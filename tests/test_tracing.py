"""Tests for ExecutionTrace — spec §20."""

import pytest

from adjacency_agents import (
    DeterministicEngine,
    EnrichedPointer,
    ExecutionTrace,
    FinalAnswer,
    Observation,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.errors import (
    ContextInjectionError,
    MaxStepsExceededError,
    ToolNotAllowedError,
)
from adjacency_agents.llm import FakeLLMClient
from adjacency_agents.tracing import REDACTED


def _ctx(*caps: str, metadata=None) -> UserContext:
    return UserContext(
        session_id="s",
        capabilities=set(caps),
        metadata=metadata or {},
    )


def _trace_text(trace: ExecutionTrace) -> str:
    return repr(trace.to_dict())


def test_trace_redacts_sensitive_values():
    trace = ExecutionTrace()
    trace.record(
        "custom",
        token="abc",
        nested={"registration_id": "reg-123", "safe": "ok"},
        values=[{"password": "secret", "safe": 1}],
    )

    event = trace.events[0]
    assert event.data["token"] == REDACTED
    assert event.data["nested"]["registration_id"] == REDACTED
    assert event.data["nested"]["safe"] == "ok"
    assert event.data["values"][0]["password"] == REDACTED


@tool_node(requires=["public"])
def trace_hello() -> str:
    return "hello"


def test_engine_records_success_trace_without_prompt_or_payload_values():
    fake = FakeLLMClient(script=[ToolCall(name="trace_hello")])
    engine = DeterministicEngine(llm=fake, tools=[trace_hello])

    out = engine.invoke(prompt="my password is 123", context=_ctx("public"))

    assert out == FinalAnswer(content="hello")
    assert engine.last_trace is not None
    assert engine.last_trace.names() == [
        "allowlist_built",
        "llm_called",
        "tool_call_received",
        "tool_call_validated",
        "tool_executed",
        "final_answer_returned",
    ]
    assert "my password is 123" not in _trace_text(engine.last_trace)


@tool_node(requires=["public"], structural_neighbors=["trace_detail"])
def trace_search() -> EnrichedPointer:
    return EnrichedPointer(
        next_tool="trace_detail",
        kwargs={"item_id": "ITEM-SECRET"},
        reason="internal reason",
    )


@tool_node(requires=["public"], llm_visible=False)
def trace_detail(item_id: str) -> str:
    return f"detail {item_id}"


def test_trace_records_pointer_events_without_pointer_values():
    fake = FakeLLMClient(script=[ToolCall(name="trace_search")])
    engine = DeterministicEngine(llm=fake, tools=[trace_search, trace_detail])

    engine.invoke(prompt="x", context=_ctx("public"))

    assert engine.last_trace is not None
    names = engine.last_trace.names()
    assert "pointer_received" in names
    assert "pointer_validated" in names
    assert "transition_executed" in names
    assert "ITEM-SECRET" not in _trace_text(engine.last_trace)
    assert "internal reason" not in _trace_text(engine.last_trace)


@tool_node(requires=["public"])
def trace_observation() -> Observation:
    return Observation(data={"secret": "payload", "count": 1})


def test_trace_records_synthesis_without_raw_observation_payload():
    fake = FakeLLMClient(
        script=[
            ToolCall(name="trace_observation"),
            FinalAnswer(content="synthesized"),
        ]
    )
    engine = DeterministicEngine(llm=fake, tools=[trace_observation])

    out = engine.invoke(prompt="x", context=_ctx("public"))

    assert out.content == "synthesized"
    assert engine.last_trace is not None
    names = engine.last_trace.names()
    assert "observation_created" in names
    assert "synthesis_requested" in names
    assert "synthesis_completed" in names
    assert "payload" not in _trace_text(engine.last_trace)


@tool_node(requires=["registered"])
def trace_registered_only() -> str:
    return "registered"


def test_trace_records_policy_denied_for_disallowed_tool_call():
    fake = FakeLLMClient(script=[ToolCall(name="trace_registered_only")])
    engine = DeterministicEngine(llm=fake, tools=[trace_registered_only])

    with pytest.raises(ToolNotAllowedError):
        engine.invoke(prompt="x", context=_ctx("guest"))

    assert engine.last_trace is not None
    assert "policy_denied" in engine.last_trace.names()


@tool_node(
    requires=["registered"],
    inject={"registration_id": "metadata.registration_id"},
)
def trace_requires_injection(registration_id: str) -> str:
    return registration_id


def test_trace_records_context_injection_failure_without_value():
    fake = FakeLLMClient(script=[ToolCall(name="trace_requires_injection")])
    engine = DeterministicEngine(llm=fake, tools=[trace_requires_injection])

    with pytest.raises(ContextInjectionError):
        engine.invoke(prompt="x", context=_ctx("registered"))

    assert engine.last_trace is not None
    assert "context_injection_failed" in engine.last_trace.names()


@tool_node(requires=["public"], structural_neighbors=["trace_loop_b"])
def trace_loop_a() -> EnrichedPointer:
    return EnrichedPointer(next_tool="trace_loop_b")


@tool_node(
    requires=["public"],
    structural_neighbors=["trace_loop_a"],
    llm_visible=False,
)
def trace_loop_b() -> EnrichedPointer:
    return EnrichedPointer(next_tool="trace_loop_a")


def test_trace_records_max_steps_exceeded():
    fake = FakeLLMClient(script=[ToolCall(name="trace_loop_a")])
    engine = DeterministicEngine(
        llm=fake,
        tools=[trace_loop_a, trace_loop_b],
        max_steps=2,
    )

    with pytest.raises(MaxStepsExceededError):
        engine.invoke(prompt="x", context=_ctx("public"))

    assert engine.last_trace is not None
    assert "max_steps_exceeded" in engine.last_trace.names()


@tool_node(requires=["public"])
def trace_boom() -> str:
    raise RuntimeError("downstream secret")


def test_trace_records_tool_execution_failure_without_exception_message():
    fake = FakeLLMClient(script=[ToolCall(name="trace_boom")])
    engine = DeterministicEngine(
        llm=fake,
        tools=[trace_boom],
        tool_error_mode="final",
    )

    out = engine.invoke(prompt="x", context=_ctx("public"))

    assert out.content == "I could not complete this operation right now."
    assert engine.last_trace is not None
    assert "tool_execution_failed" in engine.last_trace.names()
    assert "downstream secret" not in _trace_text(engine.last_trace)
