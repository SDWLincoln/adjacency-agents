"""Tests for tool runtime error handling — spec §14.9, §24.7."""

import pytest

from adjacency_agents import (
    DeterministicEngine,
    EnrichedPointer,
    FinalAnswer,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.errors import (
    ToolExecutionError,
    ToolNotAllowedError,
)
from adjacency_agents.llm import FakeLLMClient


def _ctx(*caps):
    return UserContext(session_id="s", capabilities=set(caps))


@tool_node(requires=["public"])
def boom() -> str:
    raise ValueError("downstream timeout")


@tool_node(requires=["public"])
def boom_aae() -> str:
    """Tool that raises a library exception — should still be wrapped."""
    raise ToolNotAllowedError("tool author confused themselves")


@tool_node(requires=["public"])
def boom_async() -> str:
    raise RuntimeError("async boom")


# --- raise mode --------------------------------------------------------


class TestRaiseMode:
    def test_exception_becomes_tool_execution_error(self):
        fake = FakeLLMClient(script=[ToolCall(name="boom")])
        eng = DeterministicEngine(llm=fake, tools=[boom])
        with pytest.raises(ToolExecutionError) as exc:
            eng.invoke(prompt="x", context=_ctx("public"))
        assert isinstance(exc.value.__cause__, ValueError)

    def test_library_exception_inside_tool_also_wrapped(self):
        """§14.9.3 / §35.1 — even AdjacencyAgentsError subclasses inside
        a tool become ToolExecutionError so tools cannot impersonate the
        engine's policy decisions."""
        fake = FakeLLMClient(script=[ToolCall(name="boom_aae")])
        eng = DeterministicEngine(llm=fake, tools=[boom_aae])
        with pytest.raises(ToolExecutionError) as exc:
            eng.invoke(prompt="x", context=_ctx("public"))
        assert isinstance(exc.value.__cause__, ToolNotAllowedError)


# --- final mode --------------------------------------------------------


class TestFinalMode:
    def test_returns_safe_message(self):
        fake = FakeLLMClient(script=[ToolCall(name="boom")])
        eng = DeterministicEngine(
            llm=fake,
            tools=[boom],
            tool_error_mode="final",
            default_tool_error_message="estamos com instabilidade",
        )
        out = eng.invoke(prompt="x", context=_ctx("public"))
        assert isinstance(out, FinalAnswer)
        assert out.content == "estamos com instabilidade"


# --- synthesize mode ---------------------------------------------------


class TestSynthesizeMode:
    def test_no_internal_names_or_traceback_sent(self):
        """§14.9.9 / §35.3 — synthesis must not see tool names or trace."""
        fake = FakeLLMClient(
            script=[
                ToolCall(name="boom"),
                FinalAnswer(content="desculpa, deu ruim"),
            ]
        )
        eng = DeterministicEngine(
            llm=fake,
            tools=[boom],
            tool_error_mode="synthesize",
        )
        eng.invoke(prompt="quero algo", context=_ctx("public"))

        # second call is synthesis; tools must be empty and the role=tool
        # message must NOT include the real tool name or stack trace.
        synth_call = fake.calls[1]
        assert synth_call["tools"] == []
        assert synth_call["allow_tool_calls"] is False
        tool_messages = [m for m in synth_call["messages"] if m.role == "tool"]
        for tm in tool_messages:
            assert "boom" not in (tm.name or "")
            assert "Traceback" not in tm.content
            assert "ValueError" not in tm.content


# --- failure mid-chain leaks nothing ----------------------------------


@tool_node(
    requires=["public"],
    structural_neighbors=["chain_b"],
)
def chain_a() -> EnrichedPointer:
    return EnrichedPointer(next_tool="chain_b")


@tool_node(
    requires=["public"],
    structural_neighbors=["chain_c"],
    llm_visible=False,
)
def chain_b() -> EnrichedPointer:
    return EnrichedPointer(next_tool="chain_c")


@tool_node(requires=["public"], llm_visible=False)
def chain_c() -> str:
    raise RuntimeError("end of the line")


class TestChainErrorLeakage:
    def test_chain_failure_synthesize_does_not_leak(self):
        fake = FakeLLMClient(
            script=[
                ToolCall(name="chain_a"),
                FinalAnswer(content="lamento"),
            ]
        )
        eng = DeterministicEngine(
            llm=fake,
            tools=[chain_a, chain_b, chain_c],
            tool_error_mode="synthesize",
        )
        eng.invoke(prompt="x", context=_ctx("public"))
        synth = fake.calls[1]
        leaked_terms = ["chain_a", "chain_b", "chain_c", "RuntimeError", "Traceback"]
        for msg in synth["messages"]:
            for term in leaked_terms:
                assert term not in msg.content, f"synthesis leaked {term!r} in {msg!r}"
            if msg.role == "tool":
                for term in leaked_terms:
                    assert term not in (msg.name or "")
