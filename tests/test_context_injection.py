"""Engine-level injection / pointer interactions — spec §17, §24.6."""

import pytest

from adjacency_agents import (
    DeterministicEngine,
    EnrichedPointer,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.errors import (
    ContextInjectionError,
    InvalidTransitionError,
)
from adjacency_agents.llm import FakeLLMClient


def _ctx(*c, metadata=None):
    return UserContext(
        session_id="s", capabilities=set(c), metadata=metadata or {}
    )


@tool_node(
    requires=["registered"],
    inject={"registration_id": "metadata.registration_id"},
)
def consult(registration_id: str) -> str:
    return f"got {registration_id}"


@tool_node(
    requires=["registered"],
    structural_neighbors=["consult"],
)
def search_then_consult() -> EnrichedPointer:
    return EnrichedPointer(
        next_tool="consult", kwargs={"registration_id": "leaked"}
    )


def test_pointer_supplying_injected_arg_fails():
    """§17.4.3 / §24.6.6 — pointer cannot fill injected args."""
    fake = FakeLLMClient(script=[ToolCall(name="search_then_consult")])
    eng = DeterministicEngine(llm=fake, tools=[search_then_consult, consult])
    ctx = _ctx("registered", metadata={"registration_id": "real"})
    with pytest.raises(InvalidTransitionError):
        eng.invoke(prompt="x", context=ctx)


def test_missing_injection_path_raises():
    fake = FakeLLMClient(script=[ToolCall(name="consult")])
    eng = DeterministicEngine(llm=fake, tools=[consult])
    ctx = _ctx("registered")  # missing metadata.registration_id
    with pytest.raises(ContextInjectionError):
        eng.invoke(prompt="x", context=ctx)


@tool_node(
    requires=["public"],
    inject={"session": "session_id"},
)
def use_session(session: str) -> str:
    return f"session={session}"


def test_session_id_injection():
    fake = FakeLLMClient(script=[ToolCall(name="use_session")])
    eng = DeterministicEngine(llm=fake, tools=[use_session])
    out = eng.invoke(prompt="x", context=_ctx("public"))
    assert out.content == "session=s"
