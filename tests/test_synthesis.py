"""Tests for synthesis safety — spec §14.7."""

import pytest

from adjacency_agents import (
    DeterministicEngine,
    Observation,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.errors import SynthesisError
from adjacency_agents.llm import FakeLLMClient


def _ctx(*c):
    return UserContext(session_id="s", capabilities=set(c))


@tool_node(requires=["public"])
def obs() -> Observation:
    return Observation(data={"x": 1})


@tool_node(requires=["public"])
def obs_hidden() -> Observation:
    return Observation(data={"secret": "no"}, expose_to_llm=False)


def test_synthesis_call_has_no_tools():
    fake = FakeLLMClient(script=[ToolCall(name="obs"), "synthesized text"])
    eng = DeterministicEngine(llm=fake, tools=[obs])
    out = eng.invoke(prompt="x", context=_ctx("public"))
    assert out.content == "synthesized text"
    assert fake.calls[1]["tools"] == []
    assert fake.calls[1]["allow_tool_calls"] is False


def test_synthesis_rejects_tool_call_from_llm():
    """§14.7.3 — ToolCall during synthesis is SynthesisError."""
    fake = FakeLLMClient(script=[ToolCall(name="obs"), ToolCall(name="obs")])
    eng = DeterministicEngine(llm=fake, tools=[obs])
    with pytest.raises(SynthesisError):
        eng.invoke(prompt="x", context=_ctx("public"))


def test_expose_to_llm_false_raises_without_fallback():
    """§14.7.10 — Observation marked private cannot synthesize."""
    fake = FakeLLMClient(script=[ToolCall(name="obs_hidden")])
    eng = DeterministicEngine(llm=fake, tools=[obs_hidden])
    with pytest.raises(SynthesisError):
        eng.invoke(prompt="x", context=_ctx("public"))
