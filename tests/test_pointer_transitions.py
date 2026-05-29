"""Tests for multi-hop structural chains — spec §24.8, §35.5."""

import pytest

from adjacency_agents import (
    DeterministicEngine,
    EnrichedPointer,
    FinalAnswer,
    Observation,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.errors import MaxStepsExceededError
from adjacency_agents.llm import FakeLLMClient


def _ctx(*c):
    return UserContext(session_id="s", capabilities=set(c))


# A -> B -> Observation
@tool_node(requires=["public"], structural_neighbors=["b_obs"])
def a_obs() -> EnrichedPointer:
    return EnrichedPointer(next_tool="b_obs")


@tool_node(requires=["public"], llm_visible=False)
def b_obs() -> Observation:
    return Observation(data={"final": "from B"})


# A -> B -> C -> Observation
@tool_node(requires=["public"], structural_neighbors=["b_mid"])
def a_three() -> EnrichedPointer:
    return EnrichedPointer(next_tool="b_mid")


@tool_node(
    requires=["public"],
    structural_neighbors=["c_obs"],
    llm_visible=False,
)
def b_mid() -> EnrichedPointer:
    return EnrichedPointer(next_tool="c_obs")


@tool_node(requires=["public"], llm_visible=False)
def c_obs() -> Observation:
    return Observation(data={"from": "C"})


class TestChainAtoBtoObservation:
    def test_runs_chain_and_synthesizes_without_extra_routing(self):
        fake = FakeLLMClient(
            script=[
                ToolCall(name="a_obs"),
                FinalAnswer(content="resposta de B"),
            ]
        )
        eng = DeterministicEngine(llm=fake, tools=[a_obs, b_obs])
        out = eng.invoke(prompt="x", context=_ctx("public"))
        assert out.content == "resposta de B"
        # Exactly two LLM calls: initial routing + synthesis (no intermediate
        # routing decision, §31.1).
        assert len(fake.calls) == 2
        # Synthesis has no tools.
        assert fake.calls[1]["tools"] == []
        assert fake.calls[1]["allow_tool_calls"] is False


class TestChainAtoBtoCtoObservation:
    def test_runs_full_chain_with_one_synthesis(self):
        fake = FakeLLMClient(
            script=[
                ToolCall(name="a_three"),
                FinalAnswer(content="resposta final"),
            ]
        )
        eng = DeterministicEngine(
            llm=fake, tools=[a_three, b_mid, c_obs]
        )
        out = eng.invoke(prompt="x", context=_ctx("public"))
        assert out.content == "resposta final"
        assert len(fake.calls) == 2

    def test_max_steps_too_low_aborts_before_synthesis(self):
        fake = FakeLLMClient(script=[ToolCall(name="a_three")])
        eng = DeterministicEngine(
            llm=fake, tools=[a_three, b_mid, c_obs], max_steps=2
        )
        with pytest.raises(MaxStepsExceededError):
            eng.invoke(prompt="x", context=_ctx("public"))


class TestSynthesisInputs:
    def test_synthesis_does_not_receive_trace_or_internal_names(self):
        """§31.6 — synthesis receives only normalized history,
        sanitized observation and instruction. No catalog, no policies,
        no trace."""
        fake = FakeLLMClient(
            script=[
                ToolCall(name="a_obs"),
                FinalAnswer(content="ok"),
            ]
        )
        eng = DeterministicEngine(llm=fake, tools=[a_obs, b_obs])
        eng.invoke(prompt="hello", context=_ctx("public"))
        synth_msgs = fake.calls[1]["messages"]
        roles = [m.role for m in synth_msgs]
        # No 'assistant' chatter from the previous LLM turn — only the user
        # prompt, the tool observation, and the system synthesis instruction.
        assert roles.count("user") == 1
        assert any(m.role == "tool" for m in synth_msgs)
        # Names of intermediate tools must not be in any visible text.
        for msg in synth_msgs:
            assert "a_obs" not in msg.content
