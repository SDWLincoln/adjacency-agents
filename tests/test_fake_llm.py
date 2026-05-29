"""Tests for the fake LLM client used by the test suite — spec §18, §22.6."""

import pytest

from adjacency_agents import FinalAnswer, Message, ToolCall
from adjacency_agents.llm import FakeLLMClient


def _msgs():
    return [Message(role="user", content="oi")]


def test_returns_scripted_tool_call():
    fake = FakeLLMClient(script=[ToolCall(name="t", kwargs={"x": 1})])
    out = fake.complete(messages=_msgs(), tools=[], allow_tool_calls=True)
    assert isinstance(out, ToolCall)
    assert out.name == "t"


def test_returns_scripted_final_answer():
    fake = FakeLLMClient(script=[FinalAnswer(content="done")])
    out = fake.complete(messages=_msgs(), tools=[], allow_tool_calls=True)
    assert isinstance(out, FinalAnswer)
    assert out.content == "done"


def test_returns_scripted_string_then_advances():
    fake = FakeLLMClient(script=["raw", FinalAnswer(content="b")])
    assert fake.complete(messages=_msgs(), tools=[]) == "raw"
    out = fake.complete(messages=_msgs(), tools=[])
    assert isinstance(out, FinalAnswer)


def test_running_out_of_script_raises():
    fake = FakeLLMClient(script=[])
    with pytest.raises(Exception):
        fake.complete(messages=_msgs(), tools=[])


def test_records_calls():
    fake = FakeLLMClient(script=[FinalAnswer(content="x")])
    fake.complete(messages=_msgs(), tools=[{"name": "t"}], allow_tool_calls=True)
    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["allow_tool_calls"] is True
    assert call["tools"] == [{"name": "t"}]


def test_rejects_tool_call_when_disallowed():
    """§18.3.7 — when allow_tool_calls=False, ToolCall in script is an error."""
    fake = FakeLLMClient(script=[ToolCall(name="t")])
    with pytest.raises(Exception):
        fake.complete(messages=_msgs(), tools=[], allow_tool_calls=False)


async def test_acomplete_returns_same_script_item():
    fake = FakeLLMClient(script=[FinalAnswer(content="async")])
    out = await fake.acomplete(messages=_msgs(), tools=[])
    assert isinstance(out, FinalAnswer)
    assert out.content == "async"
