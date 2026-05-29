"""Tests for the OpenAI adapter — Phase 6 of the DDD roadmap.

The adapter does not require the ``openai`` package to be installed:
we pass any object that quacks like ``openai.OpenAI`` (a duck-typed
client whose ``chat.completions.create`` returns the response shape
documented at https://platform.openai.com/docs/api-reference).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from adjacency_agents import (
    DeterministicEngine,
    FinalAnswer,
    Message,
    Observation,
    UserContext,
    tool_node,
)
from adjacency_agents.adapters.openai import (
    AsyncOpenAIClient,
    OpenAIClient,
)
from adjacency_agents.errors import SynthesisError

# --- Fake OpenAI SDK shapes ------------------------------------------


@dataclass
class _FakeFunction:
    name: str
    arguments: str


@dataclass
class _FakeToolCall:
    id: str
    function: _FakeFunction
    type: str = "function"


@dataclass
class _FakeMessage:
    content: str | None = None
    tool_calls: list[_FakeToolCall] | None = None


@dataclass
class _FakeChoice:
    message: _FakeMessage
    finish_reason: str = "stop"


@dataclass
class _FakeCompletion:
    choices: list[_FakeChoice]


class _FakeOpenAI:
    """Records every kwargs handed to chat.completions.create and yields
    the next scripted response."""

    def __init__(self, script: list[_FakeCompletion]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> _FakeCompletion:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("OpenAI fake script exhausted")
        return self._script.pop(0)


class _FakeAsyncOpenAI:
    def __init__(self, script: list[_FakeCompletion]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._acreate))

    async def _acreate(self, **kwargs: Any) -> _FakeCompletion:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("OpenAI fake script exhausted")
        return self._script.pop(0)


def _text(content: str) -> _FakeCompletion:
    return _FakeCompletion(choices=[_FakeChoice(message=_FakeMessage(content=content))])


def _tool(name: str, args: dict[str, Any], call_id: str = "call_1") -> _FakeCompletion:
    return _FakeCompletion(
        choices=[
            _FakeChoice(
                message=_FakeMessage(
                    content=None,
                    tool_calls=[
                        _FakeToolCall(
                            id=call_id,
                            function=_FakeFunction(
                                name=name, arguments=json.dumps(args)
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ]
    )


# --- Schema and message conversion ------------------------------------


@tool_node(requires=["public"])
def list_services() -> str:
    """List services."""
    return "comercial, financeiro, suporte"


@tool_node(requires=["public"])
def lookup(query: str) -> str:
    """Lookup something."""
    return f"hit:{query}"


def _ctx(*caps: str) -> UserContext:
    return UserContext(session_id="s", capabilities=set(caps))


class TestToolSchemaConversion:
    def test_internal_schema_wrapped_in_openai_function_format(self):
        fake = _FakeOpenAI(script=[_text("ok")])
        adapter = OpenAIClient(client=fake, model="gpt-4o-mini")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])

        engine.invoke(prompt="x", context=_ctx("public"))

        sent = fake.calls[0]["tools"]
        assert len(sent) == 1
        assert sent[0]["type"] == "function"
        fn = sent[0]["function"]
        assert fn["name"] == "lookup"
        assert fn["description"] == "Lookup something."
        # parameters carries our internal schema body (title/description
        # promoted up) — must still be a valid JSON schema fragment.
        assert fn["parameters"]["type"] == "object"
        assert "query" in fn["parameters"]["properties"]
        assert fn["parameters"]["required"] == ["query"]

    def test_model_arg_passed_through(self):
        fake = _FakeOpenAI(script=[_text("ok")])
        adapter = OpenAIClient(client=fake, model="gpt-4o")
        engine = DeterministicEngine(llm=adapter, tools=[list_services])
        engine.invoke(prompt="x", context=_ctx("public"))
        assert fake.calls[0]["model"] == "gpt-4o"


class TestMessageConversion:
    def test_user_messages_passed_as_is(self):
        fake = _FakeOpenAI(script=[_text("ok")])
        adapter = OpenAIClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[list_services])
        engine.invoke(
            messages=[
                Message(role="user", content="oi"),
                Message(role="assistant", content="opa"),
                Message(role="user", content="continua"),
            ],
            context=_ctx("public"),
        )
        sent_msgs = fake.calls[0]["messages"]
        assert [m["role"] for m in sent_msgs] == ["user", "assistant", "user"]
        assert sent_msgs[0]["content"] == "oi"

    def test_tool_role_repackaged_during_synthesis(self):
        """OpenAI rejects role=tool messages without a corresponding
        assistant tool_calls block. The adapter rewrites them into a
        system message tagged as a tool observation."""

        @tool_node(requires=["public"])
        def returns_obs() -> Observation:
            return Observation(data={"items": [1, 2, 3]})

        fake = _FakeOpenAI(
            script=[
                _tool("returns_obs", {}),
                _text("síntese"),
            ]
        )
        adapter = OpenAIClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[returns_obs])
        out = engine.invoke(prompt="x", context=_ctx("public"))
        assert out.content == "síntese"

        synth_msgs = fake.calls[1]["messages"]
        assert all(m["role"] != "tool" for m in synth_msgs), (
            "raw tool role would be rejected by OpenAI"
        )
        # The observation payload must still reach the model in some form.
        assert any('"items"' in m.get("content", "") for m in synth_msgs)


# --- ToolCall round-trip ----------------------------------------------


class TestToolCallParsing:
    def test_tool_call_returned_becomes_internal_tool_call(self):
        fake = _FakeOpenAI(script=[_tool("lookup", {"query": "foo"})])
        adapter = OpenAIClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        out = engine.invoke(prompt="x", context=_ctx("public"))
        assert isinstance(out, FinalAnswer)
        assert out.content == "hit:foo"

    def test_pure_text_response_becomes_final_answer(self):
        fake = _FakeOpenAI(script=[_text("apenas texto")])
        adapter = OpenAIClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        out = engine.invoke(prompt="x", context=_ctx("public"))
        assert out == FinalAnswer(content="apenas texto")


# --- tool_choice="none" on synthesis ---------------------------------


class TestSynthesisToolChoice:
    def test_synthesis_sends_tool_choice_none(self):
        @tool_node(requires=["public"])
        def returns_obs() -> Observation:
            return Observation(data={"k": "v"})

        fake = _FakeOpenAI(
            script=[
                _tool("returns_obs", {}),
                _text("ok"),
            ]
        )
        adapter = OpenAIClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[returns_obs])
        engine.invoke(prompt="x", context=_ctx("public"))

        synth_call = fake.calls[1]
        assert synth_call.get("tool_choice") == "none"
        # tools either omitted or empty list — both are valid for the API
        assert synth_call.get("tools", []) == []

    def test_unexpected_tool_call_during_synthesis_raises(self):
        @tool_node(requires=["public"])
        def returns_obs() -> Observation:
            return Observation(data={"k": "v"})

        fake = _FakeOpenAI(
            script=[
                _tool("returns_obs", {}),
                _tool("returns_obs", {}),  # OpenAI shouldn't, but if it does:
            ]
        )
        adapter = OpenAIClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[returns_obs])
        with pytest.raises(SynthesisError):
            engine.invoke(prompt="x", context=_ctx("public"))


# --- Async path -------------------------------------------------------


class TestAsyncAdapter:
    async def test_acomplete_uses_async_client(self):
        fake = _FakeAsyncOpenAI(script=[_tool("lookup", {"query": "bar"})])
        adapter = AsyncOpenAIClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        out = await engine.ainvoke(prompt="x", context=_ctx("public"))
        assert out.content == "hit:bar"
