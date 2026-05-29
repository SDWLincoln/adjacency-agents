"""Tests for the Ollama adapter — Phase 6 of the DDD roadmap.

Like the other adapters, no hard dependency on the ``ollama`` SDK: the
adapter accepts any object whose ``.chat(**kwargs)`` returns the
documented response shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from adjacency_agents.adapters.ollama import (
    AsyncOllamaClient,
    OllamaClient,
)
from adjacency_agents.errors import SynthesisError

# --- Fake Ollama SDK shapes ------------------------------------------


@dataclass
class _ToolFn:
    name: str
    arguments: dict[str, Any]


@dataclass
class _Tool:
    function: _ToolFn


@dataclass
class _RespMessage:
    content: str = ""
    tool_calls: list[_Tool] = field(default_factory=list)


@dataclass
class _RespBody:
    message: _RespMessage
    done: bool = True


class _FakeOllama:
    def __init__(self, script: list[_RespBody]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> _RespBody:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("Ollama fake script exhausted")
        return self._script.pop(0)


class _FakeAsyncOllama:
    def __init__(self, script: list[_RespBody]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> _RespBody:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("Ollama fake script exhausted")
        return self._script.pop(0)


def _text(content: str) -> _RespBody:
    return _RespBody(message=_RespMessage(content=content))


def _tool(name: str, args: dict[str, Any]) -> _RespBody:
    return _RespBody(
        message=_RespMessage(
            content="",
            tool_calls=[_Tool(function=_ToolFn(name=name, arguments=args))],
        )
    )


def _ctx(*caps: str) -> UserContext:
    return UserContext(session_id="s", capabilities=set(caps))


@tool_node(requires=["public"])
def listar() -> str:
    """Listar."""
    return "ok"


@tool_node(requires=["public"])
def lookup(query: str) -> str:
    """Lookup."""
    return f"hit:{query}"


# --- Schema conversion -----------------------------------------------


class TestToolSchemaConversion:
    def test_internal_schema_wrapped_in_function_envelope(self):
        """Ollama uses the same envelope as OpenAI."""
        fake = _FakeOllama(script=[_text("ok")])
        adapter = OllamaClient(client=fake, model="llama3.1")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        engine.invoke(prompt="x", context=_ctx("public"))

        tools = fake.calls[0]["tools"]
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        fn = tools[0]["function"]
        assert fn["name"] == "lookup"
        assert fn["description"] == "Lookup."
        assert fn["parameters"]["type"] == "object"
        assert "query" in fn["parameters"]["properties"]

    def test_model_arg_passed(self):
        fake = _FakeOllama(script=[_text("ok")])
        adapter = OllamaClient(client=fake, model="llama3.2:3b")
        engine = DeterministicEngine(llm=adapter, tools=[listar])
        engine.invoke(prompt="x", context=_ctx("public"))
        assert fake.calls[0]["model"] == "llama3.2:3b"


# --- Message conversion ----------------------------------------------


class TestMessageConversion:
    def test_user_assistant_system_pass_through(self):
        """Ollama keeps system in the messages list (unlike Anthropic)."""
        fake = _FakeOllama(script=[_text("ok")])
        adapter = OllamaClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[listar])
        engine.invoke(
            messages=[
                Message(role="system", content="seja breve"),
                Message(role="user", content="oi"),
                Message(role="assistant", content="opa"),
                Message(role="user", content="vai"),
            ],
            context=_ctx("public"),
        )
        sent = fake.calls[0]["messages"]
        assert [m["role"] for m in sent] == [
            "system",
            "user",
            "assistant",
            "user",
        ]

    def test_tool_role_repackaged_during_synthesis(self):
        @tool_node(requires=["public"])
        def returns_obs() -> Observation:
            return Observation(data={"items": [1, 2]})

        fake = _FakeOllama(script=[_tool("returns_obs", {}), _text("síntese")])
        adapter = OllamaClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[returns_obs])
        out = engine.invoke(prompt="x", context=_ctx("public"))
        assert out.content == "síntese"

        synth = fake.calls[1]["messages"]
        assert all(m["role"] != "tool" for m in synth), (
            "raw tool role should be rewrapped"
        )
        all_text = " ".join(
            m["content"] for m in synth if isinstance(m.get("content"), str)
        )
        assert '"items"' in all_text


# --- Tool call round-trip --------------------------------------------


class TestToolCallParsing:
    def test_tool_call_from_dict_arguments(self):
        """Ollama returns 'arguments' as a dict already (not a JSON
        string like OpenAI)."""
        fake = _FakeOllama(script=[_tool("lookup", {"query": "foo"})])
        adapter = OllamaClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        out = engine.invoke(prompt="x", context=_ctx("public"))
        assert out == FinalAnswer(content="hit:foo")

    def test_tool_call_with_json_string_arguments_also_accepted(self):
        """Some Ollama models / clients return arguments as a JSON string.
        The adapter should be tolerant of both shapes."""
        import json

        fake = _FakeOllama(
            script=[
                _RespBody(
                    message=_RespMessage(
                        content="",
                        tool_calls=[
                            _Tool(
                                function=_ToolFn(
                                    name="lookup",
                                    arguments=json.dumps({"query": "bar"}),  # type: ignore[arg-type]
                                )
                            )
                        ],
                    )
                )
            ]
        )
        adapter = OllamaClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        out = engine.invoke(prompt="x", context=_ctx("public"))
        assert out == FinalAnswer(content="hit:bar")

    def test_pure_text_response_becomes_final_answer(self):
        fake = _FakeOllama(script=[_text("apenas texto")])
        adapter = OllamaClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        out = engine.invoke(prompt="x", context=_ctx("public"))
        assert out == FinalAnswer(content="apenas texto")


# --- Synthesis ------------------------------------------------------


class TestSynthesis:
    def test_synthesis_omits_tools(self):
        @tool_node(requires=["public"])
        def returns_obs() -> Observation:
            return Observation(data={"k": "v"})

        fake = _FakeOllama(script=[_tool("returns_obs", {}), _text("ok")])
        adapter = OllamaClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[returns_obs])
        engine.invoke(prompt="x", context=_ctx("public"))
        assert fake.calls[1].get("tools", []) == []

    def test_unexpected_tool_call_during_synthesis_raises(self):
        @tool_node(requires=["public"])
        def returns_obs() -> Observation:
            return Observation(data={"k": "v"})

        fake = _FakeOllama(
            script=[
                _tool("returns_obs", {}),
                _tool("returns_obs", {}),
            ]
        )
        adapter = OllamaClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[returns_obs])
        with pytest.raises(SynthesisError):
            engine.invoke(prompt="x", context=_ctx("public"))


# --- Async ----------------------------------------------------------


class TestAsyncAdapter:
    async def test_acomplete_uses_async_client(self):
        fake = _FakeAsyncOllama(script=[_tool("lookup", {"query": "z"})])
        adapter = AsyncOllamaClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        out = await engine.ainvoke(prompt="x", context=_ctx("public"))
        assert out == FinalAnswer(content="hit:z")
