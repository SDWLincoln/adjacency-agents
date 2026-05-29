"""Tests for the Anthropic Messages adapter — Phase 6 of the DDD roadmap.

Like the OpenAI adapter, this does not require the ``anthropic`` SDK to
be installed: the adapter takes any object that quacks like
``anthropic.Anthropic`` and tests use a scripted fake.
"""

from __future__ import annotations

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
from adjacency_agents.adapters.anthropic import (
    AnthropicClient,
    AsyncAnthropicClient,
)
from adjacency_agents.errors import InvalidToolCallError, SynthesisError

# --- Fake Anthropic SDK shapes ---------------------------------------


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _ToolUseBlock:
    name: str
    input: dict[str, Any]
    id: str = "toolu_1"
    type: str = "tool_use"


@dataclass
class _FakeResponse:
    content: list[Any]
    stop_reason: str = "end_turn"
    role: str = "assistant"


class _FakeAnthropic:
    def __init__(self, script: list[_FakeResponse]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("Anthropic fake script exhausted")
        return self._script.pop(0)


class _FakeAsyncAnthropic:
    def __init__(self, script: list[_FakeResponse]) -> None:
        self._script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._acreate)

    async def _acreate(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        if not self._script:
            raise AssertionError("Anthropic fake script exhausted")
        return self._script.pop(0)


def _text(text: str) -> _FakeResponse:
    return _FakeResponse(content=[_TextBlock(text=text)])


def _tool(name: str, args: dict[str, Any]) -> _FakeResponse:
    return _FakeResponse(
        content=[_ToolUseBlock(name=name, input=args)],
        stop_reason="tool_use",
    )


def _ctx(*caps: str) -> UserContext:
    return UserContext(session_id="s", capabilities=set(caps))


@tool_node(requires=["public"])
def listar() -> str:
    """Listar coisas."""
    return "a, b, c"


@tool_node(requires=["public"])
def lookup(query: str) -> str:
    """Lookup."""
    return f"hit:{query}"


# --- Schema conversion ------------------------------------------------


class TestToolSchemaConversion:
    def test_internal_schema_unwrapped_into_anthropic_format(self):
        """Anthropic accepts {name, description, input_schema} directly —
        no function envelope."""
        fake = _FakeAnthropic(script=[_text("ok")])
        adapter = AnthropicClient(client=fake, model="claude-sonnet-4-6")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])

        engine.invoke(prompt="x", context=_ctx("public"))

        tools = fake.calls[0]["tools"]
        assert len(tools) == 1
        entry = tools[0]
        assert entry["name"] == "lookup"
        assert entry["description"] == "Lookup."
        # input_schema must be a JSON-schema object (no "function" wrapper).
        assert entry["input_schema"]["type"] == "object"
        assert "query" in entry["input_schema"]["properties"]
        # No leaked top-level "function" / "type" keys.
        assert "function" not in entry
        assert "type" not in entry

    def test_model_and_max_tokens_passed(self):
        fake = _FakeAnthropic(script=[_text("ok")])
        adapter = AnthropicClient(client=fake, model="claude-haiku-4-5", max_tokens=512)
        engine = DeterministicEngine(llm=adapter, tools=[listar])
        engine.invoke(prompt="x", context=_ctx("public"))
        assert fake.calls[0]["model"] == "claude-haiku-4-5"
        assert fake.calls[0]["max_tokens"] == 512

    def test_default_max_tokens(self):
        """max_tokens is required by the API — adapter must provide a default."""
        fake = _FakeAnthropic(script=[_text("ok")])
        adapter = AnthropicClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[listar])
        engine.invoke(prompt="x", context=_ctx("public"))
        assert isinstance(fake.calls[0]["max_tokens"], int)
        assert fake.calls[0]["max_tokens"] > 0


# --- Message conversion ----------------------------------------------


class TestMessageConversion:
    def test_user_and_assistant_pass_through(self):
        fake = _FakeAnthropic(script=[_text("ok")])
        adapter = AnthropicClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[listar])
        engine.invoke(
            messages=[
                Message(role="user", content="oi"),
                Message(role="assistant", content="opa"),
                Message(role="user", content="vai"),
            ],
            context=_ctx("public"),
        )
        sent = fake.calls[0]["messages"]
        assert [m["role"] for m in sent] == ["user", "assistant", "user"]

    def test_system_messages_promoted_to_system_kwarg(self):
        """Anthropic carries system prompt as a top-level kwarg, not in
        messages."""
        fake = _FakeAnthropic(script=[_text("ok")])
        adapter = AnthropicClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[listar])
        engine.invoke(
            messages=[
                Message(role="system", content="você é breve"),
                Message(role="user", content="oi"),
                Message(role="system", content="responda em pt"),
            ],
            context=_ctx("public"),
        )
        call = fake.calls[0]
        # system kwarg present, both system contents concatenated.
        assert "você é breve" in call["system"]
        assert "responda em pt" in call["system"]
        # No system role in messages list.
        assert all(m["role"] != "system" for m in call["messages"])

    def test_tool_role_repackaged_during_synthesis(self):
        """Synthesis appends Message(role='tool'). Anthropic does not
        accept that role on its own — adapter must rewrap into a user
        message with a tool-result prefix."""

        @tool_node(requires=["public"])
        def returns_obs() -> Observation:
            return Observation(data={"items": [1, 2]})

        fake = _FakeAnthropic(script=[_tool("returns_obs", {}), _text("síntese")])
        adapter = AnthropicClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[returns_obs])
        out = engine.invoke(prompt="x", context=_ctx("public"))
        assert out.content == "síntese"

        synth = fake.calls[1]["messages"]
        # No raw "tool" role.
        assert all(m["role"] != "tool" for m in synth)
        # Payload still surfaces in some user/assistant message text.
        all_text = " ".join(
            m["content"] for m in synth if isinstance(m.get("content"), str)
        )
        assert '"items"' in all_text


# --- ToolCall round-trip ---------------------------------------------


class TestToolCallParsing:
    def test_tool_use_block_becomes_tool_call(self):
        fake = _FakeAnthropic(script=[_tool("lookup", {"query": "foo"})])
        adapter = AnthropicClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        out = engine.invoke(prompt="x", context=_ctx("public"))
        assert isinstance(out, FinalAnswer)
        assert out.content == "hit:foo"

    def test_pure_text_becomes_final_answer(self):
        fake = _FakeAnthropic(script=[_text("apenas texto")])
        adapter = AnthropicClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        out = engine.invoke(prompt="x", context=_ctx("public"))
        assert out == FinalAnswer(content="apenas texto")

    def test_multiple_text_blocks_concatenated(self):
        """Anthropic may split a text response across blocks. The adapter
        joins them before returning."""
        fake = _FakeAnthropic(
            script=[
                _FakeResponse(
                    content=[
                        _TextBlock(text="parte 1\n"),
                        _TextBlock(text="parte 2"),
                    ]
                )
            ]
        )
        adapter = AnthropicClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        out = engine.invoke(prompt="x", context=_ctx("public"))
        assert out.content == "parte 1\nparte 2"


# --- Synthesis tool_choice -------------------------------------------


class TestSynthesisToolChoice:
    def test_synthesis_omits_tools_and_blocks_tool_use(self):
        @tool_node(requires=["public"])
        def returns_obs() -> Observation:
            return Observation(data={"k": "v"})

        fake = _FakeAnthropic(script=[_tool("returns_obs", {}), _text("ok")])
        adapter = AnthropicClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[returns_obs])
        engine.invoke(prompt="x", context=_ctx("public"))

        synth = fake.calls[1]
        # During synthesis the adapter must not advertise tools.
        assert synth.get("tools", []) == []
        # tool_choice may be omitted (default = no-tools when none sent),
        # but if present must indicate no tool use.
        choice = synth.get("tool_choice")
        if choice is not None:
            assert choice.get("type") in ("none", "auto")

    def test_unexpected_tool_use_during_synthesis_raises(self):
        @tool_node(requires=["public"])
        def returns_obs() -> Observation:
            return Observation(data={"k": "v"})

        fake = _FakeAnthropic(
            script=[_tool("returns_obs", {}), _tool("returns_obs", {})]
        )
        adapter = AnthropicClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[returns_obs])
        with pytest.raises(SynthesisError):
            engine.invoke(prompt="x", context=_ctx("public"))


# --- Sandbox: extra_create_kwargs may not clobber engine-controlled keys ---


class TestExtraKwargsSandbox:
    @pytest.mark.parametrize(
        "key", ["tools", "tool_choice", "messages", "model", "system", "max_tokens"]
    )
    def test_reserved_key_rejected_at_construction(self, key):
        with pytest.raises(ValueError) as exc:
            AnthropicClient(
                client=_FakeAnthropic(script=[]),
                model="m",
                extra_create_kwargs={key: "anything"},
            )
        assert key in str(exc.value)

    def test_async_reserved_key_rejected_at_construction(self):
        with pytest.raises(ValueError) as exc:
            AsyncAnthropicClient(
                client=_FakeAsyncAnthropic(script=[]),
                model="m",
                extra_create_kwargs={"tools": []},
            )
        assert "tools" in str(exc.value)

    def test_non_reserved_extra_still_forwarded(self):
        fake = _FakeAnthropic(script=[_text("ok")])
        adapter = AnthropicClient(
            client=fake, model="m", extra_create_kwargs={"temperature": 0.2}
        )
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        engine.invoke(prompt="x", context=_ctx("public"))
        assert fake.calls[0]["temperature"] == 0.2


# --- One turn = one chain (§4.7): reject parallel tool_use blocks ------


class TestMultipleToolCalls:
    def test_multiple_tool_use_blocks_rejected(self):
        multi = _FakeResponse(
            content=[
                _ToolUseBlock(name="lookup", input={"query": "x"}, id="t1"),
                _ToolUseBlock(name="lookup", input={"query": "y"}, id="t2"),
            ],
            stop_reason="tool_use",
        )
        fake = _FakeAnthropic(script=[multi])
        adapter = AnthropicClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        with pytest.raises(InvalidToolCallError):
            engine.invoke(prompt="x", context=_ctx("public"))


# --- Async ----------------------------------------------------------


class TestAsyncAdapter:
    async def test_acomplete_uses_async_client(self):
        fake = _FakeAsyncAnthropic(script=[_tool("lookup", {"query": "bar"})])
        adapter = AsyncAnthropicClient(client=fake, model="m")
        engine = DeterministicEngine(llm=adapter, tools=[lookup])
        out = await engine.ainvoke(prompt="x", context=_ctx("public"))
        assert out.content == "hit:bar"
