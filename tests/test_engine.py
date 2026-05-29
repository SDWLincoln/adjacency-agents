"""Tests for DeterministicEngine — spec §14, §24.5."""

import pytest

from adjacency_agents import (
    DeterministicEngine,
    EnrichedPointer,
    FinalAnswer,
    Message,
    Observation,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.errors import (
    InvalidToolCallError,
    InvalidTransitionError,
    MaxStepsExceededError,
    ToolNotAllowedError,
    ToolNotFoundError,
)
from adjacency_agents.llm import FakeLLMClient


def _ctx(*caps: str, metadata=None) -> UserContext:
    return UserContext(
        session_id="s1",
        capabilities=set(caps),
        metadata=metadata or {},
    )


# --- Allowlist exposure -----------------------------------------------

@tool_node(requires=["public"])
def list_services() -> str:
    """List services."""
    return "comercial, financeiro, suporte"


@tool_node(requires=["guest"])
def reissue_guest(document: str) -> str:
    """guest reissue"""
    return f"hello guest {document}"


@tool_node(requires=["registered"])
def reissue_registered() -> str:
    """registered reissue"""
    return "registered ok"


class TestAllowlistExposure:
    def test_llm_only_sees_allowed_tools(self):
        fake = FakeLLMClient(script=[FinalAnswer(content="done")])
        eng = DeterministicEngine(
            llm=fake, tools=[reissue_guest, reissue_registered]
        )
        eng.invoke(prompt="oi", context=_ctx("public", "guest"))
        tools_sent = fake.calls[0]["tools"]
        names = {t["title"] for t in tools_sent}
        assert "reissue_guest" in names
        assert "reissue_registered" not in names

    def test_llm_does_not_see_tools_outside_scenario(self):
        fake = FakeLLMClient(script=[FinalAnswer(content="done")])
        eng = DeterministicEngine(
            llm=fake, tools=[reissue_guest, reissue_registered]
        )
        eng.invoke(prompt="oi", context=_ctx("public", "registered"))
        names = {t["title"] for t in fake.calls[0]["tools"]}
        assert "reissue_registered" in names
        assert "reissue_guest" not in names


# --- ToolCall execution & validation ----------------------------------

class TestToolCall:
    def test_valid_tool_call_executes(self):
        fake = FakeLLMClient(script=[ToolCall(name="list_services")])
        eng = DeterministicEngine(llm=fake, tools=[list_services])
        out = eng.invoke(prompt="quais servicos?", context=_ctx("public"))
        assert isinstance(out, FinalAnswer)
        assert "comercial" in out.content

    def test_tool_call_for_unknown_tool_fails(self):
        fake = FakeLLMClient(script=[ToolCall(name="ghost")])
        eng = DeterministicEngine(llm=fake, tools=[list_services])
        with pytest.raises(ToolNotFoundError):
            eng.invoke(prompt="x", context=_ctx("public"))

    def test_tool_call_for_disallowed_tool_fails(self):
        fake = FakeLLMClient(script=[ToolCall(name="reissue_registered")])
        eng = DeterministicEngine(
            llm=fake, tools=[reissue_registered]
        )
        with pytest.raises(ToolNotAllowedError):
            eng.invoke(prompt="x", context=_ctx("public", "guest"))

    def test_tool_call_for_invisible_tool_from_llm_fails(self):
        @tool_node(requires=["registered"], llm_visible=False)
        def hidden() -> str:
            return "secret"

        fake = FakeLLMClient(script=[ToolCall(name="hidden")])
        eng = DeterministicEngine(llm=fake, tools=[hidden])
        with pytest.raises(ToolNotAllowedError):
            eng.invoke(prompt="x", context=_ctx("registered"))

    def test_invalid_kwargs_fail(self):
        fake = FakeLLMClient(script=[ToolCall(name="reissue_guest", kwargs={})])
        eng = DeterministicEngine(llm=fake, tools=[reissue_guest])
        with pytest.raises(InvalidToolCallError):
            eng.invoke(prompt="x", context=_ctx("public", "guest"))


# --- Response normalisation -------------------------------------------

@tool_node(requires=["public"])
def returns_dict() -> dict:
    return {"items": [1, 2, 3]}


@tool_node(requires=["public"])
def returns_string() -> str:
    return "hello"


@tool_node(requires=["public"])
def returns_observation() -> Observation:
    return Observation(data={"hits": 2})


class TestResponseHandling:
    def test_string_result_becomes_final_answer(self):
        fake = FakeLLMClient(script=[ToolCall(name="returns_string")])
        eng = DeterministicEngine(llm=fake, tools=[returns_string])
        out = eng.invoke(prompt="x", context=_ctx("public"))
        assert out == FinalAnswer(content="hello")
        assert len(fake.calls) == 1  # no synthesis call

    def test_dict_triggers_synthesis(self):
        fake = FakeLLMClient(
            script=[
                ToolCall(name="returns_dict"),
                FinalAnswer(content="você tem 3 itens"),
            ]
        )
        eng = DeterministicEngine(llm=fake, tools=[returns_dict])
        out = eng.invoke(prompt="x", context=_ctx("public"))
        assert out.content == "você tem 3 itens"
        # second call must not have tools
        assert fake.calls[1]["tools"] == []
        assert fake.calls[1]["allow_tool_calls"] is False

    def test_default_tool_result_mode_applies_to_auto_tools(self):
        fake = FakeLLMClient(script=[ToolCall(name="returns_dict")])
        eng = DeterministicEngine(
            llm=fake,
            tools=[returns_dict],
            default_tool_result_mode="final",
        )
        out = eng.invoke(prompt="x", context=_ctx("public"))
        assert out.content == '{"items": [1, 2, 3]}'
        assert len(fake.calls) == 1

    def test_observation_triggers_synthesis(self):
        fake = FakeLLMClient(
            script=[
                ToolCall(name="returns_observation"),
                FinalAnswer(content="resumido"),
            ]
        )
        eng = DeterministicEngine(llm=fake, tools=[returns_observation])
        out = eng.invoke(prompt="x", context=_ctx("public"))
        assert out.content == "resumido"

    def test_tool_call_during_synthesis_raises(self):
        fake = FakeLLMClient(
            script=[
                ToolCall(name="returns_observation"),
                ToolCall(name="returns_observation"),
            ]
        )
        eng = DeterministicEngine(llm=fake, tools=[returns_observation])
        with pytest.raises(Exception):
            eng.invoke(prompt="x", context=_ctx("public"))


# --- EnrichedPointer transitions --------------------------------------

@tool_node(
    requires=["registered"],
    structural_neighbors=["detail_step"],
)
def search_step() -> EnrichedPointer:
    return EnrichedPointer(
        next_tool="detail_step",
        kwargs={"item_id": "x1"},
        reason="found",
    )


@tool_node(requires=["registered"], llm_visible=False)
def detail_step(item_id: str) -> str:
    return f"detail of {item_id}"


@tool_node(requires=["registered"])
def search_no_neighbor() -> EnrichedPointer:
    return EnrichedPointer(next_tool="detail_step", kwargs={"item_id": "x"})


class TestPointerTransitions:
    def test_valid_pointer_executes_next_tool_without_llm(self):
        fake = FakeLLMClient(script=[ToolCall(name="search_step")])
        eng = DeterministicEngine(llm=fake, tools=[search_step, detail_step])
        out = eng.invoke(prompt="x", context=_ctx("registered"))
        assert out.content == "detail of x1"
        # only one LLM call: no second routing decision
        assert len(fake.calls) == 1

    def test_pointer_to_non_neighbor_fails(self):
        fake = FakeLLMClient(script=[ToolCall(name="search_no_neighbor")])
        eng = DeterministicEngine(
            llm=fake, tools=[search_no_neighbor, detail_step]
        )
        with pytest.raises(InvalidTransitionError):
            eng.invoke(prompt="x", context=_ctx("registered"))

    def test_pointer_to_unknown_tool_fails(self):
        @tool_node(
            requires=["registered"], structural_neighbors=["detail_step"]
        )
        def search_ghost() -> EnrichedPointer:
            return EnrichedPointer(next_tool="ghost", kwargs={})

        fake = FakeLLMClient(script=[ToolCall(name="search_ghost")])
        eng = DeterministicEngine(
            llm=fake, tools=[search_ghost, detail_step]
        )
        with pytest.raises(InvalidTransitionError):
            eng.invoke(prompt="x", context=_ctx("registered"))

    def test_pointer_to_unpermitted_tool_fails(self):
        @tool_node(requires=["admin"], llm_visible=False)
        def admin_only() -> str:
            return "secret"

        @tool_node(
            requires=["registered"],
            structural_neighbors=["admin_only"],
        )
        def search_admin() -> EnrichedPointer:
            return EnrichedPointer(next_tool="admin_only", kwargs={})

        fake = FakeLLMClient(script=[ToolCall(name="search_admin")])
        eng = DeterministicEngine(
            llm=fake, tools=[search_admin, admin_only]
        )
        with pytest.raises(InvalidTransitionError):
            eng.invoke(prompt="x", context=_ctx("registered"))


# --- Max steps ---------------------------------------------------------

@tool_node(
    requires=["public"],
    structural_neighbors=["loop_b"],
)
def loop_a() -> EnrichedPointer:
    return EnrichedPointer(next_tool="loop_b")


@tool_node(
    requires=["public"],
    structural_neighbors=["loop_a"],
    llm_visible=False,
)
def loop_b() -> EnrichedPointer:
    return EnrichedPointer(next_tool="loop_a")


class TestMaxSteps:
    def test_max_steps_interrupts_cycle(self):
        fake = FakeLLMClient(script=[ToolCall(name="loop_a")])
        eng = DeterministicEngine(
            llm=fake, tools=[loop_a, loop_b], max_steps=4
        )
        with pytest.raises(MaxStepsExceededError):
            eng.invoke(prompt="x", context=_ctx("public"))


# --- Context injection in execution -----------------------------------

@tool_node(
    requires=["registered"],
    inject={"registration_id": "metadata.registration_id"},
)
def lookup_account(registration_id: str) -> str:
    return f"account {registration_id}"


class TestContextInjection:
    def test_injected_arg_resolved_at_execution(self):
        fake = FakeLLMClient(script=[ToolCall(name="lookup_account")])
        eng = DeterministicEngine(llm=fake, tools=[lookup_account])
        ctx = _ctx("registered", metadata={"registration_id": "abc"})
        out = eng.invoke(prompt="x", context=ctx)
        assert out.content == "account abc"

    def test_injected_arg_not_in_llm_schema(self):
        fake = FakeLLMClient(script=[FinalAnswer(content="no")])
        eng = DeterministicEngine(llm=fake, tools=[lookup_account])
        ctx = _ctx("registered", metadata={"registration_id": "abc"})
        eng.invoke(prompt="x", context=ctx)
        schema = fake.calls[0]["tools"][0]
        assert "registration_id" not in schema["properties"]

    def test_llm_supplying_injected_arg_fails(self):
        fake = FakeLLMClient(
            script=[
                ToolCall(
                    name="lookup_account",
                    kwargs={"registration_id": "leaked"},
                )
            ]
        )
        eng = DeterministicEngine(llm=fake, tools=[lookup_account])
        ctx = _ctx("registered", metadata={"registration_id": "abc"})
        with pytest.raises(InvalidToolCallError):
            eng.invoke(prompt="x", context=ctx)


# --- Multi-turn messages ----------------------------------------------

class TestMessages:
    def test_messages_passed_through(self):
        fake = FakeLLMClient(script=[FinalAnswer(content="ok")])
        eng = DeterministicEngine(llm=fake, tools=[list_services])
        eng.invoke(
            messages=[
                Message(role="user", content="oi"),
                Message(role="assistant", content="opa"),
                Message(role="user", content="precisamos disso"),
            ],
            context=_ctx("public"),
        )
        sent = fake.calls[0]["messages"]
        assert [m.role for m in sent] == ["user", "assistant", "user"]

    def test_prompt_appended_to_messages(self):
        fake = FakeLLMClient(script=[FinalAnswer(content="ok")])
        eng = DeterministicEngine(llm=fake, tools=[list_services])
        original = [Message(role="user", content="oi")]
        eng.invoke(
            prompt="follow-up",
            messages=original,
            context=_ctx("public"),
        )
        sent = fake.calls[0]["messages"]
        assert len(sent) == 2
        assert sent[1].content == "follow-up"
        # original list not mutated
        assert len(original) == 1
