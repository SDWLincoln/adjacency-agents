"""Tests for the core data models — spec §9, §23."""

import pytest

from adjacency_agents import (
    EnrichedPointer,
    FinalAnswer,
    Message,
    Observation,
    ToolCall,
    ToolPolicy,
    UserContext,
)

# --- UserContext --------------------------------------------------------


class TestUserContext:
    def test_session_id_and_capabilities_are_required(self):
        ctx = UserContext(session_id="s1", capabilities={"public"})
        assert ctx.session_id == "s1"
        assert ctx.capabilities == {"public"}
        assert ctx.metadata == {}

    def test_metadata_defaults_to_empty_dict(self):
        ctx = UserContext(session_id="s1", capabilities=set())
        assert ctx.metadata == {}

    def test_is_frozen(self):
        from dataclasses import FrozenInstanceError

        ctx = UserContext(session_id="s1", capabilities={"public"})
        with pytest.raises(FrozenInstanceError):
            ctx.session_id = "other"  # type: ignore[misc]


# --- Message ------------------------------------------------------------


class TestMessage:
    def test_user_message(self):
        msg = Message(role="user", content="oi")
        assert msg.role == "user"
        assert msg.content == "oi"
        assert msg.name is None
        assert msg.metadata == {}

    def test_invalid_role_rejected(self):
        with pytest.raises(ValueError):
            Message(role="banana", content="x")  # type: ignore[arg-type]


# --- ToolPolicy ---------------------------------------------------------


class TestToolPolicy:
    def test_empty_policy_denies(self):
        assert ToolPolicy().allows({"public"}) is False

    def test_all_of_satisfied(self):
        p = ToolPolicy(all_of={"a", "b"})
        assert p.allows({"a", "b"}) is True
        assert p.allows({"a", "b", "c"}) is True

    def test_all_of_incomplete_blocks(self):
        p = ToolPolicy(all_of={"a", "b"})
        assert p.allows({"a"}) is False

    def test_any_of_one_match_allows(self):
        p = ToolPolicy(any_of={"a", "b"})
        assert p.allows({"b"}) is True

    def test_any_of_no_match_blocks(self):
        p = ToolPolicy(any_of={"a", "b"})
        assert p.allows({"c"}) is False

    def test_none_of_blocks_on_intersection(self):
        p = ToolPolicy(all_of={"a"}, none_of={"banned"})
        assert p.allows({"a"}) is True
        assert p.allows({"a", "banned"}) is False

    def test_none_of_alone_does_not_grant(self):
        p = ToolPolicy(none_of={"banned"})
        assert p.allows({"x"}) is False
        assert p.allows(set()) is False

    def test_combined_all_of_and_any_of_use_and(self):
        p = ToolPolicy(all_of={"a"}, any_of={"x", "y"})
        assert p.allows({"a", "x"}) is True
        assert p.allows({"a"}) is False  # any_of missing
        assert p.allows({"x"}) is False  # all_of missing

    def test_unknown_capability_does_not_grant(self):
        p = ToolPolicy(all_of={"a"})
        assert p.allows({"banana"}) is False


# --- EnrichedPointer ---------------------------------------------------


class TestEnrichedPointer:
    def test_basic_pointer(self):
        p = EnrichedPointer(next_tool="t2", kwargs={"x": 1}, reason="why")
        assert p.next_tool == "t2"
        assert p.kwargs == {"x": 1}
        assert p.reason == "why"

    def test_kwargs_default_empty(self):
        p = EnrichedPointer(next_tool="t2")
        assert p.kwargs == {}
        assert p.reason == ""


# --- Observation -------------------------------------------------------


class TestObservation:
    def test_basic_observation(self):
        obs = Observation(data={"item": 1})
        assert obs.data == {"item": 1}
        assert obs.summary_hint is None
        assert obs.expose_to_llm is True
        assert obs.metadata == {}

    def test_can_hide_from_llm(self):
        obs = Observation(data="secret", expose_to_llm=False)
        assert obs.expose_to_llm is False


# --- ToolCall ----------------------------------------------------------


class TestToolCall:
    def test_basic(self):
        tc = ToolCall(name="t1", kwargs={"a": 1})
        assert tc.name == "t1"
        assert tc.kwargs == {"a": 1}

    def test_kwargs_default_empty(self):
        tc = ToolCall(name="t1")
        assert tc.kwargs == {}


# --- FinalAnswer -------------------------------------------------------


class TestFinalAnswer:
    def test_basic(self):
        fa = FinalAnswer(content="hello")
        assert fa.content == "hello"
