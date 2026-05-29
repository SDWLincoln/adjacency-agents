"""Tests for @tool_node — spec §10, §24.3."""

import pytest

from adjacency_agents import ToolPolicy, tool_node
from adjacency_agents.decorators import get_spec


def test_requires_becomes_all_of_policy():
    @tool_node(requires=["registered"])
    def my_tool() -> str:
        """doc"""
        return "ok"

    spec = get_spec(my_tool)
    assert spec.policy.all_of == frozenset({"registered"})


def test_explicit_policy_preserved():
    pol = ToolPolicy(all_of={"a"}, none_of={"banned"})

    @tool_node(policy=pol)
    def my_tool() -> str:
        return "ok"

    assert get_spec(my_tool).policy == pol


def test_requires_and_policy_together_raises():
    with pytest.raises(Exception):

        @tool_node(requires=["a"], policy=ToolPolicy(all_of={"b"}))
        def my_tool() -> str:
            return "x"


def test_no_policy_no_requires_defaults_to_deny():
    """§10.2.3 — no policy means default deny (empty ToolPolicy)."""

    @tool_node()
    def my_tool() -> str:
        return "x"

    spec = get_spec(my_tool)
    assert spec.policy.allows({"public", "anything"}) is False


def test_default_name_is_function_name():
    @tool_node(requires=["public"])
    def listar_servicos() -> str:
        return "ok"

    assert get_spec(listar_servicos).name == "listar_servicos"


def test_custom_name_overrides():
    @tool_node(name="list_services", requires=["public"])
    def listar_servicos() -> str:
        return "ok"

    assert get_spec(listar_servicos).name == "list_services"


def test_docstring_becomes_description():
    @tool_node(requires=["public"])
    def my_tool() -> str:
        """The description."""
        return "ok"

    assert get_spec(my_tool).description == "The description."


def test_explicit_description_overrides_docstring():
    @tool_node(requires=["public"], description="custom")
    def my_tool() -> str:
        """ignored"""
        return "ok"

    assert get_spec(my_tool).description == "custom"


def test_llm_visible_default_true():
    @tool_node(requires=["public"])
    def my_tool() -> str:
        return "ok"

    assert get_spec(my_tool).llm_visible is True


def test_llm_visible_false_preserved():
    @tool_node(requires=["public"], llm_visible=False)
    def my_tool() -> str:
        return "ok"

    assert get_spec(my_tool).llm_visible is False


def test_structural_neighbors_preserved():
    @tool_node(requires=["public"], structural_neighbors=["other"])
    def my_tool() -> str:
        return "ok"

    assert get_spec(my_tool).structural_neighbors == ("other",)


def test_inject_preserved():
    @tool_node(
        requires=["registered"],
        inject={"registration_id": "metadata.registration_id"},
    )
    def my_tool(registration_id: str) -> str:
        return registration_id

    assert get_spec(my_tool).inject == {
        "registration_id": "metadata.registration_id"
    }


def test_inject_keys_must_match_parameters():
    """§10.2.14 — inject keys must correspond to existing parameters."""
    with pytest.raises(Exception):

        @tool_node(requires=["public"], inject={"ghost": "metadata.x"})
        def my_tool(name: str) -> str:
            return name


def test_function_remains_callable():
    @tool_node(requires=["public"])
    def my_tool(x: int) -> int:
        """double"""
        return x * 2

    assert my_tool(3) == 6


def test_async_tool_preserved_as_coroutine():
    import inspect

    @tool_node(requires=["public"])
    async def my_tool() -> str:
        return "async"

    assert inspect.iscoroutinefunction(my_tool)


def test_response_mode_default_auto():
    @tool_node(requires=["public"])
    def my_tool() -> str:
        return "x"

    assert get_spec(my_tool).response_mode == "auto"
