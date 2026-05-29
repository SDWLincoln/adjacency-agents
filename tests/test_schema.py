"""Tests for schema/validation — spec §13, §24.4."""

from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field

from adjacency_agents import tool_node
from adjacency_agents.errors import (
    ContextInjectionError,
    InvalidToolCallError,
    InvalidToolSchemaError,
)
from adjacency_agents.schema import (
    build_input_model,
    build_json_schema,
    resolve_injected_kwargs,
    validate_kwargs,
)


def _spec(fn):
    from adjacency_agents.decorators import get_spec

    return get_spec(fn)


# --- Build / introspection ---------------------------------------------


def test_function_with_type_hints_builds_model():
    @tool_node(requires=["public"])
    def t(name: str, age: int = 0) -> str:
        """doc"""
        return name

    model = build_input_model(_spec(t))
    instance = model(name="lin", age=2)
    assert instance.model_dump() == {"name": "lin", "age": 2}


def test_function_without_type_hint_raises():
    @tool_node(requires=["public"])
    def t(x) -> str:  # no annotation
        return str(x)

    with pytest.raises(InvalidToolSchemaError):
        build_input_model(_spec(t))


def test_missing_required_kwarg_fails():
    @tool_node(requires=["public"])
    def t(name: str) -> str:
        return name

    with pytest.raises(InvalidToolCallError):
        validate_kwargs(_spec(t), {})


def test_extra_kwargs_blocked_by_default():
    @tool_node(requires=["public"])
    def t(name: str) -> str:
        return name

    with pytest.raises(InvalidToolCallError):
        validate_kwargs(_spec(t), {"name": "x", "junk": 1})


def test_type_mismatch_fails():
    @tool_node(requires=["public"])
    def t(age: int) -> int:
        return age

    with pytest.raises(InvalidToolCallError):
        validate_kwargs(_spec(t), {"age": "not-int"})


def test_optional_and_literal_supported():
    @tool_node(requires=["public"])
    def t(
        kind: Literal["a", "b"],
        note: str | None = None,
    ) -> str:
        return kind

    assert validate_kwargs(_spec(t), {"kind": "a"}) == {"kind": "a", "note": None}
    assert validate_kwargs(_spec(t), {"kind": "b", "note": "x"}) == {
        "kind": "b",
        "note": "x",
    }
    with pytest.raises(InvalidToolCallError):
        validate_kwargs(_spec(t), {"kind": "c"})


def test_list_and_dict_supported():
    @tool_node(requires=["public"])
    def t(tags: list[str], meta: dict[str, int]) -> str:
        return ""

    out = validate_kwargs(_spec(t), {"tags": ["a"], "meta": {"k": 1}})
    assert out == {"tags": ["a"], "meta": {"k": 1}}


def test_annotated_field_description_in_schema():
    """§13.3.5 — per-argument description via Annotated[X, Field(description=...)]."""

    @tool_node(requires=["public"])
    def search(
        query: Annotated[str, Field(description="termo de busca")],
        limit: Annotated[int, Field(description="max resultados", ge=1)] = 10,
    ) -> str:
        return f"{query}/{limit}"

    schema = build_json_schema(_spec(search))
    props = schema["properties"]
    assert props["query"]["description"] == "termo de busca"
    assert props["limit"]["description"] == "max resultados"
    assert props["limit"]["minimum"] == 1
    # required reflects defaults
    assert "query" in schema["required"]
    assert "limit" not in schema.get("required", [])


def test_annotated_field_validates_constraints():
    """Field constraints (ge=, max_length=) must be enforced at validation."""

    @tool_node(requires=["public"])
    def search(
        limit: Annotated[int, Field(ge=1, le=100)] = 10,
    ) -> str:
        return str(limit)

    assert validate_kwargs(_spec(search), {"limit": 50}) == {"limit": 50}
    with pytest.raises(InvalidToolCallError):
        validate_kwargs(_spec(search), {"limit": 0})
    with pytest.raises(InvalidToolCallError):
        validate_kwargs(_spec(search), {"limit": 999})


def test_annotated_injected_arg_still_excluded():
    """Annotated must not break the inject-exclusion rule (§13.3.6, §17)."""
    from adjacency_agents import UserContext
    from adjacency_agents.schema import resolve_injected_kwargs

    @tool_node(
        requires=["registered"],
        inject={"registration_id": "metadata.registration_id"},
    )
    def t(
        registration_id: Annotated[str, Field(description="conta interna")],
        query: Annotated[str, Field(description="texto livre")],
    ) -> str:
        return query

    schema = build_json_schema(_spec(t))
    assert "registration_id" not in schema["properties"]
    assert schema["properties"]["query"]["description"] == "texto livre"

    ctx = UserContext(
        session_id="s",
        capabilities={"registered"},
        metadata={"registration_id": "abc"},
    )
    assert resolve_injected_kwargs(_spec(t), ctx) == {"registration_id": "abc"}


def test_user_pydantic_model_preserved():
    class Item(BaseModel):
        name: str
        qty: int

    @tool_node(requires=["public"])
    def t(item: Item) -> str:
        return item.name

    out = validate_kwargs(_spec(t), {"item": {"name": "x", "qty": 3}})
    assert out["item"].name == "x"
    assert out["item"].qty == 3


# --- Injected arguments ------------------------------------------------


def test_injected_kwarg_excluded_from_schema():
    @tool_node(
        requires=["registered"],
        inject={"registration_id": "metadata.registration_id"},
    )
    def t(registration_id: str, query: str) -> str:
        return query

    schema = build_json_schema(_spec(t))
    props = schema["properties"]
    assert "registration_id" not in props
    assert "query" in props


def test_llm_supplying_injected_arg_fails():
    @tool_node(
        requires=["registered"],
        inject={"registration_id": "metadata.registration_id"},
    )
    def t(registration_id: str, query: str) -> str:
        return query

    with pytest.raises(InvalidToolCallError):
        validate_kwargs(
            _spec(t),
            {"registration_id": "leaked", "query": "x"},
        )


def test_resolve_injected_from_metadata():
    from adjacency_agents import UserContext

    @tool_node(
        requires=["registered"],
        inject={"registration_id": "metadata.registration_id"},
    )
    def t(registration_id: str, query: str) -> str:
        return query

    ctx = UserContext(
        session_id="s1",
        capabilities={"registered"},
        metadata={"registration_id": "abc-123"},
    )
    resolved = resolve_injected_kwargs(_spec(t), ctx)
    assert resolved == {"registration_id": "abc-123"}


def test_resolve_injected_nested_metadata():
    from adjacency_agents import UserContext

    @tool_node(
        requires=["registered"],
        inject={"tenant_id": "metadata.account.tenant_id"},
    )
    def t(tenant_id: str) -> str:
        return tenant_id

    ctx = UserContext(
        session_id="s",
        capabilities={"registered"},
        metadata={"account": {"tenant_id": "T1"}},
    )
    assert resolve_injected_kwargs(_spec(t), ctx) == {"tenant_id": "T1"}


def test_resolve_injected_session_id():
    from adjacency_agents import UserContext

    @tool_node(
        requires=["public"],
        inject={"session": "session_id"},
    )
    def t(session: str) -> str:
        return session

    ctx = UserContext(session_id="abc", capabilities={"public"})
    assert resolve_injected_kwargs(_spec(t), ctx) == {"session": "abc"}


def test_resolve_injected_missing_path_raises():
    from adjacency_agents import UserContext

    @tool_node(
        requires=["registered"],
        inject={"registration_id": "metadata.registration_id"},
    )
    def t(registration_id: str) -> str:
        return registration_id

    ctx = UserContext(
        session_id="s",
        capabilities={"registered"},
        metadata={},
    )
    with pytest.raises(ContextInjectionError):
        resolve_injected_kwargs(_spec(t), ctx)
