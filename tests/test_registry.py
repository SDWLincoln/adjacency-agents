"""Tests for ToolRegistry — spec §11, §24.2."""

import pytest

from adjacency_agents import tool_node
from adjacency_agents.errors import AdjacencyAgentsError
from adjacency_agents.registry import ToolRegistry


@tool_node(requires=["public"])
def t_a() -> str:
    """A"""
    return "a"


@tool_node(requires=["public"])
def t_b() -> str:
    """B"""
    return "b"


def test_register_single_tool():
    reg = ToolRegistry([t_a])
    assert reg.get("t_a").name == "t_a"


def test_register_multiple_tools():
    reg = ToolRegistry([t_a, t_b])
    names = sorted(s.name for s in reg)
    assert names == ["t_a", "t_b"]


def test_duplicate_name_raises():
    @tool_node(name="dup", requires=["public"])
    def x() -> str:
        return "x"

    @tool_node(name="dup", requires=["public"])
    def y() -> str:
        return "y"

    with pytest.raises(AdjacencyAgentsError):
        ToolRegistry([x, y])


def test_unknown_structural_neighbor_raises():
    @tool_node(requires=["public"], structural_neighbors=["ghost"])
    def src() -> str:
        return "x"

    with pytest.raises(AdjacencyAgentsError):
        ToolRegistry([src])


def test_known_structural_neighbor_accepted():
    @tool_node(requires=["public"], structural_neighbors=["t_b"])
    def t_src() -> str:
        return "x"

    reg = ToolRegistry([t_src, t_b])
    assert reg.get("t_src").structural_neighbors == ("t_b",)


def test_get_missing_raises():
    reg = ToolRegistry([t_a])
    with pytest.raises(AdjacencyAgentsError):
        reg.get("nope")


def test_undecorated_function_raises():
    from adjacency_agents.errors import InvalidToolSchemaError

    def plain():
        return "x"

    with pytest.raises(InvalidToolSchemaError):
        ToolRegistry([plain])


def test_function_identity_preserved():
    """§11 — registry must not wrap or replace the user function."""
    reg = ToolRegistry([t_a])
    assert reg.get("t_a").fn is t_a
