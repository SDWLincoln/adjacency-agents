"""Tests for the router — spec §12, §24.1."""

from adjacency_agents import ToolPolicy, UserContext, tool_node
from adjacency_agents.registry import ToolRegistry
from adjacency_agents.router import build_allowlist, build_llm_visible_allowlist


def _ctx(*caps: str) -> UserContext:
    return UserContext(session_id="s", capabilities=set(caps))


@tool_node(requires=["public"])
def t_public() -> str:
    return "p"


@tool_node(requires=["registered"])
def t_registered() -> str:
    return "r"


@tool_node(requires=["guest"])
def t_guest() -> str:
    return "g"


@tool_node(policy=ToolPolicy(all_of={"registered"}, none_of={"blocked"}))
def t_active() -> str:
    return "a"


@tool_node(requires=["registered"], llm_visible=False)
def t_hidden() -> str:
    return "h"


def _names(allow):
    return sorted(s.name for s in allow)


def test_all_of_satisfied_admits_tool():
    reg = ToolRegistry([t_public])
    assert _names(build_allowlist(reg, _ctx("public"))) == ["t_public"]


def test_all_of_unsatisfied_blocks_tool():
    reg = ToolRegistry([t_registered])
    assert build_allowlist(reg, _ctx("public")) == []


def test_guest_vs_registered_split():
    reg = ToolRegistry([t_guest, t_registered])
    assert _names(build_allowlist(reg, _ctx("public", "guest"))) == ["t_guest"]
    assert _names(build_allowlist(reg, _ctx("public", "registered"))) == [
        "t_registered"
    ]


def test_none_of_blocks_even_when_all_of_satisfied():
    reg = ToolRegistry([t_active])
    assert _names(build_allowlist(reg, _ctx("registered"))) == ["t_active"]
    assert build_allowlist(reg, _ctx("registered", "blocked")) == []


def test_llm_visible_allowlist_excludes_hidden():
    reg = ToolRegistry([t_registered, t_hidden])
    allow = build_allowlist(reg, _ctx("registered"))
    assert _names(allow) == ["t_hidden", "t_registered"]
    visible = build_llm_visible_allowlist(allow)
    assert _names(visible) == ["t_registered"]


def test_router_is_pure():
    """§12.4 — router must not mutate registry or context."""
    reg = ToolRegistry([t_public, t_registered])
    caps_before = {"public"}
    ctx = UserContext(session_id="s", capabilities=caps_before)
    build_allowlist(reg, ctx)
    assert ctx.capabilities == {"public"}


def test_empty_policy_blocks():
    @tool_node()
    def silent() -> str:
        return "x"

    reg = ToolRegistry([silent])
    assert build_allowlist(reg, _ctx("public", "registered")) == []
