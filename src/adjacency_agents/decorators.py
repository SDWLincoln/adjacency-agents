"""@tool_node decorator and ToolNodeSpec — spec §10, §22.2.

The decorator attaches a ``ToolNodeSpec`` to the function via a single
private attribute. There is no global registry side-effect: registration
is the explicit responsibility of ``ToolRegistry`` (§4.1, §22.2).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from adjacency_agents.models import ToolPolicy

ResponseMode = Literal["auto", "final", "synthesize"]

_SPEC_ATTR = "__adjacency_tool_spec__"


@dataclass(frozen=True)
class ToolNodeSpec:
    """Metadata produced by ``@tool_node`` (§5 ``ToolNode``)."""

    name: str
    fn: Callable[..., Any]
    policy: ToolPolicy
    description: str
    structural_neighbors: tuple[str, ...] = ()
    llm_visible: bool = True
    response_mode: ResponseMode = "auto"
    inject: dict[str, str] = field(default_factory=dict)
    is_coroutine: bool = False


def tool_node(
    *,
    name: str | None = None,
    requires: list[str] | set[str] | None = None,
    policy: ToolPolicy | None = None,
    structural_neighbors: list[str] | set[str] | tuple[str, ...] | None = None,
    llm_visible: bool = True,
    description: str | None = None,
    response_mode: ResponseMode = "auto",
    inject: dict[str, str] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a function as a tool node (§10.1).

    See §10.2 for the full ruleset.
    """
    if requires is not None and policy is not None:
        raise ValueError(
            "tool_node: 'requires' and 'policy' are mutually exclusive (§10.2.1)"
        )

    if requires is not None:
        resolved_policy = ToolPolicy(all_of=frozenset(requires))
    elif policy is not None:
        resolved_policy = policy
    else:
        # §10.2.3 / §4.1 — empty policy means default deny.
        resolved_policy = ToolPolicy()

    neighbors = tuple(structural_neighbors) if structural_neighbors else ()
    inject_map = dict(inject) if inject else {}

    if response_mode not in ("auto", "final", "synthesize"):
        raise ValueError(f"invalid response_mode: {response_mode!r}")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        resolved_name = name if name is not None else fn.__name__
        resolved_description = description
        if resolved_description is None:
            resolved_description = (inspect.getdoc(fn) or "").strip()

        if inject_map:
            sig = inspect.signature(fn)
            params = set(sig.parameters)
            unknown = set(inject_map) - params
            if unknown:
                raise ValueError(
                    f"tool_node({resolved_name}): inject keys not in signature: "
                    f"{sorted(unknown)} (§10.2.14)"
                )

        spec = ToolNodeSpec(
            name=resolved_name,
            fn=fn,
            policy=resolved_policy,
            description=resolved_description,
            structural_neighbors=neighbors,
            llm_visible=llm_visible,
            response_mode=response_mode,
            inject=inject_map,
            is_coroutine=inspect.iscoroutinefunction(fn),
        )
        setattr(fn, _SPEC_ATTR, spec)
        return fn

    return decorator


def get_spec(fn: Callable[..., Any]) -> ToolNodeSpec:
    """Return the ToolNodeSpec attached by ``@tool_node``."""
    spec = getattr(fn, _SPEC_ATTR, None)
    if not isinstance(spec, ToolNodeSpec):
        raise ValueError(
            f"{getattr(fn, '__name__', fn)!r} is not a @tool_node-decorated function"
        )
    return spec


def has_spec(fn: Callable[..., Any]) -> bool:
    return hasattr(fn, _SPEC_ATTR)
