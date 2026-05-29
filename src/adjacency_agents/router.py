"""Allowlist routing — spec §12, §22.4.

Pure, deterministic, no I/O. Cannot call the LLM, cannot execute tools,
cannot read metadata (§12.4).
"""

from __future__ import annotations

from adjacency_agents.decorators import ToolNodeSpec
from adjacency_agents.models import UserContext
from adjacency_agents.registry import ToolRegistry


def build_allowlist(registry: ToolRegistry, context: UserContext) -> list[ToolNodeSpec]:
    """Return tools whose ``ToolPolicy`` admits ``context.capabilities`` (§12.2)."""
    return [spec for spec in registry if spec.policy.allows(context.capabilities)]


def build_llm_visible_allowlist(
    allowlist: list[ToolNodeSpec],
) -> list[ToolNodeSpec]:
    """Subset of the allowlist exposed to the LLM (§12.3)."""
    return [spec for spec in allowlist if spec.llm_visible]
