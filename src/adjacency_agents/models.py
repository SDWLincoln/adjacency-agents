"""Core data models — spec §9.

All models are immutable (frozen) so the engine cannot accidentally
mutate caller-owned state (invariant §9.1.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class UserContext:
    """Trusted session state at invocation time (§9.1)."""

    session_id: str
    capabilities: set[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Message:
    """Conversational history entry (§9.2)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"invalid role: {self.role!r}")


@dataclass(frozen=True)
class ToolPolicy:
    """Capability-based availability rule (§9.4).

    Empty policy denies by default (§4.1, §9.4.6).
    """

    all_of: frozenset[str] = field(default_factory=frozenset)
    any_of: frozenset[str] = field(default_factory=frozenset)
    none_of: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # Accept set or list at construction, but normalize to frozenset
        # so equality and hashing are consistent.
        object.__setattr__(self, "all_of", frozenset(self.all_of))
        object.__setattr__(self, "any_of", frozenset(self.any_of))
        object.__setattr__(self, "none_of", frozenset(self.none_of))

    def allows(self, capabilities: set[str] | frozenset[str]) -> bool:
        caps = frozenset(capabilities)
        if self.all_of and not self.all_of <= caps:
            return False
        if self.any_of and not (self.any_of & caps):
            return False
        if self.none_of & caps:
            return False
        if not self.all_of and not self.any_of:
            # none_of alone never grants — default deny (§9.4.5, §9.4.6).
            return False
        return True


@dataclass(frozen=True)
class EnrichedPointer:
    """Structural-transition request emitted by a tool (§9.5)."""

    next_tool: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class Observation:
    """Structured tool output that still needs LLM synthesis (§9.3)."""

    data: Any
    summary_hint: str | None = None
    expose_to_llm: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """LLM-selected tool with parsed kwargs (§9.6)."""

    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FinalAnswer:
    """Human-safe response payload (§9.7)."""

    content: str
