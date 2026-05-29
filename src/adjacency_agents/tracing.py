"""Execution tracing primitives — spec §20, §22.9."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


DEFAULT_REDACT_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "auth",
        "cookie",
        "document",
        "password",
        "passwd",
        "pwd",
        "registration_id",
        "secret",
        "session_id",
        "token",
    }
)
REDACTED = "<redacted>"


@dataclass(frozen=True)
class TraceEvent:
    """A single structured event emitted by the deterministic engine."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    index: int = 0


@dataclass
class ExecutionTrace:
    """Append-only trace for audit, debugging and tests.

    Values are sanitized on write. The engine records only structural
    metadata by default: tool names, event names, counts and type names, never
    raw prompts, capabilities, metadata, kwargs, tool payloads or tracebacks.
    """

    events: list[TraceEvent] = field(default_factory=list)
    redact_keys: frozenset[str] = field(default_factory=lambda: DEFAULT_REDACT_KEYS)

    def record(self, name: str, **data: Any) -> TraceEvent:
        event = TraceEvent(
            name=name,
            data=sanitize_trace_data(data, redact_keys=self.redact_keys),
            index=len(self.events),
        )
        self.events.append(event)
        return event

    def names(self) -> list[str]:
        return [event.name for event in self.events]

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [
                {
                    "index": event.index,
                    "name": event.name,
                    "data": event.data,
                }
                for event in self.events
            ]
        }


def sanitize_trace_data(
    data: Mapping[str, Any],
    *,
    redact_keys: frozenset[str] = DEFAULT_REDACT_KEYS,
) -> dict[str, Any]:
    return {
        str(key): _sanitize_value(key=str(key), value=value, redact_keys=redact_keys)
        for key, value in data.items()
    }


def _sanitize_value(
    *,
    key: str,
    value: Any,
    redact_keys: frozenset[str],
    depth: int = 0,
) -> Any:
    if _is_sensitive_key(key, redact_keys):
        return REDACTED
    if depth >= 4:
        return f"<{type(value).__name__}>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {
            str(child_key): _sanitize_value(
                key=str(child_key),
                value=child_value,
                redact_keys=redact_keys,
                depth=depth + 1,
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _sanitize_value(
                key=key,
                value=item,
                redact_keys=redact_keys,
                depth=depth + 1,
            )
            for item in value
        ]
    if isinstance(value, BaseException):
        return {"error_type": type(value).__name__}
    return f"<{type(value).__name__}>"


def _is_sensitive_key(key: str, redact_keys: frozenset[str]) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(part in normalized for part in redact_keys)
