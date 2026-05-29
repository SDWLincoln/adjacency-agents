"""LLM client protocols and a deterministic fake — spec §18, §22.6."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from adjacency_agents.errors import SynthesisError
from adjacency_agents.models import FinalAnswer, Message, ToolCall


@runtime_checkable
class LLMClient(Protocol):
    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: list[dict],
        allow_tool_calls: bool = True,
    ) -> ToolCall | FinalAnswer | str: ...


@runtime_checkable
class AsyncLLMClient(Protocol):
    async def acomplete(
        self,
        *,
        messages: Sequence[Message],
        tools: list[dict],
        allow_tool_calls: bool = True,
    ) -> ToolCall | FinalAnswer | str: ...


class FakeLLMClient:
    """Scripted client used by tests and examples (§18.3.9).

    The ``script`` is consumed in order. Each call to ``complete`` or
    ``acomplete`` pops the next item and returns it, enforcing the
    ``allow_tool_calls`` flag along the way.
    """

    def __init__(
        self,
        script: Sequence[ToolCall | FinalAnswer | str],
    ) -> None:
        self._script = list(script)
        self._cursor = 0
        self.calls: list[dict[str, Any]] = []

    def _next(
        self,
        messages: Sequence[Message],
        tools: list[dict],
        allow_tool_calls: bool,
    ) -> ToolCall | FinalAnswer | str:
        self.calls.append(
            {
                "messages": list(messages),
                "tools": tools,
                "allow_tool_calls": allow_tool_calls,
            }
        )
        if self._cursor >= len(self._script):
            raise AssertionError("FakeLLMClient script exhausted")
        out = self._script[self._cursor]
        self._cursor += 1
        if isinstance(out, ToolCall) and not allow_tool_calls:
            raise SynthesisError(
                "FakeLLMClient produced a ToolCall while allow_tool_calls=False"
            )
        return out

    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: list[dict],
        allow_tool_calls: bool = True,
    ) -> ToolCall | FinalAnswer | str:
        return self._next(messages, tools, allow_tool_calls)

    async def acomplete(
        self,
        *,
        messages: Sequence[Message],
        tools: list[dict],
        allow_tool_calls: bool = True,
    ) -> ToolCall | FinalAnswer | str:
        return self._next(messages, tools, allow_tool_calls)
