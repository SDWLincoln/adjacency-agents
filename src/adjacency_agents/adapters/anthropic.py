"""Anthropic Messages API adapter — DDD §13.4, §18.3, Phase 6.

Translates between the engine's provider-agnostic protocol and the
Anthropic Messages API.

Design notes:

* No hard dependency on ``anthropic``. The user passes an instantiated
  client (sync or async); the adapter only relies on the duck-typed
  shape ``client.messages.create(**kwargs)``.
* Anthropic's tool format is ``{name, description, input_schema}`` with
  no function envelope, and ``input_schema`` IS the JSON schema (no
  nesting under ``parameters``). The adapter strips our internal
  ``title``/``description`` keys before forwarding the schema body.
* ``role="system"`` messages are pulled out of the conversation and
  promoted to the top-level ``system`` kwarg; multiple system messages
  are concatenated with two newlines.
* ``role="tool"`` messages (produced by synthesis) are rewritten as
  ``user`` messages prefixed with ``[tool: <name>]`` so the API does
  not require a paired assistant ``tool_use`` block — which our
  synthesis flow never produces.
* ``allow_tool_calls=False`` omits ``tools`` entirely. If the model
  still returns a ``tool_use`` block, ``SynthesisError`` is raised.
* The Anthropic response is a list of content blocks. ``text`` blocks
  are concatenated; a single ``tool_use`` block returns ``ToolCall``
  with the already-decoded ``input`` dict. More than one ``tool_use``
  block raises ``InvalidToolCallError`` (one turn = one chain, §4.7).
* ``max_tokens`` is required by the API. Defaults to 1024 unless the
  caller overrides it.
* ``extra_create_kwargs`` may not contain engine-controlled keys
  (``tools``, ``tool_choice``, ``messages``, ``model``, ``system``,
  ``max_tokens``); they are rejected with ``ValueError`` at construction
  so pass-through kwargs cannot re-advertise tools on the tool-disabled
  synthesis call.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from adjacency_agents.errors import InvalidToolCallError, SynthesisError
from adjacency_agents.models import FinalAnswer, Message, ToolCall

__all__ = ["AnthropicClient", "AsyncAnthropicClient"]

_DEFAULT_MAX_TOKENS = 1024

# Keys the adapter sets itself to honor the engine's sandbox contract
# (DDD §18.3, Invariant §7). ``extra_create_kwargs`` must not override
# them — otherwise a caller could re-advertise ``tools``/``tool_choice``
# on the tool-disabled synthesis call.
_RESERVED_KEYS = frozenset(
    {"tools", "tool_choice", "messages", "model", "system", "max_tokens"}
)


def _check_extra(extra: dict[str, Any] | None) -> dict[str, Any] | None:
    if not extra:
        return None
    forbidden = _RESERVED_KEYS & extra.keys()
    if forbidden:
        raise ValueError(
            "extra_create_kwargs may not contain engine-controlled keys: "
            f"{sorted(forbidden)}"
        )
    return dict(extra)


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project internal schemas onto Anthropic's ``{name, description,
    input_schema}`` shape."""
    out: list[dict[str, Any]] = []
    for schema in tools:
        input_schema: dict[str, Any] = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        if "type" not in input_schema:
            input_schema["type"] = "object"
        entry: dict[str, Any] = {
            "name": schema.get("title", ""),
            "input_schema": input_schema,
        }
        description = schema.get("description")
        if description:
            entry["description"] = description
        out.append(entry)
    return out


def _split_messages(
    messages: Sequence[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Pull ``role=system`` messages into a single system string and
    convert the remaining messages to Anthropic's wire format."""
    system_parts: list[str] = []
    chat: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
            continue
        if msg.role == "tool":
            label = msg.name or "tool"
            chat.append(
                {
                    "role": "user",
                    "content": f"[tool: {label}] {msg.content}",
                }
            )
            continue
        chat.append({"role": msg.role, "content": msg.content})
    system = "\n\n".join(system_parts) if system_parts else None
    return system, chat


def _build_kwargs(
    *,
    model: str,
    max_tokens: int,
    messages: Sequence[Message],
    tools: list[dict[str, Any]],
    allow_tool_calls: bool,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    system, chat = _split_messages(messages)
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": chat,
    }
    if system is not None:
        payload["system"] = system
    if allow_tool_calls and tools:
        payload["tools"] = _convert_tools(tools)
        payload["tool_choice"] = {"type": "auto"}
    if extra:
        payload.update(extra)
    return payload


def _parse_response(
    response: Any, *, allow_tool_calls: bool
) -> ToolCall | FinalAnswer | str:
    blocks = list(getattr(response, "content", []) or [])
    tool_use_blocks = [b for b in blocks if getattr(b, "type", None) == "tool_use"]
    if tool_use_blocks and not allow_tool_calls:
        raise SynthesisError(
            "Anthropic returned a tool_use block during synthesis (§14.7.3)"
        )
    if len(tool_use_blocks) > 1:
        raise InvalidToolCallError(
            f"Anthropic returned {len(tool_use_blocks)} tool_use blocks; "
            "one turn = one chain (§4.7)"
        )
    text_chunks: list[str] = []
    for block in blocks:
        btype = getattr(block, "type", None)
        if btype == "tool_use":
            name = getattr(block, "name", None)
            args = getattr(block, "input", None)
            if not isinstance(name, str) or not name:
                raise InvalidToolCallError("Anthropic tool_use block missing name")
            if args is None:
                args = {}
            if not isinstance(args, dict):
                raise InvalidToolCallError(
                    f"Anthropic tool_use input must be a JSON object, "
                    f"got {type(args).__name__}"
                )
            return ToolCall(name=name, kwargs=dict(args))
        if btype == "text":
            text_chunks.append(getattr(block, "text", "") or "")

    if not text_chunks:
        raise InvalidToolCallError(
            "Anthropic response had no text and no tool_use blocks"
        )
    return FinalAnswer(content="".join(text_chunks))


class AnthropicClient:
    """Synchronous adapter for ``anthropic.Anthropic``-shaped clients."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        extra_create_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._extra = _check_extra(extra_create_kwargs)

    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: list[dict[str, Any]],
        allow_tool_calls: bool = True,
    ) -> ToolCall | FinalAnswer | str:
        payload = _build_kwargs(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
            tools=tools,
            allow_tool_calls=allow_tool_calls,
            extra=self._extra,
        )
        response = self._client.messages.create(**payload)
        return _parse_response(response, allow_tool_calls=allow_tool_calls)


class AsyncAnthropicClient:
    """Asynchronous adapter for ``anthropic.AsyncAnthropic``-shaped clients."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        extra_create_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._extra = _check_extra(extra_create_kwargs)

    async def acomplete(
        self,
        *,
        messages: Sequence[Message],
        tools: list[dict[str, Any]],
        allow_tool_calls: bool = True,
    ) -> ToolCall | FinalAnswer | str:
        payload = _build_kwargs(
            model=self._model,
            max_tokens=self._max_tokens,
            messages=messages,
            tools=tools,
            allow_tool_calls=allow_tool_calls,
            extra=self._extra,
        )
        response = await self._client.messages.create(**payload)
        return _parse_response(response, allow_tool_calls=allow_tool_calls)
