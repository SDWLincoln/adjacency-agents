"""Ollama adapter — DDD §13.4, §18.3, Phase 6.

Targets the official ``ollama`` Python SDK and any HTTP server that
mimics its ``client.chat(model=..., messages=..., tools=...)`` shape.

Design notes:

* No hard dependency on ``ollama``. The user passes any object exposing
  ``client.chat(**kwargs)`` (sync) or an awaitable equivalent (async).
* Ollama uses the same tool envelope as OpenAI:
  ``{"type": "function", "function": {"name", "description",
  "parameters"}}``. Conversion mirrors the OpenAI adapter.
* The conversation roles supported are ``system``, ``user``,
  ``assistant`` and ``tool``. Ollama keeps ``system`` inside the
  messages list (unlike Anthropic). ``role="tool"`` messages produced
  by synthesis are rewritten as ``user`` messages with a
  ``[tool: <name>]`` prefix because most Ollama models will not accept
  a bare tool role without a paired assistant ``tool_calls`` block.
* ``allow_tool_calls=False`` (synthesis) omits ``tools`` entirely. A
  ``tool_calls`` payload returned anyway raises ``SynthesisError``.
* Tool arguments returned by the SDK may be either a Python dict
  (current versions) or a JSON string (older clients / some local
  servers). Both are accepted; JSON strings are ``json.loads``ed.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from adjacency_agents.errors import InvalidToolCallError, SynthesisError
from adjacency_agents.models import FinalAnswer, Message, ToolCall

__all__ = ["OllamaClient", "AsyncOllamaClient"]


def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap internal schemas in the OpenAI-compatible function envelope
    that Ollama accepts."""
    out: list[dict[str, Any]] = []
    for schema in tools:
        params: dict[str, Any] = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        if "type" not in params:
            params["type"] = "object"
        fn_def: dict[str, Any] = {
            "name": schema.get("title", ""),
            "parameters": params,
        }
        description = schema.get("description")
        if description:
            fn_def["description"] = description
        out.append({"type": "function", "function": fn_def})
    return out


def _convert_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Map our Message dataclasses to Ollama's wire format."""
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            label = msg.name or "tool"
            out.append(
                {
                    "role": "user",
                    "content": f"[tool: {label}] {msg.content}",
                }
            )
            continue
        out.append({"role": msg.role, "content": msg.content})
    return out


def _build_kwargs(
    *,
    model: str,
    messages: Sequence[Message],
    tools: list[dict[str, Any]],
    allow_tool_calls: bool,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": _convert_messages(messages),
    }
    if allow_tool_calls and tools:
        payload["tools"] = _convert_tools(tools)
    if extra:
        payload.update(extra)
    return payload


def _decode_arguments(raw: Any, tool_name: str) -> dict[str, Any]:
    """Ollama's ``arguments`` field is a dict in modern clients but may
    be a JSON string from older servers. Accept both."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InvalidToolCallError(
                f"Ollama returned non-JSON tool arguments for {tool_name!r}: {raw!r}"
            ) from exc
        if not isinstance(decoded, dict):
            raise InvalidToolCallError(
                f"Ollama tool arguments must be a JSON object, "
                f"got {type(decoded).__name__}"
            )
        return decoded
    if raw is None:
        return {}
    raise InvalidToolCallError(
        f"Ollama tool arguments must be dict or JSON string, got {type(raw).__name__}"
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Ollama responses come back as either typed objects or plain
    dicts depending on the client version."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_response(
    response: Any, *, allow_tool_calls: bool
) -> ToolCall | FinalAnswer | str:
    message = _get(response, "message")
    if message is None:
        raise InvalidToolCallError("Ollama response had no message")
    tool_calls = _get(message, "tool_calls") or []
    if tool_calls:
        if not allow_tool_calls:
            raise SynthesisError(
                "Ollama returned tool_calls during synthesis (§14.7.3)"
            )
        first = tool_calls[0]
        function = _get(first, "function")
        if function is None:
            raise InvalidToolCallError("Ollama tool_call missing function field")
        name = _get(function, "name")
        if not isinstance(name, str) or not name:
            raise InvalidToolCallError("Ollama tool_call missing function name")
        raw_args = _get(function, "arguments")
        kwargs = _decode_arguments(raw_args, name)
        return ToolCall(name=name, kwargs=kwargs)

    content = _get(message, "content")
    if not isinstance(content, str) or content == "":
        raise InvalidToolCallError(
            "Ollama response had neither tool_calls nor non-empty content"
        )
    return FinalAnswer(content=content)


class OllamaClient:
    """Synchronous adapter for ``ollama.Client``-shaped clients."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        extra_chat_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._extra = dict(extra_chat_kwargs) if extra_chat_kwargs else None

    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: list[dict[str, Any]],
        allow_tool_calls: bool = True,
    ) -> ToolCall | FinalAnswer | str:
        payload = _build_kwargs(
            model=self._model,
            messages=messages,
            tools=tools,
            allow_tool_calls=allow_tool_calls,
            extra=self._extra,
        )
        response = self._client.chat(**payload)
        return _parse_response(response, allow_tool_calls=allow_tool_calls)


class AsyncOllamaClient:
    """Asynchronous adapter for ``ollama.AsyncClient``-shaped clients."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        extra_chat_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._extra = dict(extra_chat_kwargs) if extra_chat_kwargs else None

    async def acomplete(
        self,
        *,
        messages: Sequence[Message],
        tools: list[dict[str, Any]],
        allow_tool_calls: bool = True,
    ) -> ToolCall | FinalAnswer | str:
        payload = _build_kwargs(
            model=self._model,
            messages=messages,
            tools=tools,
            allow_tool_calls=allow_tool_calls,
            extra=self._extra,
        )
        response = await self._client.chat(**payload)
        return _parse_response(response, allow_tool_calls=allow_tool_calls)
