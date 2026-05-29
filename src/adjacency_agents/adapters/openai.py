"""OpenAI Chat Completions adapter — DDD §13.4, §18.3, Phase 6.

Translates between the engine's provider-agnostic protocol
(`LLMClient`/`AsyncLLMClient`) and the OpenAI Chat Completions API.

Design notes:

* No hard dependency on ``openai``. The user passes an instantiated
  client (sync or async) and the adapter only relies on the duck-typed
  shape ``client.chat.completions.create(**kwargs)``.
* The internal JSON schema produced by ``build_json_schema`` is wrapped
  in OpenAI's ``{"type": "function", "function": {...}}`` envelope.
* When ``allow_tool_calls=False`` the adapter sends ``tool_choice="none"``
  and **omits** the ``tools`` argument entirely. If the model still
  returns a tool call, ``SynthesisError`` is raised.
* Messages with ``role="tool"`` (produced by the engine during
  synthesis) cannot be sent as-is — OpenAI requires a paired assistant
  ``tool_calls`` block, which never exists in our flow. They are
  rewritten into ``system`` messages prefixed with ``[tool: <name>]``.
* Tool arguments returned by the model arrive as JSON strings; we
  ``json.loads`` them. Any parse error raises ``InvalidToolCallError``.
* More than one tool call in a single response raises
  ``InvalidToolCallError`` (one turn = one chain, §4.7) — the adapter
  never silently executes only the first.
* ``extra_create_kwargs`` may not contain engine-controlled keys
  (``tools``, ``tool_choice``, ``messages``, ``model``); they are
  rejected with ``ValueError`` at construction so pass-through kwargs
  cannot re-advertise tools on the tool-disabled synthesis call.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from adjacency_agents.errors import InvalidToolCallError, SynthesisError
from adjacency_agents.models import FinalAnswer, Message, ToolCall

__all__ = ["OpenAIClient", "AsyncOpenAIClient"]

# Keys the adapter sets itself to honor the engine's sandbox contract
# (DDD §18.3, Invariant §7). ``extra_create_kwargs`` must not override
# them — otherwise a caller could re-advertise ``tools``/``tool_choice``
# on the tool-disabled synthesis call.
_RESERVED_KEYS = frozenset({"tools", "tool_choice", "messages", "model"})


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
    """Wrap our internal schemas in OpenAI's function-tool envelope."""
    converted: list[dict[str, Any]] = []
    for schema in tools:
        params: dict[str, Any] = {
            k: v for k, v in schema.items() if k not in ("title", "description")
        }
        # OpenAI expects parameters to be a JSON schema object.
        if "type" not in params:
            params["type"] = "object"
        fn_def: dict[str, Any] = {
            "name": schema.get("title", ""),
            "parameters": params,
        }
        description = schema.get("description")
        if description:
            fn_def["description"] = description
        converted.append({"type": "function", "function": fn_def})
    return converted


def _convert_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Map our Message dataclasses to OpenAI's wire format.

    ``role="tool"`` messages from synthesis are repackaged as ``system``
    messages so OpenAI does not reject them (it would otherwise require
    a matching assistant ``tool_calls`` block we never produced).
    """
    out: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            label = msg.name or "tool"
            out.append(
                {
                    "role": "system",
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
        payload["tool_choice"] = "auto"
    else:
        # Synthesis: forbid tool calls outright.
        payload["tool_choice"] = "none"
    if extra:
        payload.update(extra)
    return payload


def _parse_response(
    response: Any, *, allow_tool_calls: bool
) -> ToolCall | FinalAnswer | str:
    choices = getattr(response, "choices", None)
    if not choices:
        raise InvalidToolCallError("OpenAI response had no choices")
    message = choices[0].message
    tool_calls = getattr(message, "tool_calls", None) or []

    if tool_calls:
        if not allow_tool_calls:
            raise SynthesisError(
                "OpenAI returned a tool_call during synthesis (§14.7.3)"
            )
        if len(tool_calls) > 1:
            raise InvalidToolCallError(
                f"OpenAI returned {len(tool_calls)} tool calls; one turn = "
                "one chain (§4.7)"
            )
        first = tool_calls[0]
        name = first.function.name
        raw_args = first.function.arguments or "{}"
        try:
            kwargs = json.loads(raw_args)
        except json.JSONDecodeError as exc:
            raise InvalidToolCallError(
                f"OpenAI returned non-JSON tool arguments for {name!r}: {raw_args!r}"
            ) from exc
        if not isinstance(kwargs, dict):
            raise InvalidToolCallError(
                f"OpenAI tool arguments must decode to a JSON object, "
                f"got {type(kwargs).__name__}"
            )
        return ToolCall(name=name, kwargs=kwargs)

    content = getattr(message, "content", None)
    if content is None:
        raise InvalidToolCallError("OpenAI response had neither content nor tool_calls")
    return FinalAnswer(content=content)


class OpenAIClient:
    """Synchronous adapter for ``openai.OpenAI``-shaped clients."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        extra_create_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
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
            messages=messages,
            tools=tools,
            allow_tool_calls=allow_tool_calls,
            extra=self._extra,
        )
        response = self._client.chat.completions.create(**payload)
        return _parse_response(response, allow_tool_calls=allow_tool_calls)


class AsyncOpenAIClient:
    """Asynchronous adapter for ``openai.AsyncOpenAI``-shaped clients."""

    def __init__(
        self,
        *,
        client: Any,
        model: str,
        extra_create_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._client = client
        self._model = model
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
            messages=messages,
            tools=tools,
            allow_tool_calls=allow_tool_calls,
            extra=self._extra,
        )
        response = await self._client.chat.completions.create(**payload)
        return _parse_response(response, allow_tool_calls=allow_tool_calls)
