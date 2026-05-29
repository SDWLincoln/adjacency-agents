"""Pydantic v2-backed schema and validation — spec §13, §17, §22.5.

Responsibilities:

* Build a Pydantic ``BaseModel`` from a tool's signature, excluding any
  parameters declared as ``inject`` (§13.3.6).
* Validate kwargs coming from the LLM or an ``EnrichedPointer`` against
  that model, raising ``InvalidToolCallError`` on any mismatch
  (§13.3.9–10).
* Resolve injected kwargs from ``UserContext`` (§17.4.4).
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

from pydantic import ConfigDict, ValidationError, create_model

from adjacency_agents.decorators import ToolNodeSpec
from adjacency_agents.errors import (
    ContextInjectionError,
    InvalidToolCallError,
    InvalidToolSchemaError,
)
from adjacency_agents.models import UserContext


_MISSING = object()


def _signature_fields(spec: ToolNodeSpec) -> dict[str, tuple[Any, Any]]:
    """Return ``{name: (annotation, default)}`` for params not injected."""
    sig = inspect.signature(spec.fn)
    fields: dict[str, tuple[Any, Any]] = {}
    injected = set(spec.inject)
    for name, param in sig.parameters.items():
        if name in injected:
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise InvalidToolSchemaError(
                f"{spec.name!r}: *args/**kwargs are not supported"
            )
        if param.annotation is inspect.Parameter.empty:
            raise InvalidToolSchemaError(
                f"{spec.name!r}: parameter {name!r} is missing a type hint"
            )
        default = (
            ... if param.default is inspect.Parameter.empty else param.default
        )
        fields[name] = (param.annotation, default)
    return fields


def build_input_model(spec: ToolNodeSpec):
    """Build a Pydantic model representing the LLM-visible kwargs (§13.3)."""
    fields = _signature_fields(spec)
    # Forbid extras so the LLM cannot smuggle arbitrary keys (§13.3.10).
    config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    model = create_model(
        f"{spec.name.title().replace('_', '')}Input",
        __config__=config,
        **fields,
    )
    return model


def build_full_input_model(spec: ToolNodeSpec):
    """Like ``build_input_model`` but includes injected fields too.

    Used by the engine to validate the *final* kwargs after injection
    (§15.9, §17.4.8).
    """
    sig = inspect.signature(spec.fn)
    fields: dict[str, tuple[Any, Any]] = {}
    for name, param in sig.parameters.items():
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            raise InvalidToolSchemaError(
                f"{spec.name!r}: *args/**kwargs are not supported"
            )
        if param.annotation is inspect.Parameter.empty:
            raise InvalidToolSchemaError(
                f"{spec.name!r}: parameter {name!r} is missing a type hint"
            )
        default = (
            ... if param.default is inspect.Parameter.empty else param.default
        )
        fields[name] = (param.annotation, default)
    config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)
    return create_model(
        f"{spec.name.title().replace('_', '')}FullInput",
        __config__=config,
        **fields,
    )


def build_json_schema(spec: ToolNodeSpec) -> dict[str, Any]:
    """Provider-agnostic JSON schema for the tool (§13.3.7)."""
    model = build_input_model(spec)
    schema = model.model_json_schema()
    if spec.description:
        schema["description"] = spec.description
    schema["title"] = spec.name
    return schema


def validate_kwargs(
    spec: ToolNodeSpec, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Validate externally supplied kwargs against the visible schema.

    Raises ``InvalidToolCallError`` if any injected key was supplied
    (§17.4.2-3) or if Pydantic rejects the payload.
    """
    injected = set(spec.inject)
    if injected & set(kwargs):
        leaked = sorted(injected & set(kwargs))
        raise InvalidToolCallError(
            f"{spec.name!r}: injected arguments cannot be supplied externally: "
            f"{leaked}"
        )
    return _validate_with(build_input_model(spec), spec.name, kwargs)


def _validate_with(model, tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        instance = model(**kwargs)
    except ValidationError as exc:
        raise InvalidToolCallError(
            f"{tool_name!r}: invalid kwargs: {exc.errors()}"
        ) from exc
    # Preserve user-declared BaseModel instances rather than dumping them
    # back to dicts (§13.3.12). For primitive types ``getattr`` returns the
    # value unchanged.
    out: dict[str, Any] = {}
    for name in instance.__class__.model_fields:
        out[name] = getattr(instance, name)
    return out


def validate_full_kwargs(
    spec: ToolNodeSpec, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Validate the full kwargs (including injected) before execution (§17.4.8)."""
    return _validate_with(build_full_input_model(spec), spec.name, kwargs)


def resolve_injected_kwargs(
    spec: ToolNodeSpec, context: UserContext
) -> dict[str, Any]:
    """Resolve the values for each ``inject={...}`` parameter (§17.3)."""
    resolved: dict[str, Any] = {}
    for arg_name, path in spec.inject.items():
        value = _resolve_path(path, context)
        if value is _MISSING:
            raise ContextInjectionError(
                f"{spec.name!r}: missing inject path {path!r} for arg {arg_name!r}"
            )
        resolved[arg_name] = value
    return resolved


def _resolve_path(path: str, context: UserContext) -> Any:
    if path == "session_id":
        return context.session_id
    if path.startswith("metadata."):
        parts = path.split(".")[1:]
        current: Any = context.metadata
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return _MISSING
            current = current[part]
        return current
    return _MISSING
