"""Controlled exception hierarchy — spec §19.

Engine-level exceptions keep their specific type. Exceptions raised
inside a user tool body are remapped to ``ToolExecutionError`` by the
engine (§14.9.3, §35.1) — that remap is the engine's responsibility,
not these classes'.
"""

from __future__ import annotations


class AdjacencyAgentsError(Exception):
    """Base class for all library-controlled errors."""


class ToolNotFoundError(AdjacencyAgentsError):
    pass


class ToolNotAllowedError(AdjacencyAgentsError):
    pass


class InvalidToolCallError(AdjacencyAgentsError):
    pass


class InvalidTransitionError(AdjacencyAgentsError):
    pass


class InvalidToolSchemaError(AdjacencyAgentsError):
    pass


class MaxStepsExceededError(AdjacencyAgentsError):
    pass


class ContextInjectionError(AdjacencyAgentsError):
    pass


class ToolExecutionError(AdjacencyAgentsError):
    """Wraps any exception raised inside a user tool body (§14.9.2)."""


class AsyncRequiredError(AdjacencyAgentsError):
    pass


class SynthesisError(AdjacencyAgentsError):
    pass
