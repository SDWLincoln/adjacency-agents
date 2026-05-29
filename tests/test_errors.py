"""Tests for the exception hierarchy — spec §19."""

import pytest

from adjacency_agents import errors


@pytest.mark.parametrize(
    "name",
    [
        "AdjacencyAgentsError",
        "ToolNotFoundError",
        "ToolNotAllowedError",
        "InvalidToolCallError",
        "InvalidTransitionError",
        "InvalidToolSchemaError",
        "MaxStepsExceededError",
        "ContextInjectionError",
        "ToolExecutionError",
        "AsyncRequiredError",
        "SynthesisError",
    ],
)
def test_exception_class_exists(name):
    cls = getattr(errors, name)
    assert isinstance(cls, type)


def test_all_inherit_from_base():
    base = errors.AdjacencyAgentsError
    for name in (
        "ToolNotFoundError",
        "ToolNotAllowedError",
        "InvalidToolCallError",
        "InvalidTransitionError",
        "InvalidToolSchemaError",
        "MaxStepsExceededError",
        "ContextInjectionError",
        "ToolExecutionError",
        "AsyncRequiredError",
        "SynthesisError",
    ):
        cls = getattr(errors, name)
        assert issubclass(cls, base)


def test_base_inherits_from_exception():
    assert issubclass(errors.AdjacencyAgentsError, Exception)


def test_tool_execution_error_preserves_cause():
    """Spec §14.9.2: original exception preserved as __cause__."""
    original = ValueError("boom")
    try:
        try:
            raise original
        except ValueError as e:
            raise errors.ToolExecutionError("wrapped") from e
    except errors.ToolExecutionError as wrapped:
        assert wrapped.__cause__ is original
