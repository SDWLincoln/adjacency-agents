"""ToolRegistry — spec §11, §22.3.

Per-engine instance. No global singleton (§4.1, §11.1, §22.2).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Callable

from adjacency_agents.decorators import ToolNodeSpec, get_spec, has_spec
from adjacency_agents.errors import (
    AdjacencyAgentsError,
    InvalidToolSchemaError,
    ToolNotFoundError,
)


class ToolRegistry:
    """Holds the immutable catalog of tools known to an engine instance."""

    def __init__(self, tools: Iterable[Callable[..., object]]) -> None:
        self._specs: dict[str, ToolNodeSpec] = {}

        for fn in tools:
            if not has_spec(fn):
                raise InvalidToolSchemaError(
                    f"{getattr(fn, '__name__', fn)!r} is not decorated with @tool_node"
                )
            spec = get_spec(fn)
            if spec.name in self._specs:
                raise AdjacencyAgentsError(
                    f"duplicate tool name in registry: {spec.name!r}"
                )
            self._specs[spec.name] = spec

        # §11.2.4 — validate that every declared structural neighbor exists.
        for spec in self._specs.values():
            for neighbor in spec.structural_neighbors:
                if neighbor not in self._specs:
                    raise AdjacencyAgentsError(
                        f"{spec.name!r} declares unknown structural neighbor "
                        f"{neighbor!r}"
                    )

    def get(self, name: str) -> ToolNodeSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise ToolNotFoundError(name) from exc

    def has(self, name: str) -> bool:
        return name in self._specs

    def __iter__(self) -> Iterator[ToolNodeSpec]:
        return iter(self._specs.values())

    def __len__(self) -> int:
        return len(self._specs)
