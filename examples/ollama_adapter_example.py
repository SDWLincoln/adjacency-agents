"""Example: wire DeterministicEngine to a local Ollama server.

Requires the optional dependency:

    pip install -e ".[ollama]"

And a running Ollama daemon with a tool-calling-capable model pulled,
for example ``ollama pull llama3.1``. Set OLLAMA_HOST to override the
default ``http://localhost:11434``. The script exits early if the
``ollama`` package is not installed — the import is lazy.
"""

from __future__ import annotations

import os

from adjacency_agents import (
    DeterministicEngine,
    Observation,
    UserContext,
    tool_node,
)
from adjacency_agents.adapters.ollama import OllamaClient


@tool_node(requires=["public"])
def listar_servicos() -> str:
    """Lista os serviços disponíveis."""
    return "Atendimento comercial, financeiro e suporte."


@tool_node(
    requires=["registered"],
    inject={"registration_id": "metadata.registration_id"},
)
def consultar_status(registration_id: str) -> Observation:
    """Consulta status interno do cadastro."""
    return Observation(
        data={"registration_id": registration_id, "status": "active"},
        summary_hint="Resuma em uma frase.",
    )


def main() -> None:
    try:
        from ollama import Client  # noqa: I001 — lazy import keeps ollama optional
    except ImportError:
        print("Skipping: install the 'ollama' extra to run this example.")
        return

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    client = Client(host=host)
    adapter = OllamaClient(client=client, model="llama3.1")
    engine = DeterministicEngine(llm=adapter, tools=[listar_servicos, consultar_status])

    ctx_guest = UserContext(session_id="demo_guest", capabilities={"public"})
    print(
        "Guest:",
        engine.invoke(prompt="quais serviços?", context=ctx_guest).content,
    )

    ctx_reg = UserContext(
        session_id="demo_reg",
        capabilities={"public", "registered"},
        metadata={"registration_id": "abc-123"},
    )
    print(
        "Registered:",
        engine.invoke(prompt="qual meu status?", context=ctx_reg).content,
    )


if __name__ == "__main__":
    main()
