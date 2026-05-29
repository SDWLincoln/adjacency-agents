"""Example: wire DeterministicEngine to a real OpenAI client.

Requires the optional dependency:

    pip install -e ".[openai]"

And an OPENAI_API_KEY environment variable. Without the key this script
exits early — the import is lazy.
"""

from __future__ import annotations

import os

from adjacency_agents import (
    DeterministicEngine,
    Observation,
    UserContext,
    tool_node,
)
from adjacency_agents.adapters.openai import OpenAIClient


@tool_node(requires=["public"])
def listar_servicos() -> str:
    """Lista os serviços disponíveis para o cliente."""
    return "Atendimento comercial, financeiro e suporte."


@tool_node(
    requires=["registered"],
    inject={"registration_id": "metadata.registration_id"},
)
def consultar_status(registration_id: str) -> Observation:
    """Consulta status interno do cadastro do cliente."""
    # In real life this would call the application's internal API.
    return Observation(
        data={"registration_id": registration_id, "status": "active"},
        summary_hint="Resuma o status para o cliente em uma frase.",
    )


def main() -> None:
    if "OPENAI_API_KEY" not in os.environ:
        print("Skipping: set OPENAI_API_KEY to run this example.")
        return

    from openai import OpenAI  # noqa: I001 — lazy import keeps openai optional

    client = OpenAI()
    adapter = OpenAIClient(client=client, model="gpt-4o-mini")
    engine = DeterministicEngine(llm=adapter, tools=[listar_servicos, consultar_status])

    ctx_guest = UserContext(session_id="demo_guest", capabilities={"public"})
    print("Guest:", engine.invoke(prompt="quais serviços?", context=ctx_guest).content)

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
