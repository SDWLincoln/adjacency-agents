"""Spec §8.5 — structural transition via EnrichedPointer."""

from adjacency_agents import (
    DeterministicEngine,
    EnrichedPointer,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.llm import FakeLLMClient


@tool_node(
    requires=["registered"],
    structural_neighbors=["consultar_resultado_detalhado"],
)
def buscar_item_recente() -> EnrichedPointer | str:
    """Busca item recente e redireciona para o detalhamento se encontrado."""
    item_id = "ITEM-007"
    if item_id:
        return EnrichedPointer(
            next_tool="consultar_resultado_detalhado",
            kwargs={"item_id": item_id},
            reason="Item recente encontrado. Transferindo para detalhamento.",
        )
    return "Nenhum item recente encontrado."


@tool_node(requires=["registered"], llm_visible=False)
def consultar_resultado_detalhado(item_id: str) -> str:
    """Consulta detalhes de um item específico."""
    return f"Item {item_id}: enviado em 2026-05-20."


if __name__ == "__main__":
    fake = FakeLLMClient(script=[ToolCall(name="buscar_item_recente")])
    engine = DeterministicEngine(
        llm=fake,
        tools=[buscar_item_recente, consultar_resultado_detalhado],
    )
    ctx = UserContext(
        session_id="s",
        capabilities={"public", "registered"},
    )
    response = engine.invoke(prompt="Qual meu último item?", context=ctx)
    print(response)
