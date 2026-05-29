"""Spec §8.2 — minimal quickstart using FakeLLMClient."""

from adjacency_agents import (
    DeterministicEngine,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.llm import FakeLLMClient


@tool_node(requires=["public"])
def listar_servicos() -> str:
    """Lista serviços disponíveis."""
    return "Temos atendimento comercial, financeiro e suporte."


if __name__ == "__main__":
    fake = FakeLLMClient(script=[ToolCall(name="listar_servicos")])
    engine = DeterministicEngine(llm=fake, tools=[listar_servicos])
    ctx = UserContext(session_id="s", capabilities={"public"})
    answer = engine.invoke(prompt="Quais serviços vocês oferecem?", context=ctx)
    print(answer.content)
