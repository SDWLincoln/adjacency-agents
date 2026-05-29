"""Spec §8.3 — guest/registered scenario worked end-to-end with the fake LLM."""

from adjacency_agents import (
    DeterministicEngine,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.llm import FakeLLMClient


@tool_node(requires=["guest"])
def segunda_via_sem_login(documento: str) -> str:
    """Solicita segunda via para usuário sem sessão autenticada."""
    return f"Para continuar, confirme os dados de {documento}."


@tool_node(
    requires=["registered"],
    inject={"registration_id": "metadata.registration_id"},
)
def segunda_via_com_cadastro(registration_id: str) -> str:
    """Solicita segunda via para usuário identificado pela aplicação."""
    return f"Sua segunda via para o cadastro {registration_id} foi localizada."


def run_guest_flow():
    fake = FakeLLMClient(
        script=[ToolCall(name="segunda_via_sem_login", kwargs={"documento": "111"})]
    )
    engine = DeterministicEngine(
        llm=fake,
        tools=[segunda_via_sem_login, segunda_via_com_cadastro],
    )
    ctx = UserContext(
        session_id="whatsapp_guest",
        capabilities={"public", "guest"},
    )
    return engine.invoke(prompt="Quero a segunda via", context=ctx)


def run_registered_flow():
    fake = FakeLLMClient(
        script=[ToolCall(name="segunda_via_com_cadastro")]
    )
    engine = DeterministicEngine(
        llm=fake,
        tools=[segunda_via_sem_login, segunda_via_com_cadastro],
    )
    ctx = UserContext(
        session_id="whatsapp_reg",
        capabilities={"public", "registered"},
        metadata={"registration_id": "abc-123"},
    )
    return engine.invoke(prompt="Quero a segunda via", context=ctx)


if __name__ == "__main__":
    print("guest →", run_guest_flow())
    print("registered →", run_registered_flow())
