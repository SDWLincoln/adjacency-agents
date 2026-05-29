# AdjacencyAgents — Documentation-Driven Development

**Status:** Implementation-ready 0.4  
**Tipo:** Especificação de desenvolvimento guiada por documentação  
**Projeto:** `adjacency-agents`  
**Objetivo:** Criar uma microbiblioteca Python para orquestração determinística de ferramentas em fluxos com LLMs/SLMs, com restrição genérica de parsing, segurança por contexto e transições estruturais controladas.

---

## 1. Propósito deste documento

Este documento define, antes da implementação, os contratos públicos, regras internas, invariantes, diretivas de arquitetura, critérios de teste e critérios de aceite da biblioteca `adjacency-agents`.

A implementação só deve ser considerada correta se obedecer a este documento.

Este é um documento de **Documentation-Driven Development**. Portanto:

1. Toda feature deve começar com alteração ou validação da documentação.
2. Todo comportamento público deve estar descrito antes de ser implementado.
3. Todo comportamento descrito deve ter teste automatizado correspondente.
4. Nenhuma API pública deve ser adicionada sem exemplo de uso documentado.
5. Nenhuma regra de segurança deve existir apenas no código; ela deve estar documentada como invariante.

---

## 2. Problema

Modelos menores de linguagem, especialmente SLMs, têm dificuldade em escolher corretamente ferramentas em fluxos com múltiplos cenários, múltiplos estados de sessão e ferramentas semanticamente parecidas.

Exemplo típico:

```text
Usuário quer segunda via de boleto.
Existe uma tool para usuário logado.
Existe outra tool para usuário não logado.
O LLM vê ambas e escolhe a errada.
```

Esse erro não é exatamente um erro de autorização. É um erro de **parsing contextual**: o modelo está tentando escolher e preencher argumentos para uma ferramenta que nem deveria existir no cenário atual.

O problema central é:

> O LLM não deve decidir qual cenário de negócio está ativo quando esse cenário já é conhecido pelo backend, API, banco, sessão ou webhook.

---

## 3. Solução resumida

A biblioteca deve permitir que a aplicação cliente transforme dados confiáveis do domínio em `capabilities` efêmeras. Cada tool declara uma política de disponibilidade. Antes de chamar o LLM, o engine monta uma allowlist contextual e expõe apenas as tools permitidas para aquele cenário.

Fluxo geral:

```text
API / banco / webhook / sessão
        ↓
Fatos confiáveis da aplicação
        ↓
UserContext.capabilities
        ↓
ToolPolicy
        ↓
Allowlist contextual
        ↓
Schemas enviados ao LLM
        ↓
ToolCall validada pelo engine
        ↓
Execução determinística
        ↓
Possível EnrichedPointer para próxima tool
```

O LLM atua como gatilho semântico inicial. O engine Python controla visibilidade, parsing, validação, execução, transição estrutural e segurança.

---

## 4. Princípios arquiteturais

### 4.1. Omitir não é permitir

Uma tool sem política explícita de acesso não deve ser considerada pública por acidente.

Regra:

```python
ToolPolicy.default_allows == False
```

Uma ferramenta só deve entrar na allowlist se sua política for satisfeita pelo `UserContext` atual.

---

### 4.2. O LLM não decide segurança

O LLM nunca deve decidir se uma tool pode ser usada. Ele só escolhe entre ferramentas previamente permitidas pelo engine.

O engine deve validar permissão em três momentos:

1. Antes de expor schemas ao LLM.
2. Antes de executar uma ToolCall retornada pelo LLM.
3. Antes de executar uma transição estrutural via `EnrichedPointer`.

---

### 4.3. Policy-Gated Parsing

Ferramentas incompatíveis com o cenário atual não devem ser convertidas em schema para o LLM.

A restrição de parsing é aplicada antes da chamada ao modelo:

```text
registry completo -> router -> allowlist -> schema_builder -> LLM
```

Nunca:

```text
registry completo -> schema_builder -> LLM
```

---

### 4.4. Transferência determinística

O LLM pode iniciar um fluxo chamando uma ferramenta permitida. Depois disso, ferramentas podem transferir execução para vizinhos estruturais por meio de `EnrichedPointer`.

Essa transferência não consulta o LLM novamente.

---

### 4.5. Transição estrutural não ignora política

Mesmo que uma tool retorne um `EnrichedPointer`, o engine deve validar:

1. Se a próxima tool existe.
2. Se a próxima tool é vizinha estrutural permitida da tool atual.
3. Se o contexto atual permite a próxima tool.
4. Se os argumentos são válidos para o schema da próxima tool.
5. Se o limite de passos não foi excedido.

---

### 4.6. Dados confiáveis vêm da aplicação, não do LLM

Dados como identificadores de usuário, matrícula, conta, tenant, canal, sessão, plano, permissões, flags e estados de jornada devem vir da aplicação cliente.

O LLM não deve inventar, confirmar ou substituir fatos confiáveis do backend.

---

### 4.7. Um turno executa uma única cadeia

Cada chamada de `invoke` ou `ainvoke` representa um turno de usuário. Dentro desse turno, o LLM pode tomar no máximo uma decisão de roteamento inicial: retornar uma `FinalAnswer` ou uma única `ToolCall`.

Depois que a primeira `ToolCall` é validada e executada, o LLM não deve voltar a escolher novas tools com base no resultado. A continuação do fluxo ocorre apenas por transições estruturais via `EnrichedPointer`, validadas pelo engine.

A única chamada posterior ao LLM permitida no mesmo turno é a chamada de síntese final a partir de `Observation`, sempre com tools desabilitadas.

Esta biblioteca não implementa um loop ReAct multi-step no qual o modelo observa o resultado de uma tool e decide outra tool no mesmo turno. Se o produto precisar de múltiplos passos, esses passos devem ser modelados como grafo estrutural ou como novos turnos explícitos da aplicação.

---

## 5. Linguagem ubíqua

### Capability

Um rótulo textual, efêmero e definido pela aplicação cliente, derivado de fatos confiáveis do contexto atual.

Exemplos:

```text
public
anonymous
registered
logged_in
guest
active_account
premium
kyc_verified
cart_not_empty
business_hours
blocked
fraud_suspected
```

A biblioteca não interpreta semanticamente esses nomes.

---

### UserContext

Objeto que representa o estado confiável da sessão no momento da invocação.

Contém:

- `session_id`
- `capabilities`
- `metadata`

O engine usa `capabilities` para política e pode usar `metadata` para injeção de argumentos confiáveis.

---

### ToolPolicy

Regra declarativa que define quando uma tool pode ser visível, parseável e executável.

A forma mínima deve suportar:

- `all_of`: todas as capabilities exigidas.
- `any_of`: pelo menos uma capability exigida.
- `none_of`: capabilities que não podem estar presentes.

---

### ToolNode

Representação interna de uma ferramenta registrada.

Contém:

- nome público da tool;
- função Python original;
- schema de argumentos;
- política de disponibilidade;
- vizinhos estruturais;
- flag `llm_visible`;
- metadados de documentação.

---

### Allowlist

Subconjunto de tools cujo `ToolPolicy` é satisfeito pelo `UserContext` atual.

---

### LLM-visible allowlist

Subconjunto da allowlist que pode ser exposto ao LLM.

Ferramentas com `llm_visible=False` podem ser executáveis estruturalmente, mas não devem aparecer no schema enviado ao modelo.

---

### ToolCall

Objeto retornado pelo adaptador de LLM indicando que o modelo escolheu uma ferramenta permitida e forneceu argumentos parseados.

---

### FinalAnswer

Objeto ou string indicando resposta final ao usuário.

---

### EnrichedPointer

Objeto retornado por uma tool para solicitar transferência estrutural determinística para outra tool.

Contém:

- `next_tool`
- `kwargs`
- `reason`

---

### Structural Neighbor

Ferramenta explicitamente declarada como próximo passo permitido de uma tool atual.

---

### ExecutionTrace

Registro estruturado dos passos executados pelo engine.

Deve permitir depuração, auditoria e testes.

---

## 6. Objetivos do MVP

O MVP deve entregar:

1. Decorador `@tool_node`.
2. Modelo `UserContext`.
3. Modelo `Message` para histórico conversacional.
4. Modelo `ToolPolicy`.
5. Modelo `EnrichedPointer`.
6. Modelo `Observation` para retorno estruturado de tool que exige síntese.
7. Modelo `ToolCall`.
8. Modelo `FinalAnswer`.
9. `ToolRegistry` sem singleton global.
10. Router de allowlist contextual.
11. Engine determinístico com limite de passos.
12. Assinatura `invoke(..., prompt=None, messages=None, context=...)`.
13. Assinatura assíncrona `ainvoke(..., prompt=None, messages=None, context=...)`.
14. Validação de ToolCall.
15. Validação de EnrichedPointer.
16. Modelo de execução: um turno = uma cadeia iniciada pelo LLM.
17. Síntese final opcional e controlada a partir de `Observation`.
18. Schema builder e validação de kwargs baseados em Pydantic v2.
19. Suporte mínimo a tools `def` e `async def`.
20. Injeção mínima de contexto via `inject={...}` no decorador.
21. Tratamento controlado de exceções de runtime de tools.
22. Fake LLM client para testes.
23. Exemplo completo de cenário logado/não logado.
24. Testes automatizados com pytest.

---

## 7. Fora do escopo do MVP

O MVP não precisa implementar:

1. Integração nativa com múltiplos provedores de LLM.
2. Persistência de sessão.
3. Interface gráfica.
4. Banco de dados interno.
5. Autenticação própria.
6. Autorização baseada em usuário real.
7. Aprendizado automático de policies.
8. Geração automática de capabilities a partir de qualquer API.
9. Sistema distribuído de tracing.
10. Execução paralela/distribuída de múltiplas tools.
11. Retry/backoff avançado para LLM ou ferramentas externas.
12. Streaming de tokens.

Esses itens podem ser adicionados depois, desde que não quebrem os contratos do MVP.

Atenção: execução assíncrona básica com `ainvoke` e suporte a `async def` não ficam fora do MVP. Elas fazem parte do contrato mínimo de produção.

---

## 8. Contrato público esperado

### 8.1. Instalação

Nome do pacote:

```text
adjacency-agents
```

Import público esperado:

```python
from adjacency_agents import (
    DeterministicEngine,
    EnrichedPointer,
    FinalAnswer,
    Message,
    Observation,
    ToolCall,
    ToolPolicy,
    UserContext,
    tool_node,
)
```

O arquivo `__init__.py` deve funcionar como fachada pública rigorosa. Módulos internos não devem ser necessários para uso normal.

---

### 8.2. Exemplo mínimo

```python
from adjacency_agents import DeterministicEngine, ToolPolicy, UserContext, tool_node


@tool_node(requires=["public"])
def listar_servicos() -> str:
    """Lista serviços disponíveis."""
    return "Temos atendimento comercial, financeiro e suporte."


engine = DeterministicEngine(
    llm=my_llm_client,
    tools=[listar_servicos],
)

context = UserContext(
    session_id="session_123",
    capabilities={"public"},
)

response = engine.invoke(
    prompt="Quais serviços vocês oferecem?",
    context=context,
)
```

`prompt` é uma conveniência para chamada de turno único. Em fluxos multi-turno, a aplicação deve enviar `messages`.

```python
from adjacency_agents import Message

response = engine.invoke(
    messages=[
        Message(role="user", content="Quero atendimento financeiro"),
        Message(role="assistant", content="Você já possui cadastro conosco?"),
        Message(role="user", content="Sim"),
    ],
    context=context,
)
```

---

### 8.3. Exemplo com restrição genérica de cenário

```python
from adjacency_agents import DeterministicEngine, UserContext, tool_node


@tool_node(requires=["guest"])
def segunda_via_sem_login(documento: str) -> str:
    """Solicita segunda via para usuário sem sessão autenticada."""
    return "Para continuar, confirme os dados cadastrais."


@tool_node(requires=["registered"])
def segunda_via_com_cadastro() -> str:
    """Solicita segunda via para usuário já identificado pela aplicação."""
    return "Sua segunda via foi localizada."


api_response = api.get_session_state(phone="5511999999999")

capabilities = {"public"}

if api_response.registration_id:
    capabilities.add("registered")
else:
    capabilities.add("guest")

context = UserContext(
    session_id="whatsapp_123",
    capabilities=capabilities,
    metadata={
        "registration_id": api_response.registration_id,
        "channel": "whatsapp",
    },
)

engine = DeterministicEngine(
    llm=my_llm_client,
    tools=[segunda_via_sem_login, segunda_via_com_cadastro],
)

response = engine.invoke(
    prompt="Quero a segunda via",
    context=context,
)
```

Neste exemplo, se o usuário estiver em `registered`, o LLM não deve ver `segunda_via_sem_login`. Se estiver em `guest`, o LLM não deve ver `segunda_via_com_cadastro`.

---

### 8.4. Exemplo com ToolPolicy avançada

```python
from adjacency_agents import ToolPolicy, tool_node


@tool_node(
    policy=ToolPolicy(
        all_of={"registered", "active_account"},
        none_of={"blocked", "fraud_suspected"},
    )
)
def consultar_area_restrita() -> str:
    """Consulta informação disponível apenas para conta ativa e não bloqueada."""
    return "Área restrita liberada."
```

---

### 8.5. Exemplo com transferência estrutural

```python
from adjacency_agents import EnrichedPointer, tool_node


@tool_node(
    requires=["registered"],
    structural_neighbors=["consultar_resultado_detalhado"],
)
def buscar_item_recente() -> EnrichedPointer | str:
    """Busca item recente e redireciona para o detalhamento se encontrado."""
    item = db.get_last_item()

    if item:
        return EnrichedPointer(
            next_tool="consultar_resultado_detalhado",
            kwargs={"item_id": item.id},
            reason="Item recente encontrado. Transferindo para detalhamento.",
        )

    return "Nenhum item recente encontrado."


@tool_node(
    requires=["registered"],
    llm_visible=False,
)
def consultar_resultado_detalhado(item_id: str) -> dict:
    """Consulta detalhes de um item específico."""
    return db.get_item_details(item_id)
```

Neste exemplo, o LLM pode chamar `buscar_item_recente`, mas não deve ver `consultar_resultado_detalhado`. A segunda tool só pode ser executada via `EnrichedPointer` válido.

---

### 8.6. Exemplo com injeção de contexto

```python
from adjacency_agents import UserContext, tool_node


@tool_node(
    requires=["registered"],
    inject={"registration_id": "metadata.registration_id"},
)
def consultar_dados_do_cadastro(registration_id: str) -> dict:
    """Consulta dados do cadastro já identificado pela aplicação."""
    return api.get_registration(registration_id)


context = UserContext(
    session_id="whatsapp_123",
    capabilities={"public", "registered"},
    metadata={"registration_id": "abc-123"},
)
```

Neste exemplo, `registration_id` não aparece no schema enviado ao LLM. O valor é resolvido pelo engine a partir de `UserContext.metadata` no momento da execução.

---

## 9. Modelos de dados

### 9.1. UserContext

Contrato sugerido:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class UserContext:
    session_id: str
    capabilities: set[str]
    metadata: dict[str, Any] = field(default_factory=dict)
```

Regras:

1. `session_id` deve ser obrigatório.
2. `capabilities` deve ser um conjunto de strings.
3. `metadata` deve ser opcional.
4. O engine não deve alterar o `UserContext` recebido.
5. A aplicação cliente é responsável por montar o contexto a partir de dados confiáveis.

---

### 9.2. Message

Contrato sugerido:

```python
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

Regras:

1. `Message` representa histórico conversacional, não estado confiável de autorização.
2. Histórico deve ser passado em `invoke(..., messages=[...])` ou `ainvoke(..., messages=[...])`.
3. `UserContext` não deve carregar histórico por padrão; ele carrega fatos confiáveis e capabilities.
4. A aplicação cliente é responsável por persistir e recuperar histórico entre webhooks.
5. O engine não deve mutar a lista de mensagens recebida.
6. Se `prompt` e `messages` forem fornecidos juntos, `prompt` deve ser anexado logicamente como a última mensagem `user`, sem alterar a lista original.
7. Mensagens de role `tool` criadas pelo engine para síntese são internas ao ciclo de execução e não devem ser expostas diretamente ao usuário.

---

### 9.3. Observation

`Observation` representa dados produzidos por uma tool que ainda precisam ser sintetizados pelo LLM antes de virar resposta final ao usuário.

Contrato sugerido:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Observation:
    data: Any
    summary_hint: str | None = None
    expose_to_llm: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
```

Regras:

1. `Observation` não é resposta final ao usuário.
2. `Observation` deve ser serializada de forma segura e anexada como mensagem interna de tool.
3. Após uma `Observation`, o engine pode fazer uma chamada final ao LLM para síntese.
4. A chamada de síntese final deve ocorrer com `tools=[]` ou mecanismo equivalente que impeça novas ToolCalls.
5. A síntese não pode abrir novo roteamento de ferramentas.
6. Dados sensíveis devem ser mascarados antes de virar conteúdo de `Observation`, quando a aplicação ou a tool assim definir.
7. Se `expose_to_llm=False`, a observation só pode ir para trace/log interno seguro, não para síntese.

---

### 9.4. ToolPolicy

Contrato sugerido:

```python
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolPolicy:
    all_of: set[str] = field(default_factory=set)
    any_of: set[str] = field(default_factory=set)
    none_of: set[str] = field(default_factory=set)

    def allows(self, capabilities: set[str]) -> bool:
        if self.all_of and not self.all_of <= capabilities:
            return False

        if self.any_of and not self.any_of & capabilities:
            return False

        if self.none_of and self.none_of & capabilities:
            return False

        if not self.all_of and not self.any_of:
            return False

        return True
```

Regras:

1. `all_of` exige todas as capabilities listadas.
2. `any_of` exige pelo menos uma das capabilities listadas.
3. `none_of` bloqueia se qualquer capability listada estiver presente.
4. Quando `all_of`, `any_of` e `none_of` são combinados, a semântica entre os grupos é `AND`: `all_of` precisa passar, `any_of` precisa passar quando não estiver vazio, e `none_of` não pode encontrar interseção.
5. `none_of` sozinho não concede acesso; ele apenas bloqueia.
6. Política vazia deve negar acesso por padrão.
7. `requires=[...]` no decorador deve ser atalho para `ToolPolicy(all_of={...})`.

Forma lógica:

```python
allowed = (not all_of or all_of <= capabilities) \
    and (not any_of or bool(any_of & capabilities)) \
    and not bool(none_of & capabilities) \
    and bool(all_of or any_of)
```

---

### 9.5. EnrichedPointer

Contrato sugerido:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnrichedPointer:
    next_tool: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
```

Regras:

1. `next_tool` deve apontar para tool existente.
2. `kwargs` deve ser validado contra o schema da próxima tool.
3. `reason` é útil para trace, debug e auditoria.
4. `EnrichedPointer` não deve ser enviado diretamente ao usuário como resposta final.

---

### 9.6. ToolCall

Contrato sugerido:

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)
```

Regras:

1. `ToolCall` representa a escolha do LLM.
2. O engine deve validar se `name` está na allowlist visível.
3. O engine deve validar `kwargs` contra o schema da tool.
4. ToolCall inválida não deve ser executada.

---

### 9.7. FinalAnswer

Contrato sugerido:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class FinalAnswer:
    content: str
```

Regras:

1. Pode ser retornado pelo LLM adapter.
2. Pode ser retornado pelo engine após normalização de resultado final.
3. Deve conter texto seguro para o usuário final.

---

## 10. Decorador `@tool_node`

### 10.1. Assinatura esperada

```python
def tool_node(
    *,
    name: str | None = None,
    requires: list[str] | set[str] | None = None,
    policy: ToolPolicy | None = None,
    structural_neighbors: list[str] | set[str] | None = None,
    llm_visible: bool = True,
    description: str | None = None,
    response_mode: str = "auto",  # "auto" | "final" | "synthesize"
    inject: dict[str, str] | None = None,
):
    ...
```

### 10.2. Regras

1. `requires` e `policy` não devem ser usados juntos.
2. Se `requires` for usado, deve virar `ToolPolicy(all_of=set(requires))`.
3. Se nem `requires` nem `policy` forem fornecidos, a tool deve ser negada por padrão.
4. `name`, se omitido, deve ser o nome da função.
5. `description`, se omitida, deve vir da docstring.
6. A função decorada deve preservar sua identidade Python original quando possível.
7. O decorador deve anexar metadados à função; não deve depender de registry global obrigatório.
8. Funções sem type hints nos parâmetros devem falhar no registro ou no schema builder.
9. Funções sem docstring devem ser permitidas, mas o schema terá descrição limitada.
10. `llm_visible=False` remove a tool do schema enviado ao LLM, mas não da allowlist de execução estrutural.
11. `response_mode` define como resultados comuns da tool serão tratados quando ela não retornar `FinalAnswer`, `Observation` ou `EnrichedPointer`.
12. `response_mode="auto"` deve tratar `str` como resposta final e estruturas como `dict`, `list`, `BaseModel` ou dataclass como `Observation` para síntese.
13. `inject` declara argumentos resolvidos pelo engine a partir de `UserContext`, e esses argumentos não devem aparecer no schema do LLM.
14. Chaves de `inject` devem corresponder a parâmetros existentes na assinatura da função.
15. No MVP, valores de `inject` devem suportar pelo menos `session_id` e caminhos `metadata.<campo>` ou `metadata.<campo_aninhado>`.
16. Argumentos injetados não podem ser fornecidos pelo LLM nem por `EnrichedPointer`; se aparecerem em kwargs externos, a chamada deve falhar antes da execução.

---

## 11. Registry

### 11.1. Diretiva

O registry deve ser instância atrelada à engine, não singleton global obrigatório.

Motivos:

1. Melhor isolamento em testes.
2. Melhor suporte a múltiplos tenants.
3. Melhor suporte a múltiplos bots.
4. Menor risco de estado global invisível.
5. Melhor previsibilidade em execução paralela.

### 11.2. Responsabilidades

O `ToolRegistry` deve:

1. Receber lista de funções decoradas.
2. Extrair `ToolNodeSpec` de cada função.
3. Validar nomes duplicados.
4. Validar `structural_neighbors` existentes.
5. Armazenar tools por nome.
6. Expor busca por nome.
7. Expor iteração sobre tools.

---

## 12. Router

### 12.1. Função principal

```python
def build_allowlist(registry: ToolRegistry, context: UserContext) -> list[ToolNode]:
    ...
```

### 12.2. Regra central

Uma tool entra na allowlist se:

```python
tool.policy.allows(context.capabilities) is True
```

### 12.3. LLM-visible allowlist

```python
def build_llm_visible_allowlist(allowlist: list[ToolNode]) -> list[ToolNode]:
    return [tool for tool in allowlist if tool.llm_visible]
```

### 12.4. Diretivas

1. O router não deve chamar LLM.
2. O router não deve executar tools.
3. O router não deve inferir capabilities a partir de metadata.
4. O router deve ser determinístico.
5. O router deve ser facilmente testável com entrada e saída puras.

---

## 13. Schema builder

### 13.1. Decisão arquitetural

O projeto deve adotar **Pydantic v2** como dependência central do módulo `schema.py`.

Motivo: gerar JSON Schema, validar kwargs e lidar com tipos como `Optional`, `list`, `dict`, `Literal`, `Enum`, modelos aninhados e estruturas compostas manualmente com `typing` nativo criaria um parser próprio caro, frágil e difícil de manter.

Dependência esperada no `pyproject.toml`:

```toml
dependencies = [
    "pydantic>=2.7,<3",
]
```

### 13.2. Objetivo

Gerar schemas de ferramentas para o LLM apenas a partir da LLM-visible allowlist e validar argumentos antes de execução.

### 13.3. Regras

1. Deve usar type hints da função como fonte primária.
2. Deve usar Pydantic v2 para criar modelos de entrada das tools.
3. Deve falhar para parâmetros sem type hint.
4. Deve incluir descrição da tool.
5. Deve incluir descrição dos argumentos quando possível.
6. Deve excluir argumentos injetados pelo contexto, quando o recurso de injeção existir.
7. Deve produzir um schema interno independente de provedor no core.
8. Adapters específicos podem converter o schema interno para OpenAI, Anthropic, Ollama, JSON mode ou outro formato.
9. Kwargs recebidos do LLM ou de `EnrichedPointer` devem ser validados pelo modelo Pydantic da tool.
10. Argumentos extras devem ser bloqueados por padrão.
11. `Any` deve ser permitido apenas quando declarado explicitamente, mas deve ser evitado em tools sensíveis.
12. Modelos Pydantic declarados pelo usuário devem ser preservados sempre que possível.
13. O core não deve manter parser próprio para cobrir todos os casos de `typing`.

### 13.4. Diretiva de compatibilidade com provedores

Pydantic deve gerar o schema interno. O adapter do provedor é responsável por ajustar limitações específicas do destino.

Exemplo: um provedor pode não aceitar todos os recursos de JSON Schema. Nesse caso, o adapter converte, reduz ou rejeita o schema com erro claro.

---

## 14. Engine determinístico

### 14.1. Assinatura esperada

```python
from collections.abc import Callable, Sequence
from typing import Literal


class DeterministicEngine:
    def __init__(
        self,
        *,
        llm: LLMClient | AsyncLLMClient,
        tools: list[Callable],
        max_steps: int = 8,
        default_tool_result_mode: Literal["auto", "final", "synthesize"] = "auto",
        sync_tool_strategy: Literal["thread", "direct"] = "thread",
        tool_error_mode: Literal["raise", "final", "synthesize"] = "raise",
        default_tool_error_message: str = "I could not complete this operation right now.",
    ) -> None:
        ...

    def invoke(
        self,
        *,
        context: UserContext,
        prompt: str | None = None,
        messages: Sequence[Message] | None = None,
    ) -> FinalAnswer:
        ...

    async def ainvoke(
        self,
        *,
        context: UserContext,
        prompt: str | None = None,
        messages: Sequence[Message] | None = None,
    ) -> FinalAnswer:
        ...
```

### 14.2. Regras de entrada conversacional

1. `prompt` é conveniência para turno único.
2. `messages` é o contrato recomendado para fluxos multi-turno.
3. Deve ser fornecido `prompt`, `messages` ou ambos.
4. Se `prompt` e `messages` forem fornecidos juntos, `prompt` representa a nova mensagem do usuário e deve ser anexado logicamente ao final do histórico.
5. O engine não deve persistir histórico entre chamadas.
6. O engine não deve mutar a sequência de mensagens recebida.
7. `UserContext` não substitui `messages`; ele representa estado confiável e capabilities.

### 14.3. Sincronismo e assincronismo

1. `ainvoke` é o caminho recomendado para produção.
2. `ainvoke` deve suportar tools declaradas com `def` e `async def`.
3. `ainvoke` deve suportar LLM clients síncronos e assíncronos quando possível.
4. `invoke` é conveniência para scripts, testes e aplicações síncronas.
5. `invoke` pode executar a stack assíncrona usando `asyncio.run(...)` apenas quando não houver event loop ativo.
6. `invoke` não deve tentar executar silently uma stack assíncrona dentro de event loop já ativo. Nesse caso, deve lançar `AsyncRequiredError` com mensagem clara orientando o uso de `await engine.ainvoke(...)`.
7. O decorador deve preservar se a função original é coroutine.
8. Em `ainvoke`, tools `async def` devem ser aguardadas diretamente.
9. Em `ainvoke`, tools `def` devem seguir `sync_tool_strategy`.
10. `sync_tool_strategy="thread"` deve executar tools síncronas via `asyncio.to_thread` ou mecanismo equivalente, para evitar bloqueio do event loop. Esse deve ser o padrão de produção.
11. `sync_tool_strategy="direct"` pode executar tools síncronas diretamente, útil para testes, scripts e tools comprovadamente rápidas.
12. O engine não deve tentar detectar automaticamente se uma tool síncrona é bloqueante. A estratégia deve ser explícita e previsível.

### 14.4. Tratamento de resultado da tool

A biblioteca deve suportar três caminhos explícitos após execução de uma tool:

1. `EnrichedPointer`: transição estrutural determinística para outra tool.
2. `FinalAnswer`: resposta final human-safe, sem nova chamada ao LLM.
3. `Observation`: dado bruto/estruturado que deve voltar ao LLM para síntese final.

Resultados comuns, como `str`, `dict` ou `list`, devem ser tratados conforme `response_mode` da tool ou `default_tool_result_mode` do engine.

Semântica recomendada para `response_mode="auto"`:

1. `str` -> `FinalAnswer` direto.
2. `FinalAnswer` -> `FinalAnswer` direto.
3. `EnrichedPointer` -> validação e transição.
4. `Observation` -> síntese final.
5. `dict`, `list`, `BaseModel`, dataclass ou outros objetos estruturados -> `Observation` para síntese final.

### 14.5. Loop esperado

Fluxo conceitual:

```text
1. Normalizar entrada conversacional em messages.
2. Criar allowlist a partir do contexto.
3. Criar LLM-visible allowlist.
4. Gerar schemas apenas das tools visíveis.
5. Chamar LLM adapter com messages e schemas.
6. Se LLM retornar FinalAnswer, encerrar.
7. Se LLM retornar ToolCall, validar tool e kwargs.
8. Executar tool.
9. Se resultado for EnrichedPointer, validar transição.
10. Executar próxima tool sem chamar LLM.
11. Se resultado for FinalAnswer, retornar.
12. Se resultado for Observation, anexar observation ao histórico interno.
13. Chamar LLM para síntese final com tools desabilitadas.
14. Retornar FinalAnswer sintetizada.
15. Repetir transições estruturais até resultado final, Observation ou max_steps.
```

### 14.6. Pseudocódigo

```python
async def ainvoke(self, *, context: UserContext, prompt=None, messages=None):
    conversation = normalize_messages(prompt=prompt, messages=messages)

    allowlist = build_allowlist(self.registry, context)
    visible_tools = build_llm_visible_allowlist(allowlist)
    schemas = build_schemas(visible_tools)

    model_output = await self.call_llm(
        messages=conversation,
        tools=schemas,
        allow_tool_calls=True,
    )

    if isinstance(model_output, FinalAnswer):
        return model_output

    if isinstance(model_output, str):
        return FinalAnswer(model_output)

    current_call = validate_tool_call(
        model_output,
        visible_tools,
        context,
        reject_injected_kwargs=True,
    )

    steps = 0

    while steps < self.max_steps:
        steps += 1

        try:
            result = await execute_tool_call(
                current_call,
                context,
                resolve_injected_kwargs=True,
            )
        except Exception as exc:
            return handle_tool_execution_error(
                exc,
                tool_name=current_call.name,
                mode=self.tool_error_mode,
                conversation=conversation,
            )

        if isinstance(result, EnrichedPointer):
            current_call = validate_pointer(
                pointer=result,
                current_tool=current_call.name,
                registry=self.registry,
                allowlist=allowlist,
                context=context,
                reject_injected_kwargs=True,
            )
            continue

        normalized = normalize_tool_result(
            result,
            tool=current_call.name,
            mode=get_response_mode(current_call.name),
        )

        if isinstance(normalized, FinalAnswer):
            return normalized

        if isinstance(normalized, Observation):
            synthesis_messages = build_synthesis_messages(
                conversation=conversation,
                tool_name=current_call.name,
                observation=normalized,
            )
            synthesis = await self.call_llm(
                messages=synthesis_messages,
                tools=[],
                allow_tool_calls=False,
            )
            return normalize_final_answer(synthesis)

    raise MaxStepsExceededError(...)
```

### 14.7. Diretiva de síntese final

A chamada final de síntese existe para transformar dado estruturado em linguagem natural. Ela não pode ser usada para escolher novas tools.

Portanto:

1. A síntese deve ser chamada com `tools=[]` ou equivalente.
2. O adapter deve impedir ToolCall na síntese.
3. Se o LLM retornar ToolCall durante síntese, o engine deve tratar como `InvalidToolCallError` ou `SynthesisError`.
4. O engine deve registrar no trace que houve síntese.
5. A aplicação deve poder desabilitar síntese globalmente ou por tool.
6. A síntese deve receber apenas o histórico conversacional normalizado, a `Observation` sanitizada e uma instrução interna de síntese.
7. A síntese não deve receber catálogo completo, allowlist, schemas, capabilities, metadata bruta ou trace completo.
8. A `Observation` deve ser tratada como fonte de verdade para os dados recuperados pela tool; o histórico serve para intenção, idioma, tom e contexto conversacional.
9. A instrução de síntese deve orientar o modelo a não revelar nomes internos de tools, ponteiros, policies, capabilities ou detalhes de implementação.
10. Se `Observation.expose_to_llm=False`, o engine não deve enviá-la à síntese. Nesse caso, deve retornar `FinalAnswer` seguro se houver fallback configurado ou levantar `SynthesisError`.

---

### 14.8. Semântica de turno único

Cada invocação executa no máximo uma cadeia:

```text
LLM -> ToolCall inicial -> Tool A -> EnrichedPointer -> Tool B -> ... -> FinalAnswer ou Observation -> síntese sem tools
```

Não existe neste MVP o seguinte fluxo:

```text
LLM -> Tool A -> Observation -> LLM escolhe Tool B -> Tool B -> ...
```

Essa restrição é intencional. Ela preserva determinismo, reduz loops imprevisíveis e impede que uma etapa de síntese recupere poder de roteamento.

---

### 14.9. Erros de runtime de tools

Falhas de execução da função do usuário são diferentes de falhas de validação. Exemplos:

```text
timeout de API externa
falha de conexão com banco
KeyError ou ValueError dentro da tool
erro de serialização
resposta inesperada de serviço externo
```

Diretiva:

1. O engine deve capturar exceções não controladas lançadas por tools.
2. Toda exceção levantada dentro do corpo de uma tool deve ser tratada como falha de runtime da tool e encapsulada em `ToolExecutionError`, preservando a exceção original como `__cause__` quando possível.
3. Essa regra também vale se a tool levantar manualmente uma subclasse de `AdjacencyAgentsError`, como `ToolNotAllowedError` ou `InvalidTransitionError`. Dentro da tool, essas exceções não representam uma decisão de política do engine; representam falha de execução da tool do usuário.
4. Exceções de validação e política levantadas pelo próprio engine, antes ou depois da execução da tool, mantêm seus tipos específicos.
5. Por padrão, `tool_error_mode="raise"` deve abortar o turno e propagar `ToolExecutionError` para a aplicação cliente.
6. A aplicação cliente decide como transformar `ToolExecutionError` em mensagem ao usuário final.
7. `tool_error_mode="final"` pode retornar `FinalAnswer(default_tool_error_message)` sem chamar LLM.
8. `tool_error_mode="synthesize"` pode criar uma `Observation` sanitizada de erro e chamar síntese sem tools, mas somente se explicitamente configurado.
9. Se a falha ocorrer no meio de uma cadeia `EnrichedPointer`, como `A -> B -> C`, a síntese de erro não deve receber nomes internos de tools, quantidade de saltos, ponteiros, grafo, trace ou o ponto exato da falha. O trace interno pode registrar esses detalhes para auditoria, mas a síntese deve receber apenas uma descrição segura e genérica do erro.
10. Tracebacks, mensagens internas, SQL, URLs privadas, tokens e payloads sensíveis nunca devem ser enviados ao LLM ou ao usuário por padrão.
11. Erros de negócio esperados devem preferencialmente ser modelados pela própria tool retornando `FinalAnswer` ou `Observation`, não por exceções genéricas.

O valor padrão de `default_tool_error_message` deve ser neutro e em inglês para publicação como biblioteca. Aplicações finais devem sobrescrevê-lo para seu idioma, tom de marca e canal.

---

## 15. Validação de ToolCall

O engine deve validar:

1. A tool existe.
2. A tool está na LLM-visible allowlist.
3. A tool está permitida pelo contexto.
4. Os kwargs batem com o schema.
5. Argumentos extras são tratados conforme política definida.
6. Argumentos obrigatórios estão presentes.
7. Argumentos injetados não foram fornecidos pelo LLM.
8. A validação dos kwargs externos deve ocorrer antes da resolução de argumentos injetados.
9. Após resolver argumentos injetados, o conjunto final de argumentos deve ser validado novamente contra a assinatura completa da tool.

ToolCall inválida deve gerar erro controlado, não execução parcial.

---

## 16. Validação de EnrichedPointer

O engine deve validar:

1. `pointer.next_tool` existe no registry.
2. `pointer.next_tool` está em `current_tool.structural_neighbors`.
3. `pointer.next_tool` está na allowlist de execução do contexto.
4. A próxima tool pode ter `llm_visible=False`, desde que esteja na allowlist de execução.
5. `pointer.kwargs` é compatível com o schema da próxima tool.
6. `pointer.kwargs` não contém argumentos marcados como injetados.
7. O limite de passos não foi excedido.

Falhas nessa validação são violações de política ou grafo, não erros recuperáveis pelo LLM.

Ciclos estruturais como `A -> B -> A -> B` não precisam ser proibidos estaticamente pelo registry no MVP. Eles devem ser interrompidos por `max_steps`. A documentação e os testes devem deixar claro que a proteção obrigatória contra ciclos é operacional, não análise completa do grafo.

---

## 17. Injeção de contexto

### 17.1. Problema

Alguns argumentos não devem ser preenchidos pelo LLM porque já são conhecidos pela aplicação.

Exemplos:

```text
user_id
registration_id
tenant_id
account_id
channel
session_id
customer_status
```

Esses valores fazem parte do estado confiável da sessão e não devem ser inferidos, corrigidos ou inventados pelo modelo.

### 17.2. Decisão de MVP

Injeção mínima de contexto faz parte do MVP.

A API obrigatória do MVP é declarativa no decorador:

```python
@tool_node(
    requires=["registered"],
    inject={"registration_id": "metadata.registration_id"},
)
def consultar_dados(registration_id: str) -> dict:
    ...
```

A API alternativa com `Injected[...]` pode ser adicionada futuramente, mas não é requisito do MVP:

```python
from adjacency_agents import Injected


@tool_node(requires=["registered"])
def consultar_dados(registration_id: Injected["registration_id"]) -> dict:
    ...
```

### 17.3. Fontes de injeção no MVP

O MVP deve suportar pelo menos:

```text
session_id
metadata.<campo>
metadata.<campo_aninhado>
```

Exemplos:

```python
inject={"session": "session_id"}
inject={"registration_id": "metadata.registration_id"}
inject={"tenant_id": "metadata.account.tenant_id"}
```

### 17.4. Regras

1. Argumentos injetados não devem aparecer no schema do LLM.
2. Argumentos injetados não devem ser aceitos do LLM.
3. Argumentos injetados não devem ser aceitos de `EnrichedPointer`.
4. Argumentos injetados devem ser resolvidos no momento da execução.
5. Ausência de valor injetável deve gerar `ContextInjectionError`.
6. Dados de metadata não devem conceder permissão diretamente; permissão vem de capabilities.
7. O schema visível ao LLM deve validar apenas argumentos não injetados.
8. O conjunto final de argumentos, após injeção, deve ser validado pela assinatura completa da tool.
9. Valores injetados devem vencer qualquer tentativa externa de preenchimento, mas a tentativa externa deve ser tratada como erro, não sobrescrita silenciosa.
10. Injeção não deve ser usada para esconder autorização. Ela serve para passagem segura de fatos confiáveis, não para decidir policy.

---

## 18. Contrato do LLM client

### 18.1. Protocol síncrono sugerido

```python
from collections.abc import Sequence
from typing import Protocol


class LLMClient(Protocol):
    def complete(
        self,
        *,
        messages: Sequence[Message],
        tools: list[dict],
        allow_tool_calls: bool = True,
    ) -> ToolCall | FinalAnswer | str:
        ...
```

### 18.2. Protocol assíncrono sugerido

```python
from collections.abc import Sequence
from typing import Protocol


class AsyncLLMClient(Protocol):
    async def acomplete(
        self,
        *,
        messages: Sequence[Message],
        tools: list[dict],
        allow_tool_calls: bool = True,
    ) -> ToolCall | FinalAnswer | str:
        ...
```

### 18.3. Diretivas

1. O core não deve depender de provedor específico.
2. O LLM client recebe apenas tools visíveis e permitidas.
3. O LLM client recebe `messages`, não apenas `prompt`.
4. O LLM client pode usar tool calling nativo ou JSON mode.
5. O engine deve validar qualquer saída do LLM.
6. O LLM client não deve receber tools fora da allowlist.
7. Quando `allow_tool_calls=False`, o adapter deve impedir tool calling nativo e rejeitar JSON de ToolCall.
8. A chamada de síntese final deve usar `allow_tool_calls=False`.
9. O Fake LLM de testes deve suportar respostas programadas para ToolCall, FinalAnswer e síntese.
10. Argumentos de passagem do adapter (`extra_create_kwargs` / `extra_chat_kwargs`) não podem sobrescrever chaves controladas pelo engine (`tools`, `tool_choice`, `messages`, `model` e equivalentes do provedor). O adapter deve rejeitar essas chaves na construção, preservando o contrato de sandbox da síntese (Invariante §7 da §8).
11. Se o provedor retornar mais de uma tool call em uma única resposta, o adapter deve rejeitar com `InvalidToolCallError` (um turno = uma cadeia, §4.7), nunca executar silenciosamente apenas a primeira.

---

## 19. Tratamento de erros

### 19.1. Erros internos esperados

Criar módulo `errors.py` com exceções controladas:

```python
class AdjacencyAgentsError(Exception): ...
class ToolNotFoundError(AdjacencyAgentsError): ...
class ToolNotAllowedError(AdjacencyAgentsError): ...
class InvalidToolCallError(AdjacencyAgentsError): ...
class InvalidTransitionError(AdjacencyAgentsError): ...
class InvalidToolSchemaError(AdjacencyAgentsError): ...
class MaxStepsExceededError(AdjacencyAgentsError): ...
class ContextInjectionError(AdjacencyAgentsError): ...
class ToolExecutionError(AdjacencyAgentsError): ...
class AsyncRequiredError(AdjacencyAgentsError): ...
class SynthesisError(AdjacencyAgentsError): ...
```

### 19.2. Diretiva

A biblioteca pode lançar exceções internas controladas. A aplicação cliente decide como transformar isso em resposta final ao usuário.

O engine não deve executar uma tool após violação de política.

Exceções levantadas pelo próprio engine durante validação, roteamento, policy, schema, injeção, síntese ou controle de passos devem preservar seus tipos específicos, como `ToolNotAllowedError`, `InvalidTransitionError` ou `MaxStepsExceededError`.

Exceções levantadas dentro do corpo de uma tool do usuário devem sempre ser remapeadas para `ToolExecutionError`, inclusive quando forem subclasses de `AdjacencyAgentsError`. Isso impede que uma tool faça uma falha de runtime parecer uma violação de política emitida pelo engine. A exceção original deve ser preservada como causa quando possível.

O conteúdo bruto da exceção não deve ser enviado ao LLM nem ao usuário por padrão.

---

## 20. ExecutionTrace

### 20.1. Objetivo

Permitir auditoria e debug de decisões do engine.

### 20.2. Eventos mínimos

1. `allowlist_built`
2. `llm_called`
3. `tool_call_received`
4. `tool_call_validated`
5. `tool_executed`
6. `tool_execution_failed`
7. `pointer_received`
8. `pointer_validated`
9. `transition_executed`
10. `observation_created`
11. `synthesis_requested`
12. `synthesis_completed`
13. `final_answer_returned`
14. `policy_denied`
15. `max_steps_exceeded`
16. `context_injection_resolved`
17. `context_injection_failed`

### 20.3. Diretiva de segurança

Logs e traces não devem expor dados sensíveis por padrão. Deve existir forma de mascarar ou omitir valores.

---

## 21. Estrutura de diretórios

Estrutura esperada:

```text
adjacency-agents/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   ├── ddd.md
│   ├── architecture.md
│   ├── api.md
│   ├── security.md
│   └── testing.md
├── examples/
│   ├── fake_llm_example.py
│   ├── logged_vs_guest_example.py
│   └── structural_pointer_example.py
├── src/
│   └── adjacency_agents/
│       ├── __init__.py
│       ├── decorators.py
│       ├── engine.py
│       ├── errors.py
│       ├── llm.py
│       ├── models.py
│       ├── registry.py
│       ├── router.py
│       ├── schema.py
│       └── tracing.py
├── tests/
│   ├── test_decorators.py
│   ├── test_async_engine.py
│   ├── test_engine.py
│   ├── test_messages.py
│   ├── test_pointer_transitions.py
│   ├── test_registry.py
│   ├── test_router.py
│   ├── test_schema.py
│   ├── test_synthesis.py
│   ├── test_tool_execution_errors.py
│   └── test_context_injection.py
├── .gitignore
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 22. Diretivas por módulo

### 22.1. `models.py`

Deve conter modelos puros:

- `UserContext`
- `Message`
- `ToolPolicy`
- `EnrichedPointer`
- `Observation`
- `ToolCall`
- `FinalAnswer`
- `ToolNodeSpec`
- `ExecutionTrace`, se simples

Não deve conter lógica de LLM.

---

### 22.2. `decorators.py`

Deve conter:

- `tool_node`
- validação inicial leve
- anexação de metadados à função

Não deve executar registro global obrigatório.

---

### 22.3. `registry.py`

Deve conter:

- `ToolRegistry`
- validação de duplicidade
- validação de vizinhos estruturais
- lookup por nome

---

### 22.4. `router.py`

Deve conter:

- `build_allowlist`
- `build_llm_visible_allowlist`
- helpers de policy

Deve ser determinístico e testável sem LLM.

---

### 22.5. `schema.py`

Deve conter:

- extração de assinatura Python;
- criação de modelos Pydantic v2 por tool;
- geração de schema interno;
- validação de kwargs;
- bloqueio de argumentos extras;
- suporte a argumentos injetados do MVP.

---

### 22.6. `llm.py`

Deve conter:

- `LLMClient` Protocol;
- `AsyncLLMClient` Protocol;
- `FakeLLMClient` para testes e exemplos;
- eventuais adapters opcionais no futuro.

---

### 22.7. `engine.py`

Deve conter:

- `DeterministicEngine`;
- `invoke` e `ainvoke`;
- loop de execução;
- validação de ToolCall;
- validação de EnrichedPointer;
- suporte a `Observation` e síntese final;
- limite de passos;
- normalização de resposta final;
- tratamento de `ToolExecutionError`;
- construção de trace.

---

### 22.8. `errors.py`

Deve conter exceções controladas e específicas.

---

### 22.9. `tracing.py`

Deve conter modelos e utilitários de trace, se não ficarem em `models.py`.

---

## 23. Invariantes obrigatórios

A implementação deve preservar estes invariantes:

1. Tool sem policy explícita não entra na allowlist.
2. Tool fora da allowlist não é enviada ao LLM.
3. Tool com `llm_visible=False` não é enviada ao LLM.
4. Tool fora da allowlist nunca é executada.
5. ToolCall do LLM sempre é validada antes da execução.
6. EnrichedPointer sempre é validado antes da execução da próxima tool.
7. Transição estrutural só ocorre para vizinho declarado da tool atual.
8. Transição estrutural também respeita ToolPolicy.
9. Kwargs sempre são validados antes da execução.
10. Argumentos injetados nunca aparecem no schema do LLM.
11. O LLM nunca pode fornecer argumento marcado como injetado.
12. `EnrichedPointer` nunca pode fornecer argumento marcado como injetado.
13. Argumentos injetados são resolvidos pelo engine e validados antes da chamada final da função.
14. O engine deve ter `max_steps`.
15. Ciclos não podem rodar indefinidamente; no MVP, a proteção obrigatória é `max_steps`.
16. Registry não deve depender de singleton global obrigatório.
17. A aplicação cliente é responsável por gerar capabilities.
18. O framework não interpreta semanticamente nomes de capabilities.
19. O schema do LLM deve ser derivado da allowlist atual, nunca do catálogo completo.
20. Histórico conversacional deve ser aceito como `messages` e não misturado com `UserContext`.
21. Cada turno executa no máximo uma cadeia iniciada por uma ToolCall do LLM.
22. Síntese final não pode permitir novas ToolCalls.
23. `Observation` nunca deve ser retornada diretamente ao usuário.
24. Tools `async def` devem ser executáveis via `ainvoke`.
25. Em `ainvoke`, tools `def` não devem bloquear o event loop por padrão; devem seguir `sync_tool_strategy`.
26. Schema e validação de kwargs devem usar Pydantic v2.
27. Falhas de política não devem ser corrigidas por prompt.
28. Exceções de runtime de tools devem ser encapsuladas em erro controlado.
29. Dados sensíveis não devem aparecer em logs por padrão.
30. Dados sensíveis, tracebacks e detalhes internos de exceção não devem ser enviados ao LLM por padrão.
31. Kwargs de passagem do adapter nunca podem reintroduzir `tools`/`tool_choice` (ou equivalentes) na chamada de síntese; chaves controladas pelo engine são rejeitadas na construção do adapter.
32. Um adapter nunca executa silenciosamente apenas a primeira de múltiplas tool calls retornadas; múltiplas tool calls em uma resposta são rejeitadas com `InvalidToolCallError`.

---

## 24. Testes obrigatórios

### 24.1. Router

Testar:

1. `all_of` satisfeito permite tool.
2. `all_of` incompleto bloqueia tool.
3. `any_of` com uma capability presente permite tool.
4. `any_of` sem capabilities presentes bloqueia tool.
5. `none_of` presente bloqueia tool.
6. `all_of` e `any_of` combinados usam semântica AND entre grupos.
7. `none_of` sozinho não concede acesso.
8. Política vazia bloqueia tool.
9. Capabilities desconhecidas não concedem acesso.
10. Tool com `llm_visible=False` entra na allowlist de execução, mas não na allowlist visível.

---

### 24.2. Registry

Testar:

1. Registro de função decorada.
2. Erro para nome duplicado.
3. Erro para vizinho estrutural inexistente.
4. Lookup por nome.
5. Preservação da função original.

---

### 24.3. Decorator

Testar:

1. `requires` gera policy `all_of`.
2. `policy` explícita funciona.
3. `requires` e `policy` juntos geram erro.
4. Nome default vem da função.
5. Nome customizado funciona.
6. Docstring vira descrição.
7. `llm_visible=False` é preservado.

---

### 24.4. Schema

Testar:

1. Função com type hints gera schema via Pydantic v2.
2. Função sem type hint em parâmetro falha.
3. Kwargs obrigatórios ausentes falham.
4. Kwargs extras são bloqueados por padrão.
5. Tipo incompatível falha.
6. Tipos compostos como `list[str]`, `dict[str, int]`, `Optional[str]` e `Literal[...]` são validados.
7. Modelo Pydantic declarado pelo usuário é preservado ou convertido corretamente.
8. Argumento injetado não aparece no schema.
9. Argumento injetado fornecido pelo LLM falha antes da execução.
10. Argumento injetado fornecido por `EnrichedPointer` falha antes da execução.
11. Valor injetado ausente em `UserContext` gera `ContextInjectionError`.

---

### 24.5. Engine

Testar:

1. LLM só recebe tools permitidas.
2. LLM não recebe tools fora do cenário atual.
3. ToolCall válida executa tool.
4. ToolCall para tool inexistente falha.
5. ToolCall para tool não permitida falha.
6. ToolCall para tool não visível falha se vier do LLM.
7. Resultado string vira resposta final em `response_mode="auto"`.
8. Resultado dict/list vira `Observation` e dispara síntese final em `response_mode="auto"`.
9. `Observation` explícita dispara síntese final.
10. Síntese final é chamada sem tools.
11. ToolCall retornada durante síntese final falha.
12. Após `Observation`, o LLM não pode escolher uma segunda tool no mesmo turno.
13. `EnrichedPointer` válido executa próxima tool sem chamar LLM novamente.
14. `EnrichedPointer` para não-vizinho falha.
15. `EnrichedPointer` para tool sem permissão falha.
16. `EnrichedPointer` para tool inexistente falha.
17. `EnrichedPointer` que tenta preencher argumento injetado falha.
18. `max_steps` interrompe ciclo.
19. Trace registra eventos essenciais.
20. Exceção levantada por tool vira `ToolExecutionError` em `tool_error_mode="raise"`.
21. Subclasse de `AdjacencyAgentsError` levantada dentro de tool também vira `ToolExecutionError`.
22. `tool_error_mode="final"` retorna mensagem segura configurada.
23. `tool_error_mode="synthesize"` não envia traceback bruto ao LLM.
24. Falha em cadeia `EnrichedPointer` com `tool_error_mode="synthesize"` não revela nomes de tools, saltos, ponteiros ou trace ao LLM de síntese.
25. Cadeia `A -> B -> Observation` dispara síntese final sem nova seleção de tool.
26. Cadeia `A -> B -> C -> Observation` respeita `max_steps`, valida cada aresta e dispara síntese final sem tools.
27. `messages` multi-turno são enviados ao LLM.
28. `prompt` é anexado logicamente ao histórico quando `messages` também é fornecido.
29. `ainvoke` executa tool `async def`.
30. `ainvoke` executa tool `def` via `sync_tool_strategy="thread"` por padrão.
31. `invoke` falha claramente quando precisa de stack assíncrona incompatível em event loop ativo.

---

### 24.6. Injeção de contexto

Testar:

1. Argumento injetado não aparece no schema.
2. Argumento injetado é resolvido de `metadata`.
3. Argumento injetado é resolvido de `session_id`.
4. Caminho de metadata ausente gera `ContextInjectionError`.
5. LLM tentando fornecer argumento injetado gera `InvalidToolCallError`.
6. `EnrichedPointer` tentando fornecer argumento injetado gera `InvalidTransitionError`.
7. Valor final após injeção é validado pelo modelo Pydantic completo.

---

### 24.7. Erros de execução de tools

Testar:

1. Exceção de tool é encapsulada em `ToolExecutionError`.
2. `ToolExecutionError` preserva a exceção original como causa quando possível.
3. Subclasse de `AdjacencyAgentsError` levantada dentro de tool também é encapsulada em `ToolExecutionError`.
4. Trace registra `tool_execution_failed`.
5. `tool_error_mode="raise"` propaga erro controlado.
6. `tool_error_mode="final"` retorna mensagem segura.
7. `tool_error_mode="synthesize"` envia apenas observação sanitizada à síntese.
8. Traceback bruto não aparece em prompt de síntese.
9. Erro em cadeia `A -> B -> C` não vaza nomes internos de tools, ponteiros, quantidade de saltos ou trace para a síntese.

---

### 24.8. Cadeias estruturais com síntese

Testar:

1. Cadeia `A -> B -> Observation` valida cada transição, executa síntese final sem tools e retorna `FinalAnswer`.
2. Cadeia `A -> B -> C -> Observation` valida cada aresta contra `structural_neighbors`, respeita policy de cada tool e retorna síntese final.
3. O LLM é chamado uma vez para seleção inicial e, se houver `Observation`, uma vez para síntese; não há chamada intermediária de roteamento.
4. Se `max_steps` for menor que a cadeia necessária, o engine interrompe antes da síntese com `MaxStepsExceededError`.
5. A síntese recebe apenas histórico normalizado, observation sanitizada e instrução interna; não recebe trace da cadeia.

### 24.9. Sandbox e contrato dos adapters de provedor

Para cada adapter (OpenAI, Anthropic, Ollama), testar:

1. Construir o adapter com `extra_*_kwargs` contendo uma chave controlada pelo engine (`tools`, `tool_choice`, `messages`, `model` ou equivalente) levanta `ValueError`.
2. Chaves não reservadas em `extra_*_kwargs` (ex.: `temperature`, `options`) continuam sendo encaminhadas ao provedor.
3. Uma resposta do provedor com mais de uma tool call em uma única mensagem levanta `InvalidToolCallError`, sem executar nenhuma tool.

---

## 25. Exemplo obrigatório de teste de cenário

Cenário: usuário sem cadastro.

```python
context = UserContext(
    session_id="s1",
    capabilities={"public", "guest"},
)
```

Tools:

```python
@tool_node(requires=["guest"])
def tool_guest() -> str:
    return "guest"


@tool_node(requires=["registered"])
def tool_registered() -> str:
    return "registered"
```

Expectativa:

1. `tool_guest` aparece para o LLM.
2. `tool_registered` não aparece para o LLM.
3. Se o LLM tentar chamar `tool_registered`, o engine bloqueia.

---

## 26. README mínimo obrigatório

O `README.md` deve conter:

1. Descrição curta do problema.
2. Descrição curta da solução.
3. Instalação.
4. Quickstart.
5. Exemplo de capabilities.
6. Exemplo de ToolPolicy.
7. Exemplo de `EnrichedPointer`.
8. Exemplo de `Observation` com síntese final.
9. Exemplo de `messages` em fluxo multi-turno.
10. Exemplo mínimo com `ainvoke`.
11. Exemplo de injeção de contexto.
12. Exemplo de tratamento de erro de tool.
13. Aviso de segurança: LLM não decide autorização.
14. Link para documentação avançada.
15. Status do projeto.

---

## 27. Segurança

### 27.1. Diretivas

1. Default deny.
2. Allowlist por turno.
3. Schema do LLM derivado da allowlist.
4. Validação antes de execução.
5. Ponteiros estruturais validados.
6. Logs seguros por padrão.
7. Sem singleton global para policy.
8. Sem inferência automática de permissões a partir de metadata.
9. Sem execução de tool desconhecida.
10. Sem fallback inseguro para catálogo completo.

### 27.2. Anti-patterns proibidos

```python
# Proibido: enviar catálogo completo ao LLM
llm.complete(prompt=prompt, tools=registry.all_tools)
```

```python
# Proibido: executar ToolCall sem validar allowlist
tool = registry.get(model_output.name)
tool.fn(**model_output.kwargs)
```

```python
# Proibido: executar EnrichedPointer sem validar vizinhança e policy
next_tool = registry.get(pointer.next_tool)
next_tool.fn(**pointer.kwargs)
```

```python
# Proibido: policy vazia significar público
@tool_node()
def ferramenta_publica_por_acidente():
    ...
```

---

## 28. Critérios de aceite do MVP

O MVP só está aceito quando:

1. Todos os testes obrigatórios passam.
2. Exemplo de cenário `guest` vs `registered` funciona.
3. Exemplo de `EnrichedPointer` funciona.
4. Tool fora da allowlist nunca chega ao LLM.
5. Tool fora da allowlist nunca executa.
6. Tool com `llm_visible=False` não aparece no schema.
7. Transição estrutural inválida é bloqueada.
8. `max_steps` bloqueia ciclos.
9. `messages` multi-turno são aceitos sem misturar histórico em `UserContext`.
10. Cada turno executa no máximo uma cadeia iniciada pelo LLM.
11. `Observation` dispara síntese final sem tools.
12. `dict`/`list` em `response_mode="auto"` não vazam como JSON bruto por padrão.
13. Injeção de contexto exclui argumentos do schema e resolve valores no momento da execução.
14. Exceções de runtime de tools viram `ToolExecutionError` controlado por padrão.
15. `ainvoke` executa tool `async def`.
16. `ainvoke` não bloqueia o event loop com tools `def` por padrão.
17. Schema e validação usam Pydantic v2.
18. README explica a tese da biblioteca.
19. Documentação descreve as invariantes de segurança.

---

## 29. Roadmap recomendado

### Fase 1 — Core determinístico sem LLM real

Entregar:

- modelos;
- `Message`;
- `Observation`;
- decorador;
- registry;
- router;
- fake LLM;
- engine básico;
- testes centrais.

### Fase 2 — Schema e validação robusta

Entregar:

- Pydantic v2 como dependência central;
- extração de schema;
- validação de kwargs;
- erros específicos;
- bloqueio de argumentos extras;
- type hints obrigatórios.

### Fase 3 — Transições estruturais

Entregar:

- `EnrichedPointer`;
- validação de vizinhos;
- `llm_visible=False`;
- `max_steps`;
- trace básico.

### Fase 4 — Injeção de contexto do MVP

Entregar:

- argumentos injetados;
- exclusão de schema;
- resolução por metadata;
- testes de segurança.

### Fase 5 — Histórico, síntese, async e falhas de produção

Entregar:

- `messages` multi-turno;
- síntese final a partir de `Observation`;
- bloqueio de ToolCall durante síntese;
- `ainvoke`;
- suporte a `async def`;
- `sync_tool_strategy`;
- `ToolExecutionError` e `tool_error_mode`.

### Fase 6 — Adapters de LLM

Entregar:

- adapter JSON mode;
- adapter para tool calling nativo;
- exemplos com modelos reais;
- documentação de integração.

---

## 30. Decisões de produção adicionadas na versão 0.2

### 30.1. Histórico de conversação

A API pública deve aceitar histórico conversacional por meio de `messages`.

Decisão:

```python
engine.invoke(context=context, prompt="...", messages=[...])
await engine.ainvoke(context=context, prompt="...", messages=[...])
```

`UserContext` não deve carregar histórico por padrão. A separação é intencional:

- `UserContext` contém estado confiável, capabilities e metadata da aplicação.
- `messages` contém conversa, que pode incluir texto não confiável vindo do usuário.

### 30.2. Retorno da tool para o LLM

A biblioteca deve incluir capacidade de síntese final.

Decisão:

- Tool que retorna `FinalAnswer` encerra o fluxo.
- Tool que retorna `EnrichedPointer` transfere deterministicamente para vizinho estrutural.
- Tool que retorna `Observation` envia dados estruturados para uma chamada final de síntese.
- A chamada de síntese final não pode receber tools.

Isso evita que JSON bruto vaze ao usuário e preserva o roteamento determinístico.

### 30.3. Async mínimo obrigatório

A biblioteca deve suportar `ainvoke` e tools `async def` no MVP.

Decisão:

- `ainvoke` é o caminho de produção.
- `invoke` é conveniência síncrona.
- `invoke` deve falhar com erro claro quando uma stack assíncrona for exigida.

### 30.4. Pydantic v2 como motor de schema e parsing

A biblioteca deve usar Pydantic v2 em `schema.py`.

Decisão:

- não manter parser próprio de `typing`;
- gerar modelos de entrada por tool;
- validar kwargs de LLM e de `EnrichedPointer` com Pydantic;
- gerar schema interno a partir desses modelos;
- deixar adapters de provedor converterem para formatos específicos.

---

## 31. Decisões de produção adicionadas na versão 0.3

### 31.1. Um turno = uma cadeia

O motor não é ReAct multi-step. Em cada turno, o LLM toma no máximo uma decisão inicial de roteamento. Depois disso, o fluxo só continua por `EnrichedPointer` validado ou termina em `FinalAnswer`/`Observation`.

A síntese final a partir de `Observation` não pode chamar tools.

### 31.2. Ciclos estruturais

O registry não precisa rejeitar ciclos no grafo no MVP. Ciclos são permitidos estruturalmente, mas não podem executar indefinidamente. A proteção obrigatória é `max_steps`, acompanhada de trace.

### 31.3. Injeção de contexto no MVP

A injeção mínima via `inject={...}` faz parte do MVP. Isso remove do LLM a responsabilidade de preencher identificadores confiáveis da sessão e torna o invariante de argumentos injetados obrigatório, não condicional.

### 31.4. Semântica combinada de ToolPolicy

`all_of`, `any_of` e `none_of` são combinados por `AND` entre grupos. `none_of` apenas bloqueia; não concede acesso sozinho.

### 31.5. Async em produção

`ainvoke` é o caminho recomendado de produção. Tools `async def` são aguardadas. Tools `def` rodam por padrão em thread via `sync_tool_strategy="thread"` para não bloquear o event loop.

`invoke` é conveniência síncrona e deve falhar com `AsyncRequiredError` se for usado dentro de event loop ativo para operação que exige stack assíncrona.

### 31.6. Síntese final e contexto recebido pelo LLM

A chamada de síntese recebe apenas:

1. histórico conversacional normalizado;
2. observação sanitizada da tool;
3. instrução interna para redigir resposta final sem chamar tools.

Ela não recebe catálogo completo, allowlist, schemas, trace, capabilities ou metadata bruta.

### 31.7. Erro de runtime de tool

Exceções levantadas por tools são encapsuladas em `ToolExecutionError`. O padrão é abortar o turno e propagar erro controlado para a aplicação cliente. Transformar esse erro em resposta final ou síntese é comportamento opt-in.

---

## 32. Filosofia do projeto

`adjacency-agents` não tenta tornar o LLM mais obediente por prompt. A biblioteca reduz o espaço de erro antes que o LLM seja chamado.

A tese central é:

> Não peça ao modelo para escolher corretamente entre ferramentas incompatíveis com o cenário. Remova do parser as ferramentas incompatíveis antes da chamada ao modelo.

Em outras palavras:

```text
Backend define o cenário.
Engine monta a allowlist.
LLM escolhe dentro de um espaço seguro.
Python executa e valida.
```

Esse é o núcleo do projeto.


---

## 35. Revisão 0.4 — Fechamento pré-implementação

Esta revisão fecha os últimos refinamentos antes da implementação:

1. Exceções levantadas dentro de tools são sempre tratadas como falhas de runtime e remapeadas para `ToolExecutionError`, mesmo quando a exceção original é subclasse de `AdjacencyAgentsError`.
2. Exceções emitidas pelo próprio engine fora do corpo da tool preservam seus tipos específicos.
3. `tool_error_mode="synthesize"` em cadeia `EnrichedPointer` não revela nomes de tools, número de saltos, ponteiros, grafo ou trace para o LLM de síntese.
4. `default_tool_error_message` passa a ter valor padrão neutro em inglês; aplicações devem sobrescrever a mensagem para localização e tom.
5. Testes end-to-end de cadeias `A -> B -> Observation` e `A -> B -> C -> Observation` passam a ser obrigatórios.

Com estas decisões, o documento está pronto para orientar a Fase 1 de implementação.
