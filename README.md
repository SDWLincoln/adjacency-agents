# adjacency-agents

> Backend defines the scenario. The engine builds the allowlist. The LLM
> chooses inside a safe space. Python executes and validates.

`adjacency-agents` is a microlibrary for deterministic tool
orchestration in flows that include LLMs/SLMs. Instead of asking the
model to pick the *right* tool among many semantically-similar ones, the
engine removes incompatible tools from the parser **before** calling the
model.

**Status:** MVP — Phases 1–5 of the DDD spec are implemented and
covered by tests. Provider-specific adapters (Phase 6) are not in
scope yet.

## Why

Small LLMs (and even big ones) regularly call the wrong tool when two
tools are semantically close — e.g. a "reissue boleto for guests" tool
and a "reissue boleto for registered users" tool. That is not a
permission bug. It is a **contextual parsing** bug: the model is
filling arguments for a tool that should not even exist in the current
scenario.

`adjacency-agents` does not try to make the LLM smarter via prompting.
It reduces the model's choice space before the call.

## Install

```bash
pip install -e .
# or, once published:
# pip install adjacency-agents
```

Python 3.10+. Depends on `pydantic>=2.7,<3`.

## Quickstart

```python
from adjacency_agents import DeterministicEngine, UserContext, tool_node
from adjacency_agents.llm import FakeLLMClient
from adjacency_agents import ToolCall


@tool_node(requires=["public"])
def listar_servicos() -> str:
    """Lista serviços disponíveis."""
    return "Temos atendimento comercial, financeiro e suporte."


fake = FakeLLMClient(script=[ToolCall(name="listar_servicos")])
engine = DeterministicEngine(llm=fake, tools=[listar_servicos])
ctx = UserContext(session_id="s1", capabilities={"public"})
print(engine.invoke(prompt="quais serviços?", context=ctx).content)
```

## Real LLM providers

For OpenAI Chat Completions, install the optional extra and pass any
`OpenAI`-shaped client to the bundled adapter:

```bash
pip install -e ".[openai]"
```

```python
from openai import OpenAI

from adjacency_agents import DeterministicEngine, UserContext, tool_node
from adjacency_agents.adapters.openai import OpenAIClient

adapter = OpenAIClient(client=OpenAI(), model="gpt-4o-mini")
engine = DeterministicEngine(llm=adapter, tools=[listar_servicos])
answer = engine.invoke(
    prompt="quais serviços?",
    context=UserContext(session_id="s1", capabilities={"public"}),
)
```

`AsyncOpenAIClient` is the async counterpart for use with
`engine.ainvoke(...)`. Both wrap the engine's provider-agnostic JSON
schema into OpenAI's `tools=[{type: "function", ...}]` format, parse
`tool_calls` back into the internal `ToolCall`, and send
`tool_choice="none"` during synthesis.

## Capabilities

Capabilities are short string labels derived from trusted facts in your
application (session, DB, API). The library does **not** interpret them
semantically — it only matches them against tool policies.

```python
ctx = UserContext(
    session_id="whatsapp_123",
    capabilities={"public", "registered", "active_account"},
    metadata={"registration_id": "abc-123"},
)
```

## ToolPolicy

```python
from adjacency_agents import ToolPolicy, tool_node


@tool_node(
    policy=ToolPolicy(
        all_of={"registered", "active_account"},
        none_of={"blocked", "fraud_suspected"},
    )
)
def consultar_area_restrita() -> str:
    """Disponível apenas para conta ativa e não bloqueada."""
    return "Área restrita liberada."
```

A tool with no `requires`/`policy` is denied by default. Empty policies
do not grant access (§4.1 of the spec).

## EnrichedPointer — deterministic transitions

```python
from adjacency_agents import EnrichedPointer, tool_node


@tool_node(
    requires=["registered"],
    structural_neighbors=["consultar_detalhe"],
)
def buscar_recente() -> EnrichedPointer | str:
    return EnrichedPointer(
        next_tool="consultar_detalhe",
        kwargs={"item_id": "ITEM-007"},
        reason="item encontrado",
    )


@tool_node(requires=["registered"], llm_visible=False)
def consultar_detalhe(item_id: str) -> str:
    return f"Item {item_id}: enviado em 2026-05-20."
```

The second tool is `llm_visible=False`. The LLM never sees its schema;
it can only be reached via a validated pointer from a declared neighbor.

## Observation + synthesis

A tool that returns an `Observation` (or any `dict`/`list`/`BaseModel`
under `response_mode="auto"`) triggers a single synthesis call with
**tools disabled**. The LLM cannot start a new routing decision during
synthesis.

```python
from adjacency_agents import Observation, tool_node


@tool_node(requires=["public"])
def saldo() -> Observation:
    return Observation(data={"saldo": 123.45, "moeda": "BRL"})
```

## Argument descriptions and constraints

Use `typing.Annotated[T, Field(...)]` to attach descriptions and
validation rules to individual tool arguments. They flow into the JSON
schema sent to the LLM **and** are enforced by Pydantic on every call.

```python
from typing import Annotated
from pydantic import Field
from adjacency_agents import tool_node


@tool_node(requires=["public"])
def buscar(
    query: Annotated[str, Field(description="termo de busca")],
    limit: Annotated[int, Field(description="máx. resultados", ge=1, le=100)] = 10,
) -> str:
    ...
```

## Multi-turn `messages`

```python
from adjacency_agents import Message

engine.invoke(
    messages=[
        Message(role="user", content="Quero atendimento"),
        Message(role="assistant", content="Você já é cadastrado?"),
        Message(role="user", content="Sim"),
    ],
    context=ctx,
)
```

`UserContext` carries trusted facts. `messages` carries conversation.
The engine never mixes the two.

## `ainvoke` and async tools

`ainvoke` is the recommended production path. It supports `async def`
tools natively and runs `def` tools in a worker thread by default so
they cannot block the event loop.

```python
import asyncio

from adjacency_agents import (
    DeterministicEngine,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.llm import FakeLLMClient


@tool_node(requires=["public"])
async def fetch_status() -> str:
    """Pretend this awaits an HTTP call."""
    await asyncio.sleep(0)
    return "online"


async def main() -> None:
    fake = FakeLLMClient(script=[ToolCall(name="fetch_status")])
    engine = DeterministicEngine(llm=fake, tools=[fetch_status])
    ctx = UserContext(session_id="s", capabilities={"public"})

    answer = await engine.ainvoke(prompt="qual o status?", context=ctx)
    print(answer.content)  # → "online"


asyncio.run(main())
```

`invoke()` is a convenience wrapper for synchronous scripts. Calling it
from inside an active event loop raises `AsyncRequiredError` — use
`await engine.ainvoke(...)` there.

## Context injection

Confiable values from the application (`registration_id`, `tenant_id`,
`session_id`, ...) must not be filled by the LLM. Declare them with
`inject={...}` and the engine resolves them at execution time.

```python
@tool_node(
    requires=["registered"],
    inject={"registration_id": "metadata.registration_id"},
)
def consultar_dados(registration_id: str) -> dict:
    return {"id": registration_id}
```

The injected parameter is excluded from the schema sent to the LLM.
Any attempt to supply it from the LLM or an `EnrichedPointer` is
rejected before execution.

## Tool runtime errors

By default, an exception raised inside a tool body is wrapped in
`ToolExecutionError` (preserving the original as `__cause__`) and
propagated. Configure `tool_error_mode` to convert it into a safe
final answer or sanitized synthesis instead.

```python
from adjacency_agents import (
    DeterministicEngine,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.errors import ToolExecutionError
from adjacency_agents.llm import FakeLLMClient


@tool_node(requires=["public"])
def consultar_saldo() -> str:
    raise TimeoutError("upstream took too long")


fake = FakeLLMClient(script=[ToolCall(name="consultar_saldo")])
ctx = UserContext(session_id="s", capabilities={"public"})

# 1. Default: ToolExecutionError bubbles up — the application decides
#    how to render it.
engine = DeterministicEngine(llm=fake, tools=[consultar_saldo])
try:
    engine.invoke(prompt="qual meu saldo?", context=ctx)
except ToolExecutionError as exc:
    print("falhou:", exc.__cause__)

# 2. tool_error_mode="final": the engine returns a safe canned answer
#    without calling the LLM again.
fake = FakeLLMClient(script=[ToolCall(name="consultar_saldo")])
engine = DeterministicEngine(
    llm=fake,
    tools=[consultar_saldo],
    tool_error_mode="final",
    default_tool_error_message="Não foi possível concluir agora.",
)
print(engine.invoke(prompt="qual meu saldo?", context=ctx).content)
# → "Não foi possível concluir agora."
```

`tool_error_mode="synthesize"` sends only a sanitized `Observation` to
the LLM — tool names, hop counts, pointers and tracebacks never leak.

## Execution trace

Every engine invocation stores a sanitized `ExecutionTrace` in
`engine.last_trace`. It is intended for audit, debugging and tests.

```python
from adjacency_agents import ExecutionTrace

answer = engine.invoke(prompt="...", context=ctx)
trace: ExecutionTrace | None = engine.last_trace

if trace is not None:
    print(trace.names())
```

Trace events include routing, validation, tool execution, pointer
transitions, synthesis, policy denials, context injection failures and
`max_steps` aborts. By default the trace records structural metadata only:
tool names, event names, counts and type names. It does not record raw
prompts, capabilities, `UserContext.metadata`, kwargs, tool payloads or
tracebacks.

## Security guarantees (the short list)

- **Default deny** — empty policy never grants access.
- **Allowlist per turn** — the schema sent to the LLM is built from the
  current `UserContext`, never from the full catalog.
- **Triple validation** — before schema, before tool execution, before
  every transition.
- **The LLM never decides authorization** — it only picks from a
  pre-filtered, contextual allowlist.
- **No global registry** — every `DeterministicEngine` owns its own
  `ToolRegistry`, so tests and multi-tenant deployments are isolated.

## Project layout

```
src/adjacency_agents/
├── __init__.py        # public facade
├── decorators.py      # @tool_node
├── engine.py          # DeterministicEngine
├── errors.py
├── llm.py             # protocols + FakeLLMClient
├── models.py
├── registry.py
├── router.py
├── schema.py          # Pydantic v2 schema + validation
└── tracing.py         # ExecutionTrace + sanitization
```

## Tests

```bash
.venv/bin/pytest
```

The MVP test suite covers all invariants listed in §23 of the DDD
spec.

## Documentation

The full Documentation-Driven Development specification lives in
[`adjacency_agents_documentation_driven_development_v0_4_final.md`](./adjacency_agents_documentation_driven_development_v0_4_final.md).
