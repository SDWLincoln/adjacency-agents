# Architecture

## One-line thesis

> Backend defines the scenario. The engine builds the allowlist. The
> LLM chooses inside a safe space. Python executes and validates.

## Control flow (per turn)

```
                    UserContext + messages
                            │
                            ▼
                    ToolRegistry
                            │
        build_allowlist(registry, context)         (router.py)
                            │
       build_llm_visible_allowlist(allowlist)
                            │
           build_json_schema(visible tools)        (schema.py — Pydantic v2)
                            │
                            ▼
              ┌──────  LLM #1 (routing)  ──────┐
              │            │                   │
        FinalAnswer    ToolCall              str → FinalAnswer
              │            │
              │      validate_kwargs          (schema.py)
              │            │
              │      resolve_injected_kwargs  (schema.py)
              │            │
              │      validate_full_kwargs
              │            │
              │      execute tool body        (sync_tool_strategy)
              │            │
              │   ┌────────┼─────────┐
              │   ▼        ▼         ▼
              │  Pointer  Final   Observation
              │   │        │         │
              │   └→ validate next neighbor (engine._resolve_transition)
              │             │
              │             ▼
              │       LLM #2 (synthesis, tools=[])
              │             │
              └─────────────▼
                       FinalAnswer
```

Each turn is at most one chain (§4.7). The LLM makes at most one
routing decision and, if needed, one synthesis call.

## Modules

| File | Responsibility | Spec section |
|------|----------------|--------------|
| `models.py` | Frozen dataclasses for all public data types | §9, §22.1 |
| `decorators.py` | `@tool_node` and `ToolNodeSpec` | §10, §22.2 |
| `registry.py` | Per-engine `ToolRegistry`; duplicate + neighbor validation | §11, §22.3 |
| `router.py` | `build_allowlist`, `build_llm_visible_allowlist` | §12, §22.4 |
| `schema.py` | Pydantic-v2 input models, JSON schema, injection resolution | §13, §17, §22.5 |
| `llm.py` | `LLMClient` / `AsyncLLMClient` protocols + `FakeLLMClient` | §18, §22.6 |
| `engine.py` | `DeterministicEngine`, the deterministic loop | §14, §22.7 |
| `errors.py` | Controlled exception hierarchy | §19, §22.8 |
| `tracing.py` | `ExecutionTrace` and `TraceEvent` with sanitization | §20, §22.9 |

## Key design decisions

### Default deny

`ToolPolicy()` (empty) returns `False` from `allows(...)`. A tool
without an explicit `requires` or `policy` is never callable.
(§4.1, §9.4.6)

### Triple validation

Every tool invocation is validated three times:

1. **Before schema** — only allowed tools are exposed to the LLM
   (`build_allowlist` → `build_llm_visible_allowlist`).
2. **Before tool execution** — `ToolCall.name` is checked against the
   visible allowlist, `kwargs` against the Pydantic model, and any
   injected key supplied externally is rejected.
3. **Before each structural transition** — `EnrichedPointer.next_tool`
   must exist, be a declared neighbor, be allowed for the current
   context, and not supply injected args.

(§4.2, §15, §16)

### Per-engine registry

`ToolRegistry` is constructed inside `DeterministicEngine.__init__`.
There is no global singleton. This makes tests, multi-tenant
deployments, and concurrent invocations safe. (§4.1, §11.1)

### Pydantic v2 for schema and parsing

`schema.py` builds two Pydantic models per tool:

- **Input model** — only the LLM-visible parameters; sent to the
  provider, used to validate `ToolCall.kwargs` and `EnrichedPointer.kwargs`.
- **Full input model** — every parameter including injected ones; used
  to validate the merged dict *after* injection, right before the
  function call. (§17.4.8)

Both models use `extra="forbid"` so the LLM cannot smuggle arbitrary
keys.

### One chain per turn

The engine never asks the LLM to choose a second tool after seeing a
result. After the initial `ToolCall` the chain continues only through
`EnrichedPointer` transitions, which are entirely Python-side. The
synthesis call (when there's an `Observation`) goes out with
`tools=[]` and `allow_tool_calls=False`. (§4.7, §31.1)

### Tool runtime errors

Exceptions raised *inside a user tool body* — including subclasses of
`AdjacencyAgentsError` — are always wrapped in `ToolExecutionError`,
preserving the original as `__cause__`. This prevents a tool from
impersonating an engine-level policy decision. (§14.9, §35.1)

Engine-side exceptions raised *outside* the tool body
(`ToolNotAllowedError`, `InvalidTransitionError`,
`ContextInjectionError`, etc.) keep their specific type.

### Async by default

`ainvoke` is the production path. `async def` tools are awaited
directly; `def` tools run via `asyncio.to_thread` so they cannot block
the event loop (`sync_tool_strategy="thread"`, §14.3.10).

`invoke` exists for sync scripts and **refuses to run inside an active
event loop** — it raises `AsyncRequiredError` instead.

### Execution trace

Every invocation produces an `ExecutionTrace` accessible via
`engine.last_trace`. The 17 minimum events of §20.2 are emitted with
sanitized payloads — raw prompts, user-supplied kwargs and tool
outputs never end up in the trace. (§20.3)
