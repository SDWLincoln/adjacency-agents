# Public API reference

All symbols below are exported from the top-level package and are the
*only* supported import surface (§8.1).

```python
from adjacency_agents import (
    DeterministicEngine,
    EnrichedPointer,
    ExecutionTrace,
    FinalAnswer,
    Message,
    Observation,
    ToolCall,
    ToolPolicy,
    TraceEvent,
    UserContext,
    tool_node,
)
```

## Decorator

### `tool_node`

```python
def tool_node(
    *,
    name: str | None = None,
    requires: list[str] | set[str] | None = None,
    policy: ToolPolicy | None = None,
    structural_neighbors: list[str] | set[str] | tuple[str, ...] | None = None,
    llm_visible: bool = True,
    description: str | None = None,
    response_mode: Literal["auto", "final", "synthesize"] = "auto",
    inject: dict[str, str] | None = None,
) -> Callable[[F], F]: ...
```

- `requires` is sugar for `policy=ToolPolicy(all_of=set(requires))`.
- `requires` and `policy` are mutually exclusive (§10.2.1).
- A function without `requires`/`policy` gets an empty `ToolPolicy`,
  which denies by default.
- `inject` keys must match parameters that exist on the function.
- `llm_visible=False` keeps a tool out of the schema sent to the LLM
  but still allows execution via a structural pointer.

## Engine

### `DeterministicEngine`

```python
class DeterministicEngine:
    def __init__(
        self,
        *,
        llm,                                                # has .complete and/or .acomplete
        tools: list[Callable],
        max_steps: int = 8,
        default_tool_result_mode: Literal["auto","final","synthesize"] = "auto",
        sync_tool_strategy: Literal["thread","direct"] = "thread",
        tool_error_mode: Literal["raise","final","synthesize"] = "raise",
        default_tool_error_message: str = "I could not complete this operation right now.",
    ) -> None: ...

    def invoke(
        self,
        *,
        context: UserContext,
        prompt: str | None = None,
        messages: Sequence[Message] | None = None,
    ) -> FinalAnswer: ...

    async def ainvoke(
        self,
        *,
        context: UserContext,
        prompt: str | None = None,
        messages: Sequence[Message] | None = None,
    ) -> FinalAnswer: ...

    @property
    def last_trace(self) -> ExecutionTrace | None: ...
```

Invariants enforced on every call:

- `prompt` or `messages` (or both) must be supplied.
- `context` is mandatory.
- The engine never mutates the `messages` sequence or the
  `UserContext`.
- `invoke` from inside an active event loop raises `AsyncRequiredError`.

## Data models

All models are frozen dataclasses.

### `UserContext`

```python
UserContext(session_id: str, capabilities: set[str], metadata: dict[str, Any] = {})
```

Trusted state. `capabilities` drives policy; `metadata` is the source
for `inject={...}` paths like `metadata.registration_id`. The library
does not interpret capability names semantically.

### `Message`

```python
Message(role: Literal["system","user","assistant","tool"], content: str, name: str | None = None, metadata: dict = {})
```

Conversational history. Distinct from `UserContext`.

### `ToolPolicy`

```python
ToolPolicy(all_of: set[str] = (), any_of: set[str] = (), none_of: set[str] = ())
ToolPolicy.allows(capabilities: set[str]) -> bool
```

Semantics (§9.4):

- All groups combined with `AND`.
- `all_of` empty → trivially true (only when `any_of` is non-empty).
- `none_of` alone never grants access — empty policy denies.

### `EnrichedPointer`

```python
EnrichedPointer(next_tool: str, kwargs: dict = {}, reason: str = "")
```

Returned by a tool to transfer execution to a declared structural
neighbor. Never sent to the user.

### `Observation`

```python
Observation(data: Any, summary_hint: str | None = None, expose_to_llm: bool = True, metadata: dict = {})
```

Structured output that triggers one synthesis call. `expose_to_llm=False`
means the data must not appear in the synthesis prompt.

### `ToolCall`, `FinalAnswer`

```python
ToolCall(name: str, kwargs: dict = {})
FinalAnswer(content: str)
```

## Argument descriptions and constraints

Per-argument descriptions, defaults and validation rules are picked up
from `typing.Annotated[T, Field(...)]` via Pydantic v2. They appear in
`build_json_schema(spec)` and are enforced on every `validate_kwargs`.

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

The resulting schema has:

```json
{
  "properties": {
    "query": {"type": "string", "description": "termo de busca"},
    "limit": {"type": "integer", "description": "máx. resultados",
              "minimum": 1, "maximum": 100, "default": 10}
  },
  "required": ["query"]
}
```

For more complex argument shapes, declare a user-defined `BaseModel`
and accept it as a single parameter — it is preserved end-to-end.

## LLM protocols

```python
class LLMClient(Protocol):
    def complete(self, *, messages, tools, allow_tool_calls=True) -> ToolCall | FinalAnswer | str: ...

class AsyncLLMClient(Protocol):
    async def acomplete(self, *, messages, tools, allow_tool_calls=True) -> ToolCall | FinalAnswer | str: ...
```

`FakeLLMClient(script=[...])` ships with the library for tests and
examples. It enforces `allow_tool_calls=False` at the protocol level
during synthesis.

## Provider adapters

### OpenAI

```python
from openai import OpenAI, AsyncOpenAI

from adjacency_agents.adapters.openai import OpenAIClient, AsyncOpenAIClient

adapter      = OpenAIClient(client=OpenAI(), model="gpt-4o-mini")
async_adapter = AsyncOpenAIClient(client=AsyncOpenAI(), model="gpt-4o-mini")
```

Optional install:

```bash
pip install "adjacency-agents[openai]"
```

The adapter has no hard dependency on `openai`: it expects a duck-typed
client with `client.chat.completions.create(**kwargs)` (sync) or the
awaitable equivalent (async). This makes it easy to use the same code
with a self-hosted compatibility shim or tests.

Translation rules:

- Internal JSON schema (from `build_json_schema`) is wrapped in
  `{"type": "function", "function": {"name", "description", "parameters"}}`.
- `Message(role="tool", name=X, content=Y)` is rewritten as
  `{"role": "system", "content": "[tool: X] Y"}` so OpenAI does not
  reject it for lacking a paired `tool_calls` block.
- `allow_tool_calls=False` (synthesis) sets `tool_choice="none"` and
  omits `tools`. A `tool_call` returned anyway raises `SynthesisError`.
- Tool arguments are decoded from the JSON string returned by the API;
  non-JSON or non-object payloads raise `InvalidToolCallError`.
- `extra_create_kwargs={"temperature": 0.2, ...}` lets callers pass any
  additional argument the SDK accepts.

### Anthropic

```python
from anthropic import Anthropic, AsyncAnthropic

from adjacency_agents.adapters.anthropic import (
    AnthropicClient,
    AsyncAnthropicClient,
)

adapter = AnthropicClient(
    client=Anthropic(),
    model="claude-haiku-4-5",
    max_tokens=512,
)
async_adapter = AsyncAnthropicClient(
    client=AsyncAnthropic(),
    model="claude-haiku-4-5",
)
```

Optional install:

```bash
pip install "adjacency-agents[anthropic]"
```

The adapter expects a duck-typed client with
`client.messages.create(**kwargs)`. Translation rules:

- Internal JSON schema is projected to `{"name", "description",
  "input_schema"}`. There is no `function` envelope; `input_schema` IS
  the JSON schema body.
- `Message(role="system")` entries are pulled out of the conversation
  and concatenated into the top-level `system` kwarg. Multiple system
  messages are joined with two newlines.
- `Message(role="tool", name=X, content=Y)` is rewritten as
  `{"role": "user", "content": "[tool: X] Y"}` so the API does not
  reject it for lacking a paired `tool_use` block.
- `allow_tool_calls=False` (synthesis) omits `tools` entirely. A
  `tool_use` content block returned anyway raises `SynthesisError`.
- The response is a list of content blocks; `text` blocks are
  concatenated, the first `tool_use` block becomes a `ToolCall` with
  the already-decoded `input` dict.
- `max_tokens` defaults to 1024 and is overridable per adapter
  instance.
- `extra_create_kwargs` forwards any additional SDK argument
  (`temperature`, `top_p`, `stop_sequences`, ...).

## Errors

All exceptions inherit from `AdjacencyAgentsError`:

| Exception | Raised by |
|-----------|-----------|
| `ToolNotFoundError` | Registry lookup for an unknown name |
| `ToolNotAllowedError` | Policy or LLM-visibility violation |
| `InvalidToolCallError` | Schema validation of `ToolCall.kwargs` |
| `InvalidTransitionError` | `EnrichedPointer` violates graph or policy |
| `InvalidToolSchemaError` | Tool signature is unprocessable (missing type hint, *args) |
| `ContextInjectionError` | `inject={...}` path cannot be resolved on `UserContext` |
| `ToolExecutionError` | Anything raised inside a user tool body |
| `MaxStepsExceededError` | Structural chain exceeded `max_steps` |
| `AsyncRequiredError` | `invoke()` called inside an active event loop |
| `SynthesisError` | LLM emitted a `ToolCall` during synthesis, or `Observation` is `expose_to_llm=False` without fallback |

## Execution trace

```python
engine.last_trace            # ExecutionTrace | None
engine.last_trace.events     # list[TraceEvent]
engine.last_trace.names()    # list[str]
engine.last_trace.to_dict()  # JSON-friendly dump
```

The 17 standard event names are listed in `docs/architecture.md` and
spec §20.2. Payloads are sanitized — see `docs/security.md`.
