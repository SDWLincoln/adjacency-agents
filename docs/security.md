# Security model

The library's security posture is described in §4, §23 and §27 of the
DDD spec. This document is the operational summary.

## Threat model

The LLM is treated as **untrusted** input. The application is
**trusted**.

Concretely:

- The LLM may hallucinate tool names, invent arguments, leak past
  context into new turns, or attempt to call tools the application has
  not authorized for the current scenario.
- The LLM should not see or be able to fill arguments that come from
  trusted state (`registration_id`, `tenant_id`, `account_id`, etc.).
- Tool authors are trusted code but can still raise unexpected errors;
  those errors must not leak internals to the LLM or the end user by
  default.

The library does **not** defend against malicious tool code or against
a compromised application backend.

## Invariants enforced in code

The 30 invariants of §23 are exercised by the test suite. The most
important ones are:

1. **Default deny.** A tool without `requires`/`policy` is excluded
   from every allowlist.
2. **Allowlist per turn.** The schema sent to the LLM is built from
   `UserContext.capabilities` *for this call only* — never from the
   global catalog.
3. **`llm_visible=False` keeps a tool out of the LLM schema** even
   when it is reachable via a structural pointer.
4. **Tool calls are validated three times** — before schema, before
   tool execution, before every structural transition.
5. **Injected arguments are excluded from the schema** and rejected if
   supplied by either the LLM or an `EnrichedPointer`.
6. **`EnrichedPointer` must target a declared neighbor** that is
   allowed for the current context. Cross-graph jumps are rejected.
7. **One chain per turn.** After the initial routing call the LLM
   cannot pick another tool. The synthesis call goes out with
   `tools=[]` and `allow_tool_calls=False`.
8. **Tool exceptions are wrapped.** Even subclasses of
   `AdjacencyAgentsError` raised *inside* a tool body become
   `ToolExecutionError` so a tool cannot impersonate a policy decision
   by the engine.
9. **`max_steps` always applies.** Structural cycles are bounded
   operationally even if the static graph allows them (§16, §31.2).
10. **Trace sanitization.** `ExecutionTrace` redacts prompts,
    user-supplied kwargs, raw tool outputs, pointer reasons and
    exception messages — see below.

## Trace sanitization

The events emitted by `engine.last_trace` carry only structural
information by default:

- `tool_name`, `kwarg_names` (sorted list of keys), counts, phases,
  step indexes.
- Content fields (`prompt`, `kwargs values`, `tool result`,
  `pointer reason`, `exception messages`) are explicitly omitted.

Applications that need richer telemetry should add their own sanitizer
on top of `ExecutionTrace.to_dict()`. The default is conservative on
purpose.

## Synthesis safety

The synthesis call (when a tool returns `Observation`, `dict`, `list`,
etc.) receives only:

1. Normalized conversation history.
2. A sanitized representation of the `Observation.data`.
3. A system instruction telling the LLM not to call tools and not to
   reveal internals.

It does **not** receive: the catalog, the allowlist, the schema, the
trace, capability sets, raw metadata, hop counts, tool names of
intermediate steps in a chain, or tracebacks. (§14.7, §35.3)

If the LLM produces a `ToolCall` during synthesis the engine raises
`SynthesisError`.

## Anti-patterns

Direct quotes from §27.2. Do **not** do any of this:

```python
# ❌ Sending the whole catalog to the LLM.
llm.complete(prompt=prompt, tools=registry.all_tools)

# ❌ Executing a tool call without re-validating against the allowlist.
tool = registry.get(model_output.name)
tool.fn(**model_output.kwargs)

# ❌ Executing a pointer without checking neighbor + policy.
next_tool = registry.get(pointer.next_tool)
next_tool.fn(**pointer.kwargs)

# ❌ Implicit "public" by omission.
@tool_node()
def public_by_accident():
    ...
```

## Application responsibilities

The library cannot make the application safe by itself. The application
must:

- Build `UserContext.capabilities` from **trusted** facts only (DB,
  session, authenticated API, webhook signature, etc.). Never derive
  capabilities from user-supplied text.
- Choose `tool_error_mode` deliberately. `raise` is safest by default;
  `synthesize` only with the sanitization above accepted.
- Translate `default_tool_error_message` for production — the shipped
  default is intentionally generic English (§35.4).
- Persist conversation history outside the library and pass it back as
  `messages=...` on the next turn.
