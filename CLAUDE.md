# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Source of truth

The Documentation-Driven Development spec at the repo root is authoritative:
**`adjacency_agents_documentation_driven_development_v0_4_final.md`**. Every
public behavior, invariant, and acceptance criterion is defined there. When
adding or modifying behavior, update the spec in the same change. The DDD
contains numbered invariants in §23 and a mandatory test list in §24 —
preserve them.

`docs/` mirrors the spec into smaller topic files (`ddd`, `architecture`,
`api`, `security`, `testing`). The `CHANGELOG.md` follows Keep a Changelog.

## Common commands

```bash
# Tests (always against the project's venv)
.venv/bin/pytest                                # full suite
.venv/bin/pytest tests/test_engine.py           # single file
.venv/bin/pytest tests/test_engine.py -k pointer  # single test by keyword

# Lint and format (must be clean before commit; CI enforces both)
.venv/bin/ruff check src tests examples
.venv/bin/ruff format src tests examples        # apply
.venv/bin/ruff format --check src tests examples  # verify only

# Types
.venv/bin/mypy src/adjacency_agents

# Build sdist and wheel (release dry-run)
.venv/bin/python -m build
.venv/bin/twine check --strict dist/*

# Run any example end-to-end
.venv/bin/python examples/fake_llm_example.py
```

`pyproject.toml` sets `asyncio_mode = "auto"`, so any `async def test_…` is
collected without a decorator. Tests never call out to the network — every
provider adapter is exercised against a scripted fake client living in the
test file.

## Core thesis (read this before editing the engine)

Small LLMs and SLMs reliably pick the wrong tool when two tools are
semantically close. `adjacency-agents` does **not** try to make the LLM
smarter via prompting. It **removes incompatible tools from the parser
before the model is called**. The flow is:

```
UserContext.capabilities → build_allowlist → build_llm_visible_allowlist
→ build_json_schema (Pydantic v2) → LLM (routing call, once)
→ validate ToolCall (kwargs + injected exclusion) → execute tool
→ EnrichedPointer? validate transition, loop without LLM
→ Observation? one synthesis call with tools=[] → FinalAnswer
```

## Invariants you must not break

The full list is §23 of the DDD. The ones that catch contributors out:

1. **Default deny.** `ToolPolicy()` returns `False`. A tool without
   `requires`/`policy` is never callable.
2. **One turn = one chain (§4.7, §31.1).** The LLM picks at most one tool
   at the start of a turn. After that, the chain only advances via
   `EnrichedPointer` transitions (all-Python) or terminates at
   `FinalAnswer` / `Observation`. There is **no** ReAct loop; do not add
   one.
3. **Triple validation.** Before the LLM sees a schema, before tool body
   execution, before every structural transition. Each step has its own
   `Invalid*Error` type.
4. **`llm_visible=False` keeps a tool out of the LLM schema** even when
   it is reachable via a structural pointer.
5. **Tool body exceptions become `ToolExecutionError`** — even subclasses
   of `AdjacencyAgentsError` raised *inside* a tool body. This prevents
   a tool from impersonating an engine policy decision. Engine-side
   exceptions (`ToolNotAllowedError`, `InvalidTransitionError`, etc.)
   keep their specific types.
6. **No global registry.** Every `DeterministicEngine` owns its own
   `ToolRegistry`. There is no module-level catalog. Tests and
   multi-tenant deployments rely on this.
7. **Synthesis is sandboxed.** The synthesis LLM call gets only the
   normalized conversation history, the sanitized `Observation`, and a
   system instruction. Never the catalog, the schemas, the trace,
   capabilities, raw metadata, or hop counts.
8. **Async path doesn't block the loop.** `ainvoke` is the production
   path. Sync tools (`def`) run via `asyncio.to_thread` by default
   (`sync_tool_strategy="thread"`). `invoke()` called from inside an
   active event loop raises `AsyncRequiredError`.

## Module layout (high-level)

```
src/adjacency_agents/
├── models.py        Frozen dataclasses (UserContext, Message, ToolPolicy, ...)
├── decorators.py    @tool_node + ToolNodeSpec
├── registry.py      Per-engine ToolRegistry (duplicates + neighbors validated)
├── router.py        build_allowlist, build_llm_visible_allowlist (pure)
├── schema.py        Pydantic v2 input models, JSON schema, inject resolution
├── llm.py           LLMClient/AsyncLLMClient Protocols + FakeLLMClient
├── engine.py        DeterministicEngine — the loop
├── errors.py        Controlled exception hierarchy
├── tracing.py       ExecutionTrace + TraceEvent (17 sanitized event types)
├── py.typed         PEP 561 marker (must remain in package-data)
└── adapters/
    ├── openai.py    OpenAIClient + AsyncOpenAIClient
    ├── anthropic.py AnthropicClient + AsyncAnthropicClient
    └── ollama.py    OllamaClient + AsyncOllamaClient
```

Tests in `tests/` are organized by concern: `test_<module>.py` for unit-level,
`test_engine.py` / `test_pointer_transitions.py` / `test_synthesis.py` for
the engine loop, `test_<provider>_adapter.py` for each adapter.

## Adding a new provider adapter

All three existing adapters follow the same pattern. To add another:

1. Create `src/adjacency_agents/adapters/<provider>.py` with `<Provider>Client`
   (sync) and `Async<Provider>Client`. **No top-level import** of the
   provider SDK — accept any duck-typed client.
2. Implement `_convert_tools`, `_convert_messages`, `_build_kwargs`,
   `_parse_response` as needed for the provider's wire format.
3. Decide how `role="tool"` messages from synthesis are repackaged for the
   provider. The existing adapters rewrite them as `system` (OpenAI) or
   `user` (Anthropic, Ollama) messages with a `[tool: <name>]` prefix —
   the API would otherwise require a paired assistant `tool_calls` block
   we never produce.
4. Honor `allow_tool_calls=False` — omit `tools` (and pass any
   provider-specific equivalent of `tool_choice="none"`); raise
   `SynthesisError` if a tool call still arrives.
5. Add an optional extra in `pyproject.toml`:
   `<provider> = ["<sdk>>=X"]`.
6. Add `tests/test_<provider>_adapter.py` with a scripted fake client and
   the same test classes as the others (schema conversion, message
   conversion, tool-call round-trip, synthesis safety, async).
7. Add `examples/<provider>_adapter_example.py` that skips elegantly when
   the SDK or credentials are missing.
8. Update `README.md` ("Real LLM providers") and `docs/api.md`
   ("Provider adapters").

## Release flow

`v0.1.0` is on PyPI (`pip install adjacency-agents`). For the next version:

1. Bump `version = "..."` in `pyproject.toml`.
2. Move the `[Unreleased]` block in `CHANGELOG.md` into a dated `[x.y.z]`
   section and add the comparison link at the bottom.
3. Commit, tag, push:
   ```bash
   git commit -am "release: vX.Y.Z"
   git tag vX.Y.Z
   git push origin main vX.Y.Z
   ```
4. `.github/workflows/release.yml` builds, validates with
   `twine check --strict`, and publishes to PyPI via OIDC trusted
   publishing. No token storage involved.

Both publish steps use `skip-existing: true`, so re-triggering a tag (via
force-push during a history rewrite, for example) is a safe no-op.

## Conventions

- Commit messages are in English (the README and DDD spec are bilingual,
  but the git history is English).
- Public symbols are exported only via `src/adjacency_agents/__init__.py`.
  Do not encourage callers to import from internal modules — the facade
  is rigorous (DDD §8.1).
- New behavior is added by writing the failing test first, then the
  minimal code. The DDD spec calls this out explicitly (§1).
- When a behavior added to the engine concerns sanitization (what reaches
  the LLM, what reaches a trace), add an assertion in the corresponding
  test that proves the sensitive value does **not** appear.
