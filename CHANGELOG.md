# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-05-29

First public release. Implements the full Documentation-Driven
Development specification v0.4 (`adjacency_agents_documentation_driven_development_v0_4_final.md`)
and three provider adapters.

### Added

- **Core engine** (`DeterministicEngine`) with policy-gated parsing,
  triple validation (before schema, before tool execution, before each
  structural transition), `max_steps` cycle protection and
  `ExecutionTrace` exposing 17 sanitized events via
  `engine.last_trace`.
- **Data models**: `UserContext`, `Message`, `ToolPolicy`,
  `EnrichedPointer`, `Observation`, `ToolCall`, `FinalAnswer`. All
  frozen dataclasses.
- **Decorator** `@tool_node` with `requires=`, `policy=`,
  `structural_neighbors=`, `llm_visible=`, `description=`,
  `response_mode=` and `inject=`.
- **Registry & router**: per-engine `ToolRegistry` (no global
  singleton), `build_allowlist`, `build_llm_visible_allowlist`.
- **Schema/validation** backed by Pydantic v2; supports
  `typing.Annotated[T, Field(description=...)]` for per-argument
  descriptions and constraints.
- **Async**: `ainvoke` as the production path, `async def` tools
  awaited directly, sync tools dispatched via `asyncio.to_thread` by
  default (`sync_tool_strategy="thread"`), `AsyncRequiredError` when
  `invoke` is called inside an active event loop.
- **Synthesis** of `Observation` / `dict` / `list` / `BaseModel`
  results in a single LLM call with `tools=[]` and
  `allow_tool_calls=False`; synthesis never receives the catalog,
  schemas, trace, capabilities or raw metadata.
- **Tool error handling**: any exception raised inside a tool body is
  wrapped in `ToolExecutionError` (preserving `__cause__`), even
  subclasses of `AdjacencyAgentsError`. Selectable modes: `raise`,
  `final`, `synthesize`.
- **Context injection** via `inject={"arg": "metadata.path"}` and
  `inject={"arg": "session_id"}`; injected arguments are excluded from
  the LLM schema and rejected if supplied by the LLM or an
  `EnrichedPointer`.
- **OpenAI adapter** (`adjacency_agents.adapters.openai.OpenAIClient`
  and `AsyncOpenAIClient`) for OpenAI Chat Completions; optional extra
  `[openai]`.
- **Anthropic adapter** (`adjacency_agents.adapters.anthropic.AnthropicClient`
  and `AsyncAnthropicClient`) for the Anthropic Messages API; optional
  extra `[anthropic]`.
- **Ollama adapter** (`adjacency_agents.adapters.ollama.OllamaClient`
  and `AsyncOllamaClient`) for local Ollama servers; optional extra
  `[ollama]`.
- **`FakeLLMClient`** for tests and examples, scripting `ToolCall`,
  `FinalAnswer` and `str` responses with `allow_tool_calls`
  enforcement.
- **Documentation**: README, `docs/{ddd,architecture,api,security,testing}.md`,
  six runnable examples (`fake_llm`, `logged_vs_guest`,
  `structural_pointer`, `openai_adapter`, `anthropic_adapter`,
  `ollama_adapter`).
- **PEP 561** `py.typed` marker for downstream type checkers.
- **CI** (`.github/workflows/ci.yml`) running `ruff check`,
  `ruff format --check`, `mypy` and the full `pytest` suite on Python
  3.10, 3.11 and 3.12 — plus a smoke run of every example without a
  real provider.

### Tested

- 170 pytest cases covering every mandatory invariant from the DDD
  spec (§23) and the mandatory test list (§24). Ruff and mypy clean
  across `src`, `tests` and `examples`.

[Unreleased]: https://github.com/SDWLincoln/adjacency-agents/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/SDWLincoln/adjacency-agents/releases/tag/v0.1.0
