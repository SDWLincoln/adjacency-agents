# Testing

## Running the suite

```bash
.venv/bin/pytest
```

The suite ships at **136 tests** organized by concern. There is no
network or filesystem I/O — every test runs against `FakeLLMClient`.

```bash
# Focused runs while iterating
.venv/bin/pytest tests/test_engine.py
.venv/bin/pytest tests/test_engine.py -k pointer
.venv/bin/pytest -ra                    # show short reasons for non-pass results
```

`pyproject.toml` sets `asyncio_mode = "auto"`, so any `async def test_…`
function is collected without a decorator.

## Layout

| File | Spec coverage |
|------|---------------|
| `test_models.py` | §9 — UserContext, Message, ToolPolicy semantics, EnrichedPointer, Observation, ToolCall, FinalAnswer |
| `test_errors.py` | §19 — exception hierarchy, `__cause__` preservation |
| `test_decorators.py` | §10, §24.3 — `@tool_node` rules, inject keys validated, async preserved |
| `test_registry.py` | §11, §24.2 — duplicates, missing neighbors, function identity preserved |
| `test_router.py` | §12, §24.1 — every policy semantics matrix |
| `test_schema.py` | §13, §17, §24.4 — Pydantic validation, list/dict/Literal/Optional, injected exclusion |
| `test_fake_llm.py` | §18.3.9 — scripted client contract |
| `test_engine.py` | §14, §24.5 — main loop, allowlist exposure, response handling, multi-turn |
| `test_async_engine.py` | §14.3, §24.5.29–31 — `ainvoke`, async tools, sync tool via thread, `AsyncRequiredError` |
| `test_pointer_transitions.py` | §24.8 — A→B→Obs and A→B→C→Obs chains, max_steps |
| `test_synthesis.py` | §14.7 — tools=[] enforced, SynthesisError on bad ToolCall, expose_to_llm=False |
| `test_tool_execution_errors.py` | §14.9, §24.7, §35.1 — wrapping, chain leakage |
| `test_context_injection.py` | §17, §24.6 — engine-level injection + pointer rejection |
| `test_tracing.py` | §20 — events present, no leakage of prompt/kwargs/payload/reason |

## Spec → test traceability

When you add a behavior to the spec, add the corresponding test under
the matching file. When a new test deviates from the spec, update the
spec *in the same PR*. The DDD discipline is preserved by keeping
those two files in sync.

## Adding a new tool-error scenario

```python
@tool_node(requires=["public"])
def boom() -> str:
    raise SomeError("...")

fake = FakeLLMClient(script=[ToolCall(name="boom")])
eng = DeterministicEngine(
    llm=fake,
    tools=[boom],
    tool_error_mode="...",
)
```

Any test that asserts the synthesis prompt should walk
`fake.calls[1]["messages"]` and assert that internal names, tracebacks
and tool kwargs do *not* appear (§35.3).

## Async tests

`pytest-asyncio` is installed and `asyncio_mode` is `auto`, so just
write:

```python
async def test_my_async_thing():
    ...
```

If you need an event loop in an otherwise sync test (e.g. to assert
`AsyncRequiredError`), wrap the inner call in a small `async def main`
and `asyncio.run(main())`.

## Smoke testing the examples

```bash
.venv/bin/python examples/fake_llm_example.py
.venv/bin/python examples/structural_pointer_example.py
.venv/bin/python examples/logged_vs_guest_example.py
```

CI runs these three scripts after the test suite passes.
