# Documentation-Driven Development

The full DDD specification for this library lives at the repo root:

[`adjacency_agents_documentation_driven_development_v0_4_final.md`](../adjacency_agents_documentation_driven_development_v0_4_final.md)

It is the authoritative source. Implementation is considered correct
only if it satisfies the contracts, invariants and acceptance criteria
defined there.

## Workflow

The spec mandates the following loop (§1):

1. Every feature begins by updating or validating the documentation.
2. Every public behavior is described before being implemented.
3. Every described behavior has an automated test.
4. No public API is added without a documented usage example.
5. No security rule lives only in code — it is documented as an
   invariant.

## Reading order for new contributors

1. **§2–§4** — problem, solution, architectural principles. This is the
   *thesis*. Read it before touching the code.
2. **§5** — ubiquitous language. Defines `Capability`, `ToolPolicy`,
   `EnrichedPointer`, `Observation`, etc.
3. **§8** — public contract and examples.
4. **§9, §10, §14** — data models, decorator, engine.
5. **§23** — invariant checklist. Every PR must preserve these.
6. **§24** — mandatory test cases.

## Local doc files

| File | Purpose |
|------|---------|
| [`architecture.md`](./architecture.md) | Module layout, control flow, design decisions |
| [`api.md`](./api.md) | Public surface and signatures |
| [`security.md`](./security.md) | Security invariants, threat model, anti-patterns |
| [`testing.md`](./testing.md) | How to run and extend the test suite |
