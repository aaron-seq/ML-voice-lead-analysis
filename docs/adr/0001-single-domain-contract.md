# ADR 0001 — One canonical domain contract

**Status:** Accepted

## Context

The pipeline serialised analyses with one set of field names
(`analyzed_file_name`, `lead_scoring_results`) while the API deserialised
another (`fileName`, `leadScore`). No analysis produced by the pipeline was
readable through the API. This was open as issue #23.

The defect survived because both sides returned hardcoded mock data whenever
`ENVIRONMENT=testing`, and CI never executed the test suite. Every layer had
its own idea of what a call analysis was: the pipeline had dataclasses, the API
had Pydantic models with camelCase aliases, the database had a third set of
columns, and the frontend had a fourth.

## Decision

`backend/app/domain/models.py` is the only definition of a call analysis.
The engine, the repository, the HTTP layer and the CLI all import it. One
naming convention (snake_case), one serialisation format, no translation layers.

`tests/integration/test_contract.py` asserts that an analysis round-trips
engine → storage → API unchanged, and that every field the dashboard reads is
present in the response.

## Consequences

**Good.** A schema change is a single edit, and the contract test fails loudly
if a layer drifts. The CLI and the API provably produce identical results.

**Cost.** The domain models carry presentation-adjacent fields (`label`,
`detail`, `rationale`) that a purist would push outward. That is deliberate:
those strings are the *explanation*, which is a domain concept here, not
decoration.

**Cost.** No independent versioning per layer. A breaking change requires
bumping `SCHEMA_VERSION` and migrating stored JSON.

## Revisit if

Multiple external consumers need independently versioned API schemas. At that
point add a thin, explicitly versioned API-facing DTO layer — but keep the
domain contract as the single internal truth.
