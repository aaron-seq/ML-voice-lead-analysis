# ADR 0004 — SQL for analyses, blob storage for transcripts

**Status:** Accepted

## Context

The previous design used S3 objects as the datastore. Listing calls performed
`list_objects_v2` followed by one `GET` per object to read its summary fields,
so rendering a 20-row table downloaded every analysis in the page. Filtering,
sorting and aggregation were impossible without downloading everything.
Analytics were therefore hardcoded constants
(`{"Hot": 15, "Warm": 25, "Cold": 10}`).

S3 was also a hard dependency: no AWS account meant no local development
(issue #9).

## Decision

- **Database** (`calls`, `audit_events`): the full `CallAnalysis` as JSON, plus
  the queryable facts denormalised into indexed columns (`lead_band`,
  `lead_score`, `outcome`, `created_at`, `owner`). Analytics are SQL aggregates.
- **Blob storage**: raw and parsed transcripts — large, opaque, fetched by key.
- SQLite by default so a clone runs with no services; PostgreSQL via a
  connection string, exercised as a separate CI job.

Denormalised columns are written only by `CallRepository.attach_analysis`,
never independently, so they cannot drift from the JSON.

## Consequences

**Good.** List and filter are single indexed queries. Analytics are real.
A fresh clone works offline.

**Good.** Portable: `JSON`/`String` columns rather than Postgres-native
`JSONB`/`UUID`, so one schema serves both engines. CI proves it.

**Cost.** Objection/signal/risk frequency queries read analysis JSON in Python
and will degrade past roughly 10k calls. Documented in `docs/API.md`; the fix
(a normalised signals table) is roadmap Phase 4 rather than speculative work
now.

**Cost.** Two stores to back up.

## Revisit if

Analytics latency becomes user-visible. Add the normalised signals table
written alongside `attach_analysis`; the repository boundary means nothing
outside `db/repository.py` changes.
