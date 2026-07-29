# Operations

## Observability

**Logs.** Set `LOG_FORMAT=json` in production. Every line carries `request_id`
and `principal`:

```json
{"ts":"2026-07-29T06:30:58+0000","level":"INFO","logger":"app.access",
 "message":"POST /api/v1/calls -> 201 (88.5ms)","request_id":"2736546083fc42e1",
 "principal":"key-cs_abc12","method":"POST","status":201,"duration_ms":88.5}
```

An inbound `X-Request-ID` is honoured, so traces stitch together across a proxy.
Health-check paths are excluded from the access log to keep it readable.

**Timing.** Every response carries `Server-Timing: app;dur=<ms>`. Analysis
duration is recorded per call in `metadata.processing_ms`.

**Audit.** `GET /api/v1/audit` lists ingest, re-analyse, delete and export
events with principal and timestamp.

## Common tasks

**Bulk import**

```bash
python -m app.cli.main import /path/to/transcripts --owner alice
```

**Re-analyse after a scoring change**

```bash
for id in $(curl -s 'localhost:8000/api/v1/calls?page_size=100' | jq -r '.items[].id'); do
  curl -sX POST "localhost:8000/api/v1/calls/$id/reanalyse" > /dev/null
done
```

**Backup**

SQLite: copy `data/callsense.db` plus `data/blobs/`.
Postgres: `pg_dump` plus the object-storage bucket. The database holds every
analysis; blob storage holds the raw transcripts they were derived from.

## Schema migrations

Tables are created at startup via `create_all`, which is adequate for SQLite and
first-run Postgres. For schema *changes* against a populated database:

```bash
pip install alembic
alembic init migrations
# point alembic.ini at DATABASE_URL, set target_metadata = app.db.models.Base.metadata
alembic revision --autogenerate -m "add column"
alembic upgrade head
```

Alembic is deliberately not vendored — adding migration infrastructure before
there is a production deployment with data to migrate is premature.

## Capacity

Analysis is ~3 ms per call, so the engine is never the bottleneck; the database
and blob storage are. Rough guidance:

| Scale | Setup |
|---|---|
| < 1k calls | SQLite + local filesystem, single process |
| 1k–50k | Postgres + S3, 2–4 workers |
| > 50k | Add the normalised signals table (see [ROADMAP.md](ROADMAP.md)) — analytics frequency queries read analysis JSON and will slow down |

## Failure modes

| Symptom | Cause | Action |
|---|---|---|
| `/health/ready` 503, database unhealthy | DB unreachable | Check connectivity; the API still serves cached reads but writes fail |
| `/health/ready` 503, storage unhealthy | Disk full or S3 denied | Ingestion fails; existing analyses still readable from the database |
| Calls stuck in `failed` | Unparseable transcripts | `error` on the record explains why; the raw blob is retained for re-analysis |
| Rising 429s | Rate limit reached | Raise `RATE_LIMIT_PER_MINUTE`, remembering it is per process |
