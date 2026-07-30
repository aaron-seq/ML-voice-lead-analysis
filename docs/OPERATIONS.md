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

**Metrics.** Prometheus exposition at `/metrics` (disable with
`METRICS_ENABLED=false`, which returns 404):

| Metric | Type | Labels |
|---|---|---|
| `callsense_http_requests_total` | counter | `method`, `route`, `status` |
| `callsense_http_request_duration_seconds` | histogram | `method`, `route` |
| `callsense_analyses_total` | counter | `band`, `outcome` |
| `callsense_analysis_duration_seconds` | histogram | — |
| `callsense_ingestion_failures_total` | counter | `reason` |
| `callsense_uptime_seconds` | gauge | — |

`route` is the **route template** (`/api/v1/calls/{call_id}`), not the request
path, so cardinality does not grow with the number of calls.

**Counters are per worker process.** With 4 workers, a single load-balanced
scrape sees roughly a quarter of the traffic. Scrape each worker as its own
target, or accept that absolute totals are per-instance. The same applies to the
rate limiter. A shared Redis-backed implementation is on the roadmap.

A useful starting alert set:

```promql
# Error rate above 5% over 5 minutes
sum(rate(callsense_http_requests_total{status=~"5.."}[5m]))
  / sum(rate(callsense_http_requests_total[5m])) > 0.05

# Ingestion failing repeatedly — usually a bad upstream transcript export
sum(rate(callsense_ingestion_failures_total[15m])) > 0.1

# p95 request latency above half a second
histogram_quantile(0.95,
  sum(rate(callsense_http_request_duration_seconds_bucket[5m])) by (le)) > 0.5
```

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

Measured on a single core (`pytest tests/test_benchmarks.py -s`):

| | |
|---|---|
| 40-turn call (396 words) | 2.2 ms mean, 2.4 ms p95 |
| Per 1,000 words | 5.0 ms |
| Sustained batch, one thread | 472 calls/s |

The engine is therefore never the bottleneck — the database and blob storage
are. A backlog of 100,000 transcripts imports in roughly four minutes of CPU.
Rough guidance:

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
| `/metrics` returns 404 | `METRICS_ENABLED=false` | Intended behaviour; set it to `true` |
| Metrics totals look low | Per-worker counters | Scrape each worker, or expect `total / workers` |
