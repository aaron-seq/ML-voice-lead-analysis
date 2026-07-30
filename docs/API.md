# API reference

Base URL `http://localhost:8000`, all endpoints under `/api/v1`.
Interactive docs (non-production): `/api/v1/docs`. OpenAPI: `/api/v1/openapi.json`.

## Authentication

Disabled by default (local development). When `AUTH_ENABLED=true`, send:

```
X-API-Key: cs_your_key_here
```

| Role | Can |
|---|---|
| `viewer` | Read calls and analytics |
| `analyst` | + upload, re-analyse, export |
| `admin` | + delete, full audit log |

Mint a key with `python -m app.cli.main keygen --role analyst`.

## Errors

Every error returns the same shape:

```json
{
  "code": "not_found",
  "message": "Call 'abc123' not found",
  "context": { "call_id": "abc123" },
  "request_id": "a1b2c3d4e5f6a7b8"
}
```

`request_id` also appears in the `X-Request-ID` response header and in every
server log line for that request — quote it in bug reports.

| Code | Status | Meaning |
|---|---|---|
| `validation_error` | 400 | A parameter was rejected |
| `empty_transcript` | 400 | Nothing usable after cleaning |
| `unsupported_format` | 400 | Content matched no known transcript dialect |
| `invalid_request` | 422 | Schema validation failed; `context.errors` lists fields |
| `unauthenticated` | 401 | Missing or invalid API key |
| `forbidden` | 403 | Role too low; `context` names the required role |
| `not_found` | 404 | No such call |
| `rate_limited` | 429 | Budget exceeded; honour `Retry-After` |
| `storage_error` | 502 | Blob storage unreachable |

---

## Calls

### `GET /api/v1/calls`

Paginated, filterable list. Excludes the full analysis blob by design — use the
detail endpoint for that.

| Parameter | Type | Notes |
|---|---|---|
| `page` | int ≥ 1 | Default 1 |
| `page_size` | int 1–100 | Default 20 |
| `sort_by` | enum | `created_at`, `updated_at`, `lead_score`, `sentiment`, `duration_s`, `title`. Anything else is rejected. |
| `sort_desc` | bool | Default true |
| `search` | string | Case-insensitive substring of the title |
| `band` | repeatable | `hot`, `warm`, `cold`, `disqualified` |
| `status` | repeatable | `pending`, `running`, `completed`, `failed` |
| `outcome` | repeatable | See outcome labels below |
| `owner` | string | Exact match |
| `min_score`, `max_score` | 0–100 | |
| `created_after`, `created_before` | ISO 8601 | |

```bash
curl 'localhost:8000/api/v1/calls?band=hot&band=warm&min_score=60&sort_by=lead_score'
```

```json
{
  "items": [{
    "id": "a7b2a2c1641d46b2",
    "title": "Northwind — discovery",
    "created_at": "2026-07-29T06:30:58.160197Z",
    "status": "completed",
    "source_format": "diarized_text",
    "lead_score": 73.9,
    "lead_band": "hot",
    "outcome": "meeting_booked",
    "sentiment": 0.1772,
    "duration_s": 106.8,
    "word_count": 267,
    "owner": "alice",
    "tags": ["q3"],
    "error": null
  }],
  "pagination": {
    "page": 1, "page_size": 20, "total_items": 1,
    "total_pages": 1, "has_next": false, "has_previous": false
  }
}
```

### `POST /api/v1/calls`  *(analyst)*

Ingest a transcript supplied inline. Returns **201** with the full analysis.

```json
{ "title": "Acme — discovery", "transcript": "Rep: Hi...", "owner": "alice", "tags": ["q3"] }
```

### `POST /api/v1/calls/upload`  *(analyst)*

Multipart upload. Fields: `file` (required), `title`, `owner`, `tags`
(comma-separated). Title is derived from the filename when omitted. The body is
read with an enforced size cap rather than trusting `Content-Length`.

```bash
curl -X POST localhost:8000/api/v1/calls/upload \
     -F file=@call.txt -F title='Acme — discovery' -F owner=alice
```

### `POST /api/v1/calls/analyse`  *(analyst)*

Analyse without persisting anything. Nothing touches the database or blob
storage — suitable for callers who may not store customer conversations here.

```json
{ "transcript": "Rep: Hi...\nProspect: We have budget." }
```

Returns `{ "analysis": {...}, "transcript": {...} }`.

### `GET /api/v1/calls/{call_id}`

Full call with its analysis. Add `?include_transcript=true` for the parsed,
role-attributed turns.

### `PATCH /api/v1/calls/{call_id}`  *(analyst)*

Update `title`, `owner` or `tags`. Does not re-run analysis.

### `POST /api/v1/calls/{call_id}/reanalyse`  *(analyst)*

Re-run the current engine over the stored transcript. Used after a scoring
change to bring old calls up to date. Runs synchronously — analysis is
single-digit milliseconds.

### `DELETE /api/v1/calls/{call_id}`  *(admin)*

Permanently deletes the call and its stored blobs. **204**. Audited.

---

## Analytics

### `GET /api/v1/analytics/pipeline`

`days` (1–365, default 30) sets the window for the frequency breakdowns. Totals
and band/outcome counts cover the whole dataset.

```json
{
  "total_calls": 5, "analysed_calls": 5, "failed_calls": 0,
  "avg_lead_score": 40.3, "avg_sentiment": 0.08,
  "band_breakdown": { "hot": 1, "warm": 2, "cold": 0, "disqualified": 2 },
  "outcome_breakdown": { "meeting_booked": 2, "not_interested": 1 },
  "top_objections": [{ "category": "authority", "count": 2, "unaddressed": 1 }],
  "top_signals": [{ "type": "need", "count": 4 }],
  "top_risks": [{ "risk": "no_decision_maker", "count": 4 }],
  "trend": [{ "date": "2026-07-29", "calls": 5, "avg_lead_score": 40.3 }]
}
```

**Known scaling limit:** objection/signal/risk frequencies are computed by
reading analysis JSON for calls in the window. Beyond roughly 10k calls this
should move to a normalised signals table — see [ROADMAP.md](ROADMAP.md).

### `GET /api/v1/export/calls.csv`  *(analyst)*

One row per call, flattened for a spreadsheet or CRM import. Accepts `band`,
`owner`, `min_score`, `created_after`. Audited.

Capped at 10,000 rows, newest first, because every row carries its full
analysis JSON. When the cap is hit the response sets `X-Export-Truncated: true`
— narrow the filters and export again rather than assuming you have everything.

### `GET /api/v1/audit`  *(admin)*

Recent audit events: ingest, re-analyse, delete, export. `limit` 1–500.

Admin-only: the log names the principal behind every upload, export and
deletion, so it is not readable by the analysts it records.

---

## System

| Endpoint | Purpose |
|---|---|
| `GET /health/live` | Liveness. Touches no dependencies. Use for restart probes. |
| `GET /health/ready` | Readiness. Checks database and storage; **503** when unhealthy. Use for load-balancer probes. |
| `GET /` | Service metadata |

Conflating these causes bad deploys: an orchestrator should not restart a pod
because S3 had a blip.

---

## Reference values

**Lead bands** `hot` · `warm` · `cold` · `disqualified`

**Outcomes** `meeting_booked` · `follow_up_agreed` · `information_requested` ·
`not_interested` · `not_a_fit` · `gatekeeper_only` · `no_clear_outcome`

**Signal types** `budget` · `authority` · `need` · `timeline` · `urgency` ·
`competitor` · `enthusiasm` · `next_step` · `technical_fit` · `social_proof`

**Objection categories** `price` · `timing` · `authority` · `competitor` ·
`need` · `trust` · `complexity`

**Risk flags** `no_decision_maker` · `budget_unconfirmed` · `no_next_step` ·
`competitor_incumbent` · `low_engagement` · `rep_monologue` ·
`negative_sentiment_trend`

**Action priorities** `p0` (today) · `p1` (this week) · `p2` (backlog)
