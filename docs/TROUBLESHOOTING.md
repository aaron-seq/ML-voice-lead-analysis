# Troubleshooting

## Installation

**`ModuleNotFoundError: No module named 'app'`**
Run from the `backend/` directory, or install with `pip install -e backend`.

**Old instructions mention spaCy / `en_core_web_md`.**
No longer needed. The engine is pure Python. If a stale guide tells you to
download a spaCy model, ignore it.

## Analysis

**"Speaker roles could not be inferred" warning.**
The transcript is undiarised. Signals are attributed conservatively and
engagement is scored neutrally, so scores are less reliable. Supply speaker
labels (`Rep: …` / `Prospect: …`) for a materially better analysis.

**Everything scores low.**
Usually undiarised input, or very short transcripts. Check
`analysis.metadata.warnings` and `lead_score.confidence` — low confidence means
the score rests on thin evidence. Verify with:

```bash
python -m app.cli.main analyse your-call.txt
```

The breakdown shows exactly which components scored zero.

**A phrase my team uses is not detected.**
Expected — the lexicons favour precision. Add it to
`backend/app/analysis/lexicons.py` with a test. See [SCORING.md](SCORING.md).

**Scores changed after an upgrade.**
The engine version changed. Every analysis records `metadata.engine_version`.
Re-analyse stored calls to make them comparable:

```bash
curl -X POST localhost:8000/api/v1/calls/<id>/reanalyse
```

**`empty_transcript` on a file that has content.**
Everything was removed as transcription artifacts — a file of only
`[inaudible]` markers and filler words cleans to nothing.

**`unsupported_format` on a JSON file.**
The shape was not recognised. `context.top_level_keys` in the error lists what
was found. Supported: AWS Transcribe, Whisper, `{"segments": [...]}`,
`{"turns": [...]}`, `{"transcript": "..."}`.

## API

**Connection refused from the dashboard.**
The API is not running, or `NEXT_PUBLIC_API_URL` points elsewhere. Check
`curl localhost:8000/health/ready`. Note that `NEXT_PUBLIC_*` values are baked
in **at build time** — changing one requires a rebuild, not a restart.

**CORS errors in the browser.**
Add the dashboard's exact origin to `CORS_ORIGINS`. Wildcards like
`https://*.vercel.app` do **not** work — Starlette has no glob support. Use
`CORS_ORIGIN_REGEX`.

**401 with a key that should work.**
Keys must be ≥ 16 characters and formatted `key:role` in `API_KEYS`. The server
refuses to start with a shorter key.

**503 from `/health/ready`.**
A dependency is down. The response body names which component and why.

## Database

**`no such table: calls`.**
Tables are created at startup. If you pointed `DATABASE_URL` at a fresh
database, restart the app.

**Stale data after deleting the SQLite file.**
A running server holds an open handle to the deleted inode and keeps serving the
old database. Restart it.

**Postgres connection failures.**
Install the driver: `pip install -r requirements-postgres.txt`. The URL must use
`postgresql+asyncpg://`, not `postgres://`.

## Frontend

**Build fails on a missing module.**
Run `npm ci` from `frontend/`. If it persists, `rm -rf node_modules .next` and
reinstall.

**Type errors after changing the API.**
`frontend/src/lib/types.ts` mirrors the backend contract.
`backend/tests/integration/test_contract.py` fails when they diverge — read that
failure first; it names the missing field.

## Getting help

Include the `request_id` from the error response or the `X-Request-ID` header.
It correlates to every server log line for that request.
