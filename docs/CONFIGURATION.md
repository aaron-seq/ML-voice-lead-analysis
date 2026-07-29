# Configuration

All settings come from the environment (or a `.env` file beside the backend).
Defaults are chosen so a fresh clone runs with **no configuration at all**.

Settings are validated at startup. A misconfigured production deployment fails
immediately with a specific message rather than misbehaving later.

## Core

| Variable | Default | Notes |
|---|---|---|
| `ENVIRONMENT` | `local` | `local`, `test`, `staging`, `production` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | |
| `API_PREFIX` | `/api/v1` | |
| `LOG_LEVEL` | `INFO` | |
| `LOG_FORMAT` | `console` | Use `json` in production |

Setting `ENVIRONMENT=production` also disables `/docs`, `/redoc` and
`/openapi.json`, enables HSTS, and **forces `AUTH_ENABLED=true`**.

## Database

| Variable | Default |
|---|---|
| `DATABASE_URL` | `sqlite+aiosqlite:///<repo>/data/callsense.db` |
| `DB_ECHO` | `false` |

PostgreSQL: `postgresql+asyncpg://user:pass@host:5432/callsense`
(`pip install -r requirements-postgres.txt`).

## Storage

| Variable | Default | Notes |
|---|---|---|
| `STORAGE_BACKEND` | `local` | `local` or `s3` |
| `STORAGE_LOCAL_PATH` | `<repo>/data/blobs` | |
| `S3_BUCKET` | — | Required when backend is `s3` |
| `S3_PREFIX` | `callsense/` | |
| `AWS_REGION` | `us-east-1` | |

S3 credentials come from the standard AWS provider chain (instance role,
environment, shared config). CallSense never accepts an access key as a setting,
so keys cannot be logged or persisted by accident.

## Security

| Variable | Default | Notes |
|---|---|---|
| `AUTH_ENABLED` | `false` | Forced true in staging/production |
| `API_KEYS` | — | Comma-separated `key:role` pairs |
| `SECRET_KEY` | random | Must be ≥ 32 chars in staging/production |
| `RATE_LIMIT_PER_MINUTE` | `120` | Per principal. `0` disables |
| `CORS_ORIGINS` | `http://localhost:3000,...` | `*` rejected in production |
| `CORS_ORIGIN_REGEX` | — | For dynamic origins such as preview deployments |
| `MAX_UPLOAD_BYTES` | `26214400` (25 MB) | |

```bash
API_KEYS="cs_abc123...:admin,cs_def456...:viewer"
```

Starlette does not support glob patterns in `CORS_ORIGINS` — a value like
`https://*.vercel.app` silently matches nothing. Use `CORS_ORIGIN_REGEX`.

## Analysis engine

| Variable | Default | Notes |
|---|---|---|
| `ENABLE_TRANSFORMER_SENTIMENT` | `false` | Requires `requirements-ml.txt` |
| `TRANSFORMER_MODEL` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | |

If the model cannot load, the engine logs a warning and falls back to the
lexicon analyzer rather than failing.

## Frontend

`NEXT_PUBLIC_*` values are compiled into the client bundle and are visible to
anyone using the dashboard.

| Variable | Default |
|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` |
| `NEXT_PUBLIC_API_KEY` | — |

Never put an admin key here. Use a `viewer`/`analyst` key, or put the dashboard
behind your own SSO proxy.
