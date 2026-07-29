# Deployment

## Docker Compose (recommended)

```bash
cp .env.example .env      # then edit
docker compose up --build
```

Brings up the API (8000), the dashboard (3000) and PostgreSQL. Data persists in
named volumes.

For production, generate a key and set the environment first:

```bash
docker compose run --rm api python -m app.cli.main keygen --role admin
```

## Single container

```bash
docker build -t callsense-api ./backend
docker run -p 8000:8000 \
  -e ENVIRONMENT=production \
  -e SECRET_KEY="$(openssl rand -base64 48)" \
  -e API_KEYS="cs_yourkey:admin" \
  -e CORS_ORIGINS=https://calls.example.com \
  -e DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/callsense \
  -v callsense-data:/data \
  callsense-api
```

## Without Docker

```bash
cd backend
pip install -r requirements.txt -r requirements-postgres.txt
gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 4 --bind 0.0.0.0:8000
```

Note that the rate limiter is per process — with 4 workers the effective limit
is 4× `RATE_LIMIT_PER_MINUTE`.

## Frontend

The dashboard is a static Next.js app that talks to the API from the browser.

```bash
cd frontend
NEXT_PUBLIC_API_URL=https://api.example.com npm run build
npm start
```

`NEXT_PUBLIC_*` values are compiled in at **build** time. Changing one requires
a rebuild.

## Database

SQLite is the default and is genuinely fine for a single team — analyses are
small and writes are infrequent. Move to PostgreSQL when you want concurrent
writers, backups, or more than one API process.

```bash
pip install -r requirements-postgres.txt
export DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/callsense
```

Tables are created at startup. For schema changes against a populated
production database, introduce Alembic — see [OPERATIONS.md](OPERATIONS.md).

## Object storage

Local filesystem by default. For multi-instance deployments use S3 so every
instance sees the same transcripts:

```bash
STORAGE_BACKEND=s3
S3_BUCKET=your-bucket
AWS_REGION=us-east-1
```

Credentials come from the instance role or the standard AWS chain. Do not put
access keys in the environment if you can attach a role.

## Reverse proxy

Terminate TLS upstream. Minimal nginx:

```nginx
location /api/ {
    proxy_pass         http://api:8000;
    proxy_set_header   Host $host;
    proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header   X-Forwarded-Proto $scheme;
    proxy_set_header   X-Request-ID $request_id;   # preserved end to end
    client_max_body_size 26m;                      # match MAX_UPLOAD_BYTES
}
```

## Health probes

| Probe | Endpoint | Behaviour |
|---|---|---|
| Liveness | `/health/live` | Never touches dependencies — a storage blip must not trigger a restart |
| Readiness | `/health/ready` | 503 when the database or storage is unreachable |

## Platform notes

**Render / Railway / Fly** — build `pip install -r backend/requirements.txt`,
start `uvicorn app.main:app --host 0.0.0.0 --port $PORT` with working directory
`backend`. Attach a persistent volume or use Postgres + S3; container
filesystems are ephemeral.

**Vercel** — suitable for the dashboard. The API needs a persistent database and
writable storage, so host it elsewhere and point `NEXT_PUBLIC_API_URL` at it.

**AWS Lambda** — the engine has no heavyweight dependencies, so it fits well
under the size limit. Use Postgres (RDS/Aurora Serverless) and S3; the local
SQLite and filesystem defaults will not survive between invocations.

See the production checklist in [SECURITY.md](SECURITY.md#production-checklist).
