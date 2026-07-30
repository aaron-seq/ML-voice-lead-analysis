# Security

## What is protected

| Control | Implementation |
|---|---|
| Authentication | Static API keys, SHA-256 hashed in memory, compared with `hmac.compare_digest` |
| Authorisation | Three ordered roles: viewer < analyst < admin |
| Rate limiting | Per-principal sliding window, 120 req/min default |
| Input validation | Pydantic on every request; upload size capped by reading the stream, not trusting `Content-Length` |
| SQL injection | Parameterised queries throughout; `sort_by` restricted to an allowlist |
| Path traversal | Blob keys are derived server-side from validated call IDs, regex-restricted, and re-checked against the storage root after resolution |
| Transport headers | CSP, `X-Frame-Options: DENY`, nosniff, referrer policy; HSTS in staging/production |
| CORS | Explicit origin list; `*` rejected in production; credentials disabled (auth is a header, not a cookie) |
| Secrets | AWS credentials only from the provider chain, never from settings. Keys never logged; `ApiKeyRegistry.__repr__` redacts |
| Error handling | Internal exception messages never reach clients; correlated by request ID in logs |
| Audit | Ingest, re-analyse, delete and export recorded with principal, timestamp and detail |
| Dependency scanning | `pip-audit` and `npm audit` run in CI and fail the build on a new advisory |
| Static analysis | `bandit` runs in CI over the backend |

## What is not

Be clear-eyed about this before deploying.

- **Static API keys are not user identity.** There is no login, no session, no
  per-user attribution beyond the key prefix, and no rotation workflow.
  Adequate for one team behind a VPN; not for a multi-tenant deployment.
  SSO/OIDC is Phase 4 on the [roadmap](ROADMAP.md).
- **No encryption at rest** beyond what your database and object store provide.
  S3 writes use SSE-S3; SQLite files are not encrypted.
- **Rate limiting is per process.** With N workers the effective limit is N×.
  A shared Redis limiter is on the roadmap.
- **No PII detection or redaction.** Transcripts are stored as supplied. If your
  calls contain card numbers or health information, that data lands in your
  database unmodified.
- **No retention policy.** Nothing is deleted automatically.
- **`/metrics` and the health endpoints are unauthenticated.** They expose
  counters and latencies only — no transcript content, no customer data, no
  configuration. Restrict them at the ingress if your scrape target is not on a
  trusted network.
- **One unfixable-by-upgrade advisory may remain in the frontend tree at any
  time.** `npm audit` runs in CI; when an advisory has no forward fix, it is
  recorded here rather than suppressed silently. As of the last audit the
  frontend tree is clean (four libvips CVEs in `sharp` were resolved by an
  `overrides` pin to `^0.35.3`, since Next pulls a vulnerable 0.34.x and the
  dashboard uses no `next/image`).
- **`NEXT_PUBLIC_API_KEY` is visible to anyone using the dashboard**, because
  Next.js compiles it into the client bundle. Use a viewer/analyst key, never an
  admin key, or front the dashboard with your own auth proxy.

## Production checklist

```bash
ENVIRONMENT=production          # forces auth on, disables docs, enables HSTS
API_KEYS="<generated key>:admin"
CORS_ORIGINS=https://calls.yourcompany.com
DATABASE_URL=postgresql+asyncpg://...
STORAGE_BACKEND=s3
S3_BUCKET=your-bucket
LOG_FORMAT=json
```

- [ ] Terminate TLS at a reverse proxy; never expose the app directly
- [ ] Run as a non-root user (the provided Dockerfile does)
- [ ] Restrict the S3 bucket to the instance role; block public access
- [ ] Point liveness at `/health/live` and readiness at `/health/ready`
- [ ] Ship JSON logs somewhere durable — they are the audit trail's companion
- [ ] Back up the database; it holds every analysis
- [ ] Decide and document a retention period for stored conversations

## Reporting a vulnerability

Email <aaronsequeira12@gmail.com> rather than opening a public issue.
