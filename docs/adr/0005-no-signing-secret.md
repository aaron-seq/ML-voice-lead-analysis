# ADR 0005 — No `SECRET_KEY` until something signs something

**Status:** Accepted

## Context

`Settings` declared a `secret_key`, auto-generated locally and **validated as
≥32 characters in staging and production**. A self-audit found that nothing in
the codebase ever read it.

The effect was worse than a merely unused setting: the production validator
forced operators to generate, inject and rotate a secret that protected
nothing, and its presence implied signed tokens or encrypted state existed.
`sentry_dsn` and `analysis_timeout_s` were dead in the same way.

## Decision

Remove all three, and the `SECRET_KEY` production validator with them.
Authentication compares API keys as SHA-256 digests
(`hmac.compare_digest`); nothing is signed or encrypted, so no signing key is
needed. A test asserts `"secret_key" not in Settings.model_fields` so it cannot
return without the code that uses it.

`metrics_enabled` was the fourth dead setting. Rather than delete it, a real
Prometheus endpoint was implemented (`app/core/metrics.py`), and a test asserts
`METRICS_ENABLED=false` returns 404 — the setting now does something.

## Consequences

**Good.** Configuration surface is honest: every setting changes behaviour, and
a test proves it for the ones that previously did not.

**Good.** One less secret for operators to manage.

**Cost.** Adding signed sessions or JWTs later means reintroducing the setting.
That is correct order of operations — the secret arrives with the code that
consumes it.

## Revisit if

Signed sessions, JWTs, webhook signature verification, or encrypted export
links are implemented. Introduce the secret in the same change.
