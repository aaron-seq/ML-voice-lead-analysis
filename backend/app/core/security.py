"""Authentication, authorisation and rate limiting.

Deliberately modest in scope: static API keys with roles. That is honestly
what a self-hosted, single-team deployment needs, and it is enough to make
"who deleted that call?" answerable. It is *not* multi-tenant SSO, and
``docs/SECURITY.md`` says so plainly rather than implying otherwise.

Keys are compared with ``secrets.compare_digest`` and are stored as hashes in
memory, so a heap dump or an accidental ``repr`` of the registry does not
disclose them.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import IntEnum

from app.core.errors import AuthenticationError, AuthorizationError, RateLimitError

logger = logging.getLogger(__name__)


class Role(IntEnum):
    """Ordered so that permission checks are a simple comparison.

    VIEWER  - read calls and analytics
    ANALYST - VIEWER plus upload, re-analyse and export
    ADMIN   - ANALYST plus delete and audit-log access
    """

    VIEWER = 10
    ANALYST = 20
    ADMIN = 30

    @classmethod
    def parse(cls, value: str) -> Role:
        try:
            return cls[value.strip().upper()]
        except KeyError as exc:
            raise ValueError(
                f"Unknown role '{value}'. Expected one of: "
                + ", ".join(r.name.lower() for r in cls)
            ) from exc

    @property
    def label(self) -> str:
        return self.name.lower()


@dataclass(frozen=True, slots=True)
class Principal:
    """The authenticated caller."""

    id: str
    role: Role

    def require(self, minimum: Role) -> None:
        if self.role < minimum:
            raise AuthorizationError(
                f"This action requires the '{minimum.label}' role; "
                f"'{self.id}' has '{self.role.label}'.",
                required=minimum.label,
                actual=self.role.label,
            )


# Used when auth is disabled, i.e. local development only. Named so it is
# obvious in audit logs that no real identity was established.
ANONYMOUS_ADMIN = Principal(id="local-dev", role=Role.ADMIN)


def _hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


class ApiKeyRegistry:
    """Maps API keys to principals.

    Keys are supplied as ``key:role`` strings. The key's first eight characters
    become the principal id so audit entries are traceable without storing the
    secret itself.
    """

    def __init__(self, entries: list[str]) -> None:
        self._by_hash: dict[str, Principal] = {}
        for entry in entries:
            key, _, role_text = entry.partition(":")
            key = key.strip()
            if not key:
                continue
            if len(key) < 16:
                raise ValueError(
                    "API keys must be at least 16 characters. Generate one with "
                    "`python -m app.cli.main keygen`."
                )
            role = Role.parse(role_text) if role_text.strip() else Role.ANALYST
            self._by_hash[_hash_key(key)] = Principal(id=f"key-{key[:8]}", role=role)

    def resolve(self, key: str | None) -> Principal:
        if not key:
            raise AuthenticationError("Missing API key. Send it as 'X-API-Key'.")

        candidate = _hash_key(key)
        # Compare against every entry so timing does not reveal which prefix
        # matched; the loop is over a handful of keys, so the cost is trivial.
        matched: Principal | None = None
        for stored_hash, principal in self._by_hash.items():
            if hmac.compare_digest(candidate, stored_hash):
                matched = principal
        if matched is None:
            raise AuthenticationError("Invalid API key.")
        return matched

    def __len__(self) -> int:
        return len(self._by_hash)

    def __repr__(self) -> str:  # pragma: no cover - avoids leaking hashes
        return f"<ApiKeyRegistry keys={len(self._by_hash)}>"


def generate_api_key() -> str:
    """Mint a key suitable for ``API_KEYS``."""
    return f"cs_{secrets.token_urlsafe(32)}"


class SlidingWindowRateLimiter:
    """Per-principal request budget over a rolling 60-second window.

    In-process, so each worker enforces its own budget -- with N workers the
    effective limit is N x the configured value. Documented rather than
    hidden; a shared Redis limiter is on the roadmap and the interface here
    is what it would implement.
    """

    def __init__(self, limit_per_minute: int) -> None:
        self._limit = limit_per_minute
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, principal_id: str) -> None:
        if self._limit <= 0:
            return

        now = time.monotonic()
        window = self._hits[principal_id]
        cutoff = now - 60.0
        while window and window[0] < cutoff:
            window.popleft()

        if len(window) >= self._limit:
            retry_after = max(1, int(60 - (now - window[0])))
            raise RateLimitError(
                f"Rate limit of {self._limit} requests/minute exceeded.",
                retry_after_seconds=retry_after,
            )
        window.append(now)

    def reset(self) -> None:
        self._hits.clear()
