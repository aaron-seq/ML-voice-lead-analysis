"""Auth, rate limiting, configuration invariants and blob storage."""

from __future__ import annotations

import pytest

from app.core.config import Environment, Settings
from app.core.errors import (
    AuthenticationError,
    AuthorizationError,
    NotFoundError,
    RateLimitError,
    StorageError,
)
from app.core.security import (
    ApiKeyRegistry,
    Principal,
    Role,
    SlidingWindowRateLimiter,
    generate_api_key,
)
from app.storage.local import LocalBlobStore

VALID_KEY = "cs_test_key_that_is_long_enough_123"


class TestApiKeyRegistry:
    def test_resolves_a_valid_key(self):
        registry = ApiKeyRegistry([f"{VALID_KEY}:analyst"])
        assert registry.resolve(VALID_KEY).role is Role.ANALYST

    def test_rejects_an_unknown_key(self):
        registry = ApiKeyRegistry([f"{VALID_KEY}:analyst"])
        with pytest.raises(AuthenticationError):
            registry.resolve("cs_wrong_key_also_long_enough_xx")

    def test_rejects_a_missing_key(self):
        with pytest.raises(AuthenticationError):
            ApiKeyRegistry([f"{VALID_KEY}:admin"]).resolve(None)

    def test_defaults_to_analyst_when_no_role_given(self):
        assert ApiKeyRegistry([VALID_KEY]).resolve(VALID_KEY).role is Role.ANALYST

    def test_short_keys_are_refused_at_configuration_time(self):
        """Fail at boot, not on the first unauthorised request."""
        with pytest.raises(ValueError, match="at least 16"):
            ApiKeyRegistry(["short:admin"])

    def test_unknown_role_is_refused(self):
        with pytest.raises(ValueError, match="Unknown role"):
            ApiKeyRegistry([f"{VALID_KEY}:wizard"])

    def test_repr_does_not_leak_key_material(self):
        registry = ApiKeyRegistry([f"{VALID_KEY}:admin"])
        assert VALID_KEY not in repr(registry)

    def test_generated_keys_are_long_and_unique(self):
        keys = {generate_api_key() for _ in range(50)}
        assert len(keys) == 50
        assert all(len(k) >= 32 for k in keys)


class TestRoles:
    def test_roles_are_ordered(self):
        assert Role.VIEWER < Role.ANALYST < Role.ADMIN

    def test_higher_role_satisfies_lower_requirement(self):
        Principal("p", Role.ADMIN).require(Role.ANALYST)

    def test_lower_role_is_refused(self):
        with pytest.raises(AuthorizationError):
            Principal("p", Role.VIEWER).require(Role.ADMIN)


class TestRateLimiter:
    def test_allows_traffic_under_the_limit(self):
        limiter = SlidingWindowRateLimiter(5)
        for _ in range(5):
            limiter.check("p")

    def test_blocks_traffic_over_the_limit(self):
        limiter = SlidingWindowRateLimiter(3)
        for _ in range(3):
            limiter.check("p")
        with pytest.raises(RateLimitError) as exc:
            limiter.check("p")
        assert exc.value.context["retry_after_seconds"] >= 1

    def test_budgets_are_per_principal(self):
        limiter = SlidingWindowRateLimiter(2)
        limiter.check("alice")
        limiter.check("alice")
        limiter.check("bob")  # must not be affected by alice

    def test_zero_disables_limiting(self):
        limiter = SlidingWindowRateLimiter(0)
        for _ in range(500):
            limiter.check("p")


class TestSettingsInvariants:
    def test_local_defaults_need_no_configuration(self):
        settings = Settings(environment=Environment.LOCAL)
        assert settings.auth_enabled is False
        assert settings.database_url.startswith("sqlite")

    def test_production_forces_auth_on(self):
        settings = Settings(
            environment=Environment.PRODUCTION,
            secret_key="x" * 40,
            api_keys=[f"{VALID_KEY}:admin"],
        )
        assert settings.auth_enabled is True

    def test_production_rejects_a_weak_secret(self):
        with pytest.raises(ValueError, match="SECRET_KEY"):
            Settings(
                environment=Environment.PRODUCTION,
                secret_key="short",
                api_keys=[f"{VALID_KEY}:admin"],
            )

    def test_production_rejects_wildcard_cors(self):
        with pytest.raises(ValueError, match="CORS"):
            Settings(
                environment=Environment.PRODUCTION,
                secret_key="x" * 40,
                cors_origins=["*"],
                api_keys=[f"{VALID_KEY}:admin"],
            )

    def test_auth_without_keys_is_refused(self):
        """Otherwise the API would lock everyone out silently."""
        with pytest.raises(ValueError, match="API_KEYS"):
            Settings(environment=Environment.LOCAL, auth_enabled=True, api_keys=[])

    def test_docs_are_disabled_in_production(self):
        settings = Settings(
            environment=Environment.PRODUCTION,
            secret_key="x" * 40,
            api_keys=[f"{VALID_KEY}:admin"],
        )
        assert settings.docs_url is None
        assert settings.openapi_url is None

    def test_comma_separated_env_values_are_parsed(self):
        settings = Settings(cors_origins="http://a.test, http://b.test")
        assert settings.cors_origins == ["http://a.test", "http://b.test"]


class TestLocalBlobStore:
    @pytest.fixture
    def store(self, tmp_path):
        return LocalBlobStore(tmp_path / "blobs")

    async def test_put_then_get(self, store):
        await store.put("transcripts/abc", b"hello")
        assert await store.get("transcripts/abc") == b"hello"

    async def test_missing_key_raises_not_found(self, store):
        with pytest.raises(NotFoundError):
            await store.get("transcripts/nope")

    async def test_exists(self, store):
        assert not await store.exists("k1")
        await store.put("k1", b"x")
        assert await store.exists("k1")

    async def test_delete_is_idempotent(self, store):
        await store.put("k1", b"x")
        await store.delete("k1")
        await store.delete("k1")
        assert not await store.exists("k1")

    async def test_overwrite_replaces_content(self, store):
        await store.put("k1", b"first")
        await store.put("k1", b"second")
        assert await store.get("k1") == b"second"

    @pytest.mark.parametrize("key", [
        "../escape", "transcripts/../../etc/passwd", "/absolute", "..",
    ])
    async def test_traversal_keys_are_refused(self, store, key):
        with pytest.raises(StorageError):
            await store.put(key, b"x")

    async def test_health_reports_writability(self, store):
        ok, detail = await store.health()
        assert ok is True
        assert "local:" in detail
