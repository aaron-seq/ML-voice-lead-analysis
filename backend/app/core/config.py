"""Application configuration.

One settings object, validated at import time, sourced from the environment.
Defaults are chosen so that ``uvicorn app.main:app`` works on a fresh clone
with no environment file, no AWS account and no external services (issue #9).
Every production-only requirement is enforced by a validator rather than left
to a README warning.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    LOCAL = "local"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class StorageBackend(StrEnum):
    LOCAL = "local"
    S3 = "s3"


REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- Identity ---------------------------------------------------------
    app_name: str = "CallSense"
    app_version: str = "1.0.0"
    environment: Environment = Environment.LOCAL

    # -- HTTP -------------------------------------------------------------
    # Loopback by default so `uvicorn app.main:app` on a laptop is not exposed
    # to the whole local network. Containers must bind all interfaces to be
    # reachable, so the Dockerfile passes --host 0.0.0.0 explicitly.
    host: str = "127.0.0.1"
    port: int = 8000
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://127.0.0.1:3000"]
    )
    cors_origin_regex: str | None = Field(
        None,
        description="Regex for dynamic origins, e.g. preview deployments. "
        "Starlette does not support glob patterns in cors_origins, so wildcard "
        "hosts must come through here.",
    )
    max_upload_bytes: int = Field(
        25 * 1024 * 1024, description="Hard cap on a single transcript upload"
    )

    # -- Persistence ------------------------------------------------------
    database_url: str = Field(
        f"sqlite+aiosqlite:///{REPO_ROOT / 'data' / 'callsense.db'}",
        description="Async SQLAlchemy URL. Postgres: postgresql+asyncpg://...",
    )
    db_echo: bool = False

    # -- Blob storage -----------------------------------------------------
    storage_backend: StorageBackend = StorageBackend.LOCAL
    storage_local_path: Path = REPO_ROOT / "data" / "blobs"
    s3_bucket: str | None = None
    s3_prefix: str = "callsense/"
    aws_region: str = "us-east-1"

    # -- Security ---------------------------------------------------------
    #
    # There is deliberately no SECRET_KEY. Authentication is API-key based and
    # keys are compared as SHA-256 digests -- nothing is signed or encrypted,
    # so a signing secret would be a setting that does nothing while forcing
    # operators to generate and rotate it. If signed sessions or JWTs are added
    # later, the secret arrives with them.
    auth_enabled: bool = Field(
        False,
        description="When false the API runs open with an implicit admin "
        "principal. Forced on in staging/production.",
    )
    api_keys: list[str] = Field(
        default_factory=list,
        description="Static API keys in 'key:role' form, e.g. 'sk-abc:analyst'. "
        "Suitable for a single-team deployment; see docs/SECURITY.md for the "
        "path to real identity providers.",
    )
    rate_limit_per_minute: int = Field(
        120, description="Per-principal request budget. 0 disables limiting."
    )

    # -- Analysis engine --------------------------------------------------
    enable_transformer_sentiment: bool = Field(
        False,
        description="Opt in to a transformer sentiment model. Off by default: "
        "the lexicon analyzer needs no model download and is deterministic.",
    )
    transformer_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # -- Observability ----------------------------------------------------
    log_level: str = "INFO"
    log_format: str = Field("console", description="'console' or 'json'")
    metrics_enabled: bool = Field(
        True,
        description="Expose Prometheus metrics at /metrics. Values are "
        "per-worker; see docs/OPERATIONS.md.",
    )

    # ---------------------------------------------------------------------

    @field_validator("cors_origins", "api_keys", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept both JSON arrays and plain comma-separated env values."""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith("["):
                return value  # let pydantic parse the JSON form
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value

    @field_validator("log_format")
    @classmethod
    def _valid_log_format(cls, value: str) -> str:
        if value not in {"console", "json"}:
            raise ValueError("log_format must be 'console' or 'json'")
        return value

    @model_validator(mode="after")
    def _enforce_production_invariants(self) -> Settings:
        if self.environment in (Environment.STAGING, Environment.PRODUCTION):
            problems: list[str] = []
            if "*" in self.cors_origins:
                problems.append("CORS_ORIGINS must not contain '*'")
            if self.storage_backend is StorageBackend.S3 and not self.s3_bucket:
                problems.append("S3_BUCKET is required when STORAGE_BACKEND=s3")
            if problems:
                raise ValueError(
                    f"Invalid configuration for {self.environment.value}: "
                    + "; ".join(problems)
                )
            # Auth is not optional outside local development.
            object.__setattr__(self, "auth_enabled", True)

        if self.auth_enabled and not self.api_keys:
            raise ValueError(
                "AUTH_ENABLED is true but no API_KEYS are configured. "
                "Set API_KEYS='<key>:admin' or generate one with "
                "`python -m app.cli.main keygen`."
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.environment is Environment.PRODUCTION

    @property
    def docs_url(self) -> str | None:
        """Interactive docs are disabled in production."""
        return None if self.is_production else f"{self.api_prefix}/docs"

    @property
    def openapi_url(self) -> str | None:
        return None if self.is_production else f"{self.api_prefix}/openapi.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Call ``get_settings.cache_clear()`` in tests."""
    return Settings()
