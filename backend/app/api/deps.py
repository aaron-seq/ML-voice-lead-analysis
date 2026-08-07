"""FastAPI dependencies: settings, auth, repositories, engine."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Header, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.analysis.engine import AnalysisEngine
from app.core.config import Settings, get_settings
from app.core.logging import principal_var
from app.core.security import ANONYMOUS_ADMIN, ApiKeyRegistry, Principal, Role
from app.db.repository import AuditRepository, CallRepository
from app.db.session import get_session


def get_app_settings(request: Request) -> Settings:
    """Resolve the settings this app instance was built with.

    Depending on ``get_settings`` directly would read the process-wide cached
    object instead, so anything passed to ``create_app(settings)`` would be
    silently ignored by every route that reads a setting -- which made
    per-instance configuration (and the tests that rely on it) a no-op.
    """
    settings: Settings | None = getattr(request.app.state, "settings", None)
    return settings if settings is not None else get_settings()


SettingsDep = Annotated[Settings, Depends(get_app_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_principal(
    request: Request,
    settings: SettingsDep,
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> Principal:
    """Resolve the caller.

    With ``AUTH_ENABLED=false`` (local default) every caller is an admin, which
    keeps a fresh clone usable without ceremony. Config forbids that in
    staging and production.
    """
    if not settings.auth_enabled:
        return _record(request, ANONYMOUS_ADMIN)

    registry: ApiKeyRegistry = request.app.state.api_keys
    principal = registry.resolve(x_api_key)

    limiter = request.app.state.rate_limiter
    limiter.check(principal.id)

    return _record(request, principal)


def _record(request: Request, principal: Principal) -> Principal:
    """Publish the caller to the logging context.

    Both are needed. The ContextVar is what puts the principal on log lines
    emitted inside the handler; request.state is how the access-log middleware
    reads it back, since it runs in a parent task that cannot see a ContextVar
    set here.
    """
    principal_var.set(principal.id)
    request.state.principal = principal.id
    return principal


PrincipalDep = Annotated[Principal, Depends(get_principal)]


def require_role(minimum: Role):
    """Build a dependency enforcing a minimum role."""

    async def _check(principal: PrincipalDep) -> Principal:
        principal.require(minimum)
        return principal

    return _check


RequireViewer = Annotated[Principal, Depends(require_role(Role.VIEWER))]
RequireAnalyst = Annotated[Principal, Depends(require_role(Role.ANALYST))]
RequireAdmin = Annotated[Principal, Depends(require_role(Role.ADMIN))]


async def get_call_repository(session: SessionDep) -> CallRepository:
    return CallRepository(session)


async def get_audit_repository(session: SessionDep) -> AuditRepository:
    return AuditRepository(session)


def get_analysis_engine(request: Request) -> AnalysisEngine:
    """The engine built once at startup; reused across requests."""
    return request.app.state.engine


def get_blob_store(request: Request):
    return request.app.state.blob_store


CallRepoDep = Annotated[CallRepository, Depends(get_call_repository)]
AuditRepoDep = Annotated[AuditRepository, Depends(get_audit_repository)]
EngineDep = Annotated[AnalysisEngine, Depends(get_analysis_engine)]
BlobStoreDep = Annotated[object, Depends(get_blob_store)]
