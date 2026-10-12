"""Health, readiness and service metadata.

Two distinct probes, because conflating them causes bad deploys:
  * ``/health/live``  -- is the process up? Never touches dependencies.
  * ``/health/ready`` -- can it serve traffic? Checks database and storage.

A container orchestrator restarting a pod because S3 had a blip is exactly the
failure mode this separation prevents.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Response, status
from fastapi.responses import PlainTextResponse

from app.api.deps import BlobStoreDep, SettingsDep
from app.api.schemas import HealthComponent, HealthResponse
from app.core.errors import NotFoundError
from app.core.metrics import registry
from app.db.session import check_health as check_database

router = APIRouter(tags=["System"])

_STARTED_AT = time.monotonic()


def _uptime() -> float:
    return round(time.monotonic() - _STARTED_AT, 2)


@router.get("/health/live", summary="Liveness probe")
async def liveness() -> dict:
    """Process is running. Deliberately checks nothing external."""
    return {"status": "alive", "uptime_seconds": _uptime()}


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
)
async def readiness(
    settings: SettingsDep, blobs: BlobStoreDep, response: Response
) -> HealthResponse:
    """Verify every dependency needed to serve requests.

    Returns 503 when a hard dependency is down so a load balancer stops
    routing to this instance.
    """
    components: dict[str, HealthComponent] = {}

    db_ok, db_detail = await check_database()
    components["database"] = HealthComponent(ok=db_ok, detail=db_detail)

    store_ok, store_detail = await blobs.health()
    components["storage"] = HealthComponent(ok=store_ok, detail=store_detail)

    components["engine"] = HealthComponent(ok=True, detail="lexicon-v1")

    healthy = all(component.ok for component in components.values())
    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="healthy" if healthy else "unhealthy",
        version=settings.app_version,
        environment=settings.environment.value,
        uptime_seconds=_uptime(),
        components=components,
    )


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health(
    settings: SettingsDep, blobs: BlobStoreDep, response: Response
) -> HealthResponse:
    """Alias for the readiness probe, kept for convention."""
    return await readiness(settings, blobs, response)


@router.get(
    "/metrics",
    summary="Prometheus metrics",
    response_class=PlainTextResponse,
    include_in_schema=False,
)
async def metrics(settings: SettingsDep) -> PlainTextResponse:
    """Prometheus text exposition.

    Unauthenticated by design, like the health probes: a scrape endpoint that
    needs a credential is a scrape endpoint that does not get scraped. It
    exposes only counters and latencies -- no transcript content, no customer
    data, no configuration. Restrict it at the ingress if your Prometheus is
    not on a trusted network.

    404s when METRICS_ENABLED=false so the setting genuinely does something.
    """
    if not settings.metrics_enabled:
        raise NotFoundError("Metrics are disabled (METRICS_ENABLED=false)")
    return PlainTextResponse(
        registry.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


@router.get("/", summary="Service metadata", include_in_schema=False)
async def root(settings: SettingsDep) -> dict:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment.value,
        "docs": settings.docs_url,
        "health": "/health/ready",
        "api": settings.api_prefix,
    }
