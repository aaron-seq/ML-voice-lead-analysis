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

from app.api.deps import BlobStoreDep, SettingsDep
from app.api.schemas import HealthComponent, HealthResponse
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
