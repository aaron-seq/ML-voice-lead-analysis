"""CallSense API application factory.

Run locally::

    uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.analysis.engine import AnalysisEngine
from app.api.middleware import RequestContextMiddleware, SecurityHeadersMiddleware
from app.api.routers import analytics, calls, system
from app.core.config import Environment, Settings, get_settings
from app.core.errors import CallSenseError
from app.core.logging import configure_logging
from app.core.security import ApiKeyRegistry, SlidingWindowRateLimiter
from app.db.session import create_schema, dispose_engine, init_engine
from app.storage.base import build_store

logger = logging.getLogger(__name__)

DESCRIPTION = """
Call intelligence for sales teams: turn recorded sales conversations into
qualified leads, explained scores, and a prioritised follow-up list.

**Every score is explainable.** Each lead score exposes its weighted components
and links back to the exact transcript quotes that produced it.

Authenticate with an `X-API-Key` header when `AUTH_ENABLED=true`.
"""


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. Accepts injected settings for tests."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "Starting %s v%s (%s)",
            settings.app_name, settings.app_version, settings.environment.value,
        )

        init_engine(settings.database_url, echo=settings.db_echo)
        await create_schema()

        app.state.settings = settings
        app.state.blob_store = build_store(settings)
        app.state.engine = AnalysisEngine.from_settings(settings)
        app.state.api_keys = ApiKeyRegistry(settings.api_keys)
        app.state.rate_limiter = SlidingWindowRateLimiter(
            settings.rate_limit_per_minute
        )

        if not settings.auth_enabled:
            logger.warning(
                "Authentication is DISABLED. Every caller is treated as an "
                "admin. Never run this configuration on a reachable network."
            )

        ok, detail = await app.state.blob_store.health()
        if not ok:
            # Not fatal: the API can still serve reads, and failing to boot
            # would turn a storage blip into an outage.
            logger.error("Blob storage is not healthy at startup: %s", detail)

        logger.info("Startup complete")
        yield

        await dispose_engine()
        logger.info("Shutdown complete")

    app = FastAPI(
        title=settings.app_name,
        description=DESCRIPTION,
        version=settings.app_version,
        lifespan=lifespan,
        docs_url=settings.docs_url,
        redoc_url=None if settings.is_production else f"{settings.api_prefix}/redoc",
        openapi_url=settings.openapi_url,
    )

    # Set eagerly as well as in the lifespan: dependencies resolve settings from
    # app.state, and that must work even for a sub-application mounted without
    # its own lifespan running.
    app.state.settings = settings

    _register_middleware(app, settings)
    _register_exception_handlers(app)

    app.include_router(system.router)
    app.include_router(calls.router, prefix=settings.api_prefix)
    app.include_router(analytics.router, prefix=settings.api_prefix)

    return app


def _register_middleware(app: FastAPI, settings: Settings) -> None:
    # Order matters: Starlette applies these outermost-first, so the request
    # context is established before anything else can log.
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_credentials=False,  # auth is a header, not a cookie
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        expose_headers=["X-Request-ID", "Server-Timing"],
        max_age=600,
    )
    app.add_middleware(
        SecurityHeadersMiddleware,
        hsts=settings.environment in (Environment.STAGING, Environment.PRODUCTION),
    )
    app.add_middleware(RequestContextMiddleware)


def _register_exception_handlers(app: FastAPI) -> None:
    def _request_id(request: Request) -> str | None:
        return getattr(request.state, "request_id", None)

    @app.exception_handler(CallSenseError)
    async def handle_domain_error(request: Request, exc: CallSenseError):
        payload = exc.to_dict()
        payload["request_id"] = _request_id(request)

        if exc.status_code >= 500:
            logger.exception("Unhandled domain error: %s", exc.message)

        headers = {}
        retry_after = exc.context.get("retry_after_seconds")
        if retry_after:
            headers["Retry-After"] = str(retry_after)

        return JSONResponse(
            content=payload, status_code=exc.status_code, headers=headers
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "code": "invalid_request",
                "message": "Request validation failed",
                "context": {
                    "errors": [
                        {
                            "field": ".".join(str(p) for p in err["loc"][1:]) or "body",
                            "message": err["msg"],
                        }
                        for err in exc.errors()[:20]
                    ]
                },
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "code": f"http_{exc.status_code}",
                "message": str(exc.detail),
                "context": {},
                "request_id": _request_id(request),
            },
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        # Never leak an internal exception message to a client.
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "code": "internal_error",
                "message": "An unexpected error occurred.",
                "context": {},
                "request_id": _request_id(request),
            },
        )


app = create_app()
