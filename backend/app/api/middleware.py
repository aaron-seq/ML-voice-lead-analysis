"""HTTP middleware: request correlation, access logging, security headers."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import principal_var, request_id_var

logger = logging.getLogger("app.access")

# Endpoints that would otherwise dominate the log with no diagnostic value.
_QUIET_PATHS = frozenset({"/health", "/health/live", "/health/ready", "/metrics"})


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, log the outcome, and time the request.

    The id is taken from an inbound ``X-Request-ID`` when present so traces
    stitch together across a proxy, and echoed on the response either way.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request_token = request_id_var.set(request_id)
        principal_token = principal_var.set("-")
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception:
                elapsed = (time.perf_counter() - started) * 1000
                logger.exception(
                    "%s %s failed after %.1fms",
                    request.method, request.url.path, elapsed,
                    extra={"method": request.method, "path": request.url.path,
                           "duration_ms": round(elapsed, 2)},
                )
                raise

            elapsed = (time.perf_counter() - started) * 1000
            response.headers["X-Request-ID"] = request_id
            response.headers["Server-Timing"] = f"app;dur={elapsed:.1f}"

            # ``get_principal`` records the caller on request.state rather than
            # in the ContextVar directly: BaseHTTPMiddleware runs the handler in
            # a child task, and a ContextVar set there is invisible here. Copying
            # it across is what puts a real principal on the access log line --
            # the field was previously hardcoded to "-" on every record.
            principal_var.set(getattr(request.state, "principal", "-"))

            if request.url.path not in _QUIET_PATHS:
                log = logger.warning if response.status_code >= 500 else logger.info
                log(
                    "%s %s -> %d (%.1fms)",
                    request.method, request.url.path, response.status_code, elapsed,
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "duration_ms": round(elapsed, 2),
                    },
                )
            return response
        finally:
            # Reset unconditionally: the previous code left request_id_var set
            # when a request raised.
            principal_var.reset(principal_token)
            request_id_var.reset(request_token)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Baseline security headers (issue #27)."""

    def __init__(self, app, hsts: bool = False) -> None:
        super().__init__(app)
        self._hsts = hsts

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        headers = response.headers

        headers.setdefault("X-Content-Type-Options", "nosniff")
        headers.setdefault("X-Frame-Options", "DENY")
        headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        # This is a JSON API; nothing it returns should ever be rendered or
        # framed. The docs UI needs to load its own assets, so it is exempt.
        if not request.url.path.endswith(("/docs", "/redoc", "/openapi.json")):
            headers.setdefault(
                "Content-Security-Policy",
                "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
            )

        # Only meaningful over TLS, and setting it on plain HTTP in local dev
        # can poison a developer's browser for localhost.
        if self._hsts:
            headers.setdefault(
                "Strict-Transport-Security",
                "max-age=31536000; includeSubDomains",
            )
        return response
