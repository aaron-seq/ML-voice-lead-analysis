"""Application error taxonomy.

Domain and service code raises these; a single handler in ``app.main`` turns
them into RFC-7807-ish JSON. Deliberately no FastAPI import here so the engine
and CLI can raise them without depending on the web framework.
"""

from __future__ import annotations

from typing import Any


class CallSenseError(Exception):
    """Base class. Carries the HTTP status the API should surface."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.context:
            payload["context"] = self.context
        return payload


class ValidationError(CallSenseError):
    status_code = 400
    code = "validation_error"


class UnsupportedFormatError(ValidationError):
    code = "unsupported_format"


class EmptyTranscriptError(ValidationError):
    code = "empty_transcript"


class NotFoundError(CallSenseError):
    status_code = 404
    code = "not_found"


class ConflictError(CallSenseError):
    status_code = 409
    code = "conflict"


class AuthenticationError(CallSenseError):
    status_code = 401
    code = "unauthenticated"


class AuthorizationError(CallSenseError):
    status_code = 403
    code = "forbidden"


class RateLimitError(CallSenseError):
    status_code = 429
    code = "rate_limited"


class StorageError(CallSenseError):
    status_code = 502
    code = "storage_error"


class AnalysisError(CallSenseError):
    status_code = 500
    code = "analysis_failed"
