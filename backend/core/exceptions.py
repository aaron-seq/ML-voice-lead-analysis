"""Custom Exception Classes for ML Voice Lead Analysis.

Provides domain-specific exceptions with:
- Standardized error codes
- HTTP status code mapping
- Detailed error messages
- Request context tracking
"""

from typing import Optional, Dict, Any
from fastapi import status


class APIException(Exception):
    """Base exception for all API errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "API_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for JSON response."""
        return {
            'error': {
                'code': self.error_code,
                'message': self.message,
                'status': self.status_code,
                'details': self.details
            }
        }


class AudioProcessingError(APIException):
    """Raised when audio file processing fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="AUDIO_PROCESSING_ERROR",
            status_code=status.HTTP_400_BAD_REQUEST,
            details=details
        )


class InvalidAudioFormatError(APIException):
    """Raised when audio file format is not supported."""
    
    def __init__(self, format_type: str, supported_formats: list):
        message = (
            f"Invalid audio format '{format_type}'. "
            f"Supported formats: {', '.join(supported_formats)}"
        )
        super().__init__(
            message=message,
            error_code="INVALID_AUDIO_FORMAT",
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            details={
                'received_format': format_type,
                'supported_formats': supported_formats
            }
        )


class ModelNotFoundError(APIException):
    """Raised when ML model cannot be found or loaded."""
    
    def __init__(self, model_name: str = "voice_classifier"):
        message = (
            f"ML model '{model_name}' not found. "
            "Please contact support or train models first."
        )
        super().__init__(
            message=message,
            error_code="MODEL_NOT_FOUND",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={'model_name': model_name}
        )


class ModelLoadError(APIException):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, reason: str):
        message = f"Failed to load model '{model_name}': {reason}"
        super().__init__(
            message=message,
            error_code="MODEL_LOAD_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={'model_name': model_name, 'reason': reason}
        )


class DatabaseError(APIException):
    """Raised when database operations fail."""
    
    def __init__(self, operation: str, reason: str):
        message = f"Database {operation} failed: {reason}"
        super().__init__(
            message=message,
            error_code="DATABASE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={'operation': operation, 'reason': reason}
        )


class CacheError(APIException):
    """Raised when cache operations fail."""
    
    def __init__(self, operation: str, reason: str):
        message = f"Cache {operation} failed: {reason}"
        super().__init__(
            message=message,
            error_code="CACHE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={'operation': operation, 'reason': reason}
        )


class StorageError(APIException):
    """Raised when storage operations fail."""
    
    def __init__(self, operation: str, storage_type: str, reason: str):
        message = f"{storage_type} storage {operation} failed: {reason}"
        super().__init__(
            message=message,
            error_code="STORAGE_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            details={
                'operation': operation,
                'storage_type': storage_type,
                'reason': reason
            }
        )


class ValidationError(APIException):
    """Raised when input validation fails."""
    
    def __init__(self, field: str, message: str):
        super().__init__(
            message=f"Validation error for '{field}': {message}",
            error_code="VALIDATION_ERROR",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            details={'field': field, 'validation_message': message}
        )


class ResourceNotFoundError(APIException):
    """Raised when requested resource is not found."""
    
    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} '{resource_id}' not found"
        super().__init__(
            message=message,
            error_code="RESOURCE_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
            details={'resource_type': resource_type, 'resource_id': resource_id}
        )


class AuthenticationError(APIException):
    """Raised when authentication fails."""
    
    def __init__(self, reason: str = "Invalid credentials"):
        super().__init__(
            message=reason,
            error_code="AUTHENTICATION_ERROR",
            status_code=status.HTTP_401_UNAUTHORIZED,
            details={'reason': reason}
        )


class AuthorizationError(APIException):
    """Raised when user lacks permission for action."""
    
    def __init__(self, action: str, resource: str):
        message = f"Not authorized to {action} {resource}"
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=status.HTTP_403_FORBIDDEN,
            details={'action': action, 'resource': resource}
        )


class RateLimitError(APIException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, limit: int, window_seconds: int, retry_after: int):
        message = (
            f"Rate limit exceeded: {limit} requests per {window_seconds} seconds. "
            f"Retry after {retry_after} seconds."
        )
        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            details={
                'limit': limit,
                'window_seconds': window_seconds,
                'retry_after': retry_after
            }
        )


class ExternalServiceError(APIException):
    """Raised when external service (AWS, OpenAI, etc.) fails."""
    
    def __init__(self, service_name: str, operation: str, reason: str):
        message = f"{service_name} {operation} failed: {reason}"
        super().__init__(
            message=message,
            error_code="EXTERNAL_SERVICE_ERROR",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            details={
                'service': service_name,
                'operation': operation,
                'reason': reason
            }
        )
