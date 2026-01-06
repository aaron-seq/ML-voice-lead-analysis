"""FastAPI Middleware for Request Logging and Error Handling.

Provides:
- Request/response logging with correlation IDs
- Performance monitoring
- Error handling and formatting
- Request context tracking
"""

import time
import uuid
import logging
from typing import Callable
from datetime import datetime

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from .exceptions import APIException

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging all requests and responses."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra={
                'request_id': request_id,
                'method': request.method,
                'path': request.url.path,
                'client': request.client.host if request.client else None,
                'user_agent': request.headers.get('user-agent')
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
        except Exception as e:
            # Log exception
            logger.exception(
                f"Unhandled exception during request processing",
                extra={'request_id': request_id}
            )
            raise
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Add custom headers
        response.headers['X-Request-ID'] = request_id
        response.headers['X-Process-Time'] = f"{duration_ms:.2f}ms"
        
        # Log response
        logger.info(
            f"Request completed: {response.status_code}",
            extra={
                'request_id': request_id,
                'status_code': response.status_code,
                'duration_ms': round(duration_ms, 2),
                'path': request.url.path,
                'method': request.method
            }
        )
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Middleware for standardized error handling."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Handle errors and format responses."""
        try:
            response = await call_next(request)
            return response
            
        except APIException as e:
            # Handle custom API exceptions
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.error(
                f"API Exception: {e.error_code} - {e.message}",
                extra={
                    'request_id': request_id,
                    'error_code': e.error_code,
                    'status_code': e.status_code,
                    'path': request.url.path,
                    'method': request.method,
                    'details': e.details
                }
            )
            
            error_response = {
                'success': False,
                'error': {
                    'code': e.error_code,
                    'message': e.message,
                    'details': e.details,
                    'request_id': request_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'path': request.url.path
                }
            }
            
            return JSONResponse(
                status_code=e.status_code,
                content=error_response
            )
            
        except Exception as e:
            # Handle unexpected exceptions
            request_id = getattr(request.state, 'request_id', 'unknown')
            
            logger.exception(
                f"Unhandled exception: {str(e)}",
                extra={
                    'request_id': request_id,
                    'path': request.url.path,
                    'method': request.method
                }
            )
            
            error_response = {
                'success': False,
                'error': {
                    'code': 'INTERNAL_SERVER_ERROR',
                    'message': 'An unexpected error occurred. Please try again later.',
                    'request_id': request_id,
                    'timestamp': datetime.utcnow().isoformat(),
                    'path': request.url.path
                }
            }
            
            return JSONResponse(
                status_code=500,
                content=error_response
            )
