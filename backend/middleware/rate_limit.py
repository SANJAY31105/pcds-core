"""
Rate Limiting Middleware for PCDS Enterprise
Protects against API abuse and brute force attacks
"""

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

# Initialize rate limiter with Redis backend
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200/minute"],  # Global default
    storage_uri="redis://localhost:6379",  # Redis backend for distributed rate limiting
    strategy="fixed-window"
)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom handler for rate limit exceeded errors
    """
    logger.warning(f"Rate limit exceeded for {get_remote_address(request)}: {request.url.path}")
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "detail": str(exc.detail),
            "retry_after": f"{exc.detail} seconds"
        }
    )


# Rate limit configurations for different endpoint types
RATE_LIMITS = {
    "auth_login": "5/minute",      # Max 5 login attempts per minute
    "auth_general": "20/minute",   # Max 20 auth requests per minute
    "api_write": "50/minute",      # Max 50 write operations per minute
    "api_read": "200/minute",      # Max 200 read operations per minute
    "reports": "30/minute",        # Max 30 report generations per minute
}


def get_rate_limit(endpoint_type: str) -> str:
    """
    Get rate limit string for specific endpoint type
    """
    return RATE_LIMITS.get(endpoint_type, "100/minute")
