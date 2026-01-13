"""
Authentication Middleware for FastAPI
Provides token verification and role-based access control
"""

from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List
from functools import wraps

from auth.auth import get_auth_manager, User, UserRole


# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Optional[User]:
    """
    Dependency to get current user from JWT token.
    Returns None if no valid token provided.
    """
    if not credentials:
        return None
    
    auth_manager = get_auth_manager()
    user = auth_manager.get_user_from_token(credentials.credentials)
    return user


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Dependency that requires valid authentication.
    Raises 401 if not authenticated.
    """
    if not credentials:
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    auth_manager = get_auth_manager()
    user = auth_manager.get_user_from_token(credentials.credentials)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=401,
            detail="User account is disabled"
        )
    
    return user


def require_role(allowed_roles: List[UserRole]):
    """
    Dependency factory for role-based access control.
    
    Usage:
        @router.get("/admin-only")
        async def admin_endpoint(user: User = Depends(require_role([UserRole.ADMIN]))):
            ...
    """
    async def role_checker(user: User = Depends(require_auth)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied. Required role: {[r.value for r in allowed_roles]}"
            )
        return user
    return role_checker


# Pre-defined role dependencies for convenience
require_admin = require_role([UserRole.ADMIN])
require_analyst = require_role([UserRole.ADMIN, UserRole.ANALYST])
require_viewer = require_role([UserRole.ADMIN, UserRole.ANALYST, UserRole.VIEWER])


async def get_token_from_cookie(request: Request) -> Optional[str]:
    """Extract refresh token from HttpOnly cookie"""
    return request.cookies.get("refresh_token")


class RoleChecker:
    """
    Class-based role checker for more complex scenarios
    """
    
    def __init__(self, allowed_roles: List[UserRole]):
        self.allowed_roles = allowed_roles
    
    async def __call__(self, user: User = Depends(require_auth)) -> User:
        if user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient permissions. Required: {[r.value for r in self.allowed_roles]}"
            )
        return user
