"""
Authentication Package
"""

from auth.auth import (
    get_auth_manager,
    AuthManager,
    User,
    UserRole,
    JWT_SECRET,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

from auth.middleware import (
    get_current_user,
    require_auth,
    require_role,
    require_admin,
    require_analyst,
    require_viewer
)

__all__ = [
    'get_auth_manager',
    'AuthManager',
    'User',
    'UserRole',
    'get_current_user',
    'require_auth',
    'require_role',
    'require_admin',
    'require_analyst',
    'require_viewer'
]
