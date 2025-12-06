"""
Role-Based Access Control (RBAC) for PCDS Enterprise
"""
from enum import Enum
from typing import List, Set
from fastapi import HTTPException, Depends, Header
from auth.jwt import jwt_manager
import logging

logger = logging.getLogger(__name__)


class Role(str, Enum):
    """User roles"""
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Permission(str, Enum):
    """System permissions"""
    # Read permissions
    READ_DETECTIONS = "read:detections"
    READ_ENTITIES = "read:entities"
    READ_REPORTS = "read:reports"
    READ_INVESTIGATIONS = "read:investigations"
    
    # Write permissions
    WRITE_DETECTIONS = "write:detections"
    WRITE_ENTITIES = "write:entities"
    WRITE_INVESTIGATIONS = "write:investigations"
    
    # Admin permissions
    MANAGE_USERS = "manage:users"
    MANAGE_SYSTEM = "manage:system"
    DELETE_DATA = "delete:data"


# Role → Permissions mapping
ROLE_PERMISSIONS: dict[Role, Set[Permission]] = {
    Role.ADMIN: {
        # Admins have all permissions
        Permission.READ_DETECTIONS,
        Permission.READ_ENTITIES,
        Permission.READ_REPORTS,
        Permission.READ_INVESTIGATIONS,
        Permission.WRITE_DETECTIONS,
        Permission.WRITE_ENTITIES,
        Permission.WRITE_INVESTIGATIONS,
        Permission.MANAGE_USERS,
        Permission.MANAGE_SYSTEM,
        Permission.DELETE_DATA,
    },
    Role.ANALYST: {
        # Analysts can read and write, but not manage
        Permission.READ_DETECTIONS,
        Permission.READ_ENTITIES,
        Permission.READ_REPORTS,
        Permission.READ_INVESTIGATIONS,
        Permission.WRITE_DETECTIONS,
        Permission.WRITE_ENTITIES,
        Permission.WRITE_INVESTIGATIONS,
    },
    Role.VIEWER: {
        # Viewers can only read
        Permission.READ_DETECTIONS,
        Permission.READ_ENTITIES,
        Permission.READ_REPORTS,
        Permission.READ_INVESTIGATIONS,
    },
}


class RBACManager:
    """Role-Based Access Control manager"""
    
    @staticmethod
    def has_permission(role: Role, permission: Permission) -> bool:
        """Check if role has permission"""
        return permission in ROLE_PERMISSIONS.get(role, set())
    
    @staticmethod
    def get_permissions(role: Role) -> Set[Permission]:
        """Get all permissions for role"""
        return ROLE_PERMISSIONS.get(role, set())
    
    @staticmethod
    def check_permission(role: Role, permission: Permission):
        """
        Check permission and raise HTTPException if denied
        
        Raises:
            HTTPException: 403 if permission denied
        """
        if not RBACManager.has_permission(role, permission):
            raise HTTPException(
                status_code=403,
                detail=f"Permission denied. Required: {permission}"
            )


# Dependency for FastAPI routes
async def get_current_user(authorization: str = Header(None)):
    """
    Extract and validate current user from JWT token
    
    Args:
        authorization: Bearer token from Authorization header
        
    Returns:
        Dict with user_id and role
        
    Raises:
        HTTPException: 401 if token invalid
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid authorization header"
        )
    
    token = authorization.replace("Bearer ", "")
    payload = jwt_manager.verify_access_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )
    
    return {
        "user_id": payload.get("sub"),
        "role": Role(payload.get("role", "viewer"))
    }


def require_permission(permission: Permission):
    """
    Decorator factory for requiring specific permission
    
    Usage:
        @router.delete("/detections/{id}")
        @require_permission(Permission.DELETE_DATA)
        async def delete_detection(id: int, user = Depends(get_current_user)):
            ...
    """
    async def permission_checker(user: dict = Depends(get_current_user)):
        role = user.get("role")
        RBACManager.check_permission(role, permission)
        return user
    
    return permission_checker


def require_role(required_role: Role):
    """
    Decorator factory for requiring specific role
    
    Usage:
        @router.post("/users")
        @require_role(Role.ADMIN)
        async def create_user(user = Depends(get_current_user)):
            ...
    """
    async def role_checker(user: dict = Depends(get_current_user)):
        role = user.get("role")
        if role != required_role:
            raise HTTPException(
                status_code=403,
                detail=f"Role '{required_role}' required"
            )
        return user
    
    return role_checker


# Global RBAC manager instance
rbac_manager = RBACManager()
