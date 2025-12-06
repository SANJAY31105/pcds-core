from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List
from auth.utils import decode_access_token
from config.database import get_db_connection

security = HTTPBearer()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_access_token(token)
    
    if payload is None:
        print("❌ Auth Middleware: Token decode failed (payload is None)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )
    
    email = payload.get("sub")
    user_id = payload.get("user_id")
    role = payload.get("role")
    tenant_id = payload.get("tenant_id")
    
    print(f"✅ Auth Middleware: Decoded token for user {email} ({user_id})")

    if email is None or user_id is None:
        print("❌ Auth Middleware: Missing email or user_id in payload")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    # Fetch full user info
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return {
        "user_id": user[0],
        "email": user[1],
        "full_name": user[2],
        "role": user[4],
        "tenant_id": user[5],
        "is_active": user[7],
        "created_at": user[8]
    }


def require_role(allowed_roles: List[str]):
    async def role_checker(current_user: dict = Depends(get_current_user)):
        if current_user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required roles: {allowed_roles}"
            )
        return current_user
    return role_checker


def get_tenant_filter(current_user: dict):
    """Returns SQL filter for tenant isolation"""
    if current_user["role"] == "super_admin":
        return "", []  # No filter, see all
    else:
        return "AND tenant_id = ?", [current_user["tenant_id"]]
