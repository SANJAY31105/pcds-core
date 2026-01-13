"""
Authentication API Endpoints
JWT Login, Refresh, Logout, User Management
"""

from fastapi import APIRouter, HTTPException, Response, Request, Depends
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import timedelta

from auth.auth import get_auth_manager, UserRole, ACCESS_TOKEN_EXPIRE_MINUTES
from auth.middleware import require_auth, require_admin, get_current_user, User


router = APIRouter(prefix="/auth", tags=["Authentication"])


# Request/Response Models
class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60
    user: dict


class UserResponse(BaseModel):
    user_id: str
    email: str
    name: str
    role: str
    is_active: bool


class CreateUserRequest(BaseModel):
    email: str
    name: str
    password: str
    role: str = "viewer"


class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = ACCESS_TOKEN_EXPIRE_MINUTES * 60


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, response: Response):
    """
    Authenticate user and return tokens.
    
    - Access token returned in JSON body
    - Refresh token set as HttpOnly cookie
    """
    auth_manager = get_auth_manager()
    
    user = auth_manager.authenticate(request.email, request.password)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
    
    # Generate tokens
    access_token = auth_manager.create_access_token(user)
    refresh_token = auth_manager.create_refresh_token(user)
    
    # Set refresh token as HttpOnly cookie
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,  # Set to False for local dev without HTTPS
        samesite="lax",
        max_age=7 * 24 * 60 * 60  # 7 days
    )
    
    return LoginResponse(
        access_token=access_token,
        user={
            "user_id": user.user_id,
            "email": user.email,
            "name": user.name,
            "role": user.role.value
        }
    )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(request: Request, response: Response):
    """
    Refresh access token using refresh token from cookie.
    Implements token rotation for security.
    """
    auth_manager = get_auth_manager()
    
    # Get refresh token from cookie
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(
            status_code=401,
            detail="Refresh token not found"
        )
    
    # Rotate tokens
    result = auth_manager.rotate_tokens(refresh_token)
    if not result:
        # Clear invalid cookie
        response.delete_cookie("refresh_token")
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired refresh token"
        )
    
    new_access_token, new_refresh_token = result
    
    # Set new refresh token cookie
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=7 * 24 * 60 * 60
    )
    
    return RefreshResponse(access_token=new_access_token)


@router.post("/logout")
async def logout(
    request: Request, 
    response: Response,
    user: User = Depends(require_auth)
):
    """
    Logout user - invalidate tokens and clear cookie.
    """
    auth_manager = get_auth_manager()
    
    # Get tokens
    auth_header = request.headers.get("Authorization", "")
    access_token = auth_header.replace("Bearer ", "") if auth_header else None
    refresh_token = request.cookies.get("refresh_token")
    
    # Invalidate tokens
    if access_token:
        auth_manager.logout(access_token, refresh_token)
    
    # Clear cookie
    response.delete_cookie("refresh_token")
    
    return {"message": "Logged out successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(user: User = Depends(require_auth)):
    """
    Get current authenticated user info.
    Used by frontend to check auth status on page load.
    """
    return UserResponse(
        user_id=user.user_id,
        email=user.email,
        name=user.name,
        role=user.role.value,
        is_active=user.is_active
    )


@router.post("/users", response_model=UserResponse)
async def create_user(
    request: CreateUserRequest,
    admin: User = Depends(require_admin)
):
    """
    Create new user (Admin only).
    """
    auth_manager = get_auth_manager()
    
    # Validate role
    try:
        role = UserRole(request.role)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid role. Must be one of: {[r.value for r in UserRole]}"
        )
    
    try:
        user = auth_manager.user_store.create_user(
            email=request.email,
            name=request.name,
            password=request.password,
            role=role
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return UserResponse(
        user_id=user.user_id,
        email=user.email,
        name=user.name,
        role=user.role.value,
        is_active=user.is_active
    )


@router.get("/users", response_model=List[UserResponse])
async def list_users(admin: User = Depends(require_admin)):
    """
    List all users (Admin only).
    """
    auth_manager = get_auth_manager()
    users = auth_manager.user_store.list_users()
    
    return [
        UserResponse(
            user_id=u.user_id,
            email=u.email,
            name=u.name,
            role=u.role.value,
            is_active=u.is_active
        )
        for u in users
    ]


@router.get("/status")
async def auth_status(user: Optional[User] = Depends(get_current_user)):
    """
    Check authentication status (no auth required).
    """
    if user:
        return {
            "authenticated": True,
            "user": {
                "email": user.email,
                "role": user.role.value
            }
        }
    return {"authenticated": False}
