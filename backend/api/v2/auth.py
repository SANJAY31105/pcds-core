"""
PCDS Enterprise - Authentication API
User registration, login, and token management
"""

from fastapi import APIRouter, HTTPException, Depends, Header
from typing import Optional
import uuid
from datetime import datetime

from models.user import UserCreate, UserResponse, LoginRequest, TokenResponse, UserUpdate
from auth.jwt import jwt_manager
from auth.rbac import require_role
from auth.password_hasher import hash_password, verify_password
from config.database import db_manager
from cache.memory_cache import memory_cache

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register_user(user_data: UserCreate, authorization: Optional[str] = Header(None)):
    """
    Register a new user (ADMIN ONLY)
    
    Creates user account with hashed password and returns JWT tokens
    """
    # SECURITY: Only admins can create users
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=403, detail="Admin access required for user registration")
    
    token = authorization.split(" ")[1]
    payload = jwt_manager.verify_access_token(token)
    
    if not payload or payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required for user registration")
    # Check if username exists
    existing_user = db_manager.execute_one("""
        SELECT id FROM users WHERE username = ?
    """, (user_data.username,))
    
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already exists")
    
    # Check if email exists
    existing_email = db_manager.execute_one("""
        SELECT id FROM users WHERE email = ?
    """, (user_data.email,))
    
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Hash password
    password_hash = jwt_manager.hash_password(user_data.password)
    
    # Create user
    user_id = str(uuid.uuid4())
    db_manager.execute_update("""
        INSERT INTO users (id, username, email, password_hash, role, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, user_data.username, user_data.email, password_hash, user_data.role, True, datetime.utcnow().isoformat()))
    
    # Get created user
    user = db_manager.execute_one("""
        SELECT id, username, email, role, is_active, created_at, last_login
        FROM users WHERE id = ?
    """, (user_id,))
    
    # Generate tokens
    tokens = jwt_manager.create_token_pair(user_id, user['role'])
    
    return {
        **tokens,
        "user": UserResponse(**user)
    }


@router.post("/login", response_model=TokenResponse)
async def login(credentials: LoginRequest):
    """
    User login
    
    Validates credentials and returns JWT tokens
    """
    # Get user by username
    user = db_manager.execute_one("""
        SELECT id, username, email, password_hash, role, is_active, created_at, last_login
        FROM users WHERE username = ?
    """, (credentials.username,))
    
    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password using secure PBKDF2 hasher
    password_valid = verify_password(credentials.password, user['password_hash'])
    
    if not password_valid:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Check if active
    if not user['is_active']:
        raise HTTPException(status_code=403, detail="Account is  disabled")
    
    # Update last login
    now = datetime.utcnow().isoformat()
    db_manager.execute_update("""
        UPDATE users SET last_login = ? WHERE id = ?
    """, (now, user['id']))
    
    user['last_login'] = now
    
    # Generate tokens
    tokens = jwt_manager.create_token_pair(user['id'], user['role'])
    
    # Remove password hash from response
    user_data = {k: v for k, v in user.items() if k != 'password_hash'}
    
    return {
        **tokens,
        "user": UserResponse(**user_data)
    }


@router.post("/refresh", response_model=dict)
async def refresh_token(authorization: Optional[str] = Header(None)):
    """
    Refresh access token using refresh token
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")
    
    refresh_token = authorization.split(" ")[1]
    
    # Verify refresh token
    payload = jwt_manager.verify_refresh_token(refresh_token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    user_id = payload.get("sub")
    
    # Get user role
    user = db_manager.execute_one("""
        SELECT role FROM users WHERE id = ? AND is_active = 1
    """, (user_id,))
    
    if not user:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    # Create new access token
    access_token = jwt_manager.create_access_token({
        "sub": user_id,
        "role": user['role']
    })
    
    return {
        "access_token": access_token,
        "token_type": "bearer"
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user(authorization: Optional[str] = Header(None)):
    """
    Get current authenticated user
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = authorization.split(" ")[1]
    
    # Verify token
    payload = jwt_manager.verify_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = payload.get("sub")
    
    # Get user
    user = db_manager.execute_one("""
        SELECT id, username, email, role, is_active, created_at, last_login
        FROM users WHERE id = ?
    """, (user_id,))
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return UserResponse(**user)


@router.get("/users", response_model=list[UserResponse])
@memory_cache.cache_function(ttl=30)  # Cache for 30 seconds
async def list_users(authorization: Optional[str] = Header(None)):
    """
    List all users (admin only)
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = authorization.split(" ")[1]
    payload = jwt_manager.verify_access_token(token)
    
    if not payload or payload.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get all users
    users = db_manager.execute_query("""
        SELECT id, username, email, role, is_active, created_at, last_login
        FROM users
        ORDER BY created_at DESC
    """)
    
    return [UserResponse(**user) for user in users]


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: str,
    update_data: UserUpdate,
    authorization: Optional[str] = Header(None)
):
    """
    Update user (admin only or own profile)
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = authorization.split(" ")[1]
    payload = jwt_manager.verify_access_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    current_user_id = payload.get("sub")
    is_admin = payload.get("role") == "admin"
    
    # Check permissions
    if user_id != current_user_id and not is_admin:
        raise HTTPException(status_code=403, detail="Permission denied")
    
    # Build update query
    updates = []
    params = []
    
    if update_data.email:
        updates.append("email = ?")
        params.append(update_data.email)
    
    if update_data.role and is_admin:  # Only admin can change roles
        updates.append("role = ?")
        params.append(update_data.role)
    
    if update_data.is_active is not None and is_admin:
        updates.append("is_active = ?")
        params.append(update_data.is_active)
    
    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")
    
    params.append(user_id)
    query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
    
    db_manager.execute_update(query, tuple(params))
    
    # Get updated user
    user = db_manager.execute_one("""
        SELECT id, username, email, role, is_active, created_at, last_login
        FROM users WHERE id = ?
    """, (user_id,))
    
    return UserResponse(**user)


@router.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    """
    Logout user (invalidate token)
    
    Note: With JWT, logout is handled client-side by removing token.
    This endpoint is for compatibility and future session tracking.
    """
    return {"message": "Logged out successfully"}
