from fastapi import APIRouter, Depends, HTTPException, status, Response, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import timedelta
import sqlite3
import uuid
from datetime import datetime

from .schemas import UserLogin, UserRegister, Token, UserResponse
from .utils import verify_password, get_password_hash, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
from config.database import get_db_connection
from middleware.auth import get_current_user, require_role
from middleware.rate_limit import limiter, get_rate_limit

router = APIRouter(prefix="/api/auth", tags=["Authentication"])
security = HTTPBearer()


@router.post("/login", response_model=Token)
@limiter.limit(get_rate_limit("auth_login"))
async def login(request: Request, response: Response, credentials: UserLogin):
    """
    Login endpoint with rate limiting (5 attempts/minute)
    Sets httpOnly cookie for secure token storage
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM users WHERE email = ?", (credentials.email,))
    user = cursor.fetchone()
    conn.close()
    
    if not user or not verify_password(credentials.password, user[3]):  # password at index 3
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    if not user[7]:  # is_active at index 7
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive"
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user[1],  # email
            "user_id": user[0],  # id
            "role": user[4],  # role
            "tenant_id": user[5]  # tenant_id
        },
        expires_delta=access_token_expires
    )
    
    # Set httpOnly cookie for security (prevents XSS attacks)
    response.set_cookie(
        key="access_token",
        value=f"Bearer {access_token}",
        httponly=True,  # Prevents JavaScript access
        secure=False,   # Set to True in production with HTTPS
        samesite="lax", # CSRF protection
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=UserResponse)
@limiter.limit(get_rate_limit("auth_general"))
async def register(request: Request, user_data: UserRegister, current_user: dict = Depends(require_role(["super_admin", "tenant_admin"]))):
    """
    User registration endpoint with rate limiting (20/minute)
    Requires admin privileges
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT id FROM users WHERE email = ?", (user_data.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)
    
    # If tenant_admin, force their tenant_id
    tenant_id = user_data.tenant_id
    if current_user["role"] == "tenant_admin":
        tenant_id = current_user["tenant_id"]
    
    cursor.execute("""
        INSERT INTO users (id, email, full_name, password_hash, role, tenant_id, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, user_data.email, user_data.full_name, hashed_password, user_data.role, tenant_id, True, datetime.utcnow().isoformat()))
    
    conn.commit()
    
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    return {
        "id": user[0],
        "email": user[1],
        "full_name": user[2],
        "role": user[4],
        "tenant_id": user[5],
        "is_active": user[7],
        "created_at": user[8]
    }


@router.get("/me", response_model=UserResponse)
@limiter.limit(get_rate_limit("auth_general"))
async def get_current_user_info(request: Request, current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if user exists
    cursor.execute("SELECT id FROM users WHERE email = ?", (user_data.email,))
    if cursor.fetchone():
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Create user
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)
    
    # If tenant_admin, force their tenant_id
    tenant_id = user_data.tenant_id
    if current_user["role"] == "tenant_admin":
        tenant_id = current_user["tenant_id"]
    
    cursor.execute("""
        INSERT INTO users (id, email, full_name, password_hash, role, tenant_id, is_active, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, user_data.email, user_data.full_name, hashed_password, user_data.role, tenant_id, True, datetime.utcnow().isoformat()))
    
    conn.commit()
    
    cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    user = cursor.fetchone()
    conn.close()
    
    return {
        "id": user[0],
        "email": user[1],
        "full_name": user[2],
        "role": user[4],
        "tenant_id": user[5],
        "is_active": user[7],
        "created_at": user[8]
    }


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    return {
        "id": current_user["user_id"],
        "email": current_user["email"],
        "full_name": current_user.get("full_name", ""),
        "role": current_user["role"],
        "tenant_id": current_user.get("tenant_id"),
        "is_active": True,
        "created_at": current_user.get("created_at", "")
    }
