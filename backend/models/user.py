"""
PCDS Enterprise - User Model
Database schema for user management
"""

from typing import Optional, Any
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    """Base user model"""
    username: str = Field(..., min_length=3, max_length=50)
    email: str  # Use str instead of EmailStr for flexibility
    role: str = Field(default="analyst")


class UserCreate(UserBase):
    """User creation model"""
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    """User update model"""
    email: Optional[str] = None
    role: Optional[str] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """User response model (no password)"""
    id: str
    username: str
    email: str
    role: str
    is_active: Any  # DB returns 0/1 or bool
    created_at: Optional[str] = None
    last_login: Optional[str] = None
    
    model_config = {"from_attributes": True}


class LoginRequest(BaseModel):
    """Login request model"""
    email: str
    password: str
    remember_me: bool = False


class TokenResponse(BaseModel):
    """Token response model"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserResponse
