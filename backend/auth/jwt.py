"""
JWT Authentication for PCDS Enterprise
Access and refresh token management
"""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
import logging

logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Short-lived access tokens
REFRESH_TOKEN_EXPIRE_DAYS = 7  # Long-lived refresh tokens

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class JWTManager:
    """JWT token management"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_access_token(data: Dict[str, Any]) -> str:
        """
        Create JWT access token (15 minutes)
        
        Args:
            data: Payload to encode (user_id, role, etc.)
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({
            "exp": expire,
            "type": "access"
        })
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any]) -> str:
        """
        Create JWT refresh token (7 days)
        
        Args:
            data: Payload to encode (user_id only)
            
        Returns:
            JWT token string
        """
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode.update({
            "exp": expire,
            "type": "refresh"
        })
        
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def decode_token(token: str) -> Optional[Dict[str, Any]]:
        """
        Decode and validate JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded payload or None if invalid
        """
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            logger.warning(f"JWT decode error: {e}")
            return None
    
    @staticmethod
    def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify access token"""
        payload = JWTManager.decode_token(token)
        if payload and payload.get("type") == "access":
            return payload
        return None
    
    @staticmethod
    def verify_refresh_token(token: str) -> Optional[Dict[str, Any]]:
        """Verify refresh token"""
        payload = JWTManager.decode_token(token)
        if payload and payload.get("type") == "refresh":
            return payload
        return None
    
    @staticmethod
    def create_token_pair(user_id: str, role: str, additional_data: Dict = None) -> Dict[str, str]:
        """
        Create both access and refresh tokens
        
        Args:
            user_id: User identifier
            role: User role (admin, analyst, viewer)
            additional_data: Additional claims
            
        Returns:
            Dict with access_token and refresh_token
        """
        token_data = {
            "sub": user_id,
            "role": role,
            **(additional_data or {})
        }
        
        return {
            "access_token": JWTManager.create_access_token(token_data),
            "refresh_token": JWTManager.create_refresh_token({"sub": user_id}),
            "token_type": "bearer"
        }


# Global JWT manager instance
jwt_manager = JWTManager()
