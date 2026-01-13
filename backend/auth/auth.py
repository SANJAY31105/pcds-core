"""
Enterprise Authentication Module
JWT Access/Refresh Tokens + RBAC

Features:
- JWT tokens from environment variables
- Refresh tokens as HttpOnly cookies
- Token rotation on refresh
- Role-based access control (admin, analyst, viewer)
- Password hashing with bcrypt
"""

import os
import jwt
import bcrypt
from datetime import datetime, timedelta
from typing import Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import uuid


# Configuration from environment
JWT_SECRET = os.getenv("JWT_SECRET", "pcds-jwt-secret-change-in-production")
JWT_REFRESH_SECRET = os.getenv("JWT_REFRESH_SECRET", "pcds-refresh-secret-change-in-production")
ACCESS_TOKEN_EXPIRE_MINUTES = 15
REFRESH_TOKEN_EXPIRE_DAYS = 7


class UserRole(Enum):
    ADMIN = "admin"
    ANALYST = "analyst"
    VIEWER = "viewer"


@dataclass
class User:
    user_id: str
    email: str
    name: str
    role: UserRole
    password_hash: str
    created_at: str
    is_active: bool = True


class TokenBlacklist:
    """Track invalidated tokens for logout"""
    
    def __init__(self):
        self._blacklist: set = set()
    
    def add(self, token_id: str):
        self._blacklist.add(token_id)
    
    def is_blacklisted(self, token_id: str) -> bool:
        return token_id in self._blacklist
    
    def cleanup_expired(self):
        # In production, use Redis with TTL
        pass


class UserStore:
    """In-memory user store (use database in production)"""
    
    def __init__(self):
        self._users: Dict[str, User] = {}
        self._email_index: Dict[str, str] = {}
        self._create_default_admin()
    
    def _create_default_admin(self):
        """Create default admin user"""
        admin = User(
            user_id=str(uuid.uuid4()),
            email="admin@pcds.local",
            name="Administrator",
            role=UserRole.ADMIN,
            password_hash=self._hash_password("admin123"),
            created_at=datetime.utcnow().isoformat()
        )
        self._users[admin.user_id] = admin
        self._email_index[admin.email] = admin.user_id
        
        # Create demo analyst
        analyst = User(
            user_id=str(uuid.uuid4()),
            email="analyst@pcds.local",
            name="Security Analyst",
            role=UserRole.ANALYST,
            password_hash=self._hash_password("analyst123"),
            created_at=datetime.utcnow().isoformat()
        )
        self._users[analyst.user_id] = analyst
        self._email_index[analyst.email] = analyst.user_id
        
        print(f"👤 Default users created: admin@pcds.local, analyst@pcds.local")
    
    def _hash_password(self, password: str) -> str:
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    
    def get_by_email(self, email: str) -> Optional[User]:
        user_id = self._email_index.get(email.lower())
        if user_id:
            return self._users.get(user_id)
        return None
    
    def get_by_id(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)
    
    def create_user(self, email: str, name: str, password: str, role: UserRole) -> User:
        if email.lower() in self._email_index:
            raise ValueError("Email already exists")
        
        user = User(
            user_id=str(uuid.uuid4()),
            email=email.lower(),
            name=name,
            role=role,
            password_hash=self._hash_password(password),
            created_at=datetime.utcnow().isoformat()
        )
        self._users[user.user_id] = user
        self._email_index[user.email] = user.user_id
        return user
    
    def list_users(self) -> List[User]:
        return list(self._users.values())


class AuthManager:
    """Main authentication manager"""
    
    def __init__(self):
        self.user_store = UserStore()
        self.token_blacklist = TokenBlacklist()
        print("🔐 Auth Manager initialized")
    
    def authenticate(self, email: str, password: str) -> Optional[User]:
        """Verify credentials and return user if valid"""
        user = self.user_store.get_by_email(email)
        if user and user.is_active:
            if self.user_store.verify_password(password, user.password_hash):
                return user
        return None
    
    def create_access_token(self, user: User) -> str:
        """Create short-lived access token"""
        payload = {
            "sub": user.user_id,
            "email": user.email,
            "role": user.role.value,
            "type": "access",
            "jti": str(uuid.uuid4()),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        }
        return jwt.encode(payload, JWT_SECRET, algorithm="HS256")
    
    def create_refresh_token(self, user: User) -> str:
        """Create long-lived refresh token"""
        payload = {
            "sub": user.user_id,
            "type": "refresh",
            "jti": str(uuid.uuid4()),
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        }
        return jwt.encode(payload, JWT_REFRESH_SECRET, algorithm="HS256")
    
    def verify_access_token(self, token: str) -> Optional[Dict]:
        """Verify and decode access token"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            if payload.get("type") != "access":
                return None
            if self.token_blacklist.is_blacklisted(payload.get("jti")):
                return None
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def verify_refresh_token(self, token: str) -> Optional[Dict]:
        """Verify and decode refresh token"""
        try:
            payload = jwt.decode(token, JWT_REFRESH_SECRET, algorithms=["HS256"])
            if payload.get("type") != "refresh":
                return None
            if self.token_blacklist.is_blacklisted(payload.get("jti")):
                return None
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def rotate_tokens(self, refresh_token: str) -> Optional[tuple]:
        """Rotate tokens - invalidate old, create new pair"""
        payload = self.verify_refresh_token(refresh_token)
        if not payload:
            return None
        
        # Blacklist old refresh token
        self.token_blacklist.add(payload.get("jti"))
        
        # Get user and create new tokens
        user = self.user_store.get_by_id(payload.get("sub"))
        if not user or not user.is_active:
            return None
        
        new_access = self.create_access_token(user)
        new_refresh = self.create_refresh_token(user)
        
        return new_access, new_refresh
    
    def logout(self, access_token: str, refresh_token: str = None):
        """Invalidate tokens"""
        # Blacklist access token
        payload = self.verify_access_token(access_token)
        if payload:
            self.token_blacklist.add(payload.get("jti"))
        
        # Blacklist refresh token if provided
        if refresh_token:
            payload = self.verify_refresh_token(refresh_token)
            if payload:
                self.token_blacklist.add(payload.get("jti"))
    
    def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from access token"""
        payload = self.verify_access_token(token)
        if payload:
            return self.user_store.get_by_id(payload.get("sub"))
        return None


# Global instance
_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    """Get or create auth manager"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
    return _auth_manager
