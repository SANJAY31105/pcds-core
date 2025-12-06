from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError, InvalidHash
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES

# Use Argon2id - recommended by OWASP for password hashing
# More secure than bcrypt, resistant to GPU attacks
ph = PasswordHasher(
    time_cost=2,        # Number of iterations
    memory_cost=65536,  # Memory usage in kibibytes (64 MB)
    parallelism=1,      # Number of parallel threads
    hash_len=32,        # Length of hash in bytes
    salt_len=16         # Length of salt in bytes
)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify password using Argon2id algorithm.
    Production-ready - no plain-text fallback.
    """
    try:
        # Verify password with Argon2
        ph.verify(hashed_password, plain_password)
        
        # Check if password needs rehashing (parameters changed)
        if ph.check_needs_rehash(hashed_password):
            logger.info("Password hash parameters outdated, needs rehashing")
        
        return True
    except VerifyMismatchError:
        # Password doesn't match
        return False
    except (VerificationError, InvalidHash) as e:
        # Invalid hash format
        logger.error(f"Password verification error: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during password verification: {e}")
        return False


def get_password_hash(password: str) -> str:
    """
    Hash password using Argon2id algorithm.
    Production-ready - always returns secure hash.
    """
    try:
        return ph.hash(password)
    except Exception as e:
        logger.error(f"Failed to hash password: {e}")
        raise ValueError("Password hashing failed")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        print(f"❌ JWT Decode Error: {str(e)}")
        print(f"   Token: {token[:10]}...{token[-10:]}")
        print(f"   Key used: {SECRET_KEY[:5]}...")
        return None
