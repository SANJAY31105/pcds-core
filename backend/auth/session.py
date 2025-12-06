"""
Session Management with Redis
"""
from cache.redis_client import cache_client
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SessionManager:
    """Redis-based session management"""
    
    def __init__(self, ttl: int = 86400):  #24 hours default
        self.cache = cache_client
        self.ttl = ttl
    
    def create_session(self, user_id: str, user_data: Dict[str, Any] = None) -> str:
        """
        Create new session
        
        Args:
            user_id: User identifier
            user_data: Additional user data to store
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        session_data = {
            'user_id': user_id,
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            **(user_data or {})
        }
        
        try:
            self.cache.set(f"session:{session_id}", session_data, self.ttl)
            logger.info(f"Session created for user {user_id}: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Session creation error: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        try:
            session_data = self.cache.get(f"session:{session_id}")
            if session_data:
                # Update last activity
                session_data['last_activity'] = datetime.now().isoformat()
                self.cache.set(f"session:{session_id}", session_data, self.ttl)
            return session_data
        except Exception as e:
            logger.error(f"Session get error: {e}")
            return None
    
    def get_user_id(self, session_id: str) -> Optional[str]:
        """Get user ID from session"""
        session = self.get_session(session_id)
        return session['user_id'] if session else None
    
    def delete_session(self, session_id: str):
        """Delete session (logout)"""
        try:
            self.cache.delete(f"session:{session_id}")
            logger.info(f"Session deleted: {session_id}")
        except Exception as e:
            logger.error(f"Session deletion error: {e}")
    
    def extend_session(self, session_id: str):
        """Extend session TTL"""
        session = self.get_session(session_id)
        if session:
            self.cache.set(f"session:{session_id}", session, self.ttl)


# Global session manager instance
session_manager = SessionManager()
