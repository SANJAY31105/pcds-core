"""
Audit Logging Middleware for PCDS Enterprise
Logs all user actions for compliance and security
"""
from fastapi import Request
from datetime import datetime
from cache.redis_client import cache_client
import logging
import json

logger = logging.getLogger(__name__)


class AuditLogger:
    """Audit logging for user actions"""
    
    @staticmethod
    async def log_action(
        user_id: str,
        action: str,
        resource: str,
        resource_id: str = None,
        result: str = "success",
        details: dict = None,
        request: Request = None
    ):
        """
        Log audit event
        
        Args:
            user_id: User who performed action
            action: Action type (create, read, update, delete, etc.)
            resource: Resource type (detection, entity, investigation)
            resource_id: Specific resource ID
            result: success or failure
            details: Additional context
            request: FastAPI request object
        """
        audit_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "resource_id": resource_id,
            "result": result,
            "ip_address": request.client.host if request else None,
            "user_agent": request.headers.get("user-agent") if request else None,
            "details": details or {}
        }
        
        try:
            # Store in Redis list (last 10,000 events)
            cache_client.redis.lpush("audit_log", json.dumps(audit_event))
            cache_client.redis.ltrim("audit_log", 0, 9999)
            
            # Also log to file
            logger.info(f"AUDIT: {user_id} performed {action} on {resource}/{resource_id} - {result}")
        
        except Exception as e:
            logger.error(f"Audit logging error: {e}")
    
    @staticmethod
    def get_recent_logs(limit: int = 100) -> list:
        """Get recent audit logs"""
        try:
            logs = cache_client.redis.lrange("audit_log", 0, limit - 1)
            return [json.loads(log) for log in logs]
        except Exception as e:
            logger.error(f"Error retrieving audit logs: {e}")
            return []
    
    @staticmethod
    def get_user_logs(user_id: str, limit: int = 100) -> list:
        """Get audit logs for specific user"""
        all_logs = AuditLogger.get_recent_logs(limit=1000)
        return [log for log in all_logs if log.get("user_id")  == user_id][:limit]


# Global audit logger instance
audit_logger = AuditLogger()
