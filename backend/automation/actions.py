import logging
from datetime import datetime
import uuid
from config.database import db_manager, EntityQueries

logger = logging.getLogger("pcds.automation")

class ResponseActions:
    """
    Defines available response actions for playbooks.
    """
    
    @staticmethod
    def isolate_host(entity_id: str, reason: str = "Automated Response"):
        """Isolate a host from the network"""
        logger.warning(f"🛡️ ACTION: Isolating host {entity_id}. Reason: {reason}")
        
        # Update entity status in DB
        db_manager.execute_update("""
            UPDATE entities 
            SET is_isolated = 1, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        """, (entity_id,))
        
        # Log the action
        ResponseActions._log_action("isolate_host", entity_id, "success", reason)
        return True

    @staticmethod
    def block_ip(ip_address: str, reason: str = "Automated Response"):
        """Block an IP address at the firewall (Simulated)"""
        logger.warning(f"🛡️ ACTION: Blocking IP {ip_address}. Reason: {reason}")
        # In a real system, this would call a Firewall API
        return True

    @staticmethod
    def disable_user(user_id: str, reason: str = "Automated Response"):
        """Disable a user account"""
        logger.warning(f"🛡️ ACTION: Disabling user {user_id}. Reason: {reason}")
        # Update user status (if we had a users table for monitored users, distinct from system users)
        return True

    @staticmethod
    def _log_action(action_type: str, target_id: str, status: str, details: str):
        """Log the action to the database"""
        try:
            db_manager.execute_insert("""
                INSERT INTO response_actions (id, playbook_name, action_type, target_entity_id, status, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()),
                "Automated Playbook", 
                action_type, 
                target_id, 
                status, 
                details
            ))
        except Exception as e:
            logger.error(f"Failed to log response action: {e}")
