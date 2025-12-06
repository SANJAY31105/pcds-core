"""
SOAR (Security Orchestration, Automation and Response) Engine
Automated playbooks for incident response
"""
from typing import List, Dict, Any, Callable
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class PlaybookStatus(str, Enum):
    """Playbook execution status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


class Action:
    """Single action in a playbook"""
    
    def __init__(self, name: str, function: Callable, params: Dict = None):
        self.name = name
        self.function = function
        self.params = params or {}
    
    async def execute(self, context: Dict) -> bool:
        """Execute action"""
        try:
            logger.info(f"Executing action: {self.name}")
            result = await self.function(context, **self.params)
            logger.info(f"Action {self.name} completed: {result}")
            return True
        except Exception as e:
            logger.error(f"Action {self.name} failed: {e}")
            return False


class Playbook:
    """Automated response playbook"""
    
    def __init__(self, name: str, description: str, actions: List[Action]):
        self.name = name
        self.description = description
        self.actions = actions
        self.status = PlaybookStatus.PENDING
    
    async def execute(self, context: Dict) -> PlaybookStatus:
        """Execute all actions in playbook"""
        self.status = PlaybookStatus.RUNNING
        logger.info(f"Starting playbook: {self.name}")
        
        success_count = 0
        total = len(self.actions)
        
        for action in self.actions:
            if await action.execute(context):
                success_count += 1
        
        # Determine final status
        if success_count == total:
            self.status = PlaybookStatus.SUCCESS
        elif success_count == 0:
            self.status = PlaybookStatus.FAILED
        else:
            self.status = PlaybookStatus.PARTIAL
        
        logger.info(f"Playbook {self.name} completed: {self.status} ({success_count}/{total} actions)")
        return self.status


# ===== RESPONSE ACTIONS =====

async def isolate_entity(context: Dict, entity_id: str = None):
    """Isolate compromised entity"""
    entity_id = entity_id or context.get("entity_id")
    logger.warning(f"🔒 ISOLATING ENTITY: {entity_id}")
    
    # In production: firewall API call
    # await firewall.block_ip(entity.ip_address)
    
    # Update entity status
    from config.database import db_manager
    db_manager.execute_update("""
        UPDATE entities 
        SET status = 'isolated', isolated_at = ?
        WHERE identifier = ?
    """, (datetime.utcnow(), entity_id))
    
    return True


async def block_c2_traffic(context: Dict, c2_addresses: List[str] = None):
    """Block command and control traffic"""
    c2_list = c2_addresses or context.get("c2_addresses", [])
    logger.warning(f"🚫 BLOCKING C2 TRAFFIC: {len(c2_list)} addresses")
    
    # In production: Update firewall/IDS rules
    # for address in c2_list:
    #     await firewall.block_ip(address)
    
    return True


async def create_investigation(context: Dict, title: str = None):
    """Create automated investigation"""
    title = title or f"Auto-investigation: {context.get('detection_type', 'Unknown')}"
    logger.info(f"📋 CREATING INVESTIGATION: {title}")
    
    from config.database import db_manager
    
    investigation_id = db_manager.execute_insert("""
        INSERT INTO investigations (title, status, created_at, priority)
        VALUES (?, 'open', ?, 'high')
    """, (title, datetime.utcnow()))
    
    context['investigation_id'] = investigation_id
    return True


async def alert_security_team(context: Dict, severity: str = "high"):
    """Alert security team via Slack/email"""
    logger.critical(f"🚨 ALERTING SECURITY TEAM: {severity}")
    
    # In production: Send to Slack/Teams/Email
    # await slack_client.post_message(
    #     channel="#security",
    #     text=f"🚨 Critical Alert: {context.get('description')}"
    # )
    
    return True


async def preserve_forensics(context: Dict):
    """Preserve forensic evidence"""
    logger.info("💾 PRESERVING FORENSICS")
    
    # In production: Snapshot VM, capture memory, logs
    # await forensics.capture_memory(entity_id)
    # await forensics.copy_logs(entity_id)
    
    return True


async def disable_user_account(context: Dict, user_id: str = None):
    """Disable compromised user account"""
    user_id = user_id or context.get("user_id")
    logger.warning(f"👤 DISABLING USER ACCOUNT: {user_id}")
    
    # In production: AD/LDAP API call
    # await ad.disable_account(user_id)
    
    return True


async def export_to_siem(context: Dict):
    """Export detection to SIEM"""
    logger.info("📤 EXPORTING TO SIEM")
    
    # In production: Send to Splunk, QRadar, etc.
    # await splunk.send_event(context)
    
    return True


# ===== PREDEFINED PLAYBOOKS =====

# Ransomware Response Playbook
ransomware_playbook = Playbook(
    name="Ransomware Response",
    description="Automated response to ransomware detection",
    actions=[
        Action("isolate_entity", isolate_entity),
        Action("block_c2_traffic", block_c2_traffic),
        Action("create_investigation", create_investigation, {"title": "Ransomware Incident"}),
        Action("alert_security_team", alert_security_team, {"severity": "critical"}),
        Action("preserve_forensics", preserve_forensics),
        Action("export_to_siem", export_to_siem),
    ]
)

# Data Exfiltration Playbook
exfiltration_playbook = Playbook(
    name="Data Exfiltration Response",
    description="Response to data exfiltration attempts",
    actions=[
        Action("block_c2_traffic", block_c2_traffic),
        Action("isolate_entity", isolate_entity),
        Action("create_investigation", create_investigation, {"title": "Data Exfiltration"}),
        Action("alert_security_team", alert_security_team),
        Action("preserve_forensics", preserve_forensics),
    ]
)

# Compromised Credentials Playbook
compromised_creds_playbook = Playbook(
    name="Compromised Credentials",
    description="Response to credential theft",
    actions=[
        Action("disable_user_account", disable_user_account),
        Action("create_investigation", create_investigation, {"title": "Credential Compromise"}),
        Action("alert_security_team", alert_security_team),
        Action("export_to_siem", export_to_siem),
    ]
)

# APT Detection Playbook
apt_playbook = Playbook(
    name="APT Response",
    description="Advanced persistent threat response",
    actions=[
        Action("preserve_forensics", preserve_forensics),
        Action("create_investigation", create_investigation, {"title": "APT Activity"}),
        Action("alert_security_team", alert_security_team, {"severity": "critical"}),
        Action("isolate_entity", isolate_entity),
        Action("block_c2_traffic", block_c2_traffic),
        Action("export_to_siem", export_to_siem),
    ]
)


# Playbook registry
PLAYBOOKS = {
    "ransomware": ransomware_playbook,
    "data_exfiltration": exfiltration_playbook,
    "compromised_credentials": compromised_creds_playbook,
    "apt": apt_playbook,
}


class SOAREngine:
    """Main SOAR orchestration engine"""
    
    @staticmethod
    async def execute_playbook(playbook_name: str, context: Dict) -> PlaybookStatus:
        """
        Execute named playbook
        
        Args:
            playbook_name: Name of playbook to execute
            context: Context data (detection info, entity info, etc.)
            
        Returns:
            Playbook execution status
        """
        playbook = PLAYBOOKS.get(playbook_name)
        
        if not playbook:
            logger.error(f"Playbook not found: {playbook_name}")
            return PlaybookStatus.FAILED
        
        # Execute playbook
        status = await playbook.execute(context)
        
        # Log execution
        logger.info(f"Playbook {playbook_name} completed with status: {status}")
        
        return status
    
    @staticmethod
    def get_playbook_for_detection(detection_type: str) -> str:
        """Get appropriate playbook for detection type"""
        mapping = {
            "ransomware": "ransomware",
            "data_exfiltration": "data_exfiltration",
            "credential_theft": "compromised_credentials",
            "apt": "apt",
        }
        return mapping.get(detection_type.lower())


# Global SOAR engine instance
soar_engine = SOAREngine()
