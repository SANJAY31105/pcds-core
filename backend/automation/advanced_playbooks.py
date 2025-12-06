"""
PCDS Enterprise - Advanced Automated Playbook Engine
Threat containment, remediation, and response automation
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json


class PlaybookStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class PlaybookAction:
    """Single action in a playbook"""
    id: str
    name: str
    action_type: str
    parameters: Dict = field(default_factory=dict)
    timeout_seconds: int = 30
    continue_on_failure: bool = False
    
    
@dataclass
class PlaybookExecution:
    """Execution record for a playbook run"""
    id: str
    playbook_id: str
    trigger: Dict
    status: PlaybookStatus = PlaybookStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actions_completed: int = 0
    actions_failed: int = 0
    results: List[Dict] = field(default_factory=list)


class AutomatedPlaybooks:
    """
    Enterprise automated response playbooks
    Supports containment, remediation, and notification actions
    """
    
    def __init__(self):
        self.playbooks = {}
        self.executions: List[PlaybookExecution] = []
        self.action_handlers: Dict[str, Callable] = {}
        self._register_default_actions()
        self._register_default_playbooks()
        
    def _register_default_actions(self):
        """Register built-in action handlers"""
        self.action_handlers = {
            "isolate_host": self._action_isolate_host,
            "disable_user": self._action_disable_user,
            "block_ip": self._action_block_ip,
            "terminate_process": self._action_terminate_process,
            "quarantine_file": self._action_quarantine_file,
            "send_alert": self._action_send_alert,
            "create_ticket": self._action_create_ticket,
            "collect_forensics": self._action_collect_forensics,
            "scan_entity": self._action_scan_entity,
            "notify_soc": self._action_notify_soc,
            "run_edr_scan": self._action_run_edr_scan,
            "reset_password": self._action_reset_password,
            "revoke_sessions": self._action_revoke_sessions,
            "snapshot_state": self._action_snapshot_state,
            "log_audit": self._action_log_audit,
        }
        
    def _register_default_playbooks(self):
        """Register enterprise playbooks"""
        
        # Ransomware Response Playbook
        self.playbooks["ransomware_response"] = {
            "id": "ransomware_response",
            "name": "Ransomware Rapid Response",
            "description": "Automated containment for ransomware detection",
            "trigger_conditions": {
                "technique_ids": ["T1486", "T1490", "T1485"],
                "severity": ["critical"]
            },
            "actions": [
                PlaybookAction("1", "Snapshot State", "snapshot_state", {"reason": "Pre-isolation snapshot"}),
                PlaybookAction("2", "Isolate Host", "isolate_host", {"notify": True}),
                PlaybookAction("3", "Block Lateral Movement", "block_ip", {"scope": "internal"}),
                PlaybookAction("4", "Notify SOC", "notify_soc", {"priority": "P1"}),
                PlaybookAction("5", "Create Incident", "create_ticket", {"severity": "critical", "category": "ransomware"}),
                PlaybookAction("6", "Collect Forensics", "collect_forensics", {"artifacts": ["memory", "disk", "network"]}),
                PlaybookAction("7", "Log Audit", "log_audit", {"action": "ransomware_response_triggered"}),
            ]
        }
        
        # Credential Theft Response
        self.playbooks["credential_theft"] = {
            "id": "credential_theft",
            "name": "Credential Theft Response",
            "description": "Response to credential dumping and theft",
            "trigger_conditions": {
                "technique_ids": ["T1003", "T1558", "T1552", "T1110"],
                "severity": ["critical", "high"]
            },
            "actions": [
                PlaybookAction("1", "Disable User", "disable_user", {"temporary": True}),
                PlaybookAction("2", "Revoke Sessions", "revoke_sessions", {}),
                PlaybookAction("3", "Reset Password", "reset_password", {"force": True}),
                PlaybookAction("4", "Scan Endpoints", "run_edr_scan", {"scope": "user_devices"}),
                PlaybookAction("5", "Notify SOC", "notify_soc", {"priority": "P2"}),
                PlaybookAction("6", "Create Ticket", "create_ticket", {"severity": "high", "category": "credential_theft"}),
            ]
        }
        
        # Lateral Movement Containment
        self.playbooks["lateral_movement"] = {
            "id": "lateral_movement",
            "name": "Lateral Movement Containment",
            "description": "Contain active lateral movement",
            "trigger_conditions": {
                "technique_ids": ["T1021", "T1550", "T1570"],
                "severity": ["critical", "high"]
            },
            "actions": [
                PlaybookAction("1", "Block Source IP", "block_ip", {"scope": "source"}),
                PlaybookAction("2", "Isolate Source Host", "isolate_host", {"mode": "soft"}),
                PlaybookAction("3", "Scan Destination", "scan_entity", {"type": "full"}),
                PlaybookAction("4", "Notify SOC", "notify_soc", {"priority": "P1"}),
                PlaybookAction("5", "Log Audit", "log_audit", {"action": "lateral_movement_blocked"}),
            ]
        }
        
        # Data Exfiltration Response
        self.playbooks["data_exfiltration"] = {
            "id": "data_exfiltration",
            "name": "Data Exfiltration Response",
            "description": "Block active data exfiltration",
            "trigger_conditions": {
                "technique_ids": ["T1041", "T1048", "T1567", "T1052"],
                "severity": ["critical", "high"]
            },
            "actions": [
                PlaybookAction("1", "Block External IPs", "block_ip", {"scope": "external", "duration": 3600}),
                PlaybookAction("2", "Terminate Process", "terminate_process", {"force": True}),
                PlaybookAction("3", "Isolate Host", "isolate_host", {"allow_dns": False}),
                PlaybookAction("4", "Collect Network Capture", "collect_forensics", {"artifacts": ["network"]}),
                PlaybookAction("5", "Create Incident", "create_ticket", {"severity": "critical", "category": "data_exfiltration"}),
                PlaybookAction("6", "Notify DLP Team", "send_alert", {"team": "dlp", "priority": "urgent"}),
            ]
        }
        
        # C2 Communication Response
        self.playbooks["c2_communication"] = {
            "id": "c2_communication",
            "name": "C2 Communication Response",
            "description": "Block command and control traffic",
            "trigger_conditions": {
                "technique_ids": ["T1071", "T1090", "T1572", "T1573"],
                "severity": ["critical", "high"]
            },
            "actions": [
                PlaybookAction("1", "Block C2 IP", "block_ip", {"scope": "destination", "permanent": True}),
                PlaybookAction("2", "Terminate Process", "terminate_process", {}),
                PlaybookAction("3", "Run EDR Scan", "run_edr_scan", {"deep": True}),
                PlaybookAction("4", "Quarantine Files", "quarantine_file", {"pattern": "suspicious"}),
                PlaybookAction("5", "Notify SOC", "notify_soc", {"priority": "P1"}),
            ]
        }
        
        # Insider Threat Response
        self.playbooks["insider_threat"] = {
            "id": "insider_threat",
            "name": "Insider Threat Response",
            "description": "Handle insider threat indicators",
            "trigger_conditions": {
                "entity_type": ["user"],
                "detection_types": ["policy_violation", "unusual_access", "mass_download"]
            },
            "actions": [
                PlaybookAction("1", "Log Audit", "log_audit", {"action": "insider_threat_detected"}),
                PlaybookAction("2", "Snapshot Activity", "snapshot_state", {}),
                PlaybookAction("3", "Create HR Ticket", "create_ticket", {"severity": "high", "category": "insider_threat", "route_to": "hr"}),
                PlaybookAction("4", "Notify Security Manager", "send_alert", {"team": "security_management"}),
            ]
        }
        
        # Malware Detection Response
        self.playbooks["malware_detection"] = {
            "id": "malware_detection",
            "name": "Malware Detection Response",
            "description": "Automated malware containment",
            "trigger_conditions": {
                "detection_types": ["malware", "trojan", "backdoor", "rootkit"],
                "severity": ["critical", "high"]
            },
            "actions": [
                PlaybookAction("1", "Quarantine File", "quarantine_file", {}),
                PlaybookAction("2", "Terminate Process", "terminate_process", {"force": True}),
                PlaybookAction("3", "Run Full Scan", "run_edr_scan", {"scope": "full", "deep": True}),
                PlaybookAction("4", "Isolate if Needed", "isolate_host", {"conditional": True, "threshold": 3}),
                PlaybookAction("5", "Create Ticket", "create_ticket", {"severity": "high", "category": "malware"}),
            ]
        }
    
    # ========== Action Implementations ==========
    
    async def _action_isolate_host(self, detection: Dict, params: Dict) -> Dict:
        """Isolate host from network"""
        entity_id = detection.get("entity_id", "unknown")
        print(f"🔒 ACTION: Isolating host {entity_id}")
        return {"success": True, "action": "isolate_host", "entity_id": entity_id}
    
    async def _action_disable_user(self, detection: Dict, params: Dict) -> Dict:
        """Disable user account"""
        entity_id = detection.get("entity_id", "unknown")
        print(f"👤 ACTION: Disabling user {entity_id}")
        return {"success": True, "action": "disable_user", "user": entity_id}
    
    async def _action_block_ip(self, detection: Dict, params: Dict) -> Dict:
        """Block IP address"""
        ip = detection.get("source_ip") or detection.get("destination_ip", "unknown")
        scope = params.get("scope", "both")
        print(f"🚫 ACTION: Blocking IP {ip} (scope: {scope})")
        return {"success": True, "action": "block_ip", "ip": ip, "scope": scope}
    
    async def _action_terminate_process(self, detection: Dict, params: Dict) -> Dict:
        """Terminate malicious process"""
        print(f"💀 ACTION: Terminating process")
        return {"success": True, "action": "terminate_process"}
    
    async def _action_quarantine_file(self, detection: Dict, params: Dict) -> Dict:
        """Quarantine malicious file"""
        print(f"📦 ACTION: Quarantining file")
        return {"success": True, "action": "quarantine_file"}
    
    async def _action_send_alert(self, detection: Dict, params: Dict) -> Dict:
        """Send alert to team"""
        team = params.get("team", "soc")
        print(f"📧 ACTION: Sending alert to {team}")
        return {"success": True, "action": "send_alert", "team": team}
    
    async def _action_create_ticket(self, detection: Dict, params: Dict) -> Dict:
        """Create incident ticket"""
        severity = params.get("severity", "medium")
        category = params.get("category", "security")
        print(f"🎫 ACTION: Creating {severity} ticket ({category})")
        return {"success": True, "action": "create_ticket", "severity": severity, "category": category}
    
    async def _action_collect_forensics(self, detection: Dict, params: Dict) -> Dict:
        """Collect forensic artifacts"""
        artifacts = params.get("artifacts", ["memory"])
        print(f"🔍 ACTION: Collecting forensics: {artifacts}")
        return {"success": True, "action": "collect_forensics", "artifacts": artifacts}
    
    async def _action_scan_entity(self, detection: Dict, params: Dict) -> Dict:
        """Scan entity for threats"""
        print(f"🔎 ACTION: Scanning entity")
        return {"success": True, "action": "scan_entity"}
    
    async def _action_notify_soc(self, detection: Dict, params: Dict) -> Dict:
        """Notify SOC team"""
        priority = params.get("priority", "P2")
        print(f"📢 ACTION: Notifying SOC ({priority})")
        return {"success": True, "action": "notify_soc", "priority": priority}
    
    async def _action_run_edr_scan(self, detection: Dict, params: Dict) -> Dict:
        """Run EDR scan"""
        scope = params.get("scope", "quick")
        print(f"🛡️ ACTION: Running EDR scan ({scope})")
        return {"success": True, "action": "run_edr_scan", "scope": scope}
    
    async def _action_reset_password(self, detection: Dict, params: Dict) -> Dict:
        """Reset user password"""
        print(f"🔑 ACTION: Resetting password")
        return {"success": True, "action": "reset_password"}
    
    async def _action_revoke_sessions(self, detection: Dict, params: Dict) -> Dict:
        """Revoke all user sessions"""
        print(f"🚪 ACTION: Revoking all sessions")
        return {"success": True, "action": "revoke_sessions"}
    
    async def _action_snapshot_state(self, detection: Dict, params: Dict) -> Dict:
        """Snapshot current state"""
        print(f"📸 ACTION: Taking state snapshot")
        return {"success": True, "action": "snapshot_state", "timestamp": datetime.utcnow().isoformat()}
    
    async def _action_log_audit(self, detection: Dict, params: Dict) -> Dict:
        """Log audit entry"""
        action = params.get("action", "playbook_action")
        print(f"📝 ACTION: Logging audit - {action}")
        return {"success": True, "action": "log_audit", "audit_action": action}
    
    # ========== Execution Engine ==========
    
    async def check_and_execute(self, detection: Dict) -> Optional[PlaybookExecution]:
        """Check if detection triggers any playbook and execute"""
        technique_id = detection.get("technique_id")
        severity = detection.get("severity")
        detection_type = detection.get("detection_type")
        
        for playbook in self.playbooks.values():
            conditions = playbook.get("trigger_conditions", {})
            
            # Check technique match
            if technique_id and technique_id in conditions.get("technique_ids", []):
                return await self.execute_playbook(playbook["id"], detection)
            
            # Check severity match
            if severity in conditions.get("severity", []):
                if technique_id in conditions.get("technique_ids", []):
                    return await self.execute_playbook(playbook["id"], detection)
            
            # Check detection type match
            if detection_type in conditions.get("detection_types", []):
                return await self.execute_playbook(playbook["id"], detection)
        
        return None
    
    async def execute_playbook(self, playbook_id: str, detection: Dict) -> PlaybookExecution:
        """Execute a specific playbook"""
        playbook = self.playbooks.get(playbook_id)
        if not playbook:
            raise ValueError(f"Playbook {playbook_id} not found")
        
        execution = PlaybookExecution(
            id=f"exec_{datetime.utcnow().timestamp()}",
            playbook_id=playbook_id,
            trigger=detection,
            status=PlaybookStatus.RUNNING,
            started_at=datetime.utcnow()
        )
        
        print(f"\n🚀 EXECUTING PLAYBOOK: {playbook['name']}")
        print(f"   Trigger: {detection.get('detection_type', 'unknown')}")
        print(f"   Entity: {detection.get('entity_id', 'unknown')}")
        
        for action in playbook.get("actions", []):
            try:
                handler = self.action_handlers.get(action.action_type)
                if handler:
                    result = await handler(detection, action.parameters)
                    execution.results.append(result)
                    execution.actions_completed += 1
                else:
                    print(f"   ⚠️ Unknown action: {action.action_type}")
                    execution.actions_failed += 1
            except Exception as e:
                print(f"   ❌ Action failed: {e}")
                execution.actions_failed += 1
                if not action.continue_on_failure:
                    break
        
        execution.status = PlaybookStatus.COMPLETED
        execution.completed_at = datetime.utcnow()
        self.executions.append(execution)
        
        print(f"✅ PLAYBOOK COMPLETE: {execution.actions_completed} actions, {execution.actions_failed} failed\n")
        
        return execution
    
    def get_playbook_list(self) -> List[Dict]:
        """Get list of available playbooks"""
        return [
            {
                "id": p["id"],
                "name": p["name"],
                "description": p["description"],
                "trigger_conditions": p["trigger_conditions"],
                "action_count": len(p.get("actions", []))
            }
            for p in self.playbooks.values()
        ]
    
    def get_execution_history(self, limit: int = 50) -> List[Dict]:
        """Get recent playbook executions"""
        return [
            {
                "id": e.id,
                "playbook_id": e.playbook_id,
                "status": e.status.value,
                "started_at": e.started_at.isoformat() if e.started_at else None,
                "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                "actions_completed": e.actions_completed,
                "actions_failed": e.actions_failed
            }
            for e in self.executions[-limit:]
        ]


# Global instance
advanced_playbook_engine = AutomatedPlaybooks()
