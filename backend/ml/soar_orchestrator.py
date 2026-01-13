"""
SOAR - Security Orchestration, Automation, and Response
Based on survey paper recommendations for integrated security automation

Features:
- Automated incident triage and prioritization
- Playbook-based response orchestration
- Multi-tool integration (SIEM, EDR, Firewall, Ticketing)
- Case management with evidence collection
- Threat intelligence enrichment
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import hashlib


class IncidentSeverity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class IncidentStatus(str, Enum):
    NEW = "new"
    TRIAGED = "triaged"
    INVESTIGATING = "investigating"
    CONTAINED = "contained"
    ERADICATED = "eradicated"
    RECOVERED = "recovered"
    CLOSED = "closed"


class ActionType(str, Enum):
    ALERT = "alert"
    BLOCK_IP = "block_ip"
    ISOLATE_HOST = "isolate_host"
    DISABLE_USER = "disable_user"
    COLLECT_EVIDENCE = "collect_evidence"
    ENRICH_IOC = "enrich_ioc"
    CREATE_TICKET = "create_ticket"
    NOTIFY = "notify"
    CUSTOM = "custom"


@dataclass
class SecurityIncident:
    """Security incident with full context"""
    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    source: str  # Detection source (ML, SIEM, EDR, etc.)
    created_at: str
    updated_at: str
    
    # Related entities
    affected_hosts: List[str] = field(default_factory=list)
    affected_users: List[str] = field(default_factory=list)
    iocs: List[Dict] = field(default_factory=list)  # Indicators of Compromise
    
    # ML context
    ml_prediction: Optional[Dict] = None
    attack_type: Optional[str] = None
    mitre_techniques: List[str] = field(default_factory=list)
    
    # Response tracking
    actions_taken: List[Dict] = field(default_factory=list)
    playbook_id: Optional[str] = None
    assigned_to: Optional[str] = None
    
    # Evidence
    evidence: List[Dict] = field(default_factory=list)
    timeline: List[Dict] = field(default_factory=list)


@dataclass
class SOARAction:
    """An automated response action"""
    action_id: str
    action_type: ActionType
    target: str
    parameters: Dict
    status: str  # pending, running, completed, failed
    result: Optional[Dict] = None
    executed_at: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ResponsePlaybook:
    """Automated response playbook"""
    playbook_id: str
    name: str
    description: str
    trigger_conditions: Dict  # Conditions to auto-trigger
    actions: List[Dict]  # Ordered list of actions
    enabled: bool = True
    last_executed: Optional[str] = None
    execution_count: int = 0


class SOAROrchestrator:
    """
    Security Orchestration, Automation, and Response Engine
    
    Coordinates automated incident response across:
    - ML Detection Engine
    - SIEM Integration
    - EDR Agents
    - Firewall/Network Controls
    - Ticketing Systems
    """
    
    def __init__(self):
        self.incidents: Dict[str, SecurityIncident] = {}
        self.playbooks: Dict[str, ResponsePlaybook] = {}
        self.action_handlers: Dict[ActionType, Callable] = {}
        self.integrations: Dict[str, Any] = {}
        
        # Default playbooks
        self._register_default_playbooks()
        self._register_default_handlers()
        
        # Statistics
        self.stats = {
            "incidents_created": 0,
            "incidents_auto_triaged": 0,
            "playbooks_executed": 0,
            "actions_executed": 0,
            "mean_time_to_respond": 0.0
        }
        
        print("🎯 SOAR Orchestrator initialized")
    
    def _register_default_playbooks(self):
        """Register default response playbooks"""
        
        # Critical threat response
        self.playbooks["critical_threat"] = ResponsePlaybook(
            playbook_id="critical_threat",
            name="Critical Threat Response",
            description="Immediate response to critical security threats",
            trigger_conditions={
                "severity": ["critical"],
                "attack_types": ["ransomware", "data_exfiltration", "apt"]
            },
            actions=[
                {"type": "isolate_host", "auto": True},
                {"type": "collect_evidence", "auto": True},
                {"type": "notify", "recipients": ["security_team", "ciso"]},
                {"type": "create_ticket", "priority": "P1"}
            ]
        )
        
        # DDoS response
        self.playbooks["ddos_response"] = ResponsePlaybook(
            playbook_id="ddos_response",
            name="DDoS Attack Response",
            description="Automated DDoS mitigation",
            trigger_conditions={
                "attack_types": ["ddos", "DoS Hulk", "DoS GoldenEye", "DoS Slowloris"]
            },
            actions=[
                {"type": "block_ip", "auto": True, "duration": 3600},
                {"type": "alert", "message": "DDoS attack detected and mitigated"},
                {"type": "enrich_ioc", "ioc_type": "ip"}
            ]
        )
        
        # Brute force response
        self.playbooks["brute_force"] = ResponsePlaybook(
            playbook_id="brute_force",
            name="Brute Force Response",
            description="Response to authentication attacks",
            trigger_conditions={
                "attack_types": ["FTP-Patator", "SSH-Patator", "Web Attack - Brute Force"]
            },
            actions=[
                {"type": "block_ip", "auto": True, "duration": 7200},
                {"type": "disable_user", "if_compromised": True},
                {"type": "notify", "recipients": ["security_team"]}
            ]
        )
        
        # SQL Injection response
        self.playbooks["sql_injection"] = ResponsePlaybook(
            playbook_id="sql_injection",
            name="SQL Injection Response",
            description="Web attack containment",
            trigger_conditions={
                "attack_types": ["Web Attack - SQL Injection", "Web Attack - XSS"]
            },
            actions=[
                {"type": "block_ip", "auto": True, "duration": 86400},
                {"type": "collect_evidence", "auto": True},
                {"type": "create_ticket", "priority": "P2"}
            ]
        )
        
        print(f"  ✅ Registered {len(self.playbooks)} default playbooks")
    
    def _register_default_handlers(self):
        """Register default action handlers"""
        self.action_handlers[ActionType.ALERT] = self._handle_alert
        self.action_handlers[ActionType.BLOCK_IP] = self._handle_block_ip
        self.action_handlers[ActionType.ISOLATE_HOST] = self._handle_isolate_host
        self.action_handlers[ActionType.DISABLE_USER] = self._handle_disable_user
        self.action_handlers[ActionType.COLLECT_EVIDENCE] = self._handle_collect_evidence
        self.action_handlers[ActionType.ENRICH_IOC] = self._handle_enrich_ioc
        self.action_handlers[ActionType.CREATE_TICKET] = self._handle_create_ticket
        self.action_handlers[ActionType.NOTIFY] = self._handle_notify
    
    async def create_incident(self, 
                              title: str,
                              description: str,
                              severity: IncidentSeverity,
                              source: str,
                              ml_prediction: Dict = None,
                              affected_hosts: List[str] = None,
                              affected_users: List[str] = None,
                              iocs: List[Dict] = None) -> SecurityIncident:
        """
        Create a new security incident
        """
        incident_id = f"INC-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.utcnow().isoformat()
        
        # Determine attack type from ML prediction
        attack_type = None
        if ml_prediction:
            attack_type = ml_prediction.get("class_name") or ml_prediction.get("attack_type")
        
        incident = SecurityIncident(
            incident_id=incident_id,
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.NEW,
            source=source,
            created_at=now,
            updated_at=now,
            affected_hosts=affected_hosts or [],
            affected_users=affected_users or [],
            iocs=iocs or [],
            ml_prediction=ml_prediction,
            attack_type=attack_type
        )
        
        # Add to timeline
        incident.timeline.append({
            "timestamp": now,
            "event": "Incident created",
            "details": {"source": source, "severity": severity.value}
        })
        
        self.incidents[incident_id] = incident
        self.stats["incidents_created"] += 1
        
        # Auto-triage and trigger playbooks
        await self._auto_triage(incident)
        
        return incident
    
    async def _auto_triage(self, incident: SecurityIncident):
        """
        Automatically triage incident and trigger appropriate playbooks
        """
        incident.status = IncidentStatus.TRIAGED
        self.stats["incidents_auto_triaged"] += 1
        
        # Find matching playbooks
        for playbook in self.playbooks.values():
            if not playbook.enabled:
                continue
            
            if self._matches_playbook(incident, playbook):
                await self._execute_playbook(incident, playbook)
                break
    
    def _matches_playbook(self, incident: SecurityIncident, 
                         playbook: ResponsePlaybook) -> bool:
        """Check if incident matches playbook trigger conditions"""
        conditions = playbook.trigger_conditions
        
        # Check severity
        if "severity" in conditions:
            if incident.severity.value not in conditions["severity"]:
                return False
        
        # Check attack types
        if "attack_types" in conditions and incident.attack_type:
            if incident.attack_type not in conditions["attack_types"]:
                return False
        
        return True
    
    async def _execute_playbook(self, incident: SecurityIncident, 
                               playbook: ResponsePlaybook):
        """Execute a response playbook"""
        incident.playbook_id = playbook.playbook_id
        incident.status = IncidentStatus.INVESTIGATING
        
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": f"Playbook triggered: {playbook.name}",
            "details": {"playbook_id": playbook.playbook_id}
        })
        
        for action_def in playbook.actions:
            action_type = ActionType(action_def["type"])
            
            action = SOARAction(
                action_id=str(uuid.uuid4())[:8],
                action_type=action_type,
                target=self._get_action_target(incident, action_type),
                parameters=action_def,
                status="pending"
            )
            
            # Execute action
            if action_def.get("auto", False):
                await self._execute_action(incident, action)
            
            incident.actions_taken.append(asdict(action))
        
        playbook.last_executed = datetime.utcnow().isoformat()
        playbook.execution_count += 1
        self.stats["playbooks_executed"] += 1
    
    def _get_action_target(self, incident: SecurityIncident, 
                          action_type: ActionType) -> str:
        """Determine target for an action"""
        if action_type in [ActionType.BLOCK_IP, ActionType.ENRICH_IOC]:
            # Get IP from IOCs
            for ioc in incident.iocs:
                if ioc.get("type") == "ip":
                    return ioc.get("value", "unknown")
            return "unknown"
        
        if action_type == ActionType.ISOLATE_HOST:
            return incident.affected_hosts[0] if incident.affected_hosts else "unknown"
        
        if action_type == ActionType.DISABLE_USER:
            return incident.affected_users[0] if incident.affected_users else "unknown"
        
        return incident.incident_id
    
    async def _execute_action(self, incident: SecurityIncident, action: SOARAction):
        """Execute a single action"""
        action.status = "running"
        
        handler = self.action_handlers.get(action.action_type)
        if handler:
            try:
                result = await handler(incident, action)
                action.status = "completed"
                action.result = result
                action.executed_at = datetime.utcnow().isoformat()
                self.stats["actions_executed"] += 1
            except Exception as e:
                action.status = "failed"
                action.error = str(e)
        else:
            action.status = "failed"
            action.error = f"No handler for action type: {action.action_type}"
        
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": f"Action: {action.action_type.value}",
            "details": {"status": action.status, "target": action.target}
        })
    
    # Action Handlers
    async def _handle_alert(self, incident: SecurityIncident, action: SOARAction) -> Dict:
        message = action.parameters.get("message", f"Security Alert: {incident.title}")
        print(f"🚨 ALERT: {message}")
        return {"message_sent": True, "message": message}
    
    async def _handle_block_ip(self, incident: SecurityIncident, action: SOARAction) -> Dict:
        ip = action.target
        duration = action.parameters.get("duration", 3600)
        print(f"🔒 BLOCKING IP: {ip} for {duration}s")
        return {"ip_blocked": ip, "duration": duration, "success": True}
    
    async def _handle_isolate_host(self, incident: SecurityIncident, action: SOARAction) -> Dict:
        host = action.target
        print(f"🔐 ISOLATING HOST: {host}")
        incident.status = IncidentStatus.CONTAINED
        return {"host_isolated": host, "success": True}
    
    async def _handle_disable_user(self, incident: SecurityIncident, action: SOARAction) -> Dict:
        user = action.target
        print(f"👤 DISABLING USER: {user}")
        return {"user_disabled": user, "success": True}
    
    async def _handle_collect_evidence(self, incident: SecurityIncident, action: SOARAction) -> Dict:
        print(f"📦 COLLECTING EVIDENCE for {incident.incident_id}")
        evidence = {
            "collected_at": datetime.utcnow().isoformat(),
            "ml_prediction": incident.ml_prediction,
            "iocs": incident.iocs,
            "affected_hosts": incident.affected_hosts
        }
        incident.evidence.append(evidence)
        return {"evidence_collected": True, "items": 4}
    
    async def _handle_enrich_ioc(self, incident: SecurityIncident, action: SOARAction) -> Dict:
        ioc = action.target
        print(f"🔍 ENRICHING IOC: {ioc}")
        # Simulate threat intelligence lookup
        enrichment = {
            "ioc": ioc,
            "reputation": "malicious",
            "first_seen": "2024-01-15",
            "associated_malware": ["Mirai", "Gafgyt"],
            "geo": "Unknown"
        }
        return {"enrichment": enrichment}
    
    async def _handle_create_ticket(self, incident: SecurityIncident, action: SOARAction) -> Dict:
        priority = action.parameters.get("priority", "P3")
        ticket_id = f"TKT-{uuid.uuid4().hex[:6].upper()}"
        print(f"📋 CREATING TICKET: {ticket_id} (Priority: {priority})")
        return {"ticket_id": ticket_id, "priority": priority, "created": True}
    
    async def _handle_notify(self, incident: SecurityIncident, action: SOARAction) -> Dict:
        recipients = action.parameters.get("recipients", ["security_team"])
        print(f"📧 NOTIFYING: {', '.join(recipients)}")
        return {"notified": recipients, "count": len(recipients)}
    
    def get_incident(self, incident_id: str) -> Optional[SecurityIncident]:
        return self.incidents.get(incident_id)
    
    def list_incidents(self, status: str = None, limit: int = 50) -> List[Dict]:
        incidents = list(self.incidents.values())
        
        if status:
            incidents = [i for i in incidents if i.status.value == status]
        
        # Sort by created_at descending
        incidents.sort(key=lambda x: x.created_at, reverse=True)
        
        return [asdict(i) for i in incidents[:limit]]
    
    def get_stats(self) -> Dict:
        return {
            **self.stats,
            "total_incidents": len(self.incidents),
            "total_playbooks": len(self.playbooks),
            "playbooks": {p.playbook_id: p.name for p in self.playbooks.values()}
        }


# Global instance
_soar: Optional[SOAROrchestrator] = None


def get_soar() -> SOAROrchestrator:
    """Get or create SOAR instance"""
    global _soar
    if _soar is None:
        _soar = SOAROrchestrator()
    return _soar
