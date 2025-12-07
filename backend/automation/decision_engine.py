"""
PCDS Enterprise - Response Decision Engine
Enterprise-grade automated response with human-in-the-loop

Flow:
  Detection → Decision Engine → Policy Check → Confidence Check 
  → Impact Assessment → Auto-Isolate OR Analyst Approval
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from pydantic import BaseModel


# ============================================================
# ENUMS & MODELS
# ============================================================

class ResponseAction(str, Enum):
    ISOLATE_HOST = "isolate_host"
    BLOCK_IP = "block_ip"
    DISABLE_USER = "disable_user"
    KILL_PROCESS = "kill_process"
    QUARANTINE_FILE = "quarantine_file"
    NONE = "none"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"
    EXPIRED = "expired"


class ImpactLevel(str, Enum):
    LOW = "low"           # Single user affected
    MEDIUM = "medium"     # Department affected
    HIGH = "high"         # Organization affected
    CRITICAL = "critical" # Business critical systems


@dataclass
class PolicyRule:
    """Defines when auto-response is allowed"""
    id: str
    name: str
    description: str
    
    # Conditions for auto-response
    min_confidence: float = 0.85        # ML confidence threshold
    allowed_severities: List[str] = field(default_factory=lambda: ["critical"])
    allowed_techniques: List[str] = field(default_factory=list)  # MITRE IDs
    max_impact_level: ImpactLevel = ImpactLevel.LOW
    
    # Response configuration
    allowed_actions: List[ResponseAction] = field(default_factory=list)
    require_approval: bool = True
    auto_approve_after_mins: int = 0    # 0 = never auto-approve
    
    enabled: bool = True


@dataclass
class ApprovalRequest:
    """Request for analyst approval"""
    id: str
    detection_id: str
    entity_id: str
    proposed_action: ResponseAction
    reason: str
    confidence: float
    impact_level: ImpactLevel
    policy_id: str
    
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    reviewed_at: Optional[datetime] = None
    reviewed_by: Optional[str] = None
    reviewer_notes: Optional[str] = None


class DecisionResult(BaseModel):
    """Result of decision engine evaluation"""
    detection_id: str
    entity_id: str
    
    # Decision outcome
    action: ResponseAction
    auto_execute: bool
    requires_approval: bool
    approval_id: Optional[str] = None
    
    # Reasoning
    confidence: float
    impact_level: str
    policy_matched: Optional[str] = None
    decision_reason: str
    
    # Execution status
    executed: bool = False
    execution_result: Optional[Dict] = None


# ============================================================
# DECISION ENGINE
# ============================================================

class ResponseDecisionEngine:
    """
    Enterprise Response Decision Engine
    
    Evaluates detections and determines appropriate response:
    1. Check policy rules
    2. Evaluate confidence threshold
    3. Assess impact level
    4. Auto-execute or request analyst approval
    """
    
    def __init__(self):
        self.policies: Dict[str, PolicyRule] = {}
        self.approval_queue: Dict[str, ApprovalRequest] = {}
        self.decision_history: List[DecisionResult] = []
        self.enforcement_callbacks: Dict[ResponseAction, callable] = {}
        
        # Initialize default policies
        self._init_default_policies()
        
        # Register enforcement handlers
        self._init_enforcement_handlers()
    
    def _init_default_policies(self):
        """Initialize enterprise-grade default policies"""
        
        # Policy 1: Ransomware - Auto-isolate immediately
        self.policies["ransomware_auto"] = PolicyRule(
            id="ransomware_auto",
            name="Ransomware Auto-Response",
            description="Automatically isolate hosts showing ransomware behavior",
            min_confidence=0.90,
            allowed_severities=["critical"],
            allowed_techniques=["T1486", "T1490", "T1489"],  # Ransomware TTPs
            max_impact_level=ImpactLevel.HIGH,
            allowed_actions=[ResponseAction.ISOLATE_HOST, ResponseAction.KILL_PROCESS],
            require_approval=True,  # HUMAN APPROVAL REQUIRED
            enabled=True
        )
        
        # Policy 2: C2 Communication - Auto-block
        self.policies["c2_block"] = PolicyRule(
            id="c2_block",
            name="C2 Auto-Block",
            description="Automatically block C2 communication IPs",
            min_confidence=0.85,
            allowed_severities=["critical", "high"],
            allowed_techniques=["T1071", "T1573", "T1095"],
            max_impact_level=ImpactLevel.LOW,
            allowed_actions=[ResponseAction.BLOCK_IP],
            require_approval=True,  # HUMAN APPROVAL REQUIRED
            enabled=True
        )
        
        # Policy 3: Credential Theft - Require approval
        self.policies["credential_theft"] = PolicyRule(
            id="credential_theft",
            name="Credential Theft Response",
            description="Disable user accounts showing credential theft",
            min_confidence=0.80,
            allowed_severities=["critical", "high"],
            allowed_techniques=["T1003", "T1558", "T1552"],
            max_impact_level=ImpactLevel.MEDIUM,
            allowed_actions=[ResponseAction.DISABLE_USER],
            require_approval=True,  # NEEDS APPROVAL
            auto_approve_after_mins=15,
            enabled=True
        )
        
        # Policy 4: General Threats - Always require approval
        self.policies["general_threat"] = PolicyRule(
            id="general_threat",
            name="General Threat Response",
            description="Default policy requiring analyst approval",
            min_confidence=0.70,
            allowed_severities=["critical", "high", "medium"],
            max_impact_level=ImpactLevel.HIGH,
            allowed_actions=[ResponseAction.ISOLATE_HOST, ResponseAction.BLOCK_IP],
            require_approval=True,
            enabled=True
        )
    
    def _init_enforcement_handlers(self):
        """Register actual enforcement callbacks"""
        
        async def isolate_host(entity_id: str, context: Dict) -> Dict:
            # In production: Call EDR API, NAC, or firewall
            print(f"🔒 ENFORCING: Isolating host {entity_id}")
            # Example: await edr_client.isolate(entity_id)
            return {"success": True, "action": "isolate_host", "entity": entity_id}
        
        async def block_ip(ip: str, context: Dict) -> Dict:
            # In production: Call firewall API
            print(f"🚫 ENFORCING: Blocking IP {ip}")
            # Example: await firewall_client.block_ip(ip)
            return {"success": True, "action": "block_ip", "ip": ip}
        
        async def disable_user(user_id: str, context: Dict) -> Dict:
            # In production: Call Active Directory API
            print(f"👤 ENFORCING: Disabling user {user_id}")
            # Example: await ad_client.disable_account(user_id)
            return {"success": True, "action": "disable_user", "user": user_id}
        
        async def kill_process(entity_id: str, context: Dict) -> Dict:
            print(f"💀 ENFORCING: Killing process on {entity_id}")
            return {"success": True, "action": "kill_process", "entity": entity_id}
        
        self.enforcement_callbacks = {
            ResponseAction.ISOLATE_HOST: isolate_host,
            ResponseAction.BLOCK_IP: block_ip,
            ResponseAction.DISABLE_USER: disable_user,
            ResponseAction.KILL_PROCESS: kill_process,
        }
    
    # --------------------------------------------------------
    # MAIN DECISION FLOW
    # --------------------------------------------------------
    
    async def evaluate(self, detection: Dict) -> DecisionResult:
        """
        Main entry point: Evaluate detection and decide response
        
        Flow:
        1. Extract detection details
        2. Find matching policy
        3. Check confidence threshold
        4. Assess impact
        5. Auto-execute or queue for approval
        """
        detection_id = detection.get("id", str(uuid.uuid4()))
        entity_id = detection.get("entity_id", "unknown")
        severity = detection.get("severity", "medium")
        confidence = detection.get("confidence_score", 0.5)
        technique_id = detection.get("technique_id")
        detection_type = detection.get("detection_type", "unknown")
        
        print(f"\n{'='*60}")
        print(f"🔍 DECISION ENGINE: Evaluating detection {detection_id}")
        print(f"   Entity: {entity_id} | Severity: {severity} | Confidence: {confidence:.2f}")
        
        # Step 1: Find matching policy
        policy = self._find_matching_policy(severity, confidence, technique_id)
        
        if not policy:
            print(f"   ⚪ No matching policy - no action required")
            return DecisionResult(
                detection_id=detection_id,
                entity_id=entity_id,
                action=ResponseAction.NONE,
                auto_execute=False,
                requires_approval=False,
                confidence=confidence,
                impact_level="low",
                decision_reason="No matching policy rule"
            )
        
        print(f"   📋 Matched policy: {policy.name}")
        
        # Step 2: Determine action
        action = self._determine_action(detection_type, policy)
        
        # Step 3: Assess impact
        impact = self._assess_impact(entity_id, action)
        print(f"   📊 Impact assessment: {impact.value}")
        
        # Step 4: Check if impact exceeds policy threshold
        if self._impact_exceeds_threshold(impact, policy.max_impact_level):
            print(f"   ⚠️ Impact too high for auto-response")
            policy = self.policies.get("general_threat", policy)
        
        # Step 5: Decision - Auto or Approval?
        if not policy.require_approval and confidence >= policy.min_confidence:
            # AUTO-EXECUTE
            print(f"   ✅ AUTO-EXECUTING: {action.value}")
            
            result = DecisionResult(
                detection_id=detection_id,
                entity_id=entity_id,
                action=action,
                auto_execute=True,
                requires_approval=False,
                confidence=confidence,
                impact_level=impact.value,
                policy_matched=policy.id,
                decision_reason=f"Auto-approved by policy '{policy.name}'"
            )
            
            # Execute immediately
            exec_result = await self._execute_action(action, entity_id, detection)
            result.executed = True
            result.execution_result = exec_result
            
        else:
            # REQUIRES APPROVAL
            print(f"   ⏳ QUEUING FOR APPROVAL: {action.value}")
            
            approval_id = await self._create_approval_request(
                detection_id, entity_id, action, confidence, impact, policy
            )
            
            result = DecisionResult(
                detection_id=detection_id,
                entity_id=entity_id,
                action=action,
                auto_execute=False,
                requires_approval=True,
                approval_id=approval_id,
                confidence=confidence,
                impact_level=impact.value,
                policy_matched=policy.id,
                decision_reason=f"Requires analyst approval per policy '{policy.name}'"
            )
        
        self.decision_history.append(result)
        print(f"{'='*60}\n")
        
        return result
    
    def _find_matching_policy(self, severity: str, confidence: float, 
                              technique_id: Optional[str]) -> Optional[PolicyRule]:
        """Find the best matching policy for this detection"""
        
        for policy in self.policies.values():
            if not policy.enabled:
                continue
            
            # Check severity match
            if severity not in policy.allowed_severities:
                continue
            
            # Check technique match (if specified)
            if policy.allowed_techniques and technique_id:
                if technique_id not in policy.allowed_techniques:
                    continue
            
            # Check confidence threshold
            if confidence < policy.min_confidence:
                continue
            
            return policy
        
        return None
    
    def _determine_action(self, detection_type: str, policy: PolicyRule) -> ResponseAction:
        """Determine the appropriate action based on detection type"""
        
        action_map = {
            "ransomware": ResponseAction.ISOLATE_HOST,
            "c2_communication": ResponseAction.BLOCK_IP,
            "credential_theft": ResponseAction.DISABLE_USER,
            "lateral_movement": ResponseAction.ISOLATE_HOST,
            "data_exfiltration": ResponseAction.BLOCK_IP,
            "malware": ResponseAction.KILL_PROCESS,
        }
        
        suggested = action_map.get(detection_type, ResponseAction.ISOLATE_HOST)
        
        # Ensure action is allowed by policy
        if suggested in policy.allowed_actions:
            return suggested
        
        return policy.allowed_actions[0] if policy.allowed_actions else ResponseAction.NONE
    
    def _assess_impact(self, entity_id: str, action: ResponseAction) -> ImpactLevel:
        """Assess the business impact of the proposed action"""
        
        # In production: Query CMDB, check if entity is critical
        critical_systems = ["dc01", "fileserver", "exchange", "database"]
        vip_users = ["ceo", "cfo", "admin"]
        
        entity_lower = entity_id.lower()
        
        if any(sys in entity_lower for sys in critical_systems):
            return ImpactLevel.CRITICAL
        
        if any(vip in entity_lower for vip in vip_users):
            return ImpactLevel.HIGH
        
        if action == ResponseAction.DISABLE_USER:
            return ImpactLevel.MEDIUM
        
        return ImpactLevel.LOW
    
    def _impact_exceeds_threshold(self, actual: ImpactLevel, max_allowed: ImpactLevel) -> bool:
        """Check if actual impact exceeds policy threshold"""
        levels = [ImpactLevel.LOW, ImpactLevel.MEDIUM, ImpactLevel.HIGH, ImpactLevel.CRITICAL]
        return levels.index(actual) > levels.index(max_allowed)
    
    async def _execute_action(self, action: ResponseAction, entity_id: str, 
                              context: Dict) -> Dict:
        """Execute the enforcement action"""
        
        handler = self.enforcement_callbacks.get(action)
        if handler:
            return await handler(entity_id, context)
        
        return {"success": False, "error": "No handler for action"}
    
    async def _create_approval_request(self, detection_id: str, entity_id: str,
                                       action: ResponseAction, confidence: float,
                                       impact: ImpactLevel, policy: PolicyRule) -> str:
        """Create approval request for analyst review"""
        
        approval = ApprovalRequest(
            id=str(uuid.uuid4()),
            detection_id=detection_id,
            entity_id=entity_id,
            proposed_action=action,
            reason=f"Detection triggered policy: {policy.name}",
            confidence=confidence,
            impact_level=impact,
            policy_id=policy.id
        )
        
        self.approval_queue[approval.id] = approval
        
        # In production: Send notification to SOC
        print(f"   📧 Approval request created: {approval.id}")
        
        return approval.id
    
    # --------------------------------------------------------
    # ANALYST WORKFLOW
    # --------------------------------------------------------
    
    async def approve(self, approval_id: str, analyst: str, 
                      notes: Optional[str] = None) -> Dict:
        """Analyst approves the action"""
        
        approval = self.approval_queue.get(approval_id)
        if not approval:
            return {"success": False, "error": "Approval request not found"}
        
        approval.status = ApprovalStatus.APPROVED
        approval.reviewed_at = datetime.utcnow()
        approval.reviewed_by = analyst
        approval.reviewer_notes = notes
        
        # Execute the action
        result = await self._execute_action(
            approval.proposed_action,
            approval.entity_id,
            {"approval_id": approval_id, "analyst": analyst}
        )
        
        print(f"✅ Approval {approval_id} approved by {analyst}")
        return {"success": True, "execution": result}
    
    async def reject(self, approval_id: str, analyst: str,
                     reason: str) -> Dict:
        """Analyst rejects the action"""
        
        approval = self.approval_queue.get(approval_id)
        if not approval:
            return {"success": False, "error": "Approval request not found"}
        
        approval.status = ApprovalStatus.REJECTED
        approval.reviewed_at = datetime.utcnow()
        approval.reviewed_by = analyst
        approval.reviewer_notes = reason
        
        print(f"❌ Approval {approval_id} rejected by {analyst}: {reason}")
        return {"success": True, "status": "rejected"}
    
    def get_pending_approvals(self) -> List[Dict]:
        """Get all pending approval requests"""
        
        return [
            {
                "id": a.id,
                "detection_id": a.detection_id,
                "entity_id": a.entity_id,
                "action": a.proposed_action.value,
                "confidence": a.confidence,
                "impact": a.impact_level.value,
                "created_at": a.created_at.isoformat(),
                "reason": a.reason
            }
            for a in self.approval_queue.values()
            if a.status == ApprovalStatus.PENDING
        ]
    
    def get_policies(self) -> List[Dict]:
        """Get all configured policies"""
        
        return [
            {
                "id": p.id,
                "name": p.name,
                "description": p.description,
                "min_confidence": p.min_confidence,
                "require_approval": p.require_approval,
                "enabled": p.enabled
            }
            for p in self.policies.values()
        ]


# Global instance
decision_engine = ResponseDecisionEngine()


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

async def evaluate_detection(detection: Dict) -> DecisionResult:
    """Evaluate a detection and return decision"""
    return await decision_engine.evaluate(detection)


async def approve_action(approval_id: str, analyst: str, notes: str = None) -> Dict:
    """Approve a pending action"""
    return await decision_engine.approve(approval_id, analyst, notes)


async def reject_action(approval_id: str, analyst: str, reason: str) -> Dict:
    """Reject a pending action"""
    return await decision_engine.reject(approval_id, analyst, reason)


def get_pending_approvals() -> List[Dict]:
    """Get pending approval requests"""
    return decision_engine.get_pending_approvals()
