"""
PCDS Enterprise API - Response Decision Engine
Endpoints for automated response, approvals, and policy management
"""

from fastapi import APIRouter, HTTPException
from typing import Optional, Dict, List
from pydantic import BaseModel
from datetime import datetime


router = APIRouter(prefix="/response", tags=["response-automation"])


# Request models  
class DetectionTrigger(BaseModel):
    detection_id: Optional[str] = None
    entity_id: str
    detection_type: str
    severity: str = "high"
    confidence_score: float = 0.8
    technique_id: Optional[str] = None
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None


class ApprovalAction(BaseModel):
    analyst: str
    notes: Optional[str] = None


class RejectionAction(BaseModel):
    analyst: str
    reason: str


# ============================================================
# DECISION ENGINE ENDPOINTS
# ============================================================

@router.post("/evaluate")
async def evaluate_detection(trigger: DetectionTrigger):
    """
    Evaluate a detection through the decision engine
    
    Flow: Policy Check → Confidence → Impact → Auto/Manual
    """
    from automation.decision_engine import decision_engine
    
    detection = trigger.dict()
    detection["id"] = trigger.detection_id or f"det-{datetime.utcnow().timestamp()}"
    
    result = await decision_engine.evaluate(detection)
    
    return {
        "detection_id": result.detection_id,
        "entity_id": result.entity_id,
        "action": result.action.value,
        "auto_executed": result.auto_execute,
        "requires_approval": result.requires_approval,
        "approval_id": result.approval_id,
        "confidence": result.confidence,
        "impact_level": result.impact_level,
        "policy_matched": result.policy_matched,
        "decision_reason": result.decision_reason,
        "executed": result.executed,
        "execution_result": result.execution_result
    }


@router.get("/status")
async def get_engine_status():
    """Get decision engine status and metrics"""
    from automation.decision_engine import decision_engine
    
    pending = decision_engine.get_pending_approvals()
    
    return {
        "status": "operational",
        "policies_loaded": len(decision_engine.policies),
        "pending_approvals": len(pending),
        "total_decisions": len(decision_engine.decision_history),
        "enforcement_handlers": list(decision_engine.enforcement_callbacks.keys())
    }


# ============================================================
# APPROVAL WORKFLOW
# ============================================================

@router.get("/approvals")
async def get_pending_approvals():
    """Get all pending approval requests"""
    from automation.decision_engine import decision_engine
    
    return {
        "pending": decision_engine.get_pending_approvals(),
        "count": len(decision_engine.get_pending_approvals())
    }


@router.post("/approvals/{approval_id}/approve")
async def approve_action(approval_id: str, action: ApprovalAction):
    """Approve a pending response action"""
    from automation.decision_engine import decision_engine
    
    result = await decision_engine.approve(approval_id, action.analyst, action.notes)
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))
    
    return {
        "status": "approved",
        "approval_id": approval_id,
        "approved_by": action.analyst,
        "execution": result.get("execution")
    }


@router.post("/approvals/{approval_id}/reject")
async def reject_action(approval_id: str, action: RejectionAction):
    """Reject a pending response action"""
    from automation.decision_engine import decision_engine
    
    result = await decision_engine.reject(approval_id, action.analyst, action.reason)
    
    if not result.get("success"):
        raise HTTPException(status_code=404, detail=result.get("error"))
    
    return {
        "status": "rejected",
        "approval_id": approval_id,
        "rejected_by": action.analyst,
        "reason": action.reason
    }


# ============================================================
# POLICY MANAGEMENT
# ============================================================

@router.get("/policies")
async def get_policies():
    """Get all response policies"""
    from automation.decision_engine import decision_engine
    
    return {
        "policies": decision_engine.get_policies(),
        "count": len(decision_engine.policies)
    }


@router.get("/policies/{policy_id}")
async def get_policy(policy_id: str):
    """Get specific policy details"""
    from automation.decision_engine import decision_engine
    
    policy = decision_engine.policies.get(policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    
    return {
        "id": policy.id,
        "name": policy.name,
        "description": policy.description,
        "min_confidence": policy.min_confidence,
        "allowed_severities": policy.allowed_severities,
        "allowed_techniques": policy.allowed_techniques,
        "max_impact_level": policy.max_impact_level.value,
        "allowed_actions": [a.value for a in policy.allowed_actions],
        "require_approval": policy.require_approval,
        "auto_approve_after_mins": policy.auto_approve_after_mins,
        "enabled": policy.enabled
    }


@router.patch("/policies/{policy_id}/toggle")
async def toggle_policy(policy_id: str):
    """Enable/disable a policy"""
    from automation.decision_engine import decision_engine
    
    policy = decision_engine.policies.get(policy_id)
    if not policy:
        raise HTTPException(status_code=404, detail="Policy not found")
    
    policy.enabled = not policy.enabled
    
    return {
        "policy_id": policy_id,
        "enabled": policy.enabled
    }


# ============================================================
# DECISION HISTORY
# ============================================================

@router.get("/history")
async def get_decision_history(limit: int = 50):
    """Get recent decision history"""
    from automation.decision_engine import decision_engine
    
    history = decision_engine.decision_history[-limit:]
    
    return {
        "decisions": [
            {
                "detection_id": d.detection_id,
                "entity_id": d.entity_id,
                "action": d.action.value,
                "auto_execute": d.auto_execute,
                "requires_approval": d.requires_approval,
                "policy_matched": d.policy_matched,
                "executed": d.executed
            }
            for d in reversed(history)
        ],
        "total": len(decision_engine.decision_history)
    }
