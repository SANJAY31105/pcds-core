"""
SOAR API - Security Orchestration, Automation, and Response
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Optional
from enum import Enum

router = APIRouter(prefix="/soar", tags=["SOAR"])


class SeverityEnum(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"
    info = "info"


class CreateIncidentRequest(BaseModel):
    """Create a new security incident"""
    title: str
    description: str
    severity: SeverityEnum
    source: str = "manual"
    affected_hosts: List[str] = []
    affected_users: List[str] = []
    iocs: List[Dict] = []
    ml_prediction: Optional[Dict] = None


class UpdateIncidentRequest(BaseModel):
    """Update incident status"""
    status: str = None
    assigned_to: str = None
    notes: str = None


class ExecuteActionRequest(BaseModel):
    """Execute a manual action"""
    action_type: str
    target: str
    parameters: Dict = {}


@router.post("/incidents")
async def create_incident(request: CreateIncidentRequest, background_tasks: BackgroundTasks):
    """
    Create a new security incident
    
    Automatically triggers triage and matching playbooks.
    """
    from ml.soar_orchestrator import get_soar, IncidentSeverity
    import asyncio
    
    soar = get_soar()
    
    # Map severity
    severity = IncidentSeverity(request.severity.value)
    
    # Create incident
    incident = await soar.create_incident(
        title=request.title,
        description=request.description,
        severity=severity,
        source=request.source,
        ml_prediction=request.ml_prediction,
        affected_hosts=request.affected_hosts,
        affected_users=request.affected_users,
        iocs=request.iocs
    )
    
    return {
        "status": "created",
        "incident_id": incident.incident_id,
        "severity": incident.severity.value,
        "playbook_triggered": incident.playbook_id,
        "actions_taken": len(incident.actions_taken)
    }


@router.get("/incidents")
async def list_incidents(status: str = None, limit: int = 50):
    """List all security incidents"""
    from ml.soar_orchestrator import get_soar
    
    soar = get_soar()
    incidents = soar.list_incidents(status=status, limit=limit)
    
    return {
        "total": len(incidents),
        "incidents": incidents
    }


@router.get("/incidents/{incident_id}")
async def get_incident(incident_id: str):
    """Get incident details"""
    from ml.soar_orchestrator import get_soar
    from dataclasses import asdict
    
    soar = get_soar()
    incident = soar.get_incident(incident_id)
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    return asdict(incident)


@router.patch("/incidents/{incident_id}")
async def update_incident(incident_id: str, request: UpdateIncidentRequest):
    """Update incident status or assignment"""
    from ml.soar_orchestrator import get_soar, IncidentStatus
    from datetime import datetime
    
    soar = get_soar()
    incident = soar.get_incident(incident_id)
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    if request.status:
        try:
            incident.status = IncidentStatus(request.status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {request.status}")
    
    if request.assigned_to:
        incident.assigned_to = request.assigned_to
    
    if request.notes:
        incident.timeline.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "Note added",
            "details": {"note": request.notes}
        })
    
    incident.updated_at = datetime.utcnow().isoformat()
    
    return {"status": "updated", "incident_id": incident_id}


@router.post("/incidents/{incident_id}/actions")
async def execute_action(incident_id: str, request: ExecuteActionRequest):
    """Execute a manual action on an incident"""
    from ml.soar_orchestrator import get_soar, SOARAction, ActionType
    
    soar = get_soar()
    incident = soar.get_incident(incident_id)
    
    if not incident:
        raise HTTPException(status_code=404, detail="Incident not found")
    
    try:
        action_type = ActionType(request.action_type)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid action type: {request.action_type}")
    
    action = SOARAction(
        action_id=str(uuid.uuid4())[:8],
        action_type=action_type,
        target=request.target,
        parameters=request.parameters,
        status="pending"
    )
    
    await soar._execute_action(incident, action)
    
    return {
        "status": action.status,
        "action_id": action.action_id,
        "result": action.result
    }


@router.get("/playbooks")
async def list_playbooks():
    """List all response playbooks"""
    from ml.soar_orchestrator import get_soar
    from dataclasses import asdict
    
    soar = get_soar()
    
    playbooks = []
    for pb in soar.playbooks.values():
        playbooks.append({
            "playbook_id": pb.playbook_id,
            "name": pb.name,
            "description": pb.description,
            "enabled": pb.enabled,
            "trigger_conditions": pb.trigger_conditions,
            "action_count": len(pb.actions),
            "execution_count": pb.execution_count
        })
    
    return {"playbooks": playbooks}


@router.post("/playbooks/{playbook_id}/toggle")
async def toggle_playbook(playbook_id: str, enabled: bool):
    """Enable or disable a playbook"""
    from ml.soar_orchestrator import get_soar
    
    soar = get_soar()
    
    if playbook_id not in soar.playbooks:
        raise HTTPException(status_code=404, detail="Playbook not found")
    
    soar.playbooks[playbook_id].enabled = enabled
    
    return {
        "playbook_id": playbook_id,
        "enabled": enabled
    }


@router.get("/stats")
async def get_soar_stats():
    """Get SOAR statistics"""
    from ml.soar_orchestrator import get_soar
    
    soar = get_soar()
    return soar.get_stats()


@router.post("/simulate/ml-detection")
async def simulate_ml_detection(attack_type: str = "DDoS", confidence: float = 0.95):
    """
    Simulate an ML detection to trigger SOAR response
    
    Useful for testing playbook execution.
    """
    from ml.soar_orchestrator import get_soar, IncidentSeverity
    
    soar = get_soar()
    
    # Determine severity based on attack type
    severity_map = {
        "ransomware": IncidentSeverity.CRITICAL,
        "ddos": IncidentSeverity.HIGH,
        "DDoS": IncidentSeverity.HIGH,
        "DoS Hulk": IncidentSeverity.HIGH,
        "FTP-Patator": IncidentSeverity.MEDIUM,
        "SSH-Patator": IncidentSeverity.MEDIUM,
        "Web Attack - SQL Injection": IncidentSeverity.HIGH,
        "PortScan": IncidentSeverity.LOW
    }
    
    severity = severity_map.get(attack_type, IncidentSeverity.MEDIUM)
    
    ml_prediction = {
        "class_name": attack_type,
        "confidence": confidence,
        "model": "hybrid_ensemble"
    }
    
    incident = await soar.create_incident(
        title=f"{attack_type} Attack Detected",
        description=f"ML model detected {attack_type} with {confidence*100:.1f}% confidence",
        severity=severity,
        source="ml_ensemble",
        ml_prediction=ml_prediction,
        affected_hosts=["192.168.1.100"],
        iocs=[{"type": "ip", "value": "45.33.32.156"}]
    )
    
    return {
        "status": "simulated",
        "incident_id": incident.incident_id,
        "attack_type": attack_type,
        "playbook_triggered": incident.playbook_id,
        "actions_executed": len([a for a in incident.actions_taken if a.get("status") == "completed"])
    }


# Import uuid at module level
import uuid
