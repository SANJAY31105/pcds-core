"""
PCDS Enterprise API - Playbooks Endpoints
Automated response playbook management
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Dict, List
from datetime import datetime
from pydantic import BaseModel


router = APIRouter(prefix="/playbooks", tags=["playbooks"])


# Request models
class PlaybookTrigger(BaseModel):
    detection_id: Optional[str] = None
    entity_id: str
    detection_type: str
    technique_id: Optional[str] = None
    severity: str = "high"
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None


# ============================================================
# PLAYBOOK MANAGEMENT
# ============================================================

@router.get("/")
async def list_playbooks():
    """List all available playbooks"""
    from automation.advanced_playbooks import advanced_playbook_engine
    
    return {
        "playbooks": advanced_playbook_engine.get_playbook_list(),
        "total": len(advanced_playbook_engine.playbooks)
    }


@router.get("/{playbook_id}")
async def get_playbook(playbook_id: str):
    """Get specific playbook details"""
    from automation.advanced_playbooks import advanced_playbook_engine
    
    playbook = advanced_playbook_engine.playbooks.get(playbook_id)
    if not playbook:
        raise HTTPException(status_code=404, detail="Playbook not found")
    
    return playbook


@router.post("/trigger")
async def trigger_playbook(trigger: PlaybookTrigger):
    """Manually trigger playbook evaluation"""
    from automation.advanced_playbooks import advanced_playbook_engine
    
    detection = trigger.dict()
    execution = await advanced_playbook_engine.check_and_execute(detection)
    
    if execution:
        return {
            "triggered": True,
            "playbook_id": execution.playbook_id,
            "execution_id": execution.id,
            "actions_completed": execution.actions_completed,
            "status": execution.status.value
        }
    
    return {
        "triggered": False,
        "message": "No matching playbook for this detection"
    }


@router.post("/execute/{playbook_id}")
async def execute_specific_playbook(playbook_id: str, trigger: PlaybookTrigger):
    """Execute a specific playbook"""
    from automation.advanced_playbooks import advanced_playbook_engine
    
    detection = trigger.dict()
    
    try:
        execution = await advanced_playbook_engine.execute_playbook(playbook_id, detection)
        return {
            "success": True,
            "playbook_id": playbook_id,
            "execution_id": execution.id,
            "actions_completed": execution.actions_completed,
            "actions_failed": execution.actions_failed,
            "results": execution.results
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/executions/history")
async def get_execution_history(limit: int = Query(50, ge=1, le=200)):
    """Get playbook execution history"""
    from automation.advanced_playbooks import advanced_playbook_engine
    
    return {
        "executions": advanced_playbook_engine.get_execution_history(limit),
        "total": len(advanced_playbook_engine.executions)
    }


# ============================================================
# MITRE COVERAGE
# ============================================================

@router.get("/mitre/coverage")
async def get_mitre_coverage():
    """Get MITRE ATT&CK technique coverage"""
    from engine.mitre_extended import EXTENDED_TECHNIQUES, get_technique_count
    
    # Count by tactic
    tactic_counts = {}
    for tech_id, tech_data in EXTENDED_TECHNIQUES.items():
        tactic = tech_data.get("tactic", "unknown")
        if tactic not in tactic_counts:
            tactic_counts[tactic] = 0
        tactic_counts[tactic] += 1
    
    return {
        "total_techniques": get_technique_count(),
        "extended_techniques": len(EXTENDED_TECHNIQUES),
        "by_tactic": tactic_counts,
        "coverage_percentage": round((get_technique_count() / 200) * 100, 1),
        "market_comparison": {
            "crowdstrike": 150,
            "darktrace": 120,
            "pcds_enterprise": get_technique_count()
        }
    }


@router.get("/mitre/techniques")
async def list_mitre_techniques(
    tactic: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = Query(100, ge=1, le=500)
):
    """List MITRE techniques"""
    from engine.mitre_extended import EXTENDED_TECHNIQUES
    
    techniques = []
    for tech_id, tech_data in EXTENDED_TECHNIQUES.items():
        if tactic and tech_data.get("tactic") != tactic:
            continue
        if severity and tech_data.get("severity") != severity:
            continue
        
        techniques.append({
            "id": tech_id,
            **tech_data
        })
        
        if len(techniques) >= limit:
            break
    
    return {
        "techniques": techniques,
        "total": len(techniques),
        "filters": {"tactic": tactic, "severity": severity}
    }
