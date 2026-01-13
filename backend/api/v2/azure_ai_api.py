"""
Azure AI API Endpoints for PCDS
Exposes threat explanation and analyst co-pilot features
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, List
from ml.azure_ai_service import explain_detection, copilot_query, get_azure_ai

router = APIRouter(tags=["Azure AI"])


class CopilotRequest(BaseModel):
    query: str
    context: Optional[Dict] = None


class ExplainRequest(BaseModel):
    detection_id: Optional[str] = None
    detection_type: str = "Unknown"
    severity: str = "medium"
    confidence: float = 0.5
    entity_id: Optional[str] = None
    technique_id: Optional[str] = None
    description: Optional[str] = None


@router.get("/status")
async def get_azure_status():
    """Check Azure AI service status"""
    ai = get_azure_ai()
    return {
        "azure_openai_enabled": ai.enabled,
        "deployment": ai.deployment if ai.enabled else None,
        "features": {
            "threat_explanation": True,
            "analyst_copilot": True,
            "incident_summarization": True
        },
        "fallback_available": True
    }


@router.post("/explain")
async def explain_threat(request: ExplainRequest):
    """
    Get AI-powered explanation for a detection
    
    Returns structured explanation with:
    - Summary
    - Severity reasoning
    - Attack chain
    - Recommended actions
    - MITRE context
    """
    detection = {
        "detection_id": request.detection_id,
        "detection_type": request.detection_type,
        "severity": request.severity,
        "confidence": request.confidence,
        "entity_id": request.entity_id,
        "technique_id": request.technique_id,
        "description": request.description
    }
    
    result = await explain_detection(detection)
    return result


@router.post("/copilot")
async def analyst_copilot(request: CopilotRequest):
    """
    Analyst co-pilot - ask security questions
    
    Uses Azure OpenAI to provide expert security guidance
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    result = await copilot_query(request.query, request.context)
    return result


@router.post("/summarize-incident")
async def summarize_incident(detections: List[Dict]):
    """
    Summarize multiple detections into incident narrative
    """
    if not detections:
        raise HTTPException(status_code=400, detail="No detections provided")
    
    ai = get_azure_ai()
    summary = await ai.summarize_incident(detections)
    
    return {
        "detection_count": len(detections),
        "summary": summary,
        "powered_by": "Azure OpenAI" if ai.enabled else "Local Analysis"
    }
