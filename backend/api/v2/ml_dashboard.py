"""
Live ML Dashboard API
Real-time threat visualization endpoints

Endpoints:
- /ml/status - ML engine status
- /ml/predict - Run inference
- /ml/threats - Live threat feed
- /ml/risk-score - Aggregated risk
- /ml/timeline - Attack timeline data
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
from collections import deque
import asyncio


router = APIRouter(prefix="/ml", tags=["ML Dashboard"])


# In-memory threat storage (for real-time dashboard)
threat_history = deque(maxlen=1000)
risk_history = deque(maxlen=100)


class PredictRequest(BaseModel):
    """Request for ML prediction"""
    features: List[float]
    event_type: str = "unknown"
    metadata: Dict = {}


class ThreatEvent(BaseModel):
    """Threat event for dashboard"""
    id: str
    timestamp: str
    threat_class: str
    confidence: float
    risk_level: str
    action: str
    event_type: str
    details: Dict = {}


class RiskScore(BaseModel):
    """Aggregated risk score"""
    current_score: float
    trend: str  # up, down, stable
    critical_count: int
    high_risk_count: int
    suspicious_count: int
    safe_count: int


class TimelineEvent(BaseModel):
    """Attack timeline event"""
    timestamp: str
    stage: str  # reconnaissance, initial_access, execution, etc.
    threat_class: str
    confidence: float
    related_events: List[str] = []


# Lazy load ML components
_inference_engine = None
_ensemble_engine = None


def get_inference():
    global _inference_engine
    if _inference_engine is None:
        try:
            from ml.inference_engine import get_inference_engine
            _inference_engine = get_inference_engine()
        except Exception as e:
            print(f"⚠️ Inference engine not loaded: {e}")
    return _inference_engine


def get_ensemble():
    global _ensemble_engine
    if _ensemble_engine is None:
        try:
            from ml.ensemble_engine import get_ensemble_engine
            _ensemble_engine = get_ensemble_engine()
        except Exception as e:
            print(f"⚠️ Ensemble engine not loaded: {e}")
    return _ensemble_engine


@router.get("/status")
async def get_ml_status() -> Dict:
    """
    Get ML engine status
    
    Returns model availability and stats
    """
    engine = get_inference()
    ensemble = get_ensemble()
    
    status = {
        "inference_engine": {
            "loaded": engine is not None,
            "device": str(engine.device) if engine else None,
            "stats": engine.get_stats() if engine else None
        },
        "ensemble_engine": {
            "loaded": ensemble is not None,
            "stats": ensemble.get_stats() if ensemble else None
        },
        "threat_history_size": len(threat_history),
        "risk_history_size": len(risk_history)
    }
    
    return status


@router.post("/predict")
async def predict(request: PredictRequest) -> Dict:
    """
    Run ML inference on features
    
    Returns prediction with risk level and action
    """
    engine = get_inference()
    
    if not engine:
        raise HTTPException(status_code=503, detail="ML engine not available")
    
    try:
        features = np.array(request.features, dtype=np.float32)
        result = engine.predict(features)
        
        # Create threat event
        import uuid
        threat_event = ThreatEvent(
            id=str(uuid.uuid4())[:8],
            timestamp=datetime.now().isoformat(),
            threat_class=result.class_name,
            confidence=result.confidence,
            risk_level=result.risk_level.value,
            action=result.action.value,
            event_type=request.event_type,
            details=request.metadata
        )
        
        # Store in history
        threat_history.append(threat_event.dict())
        
        # Update risk history
        risk_history.append({
            "timestamp": datetime.now().isoformat(),
            "confidence": result.confidence,
            "risk_level": result.risk_level.value,
            "class": result.predicted_class
        })
        
        return {
            "prediction": {
                "class": result.predicted_class,
                "class_name": result.class_name,
                "confidence": round(result.confidence, 4),
                "risk_level": result.risk_level.value,
                "action": result.action.value,
                "probabilities": [round(p, 4) for p in result.all_probabilities[:5]]
            },
            "inference_time_ms": round(result.inference_time_ms, 2),
            "threat_event_id": threat_event.id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/threats")
async def get_threats(
    limit: int = Query(default=50, le=200),
    risk_level: Optional[str] = Query(default=None)
) -> Dict:
    """
    Get live threat feed
    
    Returns recent threat events for dashboard
    """
    threats = list(threat_history)
    
    # Filter by risk level
    if risk_level:
        threats = [t for t in threats if t.get("risk_level") == risk_level]
    
    # Sort by timestamp (newest first)
    threats = sorted(threats, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    # Limit
    threats = threats[:limit]
    
    # Summary stats
    all_threats = list(threat_history)
    stats = {
        "total": len(all_threats),
        "critical": len([t for t in all_threats if t.get("risk_level") == "critical"]),
        "high_risk": len([t for t in all_threats if t.get("risk_level") == "high_risk"]),
        "suspicious": len([t for t in all_threats if t.get("risk_level") == "suspicious"]),
        "safe": len([t for t in all_threats if t.get("risk_level") == "safe"])
    }
    
    return {
        "threats": threats,
        "stats": stats,
        "last_updated": datetime.now().isoformat()
    }


@router.get("/risk-score")
async def get_risk_score() -> Dict:
    """
    Get aggregated risk score
    
    Returns overall system risk assessment
    """
    all_threats = list(threat_history)
    recent_risks = list(risk_history)
    
    if not all_threats:
        return {
            "current_score": 0.0,
            "trend": "stable",
            "critical_count": 0,
            "high_risk_count": 0,
            "suspicious_count": 0,
            "safe_count": 0,
            "health": "healthy"
        }
    
    # Count by risk level
    critical = len([t for t in all_threats if t.get("risk_level") == "critical"])
    high_risk = len([t for t in all_threats if t.get("risk_level") == "high_risk"])
    suspicious = len([t for t in all_threats if t.get("risk_level") == "suspicious"])
    safe = len([t for t in all_threats if t.get("risk_level") == "safe"])
    
    # Calculate weighted risk score (0-100)
    total = len(all_threats)
    risk_score = (
        (critical * 100 + high_risk * 70 + suspicious * 30 + safe * 0) / total
    ) if total > 0 else 0
    
    # Determine trend
    if len(recent_risks) >= 10:
        recent_avg = np.mean([r.get("confidence", 0) for r in list(recent_risks)[-10:]])
        older_avg = np.mean([r.get("confidence", 0) for r in list(recent_risks)[-20:-10]]) if len(recent_risks) >= 20 else recent_avg
        
        if recent_avg > older_avg + 0.1:
            trend = "up"
        elif recent_avg < older_avg - 0.1:
            trend = "down"
        else:
            trend = "stable"
    else:
        trend = "stable"
    
    # Health status
    if risk_score >= 70:
        health = "critical"
    elif risk_score >= 40:
        health = "warning"
    else:
        health = "healthy"
    
    return {
        "current_score": round(risk_score, 1),
        "trend": trend,
        "critical_count": critical,
        "high_risk_count": high_risk,
        "suspicious_count": suspicious,
        "safe_count": safe,
        "health": health,
        "last_updated": datetime.now().isoformat()
    }


@router.get("/timeline")
async def get_attack_timeline(
    hours: int = Query(default=24, le=168)
) -> Dict:
    """
    Get attack timeline data
    
    Returns events organized by kill chain stage
    """
    all_threats = list(threat_history)
    
    # Filter by time
    cutoff = datetime.now() - timedelta(hours=hours)
    recent_threats = [
        t for t in all_threats
        if datetime.fromisoformat(t.get("timestamp", "2000-01-01")) > cutoff
    ]
    
    # Map threat classes to kill chain stages
    stage_mapping = {
        "Normal": "normal",
        "DoS/DDoS": "impact",
        "Recon/Scan": "reconnaissance",
        "Brute Force": "initial_access",
        "Web/Exploit": "execution",
        "Infiltration": "initial_access",
        "Botnet": "command_control",
        "Backdoor": "persistence",
        "Worms": "lateral_movement",
        "Fuzzers": "reconnaissance",
        "Other": "unknown"
    }
    
    # Group by stage
    timeline = []
    for threat in recent_threats:
        threat_class = threat.get("threat_class", "Unknown")
        stage = stage_mapping.get(threat_class, "unknown")
        
        if stage != "normal":
            timeline.append({
                "timestamp": threat.get("timestamp"),
                "stage": stage,
                "threat_class": threat_class,
                "confidence": threat.get("confidence", 0),
                "risk_level": threat.get("risk_level", "unknown"),
                "id": threat.get("id")
            })
    
    # Sort by timestamp
    timeline = sorted(timeline, key=lambda x: x.get("timestamp", ""))
    
    # Stage counts
    stage_counts = {}
    for event in timeline:
        stage = event.get("stage", "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    
    return {
        "timeline": timeline[-100:],  # Last 100 events
        "stage_counts": stage_counts,
        "total_events": len(timeline),
        "hours": hours,
        "last_updated": datetime.now().isoformat()
    }


@router.post("/simulate")
async def simulate_threat(
    threat_type: str = Query(default="dos"),
    count: int = Query(default=5, le=50)
) -> Dict:
    """
    Simulate threats for testing dashboard
    
    Creates fake threat events to test visualization
    """
    engine = get_inference()
    
    threat_profiles = {
        "dos": {"features_range": (0.5, 1.0), "expected_class": "DoS/DDoS"},
        "scan": {"features_range": (0.2, 0.6), "expected_class": "Recon/Scan"},
        "brute_force": {"features_range": (0.3, 0.7), "expected_class": "Brute Force"},
        "normal": {"features_range": (0.0, 0.3), "expected_class": "Normal"}
    }
    
    profile = threat_profiles.get(threat_type, threat_profiles["normal"])
    
    simulated = []
    for i in range(count):
        # Generate features based on profile
        low, high = profile["features_range"]
        features = np.random.uniform(low, high, 40).astype(np.float32)
        
        if engine:
            result = engine.predict(features)
            
            import uuid
            event = {
                "id": str(uuid.uuid4())[:8],
                "timestamp": datetime.now().isoformat(),
                "threat_class": result.class_name,
                "confidence": round(result.confidence, 4),
                "risk_level": result.risk_level.value,
                "action": result.action.value,
                "event_type": "simulation",
                "details": {"threat_type": threat_type, "index": i}
            }
            
            threat_history.append(event)
            simulated.append(event)
    
    return {
        "simulated": len(simulated),
        "threat_type": threat_type,
        "events": simulated
    }


@router.delete("/threats")
async def clear_threats() -> Dict:
    """Clear threat history (for testing)"""
    threat_history.clear()
    risk_history.clear()
    return {"cleared": True}
