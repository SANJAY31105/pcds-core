"""
PCDS Enterprise API v2 - Detections Endpoints
Detection feed, filtering, and management with ML v3.0 integration
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
import uuid

from config.database import DetectionQueries, EntityQueries, db_manager
from engine import enrich_detection_with_mitre

# Import Advanced ML Engine v3.0
try:
    from ml.advanced_detector import get_advanced_engine
    ML_ENGINE_AVAILABLE = True
except ImportError:
    ML_ENGINE_AVAILABLE = False
    print("⚠️ ML Engine v3.0 not available")

router = APIRouter(prefix="/detections", tags=["detections"])


@router.get("")
async def get_detections(
    limit: int = Query(50, ge=1, le=10000),  # Increased for demo
    severity: Optional[str] = Query(None, regex="^(critical|high|medium|low)$"),
    entity_id: Optional[str] = None,
    technique_id: Optional[str] = None,
    hours: int = Query(24, ge=1, le=8760)  # Up to 1 year
):
    """
    Get detections with filtering
    
    Query params:
    - limit: Max detections to return
    - severity: Filter by severity
    - entity_id: Filter by entity
    - technique_id: Filter by MITRE technique
    - hours: Time range (1-168 hours)
    """
    if entity_id:
        detections = DetectionQueries.get_by_entity(entity_id, limit=limit)
    elif severity:
        detections = DetectionQueries.get_recent(limit=limit, severity=severity)
    else:
        detections = DetectionQueries.get_recent(limit=limit)
    
    # For the AI attack simulation demo, show ALL detections without time filtering
    # (The simulation spread attacks over 7-8 days)
    # In production, you can re-enable time filtering
    
    # detections already contains the data, no need to filter by time for demo
    filtered_detections = detections[:limit]  # Just limit the count
    
    detections = filtered_detections
    
    # Filter by technique if specified
    if technique_id:
        detections = [d for d in detections if d.get('technique_id') == technique_id]
    
    return {
        "total": len(detections),
        "filters": {
            "severity": severity,
            "entity_id": entity_id,
            "technique_id": technique_id,
            "hours": hours
        },
        "detections": detections
    }


# ============================================================
# ML ENGINE v3.0 ENDPOINTS (must be before /{detection_id})
# ============================================================

@router.get("/engine-status")
async def get_engine_status():
    """Get ML Engine v3.0 status and performance metrics"""
    if not ML_ENGINE_AVAILABLE:
        return {"status": "unavailable", "message": "ML Engine v3.0 not installed"}
    
    engine = get_advanced_engine()
    if not engine:
        return {"status": "error", "message": "ML Engine failed to initialize"}
    
    stats = engine.get_performance_stats()
    return {
        "status": "operational",
        "engine": stats,
        "capabilities": {
            "transformer": "Multi-head attention for sequence analysis",
            "lstm": "Bidirectional with temporal patterns",
            "graph_nn": "Attack chain and lateral movement",
            "explainable_ai": "Human-readable explanations"
        }
    }


@router.post("/analyze")
async def analyze_with_ml(data: dict):
    """Analyze detection data with ML Engine v3.0"""
    if not ML_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML Engine v3.0 not available")
    
    engine = get_advanced_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="ML Engine failed to initialize")
    
    entities = data.pop('entities', None)
    attack_type = data.pop('attack_type_hint', None)
    entity_id = data.get('entity_id')
    
    result = engine.detect(data=data, entity_id=entity_id, entities=entities, attack_type=attack_type)
    return {"analysis": result, "engine_version": engine.VERSION, "models_used": ["transformer", "lstm", "graph_nn", "statistical"]}


@router.get("/{detection_id}")
async def get_detection(detection_id: str):
    """Get detailed information about a specific detection"""
    detection = db_manager.execute_one("""
        SELECT d.*, 
               e.identifier as entity_identifier,
               e.entity_type,
               t.name as technique_name,
               tc.name as tactic_name
        FROM detections d
        LEFT JOIN entities e ON d.entity_id = e.id
        LEFT JOIN mitre_techniques t ON d.technique_id = t.id
        LEFT JOIN mitre_tactics tc ON d.tactic_id = tc.id
        WHERE d.id = ?
    """, (detection_id,))
    
    if not detection:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")
    
    return detection


@router.post("")
async def create_detection(detection_data: dict):
    """
    Create a new detection
    
    Automatically enriches with MITRE context and updates entity score
    """
    # Generate ID if not provided
    if 'id' not in detection_data:
        detection_data['id'] = f"det_{uuid.uuid4().hex[:12]}"
    
    # Set timestamp if not provided
    if 'detected_at' not in detection_data:
        detection_data['detected_at'] = datetime.utcnow().isoformat()
    
    # Enrich with MITRE
    enriched = enrich_detection_with_mitre(detection_data)
    
    # Ensure required fields
    if 'entity_id' not in enriched:
        raise HTTPException(status_code=400, detail="entity_id is required")
    if 'detection_type' not in enriched:
        raise HTTPException(status_code=400, detail="detection_type is required")
    if 'severity' not in enriched:
        enriched['severity'] = 'medium'
    if 'confidence_score' not in enriched:
        enriched['confidence_score'] = 0.7
    if 'risk_score' not in enriched:
        enriched['risk_score'] = 50
    
    # Store detection
    detection_id = DetectionQueries.create(enriched)
    
    # Update entity detection count
    EntityQueries.increment_detection_count(
        enriched['entity_id'],
        enriched['severity']
    )
    
    return {
        "detection_id": detection_id,
        "detection": enriched,
        "message": "Detection created successfully"
    }


@router.patch("/{detection_id}/status")
async def update_detection_status(
    detection_id: str,
    status: str = Query(..., regex="^(new|investigating|resolved|false_positive)$"),
    assigned_to: Optional[str] = None
):
    """Update detection status"""
    detection = db_manager.execute_one(
        "SELECT * FROM detections WHERE id = ?",
        (detection_id,)
    )
    
    if not detection:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")
    
    # Update status
    query = "UPDATE detections SET status = ?, updated_at = ?"
    params = [status, datetime.utcnow()]
    
    if assigned_to:
        query += ", assigned_to = ?"
        params.append(assigned_to)
    
    query += " WHERE id = ?"
    params.append(detection_id)
    
    db_manager.execute_update(query, tuple(params))
    
    return {
        "detection_id": detection_id,
        "status": status,
        "assigned_to": assigned_to,
        "message": "Status updated successfully"
    }


@router.get("/stats/severity-breakdown")
async def get_severity_breakdown(hours: int = Query(24, ge=1, le=168)):
    """Get detection count by severity for last N hours"""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    stats = db_manager.execute_query("""
        SELECT 
            severity,
            COUNT(*) as count
        FROM detections
        WHERE detected_at > ?
        GROUP BY severity
    """, (cutoff_time.isoformat(),))
    
    return {
        "time_range_hours": hours,
        "severity_breakdown": {s['severity']: s['count'] for s in stats}
    }


@router.get("/stats/technique-frequency")
async def get_technique_frequency(
    limit: int = Query(10, ge=1, le=50),
    hours: int = Query(24, ge=1, le=168)
):
    """Get most frequently detected techniques"""
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    
    stats = db_manager.execute_query("""
        SELECT 
            d.technique_id,
            t.name as technique_name,
            COUNT(*) as count
        FROM detections d
        LEFT JOIN mitre_techniques t ON d.technique_id = t.id
        WHERE d.detected_at > ? AND d.technique_id IS NOT NULL
        GROUP BY d.technique_id, t.name
        ORDER BY count DESC
        LIMIT ?
    """, (cutoff_time.isoformat(), limit))
    
    return {
        "time_range_hours": hours,
        "top_techniques": stats
    }


# ============================================================
# ML ENGINE v3.0 ENDPOINTS
# ============================================================

@router.post("/analyze")
async def analyze_with_ml(data: dict):
    """
    Analyze detection data with ML Engine v3.0
    
    Returns:
    - Anomaly score from 4-model ensemble
    - Risk level classification
    - Explainable AI breakdown
    - Recommended actions
    """
    if not ML_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML Engine v3.0 not available")
    
    engine = get_advanced_engine()
    if not engine:
        raise HTTPException(status_code=503, detail="ML Engine failed to initialize")
    
    # Get entities for graph analysis if provided
    entities = data.pop('entities', None)
    attack_type = data.pop('attack_type_hint', None)
    entity_id = data.get('entity_id')
    
    # Run ML analysis
    result = engine.detect(
        data=data,
        entity_id=entity_id,
        entities=entities,
        attack_type=attack_type
    )
    
    return {
        "analysis": result,
        "engine_version": engine.VERSION,
        "models_used": ["transformer", "lstm", "graph_nn", "statistical"]
    }


@router.get("/engine-status")
async def get_engine_status():
    """
    Get ML Engine v3.0 status and performance metrics
    """
    if not ML_ENGINE_AVAILABLE:
        return {
            "status": "unavailable",
            "message": "ML Engine v3.0 not installed"
        }
    
    engine = get_advanced_engine()
    if not engine:
        return {
            "status": "error",
            "message": "ML Engine failed to initialize"
        }
    
    stats = engine.get_performance_stats()
    
    return {
        "status": "operational",
        "engine": stats,
        "capabilities": {
            "transformer": "Multi-head attention for sequence analysis",
            "lstm": "Bidirectional with temporal patterns",
            "graph_nn": "Attack chain and lateral movement",
            "explainable_ai": "Human-readable explanations"
        }
    }


@router.post("/{detection_id}/explain")
async def explain_detection(detection_id: str):
    """
    Get detailed ML explanation for a specific detection
    """
    if not ML_ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="ML Engine v3.0 not available")
    
    # Get detection from database
    detection = db_manager.execute_one(
        "SELECT * FROM detections WHERE id = ?",
        (detection_id,)
    )
    
    if not detection:
        raise HTTPException(status_code=404, detail=f"Detection {detection_id} not found")
    
    engine = get_advanced_engine()
    
    # Run analysis on this detection
    result = engine.detect(
        data=dict(detection),
        entity_id=detection.get('entity_id')
    )
    
    return {
        "detection_id": detection_id,
        "explanation": result['explanation'],
        "model_contributions": result['model_contributions'],
        "inference_time_ms": result['inference_time_ms']
    }
