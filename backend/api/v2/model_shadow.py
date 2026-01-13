from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from ml.model_server import get_inference_server, InferenceServer
import numpy as np
import uuid
import json
from config.database import db_manager

router = APIRouter()

# --- Schemas ---

class PredictionRequest(BaseModel):
    features: List[float]
    source_ip: Optional[str] = None
    source_host: Optional[str] = None

class FeedbackRequest(BaseModel):
    prediction_id: str
    is_correct: bool
    true_class: Optional[int] = None
    notes: Optional[str] = None  # Review notes for audit
    escalate: bool = False  # Escalate to investigation

# --- Dependency ---

def get_server():
    return get_inference_server()

# --- Rate Limiting (Simple in-memory) ---
from collections import defaultdict
from datetime import datetime, timedelta
import threading

_feedback_counts: Dict[str, List[datetime]] = defaultdict(list)
_feedback_lock = threading.Lock()
FEEDBACK_RATE_LIMIT = 30  # per minute

def check_rate_limit(client_id: str) -> bool:
    """Check if client has exceeded rate limit"""
    now = datetime.now()
    cutoff = now - timedelta(minutes=1)
    
    with _feedback_lock:
        # Clean old entries
        _feedback_counts[client_id] = [t for t in _feedback_counts[client_id] if t > cutoff]
        
        if len(_feedback_counts[client_id]) >= FEEDBACK_RATE_LIMIT:
            return False
        
        _feedback_counts[client_id].append(now)
        return True

# --- Endpoints ---

@router.post("/predict")
async def shadow_predict(
    request: PredictionRequest, 
    server: InferenceServer = Depends(get_server)
):
    """
    Shadow mode prediction endpoint.
    Logs to Kafka and stores for analysis.
    """
    try:
        features = np.array(request.features, dtype=np.float32)
        result = server.predict(
            features, 
            source_ip=request.source_ip, 
            source_host=request.source_host
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_metrics(server: InferenceServer = Depends(get_server)):
    """Get server performance metrics (QPS, Latency, etc.)"""
    return server.get_metrics()

@router.get("/dashboard")
async def get_dashboard_stats(server: InferenceServer = Depends(get_server)):
    """Get full dashboard statistics including distributions"""
    return server.get_dashboard_data()

@router.get("/pending")
async def get_pending_predictions(
    limit: int = 50,
    min_confidence: float = 0,
    severity: Optional[str] = None,
    source_host: Optional[str] = None,
    server: InferenceServer = Depends(get_server)
):
    """
    Get predictions awaiting analyst review.
    
    Filters:
    - limit: Max predictions to return (default 50)
    - min_confidence: Filter by minimum confidence (0-1)
    - severity: Filter by severity (critical, high, medium, low)
    - source_host: Filter by source host (partial match)
    """
    return server.get_pending_predictions(
        limit=limit,
        min_confidence=min_confidence,
        severity=severity,
        source_host=source_host
    )

@router.post("/feedback")
async def submit_feedback(
    feedback: FeedbackRequest,
    request: Request,
    server: InferenceServer = Depends(get_server)
):
    """
    Submit analyst feedback (Ground Truth).
    
    Security:
    - Rate limited to 30 submissions per minute per IP
    - Requires X-Analyst-ID header for audit trail
    """
    # Get client identifier for rate limiting
    client_ip = request.client.host if request.client else "unknown"
    
    # Check rate limit
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded. Max {FEEDBACK_RATE_LIMIT} feedback submissions per minute."
        )
    
    # Get analyst ID from header (placeholder for full auth)
    analyst_id = request.headers.get("X-Analyst-ID", "analyst@pcds.com")
    
    result = server.submit_feedback(
        prediction_id=feedback.prediction_id,
        is_correct=feedback.is_correct,
        true_class=feedback.true_class,
        reviewed_by=analyst_id,
        notes=feedback.notes
    )
    
    # Handle escalation - Create actual investigation
    if feedback.escalate:
        try:
            # Get prediction details
            prediction = server.store.get(feedback.prediction_id)
            
            if prediction:
                investigation_id = f"inv_{uuid.uuid4().hex[:12]}"
                
                # Create investigation from prediction
                db_manager.execute_insert("""
                    INSERT INTO investigations 
                    (id, title, description, severity, priority, status, assigned_to,
                     assignee_email, entity_ids, detection_ids, campaign_id, opened_at, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    investigation_id,
                    f"ML Detection: {prediction.class_name} from {prediction.source_host or 'Unknown'}",
                    f"""ML Model detected potential threat.
                    
Prediction ID: {prediction.prediction_id}
Confidence: {prediction.confidence:.1%}
Severity: {getattr(prediction, 'severity', 'medium')}
MITRE Technique: {getattr(prediction, 'mitre_technique', 'N/A')}
Source IP: {prediction.source_ip or 'N/A'}
Source Host: {prediction.source_host or 'N/A'}

Analyst Notes: {feedback.notes or 'None provided'}
""",
                    getattr(prediction, 'severity', 'medium'),
                    'high' if getattr(prediction, 'severity', 'medium') in ['critical', 'high'] else 'medium',
                    'open',
                    analyst_id,
                    None,
                    json.dumps([]),
                    json.dumps([feedback.prediction_id]),
                    None,
                    datetime.utcnow().isoformat(),
                    json.dumps(['ml-detection', 'escalated'])
                ))
                
                result["escalated"] = True
                result["investigation_id"] = investigation_id
            else:
                result["escalated"] = False
                result["escalation_error"] = "Prediction not found"
        except Exception as e:
            result["escalated"] = False
            result["escalation_error"] = str(e)
    
    return result

@router.get("/export/{format}")
async def export_model(
    format: str,
    server: InferenceServer = Depends(get_server)
):
    """Trigger model export (onnx or torchscript)"""
    if not server.model or not server.model_version:
        raise HTTPException(status_code=404, detail="No active model loaded")
        
    try:
        if format == "onnx":
            path = server.exporter.export_onnx(
                server.model, 
                server.model_version.version
            )
        elif format == "torchscript":
            path = server.exporter.export_torchscript(
                server.model, 
                server.model_version.version
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid format. Use 'onnx' or 'torchscript'")
            
        return {"status": "exported", "path": path, "format": format}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/validation")
async def get_validation_metrics(
    model_version: Optional[str] = None,
    days: int = 30,
    server: InferenceServer = Depends(get_server)
):
    """
    Get shadow model validation metrics from analyst feedback.
    
    Returns:
    - Confusion matrix
    - Precision, Recall, F1 per class
    - Overall accuracy
    - FP/TP rates over time
    """
    try:
        # Get reviewed predictions from database
        query = """
            SELECT 
                predicted_class,
                class_name,
                COALESCE(ground_truth, predicted_class) as actual_class,
                is_tp,
                is_fp,
                reviewed_at,
                severity
            FROM ml_predictions 
            WHERE ground_truth IS NOT NULL
        """
        params = []
        
        if model_version:
            query += " AND model_version = ?"
            params.append(model_version)
        
        query += f" AND reviewed_at >= datetime('now', '-{days} days')"
        query += " ORDER BY reviewed_at DESC"
        
        results = db_manager.execute_query(query, tuple(params) if params else None)
        
        if not results:
            return {
                "total_reviewed": 0,
                "message": "No reviewed predictions found",
                "confusion_matrix": {},
                "class_metrics": {},
                "overall_metrics": {}
            }
        
        # Build confusion matrix
        class_names = ["Normal", "DoS/DDoS", "Recon/Scan", "Brute Force", 
                       "Web/Exploit", "Infiltration", "Botnet", "Backdoor",
                       "Worms", "Fuzzers", "Other"]
        
        confusion = {}  # {predicted: {actual: count}}
        tp_total = 0
        fp_total = 0
        tn_total = 0
        fn_total = 0
        
        class_tp = {}
        class_fp = {}
        class_fn = {}
        class_total = {}
        
        for row in results:
            pred_class = row['predicted_class']
            actual_class = row['actual_class']
            is_tp = row['is_tp']
            is_fp = row['is_fp']
            
            pred_name = class_names[pred_class] if pred_class < len(class_names) else f"Class_{pred_class}"
            actual_name = class_names[actual_class] if actual_class < len(class_names) else f"Class_{actual_class}"
            
            # Confusion matrix
            if pred_name not in confusion:
                confusion[pred_name] = {}
            confusion[pred_name][actual_name] = confusion[pred_name].get(actual_name, 0) + 1
            
            # Per-class metrics
            if is_tp:
                tp_total += 1
                class_tp[pred_name] = class_tp.get(pred_name, 0) + 1
            if is_fp:
                fp_total += 1
                class_fp[pred_name] = class_fp.get(pred_name, 0) + 1
            
            class_total[pred_name] = class_total.get(pred_name, 0) + 1
        
        # Calculate metrics per class
        class_metrics = {}
        for class_name in set(list(class_tp.keys()) + list(class_fp.keys()) + list(class_total.keys())):
            tp = class_tp.get(class_name, 0)
            fp = class_fp.get(class_name, 0)
            total = class_total.get(class_name, 0)
            fn = total - tp  # Simplified
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            class_metrics[class_name] = {
                "total": total,
                "true_positives": tp,
                "false_positives": fp,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4)
            }
        
        # Overall metrics
        total_reviewed = len(results)
        accuracy = tp_total / total_reviewed if total_reviewed > 0 else 0.0
        fp_rate = fp_total / total_reviewed if total_reviewed > 0 else 0.0
        
        # Trend data (last 7 days)
        trend_query = """
            SELECT 
                date(reviewed_at) as review_date,
                SUM(CASE WHEN is_tp THEN 1 ELSE 0 END) as tp_count,
                SUM(CASE WHEN is_fp THEN 1 ELSE 0 END) as fp_count,
                COUNT(*) as total
            FROM ml_predictions 
            WHERE ground_truth IS NOT NULL 
            AND reviewed_at >= datetime('now', '-7 days')
            GROUP BY date(reviewed_at)
            ORDER BY review_date DESC
        """
        trend_results = db_manager.execute_query(trend_query)
        
        daily_trend = []
        for row in trend_results:
            daily_trend.append({
                "date": row['review_date'],
                "tp": row['tp_count'],
                "fp": row['fp_count'],
                "total": row['total'],
                "accuracy": round(row['tp_count'] / row['total'], 4) if row['total'] > 0 else 0
            })
        
        return {
            "total_reviewed": total_reviewed,
            "time_range_days": days,
            "model_version": model_version or "all",
            "confusion_matrix": confusion,
            "class_metrics": class_metrics,
            "overall_metrics": {
                "accuracy": round(accuracy, 4),
                "fp_rate": round(fp_rate, 4),
                "true_positives": tp_total,
                "false_positives": fp_total,
                "precision": round(tp_total / (tp_total + fp_total), 4) if (tp_total + fp_total) > 0 else 0,
            },
            "daily_trend": daily_trend
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

