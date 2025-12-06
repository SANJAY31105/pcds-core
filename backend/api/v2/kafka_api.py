"""
PCDS Enterprise API - Kafka Endpoints
Event streaming, SIEM integration, and replay API
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from pydantic import BaseModel


router = APIRouter(prefix="/kafka", tags=["kafka"])


# Request/Response models
class EventPayload(BaseModel):
    entity_id: str
    detection_type: str
    severity: str = "medium"
    source_ip: Optional[str] = None
    destination_ip: Optional[str] = None
    data: Dict = {}


class SIEMConfigPayload(BaseModel):
    connector_type: str  # splunk, elastic, syslog
    endpoint: str
    api_key: Optional[str] = None
    enabled: bool = True


class ReplayRequest(BaseModel):
    topic: str = "pcds.detections"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    entity_id: Optional[str] = None
    severity: Optional[str] = None
    limit: int = 100


# ============================================================
# KAFKA STATUS & METRICS
# ============================================================

@router.get("/status")
async def get_kafka_status():
    """Get Kafka integration status"""
    try:
        from kafka import kafka_producer, kafka_consumer, siem_manager
        
        return {
            "status": "operational",
            "producer": kafka_producer.get_metrics(),
            "consumer": kafka_consumer.get_metrics(),
            "siem_connectors": siem_manager.get_metrics(),
            "topics": {
                "raw_events": "pcds.raw-events",
                "detections": "pcds.detections",
                "alerts": "pcds.alerts",
                "dead_letter": "pcds.dead-letter",
                "audit": "pcds.audit"
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.get("/metrics")
async def get_kafka_metrics():
    """Get detailed Kafka metrics"""
    try:
        from kafka import kafka_producer, kafka_consumer, event_store
        
        # Get event counts
        counts = event_store.count_events()
        timeline = event_store.get_timeline(hours=24)
        
        return {
            "producer_metrics": kafka_producer.get_metrics(),
            "consumer_metrics": kafka_consumer.get_metrics(),
            "event_store": counts,
            "timeline_24h": timeline
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# EVENT PUBLISHING
# ============================================================

@router.post("/publish")
async def publish_event(event: EventPayload):
    """Publish event to Kafka"""
    try:
        from kafka import kafka_producer, event_store
        from kafka.config import TOPICS
        
        event_data = {
            "entity_id": event.entity_id,
            "detection_type": event.detection_type,
            "severity": event.severity,
            "source_ip": event.source_ip,
            "destination_ip": event.destination_ip,
            **event.data
        }
        
        # Send to Kafka
        await kafka_producer.connect()
        success = await kafka_producer.send(
            topic=TOPICS["raw_events"].name,
            message=event_data,
            key=event.entity_id
        )
        
        # Store for replay
        event_store.store_event(
            TOPICS["raw_events"].name, 
            event_data, 
            key=event.entity_id
        )
        
        return {
            "success": success,
            "topic": TOPICS["raw_events"].name,
            "entity_id": event.entity_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/publish/batch")
async def publish_batch(events: List[EventPayload]):
    """Publish batch of events"""
    try:
        from kafka import kafka_producer
        from kafka.config import TOPICS
        
        await kafka_producer.connect()
        
        messages = [
            {
                "entity_id": e.entity_id,
                "detection_type": e.detection_type,
                "severity": e.severity,
                "source_ip": e.source_ip,
                "destination_ip": e.destination_ip,
                **e.data
            }
            for e in events
        ]
        
        count = await kafka_producer.send_batch(TOPICS["raw_events"].name, messages)
        
        return {
            "success": True,
            "events_sent": count,
            "total_events": len(events)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# SIEM MANAGEMENT
# ============================================================

@router.get("/siem/status")
async def get_siem_status():
    """Get SIEM connector status"""
    from kafka import siem_manager
    return {
        "connectors": siem_manager.get_metrics()
    }


@router.post("/siem/test")
async def test_siem_connectors():
    """Send test event to all SIEM connectors"""
    from kafka import siem_manager
    
    test_event = {
        "detection_type": "siem_test",
        "severity": "low",
        "entity_id": "test-entity",
        "description": "SIEM connectivity test",
        "timestamp": datetime.utcnow().isoformat()
    }
    
    results = await siem_manager.send_to_all(test_event)
    await siem_manager.flush_all()
    
    return {
        "test_event": test_event,
        "results": results
    }


# ============================================================
# EVENT REPLAY
# ============================================================

@router.post("/replay")
async def replay_events(request: ReplayRequest, background_tasks: BackgroundTasks):
    """Replay historical events"""
    from kafka import event_replay
    from kafka.replay import ReplayOptions
    
    start_time = None
    end_time = None
    
    if request.start_time:
        start_time = datetime.fromisoformat(request.start_time)
    if request.end_time:
        end_time = datetime.fromisoformat(request.end_time)
    
    options = ReplayOptions(
        topic=request.topic,
        start_time=start_time,
        end_time=end_time,
        entity_id=request.entity_id,
        severity=request.severity,
        limit=request.limit
    )
    
    events = event_replay.store.query_events(options)
    
    return {
        "events": events,
        "total": len(events),
        "options": {
            "topic": request.topic,
            "start_time": request.start_time,
            "end_time": request.end_time,
            "entity_id": request.entity_id,
            "severity": request.severity
        }
    }


@router.get("/replay/timeline")
async def get_event_timeline(hours: int = Query(24, ge=1, le=168)):
    """Get event timeline for visualization"""
    from kafka import event_store
    
    timeline = event_store.get_timeline(hours=hours)
    
    return {
        "hours": hours,
        "timeline": timeline
    }


# ============================================================
# AUDIT LOG
# ============================================================

@router.get("/audit")
async def get_audit_log(
    user: Optional[str] = None,
    action: Optional[str] = None,
    hours: int = Query(24, ge=1, le=720),
    limit: int = Query(100, ge=1, le=1000)
):
    """Query audit log"""
    from kafka import audit_log
    
    start_time = datetime.utcnow() - timedelta(hours=hours)
    
    entries = audit_log.query_audit_log(
        user=user,
        action=action,
        start_time=start_time,
        limit=limit
    )
    
    return {
        "entries": entries,
        "total": len(entries),
        "filters": {
            "user": user,
            "action": action,
            "hours": hours
        }
    }


@router.post("/audit/log")
async def create_audit_entry(
    action: str,
    user: str,
    resource: str,
    details: Dict = None
):
    """Create audit log entry"""
    from kafka import audit_log
    
    entry = await audit_log.log_action(
        user=user,
        action=action,
        resource=resource,
        details=details
    )
    
    return {"entry": entry}
