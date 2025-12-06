"""
PCDS Enterprise API v2 - Entities Endpoints
Entity scoring, management, timeline, and graph
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta

from config.database import EntityQueries, DetectionQueries
from engine import scoring_engine
from cache.redis_client import cache_client  # Redis caching

router = APIRouter(prefix="/entities", tags=["entities"])


@router.get("")
@cache_client.cache(ttl=120)  # Cache for 2 minutes
async def get_entities(
    limit: int = Query(100, ge=1, le=500),
    urgency_level: Optional[str] = Query(None, regex="^(critical|high|medium|low)$"),
    entity_type: Optional[str] = None
):
    """
    Get all entities with optional filtering
    
    Query params:
    - limit: Max entities to return (1-500)
    - urgency_level: Filter by urgency (critical/high/medium/low)
    - entity_type: Filter by type (host/user/service)
    """
    try:
        entities = EntityQueries.get_all(limit=limit, urgency_level=urgency_level)
        
        # Filter by type if specified
        if entity_type:
            entities = [e for e in entities if e.get('entity_type') == entity_type]
        
        return {
            "total": len(entities),
            "entities": entities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{entity_id}")
async def get_entity(entity_id: str):
    """Get detailed information about a specific entity"""
    entity = EntityQueries.get_by_id(entity_id)
    
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    
    # Get recent detections
    detections = DetectionQueries.get_by_entity(entity_id, limit=50)
    
    return {
        "entity": entity,
        "recent_detections": detections
    }


@router.get("/{entity_id}/timeline")
async def get_entity_timeline(
    entity_id: str,
    hours: int = Query(24, ge=1, le=168)  # 1 hour to 7 days
):
    """
    Get entity activity timeline
    
    Returns chronological list of detections for visualization
    """
    entity = EntityQueries.get_by_id(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    
    # Get detections from last N hours
    detections = DetectionQueries.get_by_entity(entity_id, limit=1000)
    
    # Filter by time range
    cutoff_time = datetime.utcnow() - timedelta(hours=hours)
    timeline_detections = [
        d for d in detections 
        if datetime.fromisoformat(d['detected_at']) > cutoff_time
    ]
    
    # Sort chronologically
    timeline_detections.sort(key=lambda d: d['detected_at'])
    
    return {
        "entity_id": entity_id,
        "entity_identifier": entity['identifier'],
        "time_range_hours": hours,
        "detection_count": len(timeline_detections),
        "timeline": timeline_detections
    }


@router.get("/{entity_id}/graph")
async def get_entity_attack_graph(entity_id: str):
    """
    Get attack graph data for entity
    
    Returns nodes (entities) and edges (relationships) for visualization
    """
    from config.database import db_manager
    
    entity = EntityQueries.get_by_id(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    
    # Get entity relationships
    relationships = db_manager.execute_query("""
        SELECT * FROM entity_relationships 
        WHERE source_entity_id = ? OR target_entity_id = ?
        ORDER BY last_seen DESC
        LIMIT 50
    """, (entity_id, entity_id))
    
    # Build nodes and edges
    nodes = {entity_id: entity}
    edges = []
    
    for rel in relationships:
        source_id = rel['source_entity_id']
        target_id = rel['target_entity_id']
        
        # Add nodes
        if source_id not in nodes and source_id != entity_id:
            source_entity = EntityQueries.get_by_id(source_id)
            if source_entity:
                nodes[source_id] = source_entity
        
        if target_id not in nodes and target_id != entity_id:
            target_entity = EntityQueries.get_by_id(target_id)
            if target_entity:
                nodes[target_id] = target_entity
        
        # Add edge
        edges.append({
            "source": source_id,
            "target": target_id,
            "relationship_type": rel['relationship_type'],
            "occurrence_count": rel['occurrence_count'],
            "first_seen": rel['first_seen'],
            "last_seen": rel['last_seen']
        })
    
    return {
        "entity_id": entity_id,
        "nodes": list(nodes.values()),
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges)
    }


@router.get("/stats/overview")
async def get_entity_stats():
    """Get entity statistics overview"""
    from config.database import db_manager
    
    # Get entity counts by urgency level
    stats = db_manager.execute_query("""
        SELECT 
            urgency_level,
            COUNT(*) as count
        FROM entities
        GROUP BY urgency_level
    """)
    
    # Get entity counts by type
    type_stats = db_manager.execute_query("""
        SELECT 
            entity_type,
            COUNT(*) as count
        FROM entities
        GROUP BY entity_type
    """)
    
    # Get total detection count
    detection_count = db_manager.execute_one("""
        SELECT COUNT(*) as total FROM detections
    """)
    
    # Calculate totals
    by_urgency = {s['urgency_level']: s['count'] for s in stats}
    total_entities = sum(by_urgency.values())
    
    # Get individual counts with defaults
    critical_count = by_urgency.get('critical', 0)
    high_count = by_urgency.get('high', 0)
    medium_count = by_urgency.get('medium', 0)
    low_count = by_urgency.get('low', 0)
    
    # Calculate distribution percentages
    def calc_percent(count):
        return round((count / total_entities * 100), 1) if total_entities > 0 else 0
    
    return {
        "total_entities": total_entities,
        "critical": critical_count,
        "high": high_count,
        "medium": medium_count,
        "low": low_count,
        "distribution": {
            "critical": calc_percent(critical_count),
            "high": calc_percent(high_count),
            "medium": calc_percent(medium_count),
            "low": calc_percent(low_count)
        },
        "by_urgency": by_urgency,
        "by_type": {s['entity_type']: s['count'] for s in type_stats},
        "total_detections": detection_count['total'] if detection_count else 0
    }


@router.post("/{entity_id}/recalculate-score")
async def recalculate_entity_score(
    entity_id: str,
    asset_value: int = Query(50, ge=0, le=100)
):
    """
    Manually trigger entity urgency score recalculation
    
    Useful for testing or forced updates
    """
    entity = EntityQueries.get_by_id(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail=f"Entity {entity_id} not found")
    
    # Get entity detections
    detections = DetectionQueries.get_by_entity(entity_id, limit=100)
    
    # Calculate new score
    score_result = scoring_engine.calculate_urgency_score(
        entity_id=entity_id,
        detections=detections,
        asset_value=asset_value,
        current_urgency=entity.get('urgency_score', 0)
    )
    
    # Update entity in database
    EntityQueries.update_urgency(
        entity_id,
        score_result['urgency_score'],
        score_result['urgency_level']
    )
    
    return {
        "entity_id": entity_id,
        "previous_score": entity.get('urgency_score', 0),
        "new_score": score_result['urgency_score'],
        "urgency_level": score_result['urgency_level'],
        "urgency_change": score_result['urgency_change'],
        "trend": score_result['trend'],
        "factors": score_result['factors'],
        "recommendations": score_result['recommendations']
    }
