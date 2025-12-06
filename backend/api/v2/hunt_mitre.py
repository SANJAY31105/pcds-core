"""
PCDS Enterprise API v2 - Threat Hunting & MITRE ATT&CK Endpoints
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime 
import uuid
import json as json_lib

from config.database import db_manager, MITREQueries, DetectionQueries

# Hunt Router
hunt_router = APIRouter(prefix="/hunt", tags=["hunt"])

# MITRE Router
mitre_router = APIRouter(prefix="/mitre", tags=["mitre"])


# ============================================
# THREAT HUNTING ENDPOINTS
# ============================================

@hunt_router.get("/queries")
async def get_hunt_queries(
    query_type: Optional[str] = Query(None, regex="^(saved|scheduled|template)$"),
    is_public: Optional[bool] = None
):
    """Get available hunt queries"""
    query = "SELECT * FROM hunt_queries WHERE 1=1"
    params = []
    
    if query_type:
        query += " AND query_type = ?"
        params.append(query_type)
    
    if is_public is not None:
        query += " AND is_public = ?"
        params.append(is_public)
    
    query += " ORDER BY created_at DESC"
    
    queries = db_manager.execute_query(query, tuple(params) if params else None)
    
    # Parse JSON fields
    for q in queries:
        if q.get('detection_types'):
            q['detection_types'] = json_lib.loads(q['detection_types'])
        if q.get('technique_ids'):
            q['technique_ids'] = json_lib.loads(q['technique_ids'])
        if q.get('filters'):
            q['filters'] = json_lib.loads(q['filters'])
    
    return {
        "total": len(queries),
        "queries": queries
    }


@hunt_router.get("/queries/{query_id}")
async def get_hunt_query(query_id: str):
    """Get specific hunt query"""
    query = db_manager.execute_one("""
        SELECT * FROM hunt_queries WHERE id = ?
    """, (query_id,))
    
    if not query:
        raise HTTPException(status_code=404, detail=f"Hunt query {query_id} not found")
    
    # Parse JSON fields
    if query.get('detection_types'):
        query['detection_types'] = json_lib.loads(query['detection_types'])
    if query.get('technique_ids'):
        query['technique_ids'] = json_lib.loads(query['technique_ids'])
    if query.get('filters'):
        query['filters'] = json_lib.loads(query['filters'])
    
    # Get recent results
    results = db_manager.execute_query("""
        SELECT * FROM hunt_results 
        WHERE query_id = ?
        ORDER BY run_at DESC
        LIMIT 10
    """, (query_id,))
    
    return {
        "query": query,
        "recent_results": results
    }


@hunt_router.post("/queries/{query_id}/run")
async def run_hunt_query(query_id: str):
    """Execute a hunt query"""
    query = db_manager.execute_one("""
        SELECT * FROM hunt_queries WHERE id = ?
    """, (query_id,))
    
    if not query:
        raise HTTPException(status_code=404, detail=f"Hunt query {query_id} not found")
    
    # Parse query parameters
    detection_types = json_lib.loads(query['detection_types']) if query.get('detection_types') else []
    technique_ids = json_lib.loads(query['technique_ids']) if query.get('technique_ids') else []
    time_range = query.get('time_range', '24h')
    
    # Convert time range to hours
    hours_map = {'24h': 24, '7d': 168, '30d': 720}
    hours = hours_map.get(time_range, 24)
    
    # Build hunt query
    hunt_sql = "SELECT * FROM detections WHERE 1=1"
    params = []
    
    if detection_types:
        placeholders = ','.join(['?' for _ in detection_types])
        hunt_sql += f" AND detection_type IN ({placeholders})"
        params.extend(detection_types)
    
    if technique_ids:
        placeholders = ','.join(['?' for _ in technique_ids])
        hunt_sql += f" AND technique_id IN ({placeholders})"
        params.extend(technique_ids)
    
    # Time filter
    from datetime import datetime, timedelta
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    hunt_sql += " AND detected_at > ?"
    params.append(cutoff.isoformat())
    
    hunt_sql += " ORDER BY detected_at DESC LIMIT 1000"
    
    # Execute hunt
    start_time = datetime.utcnow()
    findings = db_manager.execute_query(hunt_sql, tuple(params))
    execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
    
    # Store result
    db_manager.execute_insert("""
        INSERT INTO hunt_results 
        (query_id, run_at, total_findings, findings_data, execution_time_ms)
        VALUES (?, ?, ?, ?, ?)
    """, (
        query_id,
        datetime.utcnow().isoformat(),
        len(findings),
        json_lib.dumps([f['id'] for f in findings if f.get('id')]),
        execution_time
    ))
    
    # Update query last_run_at
    db_manager.execute_update("""
        UPDATE hunt_queries 
        SET last_run_at = ?
        WHERE id = ?
    """, (datetime.utcnow().isoformat(), query_id))
    
    return {
        "query_id": query_id,
        "query_name": query['name'],
        "total_findings": len(findings),
        "execution_time_ms": execution_time,
        "findings": findings[:100]  # Limit to first 100 for response
    }


# ============================================
# MITRE ATT&CK ENDPOINTS
# ============================================

@mitre_router.get("/tactics")
async def get_mitre_tactics():
    """Get all MITRE ATT&CK tactics"""
    tactics = MITREQueries.get_all_tactics()
    
    return {
        "total": len(tactics),
        "tactics": tactics
    }


@mitre_router.get("/tactics/{tactic_id}/techniques")
async def get_tactic_techniques(tactic_id: str):
    """Get techniques for a specific tactic"""
    # Verify tactic exists
    tactic = db_manager.execute_one("""
        SELECT * FROM mitre_tactics WHERE id = ?
    """, (tactic_id,))
    
    if not tactic:
        raise HTTPException(status_code=404, detail=f"Tactic {tactic_id} not found")
    
    techniques = MITREQueries.get_techniques_by_tactic(tactic_id)
    
    # Parse JSON fields
    for tech in techniques:
        if tech.get('platforms'):
            tech['platforms'] = json_lib.loads(tech['platforms'])
        if tech.get('data_sources'):
            tech['data_sources'] = json_lib.loads(tech['data_sources'])
        if tech.get('mitigations'):
            tech['mitigations'] = json_lib.loads(tech['mitigations'])
        if tech.get('detection_methods'):
            tech['detection_methods'] = json_lib.loads(tech['detection_methods'])
    
    return {
        "tactic": tactic,
        "total_techniques": len(techniques),
        "techniques": techniques
    }


@mitre_router.get("/techniques/{technique_id}")
async def get_mitre_technique(technique_id: str):
    """Get detailed technique information"""
    technique = MITREQueries.get_technique(technique_id)
    
    if not technique:
        raise HTTPException(status_code=404, detail=f"Technique {technique_id} not found")
    
    # Parse JSON fields
    if technique.get('platforms'):
        technique['platforms'] = json_lib.loads(technique['platforms'])
    if technique.get('data_sources'):
        technique['data_sources'] = json_lib.loads(technique['data_sources'])
    if technique.get('mitigations'):
        technique['mitigations'] = json_lib.loads(technique['mitigations'])
    if technique.get('detection_methods'):
        technique['detection_methods'] = json_lib.loads(technique['detection_methods'])
    
    # Get recent detections using this technique
    recent_detections = DetectionQueries.get_recent(limit=50)
    technique_detections = [d for d in recent_detections if d.get('technique_id') == technique_id]
    
    return {
        "technique": technique,
        "recent_detections": technique_detections[:10],
        "detection_count": len(technique_detections)
    }


@mitre_router.get("/matrix/heatmap")
async def get_mitre_heatmap(hours: int = Query(24, ge=1, le=720)):
    """
    Get MITRE ATT&CK matrix heatmap data
    
    Returns detection frequency for each technique
    """
    from datetime import datetime, timedelta
    
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    # Get detection counts by technique
    technique_counts = db_manager.execute_query("""
        SELECT 
            technique_id,
            COUNT(*) as count
        FROM detections
        WHERE technique_id IS NOT NULL 
        AND detected_at > ?
        GROUP BY technique_id
    """, (cutoff.isoformat(),))
    
    # Build heatmap data
    heatmap = {}
    for item in technique_counts:
        heatmap[item['technique_id']] = item['count']
    
    # Get all tactics with technique counts
    tactics = MITREQueries.get_all_tactics()
    matrix = []
    
    for tactic in tactics:
        techniques = MITREQueries.get_techniques_by_tactic(tactic['id'])
        
        tactic_data = {
            'tactic_id': tactic['id'],
            'tactic_name': tactic['name'],
            'kill_chain_order': tactic['kill_chain_order'],
            'techniques': []
        }
        
        for tech in techniques:
            tactic_data['techniques'].append({
                'technique_id': tech['id'],
                'technique_name': tech['name'],
                'detection_count': heatmap.get(tech['id'], 0)
            })
        
        matrix.append(tactic_data)
    
    return {
        "time_range_hours": hours,
        "total_detections": sum(heatmap.values()),
        "unique_techniques_detected": len(heatmap),
        "matrix": matrix
    }


@mitre_router.get("/stats/coverage")
async def get_mitre_coverage():
    """Get MITRE ATT&CK coverage statistics"""
    # Get all techniques
    all_techniques = db_manager.execute_query("""
        SELECT COUNT(*) as total FROM mitre_techniques
    """)
    
    # Get techniques with detections
    covered_techniques = db_manager.execute_query("""
        SELECT COUNT(DISTINCT technique_id) as covered 
        FROM detections 
        WHERE technique_id IS NOT NULL
    """)
    
    # Get tactics coverage
    all_tactics = MITREQueries.get_all_tactics()
    tactics_with_detections = db_manager.execute_query("""
        SELECT DISTINCT tactic_id 
        FROM detections 
        WHERE tactic_id IS NOT NULL
    """)
    
    total_techniques = all_techniques[0]['total'] if all_techniques else 0
    covered_count = covered_techniques[0]['covered'] if covered_techniques else 0
    
    coverage_percentage = (covered_count / total_techniques * 100) if total_techniques > 0 else 0
    
    return {
        "total_techniques": total_techniques,
        "covered_techniques": covered_count,
        "coverage_percentage": round(coverage_percentage, 2),
        "total_tactics": len(all_tactics),
        "covered_tactics": len(tactics_with_detections),
        "tactics_coverage_percentage": round(len(tactics_with_detections) / len(all_tactics) * 100, 2) if all_tactics else 0
    }
