"""
PCDS Enterprise API v2 - Campaigns & Investigations Endpoints
Multi-stage attack campaigns and case management
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime
import uuid

from config.database import db_manager
from engine import campaign_correlator

# Campaigns Router
campaigns_router = APIRouter(prefix="/campaigns", tags=["campaigns"])

# Investigations Router  
investigations_router = APIRouter(prefix="/investigations", tags=["investigations"])


# ============================================
# CAMPAIGNS ENDPOINTS
# ============================================

@campaigns_router.get("")
async def get_campaigns(
    status: Optional[str] = Query(None, regex="^(active|contained|resolved)$"),
    limit: int = Query(50, ge=1, le=200)
):
    """Get attack campaigns"""
    if status:
        query = """
            SELECT * FROM attack_campaigns 
            WHERE status = ?
            ORDER BY started_at DESC
            LIMIT ?
        """
        campaigns = db_manager.execute_query(query, (status, limit))
    else:
        query = """
            SELECT * FROM attack_campaigns 
            ORDER BY started_at DESC
            LIMIT ?
        """
        campaigns = db_manager.execute_query(query, (limit,))
    
    return {
        "total": len(campaigns),
        "campaigns": campaigns
    }


@campaigns_router.get("/{campaign_id}")
async def get_campaign(campaign_id: str):
    """Get campaign details including all associated detections"""
    campaign = db_manager.execute_one("""
        SELECT * FROM attack_campaigns WHERE id = ?
    """, (campaign_id,))
    
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
    
    # Get associated detections
    detections = db_manager.execute_query("""
        SELECT d.*, cd.added_at
        FROM campaign_detections cd
        JOIN detections d ON cd.detection_id = d.id
        WHERE cd.campaign_id = ?
        ORDER BY d.detected_at
    """, (campaign_id,))
    
    # Parse JSON fields
    import json
    if campaign.get('tactics_used'):
        campaign['tactics_used'] = json.loads(campaign['tactics_used'])
    if campaign.get('techniques_used'):
        campaign['techniques_used'] = json.loads(campaign['techniques_used'])
    
    return {
        "campaign": campaign,
        "detections": detections,
        "detection_count": len(detections)
    }


@campaigns_router.patch("/{campaign_id}/status")
async def update_campaign_status(
    campaign_id: str,
    status: str = Query(..., regex="^(active|contained|resolved)$")
):
    """Update campaign status"""
    campaign = db_manager.execute_one(
        "SELECT * FROM attack_campaigns WHERE id = ?",
        (campaign_id,)
    )
    
    if not campaign:
        raise HTTPException(status_code=404, detail=f"Campaign {campaign_id} not found")
    
    # Update status
    updates = {
        'status': status,
        'updated_at': datetime.utcnow().isoformat()
    }
    
    if status == 'resolved':
        updates['ended_at'] = datetime.utcnow().isoformat()
    
    db_manager.execute_update("""
        UPDATE attack_campaigns 
        SET status = ?, updated_at = ?, ended_at = ?
        WHERE id = ?
    """, (status, updates['updated_at'], updates.get('ended_at'), campaign_id))
    
    return {
        "campaign_id": campaign_id,
        "status": status,
        "message": "Campaign status updated"
    }


# ============================================
# INVESTIGATIONS ENDPOINTS
# ============================================

@investigations_router.get("")
async def get_investigations(
    status: Optional[str] = Query(None, regex="^(open|investigating|resolved|closed)$"),
    assigned_to: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200)
):
    """Get investigations"""
    query = "SELECT * FROM investigations WHERE 1=1"
    params = []
    
    if status:
        query += " AND status = ?"
        params.append(status)
    
    if assigned_to:
        query += " AND assigned_to = ?"
        params.append(assigned_to)
    
    query += " ORDER BY opened_at DESC LIMIT ?"
    params.append(limit)
    
    investigations = db_manager.execute_query(query, tuple(params))
    
    return {
        "total": len(investigations),
        "investigations": investigations
    }


@investigations_router.get("/{investigation_id}")
async def get_investigation(investigation_id: str):
    """Get investigation details with notes and evidence"""
    investigation = db_manager.execute_one("""
        SELECT * FROM investigations WHERE id = ?
    """, (investigation_id,))
    
    if not investigation:
        raise HTTPException(status_code=404, detail=f"Investigation {investigation_id} not found")
    
    # Get notes
    notes = db_manager.execute_query("""
        SELECT * FROM investigation_notes 
        WHERE investigation_id = ?
        ORDER BY created_at DESC
    """, (investigation_id,))
    
    # Get evidence
    evidence = db_manager.execute_query("""
        SELECT * FROM investigation_evidence 
        WHERE investigation_id = ?
        ORDER BY uploaded_at DESC
    """, (investigation_id,))
    
    # Parse JSON fields
    import json
    if investigation.get('entity_ids'):
        investigation['entity_ids'] = json.loads(investigation['entity_ids'])
    if investigation.get('detection_ids'):
        investigation['detection_ids'] = json.loads(investigation['detection_ids'])
    if investigation.get('tags'):
        investigation['tags'] = json.loads(investigation['tags'])
    
    return {
        "investigation": investigation,
        "notes": notes,
        "evidence": evidence
    }


@investigations_router.post("")
async def create_investigation(investigation_data: dict):
    """Create a new investigation"""
    import json
    
    investigation_id = f"inv_{uuid.uuid4().hex[:12]}"
    
    # Required fields
    if 'title' not in investigation_data:
        raise HTTPException(status_code=400, detail="title is required")
    if 'severity' not in investigation_data:
        investigation_data['severity'] = 'medium'
    
    db_manager.execute_insert("""
        INSERT INTO investigations 
        (id, title, description, severity, priority, status, assigned_to, 
         assignee_email, entity_ids, detection_ids, campaign_id, opened_at, tags)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        investigation_id,
        investigation_data['title'],
        investigation_data.get('description'),
        investigation_data['severity'],
        investigation_data.get('priority', 'medium'),
        'open',
        investigation_data.get('assigned_to'),
        investigation_data.get('assignee_email'),
        json.dumps(investigation_data.get('entity_ids', [])),
        json.dumps(investigation_data.get('detection_ids', [])),
        investigation_data.get('campaign_id'),
        datetime.utcnow().isoformat(),
        json.dumps(investigation_data.get('tags', []))
    ))
    
    return {
        "investigation_id": investigation_id,
        "message": "Investigation created successfully"
    }


@investigations_router.post("/{investigation_id}/notes")
async def add_investigation_note(
    investigation_id: str,
    note_data: dict
):
    """Add a note to an investigation"""
    investigation = db_manager.execute_one(
        "SELECT * FROM investigations WHERE id = ?",
        (investigation_id,)
    )
    
    if not investigation:
        raise HTTPException(status_code=404, detail=f"Investigation {investigation_id} not found")
    
    if 'content' not in note_data or 'author' not in note_data:
        raise HTTPException(status_code=400, detail="content and author are required")
    
    db_manager.execute_insert("""
        INSERT INTO investigation_notes (investigation_id, author, content, created_at)
        VALUES (?, ?, ?, ?)
    """, (
        investigation_id,
        note_data['author'],
        note_data['content'],
        datetime.utcnow().isoformat()
    ))
    
    return {"message": "Note added successfully"}


@investigations_router.patch("/{investigation_id}/status")
async def update_investigation_status(
    investigation_id: str,
    status: str = Query(..., regex="^(open|investigating|resolved|closed)$"),
    resolution: Optional[str] = Query(None, regex="^(true_positive|false_positive|benign)$"),
    resolution_notes: Optional[str] = None
):
    """Update investigation status"""
    investigation = db_manager.execute_one(
        "SELECT * FROM investigations WHERE id = ?",
        (investigation_id,)
    )
    
    if not investigation:
        raise HTTPException(status_code=404, detail=f"Investigation {investigation_id} not found")
    
    query = "UPDATE investigations SET status = ?, updated_at = ?"
    params = [status, datetime.utcnow().isoformat()]
    
    if status in ['resolved', 'closed']:
        query += ", closed_at = ?"
        params.append(datetime.utcnow().isoformat())
        
        if resolution:
            query += ", resolution = ?"
            params.append(resolution)
        
        if resolution_notes:
            query += ", resolution_notes = ?"
            params.append(resolution_notes)
    
    query += " WHERE id = ?"
    params.append(investigation_id)
    
    db_manager.execute_update(query, tuple(params))
    
    return {
        "investigation_id": investigation_id,
        "status": status,
        "resolution": resolution,
        "message": "Investigation updated successfully"
    }
