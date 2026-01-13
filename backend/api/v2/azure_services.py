"""
Azure Services API Endpoints
Exposes Azure Blob Storage and Azure OpenAI endpoints for the frontend

Imagine Cup Requirement: Uses 2 Azure Services
1. Azure OpenAI - AI Copilot
2. Azure Blob Storage - Report Storage
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import Azure services
try:
    from ml.azure_blob_service import get_blob_service
except ImportError:
    get_blob_service = None

try:
    from ml.azure_ai_service import get_azure_ai_service
except ImportError:
    get_azure_ai_service = None

router = APIRouter(prefix="/api/v2/azure", tags=["Azure Services"])


class ReportUploadRequest(BaseModel):
    report_type: str  # "detection", "incident", "audit"
    data: Dict[str, Any]


class AuditLogRequest(BaseModel):
    event_type: str
    details: Dict[str, Any]


# ==================== AZURE SERVICES STATUS ====================

@router.get("/status")
async def get_azure_services_status():
    """
    Get status of all Azure services used in PCDS
    
    Imagine Cup: Demonstrates use of 2+ Azure services
    """
    services = []
    
    # Azure OpenAI
    if get_azure_ai_service:
        ai_service = get_azure_ai_service()
        services.append({
            "name": "Azure OpenAI",
            "type": "AI/ML",
            "status": "connected" if ai_service else "demo_mode",
            "description": "GPT-4 powered security copilot for threat analysis",
            "features": [
                "Natural language threat queries",
                "Incident analysis and remediation",
                "Attack pattern explanation"
            ]
        })
    else:
        services.append({
            "name": "Azure OpenAI",
            "type": "AI/ML",
            "status": "demo_mode",
            "description": "GPT-4 powered security copilot"
        })
    
    # Azure Blob Storage
    if get_blob_service:
        blob_service = get_blob_service()
        stats = blob_service.get_storage_stats()
        services.append({
            "name": "Azure Blob Storage",
            "type": "Storage",
            "status": "connected" if blob_service.is_connected else "demo_mode",
            "description": "Cloud storage for detection reports and audit logs",
            "features": [
                "Detection report archival",
                "Incident report storage",
                "Audit log persistence",
                "ML model versioning"
            ],
            "stats": stats
        })
    else:
        services.append({
            "name": "Azure Blob Storage",
            "type": "Storage",
            "status": "demo_mode",
            "description": "Cloud storage for reports and logs"
        })
    
    return {
        "imagine_cup_compliant": len(services) >= 2,
        "total_services": len(services),
        "services": services,
        "message": "PCDS uses Azure OpenAI for AI-powered threat analysis and Azure Blob Storage for secure report archival"
    }


# ==================== AZURE BLOB STORAGE ENDPOINTS ====================

@router.post("/storage/upload-report")
async def upload_report(request: ReportUploadRequest):
    """Upload a detection or incident report to Azure Blob Storage"""
    if not get_blob_service:
        raise HTTPException(status_code=503, detail="Azure Blob Storage not available")
    
    blob_service = get_blob_service()
    
    # Add metadata
    request.data["uploaded_at"] = datetime.now().isoformat()
    request.data["report_type"] = request.report_type
    
    if request.report_type == "detection":
        result = blob_service.upload_detection_report(request.data)
    elif request.report_type == "incident":
        result = blob_service.upload_incident_report(request.data)
    else:
        result = blob_service.upload_report(
            f"{request.report_type}_report",
            str(request.data),
            "json"
        )
    
    return result


@router.get("/storage/reports")
async def list_reports(limit: int = 50):
    """List reports stored in Azure Blob Storage"""
    if not get_blob_service:
        # Return demo data
        return {
            "reports": [
                {
                    "name": "20241225_incident_ransomware_001.json",
                    "size": 2048,
                    "type": "incident",
                    "last_modified": datetime.now().isoformat()
                },
                {
                    "name": "20241225_detection_ddos_002.json",
                    "size": 1536,
                    "type": "detection",
                    "last_modified": datetime.now().isoformat()
                }
            ],
            "demo_mode": True
        }
    
    blob_service = get_blob_service()
    reports = blob_service.list_reports(limit)
    
    return {
        "reports": reports,
        "total": len(reports),
        "demo_mode": not blob_service.is_connected
    }


@router.post("/storage/audit-log")
async def create_audit_log(request: AuditLogRequest):
    """Create an audit log entry in Azure Blob Storage"""
    if not get_blob_service:
        return {
            "success": True,
            "demo_mode": True,
            "event_type": request.event_type
        }
    
    blob_service = get_blob_service()
    result = blob_service.log_audit_event(request.event_type, request.details)
    
    return result


@router.get("/storage/stats")
async def get_storage_stats():
    """Get Azure Blob Storage statistics"""
    if not get_blob_service:
        return {
            "service": "Azure Blob Storage",
            "connected": False,
            "demo_mode": True,
            "containers": ["pcds-reports", "pcds-models", "pcds-logs"]
        }
    
    blob_service = get_blob_service()
    return blob_service.get_storage_stats()
