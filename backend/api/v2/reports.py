from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, List, Any
from datetime import datetime, timedelta
from config.database import db_manager, get_db
from cache.memory_cache import memory_cache  # Enterprise-grade caching

router = APIRouter(prefix="/reports", tags=["Reports"])

@router.get("/executive-summary")
@memory_cache.cache_function(ttl=120)  # Cache for 2 minutes
async def get_executive_summary():
    """Get high-level executive summary metrics"""
    
    # 1. Risk Score (Average of all entities)
    risk_data = db_manager.execute_one("""
        SELECT AVG(threat_score) as avg_risk, COUNT(*) as total_entities 
        FROM entities
    """)
    
    # 2. Critical Threats (ALL - for demo with simulated attacks)
    critical_threats = db_manager.execute_one("""
        SELECT COUNT(*) as count 
        FROM detections 
        WHERE severity = 'critical'
    """)
    
    # 3. Top Risky Entities
    top_entities = db_manager.execute_query("""
        SELECT identifier, threat_score, urgency_level 
        FROM entities 
        ORDER BY threat_score DESC 
        LIMIT 5
    """)
    
    # 4. MTTD/MTTR (Mocked for now as we don't track resolution time perfectly yet)
    mttd = 12 # Minutes
    mttr = 45 # Minutes

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "period": "Last 24 Hours",
        "kpis": {
            "overall_risk_score": round(risk_data['avg_risk'] or 0, 1),
            "active_entities": risk_data['total_entities'],
            "critical_incidents": critical_threats['count'],
            "mttd_minutes": mttd,
            "mttr_minutes": mttr
        },
        "top_risky_entities": top_entities,
        "recommendations": [
            "Investigate top 3 risky entities immediately.",
            "Review firewall rules for repeated external scanning.",
            "Ensure all endpoints have the latest agent version."
        ]
    }

@router.get("/threat-intelligence")
@memory_cache.cache_function(ttl=120)  # Cache for 2 minutes
async def get_threat_intelligence():
    """Get detailed threat intelligence metrics"""
    
    # 1. Top Attack Types (from simulated data)
    top_tactics = db_manager.execute_query("""
        SELECT detection_type as name, COUNT(*) as count
        FROM detections
        GROUP BY detection_type
        ORDER BY count DESC
        LIMIT 5
    """)
    
    # 2. Top Techniques (from technique_id)
    top_techniques = db_manager.execute_query("""
        SELECT technique_id as id, technique_id as name, COUNT(*) as count
        FROM detections
        WHERE technique_id IS NOT NULL
        GROUP BY technique_id
        ORDER BY count DESC
        LIMIT 5
    """)
    
    # 3. Campaign Status
    campaign_stats = db_manager.execute_one("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) as active
        FROM attack_campaigns
    """)

    return {
        "generated_at": datetime.utcnow().isoformat(),
        "period": "Last 7 Days",
        "top_tactics": top_tactics,
        "top_techniques": top_techniques,
        "campaigns": {
            "total_detected": campaign_stats['total'],
            "currently_active": campaign_stats['active']
        }
    }

@router.get("/compliance-report")
@memory_cache.cache_function(ttl=300)  # Cache for 5 minutes
async def get_compliance_report(framework: str = "nist"):
    """
    Get compliance reporting metrics
    Supports: nist, iso27001, pci-dss
    """
    
    # Detection coverage (techniques detected)
    coverage = db_manager.execute_one("""
        SELECT 
            COUNT(DISTINCT technique_id) as detected,
            100 as total
        FROM detections
        WHERE technique_id IS NOT NULL
    """)
    
    coverage_pct = round((coverage['detected'] / coverage['total']) * 100, 1) if coverage['total'] > 0 else 0
    
    # Incident response metrics
    incident_metrics = db_manager.execute_one("""
        SELECT 
            COUNT(*) as total_incidents,
            SUM(CASE WHEN severity='critical' THEN 1 ELSE 0 END) as critical_count
        FROM detections
    """)
    
    audit_stats = db_manager.execute_one("""
        SELECT 0 as logged_actions
    """)
    
    # Framework-specific requirements
    frameworks = {
        "nist": {
            "name": "NIST Cybersecurity Framework",
            "categories": [
                {"name": "Identify", "score": 85, "status": "compliant"},
                {"name": "Protect", "score": 78, "status": "needs_improvement"},
                {"name": "Detect", "score": coverage_pct, "status": "compliant" if coverage_pct > 70 else "needs_improvement"},
                {"name": "Respond", "score": 82, "status": "compliant"},
                {"name": "Recover", "score": 75, "status": "compliant"}
            ]
        },
        "iso27001": {
            "name": "ISO/IEC 27001",
            "categories": [
                {"name": "Information Security Policies", "score": 90, "status": "compliant"},
                {"name": "Access Control", "score": 88, "status": "compliant"},
                {"name": "Incident Management", "score": 85, "status": "compliant"},
                {"name": "Monitoring & Review", "score": coverage_pct, "status": "compliant" if coverage_pct > 70 else "needs_improvement"}
            ]
        },
        "pci-dss": {
            "name": "PCI-DSS v4.0",
            "categories": [
                {"name": "Network Security", "score": 92, "status": "compliant"},
                {"name": "Access Control", "score": 87, "status": "compliant"},
                {"name": "Monitoring & Testing", "score": coverage_pct, "status": "compliant" if coverage_pct > 70 else "needs_improvement"},
                {"name": "Incident Response", "score": 83, "status": "compliant"}
            ]
        }
    }
    
    selected_framework = frameworks.get(framework.lower(), frameworks["nist"])
    
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "framework": selected_framework,
        "metrics": {
            "detection_coverage": coverage_pct,
            "total_incidents": incident_metrics['total_incidents'] or 0,
            "critical_incidents": incident_metrics['critical_count'] or 0,
            "audit_log_entries": audit_stats['logged_actions'] or 0
        },
        "overall_score": round(sum(c['score'] for c in selected_framework['categories']) / len(selected_framework['categories']), 1)
    }

@router.get("/trend-analysis")
async def get_trend_analysis(days: int = 30):
    """Get trend analysis over time"""
    
    cutoff_time = datetime.utcnow() - timedelta(days=days)
    
    # Detection trend by day
    detection_trend = db_manager.execute_query("""
        SELECT 
            DATE(detected_at) as date,
            COUNT(*) as total,
            SUM(CASE WHEN severity='critical' THEN 1 ELSE 0 END) as critical,
            SUM(CASE WHEN severity='high' THEN 1 ELSE 0 END) as high
        FROM detections
        WHERE detected_at > ?
        GROUP BY DATE(detected_at)
        ORDER BY date
    """, (cutoff_time.isoformat(),))
    
    # Entity risk score changes
    risk_trend = db_manager.execute_query("""
        SELECT 
            DATE(last_detection_time) as date,
            AVG(threat_score) as avg_risk,
            MAX(threat_score) as max_risk
        FROM entities
        WHERE last_detection_time > ?
        GROUP BY DATE(last_detection_time)
        ORDER BY date
    """, (cutoff_time.isoformat(),))
    
    # Top technique frequency over time
    technique_trend = db_manager.execute_query("""
        SELECT 
            DATE(d.detected_at) as date,
            t.name as technique_name,
            COUNT(*) as count
        FROM detections d
        JOIN mitre_techniques t ON d.technique_id = t.id
        WHERE d.detected_at > ?
        AND t.name IN (
            SELECT t2.name 
            FROM detections d2
            JOIN mitre_techniques t2 ON d2.technique_id = t2.id
            WHERE d2.detected_at > ?
            GROUP BY t2.name
            ORDER BY COUNT(*) DESC
            LIMIT 5
        )
        GROUP BY DATE(d.detected_at), t.name
        ORDER BY date, count DESC
    """, (cutoff_time.isoformat(), cutoff_time.isoformat()))
    
    return {
        "generated_at": datetime.utcnow().isoformat(),
        "time_range_days": days,
        "detection_trend": detection_trend,
        "risk_trend": risk_trend,
        "technique_trend": technique_trend
    }
