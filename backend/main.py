"""
PCDS FastAPI Main Application - Enterprise Edition
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
from typing import List, Optional
import random
import uuid

from models import (
    ThreatDetection, NetworkEvent, AlertNotification,
    DashboardStats, SystemMetrics, CountermeasureAction, ThreatSeverity
)
from websocket_manager import manager
from threat_engine import threat_engine
from data_generator import data_generator
from redis_client import redis_client
from mitre_attack import mitre_attack
from entity_manager import entity_manager, UrgencyLevel, EntityType
from detections.detection_engine import detection_engine

# API v2 Routers
try:
    from api.v2.entities import router as entities_router
    from api.v2.detections import router as detections_router
    from api.v2.campaigns_investigations import campaigns_router, investigations_router
    from api.v2.hunt_mitre import hunt_router, mitre_router
    from api.v2.dashboard import router as dashboard_router
    HAS_API_V2 = True
    print("✅ API v2 routers loaded")
except Exception as e:
    print(f"⚠️ API v2 routers not available: {e}")
    HAS_API_V2 = False

# Prometheus metrics
threat_counter = Counter('pcds_threats_detected_total', 'Total threats detected')
request_latency = Histogram('pcds_request_duration_seconds', 'Request latency')

# In-memory storage
threats_storage = []
alerts_storage = []
system_metrics_storage = []
investigations_storage = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    print("🚀 Starting PCDS Backend...")
    
    # Initialize Redis
    try:
        await redis_client.connect()
    except Exception as e:
        print(f"⚠️ Redis connection failed: {e}")
    
    # Start background tasks
    asyncio.create_task(continuous_threat_detection())
    asyncio.create_task(manager.heartbeat())
    asyncio.create_task(generate_system_metrics())
    asyncio.create_task(simulate_realistic_threats())
    
    print("✅ PCDS Backend ready!")
    
    yield
    
    # Cleanup
    await redis_client.disconnect()
    print("👋 PCDS Backend shutdown complete")


app = FastAPI(
    title="Predictive Cyber Defence System - Enterprise",
    description="AI-powered Network Detection & Response (NDR) platform",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API v2 Routers
if HAS_API_V2:
    print("🔌 Registering API v2 endpoints...")
    app.include_router(entities_router, prefix="/api/v2")
    app.include_router(detections_router, prefix="/api/v2")
    app.include_router(campaigns_router, prefix="/api/v2")
    app.include_router(investigations_router, prefix="/api/v2")
    app.include_router(hunt_router, prefix="/api/v2")
    app.include_router(mitre_router, prefix="/api/v2")
    app.include_router(dashboard_router, prefix="/api/v2")
    print("✅ API v2 endpoints registered (36 total)")



# ============= Core API Endpoints =============

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "PCDS Enterprise API",
        "version": "2.0.0",
        "status": "operational",
        "features": ["Attack Signal Intelligence", "Entity Scoring", "MITRE ATT&CK", "Threat Hunting"],
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "connections": manager.get_connection_count(),
        "entities_tracked": len(entity_manager.entities),
        "active_threats": len(threats_storage)
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# ============= Entity Management Endpoints =============

@app.get("/api/v1/entities")
async def get_entities(
    urgency: Optional[str] = Query(None, description="Filter by urgency (critical/high/medium/low)"),
    entity_type: Optional[str] = Query(None, description="Filter by type (host/user/service)"),
    limit: int = Query(100, description="Maximum results to return")
):
    """Get all entities with optional filtering"""
    entities = entity_manager.get_all_entities(
        urgency_filter=urgency,
        entity_type_filter=entity_type,
        limit=limit
    )
    return {"entities": entities, "total": len(entities)}


@app.get("/api/v1/entities/{entity_id}")
async def get_entity(entity_id: str):
    """Get detailed information about an entity"""
    entity = entity_manager.get_entity(entity_id)
    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")
    
    return {
        "entity": entity,
        "timeline": entity_manager.get_entity_timeline(entity_id),
        "attack_graph": entity_manager.get_attack_graph(entity_id)
    }


@app.get("/api/v1/entities/{entity_id}/timeline")
async def get_entity_timeline(entity_id: str):
    """Get chronological attack timeline for an entity"""
    timeline = entity_manager.get_entity_timeline(entity_id)
    if not timeline:
        raise HTTPException(status_code=404, detail="Entity not found or no detections")
    return {"entity_id": entity_id, "timeline": timeline}


@app.get("/api/v1/entities/{entity_id}/graph")
async def get_entity_attack_graph(entity_id: str):
    """Get attack graph visualization data"""
    graph = entity_manager.get_attack_graph(entity_id)
    return {"entity_id": entity_id, **graph}


@app.get("/api/v1/entities/stats")
async def get_entity_stats():
    """Get entity statistics"""
    return entity_manager.get_statistics()


# ============= Detection Endpoints =============

@app.get("/api/v1/detections")
async def get_detections(
    severity: Optional[str] = Query(None),
    limit: int = Query(50)
):
    """Get recent detections with filtering"""
    detections = threats_storage
    
    if severity:
        detections = [d for d in detections if d.severity.value == severity]
    
    return {
        "detections": sorted(detections, key=lambda x: x.timestamp, reverse=True)[:limit],
        "total": len(detections)
    }


@app.get("/api/v1/detections/{detection_id}")
async def get_detection(detection_id: str):
    """Get detailed detection information"""
    for detection in threats_storage:
        if detection.id == detection_id:
            return detection
    raise HTTPException(status_code=404, detail="Detection not found")


# ============= MITRE ATT&CK Endpoints =============

@app.get("/api/v1/mitre/tactics")
async def get_mitre_tactics():
    """Get all MITRE ATT&CK tactics"""
    return {"tactics": mitre_attack.get_all_tactics()}


@app.get("/api/v1/mitre/tactics/{tactic_id}/techniques")
async def get_mitre_techniques(tactic_id: str):
    """Get techniques for a specific tactic"""
    techniques = mitre_attack.get_techniques_by_tactic(tactic_id)
    return {"tactic_id": tactic_id, "techniques": techniques}


@app.get("/api/v1/mitre/techniques/{technique_id}")
async def get_mitre_technique(technique_id: str):
    """Get detailed technique information"""
    technique = mitre_attack.get_technique(technique_id)
    if not technique:
        raise HTTPException(status_code=404, detail="Technique not found")
    return technique


@app.get("/api/v1/mitre/matrix/heatmap")
async def get_mitre_heatmap():
    """Get MITRE ATT&CK matrix heatmap data"""
    heatmap = mitre_attack.get_matrix_heatmap(threats_storage)
    return {"heatmap": heatmap, "total_detections": len(threats_storage)}


# ============= Dashboard Endpoints =============

@app.get("/api/v1/dashboard/overview")
async def get_dashboard_overview():
    """Get comprehensive dashboard overview"""
    entity_stats = entity_manager.get_statistics()
    
    # Calculate MTTD and MTTR (simulated)
    mttd_minutes = random.randint(5, 30)
    mttr_minutes = random.randint(15, 90)
    
    # Top attacked entities
    top_entities = entity_manager.get_all_entities(limit=5)
    
    # Recent high-priority detections
    high_priority = [
        t for t in threats_storage 
        if t.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH]
    ][-10:]
    
    # Tactic distribution
    tactic_counts = {}
    for threat in threats_storage:
        mitre_data = getattr(threat, 'mitre', {})
        if isinstance(mitre_data, dict):
            tactic = mitre_data.get('tactic_name', 'Unknown')
            tactic_counts[tactic] = tactic_counts.get(tactic, 0) + 1
    
    return {
        "entity_stats": entity_stats,
        "metrics": {
            "total_detections": len(threats_storage),
            "active_campaigns": random.randint(2, 8),
            "mttd_minutes": mttd_minutes,
            "mttr_minutes": mttr_minutes
        },
        "top_entities": top_entities,
        "recent_high_priority": [t.dict() for t in high_priority],
        "tactic_distribution": tactic_counts
    }


@app.get("/api/v1/dashboard/stats", response_model=DashboardStats)
async def get_dashboard_stats():
    """Get dashboard statistics (legacy endpoint)"""
    critical_count = sum(1 for t in threats_storage if t.severity == ThreatSeverity.CRITICAL)
    total_threats = len(threats_storage)
    avg_risk = sum(t.risk_score for t in threats_storage) / max(total_threats, 1)
    
    return DashboardStats(
        total_threats=total_threats,
        critical_threats=critical_count,
        threats_today=len([t for t in threats_storage if (datetime.utcnow() - t.timestamp).days == 0]),
        threats_blocked=total_threats,
        average_risk_score=avg_risk,
        system_health="operational",
        uptime_percentage=99.8
    )


# ============= Threat Hunting Endpoints =============

@app.get("/api/v1/hunt/queries")
async def get_hunt_queries():
    """Get pre-built threat hunting queries"""
    return {
        "queries": [
            {
                "id": "hunt_001",
                "name": "Credential Theft Activity",
                "description": "Hunt for credential dumping, brute force, and password spraying",
                "techniques": ["T1003", "T1110", "T1558"],
                "severity": "high"
            },
            {
                "id": "hunt_002",
                "name": "Lateral Movement Patterns",
                "description": "Identify suspicious lateral movement via RDP, SMB, or WMI",
                "techniques": ["T1021", "T1047", "T1550"],
                "severity": "high"
            },
            {
                "id": "hunt_003",
                "name": "Command & Control Beaconing",
                "description": "Detect C2 communication patterns",
                "techniques": ["T1071", "T1090", "T1095"],
                "severity": "critical"
            },
            {
                "id": "hunt_004",
                "name": "Data Exfiltration",
                "description": "Find large outbound transfers or DNS exfiltration",
                "techniques": ["T1041", "T1048", "T1567"],
                "severity": "critical"
            },
            {
                "id": "hunt_005",
                "name": "Privilege Escalation Attempts",
                "description": "Hunt for privilege escalation techniques",
                "techniques": ["T1068", "T1134", "T1548"],
                "severity": "high"
            }
        ]
    }


@app.post("/api/v1/hunt/run")
async def run_hunt_query(query_id: str):
    """Run a specific hunt query"""
    # Simulate hunt results
    results = []
    technique_map = {
        "hunt_001": ["brute_force", "credential_dumping"],
        "hunt_002": ["rdp_lateral", "smb_lateral"],
        "hunt_003": ["c2_beaconing"],
        "hunt_004": ["data_exfiltration", "large_upload"],
        "hunt_005": ["token_manipulation", "process_injection"]
    }
    
    search_types = technique_map.get(query_id, [])
    for threat in threats_storage:
        if threat.threat_type in search_types:
            results.append(threat.dict())
    
    return {
        "query_id": query_id,
        "results": results,
        "total_findings": len(results),
        "timestamp": datetime.utcnow().isoformat()
    }


# ============= Investigation Endpoints =============

@app.get("/api/v1/investigations")
async def get_investigations():
    """Get all investigations"""
    return {"investigations": investigations_storage, "total": len(investigations_storage)}


@app.post("/api/v1/investigations")
async def create_investigation(
    title: str,
    entity_id: Optional[str] = None,
    detection_id: Optional[str] = None
):
    """Create a new investigation"""
    investigation = {
        "id": str(uuid.uuid4()),
        "title": title,
        "status": "open",
        "priority": "medium",
        "entity_id": entity_id,
        "detection_id": detection_id,
        "assignee": "Security Analyst",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "notes": [],
        "evidence": []
    }
    investigations_storage.append(investigation)
    return investigation


@app.get("/api/v1/investigations/{investigation_id}")
async def get_investigation(investigation_id: str):
    """Get investigation details"""
    for inv in investigations_storage:
        if inv["id"] == investigation_id:
            return inv
    raise HTTPException(status_code=404, detail="Investigation not found")


# ============= Legacy Endpoints =============

@app.get("/api/v1/threats", response_model=List[ThreatDetection])
async def get_threats(limit: int = 50):
    """Get recent threat detections"""
    return sorted(threats_storage, key=lambda x: x.timestamp, reverse=True)[:limit]


@app.get("/api/v1/alerts", response_model=List[AlertNotification])
async def get_alerts(limit: int = 20):
    """Get recent alerts"""
    return sorted(alerts_storage, key=lambda x: x.timestamp, reverse=True)[:limit]


@app.get("/api/v1/metrics/system", response_model=SystemMetrics)
async def get_system_metrics():
    """Get current system metrics"""
    if system_metrics_storage:
        return system_metrics_storage[-1]
    return SystemMetrics(
        cpu_usage=0,
        memory_usage=0,
        network_throughput=0,
        active_connections=0,
        threats_detected_today=0,
        threats_blocked_today=0
    )


# ============= WebSocket Endpoint =============

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await manager.connect(websocket)
    
    try:
        await manager.send_personal_message({
            "type": "connected",
            "message": "Connected to PCDS Enterprise real-time feed",
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)
        
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message({
                "type": "echo",
                "data": data
            }, websocket)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


# ============= Background Tasks =============

async def simulate_realistic_threats():
    """Generate realistic threat scenarios"""
    print("🎯 Starting realistic threat simulation...")
    
    await asyncio.sleep(5)  # Wait for startup
    
    # Create some base entities
    hosts = [
        ("192.168.1.10", "WEB-SERVER-01"),
        ("192.168.1.20", "DB-SERVER-01"),
        ("192.168.1.30", "APP-SERVER-01"),
        ("192.168.1.40", "FILE-SERVER-01"),
        ("10.0.0.5", "WORKSTATION-01"),
        ("10.0.0.15", "WORKSTATION-02"),
    ]
    
    users = [
        ("john.doe", "Administrator"),
        ("jane.smith", "Developer"),
        ("mike.johnson", "Security Analyst"),
    ]
    
    # Create entities
    for ip, hostname in hosts:
        entity_manager.create_or_update_entity(
            EntityType.HOST,
            ip,
            {"hostname": hostname, "os": "Windows Server 2019"}
        )
    
    for username, role in users:
        entity_manager.create_or_update_entity(
            EntityType.USER,
            username,
            {"role": role, "department": "IT"}
        )
    
    while True:
        try:
            # Simulate attack scenario
            if random.random() < 0.4:  # 40% chance
                await simulate_attack_scenario()
            
            await asyncio.sleep(random.uniform(8, 15))
            
        except Exception as e:
            print(f"Error in threat simulation: {e}")
            await asyncio.sleep(5)


async def simulate_attack_scenario():
    """Simulate a realistic multi-stage attack"""
    scenarios = [
        "credential_theft_chain",
        "lateral_movement_chain",
        "ransomware_chain",
        "data_exfiltration_chain"
    ]
    
    scenario = random.choice(scenarios)
    
    if scenario == "credential_theft_chain":
        # Stage 1: Brute force
        await create_detection("brute_force", "10.0.0.5", "critical")
        await asyncio.sleep(2)
        # Stage 2: Credential dumping
        await create_detection("credential_dumping", "10.0.0.5", "critical")
        
    elif scenario == "lateral_movement_chain":
        # Stage 1: Network scan
        await create_detection("network_scan", "192.168.1.10", "medium")
        await asyncio.sleep(3)
        # Stage 2: RDP lateral movement
        await create_detection("rdp_lateral", "192.168.1.10", "high")
        await asyncio.sleep(2)
        # Stage 3: Privilege escalation
        await create_detection("token_manipulation", "192.168.1.20", "high")
        
    elif scenario == "data_exfiltration_chain":
        # Stage 1: File discovery
        await create_detection("file_discovery", "192.168.1.40", "low")
        await asyncio.sleep(4)
        # Stage 2: Data staging
        await create_detection("large_upload", "192.168.1.40", "high")
        await asyncio.sleep(2)
        # Stage 3: Exfiltration
        await create_detection("data_exfiltration", "192.168.1.40", "critical")


async def create_detection(detection_type: str, source_ip: str, severity: str):
    """Create and process a detection"""
    detection = {
        'type': detection_type,
        'severity': severity,
        'timestamp': datetime.now(),
        'confidence': random.uniform(0.75, 0.98),
        'metadata': {
            'source_ip': source_ip,
            'dest_ip': f'10.0.{random.randint(0,255)}.{random.randint(1,255)}',
            'protocol': random.choice(['TCP', 'UDP', 'SMB', 'RDP', 'HTTP']),
            'port': random.choice([22, 80, 443, 445, 3389]),
        }
    }
    
    # Enrich with MITRE
    detection = mitre_attack.enrich_detection(detection)
    
    # Create threat object
    threat = ThreatDetection(
        id=str(uuid.uuid4()),
        title=detection['type'].replace('_', ' ').title(),
        description=f"Detected {detection_type} activity from {source_ip}",
        severity=ThreatSeverity(severity),
        threat_type=detection_type,
        source_ip=source_ip,
        destination_ip=detection['metadata']['dest_ip'],
        risk_score=random.randint(60, 95),
        timestamp=datetime.utcnow()
    )
    
    # Add MITRE data
    threat.mitre = detection.get('mitre', {})
    
    # Store threat
    threats_storage.append(threat)
    
    # Update entity
    entity_id = f"host_{source_ip}"
    entity_manager.create_or_update_entity(EntityType.HOST, source_ip)
    entity_manager.add_detection(entity_id, detection)
    
    # Create alert
    alert = AlertNotification(
        id=str(uuid.uuid4()),
        severity=threat.severity,
        message=f"{threat.title} detected on {source_ip}",
        threat_id=threat.id
    )
    alerts_storage.append(alert)
    
    # Broadcast
    await manager.broadcast_to_all({
        "type": "threat_detected",
        "data": threat.dict()
    })
    
    await manager.broadcast_to_all({
        "type": "alert",
        "data": alert.dict()
    })


async def continuous_threat_detection():
    """Continuously generate and analyze network events"""
    print("🔍 Starting continuous threat detection...")
    
    async def process_event(event: NetworkEvent):
        """Process a network event"""
        try:
            threat = await threat_engine.analyze_network_event(event)
            
            if threat.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH, ThreatSeverity.MEDIUM]:
                threats_storage.append(threat)
                threat_counter.inc()
                
                alert = AlertNotification(
                    id=str(uuid.uuid4()),
                    severity=threat.severity,
                    message=f"{threat.title} from {threat.source_ip}",
                    threat_id=threat.id
                )
                alerts_storage.append(alert)
                
                await manager.broadcast_to_all({
                    "type": "threat_detected",
                    "data": threat.dict()
                })
                
                await manager.broadcast_to_all({
                    "type": "alert",
                    "data": alert.dict()
                })
                
        except Exception as e:
            print(f"Error processing event: {e}")
    
    await data_generator.continuous_generation(process_event, interval=4.0)


async def generate_system_metrics():
    """Generate system metrics periodically"""
    while True:
        metrics = SystemMetrics(
            cpu_usage=random.uniform(20, 80),
            memory_usage=random.uniform(40, 70),
            network_throughput=random.uniform(100, 1000),
            active_connections=manager.get_connection_count(),
            threats_detected_today=len(threats_storage),
            threats_blocked_today=len([t for t in threats_storage if t.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH]])
        )
        
        system_metrics_storage.append(metrics)
        
        if len(system_metrics_storage) > 100:
            system_metrics_storage.pop(0)
            
        await manager.broadcast_to_all({
            "type": "system_metrics",
            "data": metrics.dict()
        })
        
        await asyncio.sleep(5)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)