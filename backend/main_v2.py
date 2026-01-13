"""
PCDS Enterprise v2 - Main FastAPI Application  
Production-ready NDR platform with full API v2 integration + Authentication + ML
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
from typing import Set, List
import json
import random
import uuid
from pydantic import BaseModel

# Configuration
from config.settings import settings
from config.database import init_db

# API v2 Routers
from api.v2.entities import router as entities_router
from api.v2.detections import router as detections_router
from api.v2.campaigns_investigations import campaigns_router, investigations_router
from api.v2.hunt_mitre import hunt_router, mitre_router
from api.v2.dashboard import router as dashboard_router
from api.v2.reports import router as reports_router
from api.v2.model_shadow import router as model_shadow_router
from api.v2.auth import router as auth_v2_router  # NEW: V2 Auth API
from api.v2.kafka_api import router as kafka_router  # Kafka Enterprise API
from api.v2.playbooks_api import router as playbooks_router  # Automated Playbooks
from api.v2.response_api import router as response_router  # Decision Engine
from api.v2.edr_api import router as edr_router  # EDR Agent API
from api.v2.ml_dashboard import router as ml_dashboard_router  # ML Dashboard API
from api.v2.threat_hunting import router as threat_hunting_router  # Threat Hunting API
from api.v2.siem_integration import router as siem_router  # SIEM Integration API
from api.v2.realtime_api import router as realtime_router  # Real-time Pipeline API
from api.v2.xai_api import router as xai_router  # Explainable AI API
from api.v2.adversarial_api import router as adversarial_router  # Adversarial Defense API
from api.v2.soar_api import router as soar_router  # SOAR API
from api.v2.network_anomaly_api import router as network_anomaly_router  # Network Anomaly API
from api.v2.advanced_ml_api import router as advanced_ml_router  # Paper 4: Phishing, RL, CNN
from api.v2.auth_api import router as auth_router  # Enterprise Authentication
from api.v2.azure_ai_api import router as azure_ai_router  # Azure AI (Imagine Cup)
from api.v2.azure_services import router as azure_services_router  # Azure Blob Storage (Imagine Cup)

# ML Engine
from ml.anomaly_detector import anomaly_detector
from ml.ueba import ueba_engine
from automation.playbooks import playbook_engine


# ============================================
# WebSocket Connection Manager
# ============================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
    
    async def broadcast(self, message: dict):
        if not self.active_connections:
            return
        message_json = json.dumps(message)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message_json)
            except:
                self.active_connections.discard(connection)

ws_manager = ConnectionManager()


# ============================================
# Background Tasks
# ============================================

async def system_heartbeat():
    while True:
        await asyncio.sleep(settings.WS_HEARTBEAT_INTERVAL)
        await ws_manager.broadcast({"type": "heartbeat", "timestamp": datetime.utcnow().isoformat()})

async def simulate_threats():
    while True:
        await asyncio.sleep(random.randint(5, 15))
        
        # Create a mock detection
        detection_id = str(uuid.uuid4())
        detection_data = {
            "id": detection_id,
            "title": "Simulated Ransomware Attack",
            "description": "Ransomware behavior detected on host",
            "severity": "critical",
            "confidence_score": 0.95,
            "entity_id": "host-001" # Mock entity
        }
        
        # Evaluate against playbooks
        playbook_engine.evaluate(detection_data)
        
        await ws_manager.broadcast({"type": "new_detection", "data": {"id": detection_id}})

async def run_ueba_cycle():
    """Run periodic UEBA analysis"""
    while True:
        try:
            # Run UEBA cycle
            await ueba_engine.run_cycle()
            # Wait 60 seconds before next cycle
            await asyncio.sleep(60)
        except Exception as e:
            print(f"❌ UEBA Background Task Error: {e}")
            await asyncio.sleep(60)

# ============================================
# App Lifespan
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting PCDS Enterprise v2.0")
    
    # Initialize Database
    if init_db():
        print("✅ Database initialized")
    
    # Start Background Tasks
    asyncio.create_task(system_heartbeat())
    asyncio.create_task(simulate_threats())
    asyncio.create_task(run_ueba_cycle())
    print("✅ Background tasks started (Heartbeat, Sim, UEBA)")
    
    yield
    print("🛑 Shutting down")


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="PCDS Enterprise",
    version="2.0",
    lifespan=lifespan,
    docs_url="/api/docs"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers - AUTH FIRST
app.include_router(auth_v2_router, prefix=settings.API_V2_PREFIX)  # V2 Auth API

# API v2 Routers
app.include_router(entities_router, prefix="/api/v2/entities", tags=["Entities"])
app.include_router(detections_router, prefix="/api/v2/detections", tags=["Detections"])
app.include_router(campaigns_router, prefix=settings.API_V2_PREFIX)
app.include_router(investigations_router, prefix="/api/v2/investigations", tags=["Investigations"])
app.include_router(hunt_router, prefix=settings.API_V2_PREFIX)
app.include_router(mitre_router, prefix=settings.API_V2_PREFIX)
app.include_router(dashboard_router, prefix="/api/v2/dashboard", tags=["Dashboard"])
app.include_router(reports_router, prefix="/api/v2/reports", tags=["Reports"])
app.include_router(kafka_router, prefix=settings.API_V2_PREFIX)  # Kafka Enterprise API
app.include_router(playbooks_router, prefix="/api/v2/playbooks", tags=["Playbooks"])  # Automated Playbooks
app.include_router(response_router, prefix=settings.API_V2_PREFIX)  # Decision Engine
app.include_router(edr_router, prefix="/api/v2/edr", tags=["EDR"])  # EDR Agent API
app.include_router(ml_dashboard_router, prefix="/api/v2/ml", tags=["ML Dashboard"])  # ML Dashboard API
app.include_router(threat_hunting_router, prefix="/api/v2/hunt", tags=["Threat Hunting"])  # Threat Hunting API
app.include_router(siem_router, prefix="/api/v2/siem", tags=["SIEM"])  # SIEM Integration API
app.include_router(model_shadow_router, prefix="/api/v2/ml/shadow", tags=["ML Shadow Mode"]) # ML Shadow Mode API
app.include_router(realtime_router, prefix="/api/v2/realtime", tags=["Real-time Pipeline"])  # Real-time ML Pipeline
app.include_router(xai_router, prefix="/api/v2", tags=["Explainable AI"])  # Explainable AI
app.include_router(adversarial_router, prefix="/api/v2", tags=["Adversarial Defense"])  # Adversarial Defense
app.include_router(soar_router, prefix="/api/v2", tags=["SOAR"])  # Security Orchestration
app.include_router(network_anomaly_router, prefix="/api/v2", tags=["Network Anomaly"])  # Network Anomaly Detection
app.include_router(advanced_ml_router, prefix="/api/v2", tags=["Advanced ML"])  # Paper 4: Phishing, RL, CNN
app.include_router(auth_router, prefix="/api/v2", tags=["Authentication"])  # Enterprise Auth
app.include_router(azure_ai_router, prefix="/api/v2/azure", tags=["Azure AI"])  # Imagine Cup - Microsoft AI
app.include_router(azure_services_router)  # Azure Blob Storage - Imagine Cup 2nd service


# ============================================
# ML Endpoints
# ============================================

class TrainingData(BaseModel):
    features: List[List[float]]

@app.post("/api/v2/ml/train")
async def train_model(data: TrainingData):
    """Trigger online training for the anomaly detector"""
    import numpy as np
    features_list = [np.array(f, dtype=np.float32) for f in data.features]
    loss = anomaly_detector.train(features_list)
    return {
        "status": "trained", 
        "loss": loss, 
        "model_version": anomaly_detector.model_version,
        "timestamp": datetime.utcnow().isoformat()
    }


# ============================================
# Core Endpoints
# ============================================

@app.get("/")
async def root():
    return {"name": "PCDS Enterprise", "version": "2.0", "auth": "/api/auth/login"}

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ============================================
# Network Monitoring Endpoints
# ============================================

from network_monitor import network_monitor, start_network_monitoring, stop_network_monitoring

@app.post("/api/v2/network/start")
async def start_network():
    """Start real network monitoring"""
    start_network_monitoring()
    return {"status": "started", "message": "Network monitoring activated"}

@app.post("/api/v2/network/stop")
async def stop_network():
    """Stop network monitoring"""
    stop_network_monitoring()
    return {"status": "stopped", "message": "Network monitoring deactivated"}

@app.get("/api/v2/network/stats")
async def get_network_stats():
    """Get network monitoring statistics"""
    return network_monitor.get_stats()

@app.get("/api/v2/network/connections")
async def get_connections(limit: int = 50):
    """Get active network connections"""
    return {"connections": network_monitor.get_connections(limit)}

@app.get("/api/v2/network/events")
async def get_network_events(limit: int = 100):
    """Get network events for live feed"""
    return {"events": network_monitor.get_events(limit)}

@app.get("/api/v2/network/suspicious")
async def get_suspicious():
    """Get suspicious connections only"""
    return {"suspicious": network_monitor.get_suspicious()}


# ============================================
# WebSocket Endpoint
# ============================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


# ============================================
# Run
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_v2:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)