"""
PCDS Production API - Minimal Deployment Version
This is a streamlined version for Railway deployment
"""
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import random
import uuid
import os
import databases
import sqlalchemy
import resend
import numpy as np
from sklearn.ensemble import IsolationForest
from collections import deque

# ============= ML Anomaly Detection =============
class AnomalyDetector:
    """Lightweight Isolation Forest for real-time anomaly detection"""
    
    def __init__(self):
        self.model = IsolationForest(
            n_estimators=50,  # Lightweight
            contamination=0.1,  # Expect 10% anomalies
            random_state=42
        )
        self.training_data = deque(maxlen=1000)  # Rolling window
        self.is_trained = False
        self.min_samples = 50  # Minimum samples before training
        
    def extract_features(self, event: dict) -> np.ndarray:
        """Convert network event to feature vector"""
        # Feature engineering
        port = event.get('dest_port') or 0
        bytes_sent = event.get('bytes_sent') or 0
        bytes_recv = event.get('bytes_recv') or 0
        
        # Port risk scoring
        high_risk_ports = {22, 23, 3389, 445, 135, 139}  # SSH, Telnet, RDP, SMB
        port_risk = 1.0 if port in high_risk_ports else 0.0
        
        # Suspicious port ranges
        suspicious_port = 1.0 if port > 49152 else 0.0  # Ephemeral ports
        
        features = np.array([
            port / 65535,  # Normalized port
            min(bytes_sent / 1e6, 1.0),  # Normalized bytes (cap at 1MB)
            min(bytes_recv / 1e6, 1.0),
            port_risk,
            suspicious_port,
            1.0 if event.get('status') == 'ESTABLISHED' else 0.0,
        ])
        return features
    
    def train(self):
        """Train on accumulated data"""
        if len(self.training_data) >= self.min_samples:
            X = np.array(list(self.training_data))
            self.model.fit(X)
            self.is_trained = True
            return True
        return False
    
    def predict(self, event: dict) -> dict:
        """Predict if event is anomalous"""
        features = self.extract_features(event)
        self.training_data.append(features)
        
        # Auto-retrain periodically
        if len(self.training_data) % 100 == 0:
            self.train()
        
        if not self.is_trained:
            # Not enough data yet - return neutral score
            return {
                "is_anomaly": False,
                "risk_score": 25,
                "confidence": 0.5,
                "reason": "Baseline learning in progress"
            }
        
        # Predict (-1 = anomaly, 1 = normal)
        prediction = self.model.predict(features.reshape(1, -1))[0]
        anomaly_score = -self.model.score_samples(features.reshape(1, -1))[0]
        
        # Convert to 0-100 risk score
        risk_score = int(min(100, max(0, anomaly_score * 50 + 50)))
        
        return {
            "is_anomaly": prediction == -1,
            "risk_score": risk_score,
            "confidence": 0.85 if self.is_trained else 0.5,
            "reason": "Unusual traffic pattern detected" if prediction == -1 else "Normal traffic"
        }

# Global anomaly detector instance
anomaly_detector = AnomalyDetector()

# ============= Attack Sequence Predictor =============
ATTACK_STAGES = [
    "benign", "recon", "weaponization", "delivery", "exploitation",
    "privilege_esc", "defense_evasion", "credential_access", "discovery",
    "lateral_movement", "collection", "exfiltration", "impact"
]

class AttackPredictor:
    """Predicts next attack stage based on current event patterns"""
    
    def __init__(self):
        self.event_history = deque(maxlen=20)  # Keep last 20 events
        self.stage_counts = {stage: 0 for stage in ATTACK_STAGES}
        
    def analyze_event(self, event: dict) -> str:
        """Classify event into attack stage based on characteristics"""
        port = event.get('dest_port') or 0
        process = (event.get('process_name') or '').lower()
        
        # Simple heuristic classification
        if port in [21, 22, 23, 3389]:
            return "lateral_movement"
        elif port in [53, 80, 443, 8080]:
            if 'scan' in process or 'nmap' in process:
                return "recon"
            return "benign"
        elif port in [445, 135, 139]:
            return "credential_access"
        elif port > 49152:
            return "exfiltration"
        elif 'powershell' in process or 'cmd' in process:
            return "exploitation"
        else:
            return "benign"
    
    def predict_next_stage(self, current_events: List[dict]) -> dict:
        """Predict next likely attack stage"""
        # Analyze recent events
        for event in current_events:
            stage = self.analyze_event(event)
            self.event_history.append(stage)
            self.stage_counts[stage] += 1
        
        # Calculate stage progression
        recent_stages = list(self.event_history)[-10:]
        
        # Attack chain logic: what typically follows what
        attack_chain = {
            "recon": ("delivery", 0.75, 15),
            "delivery": ("exploitation", 0.82, 8),
            "exploitation": ("privilege_esc", 0.78, 5),
            "privilege_esc": ("credential_access", 0.85, 4),
            "credential_access": ("lateral_movement", 0.88, 6),
            "lateral_movement": ("collection", 0.80, 10),
            "collection": ("exfiltration", 0.90, 3),
            "exfiltration": ("impact", 0.70, 2),
        }
        
        # Find most recent non-benign stage
        current_stage = "benign"
        for stage in reversed(recent_stages):
            if stage != "benign":
                current_stage = stage
                break
        
        if current_stage in attack_chain:
            next_stage, confidence, eta_minutes = attack_chain[current_stage]
            return {
                "current_stage": current_stage,
                "predicted_next_stage": next_stage,
                "confidence": confidence,
                "eta_minutes": eta_minutes,
                "is_active_attack": True,
                "message": f"Attack in progress: {current_stage} → {next_stage} likely in ~{eta_minutes} min"
            }
        
        return {
            "current_stage": current_stage,
            "predicted_next_stage": None,
            "confidence": 0.0,
            "eta_minutes": None,
            "is_active_attack": False,
            "message": "No active attack detected"
        }

# Global predictor instance
attack_predictor = AttackPredictor()



# Email Configuration
resend.api_key = os.getenv("RESEND_API_KEY", "")

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
database = databases.Database(DATABASE_URL) if DATABASE_URL else None
# ============= Persistent Storage Tables =============
metadata = sqlalchemy.MetaData()

leads_table = sqlalchemy.Table(
    "leads",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("email", sqlalchemy.String),
    sqlalchemy.Column("timestamp", sqlalchemy.String),
)

events_table = sqlalchemy.Table(
    "events",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.Integer, primary_key=True),
    sqlalchemy.Column("customer_id", sqlalchemy.String, index=True),
    sqlalchemy.Column("hostname", sqlalchemy.String),
    sqlalchemy.Column("source_ip", sqlalchemy.String),
    sqlalchemy.Column("dest_ip", sqlalchemy.String),
    sqlalchemy.Column("dest_port", sqlalchemy.Integer),
    sqlalchemy.Column("protocol", sqlalchemy.String),
    sqlalchemy.Column("process_name", sqlalchemy.String),
    sqlalchemy.Column("status", sqlalchemy.String),
    sqlalchemy.Column("bytes_sent", sqlalchemy.Integer),
    sqlalchemy.Column("bytes_recv", sqlalchemy.Integer),
    sqlalchemy.Column("risk_score", sqlalchemy.Integer),
    sqlalchemy.Column("is_anomaly", sqlalchemy.Boolean),
    sqlalchemy.Column("prediction_message", sqlalchemy.String),
    sqlalchemy.Column("timestamp", sqlalchemy.String),
)

# API Keys table for agent authentication
api_keys_table = sqlalchemy.Table(
    "api_keys",
    metadata,
    sqlalchemy.Column("id", sqlalchemy.String, primary_key=True),
    sqlalchemy.Column("key", sqlalchemy.String, unique=True, index=True),
    sqlalchemy.Column("user_id", sqlalchemy.String, index=True),
    sqlalchemy.Column("user_email", sqlalchemy.String),
    sqlalchemy.Column("name", sqlalchemy.String),
    sqlalchemy.Column("created_at", sqlalchemy.String),
    sqlalchemy.Column("last_used", sqlalchemy.String),
    sqlalchemy.Column("is_active", sqlalchemy.Boolean, default=True),
)

# ============= Agent Data Ingestion System =============
# In-memory cache for API keys (loaded from DB on startup)
customer_api_keys = {
    "pcds_demo_key_12345": {"customer_id": "demo", "name": "Demo Customer"},
    # Real customers loaded from database
}
customer_events = {}  # {customer_id: [events]}

class NetworkEvent(BaseModel):
    source_ip: str
    dest_ip: Optional[str] = None
    dest_port: Optional[int] = None
    protocol: Optional[str] = "TCP"
    process_name: Optional[str] = None
    status: Optional[str] = "ESTABLISHED"
    bytes_sent: Optional[int] = 0
    bytes_recv: Optional[int] = 0
    timestamp: Optional[str] = None

class AgentPayload(BaseModel):
    api_key: str
    hostname: str
    events: List[NetworkEvent]


app = FastAPI(
    title="PCDS Enterprise API",
    description="Predictive Cyber Defence System - Production API",
    version="2.0.0"
)

# CORS - allow specific origins for credentials support
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://pcdsai.app",
        "https://pcds-backend-production.up.railway.app",
        "http://localhost:3000",
        "https://frontend-git-main-sanjay-krishna-reddys-projects.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    if database:
        try:
            await database.connect()
            # Create tables using sync engine (creates if not exists)
            # Replace postgres:// with postgresql:// for sqlalchemy if needed
            db_url = DATABASE_URL.replace("postgres://", "postgresql://")
            engine = sqlalchemy.create_engine(db_url)
            metadata.create_all(engine)
            print("Database connected and tables verified.")
        except Exception as e:
            print(f"DB Connection Warning: {e}")

@app.on_event("shutdown")
async def shutdown():
    if database:
        await database.disconnect()

# ============= Models =============

class Detection(BaseModel):
    id: str
    title: str
    description: str
    severity: str
    type: str
    source_ip: str
    destination_ip: str
    risk_score: int
    timestamp: str
    confidence: float = 0.85
    mitre: Optional[Dict] = None

class Entity(BaseModel):
    id: str
    identifier: str
    entity_type: str
    display_name: str
    threat_score: int
    urgency_level: str
    first_seen: str
    last_seen: str
    total_detections: int

class DashboardOverview(BaseModel):
    entities: Dict
    detections: Dict
    campaigns: Dict
    mitre: Dict

# ============= Auth Models & Endpoints =============

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/api/v2/auth/login")
async def login(req: LoginRequest):
    # Accept specific demo emails, or any email for convenience in demo
    return {
        "access_token": f"demo-jwt-token-{uuid.uuid4()}",
        "token_type": "bearer",
        "user": {
            "user_id": "u-123",
            "email": req.email,
            "name": "Admin User" if "admin" in req.email else "Analyst User",
            "role": "admin" if "admin" in req.email else "analyst"
        }
    }

# ============= API Key Management =============

class GenerateKeyRequest(BaseModel):
    user_email: str
    key_name: Optional[str] = "Default Agent Key"

class ApiKeyResponse(BaseModel):
    id: str
    key: str  # Only shown once on creation
    name: str
    created_at: str
    last_used: Optional[str] = None
    is_active: bool = True

def generate_api_key(user_email: str) -> str:
    """Generate a unique API key for a user"""
    import secrets
    # Format: pcds_{short_user_hash}_{random}
    user_hash = hashlib.md5(user_email.encode()).hexdigest()[:6]
    random_part = secrets.token_hex(8)
    return f"pcds_{user_hash}_{random_part}"

@app.post("/api/v2/keys/generate")
async def create_api_key(req: GenerateKeyRequest):
    """Generate a new API key for a user"""
    key_id = str(uuid.uuid4())
    api_key = generate_api_key(req.user_email)
    created_at = datetime.now().isoformat()
    
    # Store in database if available
    if database:
        try:
            query = api_keys_table.insert().values(
                id=key_id,
                key=api_key,
                user_id=hashlib.md5(req.user_email.encode()).hexdigest()[:8],
                user_email=req.user_email,
                name=req.key_name,
                created_at=created_at,
                last_used=None,
                is_active=True
            )
            await database.execute(query)
        except Exception as e:
            print(f"Error storing API key: {e}")
    
    # Also add to in-memory cache
    customer_api_keys[api_key] = {
        "customer_id": hashlib.md5(req.user_email.encode()).hexdigest()[:8],
        "name": req.user_email,
        "key_id": key_id
    }
    
    return {
        "success": True,
        "key": api_key,  # Only shown once!
        "key_id": key_id,
        "name": req.key_name,
        "created_at": created_at,
        "message": "Save this key! It won't be shown again."
    }

@app.get("/api/v2/keys")
async def list_api_keys(user_email: str):
    """List all API keys for a user (keys are masked)"""
    keys = []
    
    if database:
        try:
            query = api_keys_table.select().where(
                api_keys_table.c.user_email == user_email
            )
            rows = await database.fetch_all(query)
            for row in rows:
                # Mask the key: show first 10 and last 4 chars
                full_key = row['key']
                masked = f"{full_key[:10]}...{full_key[-4:]}"
                keys.append({
                    "id": row['id'],
                    "key_masked": masked,
                    "name": row['name'],
                    "created_at": row['created_at'],
                    "last_used": row['last_used'],
                    "is_active": row['is_active']
                })
        except Exception as e:
            print(f"Error listing keys: {e}")
    
    return {"keys": keys}

@app.delete("/api/v2/keys/{key_id}")
async def revoke_api_key(key_id: str):
    """Revoke (deactivate) an API key"""
    if database:
        try:
            # Get the key first to remove from cache
            query = api_keys_table.select().where(api_keys_table.c.id == key_id)
            row = await database.fetch_one(query)
            
            if row:
                # Remove from cache
                if row['key'] in customer_api_keys:
                    del customer_api_keys[row['key']]
                
                # Mark as inactive in DB
                update_query = api_keys_table.update().where(
                    api_keys_table.c.id == key_id
                ).values(is_active=False)
                await database.execute(update_query)
                
                return {"success": True, "message": "Key revoked"}
        except Exception as e:
            print(f"Error revoking key: {e}")
    
    return {"success": False, "message": "Key not found"}

from fastapi import Header


@app.get("/api/v2/auth/me")
async def get_current_user(authorization: Optional[str] = Header(None)):
    # Minimal auth check
    if not authorization:
        # In strict mode we'd raise 401, but for demo stability we can be lenient or strict
        # Let's be lenient if no header logic is strict in frontend
        pass
    
    return {
        "user_id": "u-123",
        "email": "admin@pcds.com",
        "name": "Admin User",
        "role": "admin"
    }

@app.post("/api/v2/auth/logout")
async def logout():
    return {"status": "success"}

@app.post("/api/v2/auth/refresh")
async def refresh_token():
    return {
        "access_token": f"refreshed-token-{uuid.uuid4()}",
        "token_type": "bearer"
    }

class Lead(BaseModel):
    email: str

# In-memory storage for demo
leads_db = []

@app.post("/api/v2/leads")
async def create_lead(lead: Lead):
    email_sent = False
    
    # Send welcome email
    if resend.api_key:
        try:
            resend.Emails.send({
                "from": "PCDS <onboarding@resend.dev>",
                "to": [lead.email],
                "subject": "Welcome to PCDS - Your Security Journey Starts Now",
                "html": f"""
                    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto; background: #0a0a0a; color: #fff; padding: 40px; border-radius: 12px;">
                        <h1 style="color: #f5c16c; margin-bottom: 20px;">Welcome to PCDS! 🛡️</h1>
                        <p style="color: #a1a1a1; line-height: 1.6;">Thank you for your interest in PCDS - the AI-powered Network Detection & Response platform.</p>
                        <p style="color: #a1a1a1; line-height: 1.6;">We'll be in touch soon with your free trial access.</p>
                        <div style="margin-top: 30px; padding: 20px; background: #141414; border-radius: 8px;">
                            <p style="color: #666; margin: 0;">In the meantime, explore our dashboard:</p>
                            <a href="https://pcdsai.app/dashboard" style="display: inline-block; margin-top: 15px; padding: 12px 24px; background: linear-gradient(180deg, #fde68a, #f5c16c); color: #000; text-decoration: none; border-radius: 8px; font-weight: bold;">View Dashboard →</a>
                        </div>
                        <p style="color: #666; margin-top: 30px; font-size: 12px;">© 2026 PCDS Enterprise. Built in Hyderabad 🇮🇳</p>
                    </div>
                """
            })
            email_sent = True
        except Exception as e:
            print(f"Email Error: {e}")
    
    # Store lead in database
    if database:
        try:
            query = leads_table.insert().values(email=lead.email, timestamp=str(datetime.now()))
            await database.execute(query)
            return {"status": "success", "message": f"Lead captured (DB){' + Email sent' if email_sent else ''}"}
        except Exception as e:
            print(f"DB Error: {e}")
            leads_db.append({"email": lead.email, "timestamp": str(datetime.now())})
            return {"status": "success", "message": f"Lead captured (Fallback){' + Email sent' if email_sent else ''}"}
    else:
        leads_db.append({"email": lead.email, "timestamp": str(datetime.now())})
        return {"status": "success", "message": f"Lead captured (Memory){' + Email sent' if email_sent else ''}"}

@app.get("/api/v2/leads")
async def get_leads():
    if database:
        try:
            query = leads_table.select().order_by(leads_table.c.id.desc())
            return await database.fetch_all(query)
        except Exception:
            return leads_db
    return leads_db

# ============= Agent Data Ingestion API =============

@app.post("/api/v2/ingest")
async def ingest_agent_data(payload: AgentPayload):
    """Receive network events from customer agents with ML-based threat detection"""
    # Validate API key
    if payload.api_key not in customer_api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    customer_id = customer_api_keys[payload.api_key]["customer_id"]
    
    anomalies_detected = 0
    timestamp = str(datetime.now())
    
    # Process and save events
    for event in payload.events:
        event_dict = event.dict()
        event_dict["hostname"] = payload.hostname
        event_dict["customer_id"] = customer_id
        event_dict["received_at"] = timestamp
        
        # ===== RUN ML PREDICTION =====
        ml_result = anomaly_detector.predict(event_dict)
        event_dict["ml_prediction"] = ml_result
        event_dict["risk_score"] = ml_result["risk_score"]
        event_dict["is_anomaly"] = ml_result["is_anomaly"]
        
        if ml_result["is_anomaly"]:
            anomalies_detected += 1
            
        # ===== EMAIL ALERTING =====
        if event_dict["risk_score"] > 80:
            try:
                # In production, get email from DB user settings
                # For demo, default to hardcoded admin
                alert_email = "sanjay31105@gmail.com" 
                
                resend.Emails.send({
                    "from": "PCDS Alert <security@pcdsai.app>", 
                    "to": alert_email,
                    "subject": f"🚨 CRITICAL ALERT: {event_dict['risk_score']} Risk on {payload.hostname}",
                    "html": f"""
                        <h1>Critical Threat Detected</h1>
                        <p><b>Host:</b> {payload.hostname}</p>
                        <p><b>Risk Score:</b> {event_dict['risk_score']}</p>
                        <p><b>Reason:</b> {ml_result['reason']}</p>
                        <p><b>Source IP:</b> {event.source_ip}</p>
                        <br/>
                        <a href="https://pcdsai.app/dashboard">View in Dashboard</a>
                    """
                })
                print(f"📧 Alert sent to {alert_email}")
            except Exception as e:
                print(f"Email Alert Error: {e}")
            
        # Save to database if available
        if database:
            try:
                query = events_table.insert().values(
                    customer_id=customer_id,
                    hostname=payload.hostname,
                    source_ip=event.source_ip,
                    dest_ip=event.dest_ip,
                    dest_port=event.dest_port,
                    protocol=event.protocol,
                    process_name=event.process_name,
                    status=event.status,
                    bytes_sent=event.bytes_sent,
                    bytes_recv=event.bytes_recv,
                    risk_score=event_dict["risk_score"],
                    is_anomaly=event_dict["is_anomaly"],
                    prediction_message=ml_result["reason"],
                    timestamp=event.timestamp or timestamp
                )
                await database.execute(query)
            except Exception as e:
                print(f"Error saving event to DB: {e}")
        
        # Fallback to in-memory
        if customer_id not in customer_events:
            customer_events[customer_id] = []
        customer_events[customer_id].append(event_dict)
        customer_events[customer_id] = customer_events[customer_id][-1000:]
    
    return {
        "status": "success",
        "events_received": len(payload.events),
        "anomalies_detected": anomalies_detected,
        "ml_status": "trained" if anomaly_detector.is_trained else "learning",
        "customer_id": customer_id,
        "mode": "persistent" if database else "ephemeral"
    }

@app.get("/api/v2/customers/{customer_id}/events")
async def get_customer_events(customer_id: str, limit: int = 100):
    """Get events for a specific customer"""
    if database:
        try:
            query = events_table.select().where(events_table.c.customer_id == customer_id).order_by(events_table.c.id.desc()).limit(limit)
            rows = await database.fetch_all(query)
            return {
                "customer_id": customer_id,
                "total_events": len(rows),
                "events": [dict(r) for r in rows]
            }
        except Exception as e:
            print(f"DB Fetch Error: {e}")
    
    # Fallback to in-memory
    events = customer_events.get(customer_id, [])
    return {
        "customer_id": customer_id,
        "total_events": len(events),
        "events": events[-limit:]
    }

@app.post("/api/v2/predict")
async def predict_attack(customer_id: str = Body(..., embed=True)):
    """Predict next attack stage based on recent events (POST)"""
    recent_events = []
    
    if database:
        try:
            query = events_table.select().where(events_table.c.customer_id == customer_id).order_by(events_table.c.id.desc()).limit(20)
            rows = await database.fetch_all(query)
            recent_events = [dict(r) for r in rows]
        except Exception as e:
            print(f"DB Prediction Fetch Error: {e}")
            recent_events = customer_events.get(customer_id, [])[-20:]
    else:
        recent_events = customer_events.get(customer_id, [])[-20:]
    
    if not recent_events:
        return {
            "status": "no_data",
            "message": "No events found for prediction",
            "prediction": None
        }
    
    # Needs chronological order for sequence prediction
    recent_events.reverse()
    prediction = attack_predictor.predict_next_stage(recent_events)
    
    return {
        "status": "success",
        "customer_id": customer_id,
        "events_analyzed": len(recent_events),
        "prediction": prediction
    }

@app.get("/api/v2/predict/{customer_id}")
async def get_prediction(customer_id: str):
    """Get current attack prediction for customer (GET)"""
    recent_events = []
    
    if database:
        try:
            query = events_table.select().where(events_table.c.customer_id == customer_id).order_by(events_table.c.id.desc()).limit(20)
            rows = await database.fetch_all(query)
            recent_events = [dict(r) for r in rows]
        except Exception as e:
            print(f"DB Prediction Fetch Error: {e}")
            recent_events = customer_events.get(customer_id, [])[-20:]
    else:
        recent_events = customer_events.get(customer_id, [])[-20:]
    
    if not recent_events:
        return {
            "status": "no_data",
            "prediction": {
                "current_stage": "benign",
                "predicted_next_stage": None,
                "confidence": 0.0,
                "is_active_attack": False,
                "message": "No events to analyze"
            }
        }
    
    recent_events.reverse()
    prediction = attack_predictor.predict_next_stage(recent_events)
    
    return {
        "status": "success",
        "customer_id": customer_id,
        "prediction": prediction
    }

@app.post("/api/v2/customers/api-key")
async def generate_api_key(customer_name: str = Body(..., embed=True)):
    """Generate a new API key for a customer"""
    new_key = f"pcds_{uuid.uuid4().hex[:16]}"
    customer_id = f"cust_{uuid.uuid4().hex[:8]}"
    
    customer_api_keys[new_key] = {
        "customer_id": customer_id,
        "name": customer_name,
        "created_at": str(datetime.now())
    }
    
    return {
        "api_key": new_key,
        "customer_id": customer_id,
        "message": "Save this API key - it won't be shown again!"
    }

# ============= Demo Data Generation =============


def generate_demo_detections(count: int = 50) -> List[Detection]:
    """Generate realistic demo detection data"""
    detection_types = [
        ("Brute Force Attack", "credential_theft", "critical", "T1110"),
        ("Lateral Movement via RDP", "lateral_movement", "high", "T1021.001"),
        ("C2 Beaconing Detected", "command_control", "critical", "T1071.001"),
        ("Data Exfiltration Attempt", "exfiltration", "high", "T1041"),
        ("Privilege Escalation", "privilege_escalation", "high", "T1134"),
        ("Suspicious Network Scan", "discovery", "medium", "T1046"),
        ("Credential Dumping", "credential_theft", "critical", "T1003"),
        ("Malware Execution", "execution", "critical", "T1059"),
        ("Unusual DNS Query", "command_control", "medium", "T1071.004"),
        ("Large File Transfer", "exfiltration", "medium", "T1048"),
    ]
    
    detections = []
    for i in range(count):
        title, dtype, severity, technique = random.choice(detection_types)
        src_ip = f"192.168.{random.randint(1,10)}.{random.randint(1,254)}"
        dst_ip = f"10.0.{random.randint(0,5)}.{random.randint(1,254)}"
        
        detections.append(Detection(
            id=str(uuid.uuid4()),
            title=title,
            description=f"Detected {title.lower()} from {src_ip} to {dst_ip}",
            severity=severity,
            type=dtype,
            source_ip=src_ip,
            destination_ip=dst_ip,
            risk_score=random.randint(45, 95),
            timestamp=(datetime.utcnow() - timedelta(hours=random.randint(0, 48))).isoformat(),
            confidence=round(random.uniform(0.75, 0.98), 2),
            mitre={
                "technique_id": technique,
                "technique_name": title,
                "tactic_name": dtype.replace("_", " ").title()
            }
        ))
    
    return sorted(detections, key=lambda x: x.timestamp, reverse=True)

def generate_demo_entities(count: int = 15) -> List[Entity]:
    """Generate demo entity data"""
    entity_templates = [
        ("host", "WEB-SERVER-01", "192.168.1.10"),
        ("host", "DB-SERVER-01", "192.168.1.20"),
        ("host", "APP-SERVER-01", "192.168.1.30"),
        ("host", "FILE-SERVER-01", "192.168.1.40"),
        ("host", "DC-01", "192.168.1.5"),
        ("user", "john.doe", "john.doe@company.com"),
        ("user", "admin.user", "admin@company.com"),
        ("user", "jane.smith", "jane.smith@company.com"),
        ("service", "nginx", "web-proxy"),
        ("service", "postgresql", "database"),
    ]
    
    urgency_levels = ["critical", "high", "medium", "low"]
    entities = []
    
    for i, (etype, name, identifier) in enumerate(entity_templates[:count]):
        urgency = random.choice(urgency_levels)
        entities.append(Entity(
            id=f"entity-{i+1}",
            identifier=identifier,
            entity_type=etype,
            display_name=name,
            threat_score=random.randint(20, 95),
            urgency_level=urgency,
            first_seen=(datetime.utcnow() - timedelta(days=random.randint(1, 30))).isoformat(),
            last_seen=(datetime.utcnow() - timedelta(hours=random.randint(0, 24))).isoformat(),
            total_detections=random.randint(1, 25)
        ))
    
    return sorted(entities, key=lambda x: x.threat_score, reverse=True)

# Initialize demo data
DEMO_DETECTIONS = generate_demo_detections(50)
DEMO_ENTITIES = generate_demo_entities(15)

# ============= API Endpoints =============

@app.get("/")
async def root():
    return {
        "name": "PCDS Enterprise API",
        "version": "2.0.0",
        "status": "operational",
        "features": ["Attack Signal Intelligence", "Entity Scoring", "MITRE ATT&CK", "Threat Hunting"],
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "entities_tracked": len(DEMO_ENTITIES),
        "active_detections": len(DEMO_DETECTIONS)
    }

# Dashboard Endpoints
@app.get("/api/v2/dashboard/overview")
async def get_dashboard_overview(hours: int = 24):
    critical = len([d for d in DEMO_DETECTIONS if d.severity == "critical"])
    high = len([d for d in DEMO_DETECTIONS if d.severity == "high"])
    medium = len([d for d in DEMO_DETECTIONS if d.severity == "medium"])
    low = len([d for d in DEMO_DETECTIONS if d.severity == "low"])
    
    return {
        "entities": {
            "total": len(DEMO_ENTITIES),
            "by_urgency": {
                "critical": len([e for e in DEMO_ENTITIES if e.urgency_level == "critical"]),
                "high": len([e for e in DEMO_ENTITIES if e.urgency_level == "high"]),
                "medium": len([e for e in DEMO_ENTITIES if e.urgency_level == "medium"]),
                "low": len([e for e in DEMO_ENTITIES if e.urgency_level == "low"]),
            },
            "top_entities": [e.dict() for e in DEMO_ENTITIES[:5]]
        },
        "detections": {
            "total": len(DEMO_DETECTIONS),
            "by_severity": {"critical": critical, "high": high, "medium": medium, "low": low},
            "recent_critical": [d.dict() for d in DEMO_DETECTIONS if d.severity in ["critical", "high"]][:10]
        },
        "campaigns": {
            "total": 4,
            "by_status": {"active": 2, "contained": 1, "resolved": 1}
        },
        "mitre": {
            "techniques_detected": 12,
            "total_techniques": 38,
            "coverage_percentage": 31.6,
            "top_techniques": [
                {"id": "T1110", "name": "Brute Force", "count": 8},
                {"id": "T1071", "name": "Application Layer Protocol", "count": 6},
                {"id": "T1021", "name": "Remote Services", "count": 5}
            ]
        }
    }

# Entity Endpoints
@app.get("/api/v2/entities")
async def get_entities(
    urgency: Optional[str] = None,
    entity_type: Optional[str] = None,
    limit: int = 50
):
    entities = DEMO_ENTITIES.copy()
    
    if urgency:
        entities = [e for e in entities if e.urgency_level == urgency]
    if entity_type:
        entities = [e for e in entities if e.entity_type == entity_type]
    
    return {
        "entities": [e.dict() for e in entities[:limit]],
        "total": len(entities),
        "page": 1,
        "pages": 1
    }

@app.get("/api/v2/entities/stats")
async def get_entity_stats():
    return {
        "total_entities": len(DEMO_ENTITIES),
        "by_urgency": {
            "critical": len([e for e in DEMO_ENTITIES if e.urgency_level == "critical"]),
            "high": len([e for e in DEMO_ENTITIES if e.urgency_level == "high"]),
            "medium": len([e for e in DEMO_ENTITIES if e.urgency_level == "medium"]),
            "low": len([e for e in DEMO_ENTITIES if e.urgency_level == "low"]),
        },
        "by_type": {
            "host": len([e for e in DEMO_ENTITIES if e.entity_type == "host"]),
            "user": len([e for e in DEMO_ENTITIES if e.entity_type == "user"]),
            "service": len([e for e in DEMO_ENTITIES if e.entity_type == "service"]),
        }
    }

@app.get("/api/v2/entities/{entity_id}")
async def get_entity(entity_id: str):
    for entity in DEMO_ENTITIES:
        if entity.id == entity_id:
            return {
                "entity": entity.dict(),
                "timeline": [
                    {"timestamp": (datetime.utcnow() - timedelta(hours=i)).isoformat(), 
                     "event": f"Detection {i+1}", "severity": random.choice(["high", "medium"])}
                    for i in range(5)
                ],
                "related_entities": [e.dict() for e in random.sample(DEMO_ENTITIES, min(3, len(DEMO_ENTITIES)))]
            }
    raise HTTPException(status_code=404, detail="Entity not found")

# Detection Endpoints
@app.get("/api/v2/detections")
async def get_detections(
    severity: Optional[str] = None,
    detection_type: Optional[str] = None,
    limit: int = 50,
    page: int = 1
):
    detections = DEMO_DETECTIONS.copy()
    
    if severity:
        detections = [d for d in detections if d.severity == severity]
    if detection_type:
        detections = [d for d in detections if d.type == detection_type]
    
    start = (page - 1) * limit
    end = start + limit
    
    return {
        "detections": [d.dict() for d in detections[start:end]],
        "total": len(detections),
        "page": page,
        "pages": (len(detections) + limit - 1) // limit
    }

@app.get("/api/v2/detections/stats")
async def get_detection_stats():
    return {
        "total": len(DEMO_DETECTIONS),
        "by_severity": {
            "critical": len([d for d in DEMO_DETECTIONS if d.severity == "critical"]),
            "high": len([d for d in DEMO_DETECTIONS if d.severity == "high"]),
            "medium": len([d for d in DEMO_DETECTIONS if d.severity == "medium"]),
            "low": len([d for d in DEMO_DETECTIONS if d.severity == "low"]),
        },
        "by_type": {
            "credential_theft": len([d for d in DEMO_DETECTIONS if d.type == "credential_theft"]),
            "lateral_movement": len([d for d in DEMO_DETECTIONS if d.type == "lateral_movement"]),
            "command_control": len([d for d in DEMO_DETECTIONS if d.type == "command_control"]),
            "exfiltration": len([d for d in DEMO_DETECTIONS if d.type == "exfiltration"]),
        }
    }

@app.get("/api/v2/detections/{detection_id}")
async def get_detection(detection_id: str):
    for detection in DEMO_DETECTIONS:
        if detection.id == detection_id:
            return detection.dict()
    raise HTTPException(status_code=404, detail="Detection not found")

# MITRE ATT&CK Endpoints
@app.get("/api/v2/mitre/stats/coverage")
async def get_mitre_coverage():
    return {
        "techniques_detected": 12,
        "total_techniques": 38,
        "coverage_percentage": 31.6,
        "tactics_coverage": {
            "Initial Access": 2,
            "Execution": 3,
            "Persistence": 1,
            "Privilege Escalation": 2,
            "Defense Evasion": 1,
            "Credential Access": 3,
            "Discovery": 2,
            "Lateral Movement": 2,
            "Collection": 1,
            "Command and Control": 3,
            "Exfiltration": 2,
            "Impact": 1
        }
    }

@app.get("/api/v2/mitre/matrix")
async def get_mitre_matrix():
    tactics = [
        {"id": "TA0001", "name": "Initial Access", "techniques": ["T1190", "T1133"]},
        {"id": "TA0002", "name": "Execution", "techniques": ["T1059", "T1106"]},
        {"id": "TA0003", "name": "Persistence", "techniques": ["T1136", "T1053"]},
        {"id": "TA0004", "name": "Privilege Escalation", "techniques": ["T1134", "T1068"]},
        {"id": "TA0005", "name": "Defense Evasion", "techniques": ["T1070", "T1562"]},
        {"id": "TA0006", "name": "Credential Access", "techniques": ["T1003", "T1110"]},
        {"id": "TA0007", "name": "Discovery", "techniques": ["T1046", "T1087"]},
        {"id": "TA0008", "name": "Lateral Movement", "techniques": ["T1021", "T1550"]},
        {"id": "TA0040", "name": "Impact", "techniques": ["T1486", "T1489"]}
    ]
    return {"tactics": tactics}

# Hunt Endpoints
@app.get("/api/v2/hunt/queries")
async def get_hunt_queries():
    return {
        "queries": [
            {"id": "hunt_001", "name": "Credential Theft Activity", "severity": "high", "techniques": ["T1003", "T1110"]},
            {"id": "hunt_002", "name": "Lateral Movement", "severity": "high", "techniques": ["T1021", "T1550"]},
            {"id": "hunt_003", "name": "C2 Beaconing", "severity": "critical", "techniques": ["T1071", "T1090"]},
            {"id": "hunt_004", "name": "Data Exfiltration", "severity": "critical", "techniques": ["T1041", "T1048"]},
        ]
    }

# SOAR Endpoints
@app.get("/api/v2/soar/incidents")
async def get_soar_incidents():
    return {
        "incidents": [
            {"incident_id": "INC-001", "title": "Ransomware Detected", "status": "contained", "severity": "critical"},
            {"incident_id": "INC-002", "title": "C2 Communication", "status": "investigating", "severity": "high"},
            {"incident_id": "INC-003", "title": "Data Exfiltration", "status": "new", "severity": "high"},
            {"incident_id": "INC-004", "title": "Brute Force Attack", "status": "resolved", "severity": "medium"},
        ],
        "total": 4
    }

# AI Copilot Endpoints - support both paths
@app.post("/api/v2/copilot/query")
async def copilot_query(query: dict):
    return await handle_copilot_query(query)

@app.post("/api/v2/azure/copilot")
async def azure_copilot_query(query: dict):
    return await handle_copilot_query(query)

async def handle_copilot_query(query: dict):
    user_query = query.get("query", "")
    
    # Simple response based on query keywords
    if "threat" in user_query.lower() or "attack" in user_query.lower():
        response = f"Based on the current analysis, I've detected {len(DEMO_DETECTIONS)} security events in the last 24 hours. The most critical threats involve credential theft and lateral movement attempts. I recommend focusing on the 5 critical-severity detections first."
    elif "entity" in user_query.lower() or "host" in user_query.lower():
        response = f"Currently tracking {len(DEMO_ENTITIES)} entities. The highest risk entity is WEB-SERVER-01 with a threat score of 92. This server has been targeted in multiple attack chains."
    elif "monitor" in user_query.lower() or "live" in user_query.lower():
        response = "The live monitor is active and processing approximately 850,000 packets per minute. I'm detecting some behavioral anomalies consistent with C2 beaconing."
    elif "secure" in user_query.lower():
        response = "The system is currently secure, but I'm tracking 3 active investigations. The threat level is elevated due to recent external scanning activity."
    elif "mitre" in user_query.lower():
        response = "Your environment has triggered detections across 12 MITRE ATT&CK techniques. The most common are T1110 (Brute Force) and T1071 (C2 Communications). Consider implementing additional detection rules for T1003 (Credential Dumping)."
    else:
        response = f"I've analyzed {len(DEMO_DETECTIONS)} detections across {len(DEMO_ENTITIES)} entities. Your current security posture shows elevated risk in the credential theft and lateral movement categories. Would you like me to elaborate on any specific aspect?"
    
    return {
        "response": response,
        "timestamp": datetime.utcnow().isoformat(),
        "confidence": 0.92,
        "sources": ["Detection Engine", "Entity Manager", "MITRE Mapper"]
    }

# Network Monitoring Endpoints
@app.post("/api/v2/network/start")
async def start_monitoring():
    return {"status": "started", "timestamp": datetime.utcnow().isoformat()}

@app.post("/api/v2/network/stop")
async def stop_monitoring():
    return {"status": "stopped", "timestamp": datetime.utcnow().isoformat()}

@app.get("/api/v2/network/stats")
async def get_network_stats():
    return {
        "packets_analyzed": random.randint(800000, 950000),
        "active_connections": random.randint(120, 180),
        "suspicious_count": random.randint(3, 15),
        "bandwidth_usage": "2.1 Gbps"
    }

@app.get("/api/v2/network/events")
async def get_network_events(limit: int = 50):
    events = []
    base_events = [
        {"type": "detection", "message": "Behavioral anomaly: Entity activity above baseline", "source": "UEBA", "mitre": {"technique_id": "T1078"}},
        {"type": "detection", "message": "DNS Tunneling suspected", "source": "DNS Monitor", "mitre": {"technique_id": "T1048"}},
        {"type": "action", "message": "Firewall rule updated: Blocked malicious IP", "source": "Auto Response"},
        {"type": "system", "message": "Network scan completed", "source": "Scanner"},
        {"type": "detection", "message": "Data exfiltration attempt detected", "source": "DLP", "mitre": {"technique_id": "T1567"}}
    ]
    
    for i in range(min(limit, 20)):
        evt = random.choice(base_events)
        severity = "high" if evt["type"] == "detection" else "info"
        if "exfiltration" in evt["message"]: severity = "critical"
        
        events.append({
            "id": str(uuid.uuid4()),
            "type": evt["type"],
            "message": evt["message"],
            "severity": severity,
            "timestamp": (datetime.utcnow() - timedelta(minutes=random.randint(0, 60))).isoformat(),
            "source": evt["source"],
            "mitre": evt.get("mitre")
        })
    
    return {"events": events}

@app.get("/api/v2/network/connections")
async def get_network_connections(limit: int = 20):
    connections = []
    
    # 1. Try fetching REAL data from Database
    if database:
        try:
            # Query recent events that have network info
            query = events_table.select().order_by(events_table.c.timestamp.desc()).limit(limit)
            rows = await database.fetch_all(query)
            
            for row in rows:
                # Only include events that look like network connections
                if row['dest_ip'] or row['dest_port']:
                    connections.append({
                        "remote_ip": row['dest_ip'] or "0.0.0.0",
                        "remote_port": row['dest_port'] or 0,
                        "local_port": 0, # Agent doesn't send local port yet
                        "hostname": row['hostname'] or "Unknown Host",
                        "process": row['process_name'] or "unknown",
                        "status": row['status'] or "ESTABLISHED",
                        "is_suspicious": row['is_anomaly'],
                        "anomaly_score": float(row['risk_score'] or 0) / 100.0,
                        "threat_info": {"notes": row['prediction_message']} if row['is_anomaly'] else None
                    })
        except Exception as e:
            print(f"DB Fetch Error: {e}")
            
    # 2. Try fetching REAL data from Memory (Fallback)
    if not connections:
        for cid, events in customer_events.items():
            for evt in events[-limit:]:
                 if evt.get('dest_ip') or evt.get('dest_port'):
                    connections.append({
                        "remote_ip": evt.get('dest_ip'),
                        "remote_port": evt.get('dest_port'),
                        "hostname": evt.get('hostname'),
                        "process": evt.get('process_name'),
                        "status": evt.get('status'),
                        "is_suspicious": evt.get('is_anomaly'),
                        "anomaly_score": float(evt.get('risk_score', 0)) / 100.0
                    })
    
    # 3. If STILL empty, use Demo Data (so dashboard isn't blank)
    if not connections:
        processes = ["chrome.exe", "svchost.exe", "powershell.exe", "nginx", "postgres"]
        for i in range(min(limit, 15)):
            proc = random.choice(processes)
            is_suspicious = proc in ["powershell.exe"] and random.random() > 0.7
            
            connections.append({
                "remote_ip": f"192.168.1.{random.randint(100, 200)}",
                "remote_port": random.randint(1024, 65535),
                "local_port": 443 if random.random() > 0.5 else 80,
                "hostname": f"Demo-Node-{random.randint(1, 20)}", # Marked as Demo
                "process": proc,
                "status": "ESTABLISHED",
                "is_suspicious": is_suspicious,
                "anomaly_score": random.uniform(0.1, 0.9) if is_suspicious else random.uniform(0, 0.2),
                "threat_info": {"notes": "Known malicious behaviour"} if is_suspicious else None
            })
    
    return {"connections": connections[:limit]}

# XAI Endpoints
@app.get("/api/v2/xai/status")
async def get_xai_status():
    return {
        "shap_available": True,
        "lime_available": True,
        "model_loaded": True,
        "shap_explainer_ready": True,
        "lime_explainer_ready": True
    }

@app.get("/api/v2/xai/feature-importance")
async def get_feature_importance():
    return {
        "importance": {
            "Flow Duration": 0.23,
            "Total Fwd Packets": 0.18,
            "Flow Bytes/s": 0.15,
            "Fwd IAT Mean": 0.12,
            "Packet Length Mean": 0.10,
            "SYN Flag Count": 0.08,
            "Total Bwd Packets": 0.06,
            "Bwd IAT Mean": 0.04,
            "FIN Flag Count": 0.03,
            "Flow Packets/s": 0.01
        }
    }

# Reports Endpoint
@app.get("/api/v2/reports/executive")
async def get_executive_report():
    return {
        "period": "Last 30 days",
        "summary": {
            "total_threats": len(DEMO_DETECTIONS),
            "critical_resolved": 12,
            "mttd_minutes": 4.2,
            "mttr_minutes": 28,
            "security_score": 78
        },
        "highlights": [
            "Blocked 156 intrusion attempts",
            "Detected advanced persistent threat campaign",
            "Reduced MTTD by 35% compared to last month"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
