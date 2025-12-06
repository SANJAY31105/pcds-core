"""
Data models and schemas for PCDS
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from enum import Enum


class ThreatSeverity(str, Enum):
    """Threat severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class ThreatCategory(str, Enum):
    """Threat categories"""
    MALWARE = "malware"
    PHISHING = "phishing"
    DDoS = "ddos"
    INTRUSION = "intrusion"
    DATA_BREACH = "data_breach"
    ANOMALY = "anomaly"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"


class NetworkEvent(BaseModel):
    """Network event data"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source_ip: str
    destination_ip: str
    port: int
    protocol: str
    packet_size: int
    flags: Optional[str] = None
    

class ThreatDetection(BaseModel):
    """Threat detection result"""
    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    severity: ThreatSeverity
    category: ThreatCategory
    title: str
    description: str
    source_ip: str
    destination_ip: Optional[str] = None
    risk_score: float = Field(ge=0.0, le=100.0)
    confidence: float = Field(ge=0.0, le=1.0)
    indicators: List[str] = Field(default_factory=list)
    affected_systems: List[str] = Field(default_factory=list)
    mitre_attack_techniques: List[dict] = Field(default_factory=list)  # MITRE ATT&CK mapping
    kill_chain_stage: Optional[dict] = None  # Kill chain stage info
    

class CountermeasureAction(BaseModel):
    """Recommended countermeasure"""
    id: str
    threat_id: str
    action_type: str
    description: str
    priority: int
    automated: bool
    estimated_impact: str
    

class SystemMetrics(BaseModel):
    """System health metrics"""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    cpu_usage: float
    memory_usage: float
    network_throughput: float
    active_connections: int
    threats_detected_today: int
    threats_blocked_today: int
    

class AlertNotification(BaseModel):
    """Real-time alert notification"""
    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    severity: ThreatSeverity
    message: str
    threat_id: Optional[str] = None
    acknowledged: bool = False
    

class MLPrediction(BaseModel):
    """ML model prediction result"""
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    features: dict
    model_version: str
    

class DashboardStats(BaseModel):
    """Dashboard overview statistics"""
    total_threats: int
    critical_threats: int
    threats_today: int
    threats_blocked: int
    average_risk_score: float
    system_health: str
    uptime_percentage: float
