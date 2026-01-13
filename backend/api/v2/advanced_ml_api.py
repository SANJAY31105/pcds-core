"""
Paper 4 Features API
Phishing Detection, Reinforcement Learning, CNN Packet Analysis
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

router = APIRouter(prefix="/advanced-ml", tags=["Advanced ML"])


# ============ Phishing Detection ============

class URLCheckRequest(BaseModel):
    url: str


class EmailCheckRequest(BaseModel):
    content: str
    sender: str = ""
    subject: str = ""


@router.post("/phishing/check-url")
async def check_url_phishing(request: URLCheckRequest):
    """
    Check URL for phishing indicators
    
    Analyzes: suspicious TLDs, brand impersonation, URL patterns
    """
    from ml.phishing_detector import get_phishing_detector
    from dataclasses import asdict
    
    detector = get_phishing_detector()
    result = detector.analyze_url(request.url)
    
    return {
        "is_phishing": result.is_phishing,
        "confidence": result.confidence,
        "risk_score": result.risk_score,
        "indicators": result.indicators,
        "url_analysis": result.url_analysis,
        "recommendation": result.recommendation
    }


@router.post("/phishing/check-email")
async def check_email_phishing(request: EmailCheckRequest):
    """
    Check email content for phishing indicators
    
    Analyzes: urgency keywords, suspicious URLs, generic greetings
    """
    from ml.phishing_detector import get_phishing_detector
    
    detector = get_phishing_detector()
    result = detector.analyze_email(
        content=request.content,
        sender=request.sender,
        subject=request.subject
    )
    
    return {
        "is_phishing": result.is_phishing,
        "confidence": result.confidence,
        "risk_score": result.risk_score,
        "indicators": result.indicators,
        "content_analysis": result.content_analysis,
        "recommendation": result.recommendation
    }


@router.get("/phishing/stats")
async def get_phishing_stats():
    """Get phishing detector statistics"""
    from ml.phishing_detector import get_phishing_detector
    
    detector = get_phishing_detector()
    return detector.get_stats()


# ============ Reinforcement Learning ============

class RLStateRequest(BaseModel):
    threat_level: str  # none, low, medium, high, critical
    attack_type: int = 0
    affected_hosts: int = 1
    time_since_detection: int = 0


class RLFeedbackRequest(BaseModel):
    threat_mitigated: bool
    false_positive: bool = False
    response_time_seconds: int = 30


@router.post("/rl/recommend-action")
async def recommend_defense_action(request: RLStateRequest):
    """
    Get RL agent's recommended defense action
    
    Returns action with Q-values for the given threat state
    """
    from ml.rl_defense_agent import get_rl_agent, RLState, ThreatLevel, DefenseAction
    
    agent = get_rl_agent()
    
    # Map threat level
    threat_map = {
        "none": ThreatLevel.NONE,
        "low": ThreatLevel.LOW,
        "medium": ThreatLevel.MEDIUM,
        "high": ThreatLevel.HIGH,
        "critical": ThreatLevel.CRITICAL
    }
    
    threat = threat_map.get(request.threat_level.lower(), ThreatLevel.LOW)
    
    state = RLState(
        threat_level=threat,
        attack_type=request.attack_type,
        affected_hosts=request.affected_hosts,
        time_since_detection=request.time_since_detection,
        previous_action=DefenseAction.MONITOR,
        action_success_rate=0.8
    )
    
    recommendation = agent.get_policy_recommendation(state)
    
    return recommendation


@router.post("/rl/train")
async def train_rl_agent(episodes: int = 100):
    """
    Train RL agent through simulation
    """
    from ml.rl_defense_agent import get_rl_agent, RLState, ThreatLevel, DefenseAction
    import random
    
    agent = get_rl_agent()
    
    results = []
    for _ in range(episodes):
        # Random initial state
        initial_state = RLState(
            threat_level=ThreatLevel(random.randint(1, 4)),
            attack_type=random.randint(0, 15),
            affected_hosts=random.randint(1, 50),
            time_since_detection=0,
            previous_action=DefenseAction.MONITOR,
            action_success_rate=0.8
        )
        
        result = agent.train_episode(initial_state)
        results.append(result)
    
    return {
        "episodes_trained": episodes,
        "average_reward": sum(r["total_reward"] for r in results) / episodes,
        "agent_stats": agent.get_stats()
    }


@router.get("/rl/stats")
async def get_rl_stats():
    """Get RL agent statistics"""
    from ml.rl_defense_agent import get_rl_agent
    
    agent = get_rl_agent()
    return agent.get_stats()


# ============ CNN Packet Analysis ============

class PacketAnalyzeRequest(BaseModel):
    packet_hex: str  # Hex-encoded packet bytes


@router.post("/dpi/analyze")
async def analyze_packet(request: PacketAnalyzeRequest):
    """
    Deep packet inspection using CNN
    
    Analyzes raw packet bytes for malicious patterns
    """
    from ml.cnn_packet_classifier import get_packet_inspector
    
    inspector = get_packet_inspector()
    
    try:
        packet_bytes = bytes.fromhex(request.packet_hex)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid hex string")
    
    result = inspector.analyze_packet(packet_bytes)
    
    return {
        "is_malicious": result.is_malicious,
        "confidence": result.confidence,
        "attack_class": result.attack_class
    }


@router.post("/dpi/analyze-batch")
async def analyze_packets(packets: List[str]):
    """
    Analyze multiple packets
    """
    from ml.cnn_packet_classifier import get_packet_inspector
    
    inspector = get_packet_inspector()
    
    results = []
    for packet_hex in packets:
        try:
            packet_bytes = bytes.fromhex(packet_hex)
            result = inspector.analyze_packet(packet_bytes)
            results.append({
                "is_malicious": result.is_malicious,
                "confidence": result.confidence,
                "attack_class": result.attack_class
            })
        except ValueError:
            results.append({"error": "Invalid hex"})
    
    return {
        "results": results,
        "total": len(results),
        "malicious_count": sum(1 for r in results if r.get("is_malicious", False))
    }


@router.get("/dpi/stats")
async def get_dpi_stats():
    """Get deep packet inspector statistics"""
    from ml.cnn_packet_classifier import get_packet_inspector
    
    inspector = get_packet_inspector()
    return inspector.get_stats()


# ============ Combined Status ============

@router.get("/status")
async def get_advanced_ml_status():
    """Get status of all advanced ML components"""
    from ml.phishing_detector import get_phishing_detector
    from ml.rl_defense_agent import get_rl_agent
    from ml.cnn_packet_classifier import get_packet_inspector
    
    return {
        "phishing_detector": get_phishing_detector().get_stats(),
        "rl_agent": get_rl_agent().get_stats(),
        "packet_inspector": get_packet_inspector().get_stats()
    }
