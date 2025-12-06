"""
Threat Detection Engine - Core analysis and classification
"""
import uuid
from datetime import datetime
from typing import List, Dict
import numpy as np
from models import (
    ThreatDetection, ThreatSeverity, ThreatCategory,
    NetworkEvent, CountermeasureAction, MLPrediction
)
from ml.anomaly_detector import anomaly_detector
from ml.mitre_attack import mitre_mapper


class ThreatEngine:
    """Core threat detection and analysis engine"""
    
    def __init__(self):
        self.threat_signatures = self._load_threat_signatures()
        self.risk_weights = {
            'ml_score': 0.4,
            'signature_match': 0.3,
            'behavioral': 0.2,
            'reputation': 0.1
        }
        
    def _load_threat_signatures(self) -> Dict:
        """Load known threat signatures"""
        return {
            'sql_injection': {
                'patterns': ['UNION SELECT', 'DROP TABLE', '--', 'OR 1=1'],
                'severity': ThreatSeverity.CRITICAL,
                'category': ThreatCategory.INTRUSION
            },
            'port_scan': {
                'indicators': ['sequential_ports', 'high_frequency'],
                'severity': ThreatSeverity.MEDIUM,
                'category': ThreatCategory.SUSPICIOUS_ACTIVITY
            },
            'ddos_pattern': {
                'indicators': ['high_volume', 'same_source', 'syn_flood'],
                'severity': ThreatSeverity.HIGH,
                'category': ThreatCategory.DDoS
            }
        }
        
    async def analyze_network_event(self, event: NetworkEvent) -> ThreatDetection:
        """Analyze network event for threats"""
        
        # Extract features for ML model
        features = anomaly_detector.extract_features({
            'packet_size': event.packet_size,
            'port': event.port,
            'protocol': event.protocol,
            'source_ip': event.source_ip,
            'destination_ip': event.destination_ip,
            'flags': event.flags
        })
        
        # ML prediction
        is_anomaly, anomaly_score, ml_confidence = anomaly_detector.predict(features)
        
        # Signature matching
        signature_match, matched_threat = self._check_signatures(event)
        
        # Behavioral analysis
        behavioral_score = self._analyze_behavior(event)
        
        # Calculate overall risk score
        risk_score = self._calculate_risk_score(
            ml_score=anomaly_score,
            signature_score=1.0 if signature_match else 0.0,
            behavioral_score=behavioral_score
        )
        
        # Determine severity and category
        if signature_match:
            severity = matched_threat['severity']
            category = matched_threat['category']
        elif risk_score > 0.8:
            severity = ThreatSeverity.CRITICAL
            category = ThreatCategory.ANOMALY
        elif risk_score > 0.6:
            severity = ThreatSeverity.HIGH
            category = ThreatCategory.SUSPICIOUS_ACTIVITY
        elif risk_score > 0.4:
            severity = ThreatSeverity.MEDIUM
            category = ThreatCategory.SUSPICIOUS_ACTIVITY
        else:
            severity = ThreatSeverity.LOW
            category = ThreatCategory.ANOMALY
            
        # Map to MITRE ATT&CK
        mitre_techniques = mitre_mapper.map_threat(
            threat_category=category.value,
            indicators=self._extract_indicators(event, is_anomaly, signature_match),
            source_ip=event.source_ip,
            destination_ip=event.destination_ip or "",
            port=event.port
        )
        
        # Get kill chain stage
        kill_chain_stage = None
        if mitre_techniques:
            # Use first technique's tactic for kill chain
            first_tactic = mitre_techniques[0].get('tactic')
            if first_tactic:
                kill_chain_stage = mitre_mapper.get_kill_chain_stage(first_tactic)
            
        # Build threat detection object
        threat = ThreatDetection(
            id=str(uuid.uuid4()),
            timestamp=event.timestamp,
            severity=severity,
            category=category,
            title=self._generate_threat_title(category, signature_match),
            description=self._generate_threat_description(
                event, is_anomaly, signature_match, matched_threat
            ),
            source_ip=event.source_ip,
            destination_ip=event.destination_ip,
            risk_score=float(risk_score * 100),
            confidence=float(ml_confidence),
            indicators=self._extract_indicators(event, is_anomaly, signature_match),
            affected_systems=[event.destination_ip] if event.destination_ip else [],
            mitre_attack_techniques=mitre_techniques,
            kill_chain_stage=kill_chain_stage
        )
        
        return threat
        
    def _check_signatures(self, event: NetworkEvent) -> tuple:
        """Check against known threat signatures"""
        # Simplified signature matching
        if event.port in [23, 3389]:  # Telnet, RDP
            return True, self.threat_signatures['port_scan']
        if event.packet_size > 1400:
            return True, self.threat_signatures['ddos_pattern']
        return False, None
        
    def _analyze_behavior(self, event: NetworkEvent) -> float:
        """Analyze behavioral patterns"""
        score = 0.0
        
        # High packet size
        if event.packet_size > 1200:
            score += 0.3
            
        # Suspicious ports
        if event.port in [23, 135, 445, 3389, 5900]:
            score += 0.4
            
        # TCP flags analysis
        if event.flags and 'SYN' in event.flags and 'ACK' not in event.flags:
            score += 0.2
            
        return min(score, 1.0)
        
    def _calculate_risk_score(self, ml_score: float, signature_score: float, 
                             behavioral_score: float) -> float:
        """Calculate weighted risk score"""
        return (
            ml_score * self.risk_weights['ml_score'] +
            signature_score * self.risk_weights['signature_match'] +
            behavioral_score * self.risk_weights['behavioral']
        )
        
    def _generate_threat_title(self, category: ThreatCategory, 
                              signature_match: bool) -> str:
        """Generate human-readable threat title"""
        if signature_match:
            return f"Detected {category.value.replace('_', ' ').title()} Attack"
        return f"Suspicious {category.value.replace('_', ' ').title()} Detected"
        
    def _generate_threat_description(self, event: NetworkEvent, is_anomaly: bool,
                                    signature_match: bool, matched_threat: dict) -> str:
        """Generate detailed threat description"""
        if signature_match:
            return f"Known attack pattern detected from {event.source_ip} targeting port {event.port}"
        else:
            return f"Anomalous behavior detected from {event.source_ip} with risk indicators"
            
    def _extract_indicators(self, event: NetworkEvent, is_anomaly: bool,
                          signature_match: bool) -> List[str]:
        """Extract threat indicators"""
        indicators = []
        
        if is_anomaly:
            indicators.append("ML anomaly detection triggered")
        if signature_match:
            indicators.append("Known threat signature matched")
        if event.packet_size > 1200:
            indicators.append("Unusually large packet size")
        if event.port in [23, 135, 445, 3389]:
            indicators.append("Connection to sensitive port")
            
        return indicators
        
    def generate_countermeasures(self, threat: ThreatDetection) -> List[CountermeasureAction]:
        """Generate recommended countermeasures"""
        measures = []
        
        if threat.severity in [ThreatSeverity.CRITICAL, ThreatSeverity.HIGH]:
            measures.append(CountermeasureAction(
                id=str(uuid.uuid4()),
                threat_id=threat.id,
                action_type="block_ip",
                description=f"Block traffic from {threat.source_ip}",
                priority=1,
                automated=True,
                estimated_impact="High - prevents further attacks from this source"
            ))
            
        measures.append(CountermeasureAction(
            id=str(uuid.uuid4()),
            threat_id=threat.id,
            action_type="alert_admin",
            description="Notify security team for investigation",
            priority=2,
            automated=True,
            estimated_impact="Medium - enables human review"
        ))
        
        if threat.category == ThreatCategory.DDoS:
            measures.append(CountermeasureAction(
                id=str(uuid.uuid4()),
                threat_id=threat.id,
                action_type="rate_limit",
                description="Apply rate limiting on affected endpoints",
                priority=1,
                automated=True,
                estimated_impact="High - reduces service impact"
            ))
            
        return measures


# Global threat engine instance
threat_engine = ThreatEngine()
