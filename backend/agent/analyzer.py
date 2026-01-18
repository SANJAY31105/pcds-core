"""
PCDS Local ML Analyzer
Runs ML inference locally when cloud is unavailable
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from .features import FlowFeatures, FEATURE_NAMES
except ImportError:
    from features import FlowFeatures, FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class ThreatDetection:
    """Represents a detected threat"""
    flow_id: str
    threat_type: str
    confidence: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    description: str
    mitre_tactic: Optional[str] = None
    mitre_technique: Optional[str] = None
    recommended_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "threat_type": self.threat_type,
            "confidence": self.confidence,
            "severity": self.severity,
            "description": self.description,
            "mitre_tactic": self.mitre_tactic,
            "mitre_technique": self.mitre_technique,
            "recommended_action": self.recommended_action
        }


class LocalMLAnalyzer:
    """
    Local ML inference engine
    Supports multiple model types:
    - Scikit-learn (joblib)
    - PyTorch
    - Simple rule-based (always available)
    """
    
    # Threat type mapping
    THREAT_TYPES = {
        0: "Normal",
        1: "DoS",
        2: "Probe",
        3: "R2L",  # Remote to Local
        4: "U2R",  # User to Root
        5: "Backdoor",
        6: "Botnet",
        7: "Brute_Force",
        8: "DDoS",
        9: "Infiltration",
        10: "Port_Scan",
        11: "Web_Attack"
    }
    
    # MITRE ATT&CK mapping
    MITRE_MAPPING = {
        "DoS": ("Impact", "T1499 - Endpoint Denial of Service"),
        "DDoS": ("Impact", "T1498 - Network Denial of Service"),
        "Probe": ("Discovery", "T1046 - Network Service Discovery"),
        "Port_Scan": ("Discovery", "T1046 - Network Service Discovery"),
        "Brute_Force": ("Credential Access", "T1110 - Brute Force"),
        "Backdoor": ("Persistence", "T1543 - Create or Modify System Process"),
        "Botnet": ("Command and Control", "T1071 - Application Layer Protocol"),
        "Web_Attack": ("Initial Access", "T1190 - Exploit Public-Facing Application"),
        "R2L": ("Initial Access", "T1078 - Valid Accounts"),
        "U2R": ("Privilege Escalation", "T1068 - Exploitation for Privilege Escalation"),
        "Infiltration": ("Lateral Movement", "T1021 - Remote Services"),
    }
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_type = "rules"  # Default to rule-based
        self.model_path = model_path
        
        if model_path:
            self._load_model(model_path)
        
        logger.info(f"LocalMLAnalyzer initialized with {self.model_type} model")
    
    def _load_model(self, model_path: str):
        """Load ML model from file"""
        path = Path(model_path)
        
        if not path.exists():
            logger.warning(f"Model not found: {model_path}")
            return
        
        suffix = path.suffix.lower()
        
        try:
            if suffix in ['.pkl', '.joblib'] and JOBLIB_AVAILABLE:
                self.model = joblib.load(model_path)
                self.model_type = "sklearn"
                logger.info(f"Loaded sklearn model from {model_path}")
                
            elif suffix == '.pt' and TORCH_AVAILABLE:
                self.model = torch.load(model_path, map_location='cpu')
                self.model.eval()
                self.model_type = "pytorch"
                logger.info(f"Loaded PyTorch model from {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_type = "rules"
    
    def _rule_based_analysis(self, features: FlowFeatures) -> Tuple[int, float]:
        """
        Simple rule-based threat detection
        Returns (threat_class, confidence)
        """
        threat_class = 0  # Normal
        confidence = 0.0
        
        # Rule 1: High packet rate (potential DDoS)
        if features.packets_per_second > 1000:
            threat_class = 8  # DDoS
            confidence = min(0.9, features.packets_per_second / 10000)
        
        # Rule 2: Port scan pattern (many connections, low bytes)
        elif features.src_packets > 10 and features.dst_bytes < 100:
            threat_class = 10  # Port_Scan
            confidence = 0.7
        
        # Rule 3: Brute force (many short connections)
        elif features.duration < 1.0 and features.flag_rst == 1:
            threat_class = 7  # Brute_Force
            confidence = 0.6
        
        # Rule 4: Data exfiltration (high outbound bytes)
        elif features.bytes_ratio > 10 and features.src_bytes > 100000:
            threat_class = 9  # Infiltration
            confidence = 0.65
        
        # Rule 5: Uncommon port usage
        elif features.uses_uncommon_port == 1:
            threat_class = 5  # Backdoor
            confidence = 0.5
        
        # Rule 6: RST flood (DoS)
        elif features.flag_rst == 1 and features.packets_per_second > 100:
            threat_class = 1  # DoS
            confidence = 0.75
        
        return threat_class, confidence
    
    def _sklearn_predict(self, features: FlowFeatures) -> Tuple[int, float]:
        """Predict using sklearn model"""
        try:
            feature_array = features.to_array().reshape(1, -1)
            prediction = self.model.predict(feature_array)[0]
            
            # Try to get probability if available
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(feature_array)[0]
                confidence = float(max(proba))
            else:
                confidence = 0.8
            
            return int(prediction), confidence
            
        except Exception as e:
            logger.error(f"sklearn prediction error: {e}")
            return self._rule_based_analysis(features)
    
    def _pytorch_predict(self, features: FlowFeatures) -> Tuple[int, float]:
        """Predict using PyTorch model"""
        try:
            feature_array = features.to_array()
            tensor = torch.FloatTensor(feature_array).unsqueeze(0)
            
            with torch.no_grad():
                output = self.model(tensor)
                proba = torch.softmax(output, dim=1)
                confidence, prediction = torch.max(proba, dim=1)
            
            return int(prediction.item()), float(confidence.item())
            
        except Exception as e:
            logger.error(f"PyTorch prediction error: {e}")
            return self._rule_based_analysis(features)
    
    def analyze(self, features: FlowFeatures) -> Optional[ThreatDetection]:
        """
        Analyze flow features and detect threats
        Returns ThreatDetection if threat found, None if normal
        """
        # Get prediction based on model type
        if self.model_type == "sklearn" and self.model:
            threat_class, confidence = self._sklearn_predict(features)
        elif self.model_type == "pytorch" and self.model:
            threat_class, confidence = self._pytorch_predict(features)
        else:
            threat_class, confidence = self._rule_based_analysis(features)
        
        # Normal traffic
        if threat_class == 0 or confidence < 0.4:
            return None
        
        # Get threat info
        threat_type = self.THREAT_TYPES.get(threat_class, "Unknown")
        mitre_info = self.MITRE_MAPPING.get(threat_type, (None, None))
        
        # Determine severity
        if confidence >= 0.9:
            severity = "CRITICAL"
        elif confidence >= 0.75:
            severity = "HIGH"
        elif confidence >= 0.5:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        # Generate description
        description = self._generate_description(features, threat_type, confidence)
        
        # Generate recommendation
        recommendation = self._generate_recommendation(threat_type)
        
        return ThreatDetection(
            flow_id=features.flow_id,
            threat_type=threat_type,
            confidence=confidence,
            severity=severity,
            description=description,
            mitre_tactic=mitre_info[0],
            mitre_technique=mitre_info[1],
            recommended_action=recommendation
        )
    
    def analyze_batch(self, features_list: List[FlowFeatures]) -> List[ThreatDetection]:
        """Analyze multiple flows"""
        detections = []
        for features in features_list:
            detection = self.analyze(features)
            if detection:
                detections.append(detection)
        return detections
    
    def _generate_description(self, features: FlowFeatures, threat_type: str, confidence: float) -> str:
        """Generate human-readable threat description"""
        descriptions = {
            "DoS": f"Denial of Service attack detected. High packet rate ({features.packets_per_second:.0f} pps) indicates attempt to overwhelm target.",
            "DDoS": f"Distributed Denial of Service pattern. Extremely high traffic volume detected ({features.bytes_per_second:.0f} bytes/sec).",
            "Probe": "Network reconnaissance detected. Scanning activity observed across multiple ports.",
            "Port_Scan": f"Port scanning activity from source. {features.src_packets} connection attempts with minimal response.",
            "Brute_Force": "Brute force login attempt. Multiple rapid connection attempts with connection resets.",
            "Backdoor": f"Suspicious backdoor communication. Traffic on uncommon port detected.",
            "Botnet": "Potential botnet command and control traffic. Unusual outbound communication pattern.",
            "Web_Attack": "Web application attack detected. Suspicious HTTP request patterns.",
            "R2L": "Remote to Local attack. Unauthorized access attempt from external source.",
            "U2R": "Privilege escalation attempt. User trying to gain root/admin access.",
            "Infiltration": f"Data exfiltration suspected. High outbound data transfer ({features.src_bytes} bytes).",
        }
        
        return descriptions.get(threat_type, f"Suspicious activity detected: {threat_type}")
    
    def _generate_recommendation(self, threat_type: str) -> str:
        """Generate recommended action"""
        recommendations = {
            "DoS": "Block source IP. Enable rate limiting. Monitor for continued attacks.",
            "DDoS": "Enable DDoS mitigation. Contact ISP. Consider CDN protection.",
            "Probe": "Block source IP. Review firewall rules. Monitor for follow-up attacks.",
            "Port_Scan": "Block source IP. Close unnecessary ports. Enable port scan detection.",
            "Brute_Force": "Block source IP. Enable account lockout. Enforce MFA.",
            "Backdoor": "Isolate affected system. Scan for malware. Audit network connections.",
            "Botnet": "Disconnect infected system. Run antivirus scan. Check for lateral movement.",
            "Web_Attack": "Enable WAF rules. Review application logs. Patch vulnerable software.",
            "R2L": "Block source. Review access controls. Enable network segmentation.",
            "U2R": "Isolate system. Review user permissions. Audit privilege changes.",
            "Infiltration": "Block outbound connection. Investigate data accessed. Enable DLP.",
        }
        
        return recommendations.get(threat_type, "Investigate further and consider blocking source.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics"""
        return {
            "model_type": self.model_type,
            "model_loaded": self.model is not None,
            "model_path": self.model_path
        }
