"""
Multi-Model Ensemble Detector
Combines 5 specialized detection models with weighted voting

Like CrowdStrike & Darktrace - multiple models vote on each connection

Models:
1. SignatureDetector - Known malware ports/IPs (weight: 1.0)
2. MLAnomalyDetector - Trained neural network (weight: 0.8)
3. BehavioralDetector - C2, DGA, Exfiltration (weight: 0.9)
4. TimeSeriesDetector - Connection frequency patterns (weight: 0.7)
5. DNSEntropyDetector - DGA domain detection (weight: 0.8)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math
from collections import deque
import time

@dataclass
class DetectionResult:
    """Result from a single detector"""
    name: str
    is_threat: bool
    confidence: float
    threat_type: Optional[str] = None
    details: Optional[Dict] = None


class SignatureDetector:
    """Known bad signatures - ports, IPs, patterns"""
    
    WEIGHT = 1.0  # Highest weight - definite threats
    
    MALICIOUS_PORTS = {
        4444: ("Metasploit", 1.0),
        5555: ("Android Debug", 0.9),
        6666: ("IRC Botnet", 1.0),
        6667: ("IRC", 0.8),
        31337: ("Back Orifice", 1.0),
        12345: ("NetBus", 1.0),
        27374: ("SubSeven", 1.0),
        20034: ("NetBus 2", 1.0),
        1337: ("Leet Port", 0.7),
        8888: ("Alt HTTP", 0.5),
    }
    
    def detect(self, connection: Dict) -> DetectionResult:
        remote_port = connection.get('remote_port', 0)
        local_port = connection.get('local_port', 0)
        
        # Check remote port
        if remote_port in self.MALICIOUS_PORTS:
            name, confidence = self.MALICIOUS_PORTS[remote_port]
            return DetectionResult(
                name="SignatureDetector",
                is_threat=True,
                confidence=confidence,
                threat_type="malicious_port",
                details={"port": remote_port, "threat": name}
            )
        
        # Check local port
        if local_port in self.MALICIOUS_PORTS:
            name, confidence = self.MALICIOUS_PORTS[local_port]
            return DetectionResult(
                name="SignatureDetector",
                is_threat=True,
                confidence=confidence,
                threat_type="malicious_local_port",
                details={"port": local_port, "threat": name}
            )
        
        return DetectionResult(name="SignatureDetector", is_threat=False, confidence=0.0)


class TimeSeriesDetector:
    """Detects anomalous connection patterns over time"""
    
    WEIGHT = 0.7
    
    def __init__(self):
        self.connection_counts = {}  # IP -> deque of timestamps
        self.baseline_window = 60  # seconds
        self.threshold_multiplier = 3  # 3x baseline = anomaly
    
    def detect(self, connection: Dict) -> DetectionResult:
        remote_ip = connection.get('remote_ip', '')
        if not remote_ip:
            return DetectionResult(name="TimeSeriesDetector", is_threat=False, confidence=0.0)
        
        now = time.time()
        
        # Initialize or get history
        if remote_ip not in self.connection_counts:
            self.connection_counts[remote_ip] = deque(maxlen=1000)
        
        history = self.connection_counts[remote_ip]
        history.append(now)
        
        # Count connections in window
        recent = sum(1 for t in history if now - t <= self.baseline_window)
        
        # Calculate baseline (older connections)
        old = sum(1 for t in history if self.baseline_window < now - t <= self.baseline_window * 2)
        baseline = max(old, 1)
        
        # Detect anomaly
        if recent > baseline * self.threshold_multiplier and recent > 10:
            confidence = min(1.0, (recent / baseline) / 10)
            return DetectionResult(
                name="TimeSeriesDetector",
                is_threat=True,
                confidence=confidence,
                threat_type="connection_spike",
                details={"recent": recent, "baseline": baseline, "ip": remote_ip}
            )
        
        return DetectionResult(name="TimeSeriesDetector", is_threat=False, confidence=0.0)


class DNSEntropyDetector:
    """Detects DGA (Domain Generation Algorithm) domains"""
    
    WEIGHT = 0.8
    ENTROPY_THRESHOLD = 3.5
    SUSPICIOUS_TLDS = ['.top', '.xyz', '.club', '.info', '.online', '.site', '.pw', '.tk', '.ml', '.ga', '.cf']
    
    def detect(self, connection: Dict) -> DetectionResult:
        hostname = connection.get('hostname', '')
        if not hostname or '.' not in hostname:
            return DetectionResult(name="DNSEntropyDetector", is_threat=False, confidence=0.0)
        
        # Extract domain
        parts = hostname.lower().split('.')
        if len(parts) < 2:
            return DetectionResult(name="DNSEntropyDetector", is_threat=False, confidence=0.0)
        
        domain = parts[-2]
        if len(domain) < 4:
            return DetectionResult(name="DNSEntropyDetector", is_threat=False, confidence=0.0)
        
        # Calculate entropy
        entropy = self._shannon_entropy(domain)
        
        # Adjust threshold for suspicious TLDs
        has_suspicious_tld = any(hostname.endswith(tld) for tld in self.SUSPICIOUS_TLDS)
        threshold = self.ENTROPY_THRESHOLD - 0.5 if has_suspicious_tld else self.ENTROPY_THRESHOLD
        
        if entropy > threshold:
            confidence = min(1.0, (entropy - threshold) / 1.5)
            return DetectionResult(
                name="DNSEntropyDetector",
                is_threat=True,
                confidence=confidence,
                threat_type="dga_domain",
                details={"domain": hostname, "entropy": round(entropy, 2)}
            )
        
        return DetectionResult(name="DNSEntropyDetector", is_threat=False, confidence=0.0)
    
    def _shannon_entropy(self, string: str) -> float:
        if not string:
            return 0.0
        freq = {}
        for char in string:
            freq[char] = freq.get(char, 0) + 1
        length = len(string)
        return -sum((count/length) * math.log2(count/length) for count in freq.values())


class MLAnomalyDetector:
    """Wrapper for trained neural network model"""
    
    WEIGHT = 0.8
    
    def __init__(self):
        self.ml_detector = None
        self._load_model()
    
    def _load_model(self):
        try:
            from ml.ml_detector import get_ml_detector
            self.ml_detector = get_ml_detector()
        except:
            pass
    
    def detect(self, connection: Dict) -> DetectionResult:
        if not self.ml_detector or not self.ml_detector.loaded:
            return DetectionResult(name="MLAnomalyDetector", is_threat=False, confidence=0.0)
        
        try:
            features = self.ml_detector.extract_features(connection)
            is_attack, confidence = self.ml_detector.predict(features)
            
            if is_attack:
                return DetectionResult(
                    name="MLAnomalyDetector",
                    is_threat=True,
                    confidence=confidence,
                    threat_type="ml_detected",
                    details={"model_confidence": round(confidence, 3)}
                )
        except:
            pass
        
        return DetectionResult(name="MLAnomalyDetector", is_threat=False, confidence=0.0)


class BehavioralDetectorWrapper:
    """Wrapper for behavioral detector (C2, DGA, Exfil)"""
    
    WEIGHT = 0.9
    
    def __init__(self):
        self.behavioral_detector = None
        self._load_detector()
    
    def _load_detector(self):
        try:
            from ml.behavioral_detector import get_behavioral_detector
            self.behavioral_detector = get_behavioral_detector()
        except:
            pass
    
    def detect(self, connection: Dict) -> DetectionResult:
        if not self.behavioral_detector:
            return DetectionResult(name="BehavioralDetector", is_threat=False, confidence=0.0)
        
        try:
            threat = self.behavioral_detector.analyze_connection(connection)
            if threat:
                return DetectionResult(
                    name="BehavioralDetector",
                    is_threat=True,
                    confidence=0.85,  # High confidence for behavioral
                    threat_type=threat.get("type", "behavioral"),
                    details=threat
                )
        except:
            pass
        
        return DetectionResult(name="BehavioralDetector", is_threat=False, confidence=0.0)


class EnsembleDetector:
    """
    Multi-Model Ensemble Detector
    Combines all 5 detectors with weighted voting
    
    Enterprise-grade detection like CrowdStrike & Darktrace
    """
    
    def __init__(self):
        self.detectors = [
            SignatureDetector(),
            MLAnomalyDetector(),
            BehavioralDetectorWrapper(),
            TimeSeriesDetector(),
            DNSEntropyDetector(),
        ]
        
        self.weights = {
            "SignatureDetector": 1.0,      # Known bad = definite
            "MLAnomalyDetector": 0.8,       # Trained model
            "BehavioralDetector": 0.9,      # Behavioral patterns
            "TimeSeriesDetector": 0.7,      # Frequency anomalies
            "DNSEntropyDetector": 0.8,      # DGA detection
        }
        
        print("🎯 Ensemble Detector initialized with 5 models")
    
    def detect(self, connection: Dict) -> Tuple[bool, float, List[DetectionResult]]:
        """
        Run all detectors and combine results
        
        Returns:
            (is_threat, combined_confidence, individual_results)
        """
        results = []
        
        for detector in self.detectors:
            result = detector.detect(connection)
            results.append(result)
        
        # Weighted voting
        weighted_sum = 0
        total_weight = 0
        threat_count = 0
        
        for result in results:
            weight = self.weights.get(result.name, 0.5)
            if result.is_threat:
                weighted_sum += result.confidence * weight
                threat_count += 1
            total_weight += weight
        
        # Calculate combined confidence
        combined_confidence = weighted_sum / total_weight if total_weight > 0 else 0
        
        # Determine if threat (any high-confidence hit OR multiple low-confidence)
        is_threat = (
            any(r.confidence > 0.8 and r.is_threat for r in results) or  # Single high-confidence
            (threat_count >= 2 and combined_confidence > 0.5) or          # Multiple detectors agree
            combined_confidence > 0.7                                      # Overall high confidence
        )
        
        return is_threat, combined_confidence, results
    
    def get_threat_summary(self, results: List[DetectionResult]) -> Dict:
        """Generate summary of detection results"""
        threats = [r for r in results if r.is_threat]
        
        if not threats:
            return None
        
        # Get highest confidence threat
        primary = max(threats, key=lambda r: r.confidence)
        
        return {
            "type": primary.threat_type,
            "name": primary.details.get("threat", "Multi-Model Detection") if primary.details else "Threat Detected",
            "technique": "T1071",  # Default MITRE technique
            "severity": "critical" if primary.confidence > 0.9 else "high" if primary.confidence > 0.7 else "medium",
            "detectors_triggered": [r.name for r in threats],
            "confidence": round(primary.confidence, 3),
            "details": {r.name: r.details for r in threats if r.details}
        }


# Singleton instance
_ensemble_detector = None

def get_ensemble_detector() -> EnsembleDetector:
    """Get or create ensemble detector instance"""
    global _ensemble_detector
    if _ensemble_detector is None:
        _ensemble_detector = EnsembleDetector()
    return _ensemble_detector


if __name__ == "__main__":
    # Test the ensemble
    detector = EnsembleDetector()
    
    # Test with suspicious connection
    test_connection = {
        "remote_ip": "192.168.1.100",
        "remote_port": 4444,  # Metasploit!
        "local_port": 12345,
        "hostname": "xkcd42abc.top",  # DGA-like
        "process_name": "unknown.exe"
    }
    
    print("\n🧪 Testing Ensemble Detector...")
    print(f"Connection: {test_connection}")
    
    is_threat, confidence, results = detector.detect(test_connection)
    
    print(f"\n📊 Results:")
    print(f"   Is Threat: {is_threat}")
    print(f"   Combined Confidence: {confidence:.2f}")
    print(f"\n   Individual Detectors:")
    for r in results:
        status = "⚠️ THREAT" if r.is_threat else "✅ Safe"
        print(f"   - {r.name}: {status} (conf: {r.confidence:.2f})")
        if r.details:
            print(f"     Details: {r.details}")
