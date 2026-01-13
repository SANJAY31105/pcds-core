"""
Behavioral Threat Detection Module
Detects threats based on behavior patterns, not signatures:
- C2 Beaconing (regular callback intervals)
- DNS Entropy Analysis (DGA domain detection)
- Data Exfiltration (large outbound transfers)
- Lateral Movement (internal scanning)
"""

import time
import math
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class BehavioralDetector:
    """
    Behavioral-based threat detection
    Catches threats that signature-based detection misses
    """
    
    def __init__(self):
        # Connection history for pattern analysis
        self.connection_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.dns_queries: Dict[str, List[str]] = defaultdict(list)
        self.data_transfer: Dict[str, Dict] = defaultdict(lambda: {"sent": 0, "recv": 0, "start_time": None})
        
        # Detection thresholds
        self.BEACONING_INTERVAL_TOLERANCE = 0.15  # 15% tolerance
        self.BEACONING_MIN_SAMPLES = 5
        self.DNS_ENTROPY_THRESHOLD = 3.5  # High entropy = suspicious
        self.EXFIL_THRESHOLD_MB = 100  # 100MB outbound in short time
        self.EXFIL_TIME_WINDOW = 3600  # 1 hour
    
    def analyze_connection(self, connection: Dict) -> Optional[Dict]:
        """
        Analyze connection for behavioral threats
        
        Returns threat info if detected, None otherwise
        """
        remote_ip = connection.get('remote_ip', '')
        remote_port = connection.get('remote_port', 0)
        hostname = connection.get('hostname', '')
        
        # Check for C2 beaconing
        beaconing = self._check_beaconing(remote_ip, remote_port)
        if beaconing:
            return beaconing
        
        # Check for DGA domains (DNS entropy)
        dga = self._check_dns_entropy(hostname)
        if dga:
            return dga
        
        # Check for data exfiltration
        exfil = self._check_exfiltration(remote_ip, connection.get('bytes_sent', 0))
        if exfil:
            return exfil
        
        return None
    
    def _check_beaconing(self, remote_ip: str, remote_port: int) -> Optional[Dict]:
        """
        Detect C2 beaconing patterns
        
        C2 malware often calls back at regular intervals (e.g., every 60 seconds)
        """
        key = f"{remote_ip}:{remote_port}"
        now = time.time()
        
        # Record this connection time
        self.connection_history[key].append(now)
        
        history = list(self.connection_history[key])
        if len(history) < self.BEACONING_MIN_SAMPLES:
            return None
        
        # Calculate intervals between connections
        intervals = []
        for i in range(1, len(history)):
            intervals.append(history[i] - history[i-1])
        
        if not intervals:
            return None
        
        # Check if intervals are regular (beaconing pattern)
        avg_interval = sum(intervals) / len(intervals)
        if avg_interval < 5:  # Less than 5 seconds = not beaconing, just busy connection
            return None
        
        # Calculate variance
        variance = sum((x - avg_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = math.sqrt(variance)
        
        # Regular intervals with low variance = beaconing
        coefficient_of_variation = std_dev / avg_interval if avg_interval > 0 else 1
        
        if coefficient_of_variation < self.BEACONING_INTERVAL_TOLERANCE:
            return {
                "type": "c2_beaconing",
                "name": f"C2 Beaconing Detected (every {avg_interval:.0f}s)",
                "technique": "T1071",  # Application Layer Protocol
                "severity": "critical",
                "details": {
                    "avg_interval": round(avg_interval, 1),
                    "samples": len(history),
                    "variance": round(coefficient_of_variation, 3)
                }
            }
        
        return None
    
    def _check_dns_entropy(self, hostname: str) -> Optional[Dict]:
        """
        Detect DGA (Domain Generation Algorithm) domains
        
        DGA domains have high entropy (random-looking names)
        Examples: xkcd42abc.top, a7f9b2c.xyz
        """
        if not hostname or '.' not in hostname:
            return None
        
        # Extract domain name (before TLD)
        parts = hostname.lower().split('.')
        if len(parts) < 2:
            return None
        
        domain = parts[-2]  # Second-level domain
        
        if len(domain) < 4:
            return None
        
        # Calculate Shannon entropy
        entropy = self._shannon_entropy(domain)
        
        # High entropy + certain TLDs = suspicious
        suspicious_tlds = ['.top', '.xyz', '.club', '.info', '.online', '.site', '.pw', '.tk', '.ml', '.ga', '.cf']
        has_suspicious_tld = any(hostname.endswith(tld) for tld in suspicious_tlds)
        
        threshold = self.DNS_ENTROPY_THRESHOLD - 0.5 if has_suspicious_tld else self.DNS_ENTROPY_THRESHOLD
        
        if entropy > threshold:
            return {
                "type": "dga_domain",
                "name": f"Suspicious DGA Domain ({hostname})",
                "technique": "T1568.002",  # Dynamic Resolution: DGA
                "severity": "high",
                "details": {
                    "domain": hostname,
                    "entropy": round(entropy, 2),
                    "threshold": threshold
                }
            }
        
        return None
    
    def _shannon_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not string:
            return 0.0
        
        # Calculate frequency of each character
        freq = {}
        for char in string:
            freq[char] = freq.get(char, 0) + 1
        
        # Calculate entropy
        length = len(string)
        entropy = 0.0
        for count in freq.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    def _check_exfiltration(self, remote_ip: str, bytes_sent: int) -> Optional[Dict]:
        """
        Detect potential data exfiltration
        
        Large amounts of data sent to external IPs
        """
        # Skip internal IPs
        if remote_ip.startswith(('192.168.', '10.', '172.', '127.')):
            return None
        
        now = time.time()
        transfer = self.data_transfer[remote_ip]
        
        if transfer["start_time"] is None:
            transfer["start_time"] = now
        
        transfer["sent"] += bytes_sent
        
        # Check if within time window
        elapsed = now - transfer["start_time"]
        if elapsed > self.EXFIL_TIME_WINDOW:
            # Reset window
            transfer["sent"] = bytes_sent
            transfer["start_time"] = now
            return None
        
        # Check if threshold exceeded
        sent_mb = transfer["sent"] / (1024 * 1024)
        if sent_mb > self.EXFIL_THRESHOLD_MB:
            return {
                "type": "data_exfiltration",
                "name": f"Potential Data Exfiltration ({sent_mb:.1f} MB)",
                "technique": "T1041",  # Exfiltration Over C2 Channel
                "severity": "critical",
                "details": {
                    "bytes_sent": transfer["sent"],
                    "mb_sent": round(sent_mb, 1),
                    "destination": remote_ip,
                    "duration_seconds": round(elapsed, 0)
                }
            }
        
        return None
    
    def get_stats(self) -> Dict:
        """Get detection statistics"""
        return {
            "tracked_connections": len(self.connection_history),
            "tracked_transfers": len(self.data_transfer),
            "dns_queries_tracked": sum(len(v) for v in self.dns_queries.values())
        }


# Singleton instance
_behavioral_detector = None

def get_behavioral_detector() -> BehavioralDetector:
    """Get or create behavioral detector instance"""
    global _behavioral_detector
    if _behavioral_detector is None:
        _behavioral_detector = BehavioralDetector()
    return _behavioral_detector


if __name__ == "__main__":
    # Test the detector
    detector = BehavioralDetector()
    
    # Test DGA detection
    suspicious_domains = [
        "xkcd42abc.top",
        "a7f9b2c3d4.xyz",
        "randomstring123.club",
        "google.com",
        "microsoft.com"
    ]
    
    print("DGA Detection Test:")
    for domain in suspicious_domains:
        result = detector._check_dns_entropy(domain)
        status = "⚠️ SUSPICIOUS" if result else "✅ Safe"
        entropy = detector._shannon_entropy(domain.split('.')[0])
        print(f"  {domain}: {status} (entropy: {entropy:.2f})")
