"""
ML Training Pipeline - Feature Extraction
Converts raw network data to ML-ready feature vectors.

Feature Categories:
1. Flow Features - Network flow characteristics
2. Behavioral Features - User/entity behavior patterns
3. Attack Context Features - Threat-specific indicators
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime
import math


class FeatureExtractor:
    """
    Comprehensive feature extraction for network security ML
    
    Produces feature vectors for:
    - Flow-based detection
    - Behavioral anomaly detection
    - Attack classification
    """
    
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.feature_names = []
        self._build_feature_names()
    
    def _build_feature_names(self):
        """Build list of feature names for interpretability"""
        self.feature_names = [
            # Flow features (20)
            'duration', 'total_packets', 'total_bytes', 'packets_per_sec',
            'bytes_per_sec', 'avg_packet_size', 'packet_size_std',
            'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',
            'fwd_bwd_ratio', 'iat_mean', 'iat_std', 'iat_max', 'iat_min',
            'tcp_flags', 'syn_count', 'fin_count', 'rst_count',
            
            # Protocol features (8)
            'is_tcp', 'is_udp', 'is_icmp', 'is_http', 'is_https',
            'is_dns', 'is_ssh', 'is_rdp',
            
            # Port features (8)
            'src_port_norm', 'dst_port_norm', 'is_well_known_port',
            'is_registered_port', 'is_dynamic_port', 'port_entropy',
            'is_suspicious_port', 'port_class',
            
            # IP features (8)
            'ip_entropy', 'is_internal_src', 'is_internal_dst',
            'is_broadcast', 'is_multicast', 'geo_distance',
            'ip_reputation', 'is_known_bad_ip',
            
            # Behavioral features (12)
            'hour_of_day', 'day_of_week', 'is_business_hours',
            'connection_count', 'unique_dst_ips', 'unique_dst_ports',
            'failed_connections', 'success_rate', 'data_transfer_ratio',
            'session_duration', 'idle_time', 'activity_score',
            
            # Attack context features (8)
            'lateral_movement_score', 'privilege_escalation_score',
            'exfiltration_score', 'c2_beaconing_score',
            'brute_force_score', 'scan_score', 'malware_score',
            'anomaly_score'
        ]
    
    def extract_flow_features(self, flow_data: Dict) -> np.ndarray:
        """
        Extract flow-based features from network flow data
        
        Args:
            flow_data: Dictionary with flow information
        
        Returns:
            numpy array of flow features
        """
        features = []
        
        # Duration
        duration = flow_data.get('duration', 0)
        features.append(self._safe_log(duration))
        
        # Packet counts
        fwd_packets = flow_data.get('fwd_packets', flow_data.get('total_fwd_packets', 0))
        bwd_packets = flow_data.get('bwd_packets', flow_data.get('total_bwd_packets', 0))
        total_packets = fwd_packets + bwd_packets
        
        features.append(self._safe_log(total_packets))
        
        # Bytes
        fwd_bytes = flow_data.get('fwd_bytes', flow_data.get('total_fwd_bytes', 0))
        bwd_bytes = flow_data.get('bwd_bytes', flow_data.get('total_bwd_bytes', 0))
        total_bytes = fwd_bytes + bwd_bytes
        
        features.append(self._safe_log(total_bytes))
        
        # Rates
        if duration > 0:
            features.append(total_packets / duration)
            features.append(total_bytes / duration)
        else:
            features.extend([0, 0])
        
        # Packet size stats
        if total_packets > 0:
            avg_size = total_bytes / total_packets
            features.append(avg_size)
        else:
            features.append(0)
        
        features.append(flow_data.get('packet_size_std', 0))
        
        # Directional
        features.append(self._safe_log(fwd_packets))
        features.append(self._safe_log(bwd_packets))
        features.append(self._safe_log(fwd_bytes))
        features.append(self._safe_log(bwd_bytes))
        
        # Ratio
        if bwd_packets > 0:
            features.append(fwd_packets / bwd_packets)
        else:
            features.append(fwd_packets if fwd_packets > 0 else 0)
        
        # Inter-arrival time stats
        features.append(flow_data.get('iat_mean', 0))
        features.append(flow_data.get('iat_std', 0))
        features.append(flow_data.get('iat_max', 0))
        features.append(flow_data.get('iat_min', 0))
        
        # TCP flags
        features.append(flow_data.get('tcp_flags', 0))
        features.append(flow_data.get('syn_count', 0))
        features.append(flow_data.get('fin_count', 0))
        features.append(flow_data.get('rst_count', 0))
        
        return np.array(features, dtype=np.float32)
    
    def extract_protocol_features(self, protocol_data: Dict) -> np.ndarray:
        """Extract protocol-related features"""
        features = []
        
        protocol = protocol_data.get('protocol', '').upper()
        
        # One-hot style protocol encoding
        features.append(1.0 if protocol == 'TCP' else 0.0)
        features.append(1.0 if protocol == 'UDP' else 0.0)
        features.append(1.0 if protocol == 'ICMP' else 0.0)
        
        # Application layer
        service = protocol_data.get('service', '').lower()
        features.append(1.0 if service in ['http', 'www'] else 0.0)
        features.append(1.0 if service in ['https', 'ssl', 'tls'] else 0.0)
        features.append(1.0 if service == 'dns' else 0.0)
        features.append(1.0 if service == 'ssh' else 0.0)
        features.append(1.0 if service == 'rdp' else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def extract_port_features(self, port_data: Dict) -> np.ndarray:
        """Extract port-based features"""
        features = []
        
        src_port = port_data.get('src_port', 0)
        dst_port = port_data.get('dst_port', 0)
        
        # Normalized ports
        features.append(src_port / 65535.0)
        features.append(dst_port / 65535.0)
        
        # Port categories
        features.append(1.0 if dst_port < 1024 else 0.0)  # Well-known
        features.append(1.0 if 1024 <= dst_port < 49152 else 0.0)  # Registered
        features.append(1.0 if dst_port >= 49152 else 0.0)  # Dynamic
        
        # Port entropy (randomness indicator)
        features.append(self._calculate_entropy(str(dst_port)))
        
        # Suspicious ports
        suspicious_ports = {4444, 5555, 6666, 31337, 12345, 27374, 20034}
        features.append(1.0 if dst_port in suspicious_ports else 0.0)
        
        # Port class encoding
        if dst_port in [80, 443, 8080]:
            features.append(0.1)  # Web
        elif dst_port in [22, 23, 3389]:
            features.append(0.3)  # Remote access
        elif dst_port in [53, 123]:
            features.append(0.2)  # Infrastructure
        elif dst_port in [25, 587, 465]:
            features.append(0.4)  # Email
        else:
            features.append(0.5)  # Other
        
        return np.array(features, dtype=np.float32)
    
    def extract_ip_features(self, ip_data: Dict) -> np.ndarray:
        """Extract IP-based features"""
        features = []
        
        src_ip = ip_data.get('src_ip', '0.0.0.0')
        dst_ip = ip_data.get('dst_ip', '0.0.0.0')
        
        # IP entropy
        features.append(self._calculate_entropy(src_ip))
        
        # Internal/External classification
        features.append(1.0 if self._is_internal_ip(src_ip) else 0.0)
        features.append(1.0 if self._is_internal_ip(dst_ip) else 0.0)
        
        # Special addresses
        features.append(1.0 if dst_ip.endswith('.255') or dst_ip == '255.255.255.255' else 0.0)
        features.append(1.0 if dst_ip.startswith('224.') else 0.0)
        
        # Geo distance (placeholder - would use GeoIP in production)
        features.append(ip_data.get('geo_distance', 0.0))
        
        # IP reputation (placeholder - would use threat intel in production)
        features.append(ip_data.get('reputation_score', 0.5))
        
        # Known bad IP
        features.append(1.0 if ip_data.get('is_known_bad', False) else 0.0)
        
        return np.array(features, dtype=np.float32)
    
    def extract_behavioral_features(self, behavior_data: Dict) -> np.ndarray:
        """Extract behavioral/temporal features"""
        features = []
        
        # Time features
        timestamp = behavior_data.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        features.append(timestamp.hour / 24.0)  # Hour of day
        features.append(timestamp.weekday() / 7.0)  # Day of week
        features.append(1.0 if 9 <= timestamp.hour <= 17 else 0.0)  # Business hours
        
        # Connection patterns
        features.append(self._safe_log(behavior_data.get('connection_count', 0)))
        features.append(self._safe_log(behavior_data.get('unique_dst_ips', 0)))
        features.append(self._safe_log(behavior_data.get('unique_dst_ports', 0)))
        
        # Success/failure
        failed = behavior_data.get('failed_connections', 0)
        total = behavior_data.get('total_connections', 1)
        features.append(self._safe_log(failed))
        features.append((total - failed) / max(total, 1))  # Success rate
        
        # Data transfer
        sent = behavior_data.get('bytes_sent', 0)
        recv = behavior_data.get('bytes_recv', 0)
        features.append(sent / max(recv, 1) if recv > 0 else 0)  # Transfer ratio
        
        # Session
        features.append(self._safe_log(behavior_data.get('session_duration', 0)))
        features.append(self._safe_log(behavior_data.get('idle_time', 0)))
        features.append(behavior_data.get('activity_score', 0.5))
        
        return np.array(features, dtype=np.float32)
    
    def extract_attack_context_features(self, context_data: Dict) -> np.ndarray:
        """Extract attack-specific context features"""
        features = []
        
        # Attack pattern scores (0-1)
        features.append(context_data.get('lateral_movement_score', 0.0))
        features.append(context_data.get('privilege_escalation_score', 0.0))
        features.append(context_data.get('exfiltration_score', 0.0))
        features.append(context_data.get('c2_beaconing_score', 0.0))
        features.append(context_data.get('brute_force_score', 0.0))
        features.append(context_data.get('scan_score', 0.0))
        features.append(context_data.get('malware_score', 0.0))
        features.append(context_data.get('anomaly_score', 0.0))
        
        return np.array(features, dtype=np.float32)
    
    def extract_all(self, data: Dict) -> np.ndarray:
        """
        Extract all feature categories and combine into single vector
        
        Args:
            data: Dictionary containing all data fields
        
        Returns:
            numpy array of shape (feature_dim,)
        """
        # Extract each category
        flow = self.extract_flow_features(data)
        protocol = self.extract_protocol_features(data)
        port = self.extract_port_features(data)
        ip = self.extract_ip_features(data)
        behavior = self.extract_behavioral_features(data)
        context = self.extract_attack_context_features(data)
        
        # Concatenate
        all_features = np.concatenate([flow, protocol, port, ip, behavior, context])
        
        # Pad or truncate to feature_dim
        if len(all_features) < self.feature_dim:
            all_features = np.pad(all_features, (0, self.feature_dim - len(all_features)))
        elif len(all_features) > self.feature_dim:
            all_features = all_features[:self.feature_dim]
        
        return all_features.astype(np.float32)
    
    def extract_batch(self, data_list: List[Dict]) -> np.ndarray:
        """Extract features for a batch of samples"""
        return np.array([self.extract_all(d) for d in data_list])
    
    # ==================== Utility Methods ====================
    
    def _safe_log(self, x: float, base: float = 10) -> float:
        """Safe logarithm that handles zero/negative values"""
        return math.log(max(abs(x), 1e-10) + 1, base)
    
    def _calculate_entropy(self, s: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not s:
            return 0.0
        
        prob = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * math.log2(p + 1e-10) for p in prob)
    
    def _is_internal_ip(self, ip: str) -> bool:
        """Check if IP is internal/private"""
        private_prefixes = ('10.', '172.16.', '172.17.', '172.18.', '172.19.',
                           '172.20.', '172.21.', '172.22.', '172.23.', '172.24.',
                           '172.25.', '172.26.', '172.27.', '172.28.', '172.29.',
                           '172.30.', '172.31.', '192.168.', '127.')
        return ip.startswith(private_prefixes)


class AttackFeatureExtractor:
    """
    Specialized feature extractor for attack-type classification
    
    Extracts features optimized for distinguishing between:
    - Brute Force
    - Botnet/C2
    - Reconnaissance
    - Data Exfiltration
    - DDoS
    - Malware
    """
    
    ATTACK_TYPES = ['benign', 'brute_force', 'dos', 'ddos', 'botnet', 
                    'recon', 'exfiltration', 'web_attack', 'exploit', 'malware']
    
    def __init__(self):
        self.base_extractor = FeatureExtractor()
    
    def extract_brute_force_features(self, data: Dict) -> np.ndarray:
        """Features specific to brute force detection"""
        features = []
        
        # Authentication attempts
        features.append(data.get('failed_logins', 0))
        features.append(data.get('login_attempts', 0))
        features.append(data.get('unique_usernames', 0))
        
        # Timing
        features.append(data.get('attempts_per_minute', 0))
        
        # Success pattern
        failed = data.get('failed_logins', 0)
        total = data.get('login_attempts', 1)
        features.append(failed / max(total, 1))  # Failure rate
        
        return np.array(features, dtype=np.float32)
    
    def extract_botnet_features(self, data: Dict) -> np.ndarray:
        """Features specific to botnet/C2 detection"""
        features = []
        
        # Beaconing patterns
        features.append(data.get('connection_regularity', 0))  # Low variance = beaconing
        features.append(data.get('beacon_interval', 0))
        features.append(data.get('connection_count', 0))
        
        # C2 indicators
        features.append(1.0 if data.get('uses_dga_domain', False) else 0.0)
        features.append(data.get('dns_query_entropy', 0))
        
        # Payload characteristics
        features.append(data.get('encrypted_payload', 0))
        features.append(data.get('base64_encoded', 0))
        
        return np.array(features, dtype=np.float32)
    
    def extract_recon_features(self, data: Dict) -> np.ndarray:
        """Features specific to reconnaissance/scanning detection"""
        features = []
        
        # Scan patterns
        features.append(data.get('unique_ports_scanned', 0))
        features.append(data.get('unique_ips_contacted', 0))
        features.append(data.get('scan_rate', 0))
        
        # Response patterns
        features.append(data.get('rst_response_ratio', 0))
        features.append(data.get('no_response_ratio', 0))
        
        # Timing
        features.append(data.get('scan_duration', 0))
        
        return np.array(features, dtype=np.float32)
    
    def extract_exfiltration_features(self, data: Dict) -> np.ndarray:
        """Features specific to data exfiltration detection"""
        features = []
        
        # Data volume
        features.append(data.get('bytes_sent', 0))
        features.append(data.get('bytes_sent_per_hour', 0))
        
        # Transfer patterns
        features.append(data.get('upload_download_ratio', 0))
        features.append(data.get('large_transfers_count', 0))
        
        # Destination analysis
        features.append(1.0 if data.get('external_destination', False) else 0.0)
        features.append(data.get('destination_reputation', 0.5))
        
        # Encoding
        features.append(1.0 if data.get('compressed_payload', False) else 0.0)
        features.append(1.0 if data.get('encrypted_payload', False) else 0.0)
        
        return np.array(features, dtype=np.float32)


# Factory function
def create_feature_extractor(feature_dim: int = 64) -> FeatureExtractor:
    """Create a feature extractor instance"""
    return FeatureExtractor(feature_dim=feature_dim)


if __name__ == "__main__":
    print("Feature Extractor Test")
    print("=" * 50)
    
    # Test data
    sample = {
        'duration': 10.5,
        'fwd_packets': 100,
        'bwd_packets': 80,
        'fwd_bytes': 50000,
        'bwd_bytes': 40000,
        'protocol': 'TCP',
        'service': 'https',
        'src_port': 54321,
        'dst_port': 443,
        'src_ip': '192.168.1.100',
        'dst_ip': '8.8.8.8',
        'timestamp': datetime.now().isoformat(),
        'connection_count': 50,
        'unique_dst_ips': 5,
        'unique_dst_ports': 3
    }
    
    extractor = FeatureExtractor()
    features = extractor.extract_all(sample)
    
    print(f"✅ Extracted {len(features)} features")
    print(f"   Shape: {features.shape}")
    print(f"   Sample values: {features[:10]}")
