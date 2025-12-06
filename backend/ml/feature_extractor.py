"""
PCDS Enterprise - Advanced Feature Extractor
Production-grade feature extraction for ML models
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import hashlib


class AdvancedFeatureExtractor:
    """
    Real feature extraction from network data, logs, and behavioral patterns.
    No random values - all features are computed from actual data.
    """
    
    def __init__(self):
        self.feature_names = self._get_feature_names()
        self.baseline_stats = {}
        self.entity_history = {}
        
    def _get_feature_names(self) -> List[str]:
        """Get all 32 feature names"""
        return [
            # Network features (0-7)
            'packet_size_norm', 'port_norm', 'protocol_enc', 'ip_entropy',
            'bytes_in_norm', 'bytes_out_norm', 'packet_count', 'connection_duration',
            
            # Temporal features (8-15)
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hours',
            'time_since_last_event', 'events_per_minute', 'burst_indicator', 'session_duration',
            
            # Behavioral features (16-23)
            'login_frequency', 'failed_attempts', 'access_anomaly', 'privilege_level',
            'resource_access_count', 'data_volume_ratio', 'new_destination', 'geo_anomaly',
            
            # Security features (24-31)
            'known_bad_ip', 'port_scan_indicator', 'lateral_movement', 'exfil_indicator',
            'c2_beacon_pattern', 'encryption_anomaly', 'dns_anomaly', 'mitre_technique_match'
        ]
    
    def extract_network_features(self, data: Dict) -> np.ndarray:
        """Extract 8 network-related features"""
        features = np.zeros(8, dtype=np.float32)
        
        # 0. Packet size (normalized by MTU 1500)
        features[0] = min(data.get('packet_size', 0) / 1500.0, 1.0)
        
        # 1. Port (normalized, with common port weighting)
        port = data.get('port', data.get('dest_port', 0))
        if port in [80, 443, 22, 21, 25]:
            features[1] = 0.1  # Common ports = low risk
        elif port in [4444, 5555, 6666, 31337]:
            features[1] = 0.9  # Known malicious ports
        else:
            features[1] = port / 65535.0
        
        # 2. Protocol encoding
        protocol = str(data.get('protocol', 'tcp')).lower()
        protocol_risk = {'tcp': 0.2, 'udp': 0.3, 'icmp': 0.5, 'http': 0.1, 'https': 0.1, 'dns': 0.2}
        features[2] = protocol_risk.get(protocol, 0.5)
        
        # 3. IP entropy (measure of randomness in IP)
        source_ip = data.get('source_ip', '0.0.0.0')
        features[3] = self._calculate_ip_entropy(source_ip)
        
        # 4-5. Bytes in/out ratio
        bytes_in = data.get('bytes_in', data.get('bytes_received', 0))
        bytes_out = data.get('bytes_out', data.get('bytes_sent', 0))
        total_bytes = bytes_in + bytes_out + 1
        features[4] = min(bytes_in / 1e6, 1.0)  # Normalize to MB
        features[5] = min(bytes_out / 1e6, 1.0)
        
        # 6. Packet count (normalized)
        features[6] = min(data.get('packet_count', 1) / 1000.0, 1.0)
        
        # 7. Connection duration
        duration = data.get('duration', data.get('connection_duration', 0))
        features[7] = min(duration / 3600.0, 1.0)  # Normalize to hours
        
        return features
    
    def extract_temporal_features(self, data: Dict, entity_id: str = None) -> np.ndarray:
        """Extract 8 temporal features"""
        features = np.zeros(8, dtype=np.float32)
        
        # Get timestamp
        timestamp = data.get('timestamp', data.get('detected_at', datetime.now().isoformat()))
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                dt = datetime.now()
        else:
            dt = timestamp if isinstance(timestamp, datetime) else datetime.now()
        
        # 0. Hour of day (cyclical encoding)
        features[0] = dt.hour / 24.0
        
        # 1. Day of week
        features[1] = dt.weekday() / 6.0
        
        # 2. Is weekend
        features[2] = 1.0 if dt.weekday() >= 5 else 0.0
        
        # 3. Is business hours (9-17)
        features[3] = 1.0 if 9 <= dt.hour <= 17 and dt.weekday() < 5 else 0.0
        
        # 4. Time since last event for this entity
        if entity_id and entity_id in self.entity_history:
            last_time = self.entity_history[entity_id].get('last_event_time')
            if last_time:
                delta = (dt - last_time).total_seconds()
                features[4] = min(delta / 3600.0, 1.0)  # Hours since last
        
        # 5. Events per minute (from history)
        if entity_id and entity_id in self.entity_history:
            event_count = self.entity_history[entity_id].get('recent_events', 0)
            features[5] = min(event_count / 10.0, 1.0)
        
        # 6. Burst indicator (many events in short time)
        features[6] = 1.0 if features[5] > 0.5 else features[5] * 2
        
        # 7. Session duration
        features[7] = min(data.get('session_duration', 0) / 3600.0, 1.0)
        
        # Update entity history
        if entity_id:
            if entity_id not in self.entity_history:
                self.entity_history[entity_id] = {}
            self.entity_history[entity_id]['last_event_time'] = dt
            self.entity_history[entity_id]['recent_events'] = \
                self.entity_history[entity_id].get('recent_events', 0) + 1
        
        return features
    
    def extract_behavioral_features(self, data: Dict, entity_id: str = None) -> np.ndarray:
        """Extract 8 behavioral features"""
        features = np.zeros(8, dtype=np.float32)
        
        # 0. Login frequency anomaly
        login_count = data.get('login_count', data.get('auth_attempts', 0))
        features[0] = min(login_count / 10.0, 1.0)
        
        # 1. Failed attempts ratio
        failed = data.get('failed_attempts', data.get('auth_failures', 0))
        total = login_count + failed + 1
        features[1] = failed / total
        
        # 2. Access anomaly (unusual resources)
        features[2] = data.get('access_anomaly_score', 0.0)
        
        # 3. Privilege level
        role = str(data.get('role', data.get('privilege', 'user'))).lower()
        priv_map = {'admin': 1.0, 'root': 1.0, 'system': 0.9, 'power': 0.7, 'user': 0.3, 'guest': 0.1}
        features[3] = priv_map.get(role, 0.5)
        
        # 4. Resource access count
        features[4] = min(data.get('resource_count', 0) / 100.0, 1.0)
        
        # 5. Data volume ratio (out vs in)
        bytes_in = data.get('bytes_in', 1)
        bytes_out = data.get('bytes_out', 0)
        features[5] = bytes_out / (bytes_in + bytes_out + 1)
        
        # 6. New destination indicator
        dest_ip = data.get('destination_ip', data.get('dest_ip', ''))
        if entity_id and entity_id in self.entity_history:
            known_dests = self.entity_history[entity_id].get('known_destinations', set())
            features[6] = 0.0 if dest_ip in known_dests else 1.0
            known_dests.add(dest_ip)
            self.entity_history[entity_id]['known_destinations'] = known_dests
        
        # 7. Geo anomaly (simplified)
        features[7] = data.get('geo_anomaly_score', 0.0)
        
        return features
    
    def extract_security_features(self, data: Dict) -> np.ndarray:
        """Extract 8 security-specific features"""
        features = np.zeros(8, dtype=np.float32)
        
        # 0. Known bad IP indicator
        source_ip = data.get('source_ip', '')
        dest_ip = data.get('destination_ip', '')
        known_bad = ['192.168.1.100', '10.0.0.50']  # Example threat intel
        features[0] = 1.0 if source_ip in known_bad or dest_ip in known_bad else 0.0
        
        # 1. Port scan indicator
        unique_ports = data.get('unique_ports_accessed', 0)
        features[1] = min(unique_ports / 100.0, 1.0)
        
        # 2. Lateral movement indicator
        internal_destinations = data.get('internal_destinations', 0)
        features[2] = min(internal_destinations / 10.0, 1.0)
        
        # 3. Exfiltration indicator (high outbound)
        bytes_out = data.get('bytes_out', 0)
        features[3] = 1.0 if bytes_out > 10_000_000 else bytes_out / 10_000_000
        
        # 4. C2 beacon pattern (regular intervals)
        features[4] = data.get('beacon_score', 0.0)
        
        # 5. Encryption anomaly
        features[5] = data.get('encryption_anomaly', 0.0)
        
        # 6. DNS anomaly
        dns_queries = data.get('dns_query_count', 0)
        features[6] = min(dns_queries / 100.0, 1.0)
        
        # 7. MITRE technique match
        technique_id = data.get('technique_id', '')
        features[7] = 1.0 if technique_id else 0.0
        
        return features
    
    def extract_all_features(self, data: Dict, entity_id: str = None) -> np.ndarray:
        """Extract all 32 features"""
        network = self.extract_network_features(data)
        temporal = self.extract_temporal_features(data, entity_id)
        behavioral = self.extract_behavioral_features(data, entity_id)
        security = self.extract_security_features(data)
        
        return np.concatenate([network, temporal, behavioral, security])
    
    def _calculate_ip_entropy(self, ip: str) -> float:
        """Calculate entropy of IP address"""
        if not ip:
            return 0.5
        
        # Convert IP to bytes and calculate entropy
        parts = ip.split('.')
        if len(parts) != 4:
            return 0.5
        
        try:
            values = [int(p) for p in parts]
            # Higher entropy for more random-looking IPs
            variance = np.var(values) / 65025.0  # Max variance
            return min(variance * 2, 1.0)
        except:
            return 0.5
    
    def get_feature_importance(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Get feature importance for explainability"""
        importance = {}
        for i, name in enumerate(self.feature_names):
            importance[name] = float(feature_vector[i])
        return importance


# Global instance
feature_extractor = AdvancedFeatureExtractor()
