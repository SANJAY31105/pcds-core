"""Advanced Detection Engine - Integrates all detection modules"""
from typing import Dict, List
import random
from datetime import datetime, timedelta

class DetectionEngine:
    """Main detection engine integrating all detection types"""
    
    def __init__(self):
        self.detection_rules = self.load_detection_rules()
    
    def load_detection_rules(self) -> Dict:
        """Load detection rules and thresholds"""
        return {
            # Credential Access
            'brute_force': {'threshold': 5, 'window': 300, 'severity': 'high'},
            'credential_dumping': {'threshold': 1, 'window': 0, 'severity': 'critical'},
            'password_spraying': {'threshold': 3, 'window': 600, 'severity': 'high'},
            'kerberoasting': {'threshold': 1, 'window': 0, 'severity': 'critical'},
            
            # Lateral Movement
            'rdp_lateral': {'threshold': 3, 'window': 300, 'severity': 'high'},
            'smb_lateral': {'threshold': 5, 'window': 600, 'severity': 'medium'},
            'psexec': {'threshold': 1, 'window': 0, 'severity': 'high'},
            'wmi_lateral': {'threshold': 2, 'window': 300, 'severity': 'high'},
            'pass_the_hash': {'threshold': 1, 'window': 0, 'severity': 'critical'},
            
            # Privilege Escalation
            'token_manipulation': {'threshold': 1, 'window': 0, 'severity': 'high'},
            'process_injection': {'threshold': 1, 'window': 0, 'severity': 'high'},
            'uac_bypass': {'threshold': 1, 'window': 0, 'severity': 'high'},
            
            # Discovery
            'network_scan': {'threshold': 50, 'window': 60, 'severity': 'medium'},
            'account_enumeration': {'threshold': 10, 'window': 300, 'severity': 'low'},
            'file_discovery': {'threshold': 100, 'window': 60, 'severity': 'low'},
            
            # Command & Control
            'c2_beaconing': {'threshold': 10, 'window': 3600, 'severity': 'critical'},
            'dns_tunneling': {'threshold': 20, 'window': 300, 'severity': 'high'},
            'proxy_usage': {'threshold': 5, 'window': 600, 'severity': 'medium'},
            
            # Exfiltration
            'data_exfiltration': {'threshold': 1, 'window': 0, 'severity': 'critical'},
            'large_upload': {'threshold': 100, 'window': 300, 'severity': 'high'},
            'dns_exfiltration': {'threshold': 50, 'window': 300, 'severity': 'high'},
            
            # Impact
            'ransomware': {'threshold': 1, 'window': 0, 'severity': 'critical'},
            'data_destruction': {'threshold': 1, 'window': 0, 'severity': 'critical'},
            
            # Defense Evasion
            'log_deletion': {'threshold': 1, 'window': 0, 'severity': 'high'},
            'disable_security': {'threshold': 1, 'window': 0, 'severity': 'critical'},
            
            # Execution
            'powershell_execution': {'threshold': 10, 'window': 300, 'severity': 'medium'},
            'cmd_execution': {'threshold': 20, 'window': 300, 'severity': 'low'},
        }
    
    def analyze_network_traffic(self, traffic_data: Dict) -> List[Dict]:
        """Analyze network traffic for threats"""
        detections = []
        
        # Simulate various detection types based on traffic patterns
        if random.random() < 0.3:  # 30% chance of detection
            detection_types = list(self.detection_rules.keys())
            detection_type = random.choice(detection_types)
            rule = self.detection_rules[detection_type]
            
            detection = {
                'type': detection_type,
                'severity': rule['severity'],
                'timestamp': datetime.now(),
                'confidence': random.uniform(0.7, 0.99),
                'metadata': {
                    'source_ip': traffic_data.get('src_ip', f'192.168.{random.randint(1,255)}.{random.randint(1,255)}'),
                    'dest_ip': traffic_data.get('dst_ip', f'10.0.{random.randint(1,255)}.{random.randint(1,255)}'),
                    'protocol': random.choice(['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS', 'SMB', 'RDP']),
                    'port': random.choice([22, 80, 443, 445, 3389, 53, 135, 139]),
                    'bytes_transferred': random.randint(1000, 10000000),
                    'connection_count': random.randint(1, 100)
                }
            }
            detections.append(detection)
        
        return detections
    
    def detect_credential_attacks(self, auth_logs: List[Dict]) -> List[Dict]:
        """Detect credential-based attacks"""
        detections = []
        
        # Brute force detection
        failed_logins = {}
        for log in auth_logs:
            if log.get('status') == 'failed':
                source = log.get('source_ip', 'unknown')
                failed_logins[source] = failed_logins.get(source, 0) + 1
        
        for source, count in failed_logins.items():
            if count >= self.detection_rules['brute_force']['threshold']:
                detections.append({
                    'type': 'brute_force',
                    'severity': 'high',
                    'timestamp': datetime.now(),
                    'confidence': 0.95,
                    'metadata': {
                        'source_ip': source,
                        'failed_attempts': count,
                        'target_accounts': [log.get('username') for log in auth_logs if log.get('source_ip') == source]
                    }
                })
        
        return detections
    
    def detect_lateral_movement(self, event_logs: List[Dict]) -> List[Dict]:
        """Detect lateral movement patterns"""
        detections = []
        
        # Track host-to-host connections
        lateral_connections = {}
        for event in event_logs:
            if event.get('event_type') in ['smb_connection', 'rdp_connection', 'psexec', 'wmi_exec']:
                source = event.get('source_host')
                target = event.get('target_host')
                key = f"{source}->{target}"
                lateral_connections[key] = lateral_connections.get(key, 0) + 1
        
        for connection, count in lateral_connections.items():
            if count >= 3:
                source, target = connection.split('->')
                detections.append({
                    'type': 'smb_lateral',
                    'severity': 'high',
                    'timestamp': datetime.now(),
                    'confidence': 0.88,
                    'metadata': {
                        'source_host': source,
                        'target_host': target,
                        'connection_count': count
                    }
                })
        
        return detections
    
    def detect_data_exfiltration(self, network_flow: List[Dict]) -> List[Dict]:
        """Detect data exfiltration attempts"""
        detections = []
        
        # Detect large outbound transfers
        for flow in network_flow:
            if flow.get('direction') == 'outbound':
                bytes_out = flow.get('bytes', 0)
                if bytes_out > 100_000_000:  # 100MB
                    detections.append({
                        'type': 'large_upload',
                        'severity': 'high',
                        'timestamp': datetime.now(),
                        'confidence': 0.82,
                        'metadata': {
                            'source_ip': flow.get('src_ip'),
                            'dest_ip': flow.get('dst_ip'),
                            'bytes_transferred': bytes_out,
                            'destination_country': flow.get('geo_country', 'Unknown')
                        }
                    })
        
        return detections
    
    def detect_c2_activity(self, dns_logs: List[Dict], network_logs: List[Dict]) -> List[Dict]:
        """Detect Command & Control activity"""
        detections = []
        
        # Detect beaconing patterns (regular intervals)
        connection_times = {}
        for log in network_logs:
            dest = log.get('dest_ip')
            timestamp = log.get('timestamp')
            if dest not in connection_times:
                connection_times[dest] = []
            connection_times[dest].append(timestamp)
        
        for dest, times in connection_times.items():
            if len(times) >= 10:
                # Check for regular intervals
                intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
                avg_interval = sum(intervals, timedelta()) / len(intervals)
                variance = sum([(i - avg_interval).total_seconds()**2 for i in intervals]) / len(intervals)
                
                if variance < 100:  # Low variance = regular beaconing
                    detections.append({
                        'type': 'c2_beaconing',
                        'severity': 'critical',
                        'timestamp': datetime.now(),
                        'confidence': 0.93,
                        'metadata': {
                            'c2_server': dest,
                            'beacon_count': len(times),
                            'avg_interval_seconds': avg_interval.total_seconds(),
                            'regularity_score': 1 - (variance / 1000)
                        }
                    })
        
        return detections
    
    def run_full_analysis(self, data: Dict) -> List[Dict]:
        """Run all detection modules"""
        all_detections = []
        
        # Run credential attack detection
        if 'auth_logs' in data:
            all_detections.extend(self.detect_credential_attacks(data['auth_logs']))
        
        # Run lateral movement detection
        if 'event_logs' in data:
            all_detections.extend(self.detect_lateral_movement(data['event_logs']))
        
        # Run exfiltration detection
        if 'network_flow' in data:
            all_detections.extend(self.detect_data_exfiltration(data['network_flow']))
        
        # Run C2 detection
        if 'dns_logs' in data and 'network_logs' in data:
            all_detections.extend(self.detect_c2_activity(data['dns_logs'], data['network_logs']))
        
        # Run network traffic analysis
        if 'traffic_data' in data:
            all_detections.extend(self.analyze_network_traffic(data['traffic_data']))
        
        return all_detections

# Global instance
detection_engine = DetectionEngine()
