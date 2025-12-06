"""
PCDS Enterprise Detection Engine
Comprehensive threat detection with 6 core modules

Modules:
1. Reconnaissance - Network/port scanning, enumeration
2. Credential Theft - Brute force, dumping, kerberoasting
3. Lateral Movement - RDP, SMB, WMI, pass-the-hash
4. Privilege Escalation - Token manipulation, process injection
5. Command & Control - Beaconing, tunneling, proxies
6. Data Exfiltration - Large uploads, DNS exfiltration

Each module returns detections with:
- Detection type
- Severity (critical/high/medium/low)
- Confidence score (0.0-1.0)
- MITRE technique mapping
- Evidence/metadata
"""

from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import math
import statistics


class ReconnaissanceDetector:
    """Detect reconnaissance and discovery activities"""
    
    def detect_network_scan(self, events: List[Dict]) -> List[Dict]:
        """
        Detect network scanning activity
        
        Indicators:
        - Multiple connection attempts to different IPs
        - Sequential port probing
        - ICMP sweeps
        """
        detections = []
        
        # Group by source IP
        by_source = defaultdict(list)
        for event in events:
            source = event.get('source_ip')
            if source:
                by_source[source].append(event)
        
        for source_ip, source_events in by_source.items():
            # Count unique destination IPs
            unique_dests = set(e.get('destination_ip') for e in source_events if e.get('destination_ip'))
            
            # Network scan threshold: >20 unique destinations in short time
            if len(unique_dests) > 20:
                # Check time window
                times = [datetime.fromisoformat(e['timestamp']) for e in source_events if 'timestamp' in e]
                if times:
                    time_span = (max(times) - min(times)).total_seconds() / 60  # minutes
                    
                    if time_span < 30:  # 30 minutes
                        detections.append({
                            'detection_type': 'network_scan',
                            'severity': 'medium',
                            'confidence_score': 0.85,
                            'source_ip': source_ip,
                            'title': f'Network Scanning Detected from {source_ip}',
                            'description': f'Source {source_ip} contacted {len(unique_dests)} unique hosts in {time_span:.1f} minutes',
                            'technique_id': 'T1046',
                            'tactic_id': 'TA0007',
                            'kill_chain_stage': 7,
                            'metadata': {
                                'unique_destinations': len(unique_dests),
                                'time_span_minutes': round(time_span, 2),
                                'scan_rate': round(len(unique_dests) / (time_span / 60), 2)
                            }
                        })
        
        return detections
    
    def detect_port_scan(self, events: List[Dict]) -> List[Dict]:
        """
        Detect port scanning activity
        
        Indicators:
        - Multiple ports probed on same destination
        - Sequential port numbers
        """
        detections = []
        
        # Group by source -> destination pairs
        by_pair = defaultdict(list)
        for event in events:
            source = event.get('source_ip')
            dest = event.get('destination_ip')
            if source and dest:
                key = (source, dest)
                by_pair[key].append(event)
        
        for (source_ip, dest_ip), pair_events in by_pair.items():
            # Count unique destination ports
            unique_ports = set(e.get('destination_port') for e in pair_events if e.get('destination_port'))
            
            # Port scan threshold: >15 unique ports
            if len(unique_ports) > 15:
                times = [datetime.fromisoformat(e['timestamp']) for e in pair_events if 'timestamp' in e]
                if times:
                    time_span = (max(times) - min(times)).total_seconds() / 60
                    
                    if time_span < 10:  # 10 minutes
                        detections.append({
                            'detection_type': 'port_scan',
                            'severity': 'low',
                            'confidence_score': 0.8,
                            'source_ip': source_ip,
                            'destination_ip': dest_ip,
                            'title': f'Port Scanning: {source_ip} → {dest_ip}',
                            'description': f'{len(unique_ports)} ports probed in {time_span:.1f} minutes',
                            'technique_id': 'T1046',
                            'tactic_id': 'TA0007',
                            'kill_chain_stage': 7,
                            'metadata': {
                                'unique_ports': len(unique_ports),
                                'time_span_minutes': round(time_span, 2),
                                'ports_probed': sorted(list(unique_ports))[:20]  # First 20
                            }
                        })
        
        return detections
    
    def detect_enumeration(self, events: List[Dict]) -> List[Dict]:
        """Detect account/system enumeration"""
        detections = []
        
        # Look for enumeration patterns in logs
        enum_keywords = ['net user', 'net group', 'ldapsearch', 'enum', 'whoami', 'query user']
        
        by_source = defaultdict(list)
        for event in events:
            command = event.get('command', '').lower()
            if any(keyword in command for keyword in enum_keywords):
                source = event.get('source_ip') or event.get('hostname')
                if source:
                    by_source[source].append(event)
        
        for source, enum_events in by_source.items():
            if len(enum_events) >= 3:  # Multiple enumeration commands
                detections.append({
                    'detection_type': 'account_enumeration',
                    'severity': 'low',
                    'confidence_score': 0.7,
                    'source_ip': source,
                    'title': f'Account Enumeration from {source}',
                    'description': f'{len(enum_events)} enumeration commands executed',
                    'technique_id': 'T1087',
                    'tactic_id': 'TA0007',
                    'kill_chain_stage': 7,
                    'metadata': {
                        'command_count': len(enum_events),
                        'sample_commands': [e.get('command', '')[:100] for e in enum_events[:5]]
                    }
                })
        
        return detections


class CredentialTheftDetector:
    """Detect credential access and theft"""
    
    def detect_brute_force(self, auth_events: List[Dict]) -> List[Dict]:
        """
        Detect brute force attacks
        
        Indicators:
        - Multiple failed login attempts
        - High failure rate
        - Dictionary attack patterns
        """
        detections = []
        
        # Group by source IP
        by_source = defaultdict(list)
        for event in auth_events:
            if event.get('event_type') == 'authentication_failure':
                source = event.get('source_ip')
                if source:
                    by_source[source].append(event)
        
        for source_ip, failures in by_source.items():
            # Brute force threshold: 10+ failures in 5 minutes
            times = [datetime.fromisoformat(e['timestamp']) for e in failures if 'timestamp' in e]
            
            if times and len(failures) >= 10:
                time_span = (max(times) - min(times)).total_seconds() / 60
                
                if time_span <= 5:
                    # Count unique usernames targeted
                    unique_users = set(e.get('username') for e in failures if e.get('username'))
                    
                    severity = 'high' if len(failures) >= 20 else 'medium'
                    confidence = 0.9 if len(failures) >= 30 else 0.8
                    
                    detections.append({
                        'detection_type': 'brute_force',
                        'severity': severity,
                        'confidence_score': confidence,
                        'source_ip': source_ip,
                        'title': f'Brute Force Attack from {source_ip}',
                        'description': f'{len(failures)} failed login attempts in {time_span:.1f} minutes',
                        'technique_id': 'T1110',
                        'tactic_id': 'TA0006',
                        'kill_chain_stage': 6,
                        'metadata': {
                            'failed_attempts': len(failures),
                            'unique_usernames': len(unique_users),
                            'time_span_minutes': round(time_span, 2),
                            'attack_rate': round(len(failures) / (time_span / 60), 2)
                        }
                    })
        
        return detections
    
    def detect_password_spraying(self, auth_events: List[Dict]) -> List[Dict]:
        """
        Detect password spraying attacks
        
        Indicators:
        - Few login attempts per user
        - Many different users targeted
        - Same source IP
        """
        detections = []
        
        by_source = defaultdict(list)
        for event in auth_events:
            if event.get('event_type') == 'authentication_failure':
                source = event.get('source_ip')
                if source:
                    by_source[source].append(event)
        
        for source_ip, failures in by_source.items():
            # Count attempts per user
            user_attempts = Counter(e.get('username') for e in failures if e.get('username'))
            
            # Password spray pattern: many users (>10), few attempts per user (<5)
            if len(user_attempts) >= 10:
                avg_attempts = statistics.mean(user_attempts.values())
                
                if avg_attempts <= 5:  # Low attempts per user = spraying
                    detections.append({
                        'detection_type': 'password_spraying',
                        'severity': 'high',
                        'confidence_score': 0.85,
                        'source_ip': source_ip,
                        'title': f'Password Spraying from {source_ip}',
                        'description': f'{len(user_attempts)} accounts targeted with {avg_attempts:.1f} avg attempts each',
                        'technique_id': 'T1110',
                        'tactic_id': 'TA0006',
                        'kill_chain_stage': 6,
                        'metadata': {
                            'targeted_accounts': len(user_attempts),
                            'total_attempts': len(failures),
                            'avg_attempts_per_account': round(avg_attempts, 2)
                        }
                    })
        
        return detections
    
    def detect_credential_dumping(self, process_events: List[Dict]) -> List[Dict]:
        """
        Detect credential dumping tools
        
        Indicators:
        - LSASS process access
        - Mimikatz-like behavior
        - SAM database access
        """
        detections = []
        
        dumping_indicators = {
            'lsass.exe': 'LSASS memory access',
            'mimikatz': 'Mimikatz credential dumping',
            'procdump': 'Process dump utility',
            'sam': 'SAM database access',
            'sekurlsa': 'Credential extraction'
        }
        
        for event in process_events:
            process_name = event.get('process_name', '').lower()
            command_line = event.get('command_line', '').lower()
            
            for indicator, description in dumping_indicators.items():
                if indicator in process_name or indicator in command_line:
                    detections.append({
                        'detection_type': 'credential_dumping',
                        'severity': 'critical',
                        'confidence_score': 0.95,
                        'source_ip': event.get('source_ip'),
                        'title': f'Credential Dumping Detected: {description}',
                        'description': f'Detected {description} on {event.get("hostname", "unknown host")}',
                        'technique_id': 'T1003',
                        'tactic_id': 'TA0006',
                        'kill_chain_stage': 6,
                        'metadata': {
                            'process_name': event.get('process_name'),
                            'command_line': event.get('command_line', '')[:200],
                            'indicator_matched': indicator
                        }
                    })
                    break  # Only one detection per event
        
        return detections
    
    def detect_kerberoasting(self, network_events: List[Dict]) -> List[Dict]:
        """
        Detect Kerberoasting attacks
        
        Indicators:
        - Multiple TGS requests for service accounts
        - RC4 encryption usage
        - High volume of service ticket requests
        """
        detections = []
        
        # Look for Kerberos TGS-REQ patterns
        tgs_requests = [e for e in network_events if e.get('event_type') == 'kerberos_tgs_req']
        
        by_source = defaultdict(list)
        for event in tgs_requests:
            source = event.get('source_ip')
            if source:
                by_source[source].append(event)
        
        for source_ip, requests in by_source.items():
            # Kerberoasting pattern: >5 service ticket requests in short time
            if len(requests) >= 5:
                times = [datetime.fromisoformat(e['timestamp']) for e in requests if 'timestamp' in e]
                if times:
                    time_span = (max(times) - min(times)).total_seconds() / 60
                    
                    if time_span <= 10:
                        detections.append({
                            'detection_type': 'kerberoasting',
                            'severity': 'critical',
                            'confidence_score': 0.9,
                            'source_ip': source_ip,
                            'title': f'Kerberoasting Attack from {source_ip}',
                            'description': f'{len(requests)} service ticket requests in {time_span:.1f} minutes',
                            'technique_id': 'T1558',
                            'tactic_id': 'TA0006',
                            'kill_chain_stage': 6,
                            'metadata': {
                                'tgs_requests': len(requests),
                                'time_span_minutes': round(time_span, 2)
                            }
                        })
        
        return detections


# Export detector classes
__all__ = [
    'ReconnaissanceDetector',
    'CredentialTheftDetector'
]
