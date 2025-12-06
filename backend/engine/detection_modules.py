"""
Detection Engine Modules 3-6:
- Lateral Movement
- Privilege Escalation  
- Command & Control
- Data Exfiltration
"""

from typing import List, Dict
from collections import defaultdict, Counter
from datetime import datetime
import statistics
import math


class LateralMovementDetector:
    """Detect lateral movement activities"""
    
    def detect_rdp_lateral(self, network_events: List[Dict]) -> List[Dict]:
        """
        Detect lateral RDP connections
        
        Indicators:
        - RDP connections between internal hosts
        - Unusual RDP source IPs
        - RDP connection chains
        """
        detections = []
        
        rdp_events = [e for e in network_events if e.get('destination_port') == 3389]
        
        # Group by source IP
        by_source = defaultdict(list)
        for event in rdp_events:
            source = event.get('source_ip', '')
            # Filter internal-to-internal only
            if source.startswith(('192.168.', '10.', '172.')):
                by_source[source].append(event)
        
        for source_ip, events in by_source.items():
            unique_dests = set(e.get('destination_ip') for e in events if e.get('destination_ip'))
            
            # Lateral RDP: source connecting to 2+ internal hosts
            if len(unique_dests) >= 2:
                detections.append({
                    'detection_type': 'rdp_lateral',
                    'severity': 'high',
                    'confidence_score': 0.85,
                    'source_ip': source_ip,
                    'title': f'Lateral RDP Movement from {source_ip}',
                    'description': f'RDP connections to {len(unique_dests)} internal hosts',
                    'technique_id': 'T1021',
                    'tactic_id': 'TA0008',
                    'kill_chain_stage': 8,
                    'metadata': {
                        'connection_count': len(events),
                        'unique_destinations': len(unique_dests),
                        'destination_ips': list(unique_dests)
                    }
                })
        
        return detections
    
    def detect_smb_lateral(self, network_events: List[Dict]) -> List[Dict]:
        """Detect lateral SMB/File share activity"""
        detections = []
        
        smb_events = [e for e in network_events if e.get('destination_port') in [445, 139]]
        
        by_source = defaultdict(list)
        for event in smb_events:
            source = event.get('source_ip', '')
            if source.startswith(('192.168.', '10.', '172.')):
                by_source[source].append(event)
        
        for source_ip, events in by_source.items():
            unique_dests = set(e.get('destination_ip') for e in events if e.get('destination_ip'))
            
            if len(unique_dests) >= 3:  # SMB to 3+ hosts
                detections.append({
                    'detection_type': 'smb_lateral',
                    'severity': 'medium',
                    'confidence_score': 0.75,
                    'source_ip': source_ip,
                    'title': f'Lateral SMB Activity from {source_ip}',
                    'description': f'SMB connections to {len(unique_dests)} hosts',
                    'technique_id': 'T1021',
                    'tactic_id': 'TA0008',
                    'kill_chain_stage': 8,
                    'metadata': {
                        'connection_count': len(events),
                        'unique_destinations': len(unique_dests)
                    }
                })
        
        return detections
    
    def detect_wmi_lateral(self, process_events: List[Dict]) -> List[Dict]:
        """Detect WMI-based lateral movement"""
        detections = []
        
        wmi_processes = ['wmic.exe', 'wmiprvse.exe']
        
        wmi_events = [
            e for e in process_events 
            if any(proc in e.get('process_name', '').lower() for proc in wmi_processes)
        ]
        
        # Look for remote WMI execution patterns
        for event in wmi_events:
            command = event.get('command_line', '').lower()
            
            # Remote WMI indicators
            if any(keyword in command for keyword in ['/node:', 'remote', 'process call create']):
                detections.append({
                    'detection_type': 'wmi_lateral',
                    'severity': 'high',
                    'confidence_score': 0.8,
                    'source_ip': event.get('source_ip'),
                    'title': 'WMI Lateral Movement Detected',
                    'description': f'Remote WMI execution on {event.get("hostname", "unknown")}',
                    'technique_id': 'T1047',
                    'tactic_id': 'TA0008',
                    'kill_chain_stage': 8,
                    'metadata': {
                        'process_name': event.get('process_name'),
                        'command_line': event.get('command_line', '')[:200]
                    }
                })
        
        return detections
    
    def detect_pass_the_hash(self, auth_events: List[Dict]) -> List[Dict]:
        """Detect pass-the-hash attacks"""
        detections = []
        
        # Look for NTLM auth without corresponding Kerberos
        ntlm_auths = [e for e in auth_events if e.get('auth_type') == 'NTLM']
        
        by_source = defaultdict(list)
        for event in ntlm_auths:
            source = event.get('source_ip')
            if source and event.get('event_type') == 'authentication_success':
                by_source[source].append(event)
        
        for source_ip, auths in by_source.items():
            unique_users = set(e.get('username') for e in auths if e.get('username'))
            
            # Pass-the-hash pattern: Multiple successful NTLM auths, multiple users
            if len(unique_users) >= 2 and len(auths) >= 3:
                detections.append({
                    'detection_type': 'pass_the_hash',
                    'severity': 'critical',
                    'confidence_score': 0.85,
                    'source_ip': source_ip,
                    'title': f'Pass-the-Hash Attack from {source_ip}',
                    'description': f'{len(unique_users)} accounts used via NTLM authentication',
                    'technique_id': 'T1550',
                    'tactic_id': 'TA0008',
                    'kill_chain_stage': 8,
                    'metadata': {
                        'ntlm_auth_count': len(auths),
                        'unique_accounts': len(unique_users)
                    }
                })
        
        return detections


class PrivilegeEscalationDetector:
    """Detect privilege escalation attempts"""
    
    def detect_token_manipulation(self, process_events: List[Dict]) -> List[Dict]:
        """Detect access token manipulation"""
        detections = []
        
        token_keywords = ['runas', 'token', 'impersonate', 'getsystem']
        
        for event in process_events:
            command = event.get('command_line', '').lower()
            
            if any(keyword in command for keyword in token_keywords):
                detections.append({
                    'detection_type': 'token_manipulation',
                    'severity': 'high',
                    'confidence_score': 0.8,
                    'source_ip': event.get('source_ip'),
                    'title': 'Token Manipulation Detected',
                    'description': f'Suspicious token/privilege command on {event.get("hostname", "unknown")}',
                    'technique_id': 'T1134',
                    'tactic_id': 'TA0004',
                    'kill_chain_stage': 4,
                    'metadata': {
                        'process_name': event.get('process_name'),
                        'command_line': event.get('command_line', '')[:200]
                    }
                })
        
        return detections
    
    def detect_process_injection(self, process_events: List[Dict]) -> List[Dict]:
        """Detect process injection techniques"""
        detections = []
        
        injection_indicators = ['inject', 'reflective', 'hollowing', 'doppelganging']
        
        for event in process_events:
            command = event.get('command_line', '').lower()
            process_name = event.get('process_name', '').lower()
            
            if any(indicator in command or indicator in process_name for indicator in injection_indicators):
                detections.append({
                    'detection_type': 'process_injection',
                    'severity': 'high',
                    'confidence_score': 0.85,
                    'source_ip': event.get('source_ip'),
                    'title': 'Process Injection Detected',
                    'description': f'Suspected process injection on {event.get("hostname", "unknown")}',
                    'technique_id': 'T1055',
                    'tactic_id': 'TA0004',
                    'kill_chain_stage': 4,
                    'metadata': {
                        'process_name': event.get('process_name'),
                        'command_line': event.get('command_line', '')[:200]
                    }
                })
        
        return detections
    
    def detect_uac_bypass(self, process_events: List[Dict]) -> List[Dict]:
        """Detect UAC bypass attempts"""
        detections = []
        
        uac_bypass_patterns = ['eventvwr.exe', 'fodhelper.exe', 'computerdefaults.exe']
        
        for event in process_events:
            process_name = event.get('process_name', '').lower()
            parent_process = event.get('parent_process_name', '').lower()
            
            # UAC bypass: suspicious process launching high-integrity child
            if any(pattern in process_name or pattern in parent_process for pattern in uac_bypass_patterns):
                detections.append({
                    'detection_type': 'uac_bypass',
                    'severity': 'high',
                    'confidence_score': 0.75,
                    'source_ip': event.get('source_ip'),
                    'title': 'UAC Bypass Attempt Detected',
                    'description': f'Suspicious UAC bypass pattern on {event.get("hostname", "unknown")}',
                    'technique_id': 'T1548',
                    'tactic_id': 'TA0004',
                    'kill_chain_stage': 4,
                    'metadata': {
                        'process_name': event.get('process_name'),
                        'parent_process': event.get('parent_process_name')
                    }
                })
        
        return detections


class CommandAndControlDetector:
    """Detect C2 communications"""
    
    def detect_c2_beaconing(self, network_events: List[Dict]) -> List[Dict]:
        """
        Detect C2 beaconing patterns
        
        Indicators:
        - Regular intervals
        - Consistent packet sizes
        - Unusual destinations
        """
        detections = []
        
        # Group by source -> destination pairs
        by_pair = defaultdict(list)
        for event in network_events:
            source = event.get('source_ip')
            dest = event.get('destination_ip')
            if source and dest:
                by_pair[(source, dest)].append(event)
        
        for (source_ip, dest_ip), events in by_pair.items():
            if len(events) < 10:  # Need enough data points
                continue
            
            # Extract timestamps
            times = []
            for e in events:
                try:
                    t = datetime.fromisoformat(e['timestamp'])
                    times.append(t)
                except:
                    continue
            
            if len(times) < 10:
                continue
            
            # Calculate intervals
            times.sort()
            intervals = [(times[i+1] - times[i]).total_seconds() for i in range(len(times) - 1)]
            
            if len(intervals) >= 5:
                # Check for regularity (low variance = beaconing)
                avg_interval = statistics.mean(intervals)
                variance = statistics.variance(intervals)
                std_dev = math.sqrt(variance)
                
                # Coefficient of variation
                cv = (std_dev / avg_interval) if avg_interval > 0 else 1
                
                # Beaconing pattern: cv < 0.3 (highly regular) and reasonable interval
                if cv < 0.3 and 10 < avg_interval < 600:  # 10s to 10min
                    detections.append({
                        'detection_type': 'c2_beaconing',
                        'severity': 'critical',
                        'confidence_score': 0.9,
                        'source_ip': source_ip,
                        'destination_ip': dest_ip,
                        'title': f'C2 Beaconing: {source_ip} → {dest_ip}',
                        'description': f'Regular beacon every {avg_interval:.1f}s (CV: {cv:.3f})',
                        'technique_id': 'T1071',
                        'tactic_id': 'TA0010',
                        'kill_chain_stage': 10,
                        'metadata': {
                            'beacon_count': len(events),
                            'avg_interval_seconds': round(avg_interval, 2),
                            'coefficient_of_variation': round(cv, 3),
                            'regularity_score': round(1 - cv, 3)
                        }
                    })
        
        return detections
    
    def detect_dns_tunneling(self, dns_events: List[Dict]) -> List[Dict]:
        """Detect DNS tunneling"""
        detections = []
        
        # Group by source IP
        by_source = defaultdict(list)
        for event in dns_events:
            if event.get('event_type') == 'dns_query':
                source = event.get('source_ip')
                if source:
                    by_source[source].append(event)
        
        for source_ip, queries in by_source.items():
            # Count queries and unique domains
            unique_domains = set(e.get('query_name', '') for e in queries)
            
            # DNS tunneling indicators:
            # - High query volume
            # - Long domain names
            # - Unusual subdomains
            long_queries = [q for q in queries if len(q.get('query_name', '')) > 50]
            
            if len(queries) > 100 or len(long_queries) > 20:
                avg_length = statistics.mean([len(q.get('query_name', '')) for q in queries])
                
                if avg_length > 30:  # Unusually long
                    detections.append({
                        'detection_type': 'dns_tunneling',
                        'severity': 'high',
                        'confidence_score': 0.8,
                        'source_ip': source_ip,
                        'title': f'DNS Tunneling from {source_ip}',
                        'description': f'{len(queries)} DNS queries, avg length {avg_length:.1f} chars',
                        'technique_id': 'T1071',
                        'tactic_id': 'TA0010',
                        'kill_chain_stage': 10,
                        'metadata': {
                            'query_count': len(queries),
                            'unique_domains': len(unique_domains),
                            'avg_query_length': round(avg_length, 1),
                            'long_queries': len(long_queries)
                        }
                    })
        
        return detections


class DataExfiltrationDetector:
    """Detect data exfiltration"""
    
    def detect_large_upload(self, network_events: List[Dict]) -> List[Dict]:
        """Detect large outbound data transfers"""
        detections = []
        
        # Group by source IP
        by_source = defaultdict(list)
        for event in network_events:
            if event.get('direction') == 'outbound':
                source = event.get('source_ip')
                if source:
                    by_source[source].append(event)
        
        for source_ip, events in by_source.items():
            # Calculate total bytes uploaded
            total_bytes = sum(e.get('bytes_sent', 0) for e in events)
            
            # Large upload threshold: >100MB
            if total_bytes > 100 * 1024 * 1024:
                # Check time span
                times = []
                for e in events:
                    try:
                        times.append(datetime.fromisoformat(e['timestamp']))
                    except:
                        pass
                
                if times:
                    time_span = (max(times) - min(times)).total_seconds() / 3600  # hours
                    
                    unique_dests = set(e.get('destination_ip') for e in events if e.get('destination_ip'))
                    
                    detections.append({
                        'detection_type': 'large_upload',
                        'severity': 'high',
                        'confidence_score': 0.75,
                        'source_ip': source_ip,
                        'title': f'Large Data Upload from {source_ip}',
                        'description': f'{total_bytes / (1024*1024):.1f} MB uploaded in {time_span:.1f} hours',
                        'technique_id': 'T1567',
                        'tactic_id': 'TA0011',
                        'kill_chain_stage': 11,
                        'metadata': {
                            'bytes_uploaded': total_bytes,
                            'megabytes': round(total_bytes / (1024*1024), 2),
                            'unique_destinations': len(unique_dests),
                            'time_span_hours': round(time_span, 2)
                        }
                    })
        
        return detections
    
    def detect_dns_exfiltration(self, dns_events: List[Dict]) -> List[Dict]:
        """Detect data exfiltration via DNS"""
        detections = []
        
        by_source = defaultdict(list)
        for event in dns_events:
            if event.get('event_type') == 'dns_query':
                source = event.get('source_ip')
                query = event.get('query_name', '')
                
                # Look for base64-like patterns in subdomains
                if '.' in query:
                    subdomain = query.split('.')[0]
                    if len(subdomain) > 30 and subdomain.replace('-', '').replace('_', '').isalnum():
                        by_source[source].append(event)
        
        for source_ip, suspicious_queries in by_source.items():
            if len(suspicious_queries) >= 10:
                detections.append({
                    'detection_type': 'dns_exfiltration',
                    'severity': 'critical',
                    'confidence_score': 0.85,
                    'source_ip': source_ip,
                    'title': f'DNS Exfiltration from {source_ip}',
                    'description': f'{len(suspicious_queries)} suspicious DNS queries detected',
                    'technique_id': 'T1048',
                    'tactic_id': 'TA0011',
                    'kill_chain_stage': 11,
                    'metadata': {
                        'suspicious_query_count': len(suspicious_queries),
                        'sample_queries': [e.get('query_name', '') for e in suspicious_queries[:5]]
                    }
                })
        
        return detections


# Export all detector classes
__all__ = [
    'LateralMovementDetector',
    'PrivilegeEscalationDetector',
    'CommandAndControlDetector',
    'DataExfiltrationDetector'
]
