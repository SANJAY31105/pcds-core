"""MITRE ATT&CK Framework Integration Module"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from enum import Enum

class TacticID(str, Enum):
    """MITRE ATT&CK Tactic IDs"""
    INITIAL_ACCESS = "TA0001"
    EXECUTION = "TA0002"
    PERSISTENCE = "TA0003"
    PRIVILEGE_ESCALATION = "TA0004"
    DEFENSE_EVASION = "TA0005"
    CREDENTIAL_ACCESS = "TA0006"
    DISCOVERY = "TA0007"
    LATERAL_MOVEMENT = "TA0008"
    COLLECTION = "TA0009"
    COMMAND_AND_CONTROL = "TA0010"
    EXFILTRATION = "TA0011"
    IMPACT = "TA0040"

class MITREAttack:
    """MITRE ATT&CK Framework Manager"""
    
    def __init__(self):
        self.data_path = Path(__file__).parent / "data" / "mitre_attack.json"
        self.tactics: Dict = {}
        self.techniques: Dict = {}
        self.load_framework()
    
    def load_framework(self):
        """Load MITRE ATT&CK data from JSON"""
        try:
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                self.tactics = {t['id']: t for t in data['tactics']}
                self.techniques = data['techniques']
                print(f"✅ Loaded MITRE ATT&CK: {len(self.tactics)} tactics, {len(self.techniques)} techniques")
        except Exception as e:
            print(f"❌ Failed to load MITRE data: {e}")
            self.tactics = {}
            self.techniques = {}
    
    def get_tactic(self, tactic_id: str) -> Optional[Dict]:
        """Get tactic by ID"""
        return self.tactics.get(tactic_id)
    
    def get_technique(self, technique_id: str) -> Optional[Dict]:
        """Get technique by ID"""
        return self.techniques.get(technique_id)
    
    def get_all_tactics(self) -> List[Dict]:
        """Get all tactics"""
        return list(self.tactics.values())
    
    def get_techniques_by_tactic(self, tactic_id: str) -> List[Dict]:
        """Get all techniques for a tactic"""
        tactic = self.tactics.get(tactic_id)
        if not tactic:
            return []
        
        technique_ids = tactic.get('techniques', [])
        return [
            {**self.techniques[tid], 'id': tid} 
            for tid in technique_ids 
            if tid in self.techniques
        ]
    
    def map_detection_to_technique(self, detection_type: str) -> Optional[str]:
        """Map detection type to MITRE technique ID"""
        detection_mapping = {
            # Credential Access
            'brute_force': 'T1110',
            'credential_dumping': 'T1003',
            'keylogging': 'T1056',
            'kerberoasting': 'T1558',
            'password_spraying': 'T1110',
            
            # Lateral Movement
            'rdp_lateral': 'T1021',
            'smb_lateral': 'T1021',
            'psexec': 'T1021',
            'wmi_lateral': 'T1047',
            'pass_the_hash': 'T1550',
            
            # Privilege Escalation
            'token_manipulation': 'T1134',
            'process_injection': 'T1055',
            'uac_bypass': 'T1548',
            
            # Discovery
            'network_scan': 'T1046',
            'account_enumeration': 'T1087',
            'file_discovery': 'T1083',
            'process_discovery': 'T1057',
            
            # Command and Control
            'c2_beaconing': 'T1071',
            'dns_tunneling': 'T1071',
            'proxy_usage': 'T1090',
            
            # Exfiltration
            'data_exfiltration': 'T1041',
            'dns_exfiltration': 'T1048',
            'large_upload': 'T1567',
            
            # Impact
            'ransomware': 'T1486',
            'data_destruction': 'T1485',
            
            # Defense Evasion
            'log_deletion': 'T1070',
            'disable_security': 'T1562',
            
            # Execution
            'powershell_execution': 'T1059',
            'cmd_execution': 'T1059',
        }
        
        return detection_mapping.get(detection_type.lower())
    
    def get_kill_chain_stage(self, tactic_id: str) -> int:
        """Get kill chain stage number (1-12) for ordering"""
        stage_order = {
            'TA0001': 1,  # Initial Access
            'TA0002': 2,  # Execution
            'TA0003': 3,  # Persistence
            'TA0004': 4,  # Privilege Escalation
            'TA0005': 5,  # Defense Evasion
            'TA0006': 6,  # Credential Access
            'TA0007': 7,  # Discovery
            'TA0008': 8,  # Lateral Movement
            'TA0009': 9,  # Collection
            'TA0010': 10, # Command and Control
            'TA0011': 11, # Exfiltration
            'TA0040': 12, # Impact
        }
        return stage_order.get(tactic_id, 0)
    
    def enrich_detection(self, detection: Dict) -> Dict:
        """Enrich detection with MITRE ATT&CK context"""
        detection_type = detection.get('type', '')
        technique_id = self.map_detection_to_technique(detection_type)
        
        if technique_id:
            technique = self.get_technique(technique_id)
            if technique:
                tactic_id = technique['tactic']
                tactic = self.get_tactic(tactic_id)
                
                detection['mitre'] = {
                    'technique_id': technique_id,
                    'technique_name': technique['name'],
                    'tactic_id': tactic_id,
                    'tactic_name': tactic['name'] if tactic else 'Unknown',
                    'kill_chain_stage': self.get_kill_chain_stage(tactic_id),
                    'severity': technique.get('severity', 'medium')
                }
        
        return detection
    
    def get_matrix_heatmap(self, detections: List[Dict]) -> Dict:
        """Generate heatmap data for MITRE matrix visualization"""
        heatmap = {}
        
        for detection in detections:
            technique_id = detection.get('mitre', {}).get('technique_id')
            if technique_id:
                heatmap[technique_id] = heatmap.get(technique_id, 0) + 1
        
        return heatmap

# Global instance
mitre_attack = MITREAttack()
