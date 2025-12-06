"""
MITRE ATT&CK Mapper
Automatically maps detections to MITRE ATT&CK techniques

Features:
- Detection type → Technique ID mapping
- Technique enrichment with metadata
- Tactic classification
- Kill chain stage assignment
- Severity inheritance from technique
"""
from typing import Tuple, Dict, List, Optional
import json
from typing import Dict, Optional, List
from pathlib import Path
from config.settings import settings
from config.database import MITREQueries


class MITREMapper:
    """
    Automatic MITRE ATT&CK technique mapping for detections
    """
    
    def __init__(self):
        self.detection_mappings = {}
        self.techniques_cache = {}
        self.tactics_cache = {}
        self._load_mitre_data()
    
    def _load_mitre_data(self):
        """Load MITRE ATT&CK data from database and JSON"""
        try:
            # Load detection type → technique mappings from JSON
            mitre_file = settings.MITRE_DATA_FILE
            if mitre_file.exists():
                with open(mitre_file, 'r') as f:
                    data = json.load(f)
                    self.detection_mappings = data.get('detection_mappings', {})
            
            # Load techniques from database
            all_tactics = MITREQueries.get_all_tactics()
            for tactic in all_tactics:
                self.tactics_cache[tactic['id']] = tactic
            
            print(f"✅ MITRE Mapper initialized: {len(self.detection_mappings)} detection mappings")
            
        except Exception as e:
            print(f"⚠️  MITRE Mapper warning: {e}")
            # Use fallback mappings
            self._load_fallback_mappings()
    
    def _load_fallback_mappings(self):
        """Fallback detection mappings if file not available"""
        self.detection_mappings = {
            'brute_force': 'T1110',
            'password_spraying': 'T1110',
            'credential_dumping': 'T1003',
            'kerberoasting': 'T1558',
            'network_scan': 'T1046',
            'port_scan': 'T1046',
            'account_enumeration': 'T1087',
            'rdp_lateral': 'T1021',
            'smb_lateral': 'T1021',
            'wmi_lateral': 'T1047',
            'pass_the_hash': 'T1550',
            'token_manipulation': 'T1134',
            'process_injection': 'T1055',
            'uac_bypass': 'T1548',
            'c2_beaconing': 'T1071',
            'dns_tunneling': 'T1071',
            'large_upload': 'T1567',
            'dns_exfiltration': 'T1048',
            'ransomware': 'T1486',
            'data_destruction': 'T1485'
        }
    
    def enrich_detection(self, detection: Dict) -> Dict:
        """
        Enrich detection with MITRE ATT&CK context
        
        Args:
            detection: Detection dictionary
        
        Returns:
            Enriched detection with technique_id, tactic_id, kill_chain_stage
        """
        detection_type = detection.get('detection_type', '')
        
        # Get technique ID from mapping
        technique_id = detection.get('technique_id') or self.detection_mappings.get(detection_type)
        
        if not technique_id:
            # No mapping found
            return detection
        
        # Get technique details from database
        technique = MITREQueries.get_technique(technique_id)
        
        if technique:
            # Add MITRE context
            detection['technique_id'] = technique_id
            detection['technique_name'] = technique['name']
            detection['tactic_id'] = technique['tactic_id']
            
            # Get tactic details
            tactic = self.tactics_cache.get(technique['tactic_id'])
            if tactic:
                detection['tactic_name'] = tactic['name']
                detection['kill_chain_stage'] = tactic['kill_chain_order']
            
            # Add technique metadata
            detection['mitre'] = {
                'technique_id': technique_id,
                'technique_name': technique['name'],
                'tactic_id': technique['tactic_id'],
                'tactic_name': tactic['name'] if tactic else 'Unknown',
                'severity': technique.get('severity', 'medium'),
                'platforms': json.loads(technique.get('platforms', '[]')),
                'description': technique.get('description', '')
            }
            
            # Inherit severity from technique if not set
            if not detection.get('severity'):
                detection['severity'] = technique.get('severity', 'medium')
        
        return detection
    
    def get_technique_info(self, technique_id: str) -> Optional[Dict]:
        """Get detailed information about a MITRE technique"""
        if technique_id in self.techniques_cache:
            return self.techniques_cache[technique_id]
        
        technique = MITREQueries.get_technique(technique_id)
        if technique:
            self.techniques_cache[technique_id] = technique
            return technique
        
        return None
    
    def get_tactic_info(self, tactic_id: str) -> Optional[Dict]:
        """Get detailed information about a MITRE tactic"""
        return self.tactics_cache.get(tactic_id)
    
    def get_techniques_for_tactic(self, tactic_id: str) -> List[Dict]:
        """Get all techniques for a given tactic"""
        return MITREQueries.get_techniques_by_tactic(tactic_id)
    
    def get_kill_chain_stage(self, technique_id: str) -> Optional[int]:
        """Get kill chain stage number for a technique"""
        technique = self.get_technique_info(technique_id)
        if technique:
            tactic = self.tactics_cache.get(technique['tactic_id'])
            if tactic:
                return tactic['kill_chain_order']
        return None
    
    def validate_detection(self, detection: Dict) -> Tuple[bool, str]:
        """
        Validate that a detection has proper MITRE mapping
        
        Returns:
            (is_valid, error_message)
        """
        detection_type = detection.get('detection_type')
        if not detection_type:
            return False, "Missing detection_type"
        
        technique_id = detection.get('technique_id') or self.detection_mappings.get(detection_type)
        if not technique_id:
            return False, f"No MITRE mapping for detection type: {detection_type}"
        
        technique = self.get_technique_info(technique_id)
        if not technique:
            return False, f"Invalid technique ID: {technique_id}"
        
        return True, "Valid"


# Global MITRE mapper instance
mitre_mapper = MITREMapper()


def enrich_detection_with_mitre(detection: Dict) -> Dict:
    """
    Convenience function to enrich a detection with MITRE context
    
    Usage:
        detection = {'detection_type': 'brute_force', 'severity': 'high', ...}
        enriched = enrich_detection_with_mitre(detection)
        # Now has technique_id, tactic_id, kill_chain_stage, mitre metadata
    """
    return mitre_mapper.enrich_detection(detection)
