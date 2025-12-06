"""
MITRE ATT&CK Framework Integration
Maps detected threats to tactics, techniques, and procedures
"""
from typing import List, Dict, Optional
from enum import Enum


class MITRETactic(str, Enum):
    """MITRE ATT&CK Tactics"""
    RECONNAISSANCE = "reconnaissance"
    RESOURCE_DEVELOPMENT = "resource-development"
    INITIAL_ACCESS = "initial-access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege-escalation"
    DEFENSE_EVASION = "defense-evasion"
    CREDENTIAL_ACCESS = "credential-access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral-movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command-and-control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


class MITRETechnique:
    """MITRE ATT&CK Technique"""
    
    def __init__(self, technique_id: str, name: str, tactic: MITRETactic, description: str):
        self.technique_id = technique_id
        self.name = name
        self.tactic = tactic
        self.description = description


class MITREMapper:
    """
    Maps threat detections to MITRE ATT&CK framework
    """
    
    def __init__(self):
        self.techniques = self._load_techniques()
        
    def _load_techniques(self) -> Dict[str, MITRETechnique]:
        """Load MITRE ATT&CK techniques database"""
        techniques = {}
        
        # Network-based techniques (subset for demo)
        techniques_data = [
            # Initial Access
            ("T1190", "Exploit Public-Facing Application", MITRETactic.INITIAL_ACCESS,
             "Adversaries may attempt to exploit a weakness in an Internet-facing host or system"),
            ("T1133", "External Remote Services", MITRETactic.INITIAL_ACCESS,
             "Adversaries may leverage external-facing remote services to gain initial access"),
            
            # Persistence
            ("T1078", "Valid Accounts", MITRETactic.PERSISTENCE,
             "Adversaries may obtain and abuse credentials of existing accounts"),
            
            # Privilege Escalation
            ("T1068", "Exploitation for Privilege Escalation", MITRETactic.PRIVILEGE_ESCALATION,
             "Adversaries may exploit software vulnerabilities to elevate privileges"),
            
            # Defense Evasion
            ("T1070", "Indicator Removal", MITRETactic.DEFENSE_EVASION,
             "Adversaries may delete or modify artifacts to remove evidence"),
            ("T1027", "Obfuscated Files or Information", MITRETactic.DEFENSE_EVASION,
             "Adversaries may attempt to make payloads difficult to discover"),
            
            # Credential Access
            ("T1110", "Brute Force", MITRETactic.CREDENTIAL_ACCESS,
             "Adversaries may use brute force techniques to gain access to accounts"),
            ("T1003", "OS Credential Dumping", MITRETactic.CREDENTIAL_ACCESS,
             "Adversaries may attempt to dump credentials to obtain account login information"),
            
            # Discovery
            ("T1046", "Network Service Discovery", MITRETactic.DISCOVERY,
             "Adversaries may attempt to get a listing of services running on remote hosts"),
            ("T1018", "Remote System Discovery", MITRETactic.DISCOVERY,
             "Adversaries may attempt to get a listing of other systems"),
            ("T1087", "Account Discovery", MITRETactic.DISCOVERY,
             "Adversaries may attempt to get a listing of valid accounts"),
            
            # Lateral Movement
            ("T1021", "Remote Services", MITRETactic.LATERAL_MOVEMENT,
             "Adversaries may use valid accounts to log into a service"),
            ("T1210", "Exploitation of Remote Services", MITRETactic.LATERAL_MOVEMENT,
             "Adversaries may exploit remote services to gain unauthorized access"),
            
            # Collection
            ("T1005", "Data from Local System", MITRETactic.COLLECTION,
             "Adversaries may search local system sources for data"),
            ("T1039", "Data from Network Shared Drive", MITRETactic.COLLECTION,
             "Adversaries may search network shares for sensitive data"),
            
            # Command and Control
            ("T1071", "Application Layer Protocol", MITRETactic.COMMAND_AND_CONTROL,
             "Adversaries may communicate using application layer protocols"),
            ("T1095", "Non-Application Layer Protocol", MITRETactic.COMMAND_AND_CONTROL,
             "Adversaries may use non-application layer protocols for C2"),
            ("T1568", "Dynamic Resolution", MITRETactic.COMMAND_AND_CONTROL,
             "Adversaries may dynamically establish connections to C2 infrastructure"),
            
            # Exfiltration
            ("T1041", "Exfiltration Over C2 Channel", MITRETactic.EXFILTRATION,
             "Adversaries may steal data by exfiltrating it over existing C2 channel"),
            ("T1048", "Exfiltration Over Alternative Protocol", MITRETactic.EXFILTRATION,
             "Adversaries may steal data by exfiltrating it over different protocol"),
            ("T1567", "Exfiltration Over Web Service", MITRETactic.EXFILTRATION,
             "Adversaries may use external web services to exfiltrate data"),
            
            # Impact
            ("T1499", "Endpoint Denial of Service", MITRETactic.IMPACT,
             "Adversaries may perform Endpoint Denial of Service"),
            ("T1498", "Network Denial of Service", MITRETactic.IMPACT,
             "Adversaries may perform Network Denial of Service attacks"),
        ]
        
        for tech_id, name, tactic, desc in techniques_data:
            techniques[tech_id] = MITRETechnique(tech_id, name, tactic, desc)
        
        return techniques
    
    def map_threat(self, threat_category: str, indicators: List[str], 
                   source_ip: str, destination_ip: str, port: int) -> List[Dict]:
        """
        Map a threat to MITRE ATT&CK techniques
        
        Args:
            threat_category: Type of threat detected
            indicators: List of threat indicators
            source_ip: Source IP address
            destination_ip: Destination IP address
            port: Port number
            
        Returns:
            List of matched MITRE techniques with confidence scores
        """
        matched_techniques = []
        
        # Port scanning detection
        if threat_category == "suspicious_activity" and any("port" in ind.lower() for ind in indicators):
            matched_techniques.append({
                "technique_id": "T1046",
                "name": "Network Service Discovery",
                "tactic": MITRETactic.DISCOVERY,
                "confidence": 0.9,
                "description": "Port scanning detected - likely network reconnaissance",
                "evidence": [f"Sequential port access from {source_ip}"]
            })
        
        # DDoS detection
        if threat_category == "ddos":
            matched_techniques.append({
                "technique_id": "T1498",
                "name": "Network Denial of Service",
                "tactic": MITRETactic.IMPACT,
                "confidence": 0.95,
                "description": "DDoS attack pattern detected",
                "evidence": [f"High volume traffic from {source_ip}"]
            })
        
        # Brute force detection
        if "brute" in threat_category.lower() or port in [22, 3389, 21]:
            matched_techniques.append({
                "technique_id": "T1110",
                "name": "Brute Force",
                "tactic": MITRETactic.CREDENTIAL_ACCESS,
                "confidence": 0.85,
                "description": "Potential brute force attack on authentication service",
                "evidence": [f"Multiple connection attempts on port {port}"]
            })
        
        # Data exfiltration
        if threat_category == "data_breach" or "exfiltration" in str(indicators):
            matched_techniques.extend([
                {
                    "technique_id": "T1041",
                    "name": "Exfiltration Over C2 Channel",
                    "tactic": MITRETactic.EXFILTRATION,
                    "confidence": 0.8,
                    "description": "Potential data exfiltration detected",
                    "evidence": [f"Large data transfer to {destination_ip}"]
                },
                {
                    "technique_id": "T1567",
                    "name": "Exfiltration Over Web Service",
                    "tactic": MITRETactic.EXFILTRATION,
                    "confidence": 0.75,
                    "description": "Data transfer to external service",
                    "evidence": ["Unusual upload patterns"]
                }
            ])
        
        # Lateral movement
        if threat_category == "intrusion" and port in [445, 139, 3389]:
            matched_techniques.append({
                "technique_id": "T1021",
                "name": "Remote Services",
                "tactic": MITRETactic.LATERAL_MOVEMENT,
                "confidence": 0.85,
                "description": "Possible lateral movement via remote services",
                "evidence": [f"SMB/RDP access from {source_ip}"]
            })
        
        # C2 communication
        if "anomaly" in threat_category and any("unusual" in ind.lower() for ind in indicators):
            matched_techniques.append({
                "technique_id": "T1071",
                "name": "Application Layer Protocol",
                "tactic": MITRETactic.COMMAND_AND_CONTROL,
                "confidence": 0.7,
                "description": "Anomalous network behavior suggests C2 communication",
                "evidence": ["Non-standard protocol usage detected"]
            })
        
        return matched_techniques
    
    def get_kill_chain_stage(self, tactic: MITRETactic) -> Dict:
        """
        Map MITRE tactic to cyber kill chain stage
        
        Returns:
            Kill chain stage information
        """
        kill_chain_map = {
            MITRETactic.RECONNAISSANCE: {"stage": 1, "name": "Reconnaissance", "color": "#3498db"},
            MITRETactic.RESOURCE_DEVELOPMENT: {"stage": 2, "name": "Weaponization", "color": "#9b59b6"},
            MITRETactic.INITIAL_ACCESS: {"stage": 3, "name": "Delivery", "color": "#e74c3c"},
            MITRETactic.EXECUTION: {"stage": 4, "name": "Exploitation", "color": "#e67e22"},
            MITRETactic.PERSISTENCE: {"stage": 5, "name": "Installation", "color": "#f39c12"},
            MITRETactic.PRIVILEGE_ESCALATION: {"stage": 6, "name": "Privilege Escalation", "color": "#d35400"},
            MITRETactic.DEFENSE_EVASION: {"stage": 7, "name": "Defense Evasion", "color": "#c0392b"},
            MITRETactic.CREDENTIAL_ACCESS: {"stage": 8, "name": "Credential Access", "color": "#8e44ad"},
            MITRETactic.DISCOVERY: {"stage": 3, "name": "Discovery", "color": "#2980b9"},
            MITRETactic.LATERAL_MOVEMENT: {"stage": 9, "name": "Lateral Movement", "color": "#16a085"},
            MITRETactic.COLLECTION: {"stage": 10, "name": "Collection", "color": "#27ae60"},
            MITRETactic.COMMAND_AND_CONTROL: {"stage": 6, "name": "Command & Control", "color": "#2c3e50"},
            MITRETactic.EXFILTRATION: {"stage": 11, "name": "Exfiltration", "color": "#c0392b"},
            MITRETactic.IMPACT: {"stage": 12, "name": "Actions on Objectives", "color": "#e74c3c"},
        }
        
        return kill_chain_map.get(tactic, {"stage": 0, "name": "Unknown", "color": "#95a5a6"})
    
    def get_technique_details(self, technique_id: str) -> Optional[MITRETechnique]:
        """Get detailed information about a technique"""
        return self.techniques.get(technique_id)
    
    def get_techniques_by_tactic(self, tactic: MITRETactic) -> List[MITRETechnique]:
        """Get all techniques for a specific tactic"""
        return [tech for tech in self.techniques.values() if tech.tactic == tactic]


# Global MITRE mapper instance
mitre_mapper = MITREMapper()
