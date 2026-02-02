"""
Download official MITRE ATT&CK Enterprise techniques and format for PCDS
"""
import json
import requests
from typing import Dict, List

# MITRE ATT&CK STIX data URL
MITRE_URL = "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"

def get_severity(technique: dict) -> str:
    """Determine severity based on tactic"""
    tactics = technique.get("kill_chain_phases", [])
    tactic_names = [t.get("phase_name", "") for t in tactics]
    
    # Critical tactics
    if any(t in tactic_names for t in ["impact", "exfiltration", "credential-access"]):
        return "critical"
    # High severity
    if any(t in tactic_names for t in ["initial-access", "execution", "privilege-escalation", "defense-evasion", "lateral-movement"]):
        return "high"
    # Medium
    if any(t in tactic_names for t in ["persistence", "command-and-control", "collection"]):
        return "medium"
    return "low"

def get_tactic_id(phase_name: str) -> str:
    """Map phase name to tactic ID"""
    mapping = {
        "initial-access": "TA0001",
        "execution": "TA0002",
        "persistence": "TA0003",
        "privilege-escalation": "TA0004",
        "defense-evasion": "TA0005",
        "credential-access": "TA0006",
        "discovery": "TA0007",
        "lateral-movement": "TA0008",
        "collection": "TA0009",
        "command-and-control": "TA0011",
        "exfiltration": "TA0010",
        "impact": "TA0040",
        "resource-development": "TA0042",
        "reconnaissance": "TA0043",
    }
    return mapping.get(phase_name, "TA0001")

def download_mitre_techniques() -> List[Dict]:
    """Download and parse MITRE ATT&CK techniques"""
    print("Downloading MITRE ATT&CK data...")
    response = requests.get(MITRE_URL, timeout=60)
    data = response.json()
    
    techniques = []
    
    for obj in data.get("objects", []):
        # Only process attack-pattern (techniques)
        if obj.get("type") != "attack-pattern":
            continue
        
        # Skip revoked/deprecated
        if obj.get("revoked") or obj.get("x_mitre_deprecated"):
            continue
        
        # Get technique ID (e.g., T1078)
        external_refs = obj.get("external_references", [])
        technique_id = None
        for ref in external_refs:
            if ref.get("source_name") == "mitre-attack":
                technique_id = ref.get("external_id")
                break
        
        if not technique_id or not technique_id.startswith("T"):
            continue
        
        # Skip sub-techniques for now (e.g., T1078.001)
        if "." in technique_id:
            continue
        
        # Get tactics
        kill_chain = obj.get("kill_chain_phases", [])
        tactics = [kc.get("phase_name") for kc in kill_chain if kc.get("kill_chain_name") == "mitre-attack"]
        tactic_id = get_tactic_id(tactics[0]) if tactics else "TA0001"
        
        # Get platforms
        platforms = obj.get("x_mitre_platforms", ["Windows", "Linux", "macOS"])
        
        # Get data sources
        data_sources = obj.get("x_mitre_data_sources", [])
        if isinstance(data_sources, list) and data_sources:
            data_sources = data_sources[:5]  # Limit to 5
        else:
            data_sources = ["Process monitoring", "Network traffic"]
        
        # Build technique entry
        technique = {
            "id": technique_id,
            "name": obj.get("name", "Unknown"),
            "tactic_id": tactic_id,
            "severity": get_severity(obj),
            "description": (obj.get("description", "")[:200] + "...") if len(obj.get("description", "")) > 200 else obj.get("description", ""),
            "platforms": platforms[:5],
            "data_sources": data_sources,
            "detection_methods": [
                f"Monitor for {obj.get('name', 'technique')} activity",
                "Analyze behavioral patterns",
                "Correlate with threat intelligence"
            ],
            "mitigations": [
                "Network segmentation",
                "Least privilege access",
                "Security monitoring"
            ]
        }
        
        techniques.append(technique)
    
    # Sort by technique ID
    techniques.sort(key=lambda x: x["id"])
    
    print(f"Downloaded {len(techniques)} techniques")
    return techniques

def main():
    techniques = download_mitre_techniques()
    
    output = {
        "version": "15.0",
        "last_updated": "2026-02-02",
        "source": "MITRE ATT&CK Enterprise Matrix",
        "techniques": techniques
    }
    
    # Save to file
    output_path = "c:/Users/sanja/OneDrive/Desktop/pcds-core/backend/data/mitre_attack_full.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    
    print(f"Saved {len(techniques)} techniques to {output_path}")

if __name__ == "__main__":
    main()
