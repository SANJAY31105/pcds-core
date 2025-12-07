"""
PCDS Enterprise - Live Attack Simulation for Demo
Simulates a multi-stage ransomware attack with real-time detections
"""

import asyncio
import aiohttp
import random
from datetime import datetime

API_BASE = "http://localhost:8000/api/v2"

# Simulated attack stages (based on real attack chain)
ATTACK_STAGES = [
    {
        "delay": 2,
        "detection_type": "phishing",
        "title": "Suspicious Email Link Clicked",
        "description": "User clicked malicious link in phishing email from external sender",
        "source_ip": "192.168.1.50",
        "dest_ip": "185.220.101.45",
        "severity": "medium",
        "technique_id": "T1566.002",
        "technique_name": "Spearphishing Link"
    },
    {
        "delay": 3,
        "detection_type": "malware_download",
        "title": "Malware Payload Downloaded",
        "description": "Executable file downloaded from known malicious domain",
        "source_ip": "185.220.101.45",
        "dest_ip": "192.168.1.50",
        "severity": "high",
        "technique_id": "T1204.002",
        "technique_name": "Malicious File"
    },
    {
        "delay": 2,
        "detection_type": "process_injection",
        "title": "Process Injection Detected",
        "description": "Suspicious code injection into svchost.exe from unknown process",
        "source_ip": "192.168.1.50",
        "dest_ip": "192.168.1.50",
        "severity": "critical",
        "technique_id": "T1055.001",
        "technique_name": "Dynamic-link Library Injection"
    },
    {
        "delay": 2,
        "detection_type": "credential_theft",
        "title": "Credential Dumping Attempt",
        "description": "LSASS memory access detected - possible credential theft",
        "source_ip": "192.168.1.50",
        "dest_ip": "192.168.1.1",
        "severity": "critical",
        "technique_id": "T1003.001",
        "technique_name": "LSASS Memory"
    },
    {
        "delay": 3,
        "detection_type": "lateral_movement",
        "title": "Lateral Movement to Domain Controller",
        "description": "PsExec connection to domain controller from compromised workstation",
        "source_ip": "192.168.1.50",
        "dest_ip": "192.168.1.10",
        "severity": "critical",
        "technique_id": "T1021.002",
        "technique_name": "SMB/Windows Admin Shares"
    },
    {
        "delay": 2,
        "detection_type": "c2_communication",
        "title": "Command & Control Beacon",
        "description": "Periodic HTTPS beaconing to known C2 infrastructure",
        "source_ip": "192.168.1.50",
        "dest_ip": "91.234.56.78",
        "severity": "critical",
        "technique_id": "T1071.001",
        "technique_name": "Web Protocols"
    },
    {
        "delay": 3,
        "detection_type": "ransomware",
        "title": "🚨 RANSOMWARE ENCRYPTION STARTED",
        "description": "Mass file encryption detected - .locked extension being added",
        "source_ip": "192.168.1.50",
        "dest_ip": "192.168.1.200",
        "severity": "critical",
        "technique_id": "T1486",
        "technique_name": "Data Encrypted for Impact"
    }
]

async def create_detection(session, stage):
    """Send detection to PCDS API"""
    detection = {
        "title": stage["title"],
        "description": stage["description"],
        "detection_type": stage["detection_type"],
        "entity_id": stage["source_ip"],  # Required field!
        "source_ip": stage["source_ip"],
        "destination_ip": stage["dest_ip"],
        "severity": stage["severity"],
        "confidence_score": random.uniform(0.85, 0.98),
        "raw_log": f"Simulated attack stage: {stage['detection_type']}",
        "mitre_technique_id": stage["technique_id"],
        "mitre_technique_name": stage["technique_name"]
    }
    
    try:
        async with session.post(f"{API_BASE}/detections/", json=detection) as resp:
            result = await resp.json()
            return result
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

async def evaluate_response(session, detection_id, stage):
    """Send to decision engine for auto/manual response"""
    payload = {
        "id": detection_id,
        "entity_id": stage["source_ip"],
        "detection_type": stage["detection_type"],
        "severity": stage["severity"],
        "confidence_score": 0.95,
        "technique_id": stage["technique_id"]
    }
    
    try:
        async with session.post(f"{API_BASE}/response/evaluate", json=payload) as resp:
            result = await resp.json()
            return result
    except Exception as e:
        return None

async def run_attack_simulation():
    """Execute the attack simulation"""
    print("\n" + "="*60)
    print("🎬 PCDS ENTERPRISE - LIVE ATTACK SIMULATION")
    print("="*60)
    print("\n🔴 Simulating multi-stage ransomware attack...")
    print("📺 Watch the dashboard at: http://localhost:3000/detections")
    print("\n" + "-"*60)
    
    async with aiohttp.ClientSession() as session:
        for i, stage in enumerate(ATTACK_STAGES, 1):
            print(f"\n⏳ Stage {i}/{len(ATTACK_STAGES)}: Waiting {stage['delay']}s...")
            await asyncio.sleep(stage['delay'])
            
            # Create detection
            print(f"🔔 [{stage['severity'].upper()}] {stage['title']}")
            print(f"   📍 {stage['source_ip']} → {stage['dest_ip']}")
            print(f"   🎯 MITRE: {stage['technique_id']} - {stage['technique_name']}")
            
            result = await create_detection(session, stage)
            
            if result and result.get('id'):
                detection_id = result['id']
                
                # Evaluate for auto-response
                if stage['severity'] == 'critical':
                    response = await evaluate_response(session, detection_id, stage)
                    if response:
                        if response.get('requires_approval'):
                            print(f"   ⏳ QUEUED FOR APPROVAL: {response.get('action')}")
                        elif response.get('auto_execute'):
                            print(f"   ✅ AUTO-EXECUTED: {response.get('action')}")
            
            print("-"*60)
    
    print("\n" + "="*60)
    print("🎬 ATTACK SIMULATION COMPLETE")
    print("="*60)
    print("\n📊 Check results:")
    print("   • Dashboard: http://localhost:3000")
    print("   • Detections: http://localhost:3000/detections")
    print("   • Approvals: http://localhost:3000/approvals")
    print("   • Timeline: http://localhost:3000/timeline")
    print("\n")

if __name__ == "__main__":
    asyncio.run(run_attack_simulation())
