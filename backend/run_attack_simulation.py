"""
PCDS Enterprise - Realistic Attack Simulation
Uses ML Engine v3.0 for real-time threat scoring
"""

import sqlite3
import uuid
import random
from datetime import datetime, timedelta
import sys
sys.path.insert(0, '.')

# Import ML Engine v3.0
from ml.advanced_detector import get_advanced_engine

print("=" * 70)
print("🎯 PCDS ENTERPRISE - REALISTIC ATTACK SIMULATION")
print("=" * 70)

# Initialize
engine = get_advanced_engine()
conn = sqlite3.connect('pcds_enterprise.db')

# === ENTITY DEFINITIONS ===
ENTITIES = [
    {"id": "ws-001", "identifier": "WORKSTATION-CEO", "type": "host", "name": "CEO Workstation"},
    {"id": "ws-002", "identifier": "WORKSTATION-CFO", "type": "host", "name": "CFO Workstation"},
    {"id": "ws-003", "identifier": "WORKSTATION-DEV01", "type": "host", "name": "Developer Workstation"},
    {"id": "ws-004", "identifier": "WORKSTATION-DEV02", "type": "host", "name": "Developer Workstation 2"},
    {"id": "ws-005", "identifier": "WORKSTATION-HR", "type": "host", "name": "HR Department PC"},
    {"id": "srv-001", "identifier": "DC01", "type": "host", "name": "Domain Controller"},
    {"id": "srv-002", "identifier": "FILESERVER01", "type": "host", "name": "File Server"},
    {"id": "srv-003", "identifier": "SQLSERVER01", "type": "host", "name": "Database Server"},
    {"id": "srv-004", "identifier": "WEBSERVER01", "type": "host", "name": "Web Server"},
    {"id": "user-001", "identifier": "john.smith", "type": "user", "name": "John Smith CEO"},
    {"id": "user-002", "identifier": "jane.doe", "type": "user", "name": "Jane Doe CFO"},
    {"id": "user-003", "identifier": "admin.backup", "type": "user", "name": "Backup Admin"},
    {"id": "ip-001", "identifier": "185.220.101.45", "type": "ip", "name": "TOR Exit Node"},
    {"id": "ip-002", "identifier": "45.133.1.100", "type": "ip", "name": "Known C2 Server"},
]

# === ATTACK DETECTIONS ===
ATTACKS = [
    # APT Lateral Movement
    {"type": "phishing_email", "severity": "medium", "entity": "ws-005", "technique": "T1566", "desc": "Suspicious email with macro attachment"},
    {"type": "macro_execution", "severity": "high", "entity": "ws-005", "technique": "T1059", "desc": "Malicious macro executed in Word"},
    {"type": "credential_dump", "severity": "critical", "entity": "ws-005", "technique": "T1003", "desc": "Mimikatz detected - credential extraction"},
    {"type": "lateral_movement", "severity": "critical", "entity": "srv-001", "technique": "T1021", "desc": "SMB lateral movement to DC"},
    {"type": "privilege_escalation", "severity": "critical", "entity": "srv-001", "technique": "T1078", "desc": "Domain Admin compromised"},
    
    # Ransomware
    {"type": "exploit_attempt", "severity": "high", "entity": "srv-004", "technique": "T1190", "desc": "Citrix Bleed exploit CVE-2023-4966"},
    {"type": "webshell", "severity": "critical", "entity": "srv-004", "technique": "T1505", "desc": "WebShell uploaded cmd.aspx"},
    {"type": "ransomware", "severity": "critical", "entity": "srv-002", "technique": "T1486", "desc": "LockBit encryption detected"},
    {"type": "shadow_delete", "severity": "critical", "entity": "srv-002", "technique": "T1490", "desc": "Volume shadow copies deleted"},
    
    # Data Exfil
    {"type": "dns_tunneling", "severity": "high", "entity": "ws-003", "technique": "T1048", "desc": "DNS tunneling data exfiltration"},
    {"type": "large_transfer", "severity": "critical", "entity": "ws-003", "technique": "T1041", "desc": "10GB data transferred via DNS"},
    
    # Brute Force
    {"type": "brute_force", "severity": "high", "entity": "srv-003", "technique": "T1110", "desc": "500+ failed RDP login attempts"},
    {"type": "successful_login", "severity": "critical", "entity": "srv-003", "technique": "T1078", "desc": "Brute force succeeded - access gained"},
    
    # Supply Chain
    {"type": "malicious_update", "severity": "high", "entity": "ws-001", "technique": "T1195", "desc": "Trojanized software update"},
    {"type": "c2_beacon", "severity": "critical", "entity": "ws-001", "technique": "T1071", "desc": "C2 beacon to 45.133.1.100"},
    {"type": "keylogger", "severity": "critical", "entity": "ws-001", "technique": "T1056", "desc": "Keylogger on CEO workstation"},
    
    # Insider Threat
    {"type": "unusual_access", "severity": "medium", "entity": "user-003", "technique": "T1078", "desc": "After-hours access to sensitive files"},
    {"type": "mass_download", "severity": "high", "entity": "srv-002", "technique": "T1039", "desc": "500+ files downloaded"},
    {"type": "usb_exfil", "severity": "critical", "entity": "ws-004", "technique": "T1052", "desc": "USB data exfiltration detected"},
]

# === CREATE ENTITIES ===
print("\n📊 Creating entities...")
now = datetime.now()
for entity in ENTITIES:
    try:
        conn.execute("""
            INSERT INTO entities (id, identifier, entity_type, display_name, urgency_level, urgency_score, 
                                 threat_score, total_detections, first_seen, last_seen)
            VALUES (?, ?, ?, ?, 'low', 10, 20, 0, ?, ?)
        """, (entity["id"], entity["identifier"], entity["type"], entity["name"], 
              now.isoformat(), now.isoformat()))
    except Exception as e:
        print(f"   ⚠️ Entity {entity['id']}: {e}")
conn.commit()
print(f"   ✅ Created {len(ENTITIES)} entities")

# === CREATE DETECTIONS ===
print("\n🔴 Running attack simulations...")
detection_count = 0

for attack in ATTACKS:
    # ML analysis
    data = {
        "detection_type": attack["type"],
        "source_ip": random.choice(["185.220.101.45", "45.133.1.100", "192.168.1.100"]),
        "dest_ip": "10.0.0." + str(random.randint(1, 254)),
        "port": random.choice([445, 3389, 443, 80, 53, 4444]),
        "bytes_out": random.randint(1000, 50000000),
    }
    
    ml_result = engine.detect(data=data, entity_id=attack["entity"], attack_type=attack["type"])
    
    # Create detection
    det_id = "det_" + uuid.uuid4().hex[:12]
    timestamp = (now - timedelta(hours=random.randint(0, 24))).isoformat()
    
    try:
        conn.execute("""
            INSERT INTO detections (id, entity_id, detection_type, title, severity, technique_id, 
                                   description, source_ip, destination_ip, confidence_score, 
                                   risk_score, detected_at, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new')
        """, (det_id, attack["entity"], attack["type"], attack["desc"], ml_result["risk_level"], 
              attack["technique"], attack["desc"], data["source_ip"], data["dest_ip"], 
              ml_result["confidence"], ml_result["anomaly_score"] * 100, timestamp))
        
        # Update entity
        conn.execute("""
            UPDATE entities SET urgency_level = ?, total_detections = total_detections + 1, last_seen = ?
            WHERE id = ?
        """, (ml_result["risk_level"], timestamp, attack["entity"]))
        
        detection_count += 1
        icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
        print(f"   {icons.get(ml_result['risk_level'], '⚪')} {attack['type']}: score={ml_result['anomaly_score']:.2f} → {ml_result['risk_level'].upper()}")
    except Exception as e:
        print(f"   ⚠️ Detection error: {e}")

conn.commit()

# === SUMMARY ===
print("\n" + "=" * 70)
print("✅ ATTACK SIMULATION COMPLETE")
print("=" * 70)
print(f"\n📊 Results:")
print(f"   • Entities: {len(ENTITIES)}")
print(f"   • Detections: {detection_count}")
print(f"   • ML Engine: v{engine.VERSION}")

c = conn.cursor()
c.execute("SELECT severity, COUNT(*) FROM detections GROUP BY severity")
for row in c.fetchall():
    icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}
    print(f"   {icons.get(row[0], '⚪')} {row[0].upper()}: {row[1]}")

conn.close()
print("\n🚀 Refresh the PCDS dashboard to see the new data!")
print("=" * 70)
