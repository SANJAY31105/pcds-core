"""
PCDS Enterprise - Comprehensive Demo Data Generator
Populates all pages with realistic enterprise security data
"""

import sqlite3
import random
import uuid
from datetime import datetime, timedelta

DB_PATH = "pcds_enterprise.db"

# Realistic entity names
ENTITIES = [
    ("192.168.1.10", "host", "Domain Controller DC01"),
    ("192.168.1.20", "host", "Exchange Server MAIL01"),
    ("192.168.1.50", "host", "Faculty Workstation WS-FAC-01"),
    ("192.168.1.51", "host", "Admin Workstation WS-ADM-01"),
    ("192.168.1.100", "host", "Student Lab PC LAB-01"),
    ("192.168.1.101", "host", "Student Lab PC LAB-02"),
    ("192.168.1.150", "host", "Finance PC FIN-01"),
    ("10.0.0.5", "host", "Web Server WWW01"),
    ("10.0.0.10", "host", "Database Server DB01"),
    ("faculty.john@college.edu", "user", "Dr. John Smith"),
    ("admin.sarah@college.edu", "user", "Sarah Admin"),
    ("student.alex@college.edu", "user", "Alex Student"),
]

# Detection templates
DETECTIONS = [
    {
        "type": "suspicious_login",
        "title": "Suspicious Login Activity",
        "description": "Multiple failed login attempts followed by success from unusual location",
        "severity": "high",
        "technique_id": "T1078",
    },
    {
        "type": "malware",
        "title": "Malware Signature Detected",
        "description": "Known malware hash identified in downloaded file",
        "severity": "critical",
        "technique_id": "T1204",
    },
    {
        "type": "data_exfil",
        "title": "Large Data Transfer Detected",
        "description": "Unusual outbound transfer of 2.5GB to external IP",
        "severity": "critical",
        "technique_id": "T1041",
    },
    {
        "type": "lateral_movement",
        "title": "Lateral Movement Detected",
        "description": "PsExec-like activity between workstations",
        "severity": "high",
        "technique_id": "T1021",
    },
    {
        "type": "privilege_escalation",
        "title": "Privilege Escalation Attempt",
        "description": "Process attempting to elevate to SYSTEM privileges",
        "severity": "critical",
        "technique_id": "T1548",
    },
    {
        "type": "c2_beacon",
        "title": "C2 Communication Pattern",
        "description": "Regular beacon pattern to known C2 infrastructure",
        "severity": "critical",
        "technique_id": "T1071",
    },
    {
        "type": "phishing",
        "title": "Phishing Link Clicked",
        "description": "User clicked link in suspicious email from external sender",
        "severity": "medium",
        "technique_id": "T1566",
    },
    {
        "type": "bruteforce",
        "title": "Brute Force Attack",
        "description": "1000+ failed authentication attempts in 5 minutes",
        "severity": "high",
        "technique_id": "T1110",
    },
    {
        "type": "credential_dump",
        "title": "Credential Dumping",
        "description": "LSASS memory access detected",
        "severity": "critical",
        "technique_id": "T1003",
    },
    {
        "type": "ransomware",
        "title": "Ransomware Activity",
        "description": "Mass file encryption behavior detected",
        "severity": "critical",
        "technique_id": "T1486",
    }
]


def generate_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("🗑️  Cleaning existing demo data...")
    cursor.execute("DELETE FROM detections")
    cursor.execute("DELETE FROM entities")
    conn.commit()
    
    # ============ ENTITIES ============
    print("👥 Creating entities...")
    entity_ids = {}
    
    for identifier, etype, display_name in ENTITIES:
        entity_id = f"ent_{uuid.uuid4().hex[:8]}"
        entity_ids[identifier] = entity_id
        urgency_level = random.choice(['low', 'low', 'medium', 'medium', 'high', 'critical'])
        urgency_score = random.randint(20, 95)
        threat_score = random.randint(15, 85)
        total_detections = random.randint(0, 15)
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO entities (
                id, entity_type, identifier, display_name, 
                urgency_score, urgency_level, threat_score, 
                total_detections, first_seen, last_seen
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entity_id, etype, identifier, display_name, 
            urgency_score, urgency_level, threat_score,
            total_detections, now, now
        ))
    
    conn.commit()
    print(f"   ✅ Created {len(ENTITIES)} entities")
    
    # ============ DETECTIONS ============
    print("🔔 Creating detections...")
    detection_count = 0
    
    for i in range(30):
        det = random.choice(DETECTIONS)
        entity_key = random.choice(list(entity_ids.keys()))
        entity_id = entity_ids[entity_key]
        
        detection_id = f"det_{uuid.uuid4().hex[:12]}"
        timestamp = datetime.now() - timedelta(hours=random.randint(1, 72))
        confidence = round(random.uniform(0.75, 0.99), 2)
        risk_score = random.randint(40, 95)
        
        # Source and destination IPs
        source_ip = entity_key if "." in entity_key else "192.168.1." + str(random.randint(10, 200))
        dest_ip = f"{random.randint(10, 200)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        
        cursor.execute("""
            INSERT INTO detections (
                id, detection_type, title, description, severity,
                confidence_score, risk_score, entity_id,
                source_ip, destination_ip, technique_id,
                status, detected_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            detection_id, det['type'], det['title'], det['description'],
            det['severity'], confidence, risk_score, entity_id,
            source_ip, dest_ip, det['technique_id'],
            'new', timestamp.isoformat()
        ))
        detection_count += 1
    
    conn.commit()
    print(f"   ✅ Created {detection_count} detections")
    
    # ============ ATTACK CAMPAIGNS ============
    print("🎯 Creating attack campaigns...")
    campaigns = [
        ("Ransomware Campaign - BlackCat", "active", "critical", 5, 3),
        ("APT29 Suspected Activity", "active", "critical", 8, 4),
        ("Credential Harvesting", "contained", "high", 3, 2),
        ("Cryptominer Infection", "resolved", "medium", 2, 1),
    ]
    
    try:
        for name, status, severity, det_count, ent_count in campaigns:
            campaign_id = f"camp_{uuid.uuid4().hex[:8]}"
            now = datetime.now()
            cursor.execute("""
                INSERT INTO attack_campaigns (
                    id, name, description, severity, total_detections,
                    affected_entities, started_at, last_activity, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                campaign_id, name, f"Active {name.lower()} campaign targeting college network",
                severity, det_count, ent_count,
                (now - timedelta(days=random.randint(1, 5))).isoformat(),
                now.isoformat(), status
            ))
        conn.commit()
        print(f"   ✅ Created {len(campaigns)} campaigns")
    except Exception as e:
        print(f"   ⚠️ Campaigns table issue: {e}")
    
    # ============ INVESTIGATIONS ============
    print("🔍 Creating investigations...")
    investigations = [
        ("INC-2025-001", "Ransomware Incident - Finance Dept", "in_progress", "critical"),
        ("INC-2025-002", "Data Exfiltration Alert Review", "in_progress", "high"),
        ("INC-2025-003", "Phishing Campaign Response", "closed", "medium"),
        ("INC-2025-004", "Unauthorized VPN Access", "open", "high"),
    ]
    
    try:
        for case_num, title, status, severity in investigations:
            inv_id = f"inv_{uuid.uuid4().hex[:8]}"
            now = datetime.now()
            cursor.execute("""
                INSERT INTO investigations (
                    id, title, description, severity, priority,
                    status, assigned_to, opened_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                inv_id, title, f"Investigation into {title.lower()}",
                severity, severity, status, "admin",
                (now - timedelta(days=random.randint(1, 3))).isoformat()
            ))
        conn.commit()
        print(f"   ✅ Created {len(investigations)} investigations")
    except Exception as e:
        print(f"   ⚠️ Investigations table issue: {e}")
    
    # ============ HUNT QUERIES ============
    print("🔎 Creating hunt queries...")
    hunt_queries = [
        ("Ransomware Precursors", "behavior", "Find ransomware behavior patterns"),
        ("Lateral Movement Hunt", "network", "Detect internal credential abuse"),
        ("C2 Beacon Detection", "network", "Find command and control beacons"),
    ]
    
    try:
        for name, qtype, desc in hunt_queries:
            query_id = f"hunt_{uuid.uuid4().hex[:8]}"
            cursor.execute("""
                INSERT INTO hunt_queries (
                    id, name, description, query_type, created_by, is_public
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (query_id, name, desc, qtype, "admin", True))
        conn.commit()
        print(f"   ✅ Created {len(hunt_queries)} hunt queries")
    except Exception as e:
        print(f"   ⚠️ Hunt queries table issue: {e}")
    
    conn.close()
    
    print("\n" + "="*60)
    print("✅ DEMO DATA GENERATED SUCCESSFULLY!")
    print("="*60)
    print("\n📊 Summary:")
    print(f"   • 12 Entities (hosts + users)")
    print(f"   • 30 Detections (various threat types)")
    print(f"   • 4 Attack Campaigns")
    print(f"   • 4 Investigations")
    print(f"   • 3 Hunt Queries")
    print("\n🌐 Refresh your browser to see the data!")
    print("   Dashboard: http://localhost:3000")
    print("   Entities:  http://localhost:3000/entities")
    print("   Detections: http://localhost:3000/detections")


if __name__ == "__main__":
    generate_data()
