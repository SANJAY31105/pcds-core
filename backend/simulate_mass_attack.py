"""
PCDS Enterprise - Mass Attack Simulation
Generates 10,000 realistic attack scenarios for stress testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import db_manager
import uuid
import random
from datetime import datetime, timedelta

class MassAttackSimulator:
    """Simulates 10,000 realistic enterprise attacks"""
    
    def __init__(self):
        self.attack_campaigns = []
        self.entities = []
        self.detections_created = 0
        
    def run_simulation(self, total_attacks=10000):
        """Run complete attack simulation"""
        print("\n" + "="*80)
        print("🚨 PCDS ENTERPRISE - MASS ATTACK SIMULATION")
        print(f"Generating {total_attacks:,} realistic attack scenarios...")
        print("="*80 + "\n")
        
        # Step 1: Create entities (attackers & targets)
        print("[1/5] Creating entities...")
        self.create_entities()
        
        # Step 2: Create attack campaigns
        print("[2/5] Generating attack campaigns...")
        self.create_attack_campaigns()
        
        # Step 3: Generate detections
        print("[3/5] Simulating attacks...")
        self.generate_detections(total_attacks)
        
        # Step 4: Update entity scores
        print("[4/5] Calculating entity risk scores...")
        self.update_entity_scores()
        
        # Step 5: Generate report
        print("[5/5] Generating attack report...")
        self.generate_report()
        
    def create_entities(self):
        """Create simulated entities"""
        entity_types = [
            # Internal entities (targets)
            ("192.168.1.10", "device", "Finance-WS-01"),
            ("192.168.1.25", "device", "HR-WS-15"),
            ("192.168.1.50", "device", "IT-Server-01"),
            ("192.168.1.100", "device", "CEO-Laptop"),
            ("john.doe@company.com", "user", "John Doe"),
            ("admin@company.com", "user", "Admin User"),
            ("service_account", "user", "SQL Service"),
            
            # External attackers
            ("203.0.113.45", "ip", "APT29 C2"),
            ("198.51.100.89", "ip", "Ransomware Group"),
            ("104.16.249.50", "ip", "Phishing Server"),
        ]
        
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        for identifier, entity_type, display_name in entity_types:
            entity_id = str(uuid.uuid4())
            
            db_manager.execute_query("""
                INSERT OR IGNORE INTO entities 
                (id, identifier, entity_type, threat_score, urgency_level, 
                 total_detections, first_seen, last_seen, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (entity_id, identifier, entity_type, 0, 'low', 0, now, now, now, now))
            
            self.entities.append({
                'id': entity_id,
                'identifier': identifier,
                'type': entity_type,
                'name': display_name
            })
        
        print(f"   ✅ Created {len(self.entities)} entities")
    
    def create_attack_campaigns(self):
        """Create realistic attack campaigns"""
        campaigns = [
            {
                "name": "APT29 - State-Sponsored Espionage",
                "severity": "critical",
                "techniques": ["T1566", "T1059", "T1003", "T1021", "T1041"],
                "weight": 0.05,  # 5% of attacks
            },
            {
                "name": "Ransomware - Lockbit 3.0",
                "severity": "critical",
                "techniques": ["T1566", "T1059", "T1486", "T1490", "T1489"],
                "weight": 0.10,  # 10%
            },
            {
                "name": "Insider Threat - Data Exfiltration",
                "severity": "high",
                "techniques": ["T1078", "T1083", "T1567", "T1041"],
                "weight": 0.08,  # 8%
            },
            {
                "name": "Credential Stuffing Campaign",
                "severity": "high",
                "techniques": ["T1110", "T1078", "T1021"],
                "weight": 0.15,  # 15%
            },
            {
                "name": "Crypto Mining Operation",
                "severity": "medium",
                "techniques": ["T1496", "T1053", "T1059"],
                "weight": 0.12,  # 12%
            },
            {
                "name": "Phishing & Malware Distribution",
                "severity": "high",
                "techniques": ["T1566", "T1204", "T1059", "T1055"],
                "weight": 0.20,  # 20%
            },
            {
                "name": "Web Application Exploitation",
                "severity": "medium",
                "techniques": ["T1190", "T1505", "T1059"],
                "weight": 0.15,  # 15%
            },
            {
                "name": "Reconnaissance & Scanning",
                "severity": "low",
                "techniques": ["T1046", "T1087", "T1083", "T1018"],
                "weight": 0.15,  # 15%
            }
        ]
        
        now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        for campaign in campaigns:
            campaign_id = str(uuid.uuid4())
            
            db_manager.execute_query("""
                INSERT INTO attack_campaigns 
                (id, name, status, severity, started_at, last_activity, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (campaign_id, campaign['name'], 'active', campaign['severity'], 
                  now, now, now, now))
            
            campaign['id'] = campaign_id
            self.attack_campaigns.append(campaign)
        
        print(f"   ✅ Created {len(self.attack_campaigns)} attack campaigns")
    
    def generate_detections(self, total_attacks):
        """Generate mass detections"""
        print(f"   Generating {total_attacks:,} detections...")
        
        detection_templates = [
            # Credential Access
            ("Credential Dumping Detected", "credential_dumping", "T1003", "critical"),
            ("Brute Force Attack", "brute_force", "T1110", "high"),
            ("Kerberoasting Detected", "kerberoasting", "T1558", "high"),
            
            # Lateral Movement
            ("PsExec Lateral Movement", "psexec", "T1021", "high"),
            ("RDP Lateral Movement", "rdp_lateral", "T1021", "medium"),
            ("Pass-the-Hash Attack", "pass_the_hash", "T1550", "critical"),
            
            # Execution
            ("Suspicious PowerShell Execution", "powershell_execution", "T1059", "high"),
            ("Malicious Macro Execution", "macro_execution", "T1204", "medium"),
            
            # Persistence
            ("Scheduled Task Created", "scheduled_task", "T1053", "medium"),
            ("Registry Modification", "registry_mod", "T1547", "medium"),
            
            # Exfiltration
            ("Large Data Upload", "large_upload", "T1567", "critical"),
            ("DNS Exfiltration", "dns_exfiltration", "T1048", "high"),
            
            # C2 Communication
            ("C2 Beaconing Detected", "c2_beaconing", "T1071", "critical"),
            ("Proxy Usage Detected", "proxy_usage", "T1090", "medium"),
            
            # Impact
            ("Ransomware File Encryption", "ransomware", "T1486", "critical"),
            ("Data Destruction", "data_destruction", "T1485", "critical"),
            
            # Discovery
            ("Network Scanning", "network_scan", "T1046", "low"),
            ("Account Enumeration", "account_enum", "T1087", "low"),
        ]
        
        batch_size = 1000
        for batch in range(0, total_attacks, batch_size):
            batch_detections = []
            
            for i in range(batch, min(batch + batch_size, total_attacks)):
                # Select campaign based on weights
                campaign = random.choices(
                    self.attack_campaigns,
                    weights=[c['weight'] for c in self.attack_campaigns]
                )[0]
                
                # Select detection from campaign techniques
                template = random.choice(detection_templates)
                title, det_type, technique, severity = template
                
                # Random entity (target)
                target_entity = random.choice([e for e in self.entities if e['type'] in ['device', 'user']])
                attacker_entity = random.choice([e for e in self.entities if e['type'] == 'ip'])
                
                # Random timestamp in last 24 hours
                hours_ago = random.randint(0, 24)
                minutes_ago = random.randint(0, 59)
                timestamp = (datetime.utcnow() - timedelta(hours=hours_ago, minutes=minutes_ago)).strftime('%Y-%m-%d %H:%M:%S')
                
                detection_id = str(uuid.uuid4())
                
                batch_detections.append((
                    detection_id,
                    det_type,
                    f"{title} - {target_entity['name']}",
                    f"Detected {det_type} targeting {target_entity['identifier']}",
                    severity,
                    round(random.uniform(0.7, 0.99), 2),  # confidence
                    random.randint(50, 95),  # risk_score
                    target_entity['id'],
                    attacker_entity['identifier'],
                    target_entity['identifier'],
                    random.choice([80, 443, 445, 3389, 22]),
                    technique,
                    timestamp,
                    timestamp,
                    timestamp
                ))
            
            # Batch insert
            db_manager.execute_many("""
                INSERT INTO detections 
                (id, detection_type, title, description, severity, confidence_score, risk_score,
                 entity_id, source_ip, destination_ip, destination_port, technique_id,
                 detected_at, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch_detections)
            
            self.detections_created += len(batch_detections)
            progress = (self.detections_created / total_attacks) * 100
            print(f"   [{progress:5.1f}%] {self.detections_created:,} / {total_attacks:,} attacks generated", end='\r')
        
        print(f"\n   ✅ Generated {self.detections_created:,} detections")
    
    def update_entity_scores(self):
        """Update entity risk scores based on detections"""
        
        # Update detection counts and scores
        db_manager.execute_update("""
            UPDATE entities SET
                total_detections = (
                    SELECT COUNT(*) FROM detections WHERE entity_id = entities.id
                ),
                threat_score = (
                    SELECT COALESCE(AVG(risk_score), 0) FROM detections WHERE entity_id = entities.id
                ),
                urgency_level = CASE
                    WHEN (SELECT AVG(risk_score) FROM detections WHERE entity_id = entities.id) >= 80 THEN 'critical'
                    WHEN (SELECT AVG(risk_score) FROM detections WHERE entity_id = entities.id) >= 60 THEN 'high'
                    WHEN (SELECT AVG(risk_score) FROM detections WHERE entity_id = entities.id) >= 40 THEN 'medium'
                    ELSE 'low'
                END
        """)
        
        print("   ✅ Updated entity risk scores")
    
    def generate_report(self):
        """Generate attack simulation report"""
        print("\n" + "="*80)
        print("📊 ATTACK SIMULATION REPORT")
        print("="*80 + "\n")
        
        # Total detections
        total = db_manager.execute_one("SELECT COUNT(*) as count FROM detections")
        print(f"📍 Total Attacks Simulated: {total['count']:,}")
        
        # By severity
        severity_stats = db_manager.execute_query("""
            SELECT severity, COUNT(*) as count 
            FROM detections 
            GROUP BY severity 
            ORDER BY CASE severity 
                WHEN 'critical' THEN 1 
                WHEN 'high' THEN 2 
                WHEN 'medium' THEN 3 
                ELSE 4 END
        """)
        
        print("\n🔴 Severity Breakdown:")
        for stat in severity_stats:
            emoji = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "📌"}.get(stat['severity'], "•")
            pct = (stat['count'] / total['count']) * 100
            print(f"   {emoji} {stat['severity'].upper():8} : {stat['count']:,} ({pct:.1f}%)")
        
        # By technique
        technique_stats = db_manager.execute_query("""
            SELECT technique_id, COUNT(*) as count 
            FROM detections 
            WHERE technique_id IS NOT NULL
            GROUP BY technique_id 
            ORDER BY count DESC 
            LIMIT 10
        """)
        
        print("\n🎯 Top 10 MITRE Techniques:")
        for stat in technique_stats:
            print(f"   {stat['technique_id']}: {stat['count']:,} detections")
        
        # Entity stats
        entity_stats = db_manager.execute_query("""
            SELECT identifier, total_detections, threat_score, urgency_level
            FROM entities 
            WHERE total_detections > 0
            ORDER BY threat_score DESC 
            LIMIT 10
        """)
        
        print("\n🎯 Top 10 Targeted Entities:")
        for entity in entity_stats:
            emoji = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "📌"}.get(entity['urgency_level'], "•")
            print(f"   {emoji} {entity['identifier']:25} | Score: {entity['threat_score']:3.0f} | Detections: {entity['total_detections']:,}")
        
        # Campaign stats
        campaign_stats = db_manager.execute_query("""
            SELECT name, severity 
            FROM attack_campaigns 
            WHERE status = 'active'
        """)
        
        print(f"\n📋 Active Campaigns: {len(campaign_stats)}")
        for campaign in campaign_stats:
            emoji = {"critical": "🚨", "high": "⚠️", "medium": "ℹ️", "low": "📌"}.get(campaign['severity'], "•")
            print(f"   {emoji} {campaign['name']}")
        
        print("\n" + "="*80)
        print("✅ SIMULATION COMPLETE")
        print("="*80)
        print(f"\n📊 Check your dashboard: http://localhost:3000")
        print(f"🔍 View detections: http://localhost:3000/detections")
        print(f"🎯 Check entities: http://localhost:3000/entities\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='PCDS Mass Attack Simulator')
    parser.add_argument('--attacks', type=int, default=10000, help='Number of attacks to simulate (default: 10000)')
    parser.add_argument('--clear', action='store_true', help='Clear existing detections first')
    
    args = parser.parse_args()
    
    if args.clear:
        print("🗑️  Clearing existing detections...")
        db_manager.execute_update("DELETE FROM detections")
        db_manager.execute_update("UPDATE entities SET total_detections = 0, threat_score = 0")
        print("   ✅ Cleared\n")
    
    simulator = MassAttackSimulator()
    simulator.run_simulation(args.attacks)
