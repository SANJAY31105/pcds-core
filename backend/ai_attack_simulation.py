"""
PCDS Enterprise - AI-Powered Attack Simulation
Simulates 200,000 attacks based on 2025 threat landscape
Including: AI phishing, deepfakes, ransomware 3.0, IoT botnets, supply chain
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import db_manager
import uuid
from datetime import datetime, timedelta
import random
import time

class MassiveAIAttackSimulator:
    """Simulate 200,000 AI-powered attacks"""
    
    def __init__(self, total_attacks=200000):
        self.total_attacks = total_attacks
        self.batch_size = 10000  # Process in batches for performance
        self.stats = {
            'total': 0,
            'by_category': {},
            'by_severity': {},
            'start_time': None,
            'end_time': None
        }
        
        # 2025 Threat Landscape Attack Patterns
        self.attack_categories = {
            'ai_phishing': {
                'weight': 35,
                'techniques': ['T1566.001', 'T1566.002', 'T1566.003'],
                'severities': ['high', 'critical'],
                'description': 'AI-generated spearphishing (1,265% increase)',
            },
            'deepfake_fraud': {
                'weight': 10,
                'techniques': ['T1566.004', 'T1598'],
                'severities': ['critical'],
                'description': 'Synthetic media impersonation attacks',
            },
            'ransomware_3_0': {
                'weight': 20,
                'techniques': ['T1486', 'T1490', 'T1489', 'T1567'],
                'severities': ['critical'],
                'description': 'Data exfiltration + encryption + extortion',
            },
            'supply_chain': {
                'weight': 8,
                'techniques': ['T1195.002', 'T1195.001'],
                'severities': ['critical', 'high'],
                'description': 'Third-party vendor compromise',
            },
            'iot_botnet': {
                'weight': 12,
                'techniques': ['T1071', 'T1573', 'T1498'],
                'severities': ['medium', 'high'],
                'description': 'IoT device compromise for DDoS',
            },
            'cloud_misconfiguration': {
                'weight': 15,
                'techniques': ['T1530', 'T1078.004'],
                'severities': ['high', 'critical'],
                'description': 'Cloud API abuse and data exposure',
            },
            'insider_threat': {
                'weight': 5,
                'techniques': ['T1078', 'T1083', 'T1567.002'],
                'severities': ['high', 'critical'],
                'description': 'Malicious employee data exfiltration',
            },
            'apt_espionage': {
                'weight': 10,
                'techniques': ['T1003', 'T1021', 'T1071.001'],
                'severities': ['critical'],
                'description': 'Nation-state credential dumping',
            },
            'polymorphic_malware': {
                'weight': 15,
                'techniques': ['T1059.001', 'T1140', 'T1027'],
                'severities': ['high', 'critical'],
                'description': 'AI-generated evolving malware',
            },
            'quantum_harvest': {
                'weight': 5,
                'techniques': ['T1040', 'T1557'],
                'severities': ['critical'],
                'description': 'Harvest Now Decrypt Later attacks',
            }
        }
        
        # Realistic IP pools
        self.external_ips = self._generate_ip_pool('external', 500)
        self.internal_ips = self._generate_ip_pool('internal', 200)
    
    def _generate_ip_pool(self, ip_type, count):
        """Generate realistic IP addresses"""
        ips = []
        if ip_type == 'external':
            # Known malicious IP ranges
            ranges = ['203.0.113', '198.51.100', '192.0.2', '185.220', '89.248']
            for _ in range(count):
                base = random.choice(ranges)
                ips.append(f"{base}.{random.randint(1,254)}")
        else:
            # Internal network
            subnets = ['192.168.1', '10.0.0', '172.16.0']
            for _ in range(count):
                base = random.choice(subnets)
                ips.append(f"{base}.{random.randint(1,254)}")
        return ips
    
    def generate_attack_batch(self, batch_num, batch_size):
        """Generate a batch of attacks"""
        attacks = []
        
        # Determine attack distribution for this batch
        for _ in range(batch_size):
            # Select attack category based on weights
            category = random.choices(
                list(self.attack_categories.keys()),
                weights=[c['weight'] for c in self.attack_categories.values()]
            )[0]
            
            attack_type = self.attack_categories[category]
            
            # Generate attack details
            technique = random.choice(attack_type['techniques'])
            severity = random.choice(attack_type['severities'])
            
            # Timestamp spread over last 7 days
            days_ago = random.randint(0, 7)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            timestamp = datetime.utcnow() - timedelta(
                days=days_ago, 
                hours=hours_ago, 
                minutes=minutes_ago
            )
            
            # Source/Destination IPs
            if category in ['insider_threat', 'cloud_misconfiguration']:
                src_ip = random.choice(self.internal_ips)
                dst_ip = random.choice(self.external_ips)
            else:
                src_ip = random.choice(self.external_ips)
                dst_ip = random.choice(self.internal_ips)
            
            # Generate confidence score (AI attacks are sophisticated)
            if category in ['ai_phishing', 'deepfake_fraud', 'polymorphic_malware']:
                confidence = random.uniform(0.85, 0.99)
            else:
                confidence = random.uniform(0.65, 0.95)
            
            # Risk score based on severity
            risk_map = {'low': 25, 'medium': 50, 'high': 75, 'critical': 95}
            risk_score = risk_map.get(severity, 50) + random.randint(-5, 5)
            
            attack = {
                'id': str(uuid.uuid4()),
                'category': category,
                'detection_type': category,
                'title': f"{attack_type['description']}",
                'description': self._generate_description(category, technique),
                'severity': severity,
                'confidence_score': round(confidence, 2),
                'risk_score': risk_score,
                'source_ip': src_ip,
                'destination_ip': dst_ip,
                'technique_id': technique,
                'detected_at': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'created_at': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'updated_at': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            }
            
            attacks.append(attack)
            
            # Update stats
            self.stats['by_category'][category] = self.stats['by_category'].get(category, 0) + 1
            self.stats['by_severity'][severity] = self.stats['by_severity'].get(severity, 0) + 1
        
        return attacks
    
    def _generate_description(self, category, technique):
        """Generate realistic attack descriptions"""
        descriptions = {
            'ai_phishing': [
                f"AI-generated credential harvesting email with perfect grammar targeting finance dept",
                f"LLM-crafted spearphishing using LinkedIn data and recent company news",
                f"WormGPT-generated business email compromise targeting wire transfer approval",
            ],
            'deepfake_fraud': [
                f"Video conference deepfake impersonating CFO for $25M+ wire transfer",
                f"Voice-cloned CEO authorization call detected bypassing MFA",
                f"Synthetic media used to manipulate employee into sharing credentials",
            ],
            'ransomware_3_0': [
                f"RaaS deployment with data exfiltration before encryption",
                f"Triple extortion: Encryption + leak threat + DDoS",
                f"LockBit 4.0 variant bypassing EDR through polymorphic code",
            ],
            'supply_chain': [
                f"Third-party vendor compromise enabling lateral network access",
                f"Malicious npm package in CI/CD pipeline injecting backdoor",
                f"Cloud service provider breach exposing customer credentials",
            ],
            'iot_botnet': [
                f"Mirai variant recruiting IoT cameras for 22Tbps DDoS capability",
                f"BadBox 2.0 firmware-level infection of Android TV devices",
                f"SOHO router compromise establishing C2 channel",
            ],
            'cloud_misconfiguration': [
                f"Publicly exposed S3 bucket containing 100M+ customer records",
                f"Overly permissive IAM role allowing data exfiltration",
                f"API BOLA vulnerability exposing all user profiles",
            ],
            'insider_threat': [
                f"Anomalous 3 AM access from finance employee downloading customer database",
                f"Terminated employee credentials used to access corporate Dropbox",
                f"Mass file download to USB device detected via DLP",
            ],
            'apt_espionage': [
                f"Volt Typhoon lateral movement across critical infrastructure",
                f"Lazarus Group credential dumping via Mimikatz",
                f"APT29 establishing persistence through scheduled task",
            ],
            'polymorphic_malware': [
                f"AI-mutating malware rewriting code every 15 seconds evading signatures",
                f"GPT-4 generated obfuscated PowerShell payload",
                f"Self-modifying binary with 10,000+ variants detected",
            ],
            'quantum_harvest': [
                f"TLS 1.2 traffic interception for future quantum decryption",
                f"Encrypted VPN sessions captured for Harvest Now Decrypt Later",
                f"Mass collection of RSA-encrypted communications",
            ]
        }
        
        return random.choice(descriptions.get(category, ["Unknown attack detected"]))
    
    def _get_or_create_entity(self, ip_address):
        """Get or create entity for IP"""
        entity = db_manager.execute_one(
            "SELECT id FROM entities WHERE identifier = ?", 
            (ip_address,)
        )
        
        if not entity:
            entity_id = str(uuid.uuid4())
            now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            
            # Determine entity type
            if ip_address.startswith(('192.168', '10.', '172.16')):
                entity_type = 'workstation'
            else:
                entity_type = 'external_ip'
            
            db_manager.execute_query("""
                INSERT INTO entities 
                (id, identifier, entity_type, threat_score, urgency_level, 
                 total_detections, first_seen, last_seen, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (entity_id, ip_address, entity_type, 50, 'medium', 0, now, now, now, now))
            
            return entity_id
        
        return entity['id']
    
    def insert_attacks_batch(self, attacks):
        """Batch insert attacks for performance"""
        # Prepare entity associations
        entity_cache = {}
        
        detection_records = []
        for attack in attacks:
            # Get or create entity (with caching)
            src_ip = attack['source_ip']
            if src_ip not in entity_cache:
                entity_cache[src_ip] = self._get_or_create_entity(src_ip)
            
            entity_id = entity_cache[src_ip]
            
            detection_records.append((
                attack['id'],
                attack['detection_type'],
                attack['title'],
                attack['description'],
                attack['severity'],
                attack['confidence_score'],
                attack['risk_score'],
                entity_id,
                attack['source_ip'],
                attack['destination_ip'],
                None,  # destination_port
                attack['technique_id'],
                attack['detected_at'],
                attack['created_at'],
                attack['updated_at']
            ))
        
        # Batch insert
        db_manager.execute_many("""
            INSERT INTO detections 
            (id, detection_type, title, description, severity, confidence_score, 
             risk_score, entity_id, source_ip, destination_ip, destination_port,
             technique_id, detected_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, detection_records)
        
        # Update entity counts
        for entity_id in set(entity_cache.values()):
            db_manager.execute_query("""
                UPDATE entities 
                SET total_detections = (
                    SELECT COUNT(*) FROM detections WHERE entity_id = ?
                ),
                threat_score = CASE 
                    WHEN (SELECT COUNT(*) FROM detections WHERE entity_id = ? AND severity = 'critical') > 5 THEN 95
                    WHEN (SELECT COUNT(*) FROM detections WHERE entity_id = ? AND severity IN ('critical', 'high')) > 3 THEN 75
                    ELSE 50
                END,
                urgency_level = CASE
                    WHEN (SELECT COUNT(*) FROM detections WHERE entity_id = ? AND severity = 'critical') > 5 THEN 'critical'
                    WHEN (SELECT COUNT(*) FROM detections WHERE entity_id = ? AND severity IN ('critical', 'high')) > 3 THEN 'high'
                    ELSE 'medium'
                END
                WHERE id = ?
            """, (entity_id, entity_id, entity_id, entity_id, entity_id, entity_id))
    
    def run_simulation(self):
        """Execute the massive attack simulation"""
        print("\n" + "="*80)
        print("🔥 AI-POWERED ATTACK SIMULATION - 200,000 ATTACKS")
        print("Based on 2025 Global Threat Landscape")
        print("="*80 + "\n")
        
        self.stats['start_time'] = time.time()
        
        # Clear existing data for clean test
        print("Clearing previous detections...")
        db_manager.execute_query("DELETE FROM detections")
        db_manager.execute_query("DELETE FROM entities")
        print("✅ Database cleared\n")
        
        num_batches = self.total_attacks // self.batch_size
        
        for batch_num in range(num_batches):
            batch_start = time.time()
            
            print(f"Batch {batch_num + 1}/{num_batches} - Generating {self.batch_size:,} attacks...")
            
            # Generate attacks
            attacks = self.generate_attack_batch(batch_num, self.batch_size)
            
            # Insert into database
            print(f"  └─ Inserting into database...")
            self.insert_attacks_batch(attacks)
            
            batch_time = time.time() - batch_start
            self.stats['total'] += len(attacks)
            
            print(f"  └─ Completed in {batch_time:.2f}s ({len(attacks)/batch_time:.0f} attacks/sec)")
            print(f"  └─ Total: {self.stats['total']:,} / {self.total_attacks:,}\n")
        
        self.stats['end_time'] = time.time()
        self.generate_report()
    
    def generate_report(self):
        """Generate final simulation report"""
        total_time = self.stats['end_time'] - self.stats['start_time']
        
        print("\n" + "="*80)
        print("📊 ATTACK SIMULATION COMPLETE")
        print("="*80 + "\n")
        
        print(f"Total Attacks Generated: {self.stats['total']:,}")
        print(f"Total Time: {total_time:.2f} seconds")
        print(f"Average Rate: {self.stats['total']/total_time:.0f} attacks/second")
        print(f"Database Size: {self._get_db_size()}")
        
        print(f"\n📈 ATTACK BREAKDOWN BY CATEGORY:")
        for category, count in sorted(self.stats['by_category'].items(), key=lambda x: x[1], reverse=True):
            pct = (count / self.stats['total']) * 100
            bar = "█" * int(pct / 2)
            desc = self.attack_categories[category]['description']
            print(f"  {category:25} {count:7,} ({pct:5.1f}%) {bar}")
            print(f"    └─ {desc}")
        
        print(f"\n🎯 SEVERITY DISTRIBUTION:")
        for severity in ['critical', 'high', 'medium', 'low']:
            count = self.stats['by_severity'].get(severity, 0)
            if count > 0:
                pct = (count / self.stats['total']) * 100
                bar = "█" * int(pct / 2)
                print(f"  {severity.upper():10} {count:7,} ({pct:5.1f}%) {bar}")
        
        print(f"\n✅ DATABASE STATUS:")
        # Get actual counts
        detection_count = db_manager.execute_one("SELECT COUNT(*) as count FROM detections")['count']
        entity_count = db_manager.execute_one("SELECT COUNT(*) as count FROM entities")['count']
        
        print(f"  Detections: {detection_count:,}")
        print(f"  Entities: {entity_count:,}")
        print(f"  Unique IPs: {len(set(self.external_ips + self.internal_ips))}")
        
        print("\n" + "="*80)
        print("🚀 PCDS ENTERPRISE - READY FOR 2025 THREAT LANDSCAPE")
        print("="*80 + "\n")
        
        print("Access dashboard at: http://localhost:3000")
        print("See all 200,000 detections in real-time!\n")
    
    def _get_db_size(self):
        """Get database file size"""
        try:
            import os
            db_path = "data/pcds_enterprise.db"
            if os.path.exists(db_path):
                size_bytes = os.path.getsize(db_path)
                size_mb = size_bytes / (1024 * 1024)
                return f"{size_mb:.2f} MB"
        except:
            pass
        return "Unknown"


if __name__ == "__main__":
    simulator = MassiveAIAttackSimulator(total_attacks=200000)
    simulator.run_simulation()
