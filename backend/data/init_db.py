"""
Database Initialization Script for PCDS Enterprise
Runs schema creation and seeds MITRE ATT&CK data
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

# Get the directory of this script
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR
SCHEMA_FILE = DATA_DIR / "schema.sql"
MITRE_FILE = DATA_DIR / "mitre_attack_full.json"
DB_FILE = SCRIPT_DIR.parent / "pcds_enterprise.db"

def init_database():
    """Initialize the database with schema and MITRE data"""
    
    print("🗄️  PCDS Enterprise Database Initialization")
    print("=" * 60)
    
    # Remove existing database if present
    if DB_FILE.exists():
        response = input(f"Database {DB_FILE} already exists. Recreate? (yes/no): ")
        if response.lower() != 'yes':
            print("❌ Initialization cancelled")
            return
        DB_FILE.unlink()
        print(f"🗑️  Deleted existing database: {DB_FILE}")
    
    # Create new database connection
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Step 1: Execute schema
        print("\n📋 Step 1: Creating database schema...")
        with open(SCHEMA_FILE, 'r') as f:
            schema_sql = f.read()
        
        # Execute SQL statements (split by semicolon for SQLite)
        for statement in schema_sql.split(';'):
            statement = statement.strip()
            if statement:
                try:
                    cursor.execute(statement)
                except sqlite3.Error as e:
                    if "already exists" not in str(e):
                        print(f"⚠️  Warning: {e}")
        
        conn.commit()
        print("✅ Schema created successfully")
        
        # Step 2: Load and insert MITRE techniques
        print("\n📋 Step 2: Loading MITRE ATT&CK data...")
        with open(MITRE_FILE, 'r') as f:
            mitre_data = json.load(f)
        
        # Insert techniques
        techniques = mitre_data.get('techniques', [])
        print(f"📊 Inserting {len(techniques)} MITRE techniques...")
        
        for technique in techniques:
            try:
                cursor.execute("""
                    INSERT INTO mitre_techniques 
                    (id, name, description, tactic_id, severity, platforms, 
                     data_sources, mitigations, detection_methods)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    technique['id'],
                    technique['name'],
                    technique.get('description', ''),
                    technique['tactic_id'],
                    technique.get('severity', 'medium'),
                    json.dumps(technique.get('platforms', [])),
                    json.dumps(technique.get('data_sources', [])),
                    json.dumps(technique.get('mitigations', [])),
                    json.dumps(technique.get('detection_methods', []))
                ))
            except sqlite3.IntegrityError:
                print(f"⚠️  Technique {technique['id']} already exists, skipping")
        
        conn.commit()
        print(f"✅ Inserted {len(techniques)} MITRE techniques")
        
        # Step 3: Verify installation
        print("\n📋 Step 3: Verifying installation...")
        
        cursor.execute("SELECT COUNT(*) FROM mitre_tactics")
        tactics_count = cursor.fetchone()[0]
        print(f"✅ MITRE Tactics: {tactics_count}")
        
        cursor.execute("SELECT COUNT(*) FROM mitre_techniques")
        techniques_count = cursor.fetchone()[0]
        print(f"✅ MITRE Techniques: {techniques_count}")
        
        # Create initial hunt queries
        print("\n📋 Step 4: Creating default hunt queries...")
        
        hunt_queries = [
            {
                'id': 'hunt_001',
                'name': 'Credential Theft Activity',
                'description': 'Hunt for credential dumping, brute force, and password spraying attacks',
                'query_type': 'template',
                'detection_types': json.dumps(['credential_dumping', 'brute_force', 'password_spraying', 'kerberoasting']),
                'technique_ids': json.dumps(['T1003', 'T1110', 'T1558']),
                'time_range': '7d',
                'created_by': 'system',
                'is_public': True
            },
            {
                'id': 'hunt_002',
                'name': 'Lateral Movement Patterns',
                'description': 'Identify suspicious lateral movement via RDP, SMB, or WMI',
                'query_type': 'template',
                'detection_types': json.dumps(['rdp_lateral', 'smb_lateral', 'wmi_lateral', 'psexec', 'pass_the_hash']),
                'technique_ids': json.dumps(['T1021', 'T1047', 'T1550']),
                'time_range': '7d',
                'created_by': 'system',
                'is_public': True
            },
            {
                'id': 'hunt_003',
                'name': 'Command & Control Beaconing',
                'description': 'Detect C2 communication patterns and beaconing behavior',
                'query_type': 'template',
                'detection_types': json.dumps(['c2_beaconing', 'dns_tunneling', 'proxy_usage']),
                'technique_ids': json.dumps(['T1071', 'T1090', 'T1095']),
                'time_range': '24h',
                'created_by': 'system',
                'is_public': True
            },
            {
                'id': 'hunt_004',
                'name': 'Data Exfiltration',
                'description': 'Find large outbound transfers and exfiltration attempts',
                'query_type': 'template',
                'detection_types': json.dumps(['data_exfiltration', 'large_upload', 'dns_exfiltration', 'cloud_upload']),
                'technique_ids': json.dumps(['T1041', 'T1048', 'T1567']),
                'time_range': '7d',
                'created_by': 'system',
                'is_public': True
            },
            {
                'id': 'hunt_005',
                'name': 'Privilege Escalation Attempts',
                'description': 'Hunt for privilege escalation techniques and exploits',
                'query_type': 'template',
                'detection_types': json.dumps(['token_manipulation', 'process_injection', 'uac_bypass', 'privilege_escalation']),
                'technique_ids': json.dumps(['T1068', 'T1134', 'T1548', 'T1055']),
                'time_range': '7d',
                'created_by': 'system',
                'is_public': True
            },
            {
                'id': 'hunt_006',
                'name': 'Ransomware Indicators',
                'description': 'Detect ransomware deployment and impact activities',
                'query_type': 'template',
                'detection_types': json.dumps(['ransomware', 'data_destruction', 'backup_deletion', 'disable_security']),
                'technique_ids': json.dumps(['T1486', 'T1485', 'T1490', 'T1562']),
                'time_range': '24h',
                'created_by': 'system',
                'is_public': True
            }
        ]
        
        for query in hunt_queries:
            cursor.execute("""
                INSERT INTO hunt_queries 
                (id, name, description, query_type, detection_types, technique_ids, 
                 time_range, created_by, is_public)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query['id'],
                query['name'],
                query['description'],
                query['query_type'],
                query['detection_types'],
                query['technique_ids'],
                query['time_range'],
                query['created_by'],
                query['is_public']
            ))
        
        conn.commit()
        print(f"✅ Created {len(hunt_queries)} default hunt queries")
        
        # Final summary
        print("\n" + "=" * 60)
        print("✅ Database initialization complete!")
        print(f"📁 Database location: {DB_FILE}")
        print(f"📊 Total tables created: 15+")
        print(f"🎯 MITRE Tactics: {tactics_count}")
        print(f"🎯 MITRE Techniques: {techniques_count}")
        print(f"🔍 Hunt Queries: {len(hunt_queries)}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        conn.rollback()
        raise
    
    finally:
        conn.close()

if __name__ == "__main__":
    init_database()
