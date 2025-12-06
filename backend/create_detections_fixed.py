"""
Create detections with proper SQLite datetime format
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import db_manager
import uuid
from datetime import datetime

def create_detections():
    hostname = "LAPTOP-2GC24SJ1"
    
    # Get entity ID
    entity = db_manager.execute_one("SELECT id FROM entities WHERE identifier = ?", (hostname,))
    if not entity:
        print("❌ Entity not found!")
        return
    
    entity_id = entity['id']
    
    # Use SQLite datetime format: YYYY-MM-DD HH:MM:SS
    now = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    
    detections = [
        ("suspicious_powershell", "PowerShell Execution Detected", "high", "T1059"),
        ("unusual_port", "Connection to Suspicious Port", "medium", "T1071"),
        ("c2_beaconing", "Potential C2 Communication", "critical", "T1071"),
        ("large_upload", "Large Data Exfiltration Detected", "high", "T1567"),
    ]
    
    print("Creating detections with proper timestamp format...")
    for det_type, title, severity, technique in detections:
        det_id = str(uuid.uuid4())
        
        db_manager.execute_query("""
            INSERT INTO detections 
            (id, detection_type, title, description, severity, confidence_score, risk_score,
             entity_id, source_ip, technique_id, detected_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            det_id,
            det_type,
            title,
            title,
            severity,
            0.85,
            70 if severity == 'critical' else 50,
            entity_id,
            "192.168.1.100",
            technique,
            now,  # SQLite format
            now,
            now
        ))
        
        icon = "🚨" if severity == 'critical' else "⚠️" if severity == 'high' else "ℹ️"
        print(f"  {icon} {severity.upper()}: {title}")
    
    print(f"\n✅ Created {len(detections)} detections!")
    print("🔄 Refresh http://localhost:3000/detections")

if __name__ == "__main__":
    create_detections()
