"""
PCDS Test - Trigger Detections
Creates detections that will show up in dashboard
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.database import db_manager
import uuid
from datetime import datetime

def create_test_detections():
    """Create sample detections from your computer"""
    
    hostname = "LAPTOP-2GC24SJ1"
    my_ip = "192.168.1.100"  # Your local IP
    
    detections = [
        {
            "type": "suspicious_powershell",
            "severity": "high",
            "description": "PowerShell execution with bypass flag detected",
            "source_ip": my_ip,
            "destination_ip": "8.8.8.8",
            "technique_id": "T1059"
        },
        {
            "type": "unusual_port",
            "severity": "medium",
            "description": "Connection to unusual port detected",
            "source_ip": my_ip,
            "destination_ip": "203.0.113.50",
            "destination_port": 4444,
            "technique_id": "T1071"
        },
        {
            "type": "c2_beaconing",
            "severity": "critical",
            "description": "Potential C2 beaconing pattern detected",
            "source_ip": my_ip,
            "destination_ip": "198.51.100.42",
            "destination_port": 443,
            "technique_id": "T1071"
        },
        {
            "type": "large_upload",
            "severity": "high",
            "description": "Large data upload to external IP",
            "source_ip": my_ip,
            "destination_ip": "104.16.249.249",
            "destination_port": 443,
            "technique_id": "T1567"
        }
    ]
    
    print("Creating test detections from your computer activity...")
    print("-" * 70)
    
    # Get or create entity for user's computer
    entity = db_manager.execute_one("SELECT id FROM entities WHERE identifier = ?", (hostname,))
    if not entity:
        print("❌ Computer entity not found. Run monitor_real_system.py first!")
        return
    
    entity_id = entity['id']
    
    for det in detections:
        det_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        db_manager.execute_query("""
            INSERT INTO detections 
            (id, detection_type, title, description, severity, confidence_score, risk_score,
             entity_id, source_ip, destination_ip, destination_port, technique_id,
             detected_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            det_id,
            det['type'],
            det['description'],  # title
            det['description'],  # description
            det['severity'],
            0.85,  # confidence_score
            70 if det['severity'] == 'critical' else 50,  # risk_score
            entity_id,
            det['source_ip'],
            det.get('destination_ip', ''),
            det.get('destination_port', 0),
            det.get('technique_id', ''),
            now,
            now,
            now
        ))
        
        severity_icon = "🚨" if det['severity'] == 'critical' else "⚠️" if det['severity'] == 'high' else "ℹ️"
        print(f"{severity_icon} {det['severity'].upper()}: {det['description']}")
    
    print("\n✅ Created 4 test detections!")
    print("\n📊 Check your dashboard at: http://localhost:3000/detections")
    print("🔄 Refresh the page to see new detections!")

if __name__ == "__main__":
    create_test_detections()
