"""
PCDS Enterprise - Real-Time System Monitor
Monitors your computer's network connections and processes
Feeds real data to PCDS detection engine
"""

import subprocess
import time
import requests
import json
from datetime import datetime
import socket
import psutil

PCDS_API = "http://localhost:8000/api/v2"

def get_network_connections():
    """Get active network connections from your computer"""
    connections = []
    
    try:
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'ESTABLISHED':
                local_ip = conn.laddr.ip if conn.laddr else 'unknown'
                local_port = conn.laddr.port if conn.laddr else 0
                remote_ip = conn.raddr.ip if conn.raddr else 'unknown'
                remote_port = conn.raddr.port if conn.raddr else 0
                
                # Try to get process name
                try:
                    process = psutil.Process(conn.pid)
                    process_name = process.name()
                except:
                    process_name = 'unknown'
                
                connections.append({
                    'local_ip': local_ip,
                    'local_port': local_port,
                    'remote_ip': remote_ip,
                    'remote_port': remote_port,
                    'process': process_name,
                    'pid': conn.pid
                })
    except Exception as e:
        print(f"❌ Error getting connections: {e}")
    
    return connections

def get_running_processes():
    """Get list of running processes"""
    processes = []
    
    try:
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent']):
            try:
                processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'user': proc.info['username'],
                    'cpu': proc.info['cpu_percent']
                })
            except:
                pass
    except Exception as e:
        print(f"❌ Error getting processes: {e}")
    
    return processes

def create_detection_event(connection):
    """Create a detection event for PCDS"""
    # Analyze connection for suspicious activity
    timestamp = datetime.utcnow().isoformat()
    
    # Check for suspicious ports
    suspicious_ports = [4444, 5555, 6666, 7777, 8888, 9999]  # Common C2 ports
    suspicious = connection['remote_port'] in suspicious_ports
    
    # Check for unusual processes making connections
    suspicious_processes = ['powershell.exe', 'cmd.exe', 'wscript.exe', 'cscript.exe']
    if connection['process'].lower() in suspicious_processes:
        suspicious = True
    
    return {
        'source_ip': connection['local_ip'],
        'destination_ip': connection['remote_ip'],
        'destination_port': connection['remote_port'],
        'process_name': connection['process'],
        'process_id': connection['pid'],
        'timestamp': timestamp,
        'suspicious': suspicious,
        'detection_type': 'c2_communication' if suspicious else 'normal_traffic'
    }

def send_to_pcds_database(connections, processes):
    """Insert real data into PCDS database"""
    from config.database import db_manager
    import uuid
    
    # Insert connections as detections
    for conn in connections:
        event = create_detection_event(conn)
        
        if event['suspicious']:
            # Create actual detection in database
            try:
                detection_id = str(uuid.uuid4())
                
                db_manager.execute_query("""
                    INSERT INTO detections 
                    (id, type, severity, source_ip, destination_ip, destination_port, 
                     raw_data, detected_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    detection_id,
                    event['detection_type'],
                    'high' if event['suspicious'] else 'low',
                    event['source_ip'],
                    event['destination_ip'],
                    event['destination_port'],
                    json.dumps(event),
                    event['timestamp'],
                    event['timestamp']
                ))
                
                print(f"🚨 Suspicious Activity: {conn['process']} → {conn['remote_ip']}:{conn['remote_port']}")
            except Exception as e:
                print(f"❌ Database insert error: {e}")
    
    # Update entity (your computer)
    hostname = socket.gethostname()
    try:
        # Check if entity exists
        existing = db_manager.execute_one(
            "SELECT id FROM entities WHERE identifier = ?", 
            (hostname,)
        )
        
        if not existing:
            # Create entity for your computer
            entity_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            db_manager.execute_query("""
                INSERT INTO entities (id, identifier, entity_type, threat_score, urgency_level, 
                                     total_detections, first_seen, last_seen, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entity_id, hostname, 'device', 10, 'low', 0, now, now, now, now
            ))
            print(f"✅ Created entity for your computer: {hostname}")
    except Exception as e:
        print(f"❌ Entity update error: {e}")

def monitor_system():
    """Main monitoring loop"""
    print("\n" + "="*70)
    print("🔍 PCDS REAL-TIME SYSTEM MONITOR")
    print("Monitoring your computer's network activity...")
    print("="*70 + "\n")
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    
    print(f"💻 Computer: {hostname}")
    print(f"🌐 IP Address: {local_ip}")
    print(f"📊 Dashboard: http://localhost:3000")
    print(f"\nPress Ctrl+C to stop\n")
    print("-" * 70)
    
    cycle = 0
    
    try:
        while True:
            cycle += 1
            print(f"\n[Cycle {cycle}] {datetime.now().strftime('%H:%M:%S')}")
            
            # Get real data
            connections = get_network_connections()
            processes = get_running_processes()
            
            print(f"  📡 Active Connections: {len(connections)}")
            print(f"  ⚙️  Running Processes: {len(processes)}")
            
            # Show sample of connections
            if connections:
                print(f"\n  Recent Connections:")
                for conn in connections[:5]:  # Show first 5
                    status = "🚨" if conn['remote_port'] in [4444, 5555] else "✅"
                    print(f"    {status} {conn['process'][:20]:20} → {conn['remote_ip']:15}:{conn['remote_port']}")
            
            # Send to PCDS database
            send_to_pcds_database(connections, processes)
            
            # Wait before next scan
            time.sleep(30)  # Scan every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("🛑 Monitoring stopped")
        print(f"📊 View results at: http://localhost:3000")
        print("="*70)

if __name__ == "__main__":
    # Need to import from backend
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        monitor_system()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  1. PCDS backend is running (python main_v2.py)")
        print("  2. You have psutil installed (pip install psutil)")
