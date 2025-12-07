#!/usr/bin/env python3
"""
PCDS Network Monitor - Captures college network traffic
For deployment on Ubuntu 22.04 LTS with SPAN port configuration
"""

import sys
import os

# Check if running on Linux/Production
if os.name == 'nt':
    print("WARNING: This script is designed for Linux production servers with pyshark/tshark.")
    print("On Windows, it requires Wireshark installed and correctly configured.")
    # We'll mock the import for development/Windows compatibility
    try:
        import pyshark
    except ImportError:
        print("Error: pyshark not found. Run: pip install pyshark")
        sys.exit(1)
else:
    import pyshark

import requests
import json
from datetime import datetime
import time

# Configuration
PCDS_API = os.getenv("PCDS_API_URL", "http://localhost:8000/api/v2/detections")
MONITORING_INTERFACE = os.getenv("MONITOR_INTERFACE", "eth1")  # SPAN port interface

def monitor_network():
    print(f"Starting PCDS Network Monitor on {MONITORING_INTERFACE}...")
    
    try:
        # Use LiveCapture for real-time monitoring
        capture = pyshark.LiveCapture(
            interface=MONITORING_INTERFACE,
            bpf_filter='ip'  # Only capture IP traffic
        )
        
        print("Listening for traffic...")
        
        for packet in capture.sniff_continuously():
            try:
                if hasattr(packet, 'ip'):
                    analyze_packet(packet)
            except Exception as e:
                # Log error but keep running
                print(f"Error processing packet: {e}")
                continue
                
    except Exception as e:
        print(f"Critical error: {e}")
        print("Ensure tshark is installed: sudo apt install tshark")

def analyze_packet(packet):
    """Analyze packet for threat signatures"""
    try:
        # Extract basic info
        src_ip = packet.ip.src
        dst_ip = packet.ip.dst
        
        # Detect suspicious patterns
        suspicious = False
        detection_type = "normal_traffic"
        technique_id = None
        
        # 1. C2 Communication Check (Common Ports)
        if hasattr(packet, 'tcp'):
            dst_port = int(packet.tcp.dstport)
            if dst_port in [4444, 5555, 6666, 1337, 31337]:
                suspicious = True
                detection_type = "c2_communication"
                technique_id = "T1071" # Standard Application Layer Protocol
        
        # 2. SMB Lateral Movement Check
        if hasattr(packet, 'tcp'):
            dst_port = int(packet.tcp.dstport)
            if dst_port == 445:
                # Simple heuristic: excessive SMB could be lateral movement
                # In production, we'd track volume/frequency
                suspicious = False # Don't flag single packets to avoid noise
        
        # 3. Data Exfiltration Check (Large packet size)
        if hasattr(packet, 'length') and int(packet.length) > 1500:
            # Simplified check for jumbo frames/data exfil
            pass 
            
        # Send detection to PCDS if suspicious
        if suspicious:
            send_detection(src_ip, dst_ip, detection_type, technique_id)
            
    except AttributeError:
        pass

def send_detection(src_ip, dst_ip, det_type, technique_id=None):
    """Send detection to PCDS API"""
    try:
        data = {
            'detection_type': det_type,
            'title': f'{det_type.replace("_", " ").title()} Detected',
            'description': f'Suspicious traffic detected from {src_ip} to {dst_ip}',
            'severity': 'high',
            'confidence_score': 0.85,
            'risk_score': 75,
            'entity_id': src_ip, # Map to source IP for now
            'source_ip': src_ip,
            'destination_ip': dst_ip,
            'technique_id': technique_id,
            'detected_at': datetime.utcnow().isoformat(),
            'metadata': {
                'sensor': 'network_monitor_v1',
                'interface': MONITORING_INTERFACE
            }
        }
        
        # In a real deployment, we'd use a queued sender or Kafka producer directly
        # For this script, simple HTTP POST
        try:
            requests.post(PCDS_API, json=data, timeout=1)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 Alert: {src_ip} -> {dst_ip} ({det_type})")
        except requests.exceptions.RequestException:
            print(f"Error: Could not connect to PCDS API at {PCDS_API}")
            
    except Exception as e:
        print(f"Error sending detection: {e}")

if __name__ == "__main__":
    monitor_network()
