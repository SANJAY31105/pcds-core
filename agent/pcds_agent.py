#!/usr/bin/env python3
"""
PCDS Agent - Network Traffic Monitor
Monitors network connections and sends data to PCDS for threat detection.

Usage:
    python pcds_agent.py --api-key YOUR_API_KEY

Requirements:
    pip install psutil requests
"""

import argparse
import socket
import time
import requests
import psutil
import configparser
import os
import sys
from datetime import datetime
from typing import List, Dict

# PCDS API Configuration
PCDS_API_URL = "https://pcds-backend-production.up.railway.app/api/v2/ingest"
SEND_INTERVAL = 10  # seconds

def load_config():
    """Load configuration from config.ini in the same directory"""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), 'config.ini')
    
    if os.path.exists(config_path):
        config.read(config_path)
        return config
    return None

def get_hostname() -> str:
    """Get the hostname of this machine"""
    return socket.gethostname()

def get_network_connections() -> List[Dict]:
    """Get current network connections using psutil"""
    connections = []
    
    try:
        for conn in psutil.net_connections(kind='inet'):
            if conn.status == 'ESTABLISHED' or conn.status == 'LISTEN':
                # Get process name if available
                process_name = "unknown"
                if conn.pid:
                    try:
                        process = psutil.Process(conn.pid)
                        process_name = process.name()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                event = {
                    "source_ip": conn.laddr.ip if conn.laddr else "0.0.0.0",
                    "dest_ip": conn.raddr.ip if conn.raddr else None,
                    "dest_port": conn.raddr.port if conn.raddr else None,
                    "protocol": "TCP" if conn.type == socket.SOCK_STREAM else "UDP",
                    "process_name": process_name,
                    "status": conn.status,
                    "timestamp": datetime.now().isoformat()
                }
                connections.append(event)
    except psutil.AccessDenied:
        print("Warning: Some connections require elevated privileges to read")
    
    return connections

def send_to_pcds(api_key: str, hostname: str, events: List[Dict], api_url: str = PCDS_API_URL) -> bool:
    """Send events to PCDS backend"""
    if not events:
        return True
    
    payload = {
        "api_key": api_key,
        "hostname": hostname,
        "events": events
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Sent {data.get('events_received', 0)} events to PCDS")
            return True
        elif response.status_code == 401:
            print("✗ Invalid API key!")
            return False
        else:
            print(f"✗ Error: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ Connection error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='PCDS Network Agent')
    parser.add_argument('--api-key', help='Your PCDS API key')
    parser.add_argument('--url', help='PCDS Backend URL')
    parser.add_argument('--interval', type=int, default=SEND_INTERVAL, 
                        help='Send interval in seconds (default: 10)')
    args = parser.parse_args()
    
    # Load config file (priority: Args > Config File > Defaults)
    config = load_config()
    
    api_key = args.api_key
    api_url = args.url or PCDS_API_URL
    
    if not api_key and config and 'PCDS' in config:
        api_key = config['PCDS'].get('api_key')
        if config['PCDS'].get('url'):
            api_url = config['PCDS'].get('url')
            
    if not api_key:
        print("Error: API Key required. Use --api-key or create config.ini")
        return
    
    hostname = get_hostname()
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    PCDS Network Agent                        ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Hostname: {hostname:<48} ║
    ║  Interval: {args.interval} seconds{' ' * 40}║
    ║  API URL:  {api_url[:47]}...║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    print("Starting network monitoring... Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # Get current connections
            events = get_network_connections()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(events)} active connections")
            
            # Send to PCDS
            if events:
                success = send_to_pcds(api_key, hostname, events, api_url)
                if not success:
                    print("Retrying in next interval...")
            
            # Wait for next interval
            time.sleep(args.interval)
            
    except KeyboardInterrupt:
        print("\n\nAgent stopped. Goodbye!")

if __name__ == "__main__":
    main()
