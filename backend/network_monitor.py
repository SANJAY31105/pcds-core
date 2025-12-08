"""
Real Network Monitor
Captures actual network connections and traffic from the local machine.
Uses psutil for connection monitoring (no admin privileges required).
"""

import psutil
import socket
import time
import threading
from datetime import datetime
from collections import defaultdict, deque
from typing import Dict, List, Any, Optional
import json

# Try importing the anomaly detector
try:
    from ml.anomaly_detector import anomaly_detector
    ANOMALY_DETECTION_ENABLED = True
except ImportError:
    ANOMALY_DETECTION_ENABLED = False
    print("⚠️ Anomaly detector not available for network monitor")

# MITRE ATT&CK mappings for suspicious patterns
SUSPICIOUS_PORTS = {
    4444: {"name": "Metasploit", "technique": "T1571", "severity": "critical"},
    5555: {"name": "Android Debug", "technique": "T1071", "severity": "high"},
    6666: {"name": "IRC Botnet", "technique": "T1071", "severity": "critical"},
    31337: {"name": "Back Orifice", "technique": "T1571", "severity": "critical"},
    1337: {"name": "Hacker Port", "technique": "T1571", "severity": "high"},
    8080: {"name": "HTTP Proxy", "technique": "T1090", "severity": "low"},
    3389: {"name": "RDP", "technique": "T1021", "severity": "medium"},
    22: {"name": "SSH", "technique": "T1021", "severity": "low"},
    23: {"name": "Telnet", "technique": "T1021", "severity": "high"},
    445: {"name": "SMB", "technique": "T1021", "severity": "medium"},
}

KNOWN_SAFE_DOMAINS = [
    "microsoft.com", "google.com", "github.com", "amazonaws.com",
    "cloudflare.com", "akamai.net", "azure.com", "windows.net"
]


class RealNetworkMonitor:
    """Real-time network connection and traffic monitor"""
    
    def __init__(self):
        self.running = False
        self.connections: Dict[str, Dict] = {}
        self.events: deque = deque(maxlen=1000)
        self.stats = {
            "total_connections": 0,
            "active_connections": 0,
            "bytes_sent": 0,
            "bytes_recv": 0,
            "suspicious_count": 0,
            "packets_analyzed": 0,
            "start_time": None
        }
        self.connection_history: Dict[str, List] = defaultdict(list)
        self.dns_cache: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        print("🔌 Real Network Monitor initialized")
    
    def start(self):
        """Start monitoring in background thread"""
        if self.running:
            return
        
        self.running = True
        self.stats["start_time"] = datetime.now().isoformat()
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("🟢 Network monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        print("🔴 Network monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        last_io = psutil.net_io_counters()
        
        while self.running:
            try:
                # Get current connections
                connections = psutil.net_connections(kind='inet')
                current_io = psutil.net_io_counters()
                
                # Calculate traffic delta
                bytes_sent = current_io.bytes_sent - last_io.bytes_sent
                bytes_recv = current_io.bytes_recv - last_io.bytes_recv
                packets_sent = current_io.packets_sent - last_io.packets_sent
                packets_recv = current_io.packets_recv - last_io.packets_recv
                
                with self._lock:
                    self.stats["bytes_sent"] += bytes_sent
                    self.stats["bytes_recv"] += bytes_recv
                    self.stats["packets_analyzed"] += packets_sent + packets_recv
                    self.stats["active_connections"] = len([c for c in connections if c.status == 'ESTABLISHED'])
                
                last_io = current_io
                
                # Process each connection
                for conn in connections:
                    self._process_connection(conn)
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(2)
    
    def _process_connection(self, conn):
        """Process a single connection"""
        if not conn.raddr:  # No remote address
            return
        
        remote_ip = conn.raddr.ip
        remote_port = conn.raddr.port
        local_port = conn.laddr.port if conn.laddr else 0
        
        conn_key = f"{remote_ip}:{remote_port}"
        
        # Skip if already processed recently
        if conn_key in self.connections:
            return
        
        # Resolve hostname
        hostname = self._resolve_hostname(remote_ip)
        
        # Check for suspicious patterns
        threat_info = self._analyze_connection(remote_ip, remote_port, local_port, hostname)
        
        # Calculate anomaly score if available
        anomaly_score = 0.0
        if ANOMALY_DETECTION_ENABLED:
            features = anomaly_detector.extract_features({
                'source_ip': '127.0.0.1',
                'destination_ip': remote_ip,
                'port': remote_port,
                'protocol': 'tcp',
                'packet_size': 1000
            })
            _, anomaly_score, _ = anomaly_detector.predict(features)
        
        # Create connection record
        connection_data = {
            "remote_ip": remote_ip,
            "remote_port": remote_port,
            "local_port": local_port,
            "hostname": hostname,
            "status": conn.status,
            "pid": conn.pid,
            "process": self._get_process_name(conn.pid),
            "first_seen": datetime.now().isoformat(),
            "threat_info": threat_info,
            "anomaly_score": round(anomaly_score, 3),
            "is_suspicious": threat_info is not None or anomaly_score > 0.5
        }
        
        with self._lock:
            self.connections[conn_key] = connection_data
            self.stats["total_connections"] += 1
            
            if connection_data["is_suspicious"]:
                self.stats["suspicious_count"] += 1
        
        # Generate event
        self._create_event(connection_data)
    
    def _resolve_hostname(self, ip: str) -> str:
        """Resolve IP to hostname with caching"""
        if ip in self.dns_cache:
            return self.dns_cache[ip]
        
        try:
            hostname = socket.gethostbyaddr(ip)[0]
            self.dns_cache[ip] = hostname
            return hostname
        except:
            self.dns_cache[ip] = ip
            return ip
    
    def _get_process_name(self, pid: int) -> str:
        """Get process name from PID"""
        if not pid:
            return "Unknown"
        try:
            return psutil.Process(pid).name()
        except:
            return "Unknown"
    
    def _analyze_connection(self, remote_ip: str, remote_port: int, local_port: int, hostname: str) -> Optional[Dict]:
        """Analyze connection for threats"""
        
        # Check suspicious ports
        if remote_port in SUSPICIOUS_PORTS:
            return {
                "type": "suspicious_port",
                **SUSPICIOUS_PORTS[remote_port]
            }
        
        if local_port in SUSPICIOUS_PORTS:
            return {
                "type": "suspicious_local_port",
                **SUSPICIOUS_PORTS[local_port]
            }
        
        # Check for non-standard high ports with external IPs
        if remote_port > 10000 and not any(domain in hostname for domain in KNOWN_SAFE_DOMAINS):
            if not remote_ip.startswith(('192.168.', '10.', '172.', '127.')):
                return {
                    "type": "high_port_external",
                    "name": "High Port Connection",
                    "technique": "T1571",
                    "severity": "medium"
                }
        
        return None
    
    def _create_event(self, connection_data: Dict):
        """Create event for live feed"""
        severity = "low"
        event_type = "system"
        
        if connection_data["is_suspicious"]:
            event_type = "detection"
            if connection_data["threat_info"]:
                severity = connection_data["threat_info"].get("severity", "medium")
        
        message = f"Connection: {connection_data['process']} → {connection_data['hostname']}:{connection_data['remote_port']}"
        
        if connection_data["is_suspicious"]:
            message = f"⚠️ Suspicious: {connection_data['process']} → {connection_data['hostname']}:{connection_data['remote_port']}"
        
        event = {
            "id": f"net_{datetime.now().timestamp()}",
            "type": event_type,
            "severity": severity,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "source": "Network Monitor",
            "details": {
                "remote_ip": connection_data["remote_ip"],
                "remote_port": connection_data["remote_port"],
                "process": connection_data["process"],
                "anomaly_score": connection_data["anomaly_score"]
            }
        }
        
        if connection_data["threat_info"]:
            event["mitre"] = {
                "technique_id": connection_data["threat_info"].get("technique", ""),
                "technique_name": connection_data["threat_info"].get("name", "")
            }
        
        with self._lock:
            self.events.append(event)
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        with self._lock:
            return {
                **self.stats,
                "uptime_seconds": (datetime.now() - datetime.fromisoformat(self.stats["start_time"])).seconds if self.stats["start_time"] else 0
            }
    
    def get_connections(self, limit: int = 50) -> List[Dict]:
        """Get recent connections"""
        with self._lock:
            connections = list(self.connections.values())
            return sorted(connections, key=lambda x: x["first_seen"], reverse=True)[:limit]
    
    def get_events(self, limit: int = 100) -> List[Dict]:
        """Get recent events for live feed"""
        with self._lock:
            return list(self.events)[-limit:]
    
    def get_suspicious(self) -> List[Dict]:
        """Get suspicious connections"""
        with self._lock:
            return [c for c in self.connections.values() if c["is_suspicious"]]


# Global instance
network_monitor = RealNetworkMonitor()


# API functions for FastAPI integration
def start_network_monitoring():
    """Start the network monitor"""
    network_monitor.start()
    return {"status": "started"}


def stop_network_monitoring():
    """Stop the network monitor"""
    network_monitor.stop()
    return {"status": "stopped"}


def get_network_stats():
    """Get network statistics"""
    return network_monitor.get_stats()


def get_network_connections(limit: int = 50):
    """Get active connections"""
    return network_monitor.get_connections(limit)


def get_network_events(limit: int = 100):
    """Get network events for live feed"""
    return network_monitor.get_events(limit)


def get_suspicious_connections():
    """Get suspicious connections only"""
    return network_monitor.get_suspicious()


if __name__ == "__main__":
    # Test the monitor
    print("Starting network monitor test...")
    network_monitor.start()
    
    try:
        for i in range(10):
            time.sleep(2)
            stats = network_monitor.get_stats()
            print(f"\n📊 Stats: {json.dumps(stats, indent=2)}")
            
            events = network_monitor.get_events(5)
            for event in events:
                print(f"  📡 {event['message']}")
    except KeyboardInterrupt:
        pass
    finally:
        network_monitor.stop()
        print("\n✅ Test complete")
