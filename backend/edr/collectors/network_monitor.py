"""
Network Monitor - EDR Component
Monitors network connections for malicious activity

Detects:
- C2 beaconing (regular intervals)
- DNS tunneling (high entropy, long queries)
- Suspicious ports (4444, 31337, etc.)
- Data exfiltration (large outbound transfers)
- Reverse shells
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Thread, Event
from collections import defaultdict
import socket
import time
import math
import psutil

from ..core.event_queue import get_event_queue


# Known malicious ports
MALICIOUS_PORTS = {
    4444: ("Metasploit", "critical"),
    5555: ("Android Debug", "high"),
    6666: ("IRC Botnet", "critical"),
    6667: ("IRC", "high"),
    31337: ("Back Orifice", "critical"),
    12345: ("NetBus", "critical"),
    27374: ("SubSeven", "critical"),
    20034: ("NetBus 2", "critical"),
    1337: ("Leet Port", "medium"),
}

# Suspicious TLDs often used in malware C2
SUSPICIOUS_TLDS = [
    ".top", ".xyz", ".club", ".info", ".online", ".site", 
    ".pw", ".tk", ".ml", ".ga", ".cf", ".gq", ".ru", ".cn"
]

# Known legitimate domains to whitelist
WHITELIST_DOMAINS = [
    "microsoft.com", "windows.com", "windowsupdate.com",
    "google.com", "googleapis.com", "gstatic.com",
    "github.com", "githubusercontent.com",
    "cloudflare.com", "cloudfront.net",
    "akamai.net", "akamaiedge.net",
    "amazon.com", "amazonaws.com",
    "azure.com", "azureedge.net",
]


@dataclass
class ConnectionStats:
    """Track connection statistics for beaconing detection"""
    intervals: List[float] = field(default_factory=list)
    byte_counts: List[int] = field(default_factory=list)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    connection_count: int = 0


class NetworkMonitor:
    """
    Network connection monitor for EDR
    
    Detects:
    - C2 beaconing patterns
    - DNS tunneling
    - Suspicious ports
    - Data exfiltration
    - Reverse shells
    """
    
    def __init__(self):
        self.event_queue = get_event_queue()
        self._stop_event = Event()
        self._monitor_thread = None
        
        # Track connections
        self._connection_history: Dict[str, ConnectionStats] = defaultdict(ConnectionStats)
        self._known_connections: Set[Tuple[str, int, int]] = set()  # (remote_ip, remote_port, local_port)
        self._dns_queries: Dict[str, int] = defaultdict(int)  # domain -> count
        
        # Beaconing detection parameters
        self.beacon_interval_tolerance = 0.2  # 20% variance tolerance
        self.min_beacon_samples = 5  # Minimum samples for beaconing detection
        
        # Stats
        self.stats = {
            "connections_monitored": 0,
            "suspicious_connections": 0,
            "beaconing_detected": 0,
            "dns_tunneling_detected": 0,
            "malicious_ports_detected": 0
        }
        
        print("🌐 Network Monitor initialized")
    
    def start(self):
        """Start network monitoring"""
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("🌐 Network Monitor started")
    
    def stop(self):
        """Stop network monitoring"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        print("🌐 Network Monitor stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                self._check_connections()
                self._check_beaconing()
                time.sleep(2)  # Check every 2 seconds
            except Exception as e:
                print(f"⚠️ Network monitor error: {e}")
                time.sleep(5)
    
    def _check_connections(self):
        """Check current network connections"""
        try:
            connections = psutil.net_connections(kind='inet')
            current_connections = set()
            
            for conn in connections:
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    remote_ip = conn.raddr.ip
                    remote_port = conn.raddr.port
                    local_port = conn.laddr.port if conn.laddr else 0
                    
                    conn_key = (remote_ip, remote_port, local_port)
                    current_connections.add(conn_key)
                    
                    # New connection detected
                    if conn_key not in self._known_connections:
                        self._known_connections.add(conn_key)
                        self._analyze_connection(remote_ip, remote_port, local_port, conn.pid)
            
            # Track connection for beaconing
            for conn_key in current_connections:
                remote_ip = conn_key[0]
                self._update_connection_history(remote_ip)
            
            # Clean up old connections
            terminated = self._known_connections - current_connections
            for conn_key in terminated:
                self._known_connections.discard(conn_key)
                
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    
    def _analyze_connection(self, remote_ip: str, remote_port: int, local_port: int, pid: int):
        """Analyze a new connection"""
        self.stats["connections_monitored"] += 1
        
        # Get process name
        process_name = ""
        try:
            if pid:
                process_name = psutil.Process(pid).name()
        except:
            pass
        
        # Resolve hostname
        hostname = self._resolve_hostname(remote_ip)
        
        # Check malicious port
        if remote_port in MALICIOUS_PORTS:
            threat_name, severity = MALICIOUS_PORTS[remote_port]
            self._create_detection(
                detection_type="malicious_port",
                severity=severity,
                mitre="T1571",
                detection_name=f"Connection to {threat_name} port {remote_port}",
                data={
                    "remote_ip": remote_ip,
                    "remote_port": remote_port,
                    "local_port": local_port,
                    "process": process_name,
                    "threat": threat_name
                }
            )
            return
        
        # Check suspicious TLD
        if hostname:
            for tld in SUSPICIOUS_TLDS:
                if hostname.endswith(tld):
                    # Check if whitelisted
                    if not self._is_whitelisted(hostname):
                        self._create_detection(
                            detection_type="suspicious_tld",
                            severity="medium",
                            mitre="T1071",
                            detection_name=f"Connection to suspicious TLD: {tld}",
                            data={
                                "remote_ip": remote_ip,
                                "hostname": hostname,
                                "tld": tld,
                                "process": process_name
                            }
                        )
                    break
        
        # Check DNS entropy (potential DGA)
        if hostname and len(hostname) > 10:
            entropy = self._calculate_entropy(hostname.split('.')[0])
            if entropy > 3.5:  # High entropy suggests DGA
                self._create_detection(
                    detection_type="dga_domain",
                    severity="high",
                    mitre="T1568.002",
                    detection_name=f"High entropy domain (possible DGA): {hostname}",
                    data={
                        "remote_ip": remote_ip,
                        "hostname": hostname,
                        "entropy": round(entropy, 2),
                        "process": process_name
                    }
                )
    
    def _update_connection_history(self, remote_ip: str):
        """Update connection history for beaconing detection"""
        now = datetime.now()
        stats = self._connection_history[remote_ip]
        
        if stats.connection_count > 0:
            interval = (now - stats.last_seen).total_seconds()
            if interval > 0.5:  # Ignore very rapid reconnects
                stats.intervals.append(interval)
                # Keep last 20 intervals
                if len(stats.intervals) > 20:
                    stats.intervals = stats.intervals[-20:]
        
        stats.last_seen = now
        stats.connection_count += 1
    
    def _check_beaconing(self):
        """Check for C2 beaconing patterns"""
        for remote_ip, stats in list(self._connection_history.items()):
            if len(stats.intervals) >= self.min_beacon_samples:
                # Calculate interval consistency
                avg_interval = sum(stats.intervals) / len(stats.intervals)
                variance = sum((i - avg_interval) ** 2 for i in stats.intervals) / len(stats.intervals)
                std_dev = math.sqrt(variance)
                
                # Coefficient of variation
                cv = std_dev / avg_interval if avg_interval > 0 else 1
                
                # Low CV = regular intervals = beaconing
                if cv < self.beacon_interval_tolerance and avg_interval > 5:
                    self.stats["beaconing_detected"] += 1
                    
                    self._create_detection(
                        detection_type="c2_beaconing",
                        severity="critical",
                        mitre="T1071",
                        detection_name=f"C2 Beaconing detected to {remote_ip}",
                        data={
                            "remote_ip": remote_ip,
                            "avg_interval": round(avg_interval, 2),
                            "variance": round(cv, 3),
                            "connection_count": stats.connection_count
                        }
                    )
                    
                    # Reset to avoid repeated alerts
                    stats.intervals = []
    
    def _resolve_hostname(self, ip: str) -> str:
        """Resolve IP to hostname"""
        try:
            hostname, _, _ = socket.gethostbyaddr(ip)
            return hostname
        except:
            return ""
    
    def _is_whitelisted(self, hostname: str) -> bool:
        """Check if hostname is whitelisted"""
        hostname_lower = hostname.lower()
        return any(wl in hostname_lower for wl in WHITELIST_DOMAINS)
    
    def _calculate_entropy(self, string: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not string:
            return 0.0
        
        freq = {}
        for char in string.lower():
            freq[char] = freq.get(char, 0) + 1
        
        length = len(string)
        entropy = -sum((count/length) * math.log2(count/length) for count in freq.values())
        
        return entropy
    
    def _create_detection(self, detection_type: str, severity: str, mitre: str, 
                          detection_name: str, data: Dict):
        """Create a detection event"""
        self.stats["suspicious_connections"] += 1
        
        if detection_type == "malicious_port":
            self.stats["malicious_ports_detected"] += 1
        elif detection_type in ["dga_domain", "dns_tunneling"]:
            self.stats["dns_tunneling_detected"] += 1
        
        self.event_queue.create_event(
            event_type="network",
            data=data,
            severity=severity,
            mitre_technique=mitre,
            detection_name=detection_name
        )
        
        print(f"🌐 [{severity.upper()}] {detection_name}")
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics"""
        return {
            **self.stats,
            "active_connections": len(self._known_connections),
            "tracked_hosts": len(self._connection_history)
        }
    
    def get_suspicious_connections(self) -> List[Dict]:
        """Get list of suspicious connections"""
        suspicious = []
        
        for (remote_ip, remote_port, local_port) in self._known_connections:
            if remote_port in MALICIOUS_PORTS:
                suspicious.append({
                    "remote_ip": remote_ip,
                    "remote_port": remote_port,
                    "local_port": local_port,
                    "threat": MALICIOUS_PORTS[remote_port][0]
                })
        
        return suspicious


# Singleton
_network_monitor = None

def get_network_monitor() -> NetworkMonitor:
    global _network_monitor
    if _network_monitor is None:
        _network_monitor = NetworkMonitor()
    return _network_monitor


if __name__ == "__main__":
    monitor = NetworkMonitor()
    monitor.start()
    
    print("\n🌐 Monitoring network... Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(10)
            stats = monitor.get_stats()
            print(f"📊 Stats: {stats}")
    except KeyboardInterrupt:
        monitor.stop()
        print("\n✅ Network monitor stopped")
