"""
PCDS Network Capture Module
Captures network traffic from SPAN port using Scapy
"""

import logging
import threading
import queue
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib

try:
    from scapy.all import sniff, IP, TCP, UDP, DNS, Raw, Ether
    from scapy.layers.http import HTTPRequest, HTTPResponse
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("WARNING: Scapy not installed. Run: pip install scapy")

logger = logging.getLogger(__name__)


@dataclass
class NetworkFlow:
    """Represents a network connection/flow"""
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str  # TCP, UDP, ICMP
    start_time: datetime
    end_time: Optional[datetime] = None
    packets_sent: int = 0
    packets_recv: int = 0
    bytes_sent: int = 0
    bytes_recv: int = 0
    flags: List[str] = field(default_factory=list)
    app_protocol: Optional[str] = None  # HTTP, DNS, TLS, etc.
    payload_sample: Optional[bytes] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "flow_id": self.flow_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "src_port": self.src_port,
            "dst_port": self.dst_port,
            "protocol": self.protocol,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "packets_sent": self.packets_sent,
            "packets_recv": self.packets_recv,
            "bytes_sent": self.bytes_sent,
            "bytes_recv": self.bytes_recv,
            "flags": self.flags,
            "app_protocol": self.app_protocol,
            "metadata": self.metadata
        }


class PacketCapture:
    """
    High-performance packet capture engine
    Designed for SPAN port traffic mirroring
    """
    
    def __init__(
        self,
        interface: str = None,
        bpf_filter: str = None,
        packet_callback: Optional[Callable] = None,
        max_queue_size: int = 10000
    ):
        self.interface = interface
        self.bpf_filter = bpf_filter or "ip"  # Capture all IP traffic
        self.packet_callback = packet_callback
        self.max_queue_size = max_queue_size
        
        # Packet queue for processing
        self.packet_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        
        # Flow tracking
        self.active_flows: Dict[str, NetworkFlow] = {}
        self.completed_flows: queue.Queue = queue.Queue()
        
        # Statistics
        self.stats = {
            "packets_captured": 0,
            "packets_dropped": 0,
            "bytes_captured": 0,
            "flows_created": 0,
            "errors": 0
        }
        
        # Control flags
        self._running = False
        self._capture_thread: Optional[threading.Thread] = None
        self._processor_thread: Optional[threading.Thread] = None
        
        logger.info(f"PacketCapture initialized on interface: {interface}")
    
    def _generate_flow_id(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, protocol: str) -> str:
        """Generate unique flow identifier"""
        # Normalize flow (smaller IP first for bidirectional matching)
        if src_ip < dst_ip or (src_ip == dst_ip and src_port < dst_port):
            key = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        else:
            key = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        return hashlib.md5(key.encode()).hexdigest()[:16]
    
    def _parse_packet(self, packet) -> Optional[Dict[str, Any]]:
        """Parse raw packet into structured data"""
        try:
            if not packet.haslayer(IP):
                return None
            
            ip_layer = packet[IP]
            
            parsed = {
                "timestamp": datetime.now(),
                "src_ip": ip_layer.src,
                "dst_ip": ip_layer.dst,
                "protocol": "UNKNOWN",
                "src_port": 0,
                "dst_port": 0,
                "length": len(packet),
                "flags": [],
                "app_protocol": None,
                "payload": None
            }
            
            # TCP
            if packet.haslayer(TCP):
                tcp = packet[TCP]
                parsed["protocol"] = "TCP"
                parsed["src_port"] = tcp.sport
                parsed["dst_port"] = tcp.dport
                parsed["flags"] = self._get_tcp_flags(tcp)
                
                # Detect application protocol
                if tcp.dport == 80 or tcp.sport == 80:
                    parsed["app_protocol"] = "HTTP"
                elif tcp.dport == 443 or tcp.sport == 443:
                    parsed["app_protocol"] = "HTTPS"
                elif tcp.dport == 22 or tcp.sport == 22:
                    parsed["app_protocol"] = "SSH"
                elif tcp.dport == 21 or tcp.sport == 21:
                    parsed["app_protocol"] = "FTP"
                elif tcp.dport == 25 or tcp.sport == 25:
                    parsed["app_protocol"] = "SMTP"
                elif tcp.dport == 3389 or tcp.sport == 3389:
                    parsed["app_protocol"] = "RDP"
                    
            # UDP
            elif packet.haslayer(UDP):
                udp = packet[UDP]
                parsed["protocol"] = "UDP"
                parsed["src_port"] = udp.sport
                parsed["dst_port"] = udp.dport
                
                if packet.haslayer(DNS):
                    parsed["app_protocol"] = "DNS"
                    dns = packet[DNS]
                    if dns.qr == 0:  # Query
                        parsed["dns_query"] = dns.qd.qname.decode() if dns.qd else None
                elif udp.dport == 53 or udp.sport == 53:
                    parsed["app_protocol"] = "DNS"
                elif udp.dport == 123 or udp.sport == 123:
                    parsed["app_protocol"] = "NTP"
            
            # Extract payload sample (first 100 bytes)
            if packet.haslayer(Raw):
                parsed["payload"] = bytes(packet[Raw].load[:100])
            
            return parsed
            
        except Exception as e:
            logger.error(f"Packet parse error: {e}")
            self.stats["errors"] += 1
            return None
    
    def _get_tcp_flags(self, tcp) -> List[str]:
        """Extract TCP flags"""
        flags = []
        if tcp.flags.S: flags.append("SYN")
        if tcp.flags.A: flags.append("ACK")
        if tcp.flags.F: flags.append("FIN")
        if tcp.flags.R: flags.append("RST")
        if tcp.flags.P: flags.append("PSH")
        if tcp.flags.U: flags.append("URG")
        return flags
    
    def _update_flow(self, parsed: Dict[str, Any]):
        """Update or create flow from parsed packet"""
        flow_id = self._generate_flow_id(
            parsed["src_ip"],
            parsed["dst_ip"],
            parsed["src_port"],
            parsed["dst_port"],
            parsed["protocol"]
        )
        
        if flow_id not in self.active_flows:
            # Create new flow
            flow = NetworkFlow(
                flow_id=flow_id,
                src_ip=parsed["src_ip"],
                dst_ip=parsed["dst_ip"],
                src_port=parsed["src_port"],
                dst_port=parsed["dst_port"],
                protocol=parsed["protocol"],
                start_time=parsed["timestamp"],
                app_protocol=parsed["app_protocol"]
            )
            self.active_flows[flow_id] = flow
            self.stats["flows_created"] += 1
        
        flow = self.active_flows[flow_id]
        
        # Update flow stats
        if parsed["src_ip"] == flow.src_ip:
            flow.packets_sent += 1
            flow.bytes_sent += parsed["length"]
        else:
            flow.packets_recv += 1
            flow.bytes_recv += parsed["length"]
        
        flow.end_time = parsed["timestamp"]
        
        # Update flags
        for flag in parsed.get("flags", []):
            if flag not in flow.flags:
                flow.flags.append(flag)
        
        # Check for flow completion (FIN or RST)
        if "FIN" in parsed.get("flags", []) or "RST" in parsed.get("flags", []):
            self.completed_flows.put(flow)
            del self.active_flows[flow_id]
    
    def _packet_handler(self, packet):
        """Callback for each captured packet"""
        try:
            self.stats["packets_captured"] += 1
            self.stats["bytes_captured"] += len(packet)
            
            # Try to add to queue, drop if full
            try:
                self.packet_queue.put_nowait(packet)
            except queue.Full:
                self.stats["packets_dropped"] += 1
                
        except Exception as e:
            logger.error(f"Packet handler error: {e}")
            self.stats["errors"] += 1
    
    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        logger.info(f"Starting capture on {self.interface}")
        
        try:
            sniff(
                iface=self.interface,
                filter=self.bpf_filter,
                prn=self._packet_handler,
                store=False,
                stop_filter=lambda x: not self._running
            )
        except Exception as e:
            logger.error(f"Capture error: {e}")
            self._running = False
    
    def _process_loop(self):
        """Process captured packets"""
        logger.info("Starting packet processor")
        
        while self._running or not self.packet_queue.empty():
            try:
                packet = self.packet_queue.get(timeout=1.0)
                parsed = self._parse_packet(packet)
                
                if parsed:
                    self._update_flow(parsed)
                    
                    # Call external callback if provided
                    if self.packet_callback:
                        self.packet_callback(parsed)
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Process error: {e}")
    
    def start(self):
        """Start packet capture"""
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy not installed. Run: pip install scapy")
        
        if self._running:
            logger.warning("Capture already running")
            return
        
        self._running = True
        
        # Start capture thread
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        
        # Start processor thread
        self._processor_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._processor_thread.start()
        
        logger.info("Packet capture started")
    
    def stop(self):
        """Stop packet capture"""
        logger.info("Stopping packet capture...")
        self._running = False
        
        if self._capture_thread:
            self._capture_thread.join(timeout=5.0)
        if self._processor_thread:
            self._processor_thread.join(timeout=5.0)
        
        logger.info("Packet capture stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get capture statistics"""
        return {
            **self.stats,
            "active_flows": len(self.active_flows),
            "pending_packets": self.packet_queue.qsize()
        }
    
    def get_completed_flows(self) -> List[NetworkFlow]:
        """Get list of completed flows"""
        flows = []
        while not self.completed_flows.empty():
            try:
                flows.append(self.completed_flows.get_nowait())
            except queue.Empty:
                break
        return flows


# Quick test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    def on_packet(parsed):
        print(f"[{parsed['protocol']}] {parsed['src_ip']}:{parsed['src_port']} -> {parsed['dst_ip']}:{parsed['dst_port']}")
    
    # List available interfaces
    if SCAPY_AVAILABLE:
        from scapy.all import get_if_list
        print("Available interfaces:")
        for iface in get_if_list():
            print(f"  - {iface}")
    
    print("\nTo start capture, run with admin privileges and specify interface.")
