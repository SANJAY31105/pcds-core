"""
PCAP/Traffic Ingestion Connector
Ingests PCAP files and network traffic for ML pipeline processing

Features:
- PCAP file reading via scapy
- Feature extraction from packets
- Publishing to Kafka raw_events topic
- Batch processing for large PCAP files
"""

import asyncio
from typing import Dict, List, Optional, Generator, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import hashlib

# Try importing scapy
try:
    from scapy.all import rdpcap, IP, TCP, UDP, DNS, Raw
    from scapy.layers.http import HTTP, HTTPRequest
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("⚠️ scapy not installed - PCAP ingestion disabled")


@dataclass
class PacketEvent:
    """Standardized packet event for ML processing"""
    event_id: str
    timestamp: str
    source_ip: str
    dest_ip: str
    source_port: int
    dest_port: int
    protocol: str
    packet_size: int
    flags: str = ""
    payload_size: int = 0
    is_encrypted: bool = False
    dns_query: str = ""
    http_host: str = ""
    features: List[float] = None
    raw_data: Dict = None


class TrafficConnector:
    """
    PCAP/Traffic Ingestion Connector
    
    Reads PCAP files and publishes events to Kafka for ML processing.
    Can also process live packet captures.
    """
    
    def __init__(self, kafka_producer=None):
        """
        Args:
            kafka_producer: Optional Kafka producer for publishing events
        """
        self.kafka_producer = kafka_producer
        self.stats = {
            "packets_processed": 0,
            "events_published": 0,
            "errors": 0,
            "start_time": None,
            "end_time": None
        }
        
        if not SCAPY_AVAILABLE:
            print("⚠️ TrafficConnector: scapy not available, PCAP ingestion disabled")
    
    def ingest_pcap(self, pcap_path: str, batch_size: int = 100) -> Generator[List[PacketEvent], None, None]:
        """
        Read PCAP file and yield batches of packet events
        
        Args:
            pcap_path: Path to PCAP file
            batch_size: Number of packets per batch
            
        Yields:
            List of PacketEvent objects
        """
        if not SCAPY_AVAILABLE:
            print("❌ scapy not installed - cannot read PCAP")
            return
        
        self.stats["start_time"] = datetime.now().isoformat()
        print(f"📦 Reading PCAP: {pcap_path}")
        
        try:
            packets = rdpcap(pcap_path)
            print(f"   Loaded {len(packets)} packets")
            
            batch = []
            for i, pkt in enumerate(packets):
                try:
                    event = self._parse_packet(pkt, i)
                    if event:
                        batch.append(event)
                        self.stats["packets_processed"] += 1
                    
                    if len(batch) >= batch_size:
                        yield batch
                        batch = []
                        
                except Exception as e:
                    self.stats["errors"] += 1
                    continue
            
            # Yield remaining batch
            if batch:
                yield batch
                
            self.stats["end_time"] = datetime.now().isoformat()
            print(f"✅ PCAP ingestion complete: {self.stats['packets_processed']} packets")
            
        except Exception as e:
            print(f"❌ Failed to read PCAP: {e}")
            self.stats["errors"] += 1
    
    def _parse_packet(self, pkt, index: int) -> Optional[PacketEvent]:
        """Parse a scapy packet into PacketEvent"""
        if not pkt.haslayer(IP):
            return None
        
        ip_layer = pkt[IP]
        
        # Basic packet info
        source_ip = ip_layer.src
        dest_ip = ip_layer.dst
        packet_size = len(pkt)
        protocol = "other"
        source_port = 0
        dest_port = 0
        flags = ""
        payload_size = 0
        dns_query = ""
        http_host = ""
        is_encrypted = False
        
        # TCP layer
        if pkt.haslayer(TCP):
            tcp = pkt[TCP]
            protocol = "tcp"
            source_port = tcp.sport
            dest_port = tcp.dport
            flags = str(tcp.flags)
            
            # Check for TLS/HTTPS
            if dest_port == 443 or source_port == 443:
                is_encrypted = True
            
            # Payload size
            if pkt.haslayer(Raw):
                payload_size = len(pkt[Raw].load)
        
        # UDP layer
        elif pkt.haslayer(UDP):
            udp = pkt[UDP]
            protocol = "udp"
            source_port = udp.sport
            dest_port = udp.dport
            
            if pkt.haslayer(Raw):
                payload_size = len(pkt[Raw].load)
        
        # DNS queries
        if pkt.haslayer(DNS):
            dns = pkt[DNS]
            if dns.qr == 0 and dns.qd:  # Query
                dns_query = dns.qd.qname.decode() if hasattr(dns.qd, 'qname') else ""
        
        # HTTP (if available)
        try:
            if pkt.haslayer(HTTPRequest):
                http_host = pkt[HTTPRequest].Host.decode() if pkt[HTTPRequest].Host else ""
        except:
            pass
        
        # Generate event ID
        event_id = hashlib.md5(f"{source_ip}{dest_ip}{source_port}{dest_port}{index}".encode()).hexdigest()[:16]
        
        # Extract ML features
        features = self._extract_features(
            source_ip, dest_ip, source_port, dest_port,
            protocol, packet_size, payload_size, flags, is_encrypted
        )
        
        return PacketEvent(
            event_id=event_id,
            timestamp=datetime.now().isoformat(),
            source_ip=source_ip,
            dest_ip=dest_ip,
            source_port=source_port,
            dest_port=dest_port,
            protocol=protocol,
            packet_size=packet_size,
            flags=flags,
            payload_size=payload_size,
            is_encrypted=is_encrypted,
            dns_query=dns_query,
            http_host=http_host,
            features=features,
            raw_data={"index": index}
        )
    
    def _extract_features(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int,
                          protocol: str, pkt_size: int, payload_size: int, 
                          flags: str, is_encrypted: bool) -> List[float]:
        """
        Extract 40 ML features from packet data
        Matches the feature vector expected by the ensemble detector
        """
        features = []
        
        # Feature 1-4: IP-based features
        src_octets = [int(x) for x in src_ip.split('.')] if '.' in src_ip else [0, 0, 0, 0]
        features.extend([o / 255.0 for o in src_octets])
        
        # Feature 5-8: Destination IP octets
        dst_octets = [int(x) for x in dst_ip.split('.')] if '.' in dst_ip else [0, 0, 0, 0]
        features.extend([o / 255.0 for o in dst_octets])
        
        # Feature 9-10: Port features (normalized)
        features.append(src_port / 65535.0)
        features.append(dst_port / 65535.0)
        
        # Feature 11-14: Protocol encoding (one-hot)
        features.append(1.0 if protocol == "tcp" else 0.0)
        features.append(1.0 if protocol == "udp" else 0.0)
        features.append(1.0 if protocol == "icmp" else 0.0)
        features.append(1.0 if protocol == "other" else 0.0)
        
        # Feature 15-18: Packet size features
        features.append(min(pkt_size / 1500.0, 1.0))  # Normalized to MTU
        features.append(min(payload_size / 1500.0, 1.0))
        features.append(1.0 if pkt_size > 1400 else 0.0)  # Large packet
        features.append(payload_size / max(pkt_size, 1))  # Payload ratio
        
        # Feature 19-22: TCP flags (if available)
        features.append(1.0 if 'S' in flags else 0.0)  # SYN
        features.append(1.0 if 'A' in flags else 0.0)  # ACK
        features.append(1.0 if 'F' in flags else 0.0)  # FIN
        features.append(1.0 if 'R' in flags else 0.0)  # RST
        
        # Feature 23-26: Connection characteristics
        features.append(1.0 if is_encrypted else 0.0)
        features.append(1.0 if dst_port in [21, 22, 23, 25, 110, 143, 445, 3389] else 0.0)  # Sensitive ports
        features.append(1.0 if dst_port in [80, 443, 8080, 8443] else 0.0)  # Web ports
        features.append(1.0 if src_port > 1024 else 0.0)  # Ephemeral source
        
        # Feature 27-30: Suspicious indicators
        features.append(1.0 if dst_port in [4444, 5555, 6666, 1337, 31337] else 0.0)  # Known RAT ports
        features.append(1.0 if dst_port in [3389, 5900, 5901] else 0.0)  # Remote access
        features.append(1.0 if dst_port == 53 and pkt_size > 512 else 0.0)  # DNS tunneling indicator
        features.append(1.0 if 'P' in flags else 0.0)  # PSH flag (data push)
        
        # Feature 31-34: Additional timing/behavior features (placeholders for real-time)
        features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Feature 35-38: Reserved for aggregated features
        features.extend([0.0, 0.0, 0.0, 0.0])
        
        # Feature 39-40: Entropy/randomness features (placeholder)
        features.extend([0.5, 0.5])
        
        return features
    
    async def publish_to_kafka(self, events: List[PacketEvent], topic: str = "raw_events"):
        """
        Publish packet events to Kafka
        
        Args:
            events: List of PacketEvent objects
            topic: Kafka topic name
        """
        if not self.kafka_producer:
            print("⚠️ No Kafka producer configured")
            return
        
        for event in events:
            try:
                await self.kafka_producer.send(
                    topic=topic,
                    value=asdict(event),
                    key=event.source_ip
                )
                self.stats["events_published"] += 1
            except Exception as e:
                self.stats["errors"] += 1
                print(f"❌ Failed to publish event: {e}")
    
    async def ingest_and_publish(self, pcap_path: str, batch_size: int = 100):
        """
        Read PCAP and publish all events to Kafka
        
        Args:
            pcap_path: Path to PCAP file
            batch_size: Batch size for processing
        """
        for batch in self.ingest_pcap(pcap_path, batch_size):
            await self.publish_to_kafka(batch)
            print(f"   Published batch: {len(batch)} events")
    
    def get_stats(self) -> Dict:
        """Get ingestion statistics"""
        return self.stats


# Singleton instance
_traffic_connector: Optional[TrafficConnector] = None


def get_traffic_connector() -> TrafficConnector:
    """Get or create traffic connector instance"""
    global _traffic_connector
    if _traffic_connector is None:
        try:
            from kafka.producer import kafka_producer
            _traffic_connector = TrafficConnector(kafka_producer)
        except ImportError:
            _traffic_connector = TrafficConnector()
    return _traffic_connector


# CLI for testing
if __name__ == "__main__":
    import sys
    
    print("\n🔌 PCAP Traffic Connector Test\n")
    
    if len(sys.argv) > 1:
        pcap_file = sys.argv[1]
        connector = TrafficConnector()
        
        packet_count = 0
        for batch in connector.ingest_pcap(pcap_file, batch_size=50):
            packet_count += len(batch)
            print(f"   Batch: {len(batch)} packets, Total: {packet_count}")
            
            # Show sample packet
            if batch:
                sample = batch[0]
                print(f"   Sample: {sample.source_ip}:{sample.source_port} → {sample.dest_ip}:{sample.dest_port} ({sample.protocol})")
        
        print(f"\n📊 Stats: {connector.get_stats()}")
    else:
        print("Usage: python traffic_connector.py <pcap_file>")
        print("\nExample:")
        print("  python traffic_connector.py capture.pcap")
        
        if not SCAPY_AVAILABLE:
            print("\n⚠️ Install scapy: pip install scapy")
