"""
PCDS Feature Extractor
Extracts ML features from network flows for threat detection
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np

try:
    from .capture import NetworkFlow
except ImportError:
    from capture import NetworkFlow

logger = logging.getLogger(__name__)


@dataclass
class FlowFeatures:
    """ML-ready features extracted from a network flow"""
    flow_id: str
    timestamp: str
    
    # Basic features
    duration: float  # seconds
    protocol_type: int  # 0=TCP, 1=UDP, 2=ICMP
    service: int  # Encoded service type
    
    # Byte features
    src_bytes: int
    dst_bytes: int
    bytes_ratio: float  # src/dst ratio
    
    # Packet features
    src_packets: int
    dst_packets: int
    packet_ratio: float
    
    # Connection features
    flag_syn: int
    flag_ack: int
    flag_fin: int
    flag_rst: int
    flag_psh: int
    
    # Rate features
    bytes_per_second: float
    packets_per_second: float
    
    # Behavioral features
    is_from_internal: int  # 1 if src is internal IP
    uses_uncommon_port: int  # 1 if dst_port is uncommon
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML model"""
        return np.array([
            self.duration,
            self.protocol_type,
            self.service,
            self.src_bytes,
            self.dst_bytes,
            self.bytes_ratio,
            self.src_packets,
            self.dst_packets,
            self.packet_ratio,
            self.flag_syn,
            self.flag_ack,
            self.flag_fin,
            self.flag_rst,
            self.flag_psh,
            self.bytes_per_second,
            self.packets_per_second,
            self.is_from_internal,
            self.uses_uncommon_port
        ], dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API"""
        return {
            "flow_id": self.flow_id,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "protocol_type": self.protocol_type,
            "service": self.service,
            "src_bytes": self.src_bytes,
            "dst_bytes": self.dst_bytes,
            "bytes_ratio": self.bytes_ratio,
            "src_packets": self.src_packets,
            "dst_packets": self.dst_packets,
            "packet_ratio": self.packet_ratio,
            "flag_syn": self.flag_syn,
            "flag_ack": self.flag_ack,
            "flag_fin": self.flag_fin,
            "flag_rst": self.flag_rst,
            "flag_psh": self.flag_psh,
            "bytes_per_second": self.bytes_per_second,
            "packets_per_second": self.packets_per_second,
            "is_from_internal": self.is_from_internal,
            "uses_uncommon_port": self.uses_uncommon_port
        }


class FeatureExtractor:
    """
    Extracts ML features from network flows
    Compatible with PCDS ML models
    """
    
    # Service encoding
    SERVICE_MAP = {
        "HTTP": 0,
        "HTTPS": 1,
        "DNS": 2,
        "SSH": 3,
        "FTP": 4,
        "SMTP": 5,
        "RDP": 6,
        "NTP": 7,
        "UNKNOWN": 99
    }
    
    # Uncommon ports (often used in attacks)
    UNCOMMON_PORTS = {
        4444,   # Metasploit default
        5555,   # Android Debug
        6666,   # Malware
        7777,   # Malware
        8080,   # Alternative HTTP
        8443,   # Alternative HTTPS
        9001,   # Tor
        9002,   # Tor
        31337,  # Elite/Backdoor
    }
    
    # Internal IP ranges
    INTERNAL_RANGES = [
        ("10.0.0.0", "10.255.255.255"),
        ("172.16.0.0", "172.31.255.255"),
        ("192.168.0.0", "192.168.255.255"),
    ]
    
    def __init__(self):
        logger.info("FeatureExtractor initialized")
    
    def _is_internal_ip(self, ip: str) -> bool:
        """Check if IP is internal/private"""
        try:
            parts = [int(p) for p in ip.split(".")]
            ip_int = (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
            
            for start, end in self.INTERNAL_RANGES:
                start_parts = [int(p) for p in start.split(".")]
                end_parts = [int(p) for p in end.split(".")]
                start_int = (start_parts[0] << 24) + (start_parts[1] << 16) + (start_parts[2] << 8) + start_parts[3]
                end_int = (end_parts[0] << 24) + (end_parts[1] << 16) + (end_parts[2] << 8) + end_parts[3]
                
                if start_int <= ip_int <= end_int:
                    return True
            return False
        except:
            return False
    
    def _encode_protocol(self, protocol: str) -> int:
        """Encode protocol to numeric"""
        mapping = {"TCP": 0, "UDP": 1, "ICMP": 2}
        return mapping.get(protocol, 3)
    
    def _encode_service(self, app_protocol: Optional[str]) -> int:
        """Encode application protocol to numeric"""
        if not app_protocol:
            return self.SERVICE_MAP["UNKNOWN"]
        return self.SERVICE_MAP.get(app_protocol, self.SERVICE_MAP["UNKNOWN"])
    
    def extract(self, flow: NetworkFlow) -> FlowFeatures:
        """Extract features from a network flow"""
        
        # Calculate duration
        duration = 0.0
        if flow.end_time and flow.start_time:
            duration = (flow.end_time - flow.start_time).total_seconds()
        
        # Byte ratio (avoid division by zero)
        bytes_ratio = 0.0
        if flow.bytes_recv > 0:
            bytes_ratio = flow.bytes_sent / flow.bytes_recv
        elif flow.bytes_sent > 0:
            bytes_ratio = float('inf')
        
        # Packet ratio
        packet_ratio = 0.0
        if flow.packets_recv > 0:
            packet_ratio = flow.packets_sent / flow.packets_recv
        elif flow.packets_sent > 0:
            packet_ratio = float('inf')
        
        # Rate features
        total_bytes = flow.bytes_sent + flow.bytes_recv
        total_packets = flow.packets_sent + flow.packets_recv
        
        bytes_per_second = total_bytes / duration if duration > 0 else 0.0
        packets_per_second = total_packets / duration if duration > 0 else 0.0
        
        # Flag extraction
        flags = set(flow.flags)
        
        return FlowFeatures(
            flow_id=flow.flow_id,
            timestamp=flow.start_time.isoformat(),
            duration=duration,
            protocol_type=self._encode_protocol(flow.protocol),
            service=self._encode_service(flow.app_protocol),
            src_bytes=flow.bytes_sent,
            dst_bytes=flow.bytes_recv,
            bytes_ratio=min(bytes_ratio, 1000.0),  # Cap extreme values
            src_packets=flow.packets_sent,
            dst_packets=flow.packets_recv,
            packet_ratio=min(packet_ratio, 1000.0),
            flag_syn=1 if "SYN" in flags else 0,
            flag_ack=1 if "ACK" in flags else 0,
            flag_fin=1 if "FIN" in flags else 0,
            flag_rst=1 if "RST" in flags else 0,
            flag_psh=1 if "PSH" in flags else 0,
            bytes_per_second=bytes_per_second,
            packets_per_second=packets_per_second,
            is_from_internal=1 if self._is_internal_ip(flow.src_ip) else 0,
            uses_uncommon_port=1 if flow.dst_port in self.UNCOMMON_PORTS else 0
        )
    
    def extract_batch(self, flows: List[NetworkFlow]) -> List[FlowFeatures]:
        """Extract features from multiple flows"""
        return [self.extract(flow) for flow in flows]
    
    def to_model_input(self, features: List[FlowFeatures]) -> np.ndarray:
        """Convert features to model-ready numpy array"""
        return np.vstack([f.to_array() for f in features])


# Feature names for model training reference
FEATURE_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "src_bytes",
    "dst_bytes",
    "bytes_ratio",
    "src_packets",
    "dst_packets",
    "packet_ratio",
    "flag_syn",
    "flag_ack",
    "flag_fin",
    "flag_rst",
    "flag_psh",
    "bytes_per_second",
    "packets_per_second",
    "is_from_internal",
    "uses_uncommon_port"
]
