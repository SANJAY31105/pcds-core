"""
Data generator for demonstration and testing
"""
import random
import asyncio
from datetime import datetime, timedelta
import uuid
from models import NetworkEvent, ThreatSeverity
from typing import List


class DataGenerator:
    """Generate realistic network events and threat scenarios"""
    
    def __init__(self):
        self.protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS']
        self.normal_ports = [80, 443, 22, 21, 25, 110, 143, 993, 995]
        self.suspicious_ports = [23, 135, 139, 445, 3389, 5900, 1433, 3306]
        self.malicious_ips = [
            '198.51.100.42',
            '203.0.113.89',
            '192.0.2.156',
            '198.51.100.200'
        ]
        self.normal_ips = [
            f'10.0.{random.randint(1, 255)}.{random.randint(1, 255)}'
            for _ in range(10)
        ]
        
    def generate_normal_event(self) -> NetworkEvent:
        """Generate normal network traffic"""
        return NetworkEvent(
            timestamp=datetime.utcnow(),
            source_ip=random.choice(self.normal_ips),
            destination_ip=random.choice(self.normal_ips),
            port=random.choice(self.normal_ports),
            protocol=random.choice(self.protocols[:2]),
            packet_size=random.randint(64, 1200),
            flags='ACK' if random.random() > 0.5 else 'SYN,ACK'
        )
        
    def generate_suspicious_event(self) -> NetworkEvent:
        """Generate suspicious network activity"""
        return NetworkEvent(
            timestamp=datetime.utcnow(),
            source_ip=random.choice(self.malicious_ips),
            destination_ip=random.choice(self.normal_ips),
            port=random.choice(self.suspicious_ports),
            protocol=random.choice(self.protocols),
            packet_size=random.randint(1200, 1500),
            flags='SYN' if random.random() > 0.5 else 'FIN'
        )
        
    def generate_attack_scenario(self, attack_type: str) -> List[NetworkEvent]:
        """Generate specific attack scenarios"""
        events = []
        
        if attack_type == 'port_scan':
            # Sequential port scanning
            source = random.choice(self.malicious_ips)
            target = random.choice(self.normal_ips)
            for port in range(20, 30):
                events.append(NetworkEvent(
                    timestamp=datetime.utcnow(),
                    source_ip=source,
                    destination_ip=target,
                    port=port,
                    protocol='TCP',
                    packet_size=64,
                    flags='SYN'
                ))
                
        elif attack_type == 'ddos':
            # Distributed denial of service
            target = random.choice(self.normal_ips)
            for _ in range(50):
                events.append(NetworkEvent(
                    timestamp=datetime.utcnow(),
                    source_ip=random.choice(self.malicious_ips),
                    destination_ip=target,
                    port=80,
                    protocol='TCP',
                    packet_size=random.randint(1400, 1500),
                    flags='SYN'
                ))
                
        elif attack_type == 'data_exfiltration':
            # Large data transfer
            source = random.choice(self.normal_ips)
            destination = random.choice(self.malicious_ips)
            for _ in range(20):
                events.append(NetworkEvent(
                    timestamp=datetime.utcnow(),
                    source_ip=source,
                    destination_ip=destination,
                    port=443,
                    protocol='HTTPS',
                    packet_size=1500,
                    flags='PSH,ACK'
                ))
                
        return events
        
    def generate_mixed_traffic(self, 
                               normal_count: int = 10, 
                               suspicious_count: int = 2) -> List[NetworkEvent]:
        """Generate mixed normal and suspicious traffic"""
        events = []
        
        for _ in range(normal_count):
            events.append(self.generate_normal_event())
            
        for _ in range(suspicious_count):
            events.append(self.generate_suspicious_event())
            
        # Shuffle to mix normal and suspicious
        random.shuffle(events)
        return events
        
    async def continuous_generation(self, callback, interval: float = 2.0):
        """Continuously generate events and call callback"""
        while True:
            # 80% normal, 20% suspicious
            if random.random() < 0.8:
                event = self.generate_normal_event()
            else:
                event = self.generate_suspicious_event()
                
            await callback(event)
            await asyncio.sleep(interval + random.uniform(-0.5, 0.5))


# Global data generator instance
data_generator = DataGenerator()
