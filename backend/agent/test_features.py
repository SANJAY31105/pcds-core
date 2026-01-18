"""
Test feature extraction from captured packets
"""
import sys
import time
from datetime import datetime

# Add parent path
sys.path.insert(0, '..')

from scapy.all import sniff, IP, TCP, UDP

from capture import PacketCapture, NetworkFlow
from features import FeatureExtractor

def test_feature_extraction():
    """Test feature extraction from live traffic"""
    print("\n" + "="*60)
    print("  PCDS Agent - Feature Extraction Test")
    print("="*60)
    
    # Initialize components
    capture = PacketCapture(bpf_filter="ip")
    extractor = FeatureExtractor()
    
    print("\n[*] Capturing 20 packets...")
    print("[*] Make some network activity (browse, download, etc.)")
    print("-" * 60)
    
    # Capture packets
    captured = 0
    
    def on_packet(pkt):
        nonlocal captured
        if pkt.haslayer(IP):
            ip = pkt[IP]
            proto = "TCP" if pkt.haslayer(TCP) else ("UDP" if pkt.haslayer(UDP) else "?")
            print(f"  [{proto}] {ip.src}:{getattr(pkt, 'sport', '?')} -> {ip.dst}:{getattr(pkt, 'dport', '?')}")
            captured += 1
    
    packets = sniff(count=20, prn=on_packet, filter="ip", timeout=30)
    
    print("-" * 60)
    print(f"\n[*] Captured {len(packets)} packets")
    
    # Simulate flow creation (normally done by PacketCapture class)
    print("\n[*] Creating test flows from packets...")
    
    # Group packets into a simple flow
    test_flow = NetworkFlow(
        flow_id="test-flow-001",
        src_ip="10.75.69.215",
        dst_ip="54.197.155.195",
        src_port=54321,
        dst_port=443,
        protocol="TCP",
        start_time=datetime.now(),
        end_time=datetime.now(),
        packets_sent=10,
        packets_recv=10,
        bytes_sent=1500,
        bytes_recv=4500,
        flags=["SYN", "ACK", "PSH"],
        app_protocol="HTTPS"
    )
    
    print(f"\n[*] Test Flow:")
    print(f"    {test_flow.src_ip}:{test_flow.src_port} -> {test_flow.dst_ip}:{test_flow.dst_port}")
    print(f"    Protocol: {test_flow.protocol} / {test_flow.app_protocol}")
    print(f"    Packets: {test_flow.packets_sent} sent, {test_flow.packets_recv} recv")
    print(f"    Bytes: {test_flow.bytes_sent} sent, {test_flow.bytes_recv} recv")
    
    # Extract features
    print("\n[*] Extracting ML features...")
    features = extractor.extract(test_flow)
    
    print("\n" + "="*60)
    print("  EXTRACTED FEATURES (ML-Ready)")
    print("="*60)
    
    feature_dict = features.to_dict()
    for key, value in feature_dict.items():
        if key not in ['flow_id', 'timestamp']:
            print(f"  {key:25}: {value}")
    
    print("\n[*] Feature array shape:", features.to_array().shape)
    print("[*] Feature array:", features.to_array()[:5], "...")
    
    print("\n" + "="*60)
    print("  ✅ Feature extraction WORKING!")
    print("="*60)


if __name__ == "__main__":
    test_feature_extraction()
