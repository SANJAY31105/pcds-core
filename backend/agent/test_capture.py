"""Quick test to capture 10 packets"""
from scapy.all import sniff, IP, TCP, UDP

def packet_callback(pkt):
    if pkt.haslayer(IP):
        ip = pkt[IP]
        proto = "TCP" if pkt.haslayer(TCP) else ("UDP" if pkt.haslayer(UDP) else "OTHER")
        print(f"[{proto}] {ip.src} -> {ip.dst}")

print("Capturing 10 packets... (make some network activity)")
print("-" * 50)

# Capture 10 packets
packets = sniff(count=10, prn=packet_callback, filter="ip", timeout=30)

print("-" * 50)
print(f"Captured {len(packets)} packets")
