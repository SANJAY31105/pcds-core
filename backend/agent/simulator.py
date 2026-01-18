"""
PCDS Attack Simulator
Generates various attack patterns for testing the agent
"""

import socket
import time
import threading
import random
import argparse
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AttackSimulator:
    """
    Simulates various network attacks for testing PCDS agent
    WARNING: Only use on your own network for testing!
    """
    
    def __init__(self, target_ip: str = "127.0.0.1"):
        self.target_ip = target_ip
        self._running = False
    
    def port_scan(self, ports: List[int] = None, delay: float = 0.1):
        """Simulate a port scan attack"""
        if ports is None:
            ports = list(range(1, 1025))  # First 1024 ports
        
        print(f"\n[PORT SCAN] Scanning {len(ports)} ports on {self.target_ip}...")
        
        open_ports = []
        for port in ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                result = sock.connect_ex((self.target_ip, port))
                if result == 0:
                    open_ports.append(port)
                    print(f"  [OPEN] Port {port}")
                sock.close()
                time.sleep(delay)
            except Exception as e:
                pass
        
        print(f"[PORT SCAN] Complete. Found {len(open_ports)} open ports")
        return open_ports
    
    def syn_flood(self, port: int = 80, count: int = 100, delay: float = 0.01):
        """Simulate SYN flood (connection attempts)"""
        print(f"\n[SYN FLOOD] Sending {count} SYN packets to {self.target_ip}:{port}...")
        
        for i in range(count):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.1)
                sock.connect_ex((self.target_ip, port))
                sock.close()
                time.sleep(delay)
            except:
                pass
        
        print(f"[SYN FLOOD] Complete - {count} packets sent")
    
    def brute_force_simulation(self, port: int = 22, attempts: int = 50):
        """Simulate brute force login attempts"""
        print(f"\n[BRUTE FORCE] Simulating {attempts} login attempts on port {port}...")
        
        for i in range(attempts):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.5)
                sock.connect_ex((self.target_ip, port))
                # Simulate sending credentials
                sock.send(b"USER admin\r\n")
                time.sleep(0.05)
                sock.send(b"PASS password123\r\n")
                sock.close()
            except:
                pass
        
        print(f"[BRUTE FORCE] Complete - {attempts} attempts simulated")
    
    def data_exfil_simulation(self, port: int = 443, data_size: int = 100000):
        """Simulate data exfiltration (large outbound transfer)"""
        print(f"\n[DATA EXFIL] Simulating {data_size} bytes upload to {self.target_ip}:{port}...")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect_ex((self.target_ip, port))
            
            # Send large data
            data = b"X" * data_size
            sock.send(data)
            sock.close()
            print(f"[DATA EXFIL] Complete - {data_size} bytes sent")
        except Exception as e:
            print(f"[DATA EXFIL] Connection failed (expected for simulation)")
    
    def dns_tunnel_simulation(self, count: int = 50):
        """Simulate DNS tunneling (many DNS requests)"""
        print(f"\n[DNS TUNNEL] Simulating {count} DNS requests...")
        
        for i in range(count):
            try:
                # Generate random subdomain (simulating encoded data)
                random_sub = ''.join(random.choices('abcdef0123456789', k=32))
                hostname = f"{random_sub}.example.com"
                socket.gethostbyname(hostname)
            except:
                pass
            time.sleep(0.05)
        
        print(f"[DNS TUNNEL] Complete - {count} DNS requests sent")
    
    def run_all(self):
        """Run all attack simulations"""
        print("\n" + "="*60)
        print("  PCDS ATTACK SIMULATOR - Testing Mode")
        print("="*60)
        print(f"  Target: {self.target_ip}")
        print("  WARNING: Only use on networks you own!")
        print("="*60)
        
        # Run simulations
        self.port_scan(ports=list(range(1, 101)), delay=0.05)
        time.sleep(1)
        
        self.syn_flood(port=80, count=50, delay=0.02)
        time.sleep(1)
        
        self.brute_force_simulation(port=22, attempts=30)
        time.sleep(1)
        
        self.data_exfil_simulation(port=443, data_size=50000)
        time.sleep(1)
        
        self.dns_tunnel_simulation(count=30)
        
        print("\n" + "="*60)
        print("  All simulations complete!")
        print("  Check PCDS Agent for detected threats.")
        print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="PCDS Attack Simulator")
    parser.add_argument("-t", "--target", default="127.0.0.1", help="Target IP")
    parser.add_argument("-a", "--attack", choices=["scan", "syn", "brute", "exfil", "dns", "all"],
                        default="all", help="Attack type")
    
    args = parser.parse_args()
    
    sim = AttackSimulator(target_ip=args.target)
    
    if args.attack == "scan":
        sim.port_scan()
    elif args.attack == "syn":
        sim.syn_flood()
    elif args.attack == "brute":
        sim.brute_force_simulation()
    elif args.attack == "exfil":
        sim.data_exfil_simulation()
    elif args.attack == "dns":
        sim.dns_tunnel_simulation()
    else:
        sim.run_all()


if __name__ == "__main__":
    main()
