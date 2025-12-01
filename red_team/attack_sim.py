import requests
import time
import random
import sys

API_URL = "http://127.0.0.1:8000/api/sensor_input"

def send_traffic(packet_type, count=10):
    print(f"\n[>>>] Launching {packet_type} simulation ({count} packets)...")
    
    if packet_type == "DDoS":
        base_data = {'size': 64, 'protocol': 'UDP', 'port': 80, 'rate': 2000}
    elif packet_type == "Port Scan":
        base_data = {'size': 64, 'protocol': 'TCP', 'port': 22, 'rate': 50}
    elif packet_type == "Malware":
        base_data = {'size': 3000, 'protocol': 'TCP', 'port': 4444, 'rate': 5}
    else: 
        base_data = {'size': 500, 'protocol': 'HTTPS', 'port': 443, 'rate': 10}

    attacker_ip = f"192.168.66.{random.randint(10, 200)}"
    print(f"      Source IP: {attacker_ip}")

    for i in range(count):
        payload = base_data.copy()
        payload['source_ip'] = attacker_ip
        try:
            requests.post(API_URL, json=payload)
            sys.stdout.write(f"\r      Packet {i+1}/{count} sent...")
            sys.stdout.flush()
        except:
            print("\n[!] Error: Is the Backend Server running?")
            return
        time.sleep(0.1)
    
    print("\n[V] Attack wave completed.")

def main_menu():
    while True:
        print("\n" + "="*40)
        print("   PCDS RED TEAM ATTACK SIMULATOR")
        print("="*40)
        print("1. Simulate DDoS Attack")
        print("2. Simulate Port Scan")
        print("3. Simulate Malware C2")
        print("4. Send Normal Traffic")
        print("5. Exit")
        choice = input("\nSelect Option: ")
        if choice == '1': send_traffic("DDoS", 50)
        elif choice == '2': send_traffic("Port Scan", 20)
        elif choice == '3': send_traffic("Malware", 10)
        elif choice == '4': send_traffic("Normal", 20)
        elif choice == '5': break

if __name__ == "__main__":
    main_menu()