"""Quick script to list network interfaces"""
from scapy.arch.windows import get_windows_if_list

print("\n" + "="*70)
print("AVAILABLE NETWORK INTERFACES")
print("="*70)

for i, iface in enumerate(get_windows_if_list(), 1):
    name = iface.get('name', 'Unknown')[:40]
    desc = iface.get('description', 'No description')[:50]
    guid = iface.get('guid', '')
    print(f"\n[{i}] {name}")
    print(f"    Description: {desc}")
    print(f"    GUID: {guid}")

print("\n" + "="*70)
print("Use the GUID (including braces) with the -i flag to capture")
print("Example: python -m agent.main -i \"{GUID-HERE}\"")
print("="*70)
