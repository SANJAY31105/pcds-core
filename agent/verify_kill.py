import psutil
import time
import subprocess
import os
import sys

def start_dummy_malware():
    """Starts a harmless process (ping) to simulate malware"""
    print("[*] Launching Dummy Malware (ping -t)...")
    # Windows: ping -t localhost runs forever
    process = subprocess.Popen(["ping", "-t", "localhost"], stdout=subprocess.DEVNULL)
    print(f"[+] Dummy Malware started with PID: {process.pid}")
    return process

def kill_process(pid):
    """Attempts to kill a process by PID"""
    print(f"[*] Attempting to KILL PID: {pid}...")
    try:
        parent = psutil.Process(pid)
        parent.kill()
        print(f"[+] SUCCESS: Process {pid} Killed!")
        return True
    except psutil.NoSuchProcess:
        print(f"[-] Error: Process {pid} does not exist.")
        return False
    except psutil.AccessDenied:
        print(f"[-] Error: Access Denied to kill PID {pid}.")
        return False

def verify_kill(process):
    """Verifies that the process is actually gone"""
    time.sleep(1) # Wait for OS to clean up
    if process.poll() is not None:
        print("[+] VERIFIED: Process is dead (Exit Code set).")
    else:
        print("[-] FAILED: Process is still running.")

if __name__ == "__main__":
    print("--- PCDS Active Defense Test ---")
    
    # 1. Start Target
    dummy = start_dummy_malware()
    pid = dummy.pid
    
    # 2. Wait a bit
    print("    (Sleeping 2 seconds to simulate detection lag...)")
    time.sleep(2)
    
    # 3. Kill Target
    kill_process(pid)
    
    # 4. Verify
    verify_kill(dummy)
    print("--------------------------------")
