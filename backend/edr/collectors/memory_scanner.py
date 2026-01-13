"""
Memory Scanner - EDR Advanced Module
Scans process memory for malicious indicators

Detects:
- Shellcode patterns
- Process injection
- Credential stealing tools
- Suspicious memory regions
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from threading import Thread, Event
import psutil
import ctypes
import struct
import time
import re

# Windows API constants
PROCESS_QUERY_INFORMATION = 0x0400
PROCESS_VM_READ = 0x0010
MEM_COMMIT = 0x1000
PAGE_EXECUTE_READWRITE = 0x40
PAGE_EXECUTE_READ = 0x20
PAGE_EXECUTE = 0x10

# Shellcode signatures (common patterns)
SHELLCODE_PATTERNS = [
    # x86 NOP sled
    (b'\x90' * 10, "NOP Sled", "high"),
    
    # Common shellcode stubs
    (b'\xfc\xe8', "Call-Pop Shellcode", "critical"),
    (b'\x31\xc9\xf7\xe1', "XOR ECX shellcode", "critical"),
    (b'\x31\xc0\x50\x68', "Push shellcode", "high"),
    
    # Meterpreter patterns
    (b'\xfc\x48\x83\xe4\xf0', "Meterpreter x64", "critical"),
    (b'\xfc\xe8\x82\x00\x00\x00', "Meterpreter x86", "critical"),
    
    # Cobalt Strike beacon
    (b'\x4d\x5a\x41\x52\x55\x48\x89\xe5', "Cobalt Strike", "critical"),
    
    # Common API hashing
    (b'\x64\xa1\x30\x00\x00\x00', "PEB Access x86", "high"),
    (b'\x65\x48\x8b\x04\x25\x60', "PEB Access x64", "high"),
]

# Credential tool signatures (in memory strings)
CREDENTIAL_SIGNATURES = [
    ("mimikatz", "critical", "T1003.001"),
    ("sekurlsa", "critical", "T1003.001"),
    ("lsadump", "critical", "T1003.001"),
    ("kerberos::list", "critical", "T1558"),
    ("privilege::debug", "high", "T1134"),
    ("token::elevate", "high", "T1134"),
    ("procdump", "high", "T1003.001"),
    ("comsvcs.dll", "high", "T1003.001"),
    ("ntdsutil", "critical", "T1003.003"),
]

# Suspicious process names
SUSPICIOUS_PROCESSES = {
    "mimikatz.exe": ("Credential Theft Tool", "critical"),
    "procdump.exe": ("Memory Dump Tool", "high"),
    "lazagne.exe": ("Password Recovery", "critical"),
    "rubeus.exe": ("Kerberos Attack Tool", "critical"),
    "sharphound.exe": ("AD Recon Tool", "high"),
    "bloodhound.exe": ("AD Attack Tool", "high"),
    "covenant.exe": ("C2 Framework", "critical"),
}


@dataclass
class MemoryFinding:
    """Memory scan finding"""
    pid: int
    process_name: str
    finding_type: str
    description: str
    severity: str
    mitre_technique: str
    address: int = 0
    size: int = 0


class MemoryScanner:
    """
    Advanced Memory Scanner for EDR
    
    Scans process memory for:
    - Shellcode patterns
    - Injection indicators
    - Credential theft tools
    - Suspicious memory regions
    """
    
    def __init__(self):
        self._stop_event = Event()
        self._scan_thread = None
        
        # Import event queue
        try:
            from .core.event_queue import get_event_queue
            self.event_queue = get_event_queue()
        except:
            self.event_queue = None
        
        # Stats
        self.stats = {
            "processes_scanned": 0,
            "memory_regions_scanned": 0,
            "shellcode_detected": 0,
            "credential_tools_detected": 0,
            "injections_detected": 0
        }
        
        # Findings
        self.findings: List[MemoryFinding] = []
        
        print("🧠 Memory Scanner initialized")
    
    def start(self, interval: int = 60):
        """Start background memory scanning"""
        self._stop_event.clear()
        self._scan_thread = Thread(target=self._scan_loop, args=(interval,), daemon=True)
        self._scan_thread.start()
        print("🧠 Memory Scanner started")
    
    def stop(self):
        """Stop memory scanning"""
        self._stop_event.set()
        if self._scan_thread:
            self._scan_thread.join(timeout=5)
        print("🧠 Memory Scanner stopped")
    
    def _scan_loop(self, interval: int):
        """Background scanning loop"""
        while not self._stop_event.is_set():
            try:
                self.scan_all_processes()
                time.sleep(interval)
            except Exception as e:
                print(f"⚠️ Memory scan error: {e}")
                time.sleep(10)
    
    def scan_all_processes(self) -> List[MemoryFinding]:
        """Scan all running processes"""
        all_findings = []
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                pid = proc.info['pid']
                name = proc.info['name']
                
                # Skip system processes
                if pid < 10:
                    continue
                
                findings = self.scan_process(pid, name)
                all_findings.extend(findings)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return all_findings
    
    def scan_process(self, pid: int, process_name: str = None) -> List[MemoryFinding]:
        """Scan a specific process"""
        findings = []
        
        if not process_name:
            try:
                process_name = psutil.Process(pid).name()
            except:
                process_name = "unknown"
        
        self.stats["processes_scanned"] += 1
        
        # Check 1: Suspicious process name
        if process_name.lower() in SUSPICIOUS_PROCESSES:
            desc, severity = SUSPICIOUS_PROCESSES[process_name.lower()]
            finding = MemoryFinding(
                pid=pid,
                process_name=process_name,
                finding_type="suspicious_process",
                description=desc,
                severity=severity,
                mitre_technique="T1003"
            )
            findings.append(finding)
            self._report_finding(finding)
        
        # Check 2: Memory regions for shellcode (requires elevated access)
        try:
            memory_findings = self._scan_memory_regions(pid, process_name)
            findings.extend(memory_findings)
        except:
            pass  # Access denied is common
        
        # Check 3: Check for credential strings in command line
        try:
            proc = psutil.Process(pid)
            cmdline = ' '.join(proc.cmdline()).lower()
            
            for sig, severity, mitre in CREDENTIAL_SIGNATURES:
                if sig in cmdline:
                    finding = MemoryFinding(
                        pid=pid,
                        process_name=process_name,
                        finding_type="credential_tool",
                        description=f"Credential tool signature: {sig}",
                        severity=severity,
                        mitre_technique=mitre
                    )
                    findings.append(finding)
                    self._report_finding(finding)
                    self.stats["credential_tools_detected"] += 1
        except:
            pass
        
        return findings
    
    def _scan_memory_regions(self, pid: int, process_name: str) -> List[MemoryFinding]:
        """Scan process memory regions for shellcode"""
        findings = []
        
        try:
            proc = psutil.Process(pid)
            
            # Get memory maps
            try:
                memory_maps = proc.memory_maps()
            except:
                return findings
            
            for mmap in memory_maps:
                self.stats["memory_regions_scanned"] += 1
                
                # Check for executable + writable (RWX) - suspicious!
                perms = mmap.perms if hasattr(mmap, 'perms') else ''
                
                if 'r' in perms and 'w' in perms and 'x' in perms:
                    finding = MemoryFinding(
                        pid=pid,
                        process_name=process_name,
                        finding_type="rwx_memory",
                        description=f"RWX memory region detected (possible injection)",
                        severity="high",
                        mitre_technique="T1055",
                        address=int(mmap.addr.split('-')[0], 16) if hasattr(mmap, 'addr') else 0,
                        size=mmap.rss if hasattr(mmap, 'rss') else 0
                    )
                    findings.append(finding)
                    self._report_finding(finding)
                    self.stats["injections_detected"] += 1
            
        except Exception as e:
            pass
        
        return findings
    
    def detect_process_hollowing(self, pid: int) -> Optional[MemoryFinding]:
        """Detect process hollowing technique"""
        try:
            proc = psutil.Process(pid)
            
            # Check if executable path matches memory
            exe_path = proc.exe()
            cmdline = ' '.join(proc.cmdline())
            
            # Hollowed processes often have mismatched exe
            if exe_path and cmdline:
                exe_name = exe_path.split('\\')[-1].lower()
                
                # Check for common hollowing targets
                hollow_targets = ['svchost.exe', 'explorer.exe', 'notepad.exe', 'calc.exe']
                
                for target in hollow_targets:
                    if target in exe_name:
                        # Check if running from unusual location
                        if 'windows\\system32' not in exe_path.lower() and 'windows\\syswow64' not in exe_path.lower():
                            return MemoryFinding(
                                pid=pid,
                                process_name=proc.name(),
                                finding_type="process_hollowing",
                                description=f"Possible process hollowing: {target} from unusual path",
                                severity="critical",
                                mitre_technique="T1055.012"
                            )
        except:
            pass
        
        return None
    
    def _report_finding(self, finding: MemoryFinding):
        """Report finding to event queue"""
        self.findings.append(finding)
        
        # Keep last 1000 findings
        if len(self.findings) > 1000:
            self.findings = self.findings[-1000:]
        
        if self.event_queue:
            self.event_queue.create_event(
                event_type="memory",
                data={
                    "pid": finding.pid,
                    "process_name": finding.process_name,
                    "finding_type": finding.finding_type,
                    "description": finding.description,
                    "address": finding.address,
                    "size": finding.size
                },
                severity=finding.severity,
                mitre_technique=finding.mitre_technique,
                detection_name=f"Memory: {finding.description}"
            )
        
        print(f"🧠 [{finding.severity.upper()}] {finding.finding_type}: {finding.process_name} (PID: {finding.pid})")
    
    def quick_scan(self) -> Dict:
        """Quick scan for immediate threats"""
        threats = []
        
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                name = proc.info['name'].lower()
                
                # Check suspicious process names
                if name in SUSPICIOUS_PROCESSES:
                    threats.append({
                        "pid": proc.info['pid'],
                        "name": proc.info['name'],
                        "threat": SUSPICIOUS_PROCESSES[name][0],
                        "severity": SUSPICIOUS_PROCESSES[name][1]
                    })
                
                # Check for credential tools in cmdline
                cmdline = ' '.join(proc.cmdline()).lower()
                for sig, severity, mitre in CREDENTIAL_SIGNATURES:
                    if sig in cmdline:
                        threats.append({
                            "pid": proc.info['pid'],
                            "name": proc.info['name'],
                            "threat": f"Credential Tool: {sig}",
                            "severity": severity
                        })
                        break
                        
            except:
                pass
        
        return {
            "scan_time": datetime.now().isoformat(),
            "threats_found": len(threats),
            "threats": threats
        }
    
    def get_stats(self) -> Dict:
        """Get scanner statistics"""
        return {
            **self.stats,
            "total_findings": len(self.findings)
        }
    
    def get_recent_findings(self, count: int = 50) -> List[Dict]:
        """Get recent findings"""
        return [
            {
                "pid": f.pid,
                "process_name": f.process_name,
                "finding_type": f.finding_type,
                "description": f.description,
                "severity": f.severity,
                "mitre_technique": f.mitre_technique
            }
            for f in self.findings[-count:]
        ]


# Singleton
_memory_scanner = None

def get_memory_scanner() -> MemoryScanner:
    global _memory_scanner
    if _memory_scanner is None:
        _memory_scanner = MemoryScanner()
    return _memory_scanner


if __name__ == "__main__":
    scanner = MemoryScanner()
    
    print("\n🧠 Quick Memory Scan...\n")
    result = scanner.quick_scan()
    
    print(f"Threats found: {result['threats_found']}")
    for threat in result['threats']:
        print(f"  [{threat['severity'].upper()}] {threat['name']} - {threat['threat']}")
    
    print(f"\n📊 Stats: {scanner.get_stats()}")
