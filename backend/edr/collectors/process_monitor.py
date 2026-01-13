"""
Process Monitor - EDR Core Component
Monitors process creation, injection, and suspicious behavior

Detects:
- Suspicious spawning (Office → PowerShell)
- Process injection (CreateRemoteThread)
- LOLBIN abuse (certutil, wmic, etc.)
- Encoded commands
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from datetime import datetime
from threading import Thread, Event
import psutil
import time
import os

from ..core.event_queue import get_event_queue, EDREvent
from ..core.sysmon_parser import (
    get_sysmon_parser, 
    is_suspicious_process,
    LOLBINS,
    SUSPICIOUS_PARENT_CHILD
)


@dataclass
class ProcessInfo:
    """Process information snapshot"""
    pid: int
    name: str
    cmdline: str
    exe: str
    parent_pid: int
    parent_name: str
    username: str
    create_time: float
    children: List[int]


class ProcessMonitor:
    """
    Real-time process monitoring
    
    Detects:
    - New process creation
    - Suspicious parent-child relationships
    - LOLBIN execution
    - Process injection attempts
    - Encoded/hidden commands
    """
    
    def __init__(self):
        self.event_queue = get_event_queue()
        self.sysmon = get_sysmon_parser()
        self._stop_event = Event()
        self._monitor_thread = None
        self._known_pids: Set[int] = set()
        self._process_tree: Dict[int, ProcessInfo] = {}
        
        # Detection statistics
        self.stats = {
            "processes_monitored": 0,
            "detections": 0,
            "lolbins_detected": 0,
            "injection_attempts": 0
        }
        
        # Initialize with current processes
        self._snapshot_processes()
        
        print("🔍 Process Monitor initialized")
    
    def start(self):
        """Start process monitoring"""
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("🔍 Process Monitor started")
    
    def stop(self):
        """Stop process monitoring"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        print("🔍 Process Monitor stopped")
    
    def _snapshot_processes(self):
        """Take snapshot of current processes"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'exe', 'ppid', 'username', 'create_time']):
            try:
                info = proc.info
                self._known_pids.add(info['pid'])
                self._process_tree[info['pid']] = ProcessInfo(
                    pid=info['pid'],
                    name=info['name'] or "",
                    cmdline=" ".join(info['cmdline']) if info['cmdline'] else "",
                    exe=info['exe'] or "",
                    parent_pid=info['ppid'] or 0,
                    parent_name=self._get_process_name(info['ppid']),
                    username=info['username'] or "",
                    create_time=info['create_time'] or 0,
                    children=[]
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    
    def _get_process_name(self, pid: int) -> str:
        """Get process name by PID"""
        try:
            if pid and pid > 0:
                return psutil.Process(pid).name()
        except:
            pass
        return ""
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                self._check_new_processes()
                time.sleep(0.5)  # Check every 500ms
            except Exception as e:
                print(f"⚠️ Process monitor error: {e}")
                time.sleep(1)
    
    def _check_new_processes(self):
        """Check for new processes"""
        current_pids = set()
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'exe', 'ppid', 'username', 'create_time']):
            try:
                info = proc.info
                pid = info['pid']
                current_pids.add(pid)
                
                # New process detected
                if pid not in self._known_pids:
                    self._known_pids.add(pid)
                    self._handle_new_process(info)
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Clean up terminated processes
        terminated = self._known_pids - current_pids
        for pid in terminated:
            self._known_pids.discard(pid)
            self._process_tree.pop(pid, None)
    
    def _handle_new_process(self, info: Dict):
        """Handle new process creation"""
        self.stats["processes_monitored"] += 1
        
        pid = info['pid']
        name = info['name'] or ""
        cmdline = " ".join(info['cmdline']) if info['cmdline'] else ""
        parent_pid = info['ppid'] or 0
        parent_name = self._get_process_name(parent_pid)
        
        # Store in process tree
        self._process_tree[pid] = ProcessInfo(
            pid=pid,
            name=name,
            cmdline=cmdline,
            exe=info['exe'] or "",
            parent_pid=parent_pid,
            parent_name=parent_name,
            username=info['username'] or "",
            create_time=info['create_time'] or 0,
            children=[]
        )
        
        # Check for suspicious behavior
        detection = is_suspicious_process(name, parent_name, cmdline)
        
        if detection.get("is_suspicious"):
            self._handle_detection(info, detection)
    
    def _handle_detection(self, process_info: Dict, detection: Dict):
        """Handle suspicious process detection"""
        self.stats["detections"] += 1
        
        # Determine severity based on risk level
        risk = detection.get("highest_risk", "medium")
        severity_map = {
            "low": "low",
            "medium": "medium",
            "high": "high",
            "critical": "critical"
        }
        severity = severity_map.get(risk, "medium")
        
        # Determine MITRE technique
        detection_type = detection["detections"][0]["type"] if detection["detections"] else ""
        mitre_map = {
            "lolbin": "T1218",  # System Binary Proxy Execution
            "suspicious_spawn": "T1059",  # Command and Scripting Interpreter
            "encoded_command": "T1027",  # Obfuscated Files or Information
            "hidden_execution": "T1564",  # Hide Artifacts
            "certutil_download": "T1105",  # Ingress Tool Transfer
        }
        mitre = mitre_map.get(detection_type, "T1059")
        
        # Create event
        event_data = {
            "pid": process_info['pid'],
            "process_name": process_info['name'],
            "command_line": " ".join(process_info['cmdline']) if process_info['cmdline'] else "",
            "parent_pid": process_info['ppid'],
            "parent_name": self._get_process_name(process_info['ppid']),
            "username": process_info['username'],
            "detections": detection["detections"],
        }
        
        # Get detection name
        detection_name = detection["detections"][0].get("reason", "Suspicious Process") if detection["detections"] else "Suspicious Process"
        
        # Publish to event queue
        self.event_queue.create_event(
            event_type="process",
            data=event_data,
            severity=severity,
            mitre_technique=mitre,
            detection_name=detection_name
        )
        
        # Update stats
        if any(d["type"] == "lolbin" for d in detection["detections"]):
            self.stats["lolbins_detected"] += 1
        
        print(f"⚠️ [{severity.upper()}] {detection_name}: {process_info['name']} (PID: {process_info['pid']})")
    
    def get_process_tree(self, pid: int) -> List[ProcessInfo]:
        """Get process tree (ancestors) for a PID"""
        tree = []
        current_pid = pid
        visited = set()
        
        while current_pid and current_pid not in visited:
            visited.add(current_pid)
            if current_pid in self._process_tree:
                tree.append(self._process_tree[current_pid])
                current_pid = self._process_tree[current_pid].parent_pid
            else:
                break
        
        return tree
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics"""
        return {
            **self.stats,
            "active_processes": len(self._known_pids)
        }


# Singleton
_process_monitor = None

def get_process_monitor() -> ProcessMonitor:
    global _process_monitor
    if _process_monitor is None:
        _process_monitor = ProcessMonitor()
    return _process_monitor


if __name__ == "__main__":
    # Test
    monitor = ProcessMonitor()
    monitor.start()
    
    print("\n🔍 Monitoring processes... Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(10)
            stats = monitor.get_stats()
            print(f"📊 Stats: {stats}")
    except KeyboardInterrupt:
        monitor.stop()
        print("\n✅ Process monitor stopped")
