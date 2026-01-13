"""
Sysmon Parser for Windows EDR
Parses Sysmon event logs for security monitoring

MVP approach - no kernel drivers needed!
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import re
import subprocess
import json

# Sysmon Event IDs and their meanings
SYSMON_EVENTS = {
    1: "ProcessCreate",
    2: "FileCreateTime",
    3: "NetworkConnect",
    5: "ProcessTerminate",
    6: "DriverLoad",
    7: "ImageLoad",
    8: "CreateRemoteThread",
    9: "RawAccessRead",
    10: "ProcessAccess",
    11: "FileCreate",
    12: "RegistryEvent",
    13: "RegistryValueSet",
    14: "RegistryRename",
    15: "FileCreateStreamHash",
    17: "PipeEvent",
    18: "PipeEvent",
    19: "WmiEvent",
    20: "WmiEvent",
    21: "WmiEvent",
    22: "DNSQuery",
    23: "FileDelete",
    24: "ClipboardChange",
    25: "ProcessTampering",
    26: "FileDeleteDetected"
}


@dataclass
class SysmonEvent:
    """Parsed Sysmon event"""
    event_id: int
    event_type: str
    timestamp: datetime
    process_id: int
    process_name: str
    command_line: str
    parent_process_id: int
    parent_process_name: str
    user: str
    raw_data: Dict[str, Any]


class SysmonParser:
    """
    Parse Sysmon events from Windows Event Log
    
    Uses PowerShell to query Sysmon logs (no kernel driver needed)
    """
    
    def __init__(self):
        self.is_available = self._check_sysmon()
    
    def _check_sysmon(self) -> bool:
        """Check if Sysmon is installed"""
        try:
            result = subprocess.run(
                ["powershell", "-Command", "Get-Service Sysmon* -ErrorAction SilentlyContinue"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "Sysmon" in result.stdout
        except:
            return False
    
    def get_recent_events(self, count: int = 100, event_ids: List[int] = None) -> List[SysmonEvent]:
        """
        Get recent Sysmon events
        
        Args:
            count: Number of events to retrieve
            event_ids: Filter by specific event IDs (e.g., [1, 3, 11] for process, network, file)
        """
        if not self.is_available:
            return []
        
        # Build event ID filter
        id_filter = ""
        if event_ids:
            id_conditions = " or ".join([f"EventID={eid}" for eid in event_ids])
            id_filter = f"-FilterXPath '*[System[{id_conditions}]]'"
        
        # Query Sysmon logs via PowerShell
        ps_command = f"""
        Get-WinEvent -LogName 'Microsoft-Windows-Sysmon/Operational' -MaxEvents {count} {id_filter} |
        Select-Object Id, TimeCreated, Message |
        ConvertTo-Json
        """
        
        try:
            result = subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return []
            
            events_data = json.loads(result.stdout) if result.stdout.strip() else []
            
            # Handle single event (returns dict instead of list)
            if isinstance(events_data, dict):
                events_data = [events_data]
            
            return [self._parse_event(e) for e in events_data if e]
            
        except Exception as e:
            print(f"⚠️ Sysmon query error: {e}")
            return []
    
    def _parse_event(self, raw: Dict) -> Optional[SysmonEvent]:
        """Parse raw Sysmon event"""
        try:
            event_id = raw.get("Id", 0)
            message = raw.get("Message", "")
            
            # Extract fields from message
            fields = self._parse_message(message)
            
            return SysmonEvent(
                event_id=event_id,
                event_type=SYSMON_EVENTS.get(event_id, "Unknown"),
                timestamp=datetime.now(),  # Simplified
                process_id=int(fields.get("ProcessId", 0)),
                process_name=fields.get("Image", ""),
                command_line=fields.get("CommandLine", ""),
                parent_process_id=int(fields.get("ParentProcessId", 0)),
                parent_process_name=fields.get("ParentImage", ""),
                user=fields.get("User", ""),
                raw_data=fields
            )
        except:
            return None
    
    def _parse_message(self, message: str) -> Dict[str, str]:
        """Parse Sysmon message into key-value pairs"""
        fields = {}
        for line in message.split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                fields[key.strip()] = value.strip()
        return fields


# LOLBINS - Living Off The Land Binaries (suspicious when misused)
LOLBINS = {
    "powershell.exe": {"risk": "high", "reason": "Script execution"},
    "cmd.exe": {"risk": "medium", "reason": "Command execution"},
    "wmic.exe": {"risk": "high", "reason": "WMI queries"},
    "certutil.exe": {"risk": "high", "reason": "Download/decode files"},
    "bitsadmin.exe": {"risk": "high", "reason": "Background file transfer"},
    "mshta.exe": {"risk": "critical", "reason": "Execute HTA files"},
    "regsvr32.exe": {"risk": "high", "reason": "DLL registration abuse"},
    "rundll32.exe": {"risk": "high", "reason": "DLL execution"},
    "cscript.exe": {"risk": "medium", "reason": "Script execution"},
    "wscript.exe": {"risk": "medium", "reason": "Script execution"},
    "msiexec.exe": {"risk": "medium", "reason": "Install packages"},
    "schtasks.exe": {"risk": "high", "reason": "Task scheduling"},
    "at.exe": {"risk": "high", "reason": "Task scheduling"},
    "net.exe": {"risk": "medium", "reason": "Network commands"},
    "net1.exe": {"risk": "medium", "reason": "Network commands"},
    "psexec.exe": {"risk": "critical", "reason": "Remote execution"},
    "mimikatz.exe": {"risk": "critical", "reason": "Credential theft"},
}


# Suspicious parent-child relationships
SUSPICIOUS_PARENT_CHILD = [
    # Office apps spawning shells
    ("winword.exe", "cmd.exe"),
    ("winword.exe", "powershell.exe"),
    ("excel.exe", "cmd.exe"),
    ("excel.exe", "powershell.exe"),
    ("outlook.exe", "cmd.exe"),
    ("outlook.exe", "powershell.exe"),
    
    # Browser spawning shells
    ("chrome.exe", "cmd.exe"),
    ("chrome.exe", "powershell.exe"),
    ("firefox.exe", "cmd.exe"),
    ("msedge.exe", "powershell.exe"),
    
    # WMI spawning processes
    ("wmiprvse.exe", "cmd.exe"),
    ("wmiprvse.exe", "powershell.exe"),
    
    # Script hosts spawning shells
    ("wscript.exe", "cmd.exe"),
    ("cscript.exe", "powershell.exe"),
]


def is_suspicious_process(process_name: str, parent_name: str, command_line: str) -> Dict:
    """
    Check if process behavior is suspicious
    
    Returns detection info if suspicious, None otherwise
    """
    process_lower = process_name.lower()
    parent_lower = parent_name.lower() if parent_name else ""
    cmd_lower = command_line.lower() if command_line else ""
    
    detections = []
    
    # Check LOLBINS
    for lolbin, info in LOLBINS.items():
        if lolbin in process_lower:
            detections.append({
                "type": "lolbin",
                "process": lolbin,
                "risk": info["risk"],
                "reason": info["reason"]
            })
    
    # Check suspicious parent-child
    for parent, child in SUSPICIOUS_PARENT_CHILD:
        if parent in parent_lower and child in process_lower:
            detections.append({
                "type": "suspicious_spawn",
                "parent": parent,
                "child": child,
                "risk": "critical",
                "reason": f"{parent} spawned {child}"
            })
    
    # Check for Base64 encoded PowerShell
    if "powershell" in process_lower:
        if "-encodedcommand" in cmd_lower or "-enc" in cmd_lower or "-e " in cmd_lower:
            detections.append({
                "type": "encoded_command",
                "process": "powershell.exe",
                "risk": "high",
                "reason": "Base64 encoded command"
            })
        if "-noprofile" in cmd_lower and "-windowstyle hidden" in cmd_lower:
            detections.append({
                "type": "hidden_execution",
                "process": "powershell.exe",
                "risk": "high",
                "reason": "Hidden PowerShell execution"
            })
    
    # Check for suspicious certutil usage (download)
    if "certutil" in process_lower:
        if "-urlcache" in cmd_lower or "-split" in cmd_lower:
            detections.append({
                "type": "certutil_download",
                "process": "certutil.exe",
                "risk": "critical",
                "reason": "File download via certutil"
            })
    
    if detections:
        return {
            "is_suspicious": True,
            "detections": detections,
            "highest_risk": max(detections, key=lambda x: {"low": 1, "medium": 2, "high": 3, "critical": 4}.get(x["risk"], 0))["risk"]
        }
    
    return {"is_suspicious": False}


# Singleton
_sysmon_parser = None

def get_sysmon_parser() -> SysmonParser:
    global _sysmon_parser
    if _sysmon_parser is None:
        _sysmon_parser = SysmonParser()
    return _sysmon_parser
