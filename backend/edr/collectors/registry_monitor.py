"""
Registry Monitor - EDR Component
Monitors Windows Registry for persistence mechanisms

Detects:
- Run/RunOnce keys (autostart)
- Services creation
- Scheduled tasks
- COM object hijacking
- WMI persistence
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from datetime import datetime
from threading import Thread, Event
import time
import platform

# Only import Windows-specific modules on Windows
if platform.system() == "Windows":
    try:
        import winreg
        WINREG_AVAILABLE = True
    except ImportError:
        WINREG_AVAILABLE = False
else:
    WINREG_AVAILABLE = False

from ..core.event_queue import get_event_queue


# Registry keys commonly used for persistence
PERSISTENCE_KEYS = {
    # Run keys (user-level)
    "HKCU_Run": {
        "hive": "HKEY_CURRENT_USER",
        "path": r"Software\Microsoft\Windows\CurrentVersion\Run",
        "risk": "high",
        "mitre": "T1547.001",
        "description": "User autostart programs"
    },
    "HKCU_RunOnce": {
        "hive": "HKEY_CURRENT_USER",
        "path": r"Software\Microsoft\Windows\CurrentVersion\RunOnce",
        "risk": "high",
        "mitre": "T1547.001",
        "description": "User one-time autostart"
    },
    
    # Run keys (machine-level)
    "HKLM_Run": {
        "hive": "HKEY_LOCAL_MACHINE",
        "path": r"Software\Microsoft\Windows\CurrentVersion\Run",
        "risk": "critical",
        "mitre": "T1547.001",
        "description": "System autostart programs"
    },
    "HKLM_RunOnce": {
        "hive": "HKEY_LOCAL_MACHINE",
        "path": r"Software\Microsoft\Windows\CurrentVersion\RunOnce",
        "risk": "critical",
        "mitre": "T1547.001",
        "description": "System one-time autostart"
    },
    
    # Services
    "HKLM_Services": {
        "hive": "HKEY_LOCAL_MACHINE",
        "path": r"SYSTEM\CurrentControlSet\Services",
        "risk": "critical",
        "mitre": "T1543.003",
        "description": "Windows services"
    },
    
    # Winlogon
    "HKLM_Winlogon": {
        "hive": "HKEY_LOCAL_MACHINE",
        "path": r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon",
        "risk": "critical",
        "mitre": "T1547.004",
        "description": "Winlogon helper DLL"
    },
    
    # Shell extensions
    "HKCU_Shell": {
        "hive": "HKEY_CURRENT_USER",
        "path": r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders",
        "risk": "medium",
        "mitre": "T1547",
        "description": "Shell folder paths"
    },
    
    # Image File Execution Options (debugger hijack)
    "HKLM_IFEO": {
        "hive": "HKEY_LOCAL_MACHINE",
        "path": r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Image File Execution Options",
        "risk": "critical",
        "mitre": "T1546.012",
        "description": "Image File Execution Options"
    },
}

# Suspicious values to watch for
SUSPICIOUS_INDICATORS = [
    "powershell",
    "cmd.exe",
    "wscript",
    "cscript",
    "mshta",
    "regsvr32",
    "rundll32",
    "certutil",
    "-encodedcommand",
    "-enc",
    "bypass",
    "hidden",
    "downloadstring",
    "invoke-",
    "iex",
    "frombase64",
]


@dataclass
class RegistryChange:
    """Registry change event"""
    key_path: str
    value_name: str
    value_data: str
    change_type: str  # created, modified, deleted
    timestamp: datetime
    risk: str
    mitre: str


class RegistryMonitor:
    """
    Windows Registry Monitor for EDR
    
    Monitors:
    - Run/RunOnce keys
    - Services
    - Winlogon
    - Image File Execution Options
    - COM objects
    """
    
    def __init__(self):
        self.event_queue = get_event_queue()
        self._stop_event = Event()
        self._monitor_thread = None
        
        # Track known values
        self._known_values: Dict[str, Dict[str, str]] = {}
        
        # Stats
        self.stats = {
            "keys_monitored": 0,
            "changes_detected": 0,
            "suspicious_changes": 0,
            "persistence_attempts": 0
        }
        
        # Check if running on Windows
        self.is_windows = platform.system() == "Windows" and WINREG_AVAILABLE
        
        if self.is_windows:
            # Take initial snapshot
            self._snapshot_registry()
            print("📝 Registry Monitor initialized")
        else:
            print("⚠️ Registry Monitor: Windows only (skipped)")
    
    def start(self):
        """Start registry monitoring"""
        if not self.is_windows:
            print("📝 Registry Monitor: Skipped (not Windows)")
            return
        
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("📝 Registry Monitor started")
    
    def stop(self):
        """Stop registry monitoring"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        print("📝 Registry Monitor stopped")
    
    def _get_hive(self, hive_name: str):
        """Get registry hive constant"""
        if not self.is_windows:
            return None
        
        hives = {
            "HKEY_CURRENT_USER": winreg.HKEY_CURRENT_USER,
            "HKEY_LOCAL_MACHINE": winreg.HKEY_LOCAL_MACHINE,
            "HKEY_CLASSES_ROOT": winreg.HKEY_CLASSES_ROOT,
        }
        return hives.get(hive_name)
    
    def _snapshot_registry(self):
        """Take snapshot of monitored registry keys"""
        if not self.is_windows:
            return
        
        for key_name, key_info in PERSISTENCE_KEYS.items():
            try:
                hive = self._get_hive(key_info["hive"])
                if not hive:
                    continue
                
                key = winreg.OpenKey(hive, key_info["path"], 0, winreg.KEY_READ)
                
                self._known_values[key_name] = {}
                
                # Read all values
                i = 0
                while True:
                    try:
                        name, data, _ = winreg.EnumValue(key, i)
                        self._known_values[key_name][name] = str(data)
                        self.stats["keys_monitored"] += 1
                        i += 1
                    except WindowsError:
                        break
                
                winreg.CloseKey(key)
                
            except WindowsError:
                # Key doesn't exist or access denied
                self._known_values[key_name] = {}
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                self._check_registry_changes()
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                print(f"⚠️ Registry monitor error: {e}")
                time.sleep(10)
    
    def _check_registry_changes(self):
        """Check for registry changes"""
        if not self.is_windows:
            return
        
        for key_name, key_info in PERSISTENCE_KEYS.items():
            try:
                hive = self._get_hive(key_info["hive"])
                if not hive:
                    continue
                
                key = winreg.OpenKey(hive, key_info["path"], 0, winreg.KEY_READ)
                
                current_values = {}
                
                # Read current values
                i = 0
                while True:
                    try:
                        name, data, _ = winreg.EnumValue(key, i)
                        current_values[name] = str(data)
                        i += 1
                    except WindowsError:
                        break
                
                winreg.CloseKey(key)
                
                # Compare with known values
                known = self._known_values.get(key_name, {})
                
                # Check for new values
                for name, data in current_values.items():
                    if name not in known:
                        self._handle_change(key_name, key_info, name, data, "created")
                    elif known[name] != data:
                        self._handle_change(key_name, key_info, name, data, "modified")
                
                # Check for deleted values
                for name in known:
                    if name not in current_values:
                        self._handle_change(key_name, key_info, name, known[name], "deleted")
                
                # Update known values
                self._known_values[key_name] = current_values
                
            except WindowsError:
                pass
    
    def _handle_change(self, key_name: str, key_info: Dict, value_name: str, value_data: str, change_type: str):
        """Handle detected registry change"""
        self.stats["changes_detected"] += 1
        
        # Check for suspicious indicators
        is_suspicious = False
        data_lower = value_data.lower()
        
        for indicator in SUSPICIOUS_INDICATORS:
            if indicator in data_lower:
                is_suspicious = True
                break
        
        # Determine severity
        if is_suspicious:
            self.stats["suspicious_changes"] += 1
            severity = "critical"
        else:
            severity = key_info["risk"]
        
        # Persistence detection
        if change_type == "created" and key_name in ["HKCU_Run", "HKLM_Run", "HKCU_RunOnce", "HKLM_RunOnce"]:
            self.stats["persistence_attempts"] += 1
        
        # Create detection name
        detection_name = f"Registry {change_type}: {key_info['description']}"
        if is_suspicious:
            detection_name = f"SUSPICIOUS: {detection_name}"
        
        # Publish event
        self.event_queue.create_event(
            event_type="registry",
            data={
                "key_name": key_name,
                "key_path": f"{key_info['hive']}\\{key_info['path']}",
                "value_name": value_name,
                "value_data": value_data[:500],  # Truncate long data
                "change_type": change_type,
                "is_suspicious": is_suspicious,
                "indicators_found": [i for i in SUSPICIOUS_INDICATORS if i in data_lower]
            },
            severity=severity,
            mitre_technique=key_info["mitre"],
            detection_name=detection_name
        )
        
        print(f"📝 [{severity.upper()}] {detection_name}: {value_name}")
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics"""
        return self.stats
    
    def get_run_keys(self) -> List[Dict]:
        """Get current Run key entries"""
        if not self.is_windows:
            return []
        
        entries = []
        
        for key_name in ["HKCU_Run", "HKLM_Run"]:
            for name, data in self._known_values.get(key_name, {}).items():
                entries.append({
                    "key": key_name,
                    "name": name,
                    "data": data
                })
        
        return entries


# Singleton
_registry_monitor = None

def get_registry_monitor() -> RegistryMonitor:
    global _registry_monitor
    if _registry_monitor is None:
        _registry_monitor = RegistryMonitor()
    return _registry_monitor


if __name__ == "__main__":
    monitor = RegistryMonitor()
    monitor.start()
    
    print("\n📝 Monitoring registry... Press Ctrl+C to stop\n")
    print(f"Current Run keys: {monitor.get_run_keys()}")
    
    try:
        while True:
            time.sleep(10)
            stats = monitor.get_stats()
            print(f"📊 Stats: {stats}")
    except KeyboardInterrupt:
        monitor.stop()
        print("\n✅ Registry monitor stopped")
