"""
File Monitor - EDR Component
Monitors file system for malware drops and suspicious activity

Detects:
- Executable drops in temp folders
- Ransomware extensions
- Suspicious file locations
- Startup folder modifications
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass
from datetime import datetime
from threading import Thread, Event
from pathlib import Path
import os
import time

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("⚠️ watchdog not installed. File monitoring limited.")

from ..core.event_queue import get_event_queue


# Suspicious file extensions
SUSPICIOUS_EXTENSIONS = {
    # Executables
    ".exe": {"risk": "high", "type": "executable"},
    ".dll": {"risk": "high", "type": "library"},
    ".scr": {"risk": "high", "type": "screensaver"},
    ".pif": {"risk": "critical", "type": "shortcut"},
    ".com": {"risk": "high", "type": "executable"},
    
    # Scripts
    ".ps1": {"risk": "high", "type": "powershell"},
    ".vbs": {"risk": "high", "type": "vbscript"},
    ".vbe": {"risk": "high", "type": "vbscript"},
    ".js": {"risk": "medium", "type": "javascript"},
    ".jse": {"risk": "high", "type": "javascript"},
    ".wsf": {"risk": "high", "type": "script"},
    ".wsh": {"risk": "high", "type": "script"},
    ".bat": {"risk": "medium", "type": "batch"},
    ".cmd": {"risk": "medium", "type": "batch"},
    
    # Documents with macros
    ".docm": {"risk": "high", "type": "macro_doc"},
    ".xlsm": {"risk": "high", "type": "macro_doc"},
    ".pptm": {"risk": "high", "type": "macro_doc"},
    
    # Archives (potential malware delivery)
    ".iso": {"risk": "medium", "type": "disk_image"},
    ".img": {"risk": "medium", "type": "disk_image"},
    
    # Ransomware extensions
    ".encrypted": {"risk": "critical", "type": "ransomware"},
    ".locked": {"risk": "critical", "type": "ransomware"},
    ".crypted": {"risk": "critical", "type": "ransomware"},
    ".crypt": {"risk": "critical", "type": "ransomware"},
    ".enc": {"risk": "high", "type": "ransomware"},
}

# Suspicious paths to monitor
MONITORED_PATHS = {
    "temp": {
        "paths": [
            os.environ.get("TEMP", ""),
            os.environ.get("TMP", ""),
            "C:\\Windows\\Temp",
        ],
        "risk": "high",
        "reason": "Temp folder executable"
    },
    "startup": {
        "paths": [
            os.path.expandvars(r"%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup"),
            r"C:\ProgramData\Microsoft\Windows\Start Menu\Programs\Startup",
        ],
        "risk": "critical",
        "reason": "Startup persistence"
    },
    "downloads": {
        "paths": [
            os.path.expandvars(r"%USERPROFILE%\Downloads"),
        ],
        "risk": "medium",
        "reason": "Downloaded file"
    },
    "appdata": {
        "paths": [
            os.environ.get("APPDATA", ""),
            os.environ.get("LOCALAPPDATA", ""),
        ],
        "risk": "high",
        "reason": "AppData executable"
    },
    "public": {
        "paths": [
            r"C:\Users\Public",
        ],
        "risk": "high",
        "reason": "Public folder executable"
    }
}


# Only define if watchdog is available
if WATCHDOG_AVAILABLE:
    class SuspiciousFileHandler(FileSystemEventHandler):
        """Handle file system events"""
        
        def __init__(self, file_monitor):
            self.file_monitor = file_monitor
            self.event_queue = get_event_queue()
        
        def on_created(self, event):
            if not event.is_directory:
                self.file_monitor._analyze_file(event.src_path, "created")
        
        def on_modified(self, event):
            if not event.is_directory:
                self.file_monitor._analyze_file(event.src_path, "modified")


class FileMonitor:
    """
    File system monitor for EDR
    
    Monitors:
    - Temp folders
    - Startup folders
    - Downloads
    - AppData
    - Public folders
    """
    
    def __init__(self):
        self.event_queue = get_event_queue()
        self._stop_event = Event()
        self._observers: List = []
        self._fallback_thread = None
        
        # Track files
        self._known_files: Dict[str, float] = {}  # path -> mtime
        
        # Stats
        self.stats = {
            "files_monitored": 0,
            "suspicious_files": 0,
            "ransomware_detected": 0,
            "executables_in_temp": 0
        }
        
        print("📁 File Monitor initialized")
    
    def start(self):
        """Start file monitoring"""
        self._stop_event.clear()
        
        if WATCHDOG_AVAILABLE:
            self._start_watchdog()
        else:
            self._start_fallback()
        
        print("📁 File Monitor started")
    
    def stop(self):
        """Stop file monitoring"""
        self._stop_event.set()
        
        for observer in self._observers:
            observer.stop()
            observer.join(timeout=5)
        
        if self._fallback_thread:
            self._fallback_thread.join(timeout=5)
        
        print("📁 File Monitor stopped")
    
    def _start_watchdog(self):
        """Start watchdog-based monitoring"""
        handler = SuspiciousFileHandler(self)
        
        for category, config in MONITORED_PATHS.items():
            for path in config["paths"]:
                if path and os.path.exists(path):
                    observer = Observer()
                    observer.schedule(handler, path, recursive=True)
                    observer.start()
                    self._observers.append(observer)
                    print(f"   📂 Watching: {path}")
    
    def _start_fallback(self):
        """Start polling-based fallback monitoring"""
        self._fallback_thread = Thread(target=self._fallback_loop, daemon=True)
        self._fallback_thread.start()
    
    def _fallback_loop(self):
        """Fallback polling loop"""
        while not self._stop_event.is_set():
            try:
                for category, config in MONITORED_PATHS.items():
                    for base_path in config["paths"]:
                        if base_path and os.path.exists(base_path):
                            self._scan_directory(base_path)
                time.sleep(5)  # Check every 5 seconds
            except Exception as e:
                print(f"⚠️ File monitor error: {e}")
                time.sleep(10)
    
    def _scan_directory(self, directory: str):
        """Scan directory for new/modified files"""
        try:
            for root, dirs, files in os.walk(directory):
                for filename in files:
                    filepath = os.path.join(root, filename)
                    try:
                        mtime = os.path.getmtime(filepath)
                        
                        if filepath not in self._known_files:
                            self._known_files[filepath] = mtime
                            self._analyze_file(filepath, "created")
                        elif self._known_files[filepath] != mtime:
                            self._known_files[filepath] = mtime
                            self._analyze_file(filepath, "modified")
                    except:
                        pass
        except:
            pass
    
    def _analyze_file(self, filepath: str, action: str):
        """Analyze file for suspicious characteristics"""
        self.stats["files_monitored"] += 1
        
        try:
            filename = os.path.basename(filepath)
            ext = os.path.splitext(filename)[1].lower()
            
            # Check extension
            ext_info = SUSPICIOUS_EXTENSIONS.get(ext)
            
            if not ext_info:
                return  # Not suspicious extension
            
            # Determine location risk
            location_risk = "low"
            location_reason = ""
            
            for category, config in MONITORED_PATHS.items():
                for monitored_path in config["paths"]:
                    if monitored_path and filepath.lower().startswith(monitored_path.lower()):
                        location_risk = config["risk"]
                        location_reason = config["reason"]
                        break
            
            # Calculate overall risk
            risk_levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
            overall_risk = max(risk_levels.get(ext_info["risk"], 1), risk_levels.get(location_risk, 1))
            overall_risk_name = {1: "low", 2: "medium", 3: "high", 4: "critical"}[overall_risk]
            
            # Only alert on medium+ risk
            if overall_risk < 2:
                return
            
            self.stats["suspicious_files"] += 1
            
            if ext_info["type"] == "ransomware":
                self.stats["ransomware_detected"] += 1
            
            if ext_info["type"] == "executable" and location_risk in ["high", "critical"]:
                self.stats["executables_in_temp"] += 1
            
            # Determine MITRE technique
            mitre_map = {
                "executable": "T1204",  # User Execution
                "library": "T1574",  # Hijack Execution Flow
                "script": "T1059",  # Command and Scripting
                "powershell": "T1059.001",  # PowerShell
                "vbscript": "T1059.005",  # Visual Basic
                "batch": "T1059.003",  # Windows Command Shell
                "macro_doc": "T1566.001",  # Spearphishing Attachment
                "ransomware": "T1486",  # Data Encrypted for Impact
            }
            mitre = mitre_map.get(ext_info["type"], "T1204")
            
            # Create detection name
            detection_name = f"Suspicious {ext_info['type'].replace('_', ' ').title()}"
            if location_reason:
                detection_name = f"{location_reason}: {filename}"
            
            # Get file size
            try:
                file_size = os.path.getsize(filepath)
            except:
                file_size = 0
            
            # Publish event
            self.event_queue.create_event(
                event_type="file",
                data={
                    "filepath": filepath,
                    "filename": filename,
                    "extension": ext,
                    "file_type": ext_info["type"],
                    "action": action,
                    "size_bytes": file_size,
                    "location_risk": location_risk,
                    "location_reason": location_reason
                },
                severity=overall_risk_name,
                mitre_technique=mitre,
                detection_name=detection_name
            )
            
            print(f"📁 [{overall_risk_name.upper()}] {detection_name}")
            
        except Exception as e:
            pass  # Ignore errors for individual files
    
    def get_stats(self) -> Dict:
        """Get monitoring statistics"""
        return self.stats


# Singleton
_file_monitor = None

def get_file_monitor() -> FileMonitor:
    global _file_monitor
    if _file_monitor is None:
        _file_monitor = FileMonitor()
    return _file_monitor


if __name__ == "__main__":
    monitor = FileMonitor()
    monitor.start()
    
    print("\n📁 Monitoring files... Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(10)
            stats = monitor.get_stats()
            print(f"📊 Stats: {stats}")
    except KeyboardInterrupt:
        monitor.stop()
        print("\n✅ File monitor stopped")
