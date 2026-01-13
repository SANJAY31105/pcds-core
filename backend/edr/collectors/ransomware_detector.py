"""
Ransomware Detector - EDR Advanced Module
Real-time ransomware behavior detection

Detects:
- File encryption patterns (high entropy)
- Mass file modifications
- Ransom note drops
- Shadow copy deletion
- Known ransomware extensions
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import Thread, Event, Lock
from collections import defaultdict
from pathlib import Path
import math
import os
import time

# Try watchdog
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False


# Known ransomware extensions
RANSOMWARE_EXTENSIONS = {
    # Common ransomware
    ".encrypted", ".enc", ".crypted", ".crypt", ".locked", ".lock",
    ".crypto", ".crinf", ".r5a", ".XRNT", ".XTBL", ".crypt6",
    
    # Specific ransomware families
    ".locky", ".zepto", ".odin", ".aesir", ".thor", ".zzzzz",  # Locky
    ".cerber", ".cerber2", ".cerber3",  # Cerber
    ".cryptolocker", ".cryptowall",  # CryptoLocker/Wall
    ".wncry", ".wcry", ".wncrypt", ".wncryt",  # WannaCry
    ".petya", ".notpetya",  # Petya/NotPetya
    ".ryuk",  # Ryuk
    ".conti",  # Conti
    ".lockbit", ".lockbit2", ".lockbit3",  # LockBit
    ".revil", ".sodinokibi",  # REvil
    ".blackcat", ".alphv",  # BlackCat
    ".hive",  # Hive
    ".royal",  # Royal
    ".play",  # Play
    
    # Generic
    ".aaa", ".abc", ".xyz", ".zzz", ".micro", ".xxx",
}

# Ransom note filenames
RANSOM_NOTES = {
    "readme.txt", "readme.html", "how_to_decrypt.txt", "decrypt_instructions.txt",
    "restore_files.txt", "your_files.txt", "how_to_recover.txt",
    "!!!read_me!!!.txt", "!!!readme!!!.txt", "_readme.txt",
    "help_decrypt.html", "help_restore.html",
    "ryuk_readme.txt", "conti_readme.txt", "lockbit_readme.txt",
    "@please_read_me@.txt", "how_to_back_files.html",
    "decrypt_your_files.html", "recovery_key.txt",
}

# Suspicious commands (shadow copy deletion, etc.)
RANSOMWARE_COMMANDS = [
    "vssadmin delete shadows",
    "wmic shadowcopy delete",
    "bcdedit /set {default} recoveryenabled no",
    "wbadmin delete catalog",
    "cipher /w:",
    "icacls * /grant everyone:f /t /c /q",
]


@dataclass
class RansomwareAlert:
    """Ransomware detection alert"""
    alert_type: str
    description: str
    severity: str
    mitre_technique: str
    affected_path: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: Dict = field(default_factory=dict)


class RansomwareDetector:
    """
    Advanced Ransomware Detector
    
    Detection methods:
    1. File entropy analysis (encrypted files have high entropy)
    2. Mass file modification detection
    3. Ransomware extension monitoring
    4. Ransom note detection
    5. Shadow copy deletion detection
    """
    
    def __init__(self):
        self._stop_event = Event()
        self._monitor_thread = None
        self._lock = Lock()
        
        # Import event queue
        try:
            from .core.event_queue import get_event_queue
            self.event_queue = get_event_queue()
        except:
            self.event_queue = None
        
        # Track file modifications
        self._file_mods: Dict[str, List[datetime]] = defaultdict(list)  # path -> timestamps
        self._entropy_cache: Dict[str, float] = {}
        
        # Alerts
        self.alerts: List[RansomwareAlert] = []
        
        # Thresholds
        self.entropy_threshold = 7.5  # Files above this are likely encrypted (max is 8)
        self.mass_mod_threshold = 50  # Files modified per minute before alert
        self.mass_mod_window = 60  # seconds
        
        # Stats
        self.stats = {
            "files_analyzed": 0,
            "high_entropy_files": 0,
            "ransomware_extensions_detected": 0,
            "ransom_notes_detected": 0,
            "mass_modification_alerts": 0,
            "critical_alerts": 0
        }
        
        print("🔐 Ransomware Detector initialized")
    
    def start(self):
        """Start ransomware detection"""
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("🔐 Ransomware Detector started")
    
    def stop(self):
        """Stop ransomware detection"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        print("🔐 Ransomware Detector stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Check for mass file modifications
                self._check_mass_modifications()
                
                # Scan critical directories for ransomware indicators
                self._scan_critical_directories()
                
                time.sleep(5)
            except Exception as e:
                print(f"⚠️ Ransomware detector error: {e}")
                time.sleep(10)
    
    def calculate_entropy(self, file_path: str, sample_size: int = 10240) -> float:
        """
        Calculate Shannon entropy of a file
        
        High entropy (>7.5) indicates encryption or compression
        Normal files typically have entropy 4-6
        """
        try:
            with open(file_path, 'rb') as f:
                data = f.read(sample_size)
            
            if not data:
                return 0.0
            
            # Calculate byte frequency
            freq = [0] * 256
            for byte in data:
                freq[byte] += 1
            
            # Calculate entropy
            length = len(data)
            entropy = 0.0
            for count in freq:
                if count > 0:
                    p = count / length
                    entropy -= p * math.log2(p)
            
            return entropy
            
        except Exception:
            return 0.0
    
    def analyze_file(self, file_path: str) -> Optional[RansomwareAlert]:
        """Analyze a file for ransomware indicators"""
        self.stats["files_analyzed"] += 1
        
        try:
            path = Path(file_path)
            filename = path.name.lower()
            extension = path.suffix.lower()
            
            # Check 1: Ransomware extension
            if extension in RANSOMWARE_EXTENSIONS:
                self.stats["ransomware_extensions_detected"] += 1
                return self._create_alert(
                    alert_type="ransomware_extension",
                    description=f"Ransomware extension detected: {extension}",
                    severity="critical",
                    mitre="T1486",
                    path=file_path,
                    details={"extension": extension}
                )
            
            # Check 2: Ransom note
            if filename in RANSOM_NOTES:
                self.stats["ransom_notes_detected"] += 1
                return self._create_alert(
                    alert_type="ransom_note",
                    description=f"Ransom note detected: {filename}",
                    severity="critical",
                    mitre="T1486",
                    path=file_path,
                    details={"filename": filename}
                )
            
            # Check 3: High entropy (encrypted file)
            if path.exists() and path.is_file():
                try:
                    file_size = path.stat().st_size
                    
                    # Only check files between 1KB and 10MB
                    if 1024 < file_size < 10 * 1024 * 1024:
                        entropy = self.calculate_entropy(file_path)
                        
                        if entropy > self.entropy_threshold:
                            self.stats["high_entropy_files"] += 1
                            
                            # Only alert if it wasn't already high entropy
                            if file_path not in self._entropy_cache or \
                               self._entropy_cache[file_path] < self.entropy_threshold:
                                self._entropy_cache[file_path] = entropy
                                
                                return self._create_alert(
                                    alert_type="high_entropy",
                                    description=f"High entropy file detected (possible encryption)",
                                    severity="high",
                                    mitre="T1486",
                                    path=file_path,
                                    details={
                                        "entropy": round(entropy, 2),
                                        "threshold": self.entropy_threshold
                                    }
                                )
                        
                        self._entropy_cache[file_path] = entropy
                except:
                    pass
            
            return None
            
        except Exception as e:
            return None
    
    def track_file_modification(self, file_path: str):
        """Track file modification for mass modification detection"""
        with self._lock:
            now = datetime.now()
            self._file_mods[file_path].append(now)
            
            # Keep only recent modifications
            cutoff = now - timedelta(seconds=self.mass_mod_window)
            self._file_mods[file_path] = [
                t for t in self._file_mods[file_path] if t > cutoff
            ]
    
    def _check_mass_modifications(self):
        """Check for mass file modifications (ransomware behavior)"""
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.mass_mod_window)
            
            # Count recent modifications
            total_mods = 0
            for path, times in self._file_mods.items():
                recent = [t for t in times if t > cutoff]
                total_mods += len(recent)
            
            if total_mods > self.mass_mod_threshold:
                self.stats["mass_modification_alerts"] += 1
                self._create_alert(
                    alert_type="mass_modification",
                    description=f"Mass file modification detected: {total_mods} files in {self.mass_mod_window}s",
                    severity="critical",
                    mitre="T1486",
                    path="multiple",
                    details={
                        "files_modified": total_mods,
                        "window_seconds": self.mass_mod_window
                    }
                )
    
    def _scan_critical_directories(self):
        """Scan critical directories for ransomware indicators"""
        critical_dirs = [
            os.path.expanduser("~\\Documents"),
            os.path.expanduser("~\\Desktop"),
            os.path.expanduser("~\\Downloads"),
        ]
        
        for dir_path in critical_dirs:
            if not os.path.exists(dir_path):
                continue
            
            try:
                for filename in os.listdir(dir_path)[:100]:  # Limit scan
                    if filename.lower() in RANSOM_NOTES:
                        file_path = os.path.join(dir_path, filename)
                        self.analyze_file(file_path)
            except:
                pass
    
    def check_command(self, command: str) -> Optional[RansomwareAlert]:
        """Check if a command is ransomware-related"""
        cmd_lower = command.lower()
        
        for ransomware_cmd in RANSOMWARE_COMMANDS:
            if ransomware_cmd.lower() in cmd_lower:
                return self._create_alert(
                    alert_type="ransomware_command",
                    description=f"Ransomware command detected: {ransomware_cmd}",
                    severity="critical",
                    mitre="T1490",  # Inhibit System Recovery
                    path=command,
                    details={"command": command, "pattern": ransomware_cmd}
                )
        
        return None
    
    def _create_alert(self, alert_type: str, description: str, severity: str,
                      mitre: str, path: str, details: Dict = None) -> RansomwareAlert:
        """Create and report ransomware alert"""
        alert = RansomwareAlert(
            alert_type=alert_type,
            description=description,
            severity=severity,
            mitre_technique=mitre,
            affected_path=path,
            details=details or {}
        )
        
        self.alerts.append(alert)
        
        if severity == "critical":
            self.stats["critical_alerts"] += 1
        
        # Keep last 500 alerts
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-500:]
        
        # Report to event queue
        if self.event_queue:
            self.event_queue.create_event(
                event_type="ransomware",
                data={
                    "alert_type": alert_type,
                    "description": description,
                    "affected_path": path,
                    "details": details
                },
                severity=severity,
                mitre_technique=mitre,
                detection_name=f"Ransomware: {description}"
            )
        
        print(f"🔐 [{severity.upper()}] RANSOMWARE: {description}")
        
        return alert
    
    def quick_scan(self, directory: str = None) -> Dict:
        """Quick ransomware scan of a directory"""
        if directory is None:
            directory = os.path.expanduser("~\\Documents")
        
        results = {
            "scan_time": datetime.now().isoformat(),
            "directory": directory,
            "files_scanned": 0,
            "ransomware_indicators": [],
            "high_entropy_files": [],
            "ransom_notes": []
        }
        
        try:
            for root, dirs, files in os.walk(directory):
                for filename in files[:500]:  # Limit for speed
                    file_path = os.path.join(root, filename)
                    results["files_scanned"] += 1
                    
                    ext = os.path.splitext(filename)[1].lower()
                    
                    # Check extension
                    if ext in RANSOMWARE_EXTENSIONS:
                        results["ransomware_indicators"].append({
                            "type": "extension",
                            "file": file_path,
                            "indicator": ext
                        })
                    
                    # Check ransom note
                    if filename.lower() in RANSOM_NOTES:
                        results["ransom_notes"].append(file_path)
                    
                    # Check entropy (sample)
                    if results["files_scanned"] % 10 == 0:  # Every 10th file
                        try:
                            entropy = self.calculate_entropy(file_path)
                            if entropy > self.entropy_threshold:
                                results["high_entropy_files"].append({
                                    "file": file_path,
                                    "entropy": round(entropy, 2)
                                })
                        except:
                            pass
        except:
            pass
        
        return results
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        return {
            **self.stats,
            "total_alerts": len(self.alerts)
        }
    
    def get_recent_alerts(self, count: int = 50) -> List[Dict]:
        """Get recent alerts"""
        return [
            {
                "alert_type": a.alert_type,
                "description": a.description,
                "severity": a.severity,
                "mitre_technique": a.mitre_technique,
                "affected_path": a.affected_path,
                "timestamp": a.timestamp.isoformat()
            }
            for a in self.alerts[-count:]
        ]


# Singleton
_ransomware_detector = None

def get_ransomware_detector() -> RansomwareDetector:
    global _ransomware_detector
    if _ransomware_detector is None:
        _ransomware_detector = RansomwareDetector()
    return _ransomware_detector


if __name__ == "__main__":
    detector = RansomwareDetector()
    
    print("\n🔐 Quick Ransomware Scan...\n")
    
    # Test entropy calculation
    import sys
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
        entropy = detector.calculate_entropy(test_file)
        print(f"Entropy of {test_file}: {entropy:.2f}")
    
    # Quick scan
    result = detector.quick_scan()
    print(f"Files scanned: {result['files_scanned']}")
    print(f"Ransomware indicators: {len(result['ransomware_indicators'])}")
    print(f"High entropy files: {len(result['high_entropy_files'])}")
    print(f"Ransom notes: {len(result['ransom_notes'])}")
