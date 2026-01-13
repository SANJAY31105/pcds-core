"""
Behavioral Analytics Engine - EDR Advanced Module
User and Entity Behavior Analytics (UEBA)

Detects:
- Anomalous user behavior
- Lateral movement patterns
- Privilege escalation attempts
- Data exfiltration patterns
- Living off the land (LOTL) attacks
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Thread, Event, Lock
import statistics
import time
import psutil


@dataclass
class UserProfile:
    """User behavior profile"""
    username: str
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    # Process behavior
    typical_processes: Set[str] = field(default_factory=set)
    process_count_baseline: float = 0.0
    process_count_std: float = 10.0
    
    # Network behavior
    typical_connections: Set[str] = field(default_factory=set)
    connection_count_baseline: float = 0.0
    
    # File behavior
    typical_paths: Set[str] = field(default_factory=set)
    
    # Time behavior
    typical_hours: Set[int] = field(default_factory=set)
    
    # Risk score
    risk_score: float = 0.0
    alerts: int = 0


@dataclass
class BehaviorAlert:
    """Behavioral anomaly alert"""
    alert_type: str
    username: str
    description: str
    severity: str
    mitre_technique: str
    risk_delta: float
    details: Dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BehavioralAnalytics:
    """
    User and Entity Behavior Analytics (UEBA)
    
    Builds behavioral baselines and detects:
    - Process anomalies
    - Network anomalies
    - Time-based anomalies
    - Privilege escalation
    - Lateral movement
    """
    
    def __init__(self, learning_period_days: int = 7):
        self._stop_event = Event()
        self._lock = Lock()
        self._monitor_thread = None
        
        # User profiles
        self.profiles: Dict[str, UserProfile] = {}
        
        # Learning mode
        self.learning_period = timedelta(days=learning_period_days)
        self.started_at = datetime.now()
        
        # Behavioral history
        self._process_history: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)
        self._connection_history: Dict[str, List[Tuple[datetime, int]]] = defaultdict(list)
        
        # Alerts
        self.alerts: List[BehaviorAlert] = []
        
        # Attack pattern signatures
        self.attack_patterns = self._load_attack_patterns()
        
        # Stats
        self.stats = {
            "users_profiled": 0,
            "anomalies_detected": 0,
            "lateral_movement_detected": 0,
            "privilege_escalation_detected": 0,
            "lotl_attacks_detected": 0
        }
        
        print("🧬 Behavioral Analytics Engine initialized")
        print(f"   Learning period: {learning_period_days} days")
    
    def _load_attack_patterns(self) -> Dict:
        """Load attack pattern signatures"""
        return {
            # Lateral movement patterns
            "lateral_movement": {
                "processes": ["psexec", "wmic", "winrm", "net.exe", "schtasks"],
                "network_spike_threshold": 3.0,
                "mitre": "T1021"
            },
            
            # Privilege escalation
            "privilege_escalation": {
                "processes": ["runas", "powershell", "cmd.exe"],
                "indicators": ["privilege::debug", "token::elevate", "getsystem"],
                "mitre": "T1134"
            },
            
            # Reconnaissance
            "reconnaissance": {
                "processes": ["whoami", "net.exe", "ipconfig", "systeminfo", "nltest", "dsquery"],
                "time_window": 60,  # seconds
                "count_threshold": 3,
                "mitre": "T1087"
            },
            
            # Data staging
            "data_staging": {
                "processes": ["rar", "7z", "zip", "tar", "compress"],
                "extensions": [".rar", ".7z", ".zip", ".tar.gz"],
                "mitre": "T1074"
            },
            
            # LOTL (Living Off The Land)
            "lotl": {
                "processes": [
                    "powershell", "cmd", "wmic", "cscript", "wscript",
                    "mshta", "regsvr32", "rundll32", "certutil", "bitsadmin"
                ],
                "suspicious_args": [
                    "-enc", "-encodedcommand", "downloadstring", "invoke-",
                    "-nop", "-noprofile", "bypass", "hidden"
                ],
                "mitre": "T1059"
            }
        }
    
    def start(self):
        """Start behavioral monitoring"""
        self._stop_event.clear()
        self._monitor_thread = Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        print("🧬 Behavioral Analytics started")
    
    def stop(self):
        """Stop behavioral monitoring"""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        print("🧬 Behavioral Analytics stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._stop_event.is_set():
            try:
                self._collect_behavior_data()
                self._analyze_behaviors()
                time.sleep(30)  # Analyze every 30 seconds
            except Exception as e:
                print(f"⚠️ Behavioral analytics error: {e}")
                time.sleep(10)
    
    def _collect_behavior_data(self):
        """Collect current system behavior data"""
        now = datetime.now()
        current_hour = now.hour
        
        # Get processes by user
        user_processes: Dict[str, List[str]] = defaultdict(list)
        
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cmdline']):
            try:
                info = proc.info
                username = info['username']
                if username and '\\' in username:
                    username = username.split('\\')[-1]
                
                if not username:
                    continue
                
                process_name = info['name'].lower() if info['name'] else ""
                cmdline = ' '.join(info['cmdline']).lower() if info['cmdline'] else ""
                
                user_processes[username].append(process_name)
                
                # Update user profile
                self._update_profile(username, process_name, current_hour)
                
                # Check for attack patterns
                self._check_attack_patterns(username, process_name, cmdline)
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Record process counts
        for username, processes in user_processes.items():
            self._process_history[username].append((now, len(processes)))
            
            # Keep last 24 hours
            cutoff = now - timedelta(hours=24)
            self._process_history[username] = [
                (t, c) for t, c in self._process_history[username] if t > cutoff
            ]
    
    def _update_profile(self, username: str, process_name: str, hour: int):
        """Update user behavior profile"""
        with self._lock:
            if username not in self.profiles:
                self.profiles[username] = UserProfile(username=username)
                self.stats["users_profiled"] += 1
            
            profile = self.profiles[username]
            profile.last_seen = datetime.now()
            profile.typical_processes.add(process_name)
            profile.typical_hours.add(hour)
            
            # Limit set sizes
            if len(profile.typical_processes) > 100:
                # Keep most common (simplified - just limit)
                profile.typical_processes = set(list(profile.typical_processes)[:100])
    
    def _analyze_behaviors(self):
        """Analyze collected behaviors for anomalies"""
        now = datetime.now()
        is_learning = (now - self.started_at) < self.learning_period
        
        if is_learning:
            return  # Still in learning mode
        
        for username, profile in list(self.profiles.items()):
            # Check time anomaly
            current_hour = now.hour
            if profile.typical_hours and current_hour not in profile.typical_hours:
                self._create_alert(
                    alert_type="unusual_time",
                    username=username,
                    description=f"Activity at unusual hour: {current_hour}:00",
                    severity="medium",
                    mitre="T1078",
                    risk_delta=0.1,
                    details={"hour": current_hour, "typical": list(profile.typical_hours)}
                )
            
            # Check process count anomaly
            history = self._process_history.get(username, [])
            if len(history) >= 10:
                counts = [c for t, c in history]
                mean = statistics.mean(counts)
                std = statistics.stdev(counts) if len(counts) > 1 else 10
                
                current_count = counts[-1] if counts else 0
                z_score = (current_count - mean) / std if std > 0 else 0
                
                if z_score > 3:  # 3 standard deviations
                    self._create_alert(
                        alert_type="process_spike",
                        username=username,
                        description=f"Unusual process count spike: {current_count} (baseline: {mean:.1f})",
                        severity="high",
                        mitre="T1059",
                        risk_delta=0.2,
                        details={"count": current_count, "baseline": mean, "z_score": z_score}
                    )
    
    def _check_attack_patterns(self, username: str, process_name: str, cmdline: str):
        """Check for known attack patterns"""
        
        # Check LOTL
        lotl = self.attack_patterns["lotl"]
        if process_name in lotl["processes"]:
            for suspicious in lotl["suspicious_args"]:
                if suspicious in cmdline:
                    self.stats["lotl_attacks_detected"] += 1
                    self._create_alert(
                        alert_type="lotl_attack",
                        username=username,
                        description=f"Living Off The Land attack: {process_name} with {suspicious}",
                        severity="critical",
                        mitre=lotl["mitre"],
                        risk_delta=0.3,
                        details={"process": process_name, "indicator": suspicious}
                    )
                    break
        
        # Check lateral movement
        lateral = self.attack_patterns["lateral_movement"]
        if process_name in lateral["processes"]:
            self.stats["lateral_movement_detected"] += 1
            self._create_alert(
                alert_type="lateral_movement",
                username=username,
                description=f"Possible lateral movement: {process_name}",
                severity="high",
                mitre=lateral["mitre"],
                risk_delta=0.2,
                details={"process": process_name}
            )
        
        # Check privilege escalation
        priv_esc = self.attack_patterns["privilege_escalation"]
        for indicator in priv_esc["indicators"]:
            if indicator in cmdline:
                self.stats["privilege_escalation_detected"] += 1
                self._create_alert(
                    alert_type="privilege_escalation",
                    username=username,
                    description=f"Privilege escalation attempt: {indicator}",
                    severity="critical",
                    mitre=priv_esc["mitre"],
                    risk_delta=0.4,
                    details={"indicator": indicator, "process": process_name}
                )
                break
    
    def _create_alert(self, alert_type: str, username: str, description: str,
                      severity: str, mitre: str, risk_delta: float, details: Dict = None):
        """Create behavioral alert"""
        alert = BehaviorAlert(
            alert_type=alert_type,
            username=username,
            description=description,
            severity=severity,
            mitre_technique=mitre,
            risk_delta=risk_delta,
            details=details or {}
        )
        
        self.alerts.append(alert)
        self.stats["anomalies_detected"] += 1
        
        # Update user risk score
        with self._lock:
            if username in self.profiles:
                self.profiles[username].risk_score = min(
                    1.0, self.profiles[username].risk_score + risk_delta
                )
                self.profiles[username].alerts += 1
        
        # Keep last 500 alerts
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-500:]
        
        print(f"🧬 [{severity.upper()}] {alert_type}: {description}")
        
        return alert
    
    def get_user_risk(self, username: str) -> Dict:
        """Get user risk assessment"""
        with self._lock:
            if username in self.profiles:
                profile = self.profiles[username]
                return {
                    "username": username,
                    "risk_score": profile.risk_score,
                    "alerts": profile.alerts,
                    "first_seen": profile.first_seen.isoformat(),
                    "last_seen": profile.last_seen.isoformat(),
                    "typical_processes": len(profile.typical_processes),
                    "typical_hours": list(profile.typical_hours)
                }
            return {"username": username, "risk_score": 0, "alerts": 0}
    
    def get_high_risk_users(self, threshold: float = 0.5) -> List[Dict]:
        """Get users with high risk scores"""
        high_risk = []
        with self._lock:
            for username, profile in self.profiles.items():
                if profile.risk_score >= threshold:
                    high_risk.append({
                        "username": username,
                        "risk_score": profile.risk_score,
                        "alerts": profile.alerts
                    })
        return sorted(high_risk, key=lambda x: x["risk_score"], reverse=True)
    
    def get_stats(self) -> Dict:
        """Get analytics statistics"""
        is_learning = (datetime.now() - self.started_at) < self.learning_period
        return {
            **self.stats,
            "is_learning": is_learning,
            "learning_ends": (self.started_at + self.learning_period).isoformat() if is_learning else None,
            "total_alerts": len(self.alerts)
        }
    
    def get_recent_alerts(self, count: int = 50) -> List[Dict]:
        """Get recent behavioral alerts"""
        return [
            {
                "alert_type": a.alert_type,
                "username": a.username,
                "description": a.description,
                "severity": a.severity,
                "mitre_technique": a.mitre_technique,
                "risk_delta": a.risk_delta,
                "timestamp": a.timestamp.isoformat()
            }
            for a in self.alerts[-count:]
        ]


# Singleton
_behavioral_analytics = None

def get_behavioral_analytics() -> BehavioralAnalytics:
    global _behavioral_analytics
    if _behavioral_analytics is None:
        _behavioral_analytics = BehavioralAnalytics()
    return _behavioral_analytics


if __name__ == "__main__":
    analytics = BehavioralAnalytics(learning_period_days=0)  # Skip learning for test
    
    print("\n🧬 Quick Behavioral Analysis...\n")
    
    # Collect data
    analytics._collect_behavior_data()
    
    print(f"Users profiled: {analytics.stats['users_profiled']}")
    
    # Show profiles
    for username, profile in list(analytics.profiles.items())[:5]:
        print(f"\nUser: {username}")
        print(f"  Processes: {len(profile.typical_processes)}")
        print(f"  Risk Score: {profile.risk_score}")
