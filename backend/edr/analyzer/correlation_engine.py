"""
Attack Correlation Engine - EDR Advanced Module
Correlates multiple events into attack chains

Features:
- Temporal correlation (events within time window)
- Entity correlation (same user/host/process)
- Kill chain mapping
- Attack campaign detection
"""

from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from threading import Thread, Event, Lock
import uuid


# MITRE ATT&CK Kill Chain Stages
KILL_CHAIN_STAGES = {
    "initial_access": {"mitre": ["T1566", "T1190", "T1133", "T1078"], "order": 1},
    "execution": {"mitre": ["T1059", "T1204", "T1053", "T1569"], "order": 2},
    "persistence": {"mitre": ["T1547", "T1543", "T1136", "T1546"], "order": 3},
    "privilege_escalation": {"mitre": ["T1134", "T1068", "T1548", "T1055"], "order": 4},
    "defense_evasion": {"mitre": ["T1070", "T1027", "T1036", "T1562"], "order": 5},
    "credential_access": {"mitre": ["T1003", "T1558", "T1539", "T1110"], "order": 6},
    "discovery": {"mitre": ["T1087", "T1082", "T1083", "T1057"], "order": 7},
    "lateral_movement": {"mitre": ["T1021", "T1091", "T1080", "T1570"], "order": 8},
    "collection": {"mitre": ["T1074", "T1560", "T1005", "T1114"], "order": 9},
    "exfiltration": {"mitre": ["T1041", "T1048", "T1567", "T1020"], "order": 10},
    "impact": {"mitre": ["T1486", "T1489", "T1490", "T1485"], "order": 11},
}


@dataclass
class CorrelatedEvent:
    """Single event in the correlation engine"""
    event_id: str
    event_type: str
    timestamp: datetime
    entity: str  # username, hostname, or IP
    mitre_technique: str
    kill_chain_stage: str
    severity: str
    description: str
    data: Dict = field(default_factory=dict)


@dataclass
class AttackChain:
    """Correlated attack chain"""
    chain_id: str
    entity: str
    events: List[CorrelatedEvent]
    kill_chain_stages: Set[str]
    start_time: datetime
    end_time: datetime
    severity: str
    confidence: float
    description: str
    is_complete: bool = False  # True if spans multiple kill chain stages


class CorrelationEngine:
    """
    Attack Correlation Engine
    
    Capabilities:
    - Temporal event correlation
    - Entity-based correlation
    - Kill chain stage mapping
    - Attack campaign detection
    - Alert reduction (multiple events -> one incident)
    """
    
    def __init__(self, correlation_window: int = 3600):
        """
        Initialize Correlation Engine
        
        Args:
            correlation_window: Time window in seconds for correlating events
        """
        self._stop_event = Event()
        self._lock = Lock()
        self._analysis_thread = None
        
        # Configuration
        self.correlation_window = timedelta(seconds=correlation_window)
        
        # Event storage
        self.events: List[CorrelatedEvent] = []
        self.attack_chains: Dict[str, AttackChain] = {}
        
        # Entity tracking
        self._entity_events: Dict[str, List[CorrelatedEvent]] = defaultdict(list)
        
        # MITRE to kill chain mapping
        self.mitre_to_stage = self._build_mitre_mapping()
        
        # Stats
        self.stats = {
            "total_events": 0,
            "events_correlated": 0,
            "attack_chains_created": 0,
            "complete_chains_detected": 0,
            "campaigns_detected": 0
        }
        
        print("🔗 Attack Correlation Engine initialized")
    
    def _build_mitre_mapping(self) -> Dict[str, str]:
        """Build MITRE technique to kill chain stage mapping"""
        mapping = {}
        for stage, info in KILL_CHAIN_STAGES.items():
            for mitre in info["mitre"]:
                mapping[mitre] = stage
                # Also map sub-techniques
                for i in range(10):
                    mapping[f"{mitre}.{i:03d}"] = stage
        return mapping
    
    def start(self):
        """Start correlation engine"""
        self._stop_event.clear()
        self._analysis_thread = Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()
        print("🔗 Correlation Engine started")
    
    def stop(self):
        """Stop correlation engine"""
        self._stop_event.set()
        if self._analysis_thread:
            self._analysis_thread.join(timeout=5)
        print("🔗 Correlation Engine stopped")
    
    def _analysis_loop(self):
        """Background correlation analysis"""
        while not self._stop_event.is_set():
            try:
                self._analyze_correlations()
                self._cleanup_old_events()
                time.sleep(30)
            except Exception as e:
                print(f"⚠️ Correlation engine error: {e}")
                import time
                time.sleep(10)
    
    def add_event(self, event_type: str, entity: str, mitre_technique: str,
                  severity: str, description: str, data: Dict = None) -> CorrelatedEvent:
        """
        Add an event to the correlation engine
        
        Args:
            event_type: Type of event (process, file, network, etc.)
            entity: Entity involved (username, hostname, IP)
            mitre_technique: MITRE ATT&CK technique ID
            severity: Event severity (low, medium, high, critical)
            description: Event description
            data: Additional event data
        """
        # Determine kill chain stage
        stage = self.mitre_to_stage.get(mitre_technique, "unknown")
        if stage == "unknown":
            # Try base technique
            base_technique = mitre_technique.split('.')[0]
            stage = self.mitre_to_stage.get(base_technique, "unknown")
        
        event = CorrelatedEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            entity=entity,
            mitre_technique=mitre_technique,
            kill_chain_stage=stage,
            severity=severity,
            description=description,
            data=data or {}
        )
        
        with self._lock:
            self.events.append(event)
            self._entity_events[entity].append(event)
            self.stats["total_events"] += 1
        
        # Try immediate correlation
        self._correlate_event(event)
        
        return event
    
    def _correlate_event(self, new_event: CorrelatedEvent):
        """Try to correlate a new event with existing chains"""
        entity = new_event.entity
        now = datetime.now()
        
        with self._lock:
            # Find existing chain for this entity
            existing_chain = None
            for chain_id, chain in self.attack_chains.items():
                if chain.entity == entity:
                    age = now - chain.end_time
                    if age < self.correlation_window:
                        existing_chain = chain
                        break
            
            if existing_chain:
                # Add to existing chain
                existing_chain.events.append(new_event)
                existing_chain.kill_chain_stages.add(new_event.kill_chain_stage)
                existing_chain.end_time = now
                self._update_chain_severity(existing_chain)
                self.stats["events_correlated"] += 1
                
                # Check if chain spans multiple stages (more sophisticated attack)
                if len(existing_chain.kill_chain_stages) >= 3:
                    if not existing_chain.is_complete:
                        existing_chain.is_complete = True
                        self.stats["complete_chains_detected"] += 1
                        print(f"🔗 [CRITICAL] Complete attack chain: {existing_chain.description}")
            else:
                # Create new chain
                chain_id = str(uuid.uuid4())
                chain = AttackChain(
                    chain_id=chain_id,
                    entity=entity,
                    events=[new_event],
                    kill_chain_stages={new_event.kill_chain_stage},
                    start_time=now,
                    end_time=now,
                    severity=new_event.severity,
                    confidence=0.5,
                    description=f"Attack chain on {entity}"
                )
                self.attack_chains[chain_id] = chain
                self.stats["attack_chains_created"] += 1
    
    def _update_chain_severity(self, chain: AttackChain):
        """Update chain severity and confidence based on events"""
        severities = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        
        max_severity = max(severities.get(e.severity, 1) for e in chain.events)
        chain.severity = {1: "low", 2: "medium", 3: "high", 4: "critical"}[max_severity]
        
        # Confidence increases with more events and kill chain coverage
        event_factor = min(len(chain.events) / 10, 0.5)
        stage_factor = len(chain.kill_chain_stages) / len(KILL_CHAIN_STAGES) * 0.5
        chain.confidence = min(1.0, event_factor + stage_factor)
        
        # Update description
        stages = sorted(chain.kill_chain_stages, 
                       key=lambda s: KILL_CHAIN_STAGES.get(s, {}).get("order", 99))
        chain.description = f"Attack chain: {' → '.join(stages)}"
    
    def _analyze_correlations(self):
        """Analyze all events for correlations"""
        now = datetime.now()
        
        with self._lock:
            # Group recent events by entity
            for entity, events in self._entity_events.items():
                recent = [e for e in events if now - e.timestamp < self.correlation_window]
                
                if len(recent) >= 3:  # Minimum events for meaningful correlation
                    # Check for attack progression
                    stages = set(e.kill_chain_stage for e in recent)
                    
                    if len(stages) >= 2:
                        # Multiple kill chain stages = potential attack
                        stage_order = sorted(
                            stages,
                            key=lambda s: KILL_CHAIN_STAGES.get(s, {}).get("order", 99)
                        )
                        
                        # Check for logical progression
                        if self._is_attack_progression(stage_order):
                            # This is likely a real attack
                            pass  # Already handled in _correlate_event
    
    def _is_attack_progression(self, stages: List[str]) -> bool:
        """Check if stages represent attack progression"""
        orders = [KILL_CHAIN_STAGES.get(s, {}).get("order", 99) for s in stages]
        
        # Check if generally ascending (attack progression)
        increases = sum(1 for i in range(1, len(orders)) if orders[i] > orders[i-1])
        return increases >= len(orders) / 2
    
    def _cleanup_old_events(self):
        """Clean up old events outside correlation window"""
        cutoff = datetime.now() - self.correlation_window * 2
        
        with self._lock:
            self.events = [e for e in self.events if e.timestamp > cutoff]
            
            for entity in list(self._entity_events.keys()):
                self._entity_events[entity] = [
                    e for e in self._entity_events[entity] if e.timestamp > cutoff
                ]
                if not self._entity_events[entity]:
                    del self._entity_events[entity]
    
    def get_active_chains(self) -> List[Dict]:
        """Get active attack chains"""
        now = datetime.now()
        active = []
        
        with self._lock:
            for chain_id, chain in self.attack_chains.items():
                if now - chain.end_time < self.correlation_window:
                    active.append({
                        "chain_id": chain_id,
                        "entity": chain.entity,
                        "event_count": len(chain.events),
                        "stages": list(chain.kill_chain_stages),
                        "severity": chain.severity,
                        "confidence": round(chain.confidence, 2),
                        "description": chain.description,
                        "start_time": chain.start_time.isoformat(),
                        "end_time": chain.end_time.isoformat(),
                        "is_complete": chain.is_complete
                    })
        
        return sorted(active, key=lambda x: x["confidence"], reverse=True)
    
    def get_chain_details(self, chain_id: str) -> Optional[Dict]:
        """Get detailed information about an attack chain"""
        with self._lock:
            if chain_id not in self.attack_chains:
                return None
            
            chain = self.attack_chains[chain_id]
            return {
                "chain_id": chain_id,
                "entity": chain.entity,
                "severity": chain.severity,
                "confidence": chain.confidence,
                "description": chain.description,
                "is_complete": chain.is_complete,
                "start_time": chain.start_time.isoformat(),
                "end_time": chain.end_time.isoformat(),
                "stages": list(chain.kill_chain_stages),
                "events": [
                    {
                        "event_id": e.event_id,
                        "event_type": e.event_type,
                        "timestamp": e.timestamp.isoformat(),
                        "mitre_technique": e.mitre_technique,
                        "kill_chain_stage": e.kill_chain_stage,
                        "severity": e.severity,
                        "description": e.description
                    }
                    for e in chain.events
                ]
            }
    
    def get_stats(self) -> Dict:
        """Get correlation engine statistics"""
        return {
            **self.stats,
            "active_chains": len([c for c in self.attack_chains.values() 
                                 if datetime.now() - c.end_time < self.correlation_window]),
            "total_chains": len(self.attack_chains)
        }


# Singleton
_correlation_engine = None

def get_correlation_engine() -> CorrelationEngine:
    global _correlation_engine
    if _correlation_engine is None:
        _correlation_engine = CorrelationEngine()
    return _correlation_engine


# Need time import
import time

if __name__ == "__main__":
    engine = CorrelationEngine()
    
    print("\n🔗 Correlation Engine Test\n")
    
    # Simulate attack chain
    print("Simulating attack chain...")
    
    # Step 1: Initial access
    engine.add_event(
        event_type="network",
        entity="testuser",
        mitre_technique="T1566",
        severity="medium",
        description="Phishing email delivered"
    )
    
    # Step 2: Execution
    engine.add_event(
        event_type="process",
        entity="testuser",
        mitre_technique="T1059",
        severity="high",
        description="PowerShell execution"
    )
    
    # Step 3: Persistence
    engine.add_event(
        event_type="registry",
        entity="testuser",
        mitre_technique="T1547",
        severity="high",
        description="Run key added"
    )
    
    # Step 4: Credential access
    engine.add_event(
        event_type="process",
        entity="testuser",
        mitre_technique="T1003",
        severity="critical",
        description="Mimikatz-like behavior"
    )
    
    print(f"\nActive chains: {len(engine.get_active_chains())}")
    
    for chain in engine.get_active_chains():
        print(f"\n  Chain: {chain['description']}")
        print(f"  Events: {chain['event_count']}")
        print(f"  Stages: {chain['stages']}")
        print(f"  Severity: {chain['severity']}")
        print(f"  Complete: {chain['is_complete']}")
    
    print(f"\n📊 Stats: {engine.get_stats()}")
