"""
EDR ML Integration Layer
Connects EDR agent telemetry to ML inference pipeline

Pipeline:
features → inference → decision → action

This is how you go from:
ML research project → ENTERPRISE SECURITY PRODUCT
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from threading import Thread, Lock, Event
from queue import Queue
import time


@dataclass
class TelemetryEvent:
    """Standardized telemetry from EDR collectors"""
    timestamp: datetime
    event_type: str  # process, file, registry, network, memory
    source_collector: str
    raw_data: Dict
    features: List[float] = None


@dataclass
class ThreatDecision:
    """Decision output from ML pipeline"""
    event: TelemetryEvent
    threat_class: str
    confidence: float
    risk_level: str  # safe, suspicious, high_risk, critical
    recommended_action: str  # none, log, review, isolate
    took_action: bool = False
    action_result: str = None


class EDRMLIntegration:
    """
    Integrates EDR agent with ML inference
    
    Pipeline:
    1. Receive telemetry from collectors
    2. Extract features
    3. Run inference
    4. Make decision
    5. Execute action (if automated)
    
    Features:
    - Real-time inference
    - Action automation
    - Analyst queue
    - Performance optimization
    """
    
    def __init__(self, 
                 enable_auto_actions: bool = False,
                 inference_batch_size: int = 10,
                 inference_interval_ms: int = 100):
        """
        Initialize EDR ML integration
        
        Args:
            enable_auto_actions: Auto-execute actions for critical threats
            inference_batch_size: Batch events for efficiency
            inference_interval_ms: Process queue every N ms
        """
        self._lock = Lock()
        self._stop_event = Event()
        
        self.enable_auto_actions = enable_auto_actions
        self.inference_batch_size = inference_batch_size
        self.inference_interval_ms = inference_interval_ms
        
        # Event queue
        self.event_queue: Queue = Queue()
        
        # Analyst review queue
        self.analyst_queue: List[ThreatDecision] = []
        
        # ML components
        self.inference_engine = None
        self.ensemble_engine = None
        
        # Response actions
        self.response_actions = None
        
        # Stats
        self.stats = {
            "events_processed": 0,
            "threats_detected": 0,
            "auto_actions_taken": 0,
            "analyst_reviews_queued": 0,
            "avg_inference_time_ms": 0.0
        }
        
        # Callbacks
        self.threat_callbacks: List[Callable] = []
        
        # Background thread
        self._inference_thread = None
        
        self._load_components()
        
        print("🔗 EDR ML Integration initialized")
        print(f"   Auto-actions: {self.enable_auto_actions}")
    
    def _load_components(self):
        """Load ML components and response actions"""
        try:
            from ml.inference_engine import get_inference_engine
            self.inference_engine = get_inference_engine()
            print("   ✅ Inference engine loaded")
        except Exception as e:
            print(f"   ⚠️ Inference engine: {e}")
        
        try:
            from ml.ensemble_engine import get_ensemble_engine
            self.ensemble_engine = get_ensemble_engine()
            print("   ✅ Ensemble engine loaded")
        except Exception as e:
            print(f"   ⚠️ Ensemble engine: {e}")
        
        try:
            from edr.actions.response_actions import get_response_actions
            self.response_actions = get_response_actions()
            print("   ✅ Response actions loaded")
        except Exception as e:
            print(f"   ⚠️ Response actions: {e}")
    
    def start(self):
        """Start background inference thread"""
        if self._inference_thread and self._inference_thread.is_alive():
            return
        
        self._stop_event.clear()
        self._inference_thread = Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()
        print("🔗 EDR ML Integration started")
    
    def stop(self):
        """Stop background inference"""
        self._stop_event.set()
        if self._inference_thread:
            self._inference_thread.join(timeout=5)
        print("🔗 EDR ML Integration stopped")
    
    def submit_event(self, event: TelemetryEvent):
        """
        Submit telemetry event for processing
        
        Called by EDR collectors (process, file, network, etc.)
        """
        self.event_queue.put(event)
    
    def submit_raw_event(self, event_type: str, data: Dict, source: str = "edr"):
        """Submit raw event data"""
        event = TelemetryEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            source_collector=source,
            raw_data=data
        )
        
        # Extract features
        event.features = self._extract_features(event)
        
        self.submit_event(event)
    
    def _extract_features(self, event: TelemetryEvent) -> List[float]:
        """Extract ML features from telemetry event"""
        features = []
        
        # Event type encoding
        event_types = ["process", "file", "registry", "network", "memory", "dns"]
        for et in event_types:
            features.append(1.0 if event.event_type == et else 0.0)
        
        # Time features
        features.append(event.timestamp.hour / 24)
        features.append(event.timestamp.weekday() / 7)
        features.append(1.0 if event.timestamp.hour < 6 or event.timestamp.hour > 22 else 0.0)
        
        # Extract from raw data
        data = event.raw_data or {}
        
        # Process features
        if "pid" in data:
            features.append(data["pid"] / 65535)
        else:
            features.append(0.0)
        
        if "ppid" in data:
            features.append(data["ppid"] / 65535)
        else:
            features.append(0.0)
        
        # Network features
        if "dst_port" in data:
            features.append(data["dst_port"] / 65535)
        else:
            features.append(0.0)
        
        if "bytes_sent" in data:
            features.append(min(data["bytes_sent"] / 1e6, 1.0))
        else:
            features.append(0.0)
        
        # File features
        if "file_size" in data:
            features.append(min(data["file_size"] / 1e9, 1.0))
        else:
            features.append(0.0)
        
        # Registry features
        if "value_size" in data:
            features.append(min(data["value_size"] / 1e6, 1.0))
        else:
            features.append(0.0)
        
        # Suspicious indicators
        process_name = data.get("process_name", "").lower()
        suspicious = ["powershell", "cmd", "wscript", "cscript", "mshta", "regsvr32",
                     "rundll32", "certutil", "bitsadmin", "msiexec"]
        features.append(1.0 if any(s in process_name for s in suspicious) else 0.0)
        
        # Command line length (long = potentially obfuscated)
        cmdline = data.get("command_line", "")
        features.append(min(len(cmdline) / 1000, 1.0))
        
        # Has remote IP
        features.append(1.0 if data.get("remote_ip") else 0.0)
        
        # Is system account
        user = data.get("user", "").lower()
        features.append(1.0 if "system" in user else 0.0)
        features.append(1.0 if "admin" in user else 0.0)
        
        # Pad to 40 features (model input size)
        while len(features) < 40:
            features.append(0.0)
        
        return features[:40]
    
    def _inference_loop(self):
        """Background inference processing loop"""
        batch = []
        
        while not self._stop_event.is_set():
            try:
                # Collect batch
                while len(batch) < self.inference_batch_size:
                    try:
                        event = self.event_queue.get(timeout=self.inference_interval_ms / 1000)
                        batch.append(event)
                    except:
                        break
                
                if batch:
                    self._process_batch(batch)
                    batch = []
                else:
                    time.sleep(self.inference_interval_ms / 1000)
                    
            except Exception as e:
                print(f"⚠️ Inference loop error: {e}")
    
    def _process_batch(self, events: List[TelemetryEvent]):
        """Process a batch of events"""
        for event in events:
            try:
                decision = self._process_event(event)
                
                with self._lock:
                    self.stats["events_processed"] += 1
                
                if decision and decision.risk_level in ["high_risk", "critical"]:
                    with self._lock:
                        self.stats["threats_detected"] += 1
                    
                    # Execute callbacks
                    for callback in self.threat_callbacks:
                        try:
                            callback(decision)
                        except:
                            pass
                    
            except Exception as e:
                print(f"⚠️ Event processing error: {e}")
    
    def _process_event(self, event: TelemetryEvent) -> Optional[ThreatDecision]:
        """Process a single event through the pipeline"""
        if not event.features or not self.inference_engine:
            return None
        
        start_time = time.time()
        
        # Run inference
        result = self.inference_engine.predict(np.array(event.features))
        
        # Update timing stats
        inference_time = (time.time() - start_time) * 1000
        with self._lock:
            self.stats["avg_inference_time_ms"] = (
                self.stats["avg_inference_time_ms"] * (self.stats["events_processed"]) +
                inference_time
            ) / (self.stats["events_processed"] + 1)
        
        # Create decision
        decision = ThreatDecision(
            event=event,
            threat_class=result.class_name,
            confidence=result.confidence,
            risk_level=result.risk_level.value,
            recommended_action=result.action.value
        )
        
        # Handle based on risk level
        if result.risk_level.value == "critical" and self.enable_auto_actions:
            # Auto-isolate for critical
            decision = self._execute_action(decision)
        elif result.risk_level.value == "high_risk":
            # Queue for analyst
            self._queue_for_analyst(decision)
        
        return decision
    
    def _execute_action(self, decision: ThreatDecision) -> ThreatDecision:
        """Execute automated response action"""
        if not self.response_actions:
            return decision
        
        try:
            action = decision.recommended_action
            data = decision.event.raw_data
            
            if action == "auto_isolate":
                # Kill process and isolate
                if "pid" in data:
                    result = self.response_actions.kill_process(data["pid"])
                    decision.took_action = True
                    decision.action_result = f"Killed PID {data['pid']}"
                    
                    with self._lock:
                        self.stats["auto_actions_taken"] += 1
                        
        except Exception as e:
            decision.action_result = f"Action failed: {e}"
        
        return decision
    
    def _queue_for_analyst(self, decision: ThreatDecision):
        """Add to analyst review queue"""
        with self._lock:
            self.analyst_queue.append(decision)
            self.stats["analyst_reviews_queued"] += 1
            
            # Limit queue size
            if len(self.analyst_queue) > 1000:
                self.analyst_queue = self.analyst_queue[-1000:]
    
    def add_threat_callback(self, callback: Callable):
        """Add callback for threat detections"""
        self.threat_callbacks.append(callback)
    
    def get_analyst_queue(self) -> List[Dict]:
        """Get analyst review queue"""
        with self._lock:
            return [
                {
                    "timestamp": d.event.timestamp.isoformat(),
                    "event_type": d.event.event_type,
                    "threat_class": d.threat_class,
                    "confidence": d.confidence,
                    "risk_level": d.risk_level,
                    "recommended_action": d.recommended_action,
                    "raw_data": d.event.raw_data
                }
                for d in self.analyst_queue[-100:]  # Last 100
            ]
    
    def get_stats(self) -> Dict:
        """Get integration statistics"""
        with self._lock:
            return {
                **self.stats,
                "queue_size": self.event_queue.qsize(),
                "analyst_queue_size": len(self.analyst_queue)
            }


# Singleton
_edr_ml_integration = None

def get_edr_ml_integration() -> EDRMLIntegration:
    global _edr_ml_integration
    if _edr_ml_integration is None:
        _edr_ml_integration = EDRMLIntegration()
    return _edr_ml_integration


if __name__ == "__main__":
    print("=" * 60)
    print("🔗 EDR ML INTEGRATION TEST")
    print("=" * 60)
    
    integration = EDRMLIntegration(enable_auto_actions=False)
    integration.start()
    
    # Simulate events
    print("\n📍 Submitting test events...")
    
    test_events = [
        {"type": "process", "data": {"pid": 1234, "process_name": "notepad.exe"}},
        {"type": "process", "data": {"pid": 5678, "process_name": "powershell.exe", "command_line": "-enc " + "A" * 500}},
        {"type": "network", "data": {"remote_ip": "185.220.101.1", "dst_port": 4444}},
        {"type": "file", "data": {"path": "C:\\Users\\test\\malware.exe", "file_size": 1024}},
    ]
    
    for event in test_events:
        integration.submit_raw_event(event["type"], event["data"])
    
    # Wait for processing
    import time
    time.sleep(2)
    
    print(f"\n📊 Stats: {integration.get_stats()}")
    print(f"📋 Analyst Queue: {len(integration.get_analyst_queue())} items")
    
    integration.stop()
