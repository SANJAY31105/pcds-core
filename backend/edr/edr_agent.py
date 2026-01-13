"""
PCDS EDR Agent Service
Main orchestrator for all EDR components

Architecture:
- Core: Event queue, Sysmon parser
- Collectors: Process, File, Registry, Network
- Analyzer: Rules, ML, Correlation
- Actions: Kill, Quarantine, Block, Isolate
- Transport: Kafka, Offline buffer
"""

from typing import Dict, Any, List, Optional
from threading import Thread, Event
from datetime import datetime
import time
import json

# Core
from .core.event_queue import get_event_queue, EDREvent

# Collectors
from .collectors.process_monitor import get_process_monitor
from .collectors.file_monitor import get_file_monitor
from .collectors.registry_monitor import get_registry_monitor
from .collectors.network_monitor import get_network_monitor

# Actions
from .actions.response_actions import get_response_actions, auto_respond

# ML Integration
from .edr_ml_integration import get_edr_ml_integration, TelemetryEvent


class EDRAgent:
    """
    PCDS EDR Agent
    
    Enterprise-grade Endpoint Detection and Response
    
    Features:
    - Real-time process monitoring
    - File system monitoring
    - Automated response
    - Confidence-based actions
    """
    
    def __init__(self, auto_response: bool = True, response_threshold: float = 0.90):
        """
        Initialize EDR Agent
        
        Args:
            auto_response: Enable automatic response actions
            response_threshold: Confidence threshold for auto-response (0.90 = 90%)
        """
        # Configuration
        self.auto_response = auto_response
        self.response_threshold = response_threshold
        
        # Components
        self.event_queue = get_event_queue()
        self.process_monitor = get_process_monitor()
        self.file_monitor = get_file_monitor()
        self.registry_monitor = get_registry_monitor()
        self.network_monitor = get_network_monitor()
        self.response_actions = get_response_actions()
        
        # ML Integration
        self.ml_integration = get_edr_ml_integration()
        
        # State
        self._running = False
        self._analysis_thread = None
        self._stop_event = Event()
        
        # Stats
        self.stats = {
            "started_at": None,
            "events_processed": 0,
            "detections": 0,
            "ml_predictions": 0,
            "auto_responses": 0,
            "false_positives_prevented": 0
        }
        
        # Detection history
        self.detection_history: List[Dict] = []
        
        # Subscribe to events
        self.event_queue.subscribe(self._handle_event)
        
        print("=" * 60)
        print("🛡️  PCDS EDR Agent + ML")
        print("=" * 60)
        print(f"   Auto Response: {auto_response}")
        print(f"   Response Threshold: {response_threshold * 100:.0f}%")
        print(f"   ML Integration: ENABLED")
        print("=" * 60)
    
    def start(self):
        """Start the EDR agent"""
        if self._running:
            print("⚠️ EDR Agent already running")
            return
        
        self._running = True
        self._stop_event.clear()
        self.stats["started_at"] = datetime.now().isoformat()
        
        # Start components
        self.event_queue.start()
        self.process_monitor.start()
        self.file_monitor.start()
        self.registry_monitor.start()
        self.network_monitor.start()
        self.ml_integration.start()  # Start ML pipeline
        
        # Start analysis thread
        self._analysis_thread = Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()
        
        print("\n🚀 EDR Agent + ML STARTED")
        print("   Monitoring: Processes, Files, Registry, Network")
        print("   ML Inference: ENABLED")
        print("   Response: " + ("ENABLED" if self.auto_response else "DISABLED"))
        print("\n")
    
    def stop(self):
        """Stop the EDR agent"""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        # Stop components
        self.ml_integration.stop()  # Stop ML pipeline
        self.process_monitor.stop()
        self.file_monitor.stop()
        self.registry_monitor.stop()
        self.network_monitor.stop()
        self.event_queue.stop()
        
        if self._analysis_thread:
            self._analysis_thread.join(timeout=5)
        
        print("\n🛑 EDR Agent STOPPED")
    
    def _handle_event(self, event: EDREvent):
        """Handle incoming EDR events"""
        self.stats["events_processed"] += 1
        
        # Submit ALL events to ML for inference
        try:
            ml_event = TelemetryEvent(
                timestamp=event.timestamp,
                event_type=event.event_type,
                source_collector="edr_agent",
                raw_data=event.data or {}
            )
            self.ml_integration.submit_event(ml_event)
            self.stats["ml_predictions"] += 1
        except Exception as e:
            pass  # Don't block on ML errors
        
        # Process detection events (rule-based)
        if event.severity in ["medium", "high", "critical"]:
            self.stats["detections"] += 1
            
            # Store in history
            self.detection_history.append({
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "severity": event.severity,
                "detection_name": event.detection_name,
                "mitre_technique": event.mitre_technique,
                "data": event.data
            })
            
            # Keep last 1000 detections
            if len(self.detection_history) > 1000:
                self.detection_history = self.detection_history[-1000:]
            
            # Auto response if enabled
            if self.auto_response:
                self._try_auto_respond(event)
    
    def _try_auto_respond(self, event: EDREvent):
        """Attempt automated response"""
        # Map severity to confidence
        severity_confidence = {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.95
        }
        
        confidence = severity_confidence.get(event.severity, 0.5)
        
        if confidence >= self.response_threshold:
            detection = {
                "type": event.event_type,
                "data": event.data,
                "severity": event.severity
            }
            
            result = auto_respond(detection, confidence, self.response_actions)
            
            if result and result.get("success"):
                self.stats["auto_responses"] += 1
                print(f"🛡️ AUTO-RESPONSE: {result.get('action')} executed")
    
    def _analysis_loop(self):
        """Background analysis loop"""
        while not self._stop_event.is_set():
            try:
                # Periodic stats logging
                time.sleep(60)  # Every minute
                
                if self._running:
                    process_stats = self.process_monitor.get_stats()
                    file_stats = self.file_monitor.get_stats()
                    
                    # Could send to backend here
                    
            except Exception as e:
                print(f"⚠️ Analysis error: {e}")
    
    def get_stats(self) -> Dict:
        """Get comprehensive agent statistics"""
        return {
            **self.stats,
            "process_monitor": self.process_monitor.get_stats(),
            "file_monitor": self.file_monitor.get_stats(),
            "registry_monitor": self.registry_monitor.get_stats(),
            "network_monitor": self.network_monitor.get_stats(),
            "response_actions": len(self.response_actions.get_action_log()),
            "ml_integration": self.ml_integration.get_stats()
        }
    
    def get_recent_detections(self, count: int = 50) -> List[Dict]:
        """Get recent detections"""
        return self.detection_history[-count:]
    
    def set_auto_response(self, enabled: bool):
        """Enable/disable auto response"""
        self.auto_response = enabled
        print(f"🛡️ Auto-response: {'ENABLED' if enabled else 'DISABLED'}")
    
    def set_response_threshold(self, threshold: float):
        """Set response threshold (0.0 - 1.0)"""
        self.response_threshold = max(0.0, min(1.0, threshold))
        print(f"🛡️ Response threshold: {self.response_threshold * 100:.0f}%")


# Singleton
_edr_agent = None

def get_edr_agent(auto_response: bool = True, response_threshold: float = 0.90) -> EDRAgent:
    """Get or create EDR agent instance"""
    global _edr_agent
    if _edr_agent is None:
        _edr_agent = EDRAgent(auto_response, response_threshold)
    return _edr_agent


def start_edr_agent(auto_response: bool = True, response_threshold: float = 0.90):
    """Quick start EDR agent"""
    agent = get_edr_agent(auto_response, response_threshold)
    agent.start()
    return agent


if __name__ == "__main__":
    # Run standalone
    agent = start_edr_agent(auto_response=True, response_threshold=0.90)
    
    print("\n🛡️ EDR Agent running... Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(30)
            stats = agent.get_stats()
            print(f"\n📊 EDR Stats:")
            print(f"   Events: {stats['events_processed']}")
            print(f"   Detections: {stats['detections']}")
            print(f"   Auto-responses: {stats['auto_responses']}")
    except KeyboardInterrupt:
        agent.stop()
        print("\n✅ EDR Agent stopped")
