"""
EDR Event Queue
Central event bus for all EDR telemetry
"""

from typing import Dict, Any, Callable, List
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue, Empty
from threading import Thread, Event
import json
import uuid

@dataclass
class EDREvent:
    """Standardized EDR event"""
    event_id: str
    event_type: str  # process, file, registry, network, memory
    timestamp: datetime
    data: Dict[str, Any]
    severity: str = "info"  # info, low, medium, high, critical
    mitre_technique: str = ""
    detection_name: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "severity": self.severity,
            "mitre_technique": self.mitre_technique,
            "detection_name": self.detection_name
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class EventQueue:
    """
    Central event queue for EDR telemetry
    
    All collectors push events here.
    Analyzers consume from here.
    """
    
    def __init__(self, max_size: int = 100000):
        self.queue = Queue(maxsize=max_size)
        self.subscribers: List[Callable[[EDREvent], None]] = []
        self._stop_event = Event()
        self._dispatcher_thread = None
        
    def start(self):
        """Start the event dispatcher"""
        self._stop_event.clear()
        self._dispatcher_thread = Thread(target=self._dispatch_loop, daemon=True)
        self._dispatcher_thread.start()
        print("📥 EDR Event Queue started")
    
    def stop(self):
        """Stop the event dispatcher"""
        self._stop_event.set()
        if self._dispatcher_thread:
            self._dispatcher_thread.join(timeout=5)
        print("📥 EDR Event Queue stopped")
    
    def subscribe(self, callback: Callable[[EDREvent], None]):
        """Subscribe to events"""
        self.subscribers.append(callback)
    
    def publish(self, event: EDREvent):
        """Publish an event to the queue"""
        try:
            self.queue.put_nowait(event)
        except:
            # Queue full - drop oldest
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(event)
            except:
                pass
    
    def create_event(
        self,
        event_type: str,
        data: Dict[str, Any],
        severity: str = "info",
        mitre_technique: str = "",
        detection_name: str = ""
    ) -> EDREvent:
        """Create and publish an event"""
        event = EDREvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.now(),
            data=data,
            severity=severity,
            mitre_technique=mitre_technique,
            detection_name=detection_name
        )
        self.publish(event)
        return event
    
    def _dispatch_loop(self):
        """Dispatch events to subscribers"""
        while not self._stop_event.is_set():
            try:
                event = self.queue.get(timeout=0.1)
                for callback in self.subscribers:
                    try:
                        callback(event)
                    except Exception as e:
                        print(f"⚠️ Subscriber error: {e}")
            except Empty:
                continue


# Singleton instance
_event_queue = None

def get_event_queue() -> EventQueue:
    global _event_queue
    if _event_queue is None:
        _event_queue = EventQueue()
    return _event_queue
