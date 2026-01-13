"""
Kafka Transport - EDR Telemetry Pipeline
Sends EDR events to backend via Kafka (or local buffer)

Features:
- Async event sending
- Offline buffering (when Kafka unavailable)
- Batch processing
- Configurable topics
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from threading import Thread, Event, Lock
from queue import Queue, Empty
import json
import time
from pathlib import Path

# Try to import kafka-python
try:
    from kafka import KafkaProducer
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("⚠️ kafka-python not installed. Using local buffer only.")

from ..core.event_queue import get_event_queue, EDREvent


@dataclass
class TelemetryConfig:
    """Kafka transport configuration"""
    bootstrap_servers: str = "localhost:9092"
    topic_events: str = "edr-events"
    topic_alerts: str = "edr-alerts"
    topic_metrics: str = "edr-metrics"
    batch_size: int = 100
    flush_interval: int = 5  # seconds
    offline_buffer_path: str = "C:/ProgramData/PCDS/buffer"
    max_buffer_size: int = 100000


class OfflineBuffer:
    """
    Local buffer for when Kafka is unavailable
    
    Stores events to disk and replays when connection restored
    """
    
    def __init__(self, buffer_path: str, max_size: int = 100000):
        self.buffer_path = Path(buffer_path)
        self.buffer_path.mkdir(parents=True, exist_ok=True)
        self.buffer_file = self.buffer_path / "events_buffer.jsonl"
        self.max_size = max_size
        self._lock = Lock()
        self._count = 0
        
        # Count existing events
        if self.buffer_file.exists():
            with open(self.buffer_file, 'r') as f:
                self._count = sum(1 for _ in f)
    
    def add(self, event: Dict) -> bool:
        """Add event to buffer"""
        if self._count >= self.max_size:
            return False
        
        with self._lock:
            with open(self.buffer_file, 'a') as f:
                f.write(json.dumps(event) + '\n')
            self._count += 1
        
        return True
    
    def get_batch(self, batch_size: int = 100) -> List[Dict]:
        """Get batch of events from buffer"""
        events = []
        
        with self._lock:
            if not self.buffer_file.exists():
                return events
            
            # Read all lines
            with open(self.buffer_file, 'r') as f:
                lines = f.readlines()
            
            # Get batch
            for line in lines[:batch_size]:
                try:
                    events.append(json.loads(line.strip()))
                except:
                    pass
            
            # Write remaining lines back
            remaining = lines[batch_size:]
            with open(self.buffer_file, 'w') as f:
                f.writelines(remaining)
            
            self._count = len(remaining)
        
        return events
    
    def count(self) -> int:
        """Get buffered event count"""
        return self._count
    
    def clear(self):
        """Clear buffer"""
        with self._lock:
            if self.buffer_file.exists():
                self.buffer_file.unlink()
            self._count = 0


class KafkaTransport:
    """
    Kafka Transport Layer for EDR
    
    Sends EDR events to Kafka topics with offline buffering
    """
    
    def __init__(self, config: TelemetryConfig = None):
        self.config = config or TelemetryConfig()
        self.event_queue = get_event_queue()
        
        # Kafka producer
        self.producer = None
        self.is_connected = False
        
        # Offline buffer
        self.buffer = OfflineBuffer(
            self.config.offline_buffer_path,
            self.config.max_buffer_size
        )
        
        # Event batch
        self._batch: List[Dict] = []
        self._batch_lock = Lock()
        
        # Threading
        self._stop_event = Event()
        self._sender_thread = None
        self._replay_thread = None
        
        # Stats
        self.stats = {
            "events_sent": 0,
            "events_buffered": 0,
            "events_failed": 0,
            "batches_sent": 0,
            "connection_retries": 0
        }
        
        # Subscribe to EDR events
        self.event_queue.subscribe(self._on_event)
        
        print("📤 Kafka Transport initialized")
    
    def start(self):
        """Start transport layer"""
        self._stop_event.clear()
        
        # Try to connect to Kafka
        self._connect()
        
        # Start sender thread
        self._sender_thread = Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()
        
        # Start replay thread (for buffered events)
        self._replay_thread = Thread(target=self._replay_loop, daemon=True)
        self._replay_thread.start()
        
        print("📤 Kafka Transport started")
    
    def stop(self):
        """Stop transport layer"""
        self._stop_event.set()
        
        # Flush remaining events
        self._flush_batch()
        
        # Close producer
        if self.producer:
            self.producer.close()
        
        if self._sender_thread:
            self._sender_thread.join(timeout=5)
        if self._replay_thread:
            self._replay_thread.join(timeout=5)
        
        print("📤 Kafka Transport stopped")
    
    def _connect(self) -> bool:
        """Connect to Kafka"""
        if not KAFKA_AVAILABLE:
            print("   ⚠️ Kafka not available - using local buffer")
            return False
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.config.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=3,
                acks='all'
            )
            self.is_connected = True
            print(f"   ✅ Connected to Kafka: {self.config.bootstrap_servers}")
            return True
        except Exception as e:
            print(f"   ⚠️ Kafka connection failed: {e}")
            self.is_connected = False
            self.stats["connection_retries"] += 1
            return False
    
    def _on_event(self, event: EDREvent):
        """Handle incoming EDR event"""
        event_dict = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "severity": event.severity,
            "mitre_technique": event.mitre_technique,
            "detection_name": event.detection_name,
            "data": event.data
        }
        
        with self._batch_lock:
            self._batch.append(event_dict)
            
            # Flush if batch is full
            if len(self._batch) >= self.config.batch_size:
                self._flush_batch()
    
    def _flush_batch(self):
        """Flush event batch"""
        with self._batch_lock:
            if not self._batch:
                return
            
            batch = self._batch.copy()
            self._batch.clear()
        
        for event in batch:
            self._send_event(event)
        
        self.stats["batches_sent"] += 1
    
    def _send_event(self, event: Dict):
        """Send event to Kafka or buffer"""
        # Determine topic based on severity
        if event.get("severity") in ["high", "critical"]:
            topic = self.config.topic_alerts
        else:
            topic = self.config.topic_events
        
        if self.is_connected and self.producer:
            try:
                self.producer.send(topic, event)
                self.stats["events_sent"] += 1
            except Exception as e:
                # Buffer on failure
                self.buffer.add(event)
                self.stats["events_buffered"] += 1
                self.is_connected = False
        else:
            # Buffer when not connected
            self.buffer.add(event)
            self.stats["events_buffered"] += 1
    
    def _sender_loop(self):
        """Background sender loop"""
        while not self._stop_event.is_set():
            try:
                time.sleep(self.config.flush_interval)
                self._flush_batch()
                
                # Flush producer
                if self.producer:
                    self.producer.flush()
                    
            except Exception as e:
                print(f"⚠️ Sender error: {e}")
    
    def _replay_loop(self):
        """Replay buffered events when connection restored"""
        while not self._stop_event.is_set():
            try:
                time.sleep(30)  # Check every 30 seconds
                
                # Check if we have buffered events
                buffered_count = self.buffer.count()
                if buffered_count == 0:
                    continue
                
                # Try to reconnect if needed
                if not self.is_connected:
                    self._connect()
                
                # Replay buffered events
                if self.is_connected and self.producer:
                    batch = self.buffer.get_batch(100)
                    for event in batch:
                        try:
                            topic = self.config.topic_events
                            if event.get("severity") in ["high", "critical"]:
                                topic = self.config.topic_alerts
                            self.producer.send(topic, event)
                            self.stats["events_sent"] += 1
                        except:
                            # Put back in buffer
                            self.buffer.add(event)
                    
                    self.producer.flush()
                    print(f"📤 Replayed {len(batch)} buffered events")
                    
            except Exception as e:
                print(f"⚠️ Replay error: {e}")
    
    def send_metrics(self, metrics: Dict):
        """Send metrics to metrics topic"""
        metrics["timestamp"] = datetime.now().isoformat()
        
        if self.is_connected and self.producer:
            try:
                self.producer.send(self.config.topic_metrics, metrics)
            except:
                pass
    
    def get_stats(self) -> Dict:
        """Get transport statistics"""
        return {
            **self.stats,
            "is_connected": self.is_connected,
            "buffered_events": self.buffer.count(),
            "pending_batch": len(self._batch)
        }


# Singleton
_kafka_transport = None

def get_kafka_transport(config: TelemetryConfig = None) -> KafkaTransport:
    global _kafka_transport
    if _kafka_transport is None:
        _kafka_transport = KafkaTransport(config)
    return _kafka_transport


if __name__ == "__main__":
    transport = KafkaTransport()
    transport.start()
    
    print("\n📤 Kafka Transport running... Press Ctrl+C to stop\n")
    
    try:
        while True:
            time.sleep(10)
            stats = transport.get_stats()
            print(f"📊 Stats: {stats}")
    except KeyboardInterrupt:
        transport.stop()
        print("\n✅ Kafka Transport stopped")
