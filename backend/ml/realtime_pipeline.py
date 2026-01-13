"""
Real-time ML Pipeline Orchestrator
Connects Kafka streams to ML inference for live predictions

Flow: Kafka raw_events → ML Inference → Kafka ml_predictions → WebSocket → Dashboard
"""

import asyncio
import json
import threading
from datetime import datetime
from typing import Dict, Callable, Optional
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class PipelineEvent:
    """Raw event from Kafka for ML processing"""
    event_id: str
    timestamp: str
    source_ip: str
    source_host: str
    features: list
    raw_data: Dict


class RealtimeMLPipeline:
    """
    Real-time ML Pipeline that:
    1. Consumes events from Kafka (or simulated)
    2. Runs ML inference
    3. Publishes predictions back to Kafka
    4. Broadcasts to WebSocket for dashboard
    """
    
    def __init__(self):
        self.inference_server = None
        self.kafka_producer = None
        self.ws_broadcast_callback: Optional[Callable] = None
        self.running = False
        self._event_queue = asyncio.Queue()
        
        # Stats
        self.stats = {
            "events_processed": 0,
            "predictions_made": 0,
            "errors": 0,
            "start_time": None,
            "last_event_time": None
        }
        
        print("🔄 Real-time ML Pipeline initialized")
    
    def set_ws_callback(self, callback: Callable):
        """Set WebSocket broadcast callback"""
        self.ws_broadcast_callback = callback
    
    async def start(self):
        """Start the pipeline"""
        from ml.model_server import get_inference_server
        
        self.inference_server = get_inference_server()
        self.running = True
        self.stats["start_time"] = datetime.utcnow().isoformat()
        
        # Try to connect to Kafka
        await self._connect_kafka()
        
        # Start processing loop
        asyncio.create_task(self._process_loop())
        
        print("✅ Real-time ML Pipeline started")
    
    async def _connect_kafka(self):
        """Connect to Kafka for consuming/producing"""
        try:
            from confluent_kafka import Producer
            
            self.kafka_producer = Producer({
                'bootstrap.servers': 'localhost:9092',
                'client.id': 'pcds-realtime-pipeline'
            })
            print("  ✅ Kafka producer connected")
            
        except Exception as e:
            print(f"  ⚠️ Kafka not available, using local mode: {e}")
            self.kafka_producer = None
    
    async def _process_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Get event from queue (with timeout)
                try:
                    event = await asyncio.wait_for(
                        self._event_queue.get(), 
                        timeout=0.1
                    )
                    await self._process_event(event)
                except asyncio.TimeoutError:
                    pass
                    
            except Exception as e:
                self.stats["errors"] += 1
                print(f"  ⚠️ Pipeline error: {e}")
    
    async def _process_event(self, event: PipelineEvent):
        """Process single event through ML pipeline with SOAR integration"""
        try:
            # Run ML inference
            features = np.array(event.features, dtype=np.float32)
            
            prediction = self.inference_server.predict(
                features=features,
                source_ip=event.source_ip,
                source_host=event.source_host
            )
            
            self.stats["events_processed"] += 1
            self.stats["predictions_made"] += 1
            self.stats["last_event_time"] = datetime.utcnow().isoformat()
            
            # SOAR Integration: Auto-create incident for high-risk detections
            risk_level = prediction.get("risk_level", "safe")
            if risk_level in ["critical", "high_risk"]:
                await self._trigger_soar_incident(prediction, event)
            
            # Publish to Kafka
            if self.kafka_producer:
                self._publish_prediction(prediction)
            
            # Broadcast to WebSocket
            if self.ws_broadcast_callback:
                await self._broadcast_prediction(prediction)
                
        except Exception as e:
            self.stats["errors"] += 1
            print(f"  ⚠️ Event processing error: {e}")
    
    async def _trigger_soar_incident(self, prediction: Dict, event: PipelineEvent):
        """Trigger SOAR incident for high-risk detections"""
        try:
            from ml.soar_orchestrator import get_soar
            
            soar = get_soar()
            threat_class = prediction.get("threat_class", "Unknown")
            confidence = prediction.get("confidence", 0)
            risk_level = prediction.get("risk_level", "unknown")
            
            incident = soar.create_incident(
                title=f"ML Detection: {threat_class}",
                description=f"Automated detection from real-time ML pipeline. "
                           f"Confidence: {confidence:.1%}, Risk: {risk_level}",
                severity=risk_level.replace("_", " ").title(),
                source="ml_realtime_pipeline",
                attack_type=threat_class,
                affected_hosts=[event.source_host],
                iocs=[event.source_ip],
                ml_confidence=confidence
            )
            
            self.stats.setdefault("soar_incidents_created", 0)
            self.stats["soar_incidents_created"] += 1
            
            print(f"  🎯 SOAR Incident: {incident.incident_id} - {threat_class}")
            
        except Exception as e:
            print(f"  ⚠️ SOAR integration error: {e}")
    
    def _publish_prediction(self, prediction: Dict):
        """Publish prediction to Kafka"""
        try:
            msg = json.dumps(prediction, default=str).encode('utf-8')
            self.kafka_producer.produce('ml_predictions', value=msg)
            self.kafka_producer.poll(0)
        except Exception as e:
            pass
    
    async def _broadcast_prediction(self, prediction: Dict):
        """Broadcast prediction to WebSocket clients"""
        if self.ws_broadcast_callback:
            try:
                await self.ws_broadcast_callback({
                    "type": "ml_prediction",
                    "data": prediction
                })
            except Exception as e:
                pass
    
    async def inject_event(self, event: PipelineEvent):
        """Inject event for processing (used for testing/simulation)"""
        await self._event_queue.put(event)
    
    def get_stats(self) -> Dict:
        """Get pipeline stats"""
        return {
            **self.stats,
            "queue_size": self._event_queue.qsize(),
            "running": self.running,
            "kafka_connected": self.kafka_producer is not None
        }
    
    async def stop(self):
        """Stop the pipeline"""
        self.running = False
        if self.kafka_producer:
            self.kafka_producer.flush(timeout=5)
        print("🛑 Real-time ML Pipeline stopped")


class EventSimulator:
    """
    Simulates network events for testing the pipeline
    Generates realistic-looking security events
    """
    
    def __init__(self, pipeline: RealtimeMLPipeline):
        self.pipeline = pipeline
        self.running = False
        self._event_count = 0
        
        # Attack patterns for simulation
        self.attack_patterns = [
            {"type": "normal", "weight": 0.7},
            {"type": "port_scan", "weight": 0.1},
            {"type": "brute_force", "weight": 0.08},
            {"type": "dos", "weight": 0.05},
            {"type": "malware", "weight": 0.04},
            {"type": "exfiltration", "weight": 0.03},
        ]
    
    async def start(self, events_per_second: float = 2.0):
        """Start event simulation"""
        self.running = True
        interval = 1.0 / events_per_second
        
        print(f"🎲 Event Simulator started ({events_per_second} events/sec)")
        
        while self.running:
            event = self._generate_event()
            await self.pipeline.inject_event(event)
            await asyncio.sleep(interval)
    
    def _generate_event(self) -> PipelineEvent:
        """Generate a simulated security event"""
        import random
        import uuid
        
        # Pick attack type
        r = random.random()
        cumulative = 0
        event_type = "normal"
        for pattern in self.attack_patterns:
            cumulative += pattern["weight"]
            if r < cumulative:
                event_type = pattern["type"]
                break
        
        # Generate features based on event type
        features = self._generate_features(event_type)
        
        self._event_count += 1
        
        return PipelineEvent(
            event_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow().isoformat(),
            source_ip=f"192.168.{random.randint(1,254)}.{random.randint(1,254)}",
            source_host=f"host-{random.randint(1,100):03d}",
            features=features,
            raw_data={
                "type": event_type,
                "bytes": random.randint(64, 65535),
                "packets": random.randint(1, 1000),
                "protocol": random.choice(["TCP", "UDP", "ICMP"]),
                "port": random.randint(1, 65535)
            }
        )
    
    def _generate_features(self, event_type: str) -> list:
        """Generate 40 features based on event type"""
        import random
        
        # Base features (all random normal)
        features = [random.gauss(0, 1) for _ in range(40)]
        
        # Modify based on attack type
        if event_type == "port_scan":
            features[0] = random.gauss(3, 0.5)  # High unique ports
            features[5] = random.gauss(2, 0.5)  # High connection rate
        elif event_type == "brute_force":
            features[1] = random.gauss(4, 0.5)  # High failed attempts
            features[10] = random.gauss(3, 0.5)  # Repetitive pattern
        elif event_type == "dos":
            features[2] = random.gauss(5, 0.5)  # High packet rate
            features[6] = random.gauss(4, 0.5)  # Large volume
        elif event_type == "malware":
            features[15] = random.gauss(3, 0.5)  # Suspicious process
            features[20] = random.gauss(2, 0.5)  # Encoded commands
        elif event_type == "exfiltration":
            features[25] = random.gauss(4, 0.5)  # High outbound
            features[30] = random.gauss(3, 0.5)  # Unusual destination
        
        return features
    
    def stop(self):
        """Stop simulation"""
        self.running = False
        print("🛑 Event Simulator stopped")


# Global instances
_pipeline: Optional[RealtimeMLPipeline] = None
_simulator: Optional[EventSimulator] = None


def get_realtime_pipeline() -> RealtimeMLPipeline:
    """Get or create real-time pipeline"""
    global _pipeline
    if _pipeline is None:
        _pipeline = RealtimeMLPipeline()
    return _pipeline


def get_event_simulator() -> EventSimulator:
    """Get or create event simulator"""
    global _simulator, _pipeline
    if _simulator is None:
        if _pipeline is None:
            _pipeline = RealtimeMLPipeline()
        _simulator = EventSimulator(_pipeline)
    return _simulator
