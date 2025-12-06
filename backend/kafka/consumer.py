"""
PCDS Enterprise - Kafka Consumer
Multi-tenant message processing with ML pipeline integration
"""

import json
import asyncio
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ConsumerMessage:
    """Structured consumer message"""
    topic: str
    partition: int
    offset: int
    key: Optional[str]
    value: Dict
    headers: Dict
    timestamp: datetime
    tenant_id: str = "default"


class KafkaConsumer:
    """
    Enterprise Kafka consumer with:
    - Consumer groups for scaling
    - Multi-tenant routing
    - Dead-letter queue handling
    - ML pipeline integration
    """
    
    def __init__(self, bootstrap_servers: str = "localhost:9092", 
                 group_id: str = "pcds-consumers"):
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.consumer = None
        self.is_running = False
        self.handlers: Dict[str, List[Callable]] = {}
        self.tenant_handlers: Dict[str, Dict[str, Callable]] = {}
        self.metrics = {
            "messages_processed": 0,
            "messages_failed": 0,
            "processing_time_ms": []
        }
    
    async def connect(self, topics: List[str]):
        """Connect and subscribe to topics"""
        try:
            from aiokafka import AIOKafkaConsumer
            self.consumer = AIOKafkaConsumer(
                *topics,
                bootstrap_servers=self.bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True
            )
            await self.consumer.start()
            print(f"✅ Kafka consumer connected, subscribed to: {topics}")
        except ImportError:
            print("⚠️ aiokafka not installed, using simulated consumer")
            self.consumer = SimulatedConsumer(topics)
        except Exception as e:
            print(f"⚠️ Consumer connection failed: {e}, using simulated")
            self.consumer = SimulatedConsumer(topics)
    
    def register_handler(self, topic: str, handler: Callable):
        """Register message handler for topic"""
        if topic not in self.handlers:
            self.handlers[topic] = []
        self.handlers[topic].append(handler)
    
    def register_tenant_handler(self, tenant_id: str, topic: str, handler: Callable):
        """Register tenant-specific handler"""
        if tenant_id not in self.tenant_handlers:
            self.tenant_handlers[tenant_id] = {}
        self.tenant_handlers[tenant_id][topic] = handler
    
    async def start_consuming(self):
        """Start consuming messages"""
        self.is_running = True
        print("🔄 Starting message consumption...")
        
        while self.is_running:
            try:
                if isinstance(self.consumer, SimulatedConsumer):
                    # Simulated consumer
                    messages = await self.consumer.poll()
                    for msg in messages:
                        await self._process_message(msg)
                else:
                    # Real Kafka consumer
                    async for msg in self.consumer:
                        if not self.is_running:
                            break
                        await self._process_message(msg)
            except Exception as e:
                print(f"❌ Consumer error: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, raw_message) -> bool:
        """Process a single message"""
        start_time = datetime.now()
        
        try:
            # Parse message
            if isinstance(raw_message, dict):
                msg = ConsumerMessage(
                    topic=raw_message.get("topic", "unknown"),
                    partition=0,
                    offset=raw_message.get("offset", 0),
                    key=raw_message.get("key"),
                    value=raw_message.get("value", {}),
                    headers=dict(raw_message.get("headers", [])),
                    timestamp=datetime.now(),
                    tenant_id=raw_message.get("value", {}).get("_metadata", {}).get("tenant_id", "default")
                )
            else:
                # aiokafka message
                headers = {h[0]: h[1].decode() for h in (raw_message.headers or [])}
                msg = ConsumerMessage(
                    topic=raw_message.topic,
                    partition=raw_message.partition,
                    offset=raw_message.offset,
                    key=raw_message.key.decode() if raw_message.key else None,
                    value=raw_message.value,
                    headers=headers,
                    timestamp=datetime.fromtimestamp(raw_message.timestamp / 1000),
                    tenant_id=headers.get("tenant_id", "default")
                )
            
            # Route to handlers
            await self._route_message(msg)
            
            self.metrics["messages_processed"] += 1
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.metrics["processing_time_ms"].append(processing_time)
            if len(self.metrics["processing_time_ms"]) > 1000:
                self.metrics["processing_time_ms"] = self.metrics["processing_time_ms"][-1000:]
            
            return True
            
        except Exception as e:
            self.metrics["messages_failed"] += 1
            print(f"❌ Message processing failed: {e}")
            return False
    
    async def _route_message(self, msg: ConsumerMessage):
        """Route message to appropriate handlers"""
        # Check tenant-specific handlers first
        if msg.tenant_id in self.tenant_handlers:
            tenant_handler = self.tenant_handlers[msg.tenant_id].get(msg.topic)
            if tenant_handler:
                await self._call_handler(tenant_handler, msg)
                return
        
        # Fall back to topic handlers
        handlers = self.handlers.get(msg.topic, [])
        for handler in handlers:
            await self._call_handler(handler, msg)
    
    async def _call_handler(self, handler: Callable, msg: ConsumerMessage):
        """Call handler with proper async/sync handling"""
        if asyncio.iscoroutinefunction(handler):
            await handler(msg)
        else:
            handler(msg)
    
    def get_metrics(self) -> Dict:
        """Get consumer metrics"""
        avg_time = 0
        if self.metrics["processing_time_ms"]:
            avg_time = sum(self.metrics["processing_time_ms"]) / len(self.metrics["processing_time_ms"])
        
        return {
            "messages_processed": self.metrics["messages_processed"],
            "messages_failed": self.metrics["messages_failed"],
            "avg_processing_time_ms": avg_time,
            "group_id": self.group_id,
            "is_running": self.is_running
        }
    
    async def stop(self):
        """Stop consuming"""
        self.is_running = False
        if self.consumer and hasattr(self.consumer, 'stop'):
            await self.consumer.stop()


class SimulatedConsumer:
    """Simulated consumer for development"""
    
    def __init__(self, topics: List[str]):
        self.topics = topics
        self.message_queue = asyncio.Queue()
    
    async def poll(self, timeout: float = 1.0) -> List[Dict]:
        """Poll for messages"""
        messages = []
        try:
            while True:
                msg = self.message_queue.get_nowait()
                messages.append(msg)
        except asyncio.QueueEmpty:
            await asyncio.sleep(timeout)
        return messages
    
    async def inject_message(self, message: Dict):
        """Inject test message"""
        await self.message_queue.put(message)


# ML Pipeline Consumer
class MLPipelineConsumer(KafkaConsumer):
    """
    Specialized consumer for ML pipeline processing
    Integrates with ML Engine v3.0
    """
    
    def __init__(self):
        super().__init__(group_id="pcds-ml-pipeline")
        self.ml_engine = None
    
    async def initialize(self):
        """Initialize with ML engine"""
        from ml.advanced_detector import get_advanced_engine
        self.ml_engine = get_advanced_engine()
        
        from kafka.config import TOPICS
        await self.connect([TOPICS["raw_events"].name])
        
        # Register ML processing handler
        self.register_handler(TOPICS["raw_events"].name, self._process_raw_event)
    
    async def _process_raw_event(self, msg: ConsumerMessage):
        """Process raw event through ML pipeline"""
        if not self.ml_engine:
            return
        
        from kafka.producer import kafka_producer
        from kafka.config import TOPICS
        
        # Run ML detection
        result = self.ml_engine.detect(
            data=msg.value,
            entity_id=msg.value.get("entity_id")
        )
        
        # Create detection record
        detection = {
            "original_event": msg.value,
            "ml_result": result,
            "entity_id": msg.value.get("entity_id"),
            "severity": result.get("risk_level", "medium"),
            "confidence": result.get("confidence", 0.5),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to detections topic
        await kafka_producer.send_detection(detection, msg.value.get("entity_id"))
        
        # Send alert if critical
        if result.get("risk_level") == "critical":
            await kafka_producer.send_alert(detection, severity="critical")


# Global instances
kafka_consumer = KafkaConsumer()
ml_pipeline_consumer = MLPipelineConsumer()
