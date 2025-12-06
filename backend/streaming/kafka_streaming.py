"""
Kafka Event Streaming for PCDS Enterprise
Real-time detection event processing at scale (100K+ events/second)
"""
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import json
import logging
from typing import Dict, Any, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']
KAFKA_TOPICS = {
    'detections': 'pcds.detections',
    'entities': 'pcds.entities',
    'alerts': 'pcds.alerts',
    'correlations': 'pcds.correlations',
}


class KafkaProducerClient:
    """Kafka producer for publishing events"""
    
    def __init__(self, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',  # Wait for all replicas
            retries=3,
            max_in_flight_requests_per_connection=5,
            compression_type='gzip',
            linger_ms=10,  # Batch messages for better throughput
            batch_size=32768,  # 32KB batches
        )
        logger.info(f"Kafka producer initialized: {bootstrap_servers}")
    
    def publish_detection(self, detection: Dict[str, Any]):
        """
        Publish detection event to Kafka
        
        Args:
            detection: Detection data
        """
        try:
            topic = KAFKA_TOPICS['detections']
            key = str(detection.get('id', ''))
            
            # Add metadata
            event = {
                **detection,
                'event_type': 'detection',
                'published_at': datetime.utcnow().isoformat(),
            }
            
            # Async send
            future = self.producer.send(topic, key=key, value=event)
            
            # Optional: wait for confirmation (blocking)
            # record_metadata = future.get(timeout=10)
            # logger.debug(f"Published to {record_metadata.topic}:{record_metadata.partition}")
            
            logger.debug(f"Detection {key} published to Kafka")
        
        except KafkaError as e:
            logger.error(f"Failed to publish detection: {e}")
            raise
    
    def publish_entity_update(self, entity: Dict[str, Any]):
        """Publish entity update event"""
        try:
            topic = KAFKA_TOPICS['entities']
            key = entity.get('identifier', '')
            
            event = {
                **entity,
                'event_type': 'entity_update',
                'published_at': datetime.utcnow().isoformat(),
            }
            
            self.producer.send(topic, key=key, value=event)
            logger.debug(f"Entity {key} published to Kafka")
        
        except KafkaError as e:
            logger.error(f"Failed to publish entity: {e}")
            raise
    
    def publish_alert(self, alert: Dict[str, Any]):
        """Publish critical alert"""
        try:
            topic = KAFKA_TOPICS['alerts']
            
            event = {
                **alert,
                'event_type': 'alert',
                'published_at': datetime.utcnow().isoformat(),
            }
            
            self.producer.send(topic, value=event)
            logger.info(f"Alert published: {alert.get('title')}")
        
        except KafkaError as e:
            logger.error(f"Failed to publish alert: {e}")
            raise
    
    def flush(self):
        """Flush any pending messages"""
        self.producer.flush()
    
    def close(self):
        """Close producer"""
        self.producer.close()


class KafkaConsumerClient:
    """Kafka consumer for processing events"""
    
    def __init__(self, topics, group_id, bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS):
        self.consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            key_deserializer=lambda k: k.decode('utf-8') if k else None,
            auto_offset_reset='latest',  # Start from latest message
            enable_auto_commit=True,
            max_poll_records=500,  # Process 500 messages at a time
            session_timeout_ms=30000,
        )
        logger.info(f"Kafka consumer initialized: {group_id} on {topics}")
    
    def consume(self, handler: Callable):
        """
        Consume messages and process with handler
        
        Args:
            handler: Function to process each message
        """
        try:
            logger.info("Starting Kafka consumer...")
            
            for message in self.consumer:
                try:
                    # Process message
                    event = message.value
                    handler(event)
                    
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Continue processing other messages
                    continue
        
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
        finally:
            self.close()
    
    def close(self):
        """Close consumer"""
        self.consumer.close()


# ===== STREAM PROCESSORS =====

class DetectionStreamProcessor:
    """Process detection events in real-time"""
    
    def __init__(self):
        self.consumer = KafkaConsumerClient(
            topics=[KAFKA_TOPICS['detections']],
            group_id='detection-processor'
        )
    
    def process_detection(self, event: Dict):
        """Process single detection event"""
        logger.info(f"Processing detection: {event.get('id')}")
        
        # 1. Calculate entity risk score (async task)
        if 'entity_id' in event:
            from tasks import queue_entity_scoring
            queue_entity_scoring(event['entity_id'])
        
        # 2. Correlate to campaigns (async task)
        if 'id' in event:
            from tasks import queue_campaign_correlation
            queue_campaign_correlation(event['id'])
        
        # 3. Check for critical severity -> auto-response
        if event.get('severity') == 'critical':
            self._trigger_auto_response(event)
    
    def _trigger_auto_response(self, detection: Dict):
        """Trigger automated response for critical detections"""
        from engine.soar_engine import soar_engine
        
        # Determine playbook
        detection_type = detection.get('type', '').lower()
        
        if 'ransomware' in detection_type:
            playbook = 'ransomware'
        elif 'exfiltration' in detection_type:
            playbook = 'data_exfiltration'
        elif 'credential' in detection_type:
            playbook = 'compromised_credentials'
        else:
            playbook = 'apt'  # Default to APT response
        
        # Execute SOAR playbook (async)
        logger.warning(f"🚨 Auto-triggering SOAR playbook: {playbook}")
        # In production: run this async
        # asyncio.create_task(soar_engine.execute_playbook(playbook, detection))
    
    def start(self):
        """Start processing detections"""
        self.consumer.consume(self.process_detection)


class CorrelationStreamProcessor:
    """Real-time event correlation"""
    
    def __init__(self):
        # Subscribe to multiple topics for correlation
        self.consumer = KafkaConsumerClient(
            topics=[KAFKA_TOPICS['detections'], KAFKA_TOPICS['entities']],
            group_id='correlation-processor'
        )
        self.event_window = []  # Sliding window for correlation
    
    def correlate_events(self, event: Dict):
        """Correlate events in real-time"""
        self.event_window.append(event)
        
        # Keep last 1000 events in window
        if len(self.event_window) > 1000:
            self.event_window.pop(0)
        
        # Look for patterns
        # Example: Multiple detections on same entity within 5 minutes
        # This is simplified - in production, use more sophisticated correlation
        
        if event.get('event_type') == 'detection':
            entity_id = event.get('entity_id')
            if entity_id:
                related_events = [
                    e for e in self.event_window 
                    if e.get('entity_id') == entity_id
                ]
                
                if len(related_events) >= 5:
                    logger.warning(f"🔥 Campaign detected: {len(related_events)} events on {entity_id}")
                    # Publish correlation alert
                    kafka_producer.publish_alert({
                        'title': f'Potential Campaign: {len(related_events)} events on {entity_id}',
                        'severity': 'high',
                        'entity_id': entity_id,
                        'event_count': len(related_events)
                    })
    
    def start(self):
        """Start correlation processing"""
        self.consumer.consume(self.correlate_events)


# Global instances
kafka_producer = KafkaProducerClient()


# Helper functions
def publish_detection_to_kafka(detection: Dict):
    """Publish detection to Kafka stream"""
    kafka_producer.publish_detection(detection)


def publish_entity_to_kafka(entity: Dict):
    """Publish entity update to Kafka stream"""
    kafka_producer.publish_entity_update(entity)


# Example: Start consumers (run in separate processes)
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        processor_type = sys.argv[1]
        
        if processor_type == "detection":
            processor = DetectionStreamProcessor()
            processor.start()
        
        elif processor_type == "correlation":
            processor = CorrelationStreamProcessor()
            processor.start()
        
        else:
            print("Usage: python kafka_streaming.py [detection|correlation]")
    else:
        print("Usage: python kafka_streaming.py [detection|correlation]")
