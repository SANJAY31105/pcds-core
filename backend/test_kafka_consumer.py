"""Kafka consumer test using confluent-kafka"""
from confluent_kafka import Consumer
import json

conf = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'test-consumer',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(conf)
consumer.subscribe(['ml_predictions'])

print("Reading from Kafka topic 'ml_predictions'...")
count = 0
timeout_count = 0

while timeout_count < 3:  # Exit after 3 empty polls
    msg = consumer.poll(2.0)
    
    if msg is None:
        timeout_count += 1
        continue
    
    if msg.error():
        print(f"Error: {msg.error()}")
        continue
    
    timeout_count = 0
    count += 1
    
    try:
        data = json.loads(msg.value().decode('utf-8'))
        print(f"  [{count}] {data.get('prediction_id', 'N/A')}: {data.get('class_name', 'N/A')} ({data.get('source_host', 'N/A')})")
    except:
        print(f"  [{count}] Raw: {msg.value()[:50]}")
    
    if count >= 10:
        break

print(f"\n✅ Total messages: {count}")
consumer.close()
