"""
PCDS Enterprise - Event Replay
Audit trail queries and historical event replay
"""

import json
from typing import Dict, List, Optional, Callable, Generator
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import sqlite3


@dataclass
class ReplayOptions:
    """Options for event replay"""
    topic: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    entity_id: Optional[str] = None
    severity: Optional[str] = None
    limit: int = 1000
    offset: int = 0


class EventStore:
    """
    Persistent event store for replay capability
    Uses SQLite for local storage (switch to ClickHouse/TimescaleDB for production)
    """
    
    def __init__(self, db_path: str = "pcds_events.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize event store database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS kafka_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                partition INTEGER DEFAULT 0,
                offset_id INTEGER,
                key TEXT,
                value TEXT NOT NULL,
                headers TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                tenant_id TEXT DEFAULT 'default',
                entity_id TEXT,
                severity TEXT,
                indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_topic ON kafka_events(topic)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON kafka_events(timestamp)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entity ON kafka_events(entity_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_severity ON kafka_events(severity)")
        conn.commit()
        conn.close()
    
    def store_event(self, topic: str, value: Dict, key: str = None,
                   headers: Dict = None, tenant_id: str = "default"):
        """Store event for replay"""
        conn = sqlite3.connect(self.db_path)
        
        entity_id = value.get("entity_id") or value.get("original_event", {}).get("entity_id")
        severity = value.get("severity") or value.get("ml_result", {}).get("risk_level")
        
        conn.execute("""
            INSERT INTO kafka_events (topic, key, value, headers, tenant_id, entity_id, severity)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            topic,
            key,
            json.dumps(value),
            json.dumps(headers) if headers else None,
            tenant_id,
            entity_id,
            severity
        ))
        conn.commit()
        conn.close()
    
    def query_events(self, options: ReplayOptions) -> List[Dict]:
        """Query events based on options"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        query = "SELECT * FROM kafka_events WHERE topic = ?"
        params = [options.topic]
        
        if options.start_time:
            query += " AND timestamp >= ?"
            params.append(options.start_time.isoformat())
        
        if options.end_time:
            query += " AND timestamp <= ?"
            params.append(options.end_time.isoformat())
        
        if options.entity_id:
            query += " AND entity_id = ?"
            params.append(options.entity_id)
        
        if options.severity:
            query += " AND severity = ?"
            params.append(options.severity)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([options.limit, options.offset])
        
        cursor = conn.execute(query, params)
        events = []
        for row in cursor.fetchall():
            events.append({
                "id": row["id"],
                "topic": row["topic"],
                "key": row["key"],
                "value": json.loads(row["value"]),
                "timestamp": row["timestamp"],
                "entity_id": row["entity_id"],
                "severity": row["severity"]
            })
        
        conn.close()
        return events
    
    def count_events(self, topic: str = None, 
                     start_time: datetime = None, 
                     end_time: datetime = None) -> Dict:
        """Get event counts"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT COUNT(*) as total"
        params = []
        
        if topic:
            query += ", topic"
            
        query += " FROM kafka_events WHERE 1=1"
        
        if topic:
            query += " AND topic = ?"
            params.append(topic)
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if topic:
            query += " GROUP BY topic"
        
        cursor = conn.execute(query, params)
        result = cursor.fetchone()
        conn.close()
        
        return {"total": result[0] if result else 0}
    
    def get_timeline(self, hours: int = 24, interval_minutes: int = 60) -> List[Dict]:
        """Get event timeline for visualization"""
        conn = sqlite3.connect(self.db_path)
        
        now = datetime.utcnow()
        start = now - timedelta(hours=hours)
        
        # SQLite strftime for grouping
        cursor = conn.execute("""
            SELECT 
                strftime('%Y-%m-%d %H:00:00', timestamp) as interval,
                severity,
                COUNT(*) as count
            FROM kafka_events
            WHERE timestamp >= ?
            GROUP BY interval, severity
            ORDER BY interval
        """, (start.isoformat(),))
        
        timeline = {}
        for row in cursor.fetchall():
            interval = row[0]
            severity = row[1] or "unknown"
            count = row[2]
            
            if interval not in timeline:
                timeline[interval] = {"timestamp": interval}
            timeline[interval][severity] = count
        
        conn.close()
        return list(timeline.values())


class EventReplay:
    """
    Event replay service
    Allows historical event replay through ML pipeline
    """
    
    def __init__(self, store: EventStore = None):
        self.store = store or EventStore()
        self.replay_in_progress = False
        self.current_replay_id = None
    
    async def replay_events(self, options: ReplayOptions, 
                           handler: Callable) -> Dict:
        """Replay events through a handler"""
        self.replay_in_progress = True
        self.current_replay_id = datetime.utcnow().isoformat()
        
        events = self.store.query_events(options)
        processed = 0
        failed = 0
        
        for event in events:
            try:
                await handler(event)
                processed += 1
            except Exception as e:
                failed += 1
        
        self.replay_in_progress = False
        
        return {
            "replay_id": self.current_replay_id,
            "total_events": len(events),
            "processed": processed,
            "failed": failed,
            "options": {
                "topic": options.topic,
                "start_time": options.start_time.isoformat() if options.start_time else None,
                "end_time": options.end_time.isoformat() if options.end_time else None
            }
        }
    
    def stream_events(self, options: ReplayOptions) -> Generator[Dict, None, None]:
        """Stream events as generator"""
        events = self.store.query_events(options)
        for event in events:
            yield event
    
    async def replay_for_entity(self, entity_id: str, handler: Callable,
                               hours: int = 24) -> Dict:
        """Replay all events for an entity"""
        from kafka.config import TOPICS
        
        options = ReplayOptions(
            topic=TOPICS["detections"].name,
            start_time=datetime.utcnow() - timedelta(hours=hours),
            entity_id=entity_id
        )
        return await self.replay_events(options, handler)


class AuditLog:
    """
    Audit log for compliance
    Tracks all user actions and system events
    """
    
    def __init__(self, store: EventStore = None):
        self.store = store or EventStore()
    
    async def log_action(self, user: str, action: str, 
                        resource: str, details: Dict = None):
        """Log an audit action"""
        from kafka.producer import kafka_producer
        from kafka.config import TOPICS
        
        audit_entry = {
            "user": user,
            "action": action,
            "resource": resource,
            "details": details or {},
            "timestamp": datetime.utcnow().isoformat(),
            "source_ip": details.get("source_ip") if details else None
        }
        
        # Send to Kafka
        await kafka_producer.send(TOPICS["audit"].name, audit_entry)
        
        # Store locally
        self.store.store_event(TOPICS["audit"].name, audit_entry)
        
        return audit_entry
    
    def query_audit_log(self, user: str = None, action: str = None,
                       start_time: datetime = None, end_time: datetime = None,
                       limit: int = 100) -> List[Dict]:
        """Query audit log"""
        from kafka.config import TOPICS
        
        options = ReplayOptions(
            topic=TOPICS["audit"].name,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        
        events = self.store.query_events(options)
        
        # Filter by user/action if specified
        if user:
            events = [e for e in events if e["value"].get("user") == user]
        if action:
            events = [e for e in events if e["value"].get("action") == action]
        
        return events


# Global instances
event_store = EventStore()
event_replay = EventReplay(event_store)
audit_log = AuditLog(event_store)
