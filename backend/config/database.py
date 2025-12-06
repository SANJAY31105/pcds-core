"""
Database Connection Manager for PCDS Enterprise
Supports SQLite (default) and PostgreSQL
"""

import sqlite3
from contextlib import contextmanager
from typing import Generator
import json
from datetime import datetime
from config.settings import settings


class DatabaseManager:
    """Database connection and query management"""
    
    def __init__(self):
        self.db_path = self._get_db_path()
        self._ensure_database_exists()
    
    def _get_db_path(self) -> str:
        """Extract database path from DATABASE_URL"""
        url = settings.DATABASE_URL
        if url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "")
        return "./pcds_enterprise.db"
    
    def _ensure_database_exists(self):
        """Ensure database file exists"""
        import os
        if not os.path.exists(self.db_path):
            print(f"⚠️  Database not found at {self.db_path}")
            print("💡 Run: python data/init_db.py")
    
    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with automatic commit/rollback"""
        # Don't parse timestamps - treat them as text to avoid format issues
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
    
    def execute_query(self, query: str, params: tuple = None) -> list:
        """Execute SELECT query and return results as list of dicts"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def execute_one(self, query: str, params: tuple = None) -> dict:
        """Execute SELECT query and return single result as dict"""
        results = self.execute_query(query, params)
        return results[0] if results else None
    
    def execute_insert(self, query: str, params: tuple = None) -> int:
        """Execute INSERT and return last row ID"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.lastrowid
    
    def execute_update(self, query: str, params: tuple = None) -> int:
        """Execute UPDATE/DELETE and return affected rows count"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return cursor.rowcount
    
    def execute_many(self, query: str, params_list: list) -> int:
        """Execute batch INSERT and return count"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(query, params_list)
            return cursor.rowcount


# Helper functions for JSON handling in SQLite
def json_serialize(obj):
    """Serialize Python object to JSON string for database storage"""
    if obj is None:
        return None
    if isinstance(obj, (dict, list)):
        return json.dumps(obj)
    return str(obj)


def json_deserialize(text):
    """Deserialize JSON string from database to Python object"""
    if text is None:
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text


# Global database manager instance
db_manager = DatabaseManager()


# FastAPI dependency for getting database connection
def get_db():
    """FastAPI dependency to get database connection"""
    with db_manager.get_connection() as conn:
        yield conn


def get_db_connection():
    """Simple helper to get a direct database connection for auth module"""
    return sqlite3.connect(
        db_manager.db_path,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
    )


# Database initialization check
def init_db():
    """Verify database is initialized"""
    try:
        tables = db_manager.execute_query("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            ORDER BY name
        """)
        
        required_tables = [
            'entities', 'detections', 'mitre_tactics', 
            'mitre_techniques', 'investigations'
        ]
        
        table_names = [t['name'] for t in tables]
        missing = [t for t in required_tables if t not in table_names]
        
        if missing:
            print(f"❌ Missing tables: {', '.join(missing)}")
            print("💡 Run: python data/init_db.py")
            return False
        
        print(f"✅ Database ready: {len(tables)} tables found")
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False


# Utility functions for common queries
class EntityQueries:
    """Common entity queries"""
    
    @staticmethod
    def get_by_id(entity_id: str) -> dict:
        """Get entity by ID"""
        return db_manager.execute_one(
            "SELECT * FROM entities WHERE id = ?",
            (entity_id,)
        )
    
    @staticmethod
    def get_all(limit: int = 100, urgency_level: str = None) -> list:
        """Get all entities with optional filtering"""
        if urgency_level:
            query = """
                SELECT * FROM entities 
                WHERE urgency_level = ? 
                ORDER BY urgency_score DESC, last_seen DESC 
                LIMIT ?
            """
            return db_manager.execute_query(query, (urgency_level, limit))
        else:
            query = """
                SELECT * FROM entities 
                ORDER BY urgency_score DESC, last_seen DESC 
                LIMIT ?
            """
            return db_manager.execute_query(query, (limit,))
    
    @staticmethod
    def update_urgency(entity_id: str, urgency_score: int, urgency_level: str) -> int:
        """Update entity urgency score"""
        return db_manager.execute_update("""
            UPDATE entities 
            SET urgency_score = ?, urgency_level = ?, updated_at = ?
            WHERE id = ?
        """, (urgency_score, urgency_level, datetime.utcnow(), entity_id))
    
    @staticmethod
    def increment_detection_count(entity_id: str, severity: str) -> int:
        """Increment detection counters"""
        severity_field = f"{severity}_detections" if severity in ['critical', 'high'] else 'total_detections'
        
        return db_manager.execute_update(f"""
            UPDATE entities 
            SET total_detections = total_detections + 1,
                {severity_field} = {severity_field} + 1,
                last_detection_time = ?,
                updated_at = ?
            WHERE id = ?
        """, (datetime.utcnow(), datetime.utcnow(), entity_id))


class DetectionQueries:
    """Common detection queries"""
    
    @staticmethod
    def create(detection_data: dict) -> str:
        """Create new detection and publish to Kafka"""
        db_manager.execute_insert("""
            INSERT INTO detections 
            (id, detection_type, title, description, severity, confidence_score, 
             risk_score, entity_id, source_ip, destination_ip, tactic_id, 
             technique_id, kill_chain_stage, detected_at, status, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            detection_data['id'],
            detection_data['detection_type'],
            detection_data['title'],
            detection_data['description'],
            detection_data['severity'],
            detection_data['confidence_score'],
            detection_data['risk_score'],
            detection_data['entity_id'],
            detection_data.get('source_ip'),
            detection_data.get('destination_ip'),
            detection_data.get('tactic_id'),
            detection_data.get('technique_id'),
            detection_data.get('kill_chain_stage'),
            detection_data['detected_at'],
            detection_data.get('status', 'new'),
            json_serialize(detection_data.get('metadata'))
        ))
        
        # Publish to Kafka for real-time streaming
        try:
            import asyncio
            from kafka.producer import kafka_producer
            from kafka.connectors import siem_manager
            from kafka.replay import event_store
            from kafka.config import TOPICS
            
            async def publish():
                await kafka_producer.connect()
                await kafka_producer.send_detection(detection_data, detection_data['entity_id'])
                await siem_manager.send_to_all(detection_data)
                event_store.store_event(TOPICS["detections"].name, detection_data, detection_data['entity_id'])
            
            # Run in background
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(publish())
            except RuntimeError:
                asyncio.run(publish())
        except Exception as e:
            # Don't fail detection creation if Kafka fails
            pass
        
        return detection_data['id']
    
    @staticmethod
    def get_by_entity(entity_id: str, limit: int = 100) -> list:
        """Get detections for an entity"""
        return db_manager.execute_query("""
            SELECT d.*, t.name as technique_name, tc.name as tactic_name
            FROM detections d
            LEFT JOIN mitre_techniques t ON d.technique_id = t.id
            LEFT JOIN mitre_tactics tc ON d.tactic_id = tc.id
            WHERE d.entity_id = ?
            ORDER BY d.detected_at DESC
            LIMIT ?
        """, (entity_id, limit))
    
    @staticmethod
    def get_recent(limit: int = 50, severity: str = None) -> list:
        """Get recent detections"""
        if severity:
            query = """
                SELECT d.*, e.identifier as entity_identifier, 
                       t.name as technique_name, tc.name as tactic_name
                FROM detections d
                LEFT JOIN entities e ON d.entity_id = e.id
                LEFT JOIN mitre_techniques t ON d.technique_id = t.id
                LEFT JOIN mitre_tactics tc ON d.tactic_id = tc.id
                WHERE d.severity = ?
                ORDER BY d.detected_at DESC
                LIMIT ?
            """
            return db_manager.execute_query(query, (severity, limit))
        else:
            query = """
                SELECT d.*, e.identifier as entity_identifier,
                       t.name as technique_name, tc.name as tactic_name
                FROM detections d
                LEFT JOIN entities e ON d.entity_id = e.id
                LEFT JOIN mitre_techniques t ON d.technique_id = t.id
                LEFT JOIN mitre_tactics tc ON d.tactic_id = tc.id
                ORDER BY d.detected_at DESC
                LIMIT ?
            """
            return db_manager.execute_query(query, (limit,))


class MITREQueries:
    """MITRE ATT&CK queries"""
    
    @staticmethod
    def get_all_tactics() -> list:
        """Get all MITRE tactics"""
        return db_manager.execute_query("""
            SELECT * FROM mitre_tactics 
            ORDER BY kill_chain_order
        """)
    
    @staticmethod
    def get_techniques_by_tactic(tactic_id: str) -> list:
        """Get techniques for a tactic"""
        return db_manager.execute_query("""
            SELECT * FROM mitre_techniques 
            WHERE tactic_id = ? 
            ORDER BY name
        """, (tactic_id,))
    
    @staticmethod
    def get_technique(technique_id: str) -> dict:
        """Get technique by ID"""
        return db_manager.execute_one("""
            SELECT * FROM mitre_techniques WHERE id = ?
        """, (technique_id,))


# Export commonly used query classes
__all__ = [
    'db_manager',
    'get_db',
    'init_db',
    'EntityQueries',
    'DetectionQueries',
    'MITREQueries',
    'json_serialize',
    'json_deserialize'
]
