import logging
from datetime import datetime, timedelta
import json
import numpy as np
from config.database import db_manager, DetectionQueries, EntityQueries

logger = logging.getLogger("pcds.ueba")

class UEBAEngine:
    """
    User & Entity Behavior Analytics Engine
    Establishes baselines and detects anomalies in entity behavior.
    """
    
    def __init__(self):
        self.metrics = [
            "detection_frequency",  # Detections per hour
            "severity_ratio",       # Ratio of high/critical to total
            "unique_tactics"        # Number of unique tactics used
        ]
        self.learning_period_hours = 24
        self.anomaly_threshold = 2.0  # Standard deviations
    
    async def run_cycle(self):
        """Run a full UEBA cycle: calculate baselines -> detect anomalies"""
        logger.info("🔄 Starting UEBA cycle")
        try:
            self.calculate_baselines()
            self.detect_anomalies()
            logger.info("✅ UEBA cycle completed")
        except Exception as e:
            logger.error(f"❌ UEBA cycle failed: {e}")

    def calculate_baselines(self):
        """Update baselines for all active entities based on historical data"""
        # Get all entities that have had activity recently
        entities = db_manager.execute_query("""
            SELECT id, identifier FROM entities 
            WHERE last_seen > datetime('now', '-7 days')
        """)
        
        for entity in entities:
            self._update_entity_baseline(entity['id'])

    def _update_entity_baseline(self, entity_id: str):
        """Calculate and store baseline metrics for a single entity"""
        # Fetch historical detections for this entity
        detections = DetectionQueries.get_by_entity(entity_id, limit=1000)
        
        if not detections:
            return

        # 1. Detection Frequency (per hour)
        if len(detections) > 1:
            timestamps = [datetime.fromisoformat(d['detected_at']) for d in detections]
            timestamps.sort()
            duration_hours = (timestamps[-1] - timestamps[0]).total_seconds() / 3600
            if duration_hours > 0:
                freq = len(detections) / duration_hours
                self._store_baseline(entity_id, "detection_frequency", freq)

        # 2. Severity Ratio
        high_critical = sum(1 for d in detections if d['severity'] in ['high', 'critical'])
        if len(detections) > 0:
            ratio = high_critical / len(detections)
            self._store_baseline(entity_id, "severity_ratio", ratio)

    def _store_baseline(self, entity_id: str, metric: str, value: float):
        """Upsert baseline value into database"""
        # Simple moving average update if exists, or insert new
        current = db_manager.execute_one("""
            SELECT baseline_value, sample_count FROM entity_baselines
            WHERE entity_id = ? AND metric_name = ?
        """, (entity_id, metric))
        
        if current:
            # Update moving average
            n = current['sample_count']
            old_val = current['baseline_value']
            new_val = (old_val * n + value) / (n + 1)
            
            db_manager.execute_update("""
                UPDATE entity_baselines 
                SET baseline_value = ?, sample_count = sample_count + 1, last_updated = CURRENT_TIMESTAMP
                WHERE entity_id = ? AND metric_name = ?
            """, (new_val, entity_id, metric))
        else:
            # Insert new
            db_manager.execute_insert("""
                INSERT INTO entity_baselines (entity_id, metric_name, baseline_value, deviation_threshold, sample_count)
                VALUES (?, ?, ?, ?, 1)
            """, (entity_id, metric, value, self.anomaly_threshold))

    def detect_anomalies(self):
        """Compare current activity against baselines to find anomalies"""
        # Look at activity in the last hour
        recent_detections = db_manager.execute_query("""
            SELECT entity_id, COUNT(*) as count 
            FROM detections 
            WHERE detected_at > datetime('now', '-1 hour')
            GROUP BY entity_id
        """)
        
        for item in recent_detections:
            entity_id = item['entity_id']
            current_freq = item['count'] # 1 hour window
            
            # Check against baseline
            baseline = db_manager.execute_one("""
                SELECT baseline_value, deviation_threshold FROM entity_baselines
                WHERE entity_id = ? AND metric_name = 'detection_frequency'
            """, (entity_id,))
            
            if baseline:
                avg = baseline['baseline_value']
                threshold = baseline['deviation_threshold']
                
                # If current frequency is significantly higher than average
                if avg > 0 and (current_freq / avg) > threshold:
                    self._create_anomaly_alert(entity_id, current_freq, avg)

    def _create_anomaly_alert(self, entity_id: str, current: float, baseline: float):
        """Generate a detection for the behavioral anomaly"""
        import uuid
        
        # Check if we already alerted recently to avoid spam
        existing = db_manager.execute_one("""
            SELECT id FROM detections 
            WHERE entity_id = ? AND detection_type = 'behavioral_anomaly'
            AND detected_at > datetime('now', '-1 hour')
        """, (entity_id,))
        
        if existing:
            return

        detection_data = {
            "id": str(uuid.uuid4()),
            "detection_type": "behavioral_anomaly",
            "title": "Abnormal Behavior Detected",
            "description": f"Entity activity ({current}/hr) is significantly higher than baseline ({baseline:.2f}/hr).",
            "severity": "high",
            "confidence_score": 0.85,
            "risk_score": 75,
            "entity_id": entity_id,
            "detected_at": datetime.utcnow().isoformat(),
            "metadata": {
                "metric": "detection_frequency",
                "current_value": current,
                "baseline_value": baseline,
                "deviation": current / baseline if baseline else 0
            }
        }
        
        DetectionQueries.create(detection_data)
        logger.warning(f"🚨 Created behavioral anomaly for entity {entity_id}")

# Global instance
ueba_engine = UEBAEngine()
