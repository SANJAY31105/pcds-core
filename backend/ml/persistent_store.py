"""
Persistent Prediction Store
SQLite-backed storage for ML predictions with 30-day retention

Features:
- Persistent storage across restarts
- Fast indexed queries
- Automatic cleanup of old predictions
- Ground truth tracking for retraining
"""

import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from config.database import db_manager


@dataclass
class Prediction:
    """ML Prediction record"""
    prediction_id: str
    timestamp: str
    model_version: str
    predicted_class: int
    class_name: str
    confidence: float
    source_ip: Optional[str] = None
    source_host: Optional[str] = None
    ground_truth: Optional[int] = None
    is_tp: bool = False
    is_fp: bool = False
    severity: str = "medium"
    mitre_technique: Optional[str] = None
    mitre_tactic: Optional[str] = None
    top_features: Optional[List[str]] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None
    review_notes: Optional[str] = None


class PersistentPredictionStore:
    """
    SQLite-backed prediction storage
    
    Replaces in-memory PredictionStore for production use
    """
    
    def __init__(self, retention_days: int = 30):
        self.retention_days = retention_days
        self._lock = threading.Lock()
        self._ensure_table()
        
        # Cache for stats (updated periodically)
        self._stats_cache = None
        self._stats_cache_time = None
        self._stats_cache_ttl = 10  # seconds
        
        print("💾 Persistent Prediction Store initialized")
    
    def _ensure_table(self):
        """Ensure ml_predictions table exists"""
        try:
            db_manager.execute_query("""
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id VARCHAR(100) NOT NULL UNIQUE,
                    model_version VARCHAR(50) NOT NULL,
                    predicted_class INTEGER NOT NULL,
                    class_name VARCHAR(100) NOT NULL,
                    confidence DECIMAL(5,4) NOT NULL,
                    severity VARCHAR(20) DEFAULT 'medium',
                    mitre_technique VARCHAR(20),
                    mitre_tactic VARCHAR(50),
                    top_features TEXT,
                    source_ip VARCHAR(45),
                    source_host VARCHAR(200),
                    prediction_timestamp TIMESTAMP NOT NULL,
                    ground_truth INTEGER,
                    is_tp BOOLEAN DEFAULT FALSE,
                    is_fp BOOLEAN DEFAULT FALSE,
                    reviewed_by VARCHAR(100),
                    reviewed_at TIMESTAMP,
                    review_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
        except Exception as e:
            print(f"   ⚠️ Table check: {e}")
    
    def add(self, prediction: Prediction):
        """Add prediction to store"""
        with self._lock:
            try:
                top_features_json = json.dumps(prediction.top_features) if prediction.top_features else None
                
                db_manager.execute_insert("""
                    INSERT OR REPLACE INTO ml_predictions 
                    (prediction_id, model_version, predicted_class, class_name, confidence,
                     severity, mitre_technique, mitre_tactic, top_features,
                     source_ip, source_host, prediction_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.prediction_id,
                    prediction.model_version,
                    prediction.predicted_class,
                    prediction.class_name,
                    prediction.confidence,
                    prediction.severity,
                    prediction.mitre_technique,
                    prediction.mitre_tactic,
                    top_features_json,
                    prediction.source_ip,
                    prediction.source_host,
                    prediction.timestamp
                ))
                
                # Invalidate stats cache
                self._stats_cache = None
                
            except Exception as e:
                print(f"   ⚠️ Error storing prediction: {e}")
    
    def get(self, prediction_id: str) -> Optional[Prediction]:
        """Get prediction by ID"""
        try:
            result = db_manager.execute_one("""
                SELECT * FROM ml_predictions WHERE prediction_id = ?
            """, (prediction_id,))
            
            if result:
                return self._row_to_prediction(result)
            return None
        except Exception as e:
            print(f"   ⚠️ Error getting prediction: {e}")
            return None
    
    def set_ground_truth(self, prediction_id: str, ground_truth: int,
                        is_tp: bool, is_fp: bool,
                        reviewed_by: str = None, notes: str = None):
        """Set ground truth from analyst feedback"""
        with self._lock:
            try:
                db_manager.execute_update("""
                    UPDATE ml_predictions 
                    SET ground_truth = ?, is_tp = ?, is_fp = ?,
                        reviewed_by = ?, reviewed_at = ?, review_notes = ?
                    WHERE prediction_id = ?
                """, (
                    ground_truth, is_tp, is_fp,
                    reviewed_by, datetime.now().isoformat(), notes,
                    prediction_id
                ))
                
                # Also insert into feedback history for audit
                pred = self.get(prediction_id)
                if pred:
                    feedback_type = 'TP' if is_tp else ('FP' if is_fp else 'UNKNOWN')
                    try:
                        db_manager.execute_insert("""
                            INSERT INTO ml_feedback_history
                            (prediction_id, model_version, predicted_class, predicted_class_name,
                             confidence, severity, mitre_technique, mitre_tactic,
                             source_ip, source_host, prediction_timestamp,
                             feedback_type, true_class, reviewed_by, reviewed_at, review_notes)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            prediction_id, pred.model_version, pred.predicted_class, pred.class_name,
                            pred.confidence, pred.severity, pred.mitre_technique, pred.mitre_tactic,
                            pred.source_ip, pred.source_host, pred.timestamp,
                            feedback_type, ground_truth, reviewed_by, datetime.now().isoformat(), notes
                        ))
                    except Exception as e:
                        # Might fail if already exists (duplicate feedback)
                        pass
                
                # Invalidate stats cache
                self._stats_cache = None
                
            except Exception as e:
                print(f"   ⚠️ Error setting ground truth: {e}")
    
    def get_recent(self, limit: int = 100) -> List[Prediction]:
        """Get recent predictions"""
        try:
            results = db_manager.execute_query("""
                SELECT * FROM ml_predictions 
                ORDER BY prediction_timestamp DESC LIMIT ?
            """, (limit,))
            return [self._row_to_prediction(r) for r in results]
        except Exception as e:
            print(f"   ⚠️ Error getting recent: {e}")
            return []
    
    def get_pending_review(self, limit: int = 50, min_confidence: float = 0,
                          severity: str = None, source_host: str = None) -> List[Prediction]:
        """Get predictions pending analyst review with filters"""
        try:
            query = "SELECT * FROM ml_predictions WHERE ground_truth IS NULL"
            params = []
            
            if min_confidence > 0:
                query += " AND confidence >= ?"
                params.append(min_confidence)
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            if source_host:
                query += " AND source_host LIKE ?"
                params.append(f"%{source_host}%")
            
            query += " ORDER BY prediction_timestamp DESC LIMIT ?"
            params.append(limit)
            
            results = db_manager.execute_query(query, tuple(params))
            return [self._row_to_prediction(r) for r in results]
        except Exception as e:
            print(f"   ⚠️ Error getting pending: {e}")
            return []
    
    def get_by_class(self, class_id: int, limit: int = 100) -> List[Prediction]:
        """Get predictions by class"""
        try:
            results = db_manager.execute_query("""
                SELECT * FROM ml_predictions 
                WHERE predicted_class = ?
                ORDER BY prediction_timestamp DESC LIMIT ?
            """, (class_id, limit))
            return [self._row_to_prediction(r) for r in results]
        except Exception as e:
            return []
    
    def get_stats(self) -> Dict:
        """Get prediction statistics with caching"""
        now = datetime.now()
        
        # Check cache
        if (self._stats_cache and self._stats_cache_time and 
            (now - self._stats_cache_time).seconds < self._stats_cache_ttl):
            return self._stats_cache
        
        try:
            # Total count
            total_row = db_manager.execute_one("SELECT COUNT(*) as cnt FROM ml_predictions")
            total = total_row['cnt'] if total_row else 0
            
            # TP/FP counts
            tp_row = db_manager.execute_one("SELECT COUNT(*) as cnt FROM ml_predictions WHERE is_tp = 1")
            fp_row = db_manager.execute_one("SELECT COUNT(*) as cnt FROM ml_predictions WHERE is_fp = 1")
            tp_count = tp_row['cnt'] if tp_row else 0
            fp_count = fp_row['cnt'] if fp_row else 0
            
            # Pending review
            pending_row = db_manager.execute_one(
                "SELECT COUNT(*) as cnt FROM ml_predictions WHERE ground_truth IS NULL"
            )
            pending = pending_row['cnt'] if pending_row else 0
            
            # By class
            class_results = db_manager.execute_query("""
                SELECT class_name, COUNT(*) as cnt FROM ml_predictions 
                GROUP BY class_name ORDER BY cnt DESC LIMIT 10
            """)
            by_class = {r['class_name']: r['cnt'] for r in class_results}
            
            # By confidence bucket
            conf_results = db_manager.execute_query("""
                SELECT 
                    CASE 
                        WHEN confidence < 0.5 THEN '0-50'
                        WHEN confidence < 0.7 THEN '50-70'
                        WHEN confidence < 0.9 THEN '70-90'
                        ELSE '90-100'
                    END as bucket,
                    COUNT(*) as cnt
                FROM ml_predictions GROUP BY bucket
            """)
            by_confidence = {"0-50": 0, "50-70": 0, "70-90": 0, "90-100": 0}
            for r in conf_results:
                by_confidence[r['bucket']] = r['cnt']
            
            # Top hosts
            host_results = db_manager.execute_query("""
                SELECT source_host, COUNT(*) as cnt FROM ml_predictions 
                WHERE source_host IS NOT NULL
                GROUP BY source_host ORDER BY cnt DESC LIMIT 10
            """)
            top_hosts = [(r['source_host'], r['cnt']) for r in host_results]
            
            # Top classes
            top_classes = [(k, v) for k, v in by_class.items()][:10]
            
            # FP rate
            reviewed = tp_count + fp_count
            fp_rate = fp_count / reviewed if reviewed > 0 else 0.0
            
            self._stats_cache = {
                "total": total,
                "by_class": by_class,
                "by_confidence_bucket": by_confidence,
                "tp_count": tp_count,
                "fp_count": fp_count,
                "fp_rate": fp_rate,
                "pending_review": pending,
                "top_classes": top_classes,
                "top_hosts": top_hosts
            }
            self._stats_cache_time = now
            
            return self._stats_cache
            
        except Exception as e:
            print(f"   ⚠️ Error getting stats: {e}")
            return {
                "total": 0, "by_class": {}, 
                "by_confidence_bucket": {"0-50": 0, "50-70": 0, "70-90": 0, "90-100": 0},
                "tp_count": 0, "fp_count": 0, "fp_rate": 0.0, "pending_review": 0,
                "top_classes": [], "top_hosts": []
            }
    
    def cleanup_old_predictions(self):
        """Remove predictions older than retention period"""
        try:
            cutoff = datetime.now() - timedelta(days=self.retention_days)
            db_manager.execute_update("""
                DELETE FROM ml_predictions 
                WHERE prediction_timestamp < ? AND ground_truth IS NOT NULL
            """, (cutoff.isoformat(),))
            print(f"   🧹 Cleaned up predictions older than {self.retention_days} days")
        except Exception as e:
            print(f"   ⚠️ Cleanup error: {e}")
    
    def _row_to_prediction(self, row: Dict) -> Prediction:
        """Convert database row to Prediction dataclass"""
        top_features = None
        if row.get('top_features'):
            try:
                top_features = json.loads(row['top_features'])
            except:
                pass
        
        return Prediction(
            prediction_id=row['prediction_id'],
            timestamp=row['prediction_timestamp'],
            model_version=row['model_version'],
            predicted_class=row['predicted_class'],
            class_name=row['class_name'],
            confidence=float(row['confidence']),
            source_ip=row.get('source_ip'),
            source_host=row.get('source_host'),
            ground_truth=row.get('ground_truth'),
            is_tp=bool(row.get('is_tp')),
            is_fp=bool(row.get('is_fp')),
            severity=row.get('severity', 'medium'),
            mitre_technique=row.get('mitre_technique'),
            mitre_tactic=row.get('mitre_tactic'),
            top_features=top_features,
            reviewed_by=row.get('reviewed_by'),
            reviewed_at=row.get('reviewed_at'),
            review_notes=row.get('review_notes')
        )


# Singleton
_persistent_store: Optional[PersistentPredictionStore] = None


def get_persistent_store() -> PersistentPredictionStore:
    """Get or create persistent prediction store"""
    global _persistent_store
    if _persistent_store is None:
        _persistent_store = PersistentPredictionStore()
    return _persistent_store
