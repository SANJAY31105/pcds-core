"""
Production Model Server
TorchScript/ONNX deployment with versioning and monitoring

Features:
- Model export (TorchScript, ONNX)
- Version metadata
- Shadow mode inference
- Kafka logging
- 30-day prediction storage
- Latency tracking
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque
import json
import time
import hashlib
import threading


# ============================================
# Model Registry
# ============================================

@dataclass
class ModelVersion:
    """Model version metadata"""
    version: str
    name: str
    accuracy: float
    created_at: str
    input_dim: int
    n_classes: int
    model_hash: str
    training_samples: int
    framework: str  # pytorch, torchscript, onnx


class ModelRegistry:
    """Central model registry with versioning"""
    
    def __init__(self, registry_path: str = "ml/models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, ModelVersion] = {}
        self.active_version: Optional[str] = None
        
        self._load_registry()
    
    def _load_registry(self):
        """Load registry from disk"""
        registry_file = self.registry_path / "registry.json"
        if registry_file.exists():
            with open(registry_file) as f:
                data = json.load(f)
                for v, meta in data.get("models", {}).items():
                    self.models[v] = ModelVersion(**meta)
                self.active_version = data.get("active_version")
    
    def _save_registry(self):
        """Save registry to disk"""
        data = {
            "models": {v: asdict(m) for v, m in self.models.items()},
            "active_version": self.active_version
        }
        with open(self.registry_path / "registry.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def register(self, version: ModelVersion, model_path: str) -> str:
        """Register a new model version"""
        self.models[version.version] = version
        self._save_registry()
        return version.version
    
    def get_active(self) -> Optional[ModelVersion]:
        """Get active model version"""
        if self.active_version:
            return self.models.get(self.active_version)
        return None
    
    def set_active(self, version: str):
        """Set active model version"""
        if version in self.models:
            self.active_version = version
            self._save_registry()


# ============================================
# Model Exporter
# ============================================

class ModelExporter:
    """Export models to TorchScript and ONNX"""
    
    def __init__(self, output_dir: str = "ml/models/exported"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_torchscript(self, model: nn.Module, version: str, 
                          input_dim: int = 40) -> str:
        """Export model to TorchScript"""
        model.eval()
        
        # Create example input
        example_input = torch.randn(1, input_dim)
        
        # Trace the model
        traced = torch.jit.trace(model, example_input)
        
        # Save
        output_path = self.output_dir / f"{version}_torchscript.pt"
        traced.save(str(output_path))
        
        print(f"✅ Exported TorchScript: {output_path}")
        return str(output_path)
    
    def export_onnx(self, model: nn.Module, version: str,
                   input_dim: int = 40) -> str:
        """Export model to ONNX"""
        model.eval()
        
        # Create example input
        example_input = torch.randn(1, input_dim)
        
        output_path = self.output_dir / f"{version}.onnx"
        
        torch.onnx.export(
            model,
            example_input,
            str(output_path),
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={
                "features": {0: "batch_size"},
                "logits": {0: "batch_size"}
            },
            opset_version=14
        )
        
        print(f"✅ Exported ONNX: {output_path}")
        return str(output_path)
    
    def compute_model_hash(self, model: nn.Module) -> str:
        """Compute hash of model weights"""
        hasher = hashlib.md5()
        for param in model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        return hasher.hexdigest()[:12]


# ============================================
# Prediction Store (30-day retention)
# ============================================

@dataclass
class Prediction:
    """Single prediction record"""
    prediction_id: str
    timestamp: str
    model_version: str
    input_hash: str
    features: List[float]
    predicted_class: int
    class_name: str
    confidence: float
    probabilities: List[float]
    latency_ms: float
    source_ip: Optional[str] = None
    source_host: Optional[str] = None
    ground_truth: Optional[int] = None  # Set by analyst
    is_tp: Optional[bool] = None
    is_fp: Optional[bool] = None
    # Enterprise fields
    severity: str = "medium"  # critical, high, medium, low
    mitre_technique: Optional[str] = None  # T1059, etc.
    mitre_tactic: Optional[str] = None  # Execution, etc.
    top_features: Optional[List[str]] = None  # Top contributing features
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[str] = None


class PredictionStore:
    """Store predictions with 30-day retention"""
    
    def __init__(self, max_days: int = 30, max_predictions: int = 1000000):
        self.max_days = max_days
        self.max_predictions = max_predictions
        
        self.predictions: deque = deque(maxlen=max_predictions)
        self.predictions_by_id: Dict[str, Prediction] = {}
        
        # Indexes for fast lookup
        self.by_class: Dict[int, List[str]] = {}
        self.by_host: Dict[str, List[str]] = {}
        
        # Stats
        self.stats = {
            "total": 0,
            "by_class": {},
            "by_confidence_bucket": {"0-50": 0, "50-70": 0, "70-90": 0, "90-100": 0},
            "tp_count": 0,
            "fp_count": 0,
            "pending_review": 0
        }
        
        self._lock = threading.Lock()
    
    def add(self, prediction: Prediction):
        """Add prediction to store"""
        with self._lock:
            self.predictions.append(prediction)
            self.predictions_by_id[prediction.prediction_id] = prediction
            
            # Update indexes
            if prediction.predicted_class not in self.by_class:
                self.by_class[prediction.predicted_class] = []
            self.by_class[prediction.predicted_class].append(prediction.prediction_id)
            
            if prediction.source_host:
                if prediction.source_host not in self.by_host:
                    self.by_host[prediction.source_host] = []
                self.by_host[prediction.source_host].append(prediction.prediction_id)
            
            # Update stats
            self.stats["total"] += 1
            self.stats["by_class"][prediction.class_name] = \
                self.stats["by_class"].get(prediction.class_name, 0) + 1
            
            conf = prediction.confidence * 100
            if conf < 50:
                self.stats["by_confidence_bucket"]["0-50"] += 1
            elif conf < 70:
                self.stats["by_confidence_bucket"]["50-70"] += 1
            elif conf < 90:
                self.stats["by_confidence_bucket"]["70-90"] += 1
            else:
                self.stats["by_confidence_bucket"]["90-100"] += 1
            
            self.stats["pending_review"] += 1
    
    def set_ground_truth(self, prediction_id: str, ground_truth: int, 
                        is_tp: bool, is_fp: bool,
                        reviewed_by: str = None, notes: str = None):
        """Set ground truth from analyst feedback with audit info"""
        with self._lock:
            if prediction_id in self.predictions_by_id:
                pred = self.predictions_by_id[prediction_id]
                pred.ground_truth = ground_truth
                pred.is_tp = is_tp
                pred.is_fp = is_fp
                pred.reviewed_by = reviewed_by
                pred.reviewed_at = datetime.now().isoformat()
                
                if is_tp:
                    self.stats["tp_count"] += 1
                if is_fp:
                    self.stats["fp_count"] += 1
                self.stats["pending_review"] -= 1
    
    def get_recent(self, limit: int = 100) -> List[Prediction]:
        """Get recent predictions"""
        return list(self.predictions)[-limit:]
    
    def get(self, prediction_id: str) -> Optional[Prediction]:
        """Get prediction by ID"""
        return self.predictions_by_id.get(prediction_id)
    
    def get_by_class(self, class_id: int, limit: int = 100) -> List[Prediction]:
        """Get predictions by class"""
        ids = self.by_class.get(class_id, [])[-limit:]
        return [self.predictions_by_id[pid] for pid in ids if pid in self.predictions_by_id]
    
    def get_pending_review(self, limit: int = 50) -> List[Prediction]:
        """Get predictions pending analyst review"""
        pending = [p for p in self.predictions if p.ground_truth is None]
        return pending[-limit:]
    
    def get_stats(self) -> Dict:
        """Get prediction statistics"""
        with self._lock:
            return {
                **self.stats,
                "fp_rate": self.stats["fp_count"] / max(self.stats["tp_count"] + self.stats["fp_count"], 1),
                "top_classes": sorted(self.stats["by_class"].items(), key=lambda x: -x[1])[:5],
                "top_hosts": sorted([(h, len(ids)) for h, ids in self.by_host.items()], 
                                   key=lambda x: -x[1])[:5]
            }
    
    def cleanup_old(self):
        """Remove predictions older than max_days"""
        cutoff = datetime.now() - timedelta(days=self.max_days)
        cutoff_str = cutoff.isoformat()
        
        with self._lock:
            # Remove old predictions
            while self.predictions and self.predictions[0].timestamp < cutoff_str:
                old = self.predictions.popleft()
                if old.prediction_id in self.predictions_by_id:
                    del self.predictions_by_id[old.prediction_id]


# ============================================
# Kafka Logger (Shadow Mode)
# ============================================

class KafkaLogger:
    """Log predictions to Kafka for shadow mode using confluent-kafka"""
    
    def __init__(self, topic: str = "ml_predictions", 
                 bootstrap_servers: str = "localhost:9092"):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.enabled = False
        
        self._buffer: List[Dict] = []
        self._buffer_lock = threading.Lock()
        self._delivery_count = 0
        
        self._try_connect()
    
    def _try_connect(self):
        """Try to connect to Kafka using confluent-kafka"""
        try:
            from confluent_kafka import Producer
            
            conf = {
                'bootstrap.servers': self.bootstrap_servers,
                'client.id': 'pcds-ml-inference',
                'acks': 'all',
                'retries': 3,
                'socket.timeout.ms': 5000,
                'message.timeout.ms': 10000
            }
            
            self.producer = Producer(conf)
            self.enabled = True
            print(f"✅ Kafka connected: {self.topic}")
            
        except ImportError:
            # Fallback to kafka-python
            try:
                from kafka import KafkaProducer
                self.producer = KafkaProducer(
                    bootstrap_servers=self.bootstrap_servers
                )
                self.enabled = True
                self._use_confluent = False
                print(f"✅ Kafka connected (kafka-python): {self.topic}")
            except Exception as e:
                print(f"⚠️ Kafka not available: {e}")
                self.enabled = False
                
        except Exception as e:
            print(f"⚠️ Kafka not available: {e}")
            self.enabled = False
    
    def _delivery_callback(self, err, msg):
        """Callback for delivery reports"""
        if err:
            print(f"⚠️ Kafka delivery failed: {err}")
        else:
            self._delivery_count += 1
    
    def log(self, prediction: Prediction):
        """Log prediction to Kafka"""
        record = asdict(prediction)
        record_bytes = json.dumps(record, default=str).encode('utf-8')
        
        if self.enabled and self.producer:
            try:
                # confluent-kafka
                if hasattr(self.producer, 'produce'):
                    self.producer.produce(
                        self.topic, 
                        value=record_bytes,
                        callback=self._delivery_callback
                    )
                    self.producer.poll(0)  # Non-blocking
                # kafka-python fallback
                elif hasattr(self.producer, 'send'):
                    self.producer.send(self.topic, record_bytes)
            except Exception as e:
                self._buffer_prediction(record)
        else:
            self._buffer_prediction(record)
    
    def _buffer_prediction(self, record: Dict):
        """Buffer prediction when Kafka unavailable"""
        with self._buffer_lock:
            self._buffer.append(record)
            # Keep only last 10000
            if len(self._buffer) > 10000:
                self._buffer = self._buffer[-10000:]
    
    def flush_buffer(self):
        """Flush buffered predictions to Kafka"""
        if not self.enabled:
            return
        
        with self._buffer_lock:
            for record in self._buffer:
                try:
                    record_bytes = json.dumps(record, default=str).encode('utf-8')
                    if hasattr(self.producer, 'produce'):
                        self.producer.produce(self.topic, value=record_bytes)
                    elif hasattr(self.producer, 'send'):
                        self.producer.send(self.topic, record_bytes)
                except:
                    break
            self._buffer = []
            
            # Flush confluent-kafka
            if hasattr(self.producer, 'flush'):
                self.producer.flush(timeout=5)
    
    def get_stats(self) -> Dict:
        """Get Kafka stats"""
        return {
            "enabled": self.enabled,
            "topic": self.topic,
            "buffer_size": len(self._buffer),
            "delivery_count": self._delivery_count
        }


# ============================================
# Production Inference Server
# ============================================

class InferenceServer:
    """
    Production inference server with shadow mode
    
    Features:
    - Model versioning
    - Latency tracking
    - Kafka logging
    - 30-day retention
    - Analyst feedback
    """
    
    def __init__(self, model_path: str = "ml/models/trained/enhanced_classifier.pt", 
                 use_persistent_store: bool = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model
        self.model = None
        self.model_version = None
        self.model_path = model_path
        
        # Components
        self.registry = ModelRegistry()
        self.exporter = ModelExporter()
        self.kafka = KafkaLogger()
        
        # Use persistent store if available, fallback to in-memory
        if use_persistent_store:
            try:
                from ml.persistent_store import get_persistent_store
                self.store = get_persistent_store()
                print("   💾 Using persistent prediction store")
            except Exception as e:
                print(f"   ⚠️ Persistent store unavailable: {e}")
                self.store = PredictionStore()
        else:
            self.store = PredictionStore()
        
        # Metrics
        self.metrics = {
            "total_inferences": 0,
            "total_latency_ms": 0,
            "latency_samples": deque(maxlen=1000),
            "errors": 0
        }
        
        # Class names
        self.class_names = [
            "Normal", "DoS/DDoS", "Recon/Scan", "Brute Force",
            "Web/Exploit", "Infiltration", "Botnet", "Backdoor",
            "Worms", "Fuzzers", "Other"
        ]
        
        self._load_model()
    
    def _load_model(self):
        """Load the production model"""
        try:
            # Load enhanced model
            from train_enhanced import EnhancedClassifier
            
            self.model = EnhancedClassifier().to(self.device)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            # Create version metadata
            model_hash = self.exporter.compute_model_hash(self.model)
            self.model_version = ModelVersion(
                version=f"v1.0.0-{model_hash}",
                name="enhanced_classifier",
                accuracy=95.71,
                created_at=datetime.now().isoformat(),
                input_dim=40,
                n_classes=11,
                model_hash=model_hash,
                training_samples=2000000,
                framework="pytorch"
            )
            
            # Register
            self.registry.register(self.model_version, self.model_path)
            self.registry.set_active(self.model_version.version)
            
            print(f"✅ Model loaded: {self.model_version.version} ({self.model_version.accuracy}%)")
            
        except Exception as e:
            print(f"⚠️ Could not load enhanced model: {e}")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load fallback synthetic model"""
        try:
            class SyntheticClassifier(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(40, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                        nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(128, 11)
                    )
                def forward(self, x): return self.network(x)
            
            self.model = SyntheticClassifier().to(self.device)
            checkpoint = torch.load("ml/models/trained/synthetic_classifier.pt", 
                                   map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint)
            self.model.eval()
            
            self.model_version = ModelVersion(
                version="v0.9.0-fallback",
                name="synthetic_classifier",
                accuracy=90.08,
                created_at=datetime.now().isoformat(),
                input_dim=40,
                n_classes=11,
                model_hash="fallback",
                training_samples=500000,
                framework="pytorch"
            )
            
            print(f"✅ Fallback model loaded: {self.model_version.version}")
            
        except Exception as e:
            print(f"❌ No models available: {e}")
    
    def predict(self, features: np.ndarray, 
               source_ip: str = None,
               source_host: str = None) -> Dict:
        """
        Make prediction with full metadata
        
        Returns:
            {
                prob: float,
                class: int,
                class_name: str,
                model_version: str,
                model_confidence: float,
                features_used: int,
                latency_ms: float,
                prediction_id: str
            }
        """
        start_time = time.perf_counter()
        
        try:
            # Prepare input
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            x = torch.FloatTensor(features).to(self.device)
            
            # Inference
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1)
                
                pred_class = probs.argmax(dim=1).item()
                confidence = probs.max(dim=1)[0].item()
                all_probs = probs[0].cpu().numpy().tolist()
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Create prediction record
            prediction_id = f"pred_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
            input_hash = hashlib.md5(features.tobytes()).hexdigest()[:8]
            
            # Calculate severity and MITRE mapping
            severity = self._calculate_severity(pred_class, confidence)
            mitre_info = self._get_mitre_mapping(pred_class)
            
            prediction = Prediction(
                prediction_id=prediction_id,
                timestamp=datetime.now().isoformat(),
                model_version=self.model_version.version if self.model_version else "unknown",
                input_hash=input_hash,
                features=features.flatten().tolist()[:10],  # Store first 10 features
                predicted_class=pred_class,
                class_name=self.class_names[pred_class],
                confidence=confidence,
                probabilities=all_probs,
                latency_ms=latency_ms,
                source_ip=source_ip,
                source_host=source_host,
                severity=severity,
                mitre_technique=mitre_info.get("technique"),
                mitre_tactic=mitre_info.get("tactic"),
                top_features=self._get_top_features(features, pred_class)
            )
            
            # Store and log
            self.store.add(prediction)
            self.kafka.log(prediction)
            
            # Update metrics
            self.metrics["total_inferences"] += 1
            self.metrics["total_latency_ms"] += latency_ms
            self.metrics["latency_samples"].append(latency_ms)
            
            return {
                "prob": confidence,
                "class": pred_class,
                "class_name": self.class_names[pred_class],
                "model_version": self.model_version.version if self.model_version else "unknown",
                "model_confidence": confidence,
                "features_used": len(features.flatten()),
                "latency_ms": round(latency_ms, 2),
                "prediction_id": prediction_id,
                "probabilities": {self.class_names[i]: round(p, 4) for i, p in enumerate(all_probs)}
            }
            
        except Exception as e:
            self.metrics["errors"] += 1
            return {"error": str(e)}
    
    def get_metrics(self) -> Dict:
        """Get inference metrics"""
        latencies = list(self.metrics["latency_samples"])
        
        return {
            "total_inferences": self.metrics["total_inferences"],
            "errors": self.metrics["errors"],
            "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
            "p50_latency_ms": float(np.percentile(latencies, 50)) if latencies else 0.0,
            "p95_latency_ms": float(np.percentile(latencies, 95)) if latencies else 0.0,
            "p99_latency_ms": float(np.percentile(latencies, 99)) if latencies else 0.0,
            "qps": float(self._calculate_qps()),
            "model_version": self.model_version.version if self.model_version else "none",
            "model_accuracy": float(self.model_version.accuracy) if self.model_version else 0.0
        }
    
    def _calculate_qps(self) -> float:
        """Calculate queries per second"""
        # Approximate from recent predictions
        recent = self.store.get_recent(100)
        if len(recent) < 2:
            return 0
        
        try:
            first = datetime.fromisoformat(recent[0].timestamp)
            last = datetime.fromisoformat(recent[-1].timestamp)
            duration = (last - first).total_seconds()
            if duration > 0:
                return len(recent) / duration
        except:
            pass
        return 0
    
    def _calculate_severity(self, pred_class: int, confidence: float) -> str:
        """Calculate severity based on class and confidence"""
        # High-risk classes
        critical_classes = {1, 6, 7, 8}  # DoS, Botnet, Backdoor, Worms
        high_classes = {4, 5}  # Web/Exploit, Infiltration
        
        if pred_class in critical_classes and confidence > 0.8:
            return "critical"
        elif pred_class in critical_classes or (pred_class in high_classes and confidence > 0.7):
            return "high"
        elif confidence > 0.6:
            return "medium"
        return "low"
    
    def _get_mitre_mapping(self, pred_class: int) -> Dict:
        """Map prediction class to MITRE ATT&CK"""
        mitre_map = {
            0: {"technique": None, "tactic": None},  # Normal
            1: {"technique": "T1498", "tactic": "Impact"},  # DoS/DDoS
            2: {"technique": "T1046", "tactic": "Discovery"},  # Recon/Scan
            3: {"technique": "T1110", "tactic": "Credential Access"},  # Brute Force
            4: {"technique": "T1190", "tactic": "Initial Access"},  # Web/Exploit
            5: {"technique": "T1071", "tactic": "Command and Control"},  # Infiltration
            6: {"technique": "T1583", "tactic": "Resource Development"},  # Botnet
            7: {"technique": "T1059", "tactic": "Execution"},  # Backdoor
            8: {"technique": "T1210", "tactic": "Lateral Movement"},  # Worms
            9: {"technique": "T1499", "tactic": "Impact"},  # Fuzzers
            10: {"technique": None, "tactic": None},  # Other
        }
        return mitre_map.get(pred_class, {"technique": None, "tactic": None})
    
    def _get_top_features(self, features: np.ndarray, pred_class: int) -> List[str]:
        """Get top contributing features (simplified SHAP-like)"""
        # Feature names for network traffic
        feature_names = [
            "duration", "protocol", "src_bytes", "dst_bytes", "count",
            "same_srv_rate", "diff_srv_rate", "srv_count", "dst_host_count",
            "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
            "flag", "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
            "logged_in", "num_compromised", "root_shell", "su_attempted",
            "num_root", "num_file_creations", "num_shells", "num_access_files",
            "num_outbound_cmds", "is_host_login", "is_guest_login", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_serror_rate",
            "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]
        
        # Get feature values and find top 3 by magnitude
        flat = features.flatten()[:len(feature_names)]
        indexed = [(i, abs(v), v) for i, v in enumerate(flat)]
        indexed.sort(key=lambda x: -x[1])
        
        top_3 = []
        for i, _, val in indexed[:3]:
            if i < len(feature_names):
                top_3.append(f"{feature_names[i]}={val:.2f}")
        
        return top_3
    
    def submit_feedback(self, prediction_id: str, is_correct: bool, 
                       true_class: int = None, reviewed_by: str = "analyst",
                       notes: str = None) -> Dict:
        """Submit analyst feedback with audit trail"""
        is_tp = is_correct
        is_fp = not is_correct
        
        # If true_class not provided, get the predicted class from the prediction
        if true_class is None:
            pred = self.store.get(prediction_id)
            if pred:
                true_class = pred.predicted_class
            else:
                true_class = 0  # Default to Normal
        
        # Update prediction in store
        self.store.set_ground_truth(
            prediction_id, true_class, is_tp, is_fp,
            reviewed_by=reviewed_by, notes=notes
        )
        
        return {
            "success": True, 
            "prediction_id": prediction_id,
            "feedback_type": "TP" if is_tp else "FP",
            "reviewed_by": reviewed_by
        }
    
    def get_pending_predictions(self, limit: int = 50,
                                min_confidence: float = 0,
                                severity: str = None,
                                source_host: str = None) -> Dict:
        """Get predictions awaiting analyst review with filters"""
        pending = self.store.get_pending_review(limit=500)  # Get more, then filter
        
        # Apply filters
        if min_confidence > 0:
            pending = [p for p in pending if p.confidence >= min_confidence]
        if severity:
            pending = [p for p in pending if getattr(p, 'severity', 'medium') == severity]
        if source_host:
            pending = [p for p in pending if source_host.lower() in (p.source_host or "").lower()]
        
        # Return limited results with full metadata
        return {
            "predictions": [self._prediction_to_dict(p) for p in pending[:limit]],
            "total_pending": self.store.stats["pending_review"],
            "filtered_count": len(pending[:limit])
        }
    
    def _prediction_to_dict(self, p: Prediction) -> Dict:
        """Convert prediction to frontend-friendly dict"""
        return {
            "prediction_id": p.prediction_id,
            "timestamp": p.timestamp,
            "model_version": p.model_version,
            "predicted_class": p.predicted_class,
            "class_name": p.class_name,
            "confidence": round(p.confidence, 4),
            "severity": getattr(p, 'severity', 'medium'),
            "mitre_technique": getattr(p, 'mitre_technique', None),
            "mitre_tactic": getattr(p, 'mitre_tactic', None),
            "top_features": getattr(p, 'top_features', None),
            "source_ip": p.source_ip,
            "source_host": p.source_host
        }
    
    def get_dashboard_data(self) -> Dict:
        """Get data for monitoring dashboard"""
        stats = self.store.get_stats()
        metrics = self.get_metrics()
        
        # Ensure all values are JSON-serializable
        return {
            "inference_rate": float(metrics["qps"]),
            "total_predictions": int(stats["total"]),
            "latency": {
                "avg": float(metrics["avg_latency_ms"]),
                "p50": float(metrics["p50_latency_ms"]),
                "p95": float(metrics["p95_latency_ms"]),
                "target_met": bool(metrics["p95_latency_ms"] < 50)
            },
            "top_classes": [(str(k), int(v)) for k, v in stats["top_classes"]],
            "top_hosts": [(str(k), int(v)) for k, v in stats["top_hosts"]],
            "confidence_distribution": {str(k): int(v) for k, v in stats["by_confidence_bucket"].items()},
            "feedback": {
                "tp_count": int(stats["tp_count"]),
                "fp_count": int(stats["fp_count"]),
                "fp_rate": float(stats["fp_rate"]),
                "pending_review": int(stats["pending_review"])
            },
            "model": {
                "version": str(self.model_version.version) if self.model_version else "none",
                "accuracy": float(self.model_version.accuracy) if self.model_version else 0.0
            }
        }


# Global instance
_server: Optional[InferenceServer] = None


def get_inference_server() -> InferenceServer:
    """Get or create inference server"""
    global _server
    if _server is None:
        _server = InferenceServer()
    return _server


if __name__ == "__main__":
    # Test the server
    server = get_inference_server()
    
    # Test prediction
    features = np.random.uniform(0, 1, 40).astype(np.float32)
    result = server.predict(features, source_ip="192.168.1.100", source_host="workstation-01")
    
    print("\n📊 Test Prediction:")
    for k, v in result.items():
        if k != "probabilities":
            print(f"   {k}: {v}")
    
    print("\n📈 Metrics:")
    for k, v in server.get_metrics().items():
        print(f"   {k}: {v}")
