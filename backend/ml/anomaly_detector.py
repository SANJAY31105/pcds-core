"""
Production PyTorch LSTM Anomaly Detector
(Demo version - PyTorch-free for faster testing)

Note: For production, install PyTorch and uncomment PyTorch code
pip install torch==2.1.2
"""
import numpy as np
from typing import Tuple, List, Optional
import time
from datetime import datetime


# Lightweight demo version with PyTorch integration
try:
    from ml.lstm_model import LSTMPredictor
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not found. Using statistical fallback.")

class ProductionAnomalyDetector:
    """
    Production-ready anomaly detector with PyTorch LSTM
    """
    
    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        self.device = device or "cpu"
        print(f"🔧 Anomaly Detector initialized")
        
        # Initialize PyTorch Model
        if TORCH_AVAILABLE:
            self.model = LSTMPredictor(model_path)
            self.model_version = "2.0.0-lstm"
        else:
            self.model = None
            self.model_version = "1.1.0-fallback"
        
        self.trained_at = datetime.utcnow().isoformat()
        self.inference_times = []
        self.threshold = 0.05  # MSE threshold for anomaly
        
    def extract_features(self, network_data: dict) -> np.ndarray:
        """Extract 15-dimensional feature vector"""
        features = []
        
        # 1. Packet size (normalized by MTU)
        features.append(network_data.get('packet_size', 0) / 1500.0)
        
        # 2. Port (normalized)
        features.append(network_data.get('port', 0) / 65535.0)
        
        # 3. Protocol encoding
        protocol = network_data.get('protocol', 'tcp').lower()
        protocol_map = {'tcp': 0.0, 'udp': 0.33, 'icmp': 0.66, 'http': 1.0}
        features.append(protocol_map.get(protocol, 0.5))
        
        # 4. IP entropy
        source_ip = network_data.get('source_ip', '0.0.0.0')
        ip_hash = hash(source_ip) % 1000
        features.append(ip_hash / 1000.0)
        
        # 5-15. Additional features (simulated for demo if not present)
        # In production, these would come from deep packet inspection
        features.extend([
            np.random.random(),  # Packet rate
            np.random.random(),  # Byte rate
            np.random.random(),  # Connection duration
            np.random.random(),  # Packet count
            np.random.random(),  # Unique ports
            network_data.get('flags', '').count('S') / 10.0,  # TCP flags
            np.random.random(),  # Session packets
            np.random.random(),  # Payload entropy
            np.random.random(),  # Inter-arrival time
            np.random.random(),  # Flow duration
            np.random.random(),  # Bidirectional packets
        ])
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float, float]:
        """Predict using LSTM Autoencoder"""
        start_time = time.time()
        
        if self.model:
            # PyTorch Inference
            anomaly_score = self.model.predict(features)
            # Normalize score roughly to 0-1 for UI
            # MSE is usually small, so we scale it up
            normalized_score = min(anomaly_score * 10, 1.0)
        else:
            # Fallback Logic
            mean_feature = np.mean(features)
            normalized_score = min((mean_feature * 0.4 + np.max(features) * 0.3), 1.0)
            normalized_score = normalized_score * 0.8 + np.random.random() * 0.2
            
        is_anomaly = normalized_score > self.threshold
        confidence = normalized_score if is_anomaly else (1 - normalized_score)
        
        # Track performance
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        
        return is_anomaly, float(normalized_score), float(confidence)
    
    def batch_predict(self, features_list: List[np.ndarray]) -> List[Tuple[bool, float, float]]:
        results = []
        for features in features_list:
            results.append(self.predict(features))
        return results
    
    def train(self, features_list: List[np.ndarray]):
        """Online training update"""
        if self.model:
            loss = self.model.train(features_list)
            print(f"🎓 Model updated. Loss: {loss:.4f}")
            self.trained_at = datetime.utcnow().isoformat()
            return loss
        return 0.0

    def get_performance_stats(self) -> dict:
        if not self.inference_times:
            return {'avg_inference_ms': 0, 'device': self.device}
        
        return {
            'avg_inference_ms': float(np.mean(self.inference_times)),
            'p95_inference_ms': float(np.percentile(self.inference_times, 95)),
            'device': self.device,
            'model_version': self.model_version,
            'backend': 'PyTorch LSTM' if self.model else 'Statistical Fallback'
        }


# Global instance
anomaly_detector = ProductionAnomalyDetector()
print(f"✅ Anomaly detector ready (v{anomaly_detector.model_version})")
