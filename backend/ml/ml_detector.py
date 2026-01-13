"""
ML-Powered Network Detection
Loads trained model for real-time threat detection
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict
import json

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

MODEL_PATH = Path(__file__).parent / "models" / "trained" / "attack_detector.pt"
RESULTS_PATH = Path(__file__).parent / "models" / "trained" / "training_results.json"


class AttackClassifierNN(nn.Module):
    """Neural network for attack classification"""
    
    def __init__(self, input_dim: int = 38, hidden_dims: list = [256, 128, 64]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLNetworkDetector:
    """
    ML-powered network threat detector
    Uses trained model for real-time classification
    Supports GPU acceleration (RTX 4060, etc.)
    """
    
    def __init__(self):
        self.model = None
        self.loaded = False
        self.training_results = {}
        
        # Auto-detect GPU
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self.device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU detected: {gpu_name}")
        else:
            self.device = torch.device("cpu")
        
        if TORCH_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load trained model from disk and move to GPU if available"""
        if not MODEL_PATH.exists():
            print(f"⚠️ Trained model not found at {MODEL_PATH}")
            return
        
        try:
            # Load model state
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            
            # Create model architecture
            input_dim = checkpoint.get('input_dim', 38)
            self.model = AttackClassifierNN(input_dim=input_dim)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)  # Move to GPU
            self.model.eval()
            
            # Load training results
            if RESULTS_PATH.exists():
                with open(RESULTS_PATH) as f:
                    self.training_results = json.load(f)
            
            self.loaded = True
            device_str = "GPU" if self.device.type == "cuda" else "CPU"
            print(f"✅ ML Detector loaded on {device_str}: {self.training_results.get('accuracy', 0)*100:.1f}% accuracy")
            
        except Exception as e:
            print(f"⚠️ Failed to load ML model: {e}")
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Predict if connection is malicious
        
        Args:
            features: Network flow features (38-dim array)
        
        Returns:
            (is_attack, confidence)
        """
        if not self.loaded or self.model is None:
            return False, 0.0
        
        try:
            # Prepare input
            if len(features) < 38:
                features = np.pad(features, (0, 38 - len(features)))
            elif len(features) > 38:
                features = features[:38]
            
            x = torch.FloatTensor(features).unsqueeze(0)
            
            # Predict
            with torch.no_grad():
                output = self.model(x)
                confidence = output.item()
            
            # Threshold at 0.5
            is_attack = confidence > 0.5
            
            return is_attack, confidence
            
        except Exception as e:
            return False, 0.0
    
    def extract_features(self, connection: Dict) -> np.ndarray:
        """
        Extract features from connection data for ML prediction
        
        Features based on CIC-IDS 2017 + UNSW-NB15 feature set
        """
        features = np.zeros(38)
        
        try:
            # Port-based features
            remote_port = connection.get('remote_port', 0)
            local_port = connection.get('local_port', 0)
            
            # Normalize ports
            features[0] = remote_port / 65535.0
            features[1] = local_port / 65535.0
            
            # Port categories
            features[2] = 1.0 if remote_port < 1024 else 0.0  # Well-known
            features[3] = 1.0 if 1024 <= remote_port < 49152 else 0.0  # Registered
            features[4] = 1.0 if remote_port >= 49152 else 0.0  # Dynamic
            
            # Suspicious port indicators
            suspicious_ports = [4444, 5555, 6666, 6667, 12345, 31337, 27374]
            features[5] = 1.0 if remote_port in suspicious_ports else 0.0
            features[6] = 1.0 if local_port in suspicious_ports else 0.0
            
            # Common safe ports
            safe_ports = [80, 443, 8080, 8443, 53, 22, 21, 25, 587]
            features[7] = 1.0 if remote_port in safe_ports else 0.0
            
            # IP-based features (if available from connection data)
            remote_ip = connection.get('remote_ip', '')
            features[8] = 1.0 if remote_ip.startswith(('192.168.', '10.', '172.')) else 0.0
            
            # Status-based features
            status = connection.get('status', '')
            features[9] = 1.0 if status == 'ESTABLISHED' else 0.0
            features[10] = 1.0 if status == 'TIME_WAIT' else 0.0
            features[11] = 1.0 if status == 'CLOSE_WAIT' else 0.0
            
            # Anomaly score (if previously calculated)
            features[12] = connection.get('anomaly_score', 0.0)
            
        except Exception:
            pass
        
        return features
    
    def get_model_info(self) -> Dict:
        """Return model performance metrics"""
        return {
            "loaded": self.loaded,
            "accuracy": self.training_results.get("accuracy", 0),
            "precision": self.training_results.get("precision", 0),
            "recall": self.training_results.get("recall", 0),
            "f1": self.training_results.get("f1", 0),
            "fpr": self.training_results.get("fpr", 0),
            "samples_trained": self.training_results.get("samples", 0)
        }


# Singleton instance
_detector = None

def get_ml_detector() -> MLNetworkDetector:
    """Get or create ML detector instance"""
    global _detector
    if _detector is None:
        _detector = MLNetworkDetector()
    return _detector


if __name__ == "__main__":
    # Test the detector
    detector = get_ml_detector()
    print("\nModel Info:")
    for k, v in detector.get_model_info().items():
        print(f"  {k}: {v}")
    
    # Test prediction
    test_features = np.random.randn(38)
    is_attack, confidence = detector.predict(test_features)
    print(f"\nTest prediction: is_attack={is_attack}, confidence={confidence:.3f}")
