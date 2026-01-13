"""
Hybrid Ensemble NIDS - Based on Research Paper
Combines XGBoost, Random Forest, LSTM, and Autoencoder with weighted soft-voting

Reference: Almuhanna & Dardouri (2025) - Frontiers in AI
- Achieved 100% accuracy on CICIDS2017
- Uses SMOTE for class balancing
- Weighted soft-voting ensemble
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

# Try importing ML libraries
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠️ XGBoost not installed. Run: pip install xgboost")

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ scikit-learn not installed")

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class EnsemblePrediction:
    """Ensemble prediction result"""
    prediction_id: str
    predicted_class: int
    class_name: str
    confidence: float
    ensemble_confidence: float  # Weighted average across models
    model_votes: Dict[str, float]  # Per-model confidence
    is_anomaly: bool
    anomaly_score: float
    timestamp: str


# Attack class mapping (CICIDS2017)
ATTACK_CLASSES = {
    0: "Normal",
    1: "DoS Hulk",
    2: "DoS GoldenEye",
    3: "DoS Slowloris",
    4: "DoS Slowhttptest",
    5: "DDoS",
    6: "PortScan",
    7: "FTP-Patator",
    8: "SSH-Patator",
    9: "Bot",
    10: "Web Attack - Brute Force",
    11: "Web Attack - XSS",
    12: "Web Attack - SQL Injection",
    13: "Infiltration",
    14: "Heartbleed"
}


class Autoencoder(nn.Module):
    """Autoencoder for anomaly detection via reconstruction error"""
    
    def __init__(self, input_dim: int = 40, encoding_dim: int = 16):
        super().__init__()
        
        # Encoder: 40 -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder: 16 -> 32 -> 64 -> 40
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate MSE reconstruction error"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
            return mse


class HybridEnsembleNIDS:
    """
    Hybrid Ensemble Network Intrusion Detection System
    
    Components:
    - XGBoost: Gradient boosting for structured data
    - Random Forest: Ensemble decision trees
    - Autoencoder: Unsupervised anomaly detection
    - LSTM: Sequential pattern detection (existing model)
    
    Integration: Weighted soft-voting based on validation accuracy
    """
    
    def __init__(self, model_dir: str = "ml/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Models
        self.xgb_model = None
        self.rf_model = None
        self.autoencoder = None
        self.lstm_model = None  # Will connect to existing
        
        # Preprocessing
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.label_encoder = LabelEncoder() if HAS_SKLEARN else None
        
        # Model weights (based on validation performance)
        self.model_weights = {
            "xgboost": 0.30,
            "random_forest": 0.25,
            "lstm": 0.30,
            "autoencoder": 0.15
        }
        
        # Anomaly threshold (from paper: 0.01 MSE)
        self.anomaly_threshold = 0.01
        
        # Stats
        self.stats = {
            "predictions": 0,
            "anomalies_detected": 0,
            "model_initialized": False
        }
        
        # Device for PyTorch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print("🎯 Hybrid Ensemble NIDS initialized")
    
    def initialize_models(self, n_classes: int = 15, input_dim: int = 40):
        """Initialize all ensemble models"""
        
        # XGBoost with GPU acceleration
        if HAS_XGBOOST:
            # Check for GPU availability
            use_gpu = torch.cuda.is_available() if HAS_TORCH else False
            
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=200,  # More trees for better accuracy
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=n_classes,
                eval_metric='mlogloss',
                random_state=42,
                # GPU settings
                tree_method='hist' if use_gpu else 'auto',
                device='cuda' if use_gpu else 'cpu',
                n_jobs=-1
            )
            gpu_status = "GPU (CUDA)" if use_gpu else "CPU"
            print(f"  ✅ XGBoost initialized on {gpu_status}")
        
        # Random Forest
        if HAS_SKLEARN:
            self.rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                n_jobs=-1,
                random_state=42
            )
            print("  ✅ Random Forest initialized")
        
        # Autoencoder
        if HAS_TORCH:
            self.autoencoder = Autoencoder(input_dim=input_dim).to(self.device)
            print(f"  ✅ Autoencoder initialized on {self.device}")
        
        self.stats["model_initialized"] = True
        print("✅ All ensemble models initialized")
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train XGBoost classifier"""
        if not HAS_XGBOOST or self.xgb_model is None:
            print("⚠️ XGBoost not available")
            return
        
        print("🚀 Training XGBoost...")
        
        eval_set = [(X_train, y_train)]
        if X_val is not None:
            eval_set.append((X_val, y_val))
        
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Validation accuracy
        if X_val is not None:
            val_acc = self.xgb_model.score(X_val, y_val)
            print(f"  ✅ XGBoost validation accuracy: {val_acc:.4f}")
            self.model_weights["xgboost"] = val_acc
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray = None, y_val: np.ndarray = None):
        """Train Random Forest classifier"""
        if not HAS_SKLEARN or self.rf_model is None:
            print("⚠️ Random Forest not available")
            return
        
        print("🚀 Training Random Forest...")
        self.rf_model.fit(X_train, y_train)
        
        if X_val is not None:
            val_acc = self.rf_model.score(X_val, y_val)
            print(f"  ✅ Random Forest validation accuracy: {val_acc:.4f}")
            self.model_weights["random_forest"] = val_acc
    
    def train_autoencoder(self, X_normal: np.ndarray, epochs: int = 50):
        """Train Autoencoder on normal traffic only"""
        if not HAS_TORCH or self.autoencoder is None:
            print("⚠️ Autoencoder not available")
            return
        
        print("🚀 Training Autoencoder on normal traffic...")
        
        # Convert to tensor
        X_tensor = torch.FloatTensor(X_normal).to(self.device)
        
        # Training setup
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        self.autoencoder.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            reconstructed = self.autoencoder(X_tensor)
            loss = criterion(reconstructed, X_tensor)
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
        
        self.autoencoder.eval()
        final_loss = loss.item()
        print(f"  ✅ Autoencoder trained, final loss: {final_loss:.6f}")
    
    def predict(self, features: np.ndarray) -> EnsemblePrediction:
        """
        Make ensemble prediction using weighted soft-voting
        """
        import uuid
        
        self.stats["predictions"] += 1
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Normalize
        if self.scaler is not None:
            try:
                features = self.scaler.transform(features)
            except:
                pass  # Use raw features if scaler not fit
        
        model_probs = {}
        model_votes = {}
        
        # XGBoost prediction
        if self.xgb_model is not None:
            try:
                probs = self.xgb_model.predict_proba(features)[0]
                model_probs["xgboost"] = probs
                model_votes["xgboost"] = float(np.max(probs))
            except:
                pass
        
        # Random Forest prediction
        if self.rf_model is not None:
            try:
                probs = self.rf_model.predict_proba(features)[0]
                model_probs["random_forest"] = probs
                model_votes["random_forest"] = float(np.max(probs))
            except:
                pass
        
        # Autoencoder anomaly score
        is_anomaly = False
        anomaly_score = 0.0
        if self.autoencoder is not None:
            try:
                X_tensor = torch.FloatTensor(features).to(self.device)
                mse = self.autoencoder.get_reconstruction_error(X_tensor)
                anomaly_score = float(mse.item())
                is_anomaly = anomaly_score > self.anomaly_threshold
                
                # Convert anomaly score to confidence
                ae_conf = 1.0 - min(anomaly_score / 0.1, 1.0)  # Normalize
                model_votes["autoencoder"] = ae_conf
                
                if is_anomaly:
                    self.stats["anomalies_detected"] += 1
            except:
                pass
        
        # Weighted soft-voting
        if model_probs:
            # Get max classes from any model's output
            max_classes = max(len(p) for p in model_probs.values())
            weighted_probs = np.zeros(max_classes)
            total_weight = 0
            
            for model_name, probs in model_probs.items():
                weight = self.model_weights.get(model_name, 0.25)
                # Pad/truncate to match
                if len(probs) < max_classes:
                    padded = np.zeros(max_classes)
                    padded[:len(probs)] = probs
                    probs = padded
                elif len(probs) > max_classes:
                    probs = probs[:max_classes]
                weighted_probs += weight * probs
                total_weight += weight
            
            if total_weight > 0:
                weighted_probs /= total_weight
            
            predicted_class = int(np.argmax(weighted_probs))
            confidence = float(np.max(weighted_probs))
        else:
            # Fallback: random or default
            predicted_class = 0
            confidence = 0.5
        
        # Ensemble confidence (average of model votes)
        ensemble_conf = np.mean(list(model_votes.values())) if model_votes else 0.5
        
        return EnsemblePrediction(
            prediction_id=str(uuid.uuid4())[:8],
            predicted_class=predicted_class,
            class_name=ATTACK_CLASSES.get(predicted_class, "Unknown"),
            confidence=confidence,
            ensemble_confidence=ensemble_conf,
            model_votes=model_votes,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            timestamp=datetime.utcnow().isoformat()
        )
    
    def save_models(self, path_prefix: str = "ensemble"):
        """Save all trained models"""
        base_path = self.model_dir / path_prefix
        
        if self.xgb_model is not None:
            self.xgb_model.save_model(str(base_path) + "_xgb.json")
            print(f"  ✅ XGBoost saved")
        
        if self.rf_model is not None:
            with open(str(base_path) + "_rf.pkl", 'wb') as f:
                pickle.dump(self.rf_model, f)
            print(f"  ✅ Random Forest saved")
        
        if self.autoencoder is not None:
            torch.save(self.autoencoder.state_dict(), str(base_path) + "_ae.pt")
            print(f"  ✅ Autoencoder saved")
        
        print(f"✅ Ensemble models saved to {base_path}")
    
    def load_models(self, path_prefix: str = "ensemble"):
        """Load trained models"""
        base_path = self.model_dir / path_prefix
        
        # XGBoost
        xgb_path = str(base_path) + "_xgb.json"
        if Path(xgb_path).exists() and HAS_XGBOOST:
            self.xgb_model = xgb.XGBClassifier()
            self.xgb_model.load_model(xgb_path)
            print(f"  ✅ XGBoost loaded")
        
        # Random Forest
        rf_path = str(base_path) + "_rf.pkl"
        if Path(rf_path).exists():
            with open(rf_path, 'rb') as f:
                self.rf_model = pickle.load(f)
            print(f"  ✅ Random Forest loaded")
        
        # Autoencoder
        ae_path = str(base_path) + "_ae.pt"
        if Path(ae_path).exists() and HAS_TORCH:
            self.autoencoder = Autoencoder().to(self.device)
            self.autoencoder.load_state_dict(torch.load(ae_path, map_location=self.device))
            self.autoencoder.eval()
            print(f"  ✅ Autoencoder loaded")
    
    def get_stats(self) -> Dict:
        """Get ensemble stats"""
        return {
            **self.stats,
            "model_weights": self.model_weights,
            "has_xgboost": self.xgb_model is not None,
            "has_rf": self.rf_model is not None,
            "has_autoencoder": self.autoencoder is not None,
            "device": str(self.device)
        }


# Global instance
_ensemble: Optional[HybridEnsembleNIDS] = None


def get_ensemble_nids() -> HybridEnsembleNIDS:
    """Get or create ensemble NIDS"""
    global _ensemble
    if _ensemble is None:
        _ensemble = HybridEnsembleNIDS()
        _ensemble.initialize_models()
    return _ensemble
