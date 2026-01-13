"""
Production Ensemble Classifier
Combines best models for maximum accuracy

Models:
- synthetic_classifier.pt (90.08%) - Best on synthetic attacks
- combined_classifier.pt (84.78%) - Best on real CICIDS/UNSW data

Voting Strategy: Weighted average with confidence calibration
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


class SyntheticClassifier(nn.Module):
    """Architecture matching synthetic_classifier.pt"""
    def __init__(self, input_dim=40, hidden_dim=256, n_classes=11):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class CombinedClassifier(nn.Module):
    """Architecture matching combined_classifier.pt (ResidualAttackClassifier)"""
    def __init__(self, input_dim=40, hidden_dim=256, n_classes=12):
        super().__init__()
        
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        self.fc4 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn4 = nn.BatchNorm1d(hidden_dim // 2)
        
        self.fc_out = nn.Linear(hidden_dim // 2, n_classes)
        
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.input_bn(x)
        
        # Block 1
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        
        # Block 2 with residual
        identity = out
        out = self.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        out = out + identity
        
        # Block 3 with residual
        identity = out
        out = self.relu(self.bn3(self.fc3(out)))
        out = self.dropout(out)
        out = out + identity
        
        # Output block
        out = self.relu(self.bn4(self.fc4(out)))
        out = self.dropout(out)
        out = self.fc_out(out)
        
        return out


class ProductionEnsemble:
    """
    Production-ready ensemble combining best models
    
    Uses weighted voting based on model confidence and historical accuracy
    """
    
    def __init__(self, models_dir: str = "ml/models/trained"):
        self.models_dir = Path(models_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.models: Dict[str, nn.Module] = {}
        self.weights: Dict[str, float] = {}
        
        # Class names
        self.class_names = [
            "Normal", "DoS/DDoS", "Recon/Scan", "Brute Force",
            "Web/Exploit", "Infiltration", "Botnet", "Backdoor",
            "Worms", "Fuzzers", "Other", "Unknown"
        ]
        
        # Load models
        self._load_models()
        
        print(f"\n🧠 Production Ensemble Ready")
        print(f"   Device: {self.device}")
        print(f"   Models: {len(self.models)}")
        for name, weight in self.weights.items():
            print(f"   - {name}: weight={weight:.2f}")
    
    def _load_models(self):
        """Load all available models"""
        
        # Model 1: Synthetic Classifier (90.08%)
        try:
            model = SyntheticClassifier().to(self.device)
            weights_path = self.models_dir / "synthetic_classifier.pt"
            model.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=True))
            model.eval()
            self.models["synthetic"] = model
            self.weights["synthetic"] = 0.55  # Higher weight (better accuracy)
            print(f"✅ Loaded synthetic_classifier.pt (90.08%)")
        except Exception as e:
            print(f"⚠️ Could not load synthetic_classifier: {e}")
        
        # Model 2: Combined Classifier (84.78%)
        try:
            model = CombinedClassifier().to(self.device)
            weights_path = self.models_dir / "combined_classifier.pt"
            checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
            
            # Handle different save formats
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
            
            model.eval()
            self.models["combined"] = model
            self.weights["combined"] = 0.45  # Lower weight
            print(f"✅ Loaded combined_classifier.pt (84.78%)")
        except Exception as e:
            print(f"⚠️ Could not load combined_classifier: {e}")
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Make ensemble prediction
        
        Args:
            features: Input features (n_samples, n_features) or (n_features,)
        
        Returns:
            Dictionary with prediction details
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Convert to tensor
        x = torch.FloatTensor(features).to(self.device)
        
        # Get predictions from each model
        all_probs = []
        model_predictions = {}
        
        with torch.no_grad():
            for name, model in self.models.items():
                try:
                    outputs = model(x)
                    
                    # Handle different output sizes
                    if name == "combined" and outputs.shape[1] == 12:
                        # Merge class 11 (Unknown) into class 10 (Other)
                        outputs_11 = outputs[:, :11]
                        outputs_11[:, 10] += outputs[:, 11]
                        outputs = outputs_11
                    
                    probs = torch.softmax(outputs, dim=1)
                    
                    # Weight the probabilities
                    weighted_probs = probs * self.weights[name]
                    all_probs.append(weighted_probs)
                    
                    # Store individual predictions
                    pred_class = probs.argmax(dim=1).cpu().numpy()[0]
                    confidence = probs.max(dim=1)[0].cpu().numpy()[0]
                    model_predictions[name] = {
                        "class": int(pred_class),
                        "class_name": self.class_names[pred_class] if pred_class < len(self.class_names) else "Unknown",
                        "confidence": float(confidence)
                    }
                except Exception as e:
                    print(f"⚠️ Prediction error for {name}: {e}")
        
        if not all_probs:
            return {"error": "No models available"}
        
        # Combine weighted probabilities
        ensemble_probs = torch.stack(all_probs).sum(dim=0)
        ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=1, keepdim=True)  # Normalize
        
        # Get ensemble prediction
        pred_class = ensemble_probs.argmax(dim=1).cpu().numpy()[0]
        confidence = ensemble_probs.max(dim=1)[0].cpu().numpy()[0]
        
        # Risk level
        if confidence > 0.9:
            risk_level = "Critical" if pred_class > 0 else "Safe"
        elif confidence > 0.7:
            risk_level = "High Risk" if pred_class > 0 else "Low Risk"
        elif confidence > 0.5:
            risk_level = "Suspicious"
        else:
            risk_level = "Unknown"
        
        return {
            "ensemble": {
                "class": int(pred_class),
                "class_name": self.class_names[pred_class] if pred_class < len(self.class_names) else "Unknown",
                "confidence": float(confidence),
                "risk_level": risk_level,
                "probabilities": ensemble_probs[0].cpu().numpy().tolist()
            },
            "individual_models": model_predictions,
            "weights": self.weights
        }
    
    def predict_batch(self, features: np.ndarray) -> List[Dict]:
        """Predict on batch of samples"""
        results = []
        for i in range(len(features)):
            results.append(self.predict(features[i]))
        return results
    
    def get_model_info(self) -> Dict:
        """Get ensemble model information"""
        return {
            "models": list(self.models.keys()),
            "weights": self.weights,
            "device": str(self.device),
            "n_classes": len(self.class_names),
            "class_names": self.class_names
        }


def test_ensemble():
    """Test the production ensemble"""
    print("=" * 60)
    print("🧠 PRODUCTION ENSEMBLE TEST")
    print("=" * 60)
    
    ensemble = ProductionEnsemble()
    
    # Test with random samples
    print("\n📊 Testing with sample inputs...\n")
    
    test_cases = [
        ("Normal traffic", np.random.uniform(0.0, 0.2, 40).astype(np.float32)),
        ("DoS attack", np.concatenate([np.array([0.01, 0.9, 0.5, 0.1]), np.random.uniform(0.2, 0.6, 36)]).astype(np.float32)),
        ("Port scan", np.concatenate([np.array([0.001, 0.05, 0.1, 0.9]), np.random.uniform(0.3, 0.7, 36)]).astype(np.float32)),
        ("Brute force", np.concatenate([np.array([0.5, 0.3, 0.2, 0.1, 0.9]), np.random.uniform(0.2, 0.5, 35)]).astype(np.float32)),
    ]
    
    for name, features in test_cases:
        result = ensemble.predict(features)
        
        print(f"Test: {name}")
        print(f"  Ensemble: {result['ensemble']['class_name']} "
              f"({result['ensemble']['confidence']:.1%}) - {result['ensemble']['risk_level']}")
        
        for model_name, pred in result['individual_models'].items():
            print(f"    {model_name}: {pred['class_name']} ({pred['confidence']:.1%})")
        print()
    
    print("=" * 60)
    print("✅ Ensemble test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_ensemble()
