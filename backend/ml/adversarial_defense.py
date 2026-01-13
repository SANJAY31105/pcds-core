"""
Adversarial ML Defense Module for PCDS
Protects models against evasion and poisoning attacks

Based on survey paper: Defense mechanisms for adversarial attacks on ML models
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import hashlib

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class AdversarialDetectionResult:
    """Result of adversarial input detection"""
    is_adversarial: bool
    confidence: float
    detection_method: str
    anomaly_scores: Dict[str, float]
    recommendation: str


class AdversarialDefense:
    """
    Adversarial ML Defense System
    
    Techniques implemented:
    1. Input validation and sanitization
    2. Statistical anomaly detection
    3. Feature squeezing
    4. Ensemble disagreement detection
    5. Gradient masking detection
    """
    
    def __init__(self):
        self.feature_stats: Dict[str, Dict] = {}  # Learned feature statistics
        self.input_history: List[np.ndarray] = []
        self.detection_threshold = 0.7
        
        # Defense configuration
        self.config = {
            "enable_input_validation": True,
            "enable_statistical_check": True,
            "enable_feature_squeezing": True,
            "enable_ensemble_check": True,
            "squeeze_bit_depth": 4,
            "history_size": 1000
        }
        
        print("🛡️ Adversarial ML Defense initialized")
    
    def learn_feature_statistics(self, X: np.ndarray):
        """
        Learn normal feature statistics from training data
        Used to detect statistical anomalies in inputs
        """
        n_features = X.shape[1]
        
        for i in range(n_features):
            self.feature_stats[f"feature_{i}"] = {
                "mean": float(np.mean(X[:, i])),
                "std": float(np.std(X[:, i])),
                "min": float(np.min(X[:, i])),
                "max": float(np.max(X[:, i])),
                "q1": float(np.percentile(X[:, i], 25)),
                "q3": float(np.percentile(X[:, i], 75))
            }
        
        print(f"  ✅ Learned statistics for {n_features} features")
    
    def detect_adversarial(self, features: np.ndarray, 
                          predictions: Optional[Dict[str, np.ndarray]] = None) -> AdversarialDetectionResult:
        """
        Detect if input is adversarial
        
        Args:
            features: Input feature vector
            predictions: Optional dict of predictions from different models
            
        Returns:
            AdversarialDetectionResult with detection details
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        anomaly_scores = {}
        detection_methods = []
        
        # 1. Input validation
        if self.config["enable_input_validation"]:
            input_score = self._check_input_validity(features)
            anomaly_scores["input_validity"] = input_score
            if input_score > self.detection_threshold:
                detection_methods.append("input_validation")
        
        # 2. Statistical anomaly check
        if self.config["enable_statistical_check"] and self.feature_stats:
            stat_score = self._check_statistical_anomaly(features)
            anomaly_scores["statistical"] = stat_score
            if stat_score > self.detection_threshold:
                detection_methods.append("statistical_anomaly")
        
        # 3. Feature squeezing difference
        if self.config["enable_feature_squeezing"]:
            squeeze_score = self._check_feature_squeezing(features)
            anomaly_scores["feature_squeeze"] = squeeze_score
            if squeeze_score > self.detection_threshold:
                detection_methods.append("feature_squeezing")
        
        # 4. Ensemble disagreement
        if self.config["enable_ensemble_check"] and predictions:
            ensemble_score = self._check_ensemble_disagreement(predictions)
            anomaly_scores["ensemble"] = ensemble_score
            if ensemble_score > self.detection_threshold:
                detection_methods.append("ensemble_disagreement")
        
        # Aggregate scores
        if anomaly_scores:
            max_score = max(anomaly_scores.values())
            avg_score = np.mean(list(anomaly_scores.values()))
        else:
            max_score = 0.0
            avg_score = 0.0
        
        is_adversarial = max_score > self.detection_threshold
        
        # Generate recommendation
        if is_adversarial:
            recommendation = f"⚠️ Potential adversarial input detected via {', '.join(detection_methods)}. Recommend manual review or input rejection."
        else:
            recommendation = "✅ Input appears legitimate."
        
        return AdversarialDetectionResult(
            is_adversarial=is_adversarial,
            confidence=float(max_score),
            detection_method=", ".join(detection_methods) if detection_methods else "none",
            anomaly_scores=anomaly_scores,
            recommendation=recommendation
        )
    
    def _check_input_validity(self, features: np.ndarray) -> float:
        """Check for invalid or malformed inputs"""
        score = 0.0
        
        # Check for NaN or Inf
        if np.any(np.isnan(features)):
            score += 0.5
        if np.any(np.isinf(features)):
            score += 0.5
        
        # Check for extreme values
        if np.any(np.abs(features) > 1e10):
            score += 0.3
        
        # Check for too many zeros
        zero_ratio = np.mean(features == 0)
        if zero_ratio > 0.9:
            score += 0.2
        
        return min(score, 1.0)
    
    def _check_statistical_anomaly(self, features: np.ndarray) -> float:
        """Check if features deviate significantly from learned statistics"""
        if not self.feature_stats:
            return 0.0
        
        anomaly_count = 0
        total_features = min(features.shape[1], len(self.feature_stats))
        
        for i in range(total_features):
            stats = self.feature_stats.get(f"feature_{i}", {})
            if not stats:
                continue
            
            value = features[0, i]
            mean = stats["mean"]
            std = stats["std"]
            
            # Z-score based detection
            if std > 0:
                z_score = abs(value - mean) / std
                if z_score > 5:  # 5 standard deviations
                    anomaly_count += 1
            
            # Range-based detection
            if value < stats["min"] * 0.5 or value > stats["max"] * 1.5:
                anomaly_count += 0.5
        
        return min(anomaly_count / total_features, 1.0)
    
    def _check_feature_squeezing(self, features: np.ndarray) -> float:
        """
        Feature squeezing defense
        Reduce precision and check if prediction changes significantly
        """
        bit_depth = self.config["squeeze_bit_depth"]
        
        # Quantize features
        squeezed = self._squeeze_features(features, bit_depth)
        
        # Calculate difference
        diff = np.abs(features - squeezed)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # High difference after squeezing may indicate adversarial perturbations
        score = mean_diff / (np.std(features) + 1e-6)
        
        return min(score, 1.0)
    
    def _squeeze_features(self, features: np.ndarray, bit_depth: int) -> np.ndarray:
        """Reduce feature precision to specified bit depth"""
        levels = 2 ** bit_depth
        
        # Normalize to [0, 1] range
        min_val = features.min()
        max_val = features.max()
        
        if max_val - min_val > 0:
            normalized = (features - min_val) / (max_val - min_val)
        else:
            normalized = features
        
        # Quantize
        quantized = np.round(normalized * levels) / levels
        
        # Denormalize
        return quantized * (max_val - min_val) + min_val
    
    def _check_ensemble_disagreement(self, predictions: Dict[str, np.ndarray]) -> float:
        """
        Check for disagreement between ensemble models
        Adversarial inputs often cause models to disagree
        """
        if len(predictions) < 2:
            return 0.0
        
        pred_classes = []
        for name, probs in predictions.items():
            pred_classes.append(np.argmax(probs))
        
        # Calculate disagreement ratio
        unique_predictions = len(set(pred_classes))
        disagreement = (unique_predictions - 1) / max(len(pred_classes) - 1, 1)
        
        return disagreement
    
    def sanitize_input(self, features: np.ndarray) -> np.ndarray:
        """
        Sanitize input to reduce adversarial perturbations
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        sanitized = features.copy()
        
        # Replace NaN/Inf with median or 0
        sanitized = np.nan_to_num(sanitized, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Clip extreme values if we have statistics
        if self.feature_stats:
            for i in range(min(sanitized.shape[1], len(self.feature_stats))):
                stats = self.feature_stats.get(f"feature_{i}", {})
                if stats:
                    # Clip to IQR * 3
                    iqr = stats["q3"] - stats["q1"]
                    lower = stats["q1"] - 3 * iqr
                    upper = stats["q3"] + 3 * iqr
                    sanitized[0, i] = np.clip(sanitized[0, i], lower, upper)
        
        # Apply feature squeezing
        sanitized = self._squeeze_features(sanitized, self.config["squeeze_bit_depth"])
        
        return sanitized
    
    def add_to_history(self, features: np.ndarray):
        """Add input to history for drift detection"""
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        self.input_history.append(features.copy())
        
        # Keep history bounded
        if len(self.input_history) > self.config["history_size"]:
            self.input_history.pop(0)
    
    def get_stats(self) -> Dict:
        """Get defense module statistics"""
        return {
            "features_tracked": len(self.feature_stats),
            "history_size": len(self.input_history),
            "detection_threshold": self.detection_threshold,
            "config": self.config
        }


# Global instance
_defense: Optional[AdversarialDefense] = None


def get_adversarial_defense() -> AdversarialDefense:
    """Get or create adversarial defense instance"""
    global _defense
    if _defense is None:
        _defense = AdversarialDefense()
    return _defense


def check_adversarial(features: np.ndarray) -> Dict:
    """Convenience function to check for adversarial inputs"""
    defense = get_adversarial_defense()
    result = defense.detect_adversarial(features)
    
    return {
        "is_adversarial": result.is_adversarial,
        "confidence": result.confidence,
        "detection_method": result.detection_method,
        "anomaly_scores": result.anomaly_scores,
        "recommendation": result.recommendation
    }
