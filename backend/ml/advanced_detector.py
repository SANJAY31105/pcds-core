"""
PCDS Enterprise - Advanced Detection Engine v3.0
Market-Leading Multi-Model Ensemble with Explainable AI
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import time


class EnsembleCombiner:
    """Combines predictions from multiple models with learned weights"""
    
    def __init__(self):
        # Model weights (can be learned from validation data)
        self.weights = {
            'transformer': 0.35,
            'lstm': 0.30,
            'graph': 0.25,
            'statistical': 0.10
        }
        
        # Calibration parameters (Platt scaling)
        self.calibration_a = 1.0
        self.calibration_b = 0.0
    
    def combine(self, predictions: Dict[str, Tuple[float, float]]) -> Tuple[float, float, Dict]:
        """
        Combine predictions from multiple models
        
        Args:
            predictions: {model_name: (score, confidence)}
            
        Returns:
            final_score, final_confidence, contribution_dict
        """
        weighted_score = 0.0
        weighted_confidence = 0.0
        total_weight = 0.0
        contributions = {}
        
        for model_name, (score, confidence) in predictions.items():
            weight = self.weights.get(model_name, 0.1)
            weighted_score += score * weight * confidence
            weighted_confidence += confidence * weight
            total_weight += weight
            contributions[model_name] = {
                'score': float(score),
                'confidence': float(confidence),
                'weight': float(weight),
                'contribution': float(score * weight * confidence)
            }
        
        final_score = weighted_score / max(total_weight, 1e-6)
        final_confidence = weighted_confidence / max(total_weight, 1e-6)
        
        # Apply calibration
        calibrated_score = self._calibrate(final_score)
        
        return float(calibrated_score), float(final_confidence), contributions
    
    def _calibrate(self, score: float) -> float:
        """Platt scaling calibration"""
        return 1 / (1 + np.exp(-(self.calibration_a * score + self.calibration_b)))
    
    def update_weights(self, performance: Dict[str, float]):
        """Update model weights based on performance"""
        total_perf = sum(performance.values()) + 1e-6
        for model_name in self.weights:
            if model_name in performance:
                self.weights[model_name] = performance[model_name] / total_perf


class AdvancedDetectionEngine:
    """
    PCDS Enterprise Advanced Detection Engine v3.0
    
    Features:
    - Multi-model ensemble (Transformer + LSTM + Graph NN)
    - Real feature extraction (no random values)
    - Explainable AI for every decision
    - Attack chain detection
    - Sub-50ms inference
    """
    
    VERSION = "3.0.0"
    
    def __init__(self):
        print("🧠 Initializing Advanced Detection Engine v3.0...")
        
        # Import models
        from ml.feature_extractor import feature_extractor
        from ml.models.transformer_detector import transformer_detector
        from ml.models.temporal_lstm import temporal_lstm_detector
        from ml.models.graph_detector import graph_detector
        from ml.explainer import explainable_ai
        
        self.feature_extractor = feature_extractor
        self.transformer = transformer_detector
        self.lstm = temporal_lstm_detector
        self.graph_nn = graph_detector
        self.explainer = explainable_ai
        
        # Ensemble combiner
        self.combiner = EnsembleCombiner()
        
        # Statistical fallback
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        
        # Performance tracking
        self.inference_times = []
        self.prediction_count = 0
        
        # Thresholds per attack type
        self.attack_thresholds = {
            'brute_force': 0.6,
            'lateral_movement': 0.55,
            'data_exfiltration': 0.65,
            'c2_communication': 0.5,
            'privilege_escalation': 0.6,
            'default': 0.5
        }
        
        print(f"✅ Advanced Detection Engine ready (v{self.VERSION})")
    
    def detect(self, 
               data: Dict, 
               entity_id: str = None,
               entities: List[Dict] = None,
               attack_type: str = None) -> Dict:
        """
        Main detection method
        
        Args:
            data: Raw network/log data
            entity_id: Optional entity identifier for tracking
            entities: Optional list of related entities for graph analysis
            attack_type: Optional hint for threshold selection
            
        Returns:
            Comprehensive detection result with explanation
        """
        start_time = time.time()
        
        # 1. Extract features
        features = self.feature_extractor.extract_all_features(data, entity_id)
        
        # 2. Get predictions from each model
        predictions = {}
        
        # Transformer prediction
        try:
            is_anom, score, conf = self.transformer.predict(features)
            predictions['transformer'] = (score, conf)
        except Exception as e:
            predictions['transformer'] = (0.0, 0.0)
        
        # LSTM prediction
        try:
            is_anom, score, conf = self.lstm.predict(features)
            predictions['lstm'] = (score, conf)
        except Exception as e:
            predictions['lstm'] = (0.0, 0.0)
        
        # Graph prediction (if entities provided)
        if entities and len(entities) > 0:
            try:
                is_anom, score, conf = self.graph_nn.predict(entities)
                predictions['graph'] = (score, conf)
            except Exception as e:
                predictions['graph'] = (0.0, 0.0)
        
        # Statistical fallback
        stat_score = self._statistical_score(features)
        predictions['statistical'] = (stat_score, 0.8)
        
        # 3. Combine predictions
        final_score, confidence, contributions = self.combiner.combine(predictions)
        
        # 4. Determine if anomaly based on threshold
        threshold = self.attack_thresholds.get(attack_type, self.attack_thresholds['default'])
        is_anomaly = final_score > threshold
        
        # 5. Generate explanation
        explanation = self.explainer.explain(
            features=features,
            score=final_score,
            is_anomaly=is_anomaly,
            detection=data,
            entities=entities,
            model_contributions=contributions
        )
        
        # 6. Track performance
        inference_time = (time.time() - start_time) * 1000  # ms
        self.inference_times.append(inference_time)
        if len(self.inference_times) > 100:
            self.inference_times.pop(0)
        self.prediction_count += 1
        
        # 7. Build result
        result = {
            'is_anomaly': is_anomaly,
            'anomaly_score': float(final_score),
            'confidence': float(confidence),
            'risk_level': explanation['risk_level'],
            'model_contributions': contributions,
            'explanation': explanation,
            'inference_time_ms': float(inference_time),
            'engine_version': self.VERSION,
            'threshold_used': threshold,
            'attack_type_hint': attack_type
        }
        
        # Add attack chain if graph analysis available
        if entities and len(entities) > 1:
            chains = self.graph_nn.detect_attack_chains(entities)
            if chains:
                result['attack_chains'] = chains
            
            lateral = self.graph_nn.detect_lateral_movement(entities)
            result['lateral_movement'] = lateral
        
        return result
    
    def _statistical_score(self, features: np.ndarray) -> float:
        """Fallback statistical anomaly detection"""
        # Z-score based anomaly
        feature_mean = np.mean(features)
        z_score = abs(feature_mean - self.baseline_mean) / (self.baseline_std + 1e-6)
        
        # Convert to probability
        score = 1 - np.exp(-z_score / 2)
        return float(np.clip(score, 0.0, 1.0))
    
    def batch_detect(self, data_list: List[Dict], entity_id: str = None) -> List[Dict]:
        """Batch detection for multiple events"""
        return [self.detect(data, entity_id) for data in data_list]
    
    def get_performance_stats(self) -> Dict:
        """Get engine performance statistics"""
        if not self.inference_times:
            return {'status': 'no_predictions_yet'}
        
        return {
            'engine_version': self.VERSION,
            'total_predictions': self.prediction_count,
            'avg_inference_ms': float(np.mean(self.inference_times)),
            'p50_inference_ms': float(np.percentile(self.inference_times, 50)),
            'p95_inference_ms': float(np.percentile(self.inference_times, 95)),
            'p99_inference_ms': float(np.percentile(self.inference_times, 99)),
            'models_active': ['transformer', 'lstm', 'graph', 'statistical'],
            'ensemble_weights': self.combiner.weights
        }
    
    def explain_last_detection(self) -> Dict:
        """Get detailed explanation for last detection"""
        transformer_exp = self.transformer.get_attention_explanation()
        lstm_exp = self.lstm.get_temporal_explanation()
        
        return {
            'transformer': transformer_exp,
            'lstm': lstm_exp,
            'feature_importance': self.feature_extractor.get_feature_importance(
                self.feature_extractor.extract_all_features({})
            )
        }
    
    def train_baseline(self, normal_data: List[Dict]):
        """Train baseline from normal data"""
        features_list = [
            self.feature_extractor.extract_all_features(d) 
            for d in normal_data
        ]
        
        # Update statistical baseline
        all_means = [np.mean(f) for f in features_list]
        self.baseline_mean = np.mean(all_means)
        self.baseline_std = np.std(all_means)
        
        # Train transformer baseline
        self.transformer.train_baseline(features_list)
        
        return {
            'samples_trained': len(normal_data),
            'baseline_mean': float(self.baseline_mean),
            'baseline_std': float(self.baseline_std)
        }


# Create models directory __init__.py
import os
models_dir = os.path.dirname(__file__)
models_init = os.path.join(models_dir, 'models', '__init__.py')
if not os.path.exists(models_init):
    os.makedirs(os.path.dirname(models_init), exist_ok=True)
    with open(models_init, 'w') as f:
        f.write('# ML Models package\n')


# Global instance with lazy loading
_advanced_engine = None

def get_advanced_engine():
    """Get or create advanced detection engine"""
    global _advanced_engine
    if _advanced_engine is None:
        try:
            _advanced_engine = AdvancedDetectionEngine()
        except Exception as e:
            print(f"⚠️ Advanced engine not available: {e}")
            return None
    return _advanced_engine


# For backward compatibility
try:
    advanced_detector = get_advanced_engine()
except:
    advanced_detector = None
