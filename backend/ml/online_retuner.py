"""
Online Retuning System
Phase 3: Continuous model improvement with analyst feedback

Features:
- Analyst feedback integration (FP/FN corrections)
- False positive suppression
- Confidence threshold tuning
- Incremental model updates
- Model performance tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple


class OnlineRetuner:
    """
    Online model retuning with analyst feedback
    
    Capabilities:
    - Track analyst corrections
    - Adjust confidence thresholds
    - Fine-tune model with corrections
    - Suppress false positives
    - Track model drift
    """
    
    def __init__(self, model_path: str = None, 
                 feedback_buffer_size: int = 1000,
                 retune_interval: int = 100):
        """
        Args:
            model_path: Path to pre-trained model
            feedback_buffer_size: Number of feedback samples to buffer
            retune_interval: How often to retune (number of feedbacks)
        """
        self._lock = Lock()
        
        # Model
        self.model = None
        self.model_path = model_path
        
        # Feedback buffer
        self.feedback_buffer = deque(maxlen=feedback_buffer_size)
        self.fp_buffer = deque(maxlen=500)  # False positives
        self.fn_buffer = deque(maxlen=500)  # False negatives
        
        # Retune settings
        self.retune_interval = retune_interval
        self.feedback_count = 0
        
        # Confidence thresholds (per class)
        self.confidence_thresholds: Dict[int, float] = {}
        self.default_threshold = 0.5
        
        # Suppression rules
        self.suppression_rules: List[Dict] = []
        
        # Performance tracking
        self.performance_history: List[Dict] = []
        self.current_metrics = {
            "total_predictions": 0,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "analyst_corrections": 0
        }
        
        # Load model if provided
        if model_path:
            self._load_model(model_path)
        
        print("🔄 Online Retuner initialized")
    
    def _load_model(self, path: str):
        """Load pre-trained model"""
        try:
            checkpoint = torch.load(path, map_location='cpu')
            self.model = checkpoint.get('model_state_dict')
            print(f"   Loaded model from {path}")
        except Exception as e:
            print(f"   ⚠️ Could not load model: {e}")
    
    def add_prediction(self, prediction: Dict):
        """
        Track a prediction made by the model
        
        Args:
            prediction: dict with keys:
                - prediction_id
                - features
                - predicted_class
                - confidence
                - timestamp
        """
        with self._lock:
            self.current_metrics["total_predictions"] += 1
            
            # Apply confidence threshold
            class_id = prediction.get("predicted_class", 0)
            threshold = self.confidence_thresholds.get(class_id, self.default_threshold)
            
            if prediction.get("confidence", 0) < threshold:
                prediction["suppressed"] = True
            
            # Check suppression rules
            for rule in self.suppression_rules:
                if self._matches_suppression_rule(prediction, rule):
                    prediction["suppressed"] = True
                    prediction["suppression_reason"] = rule.get("reason", "rule match")
    
    def add_feedback(self, prediction_id: str, 
                     correct_class: int,
                     feedback_type: str,
                     analyst_notes: str = None):
        """
        Add analyst feedback for a prediction
        
        Args:
            prediction_id: ID of the prediction being corrected
            correct_class: The actual correct class
            feedback_type: 'true_positive', 'false_positive', 'false_negative'
            analyst_notes: Optional notes from analyst
        """
        feedback = {
            "prediction_id": prediction_id,
            "correct_class": correct_class,
            "feedback_type": feedback_type,
            "analyst_notes": analyst_notes,
            "timestamp": datetime.now().isoformat()
        }
        
        with self._lock:
            self.feedback_buffer.append(feedback)
            self.current_metrics["analyst_corrections"] += 1
            
            if feedback_type == "true_positive":
                self.current_metrics["true_positives"] += 1
            elif feedback_type == "false_positive":
                self.current_metrics["false_positives"] += 1
                self.fp_buffer.append(feedback)
            elif feedback_type == "false_negative":
                self.current_metrics["false_negatives"] += 1
                self.fn_buffer.append(feedback)
            
            self.feedback_count += 1
            
            # Check if retune needed
            if self.feedback_count >= self.retune_interval:
                self._trigger_retune()
    
    def _trigger_retune(self):
        """Trigger model retuning"""
        print("🔄 Triggering model retune...")
        
        # Adjust confidence thresholds based on FP/FN rates
        self._adjust_thresholds()
        
        # Generate suppression rules from FP patterns
        self._generate_suppression_rules()
        
        # Fine-tune model if we have enough samples
        if len(self.feedback_buffer) >= 50:
            self._fine_tune_model()
        
        # Record performance
        self._record_performance()
        
        self.feedback_count = 0
        print("🔄 Retune complete")
    
    def _adjust_thresholds(self):
        """Adjust confidence thresholds based on FP/FN rates"""
        with self._lock:
            # Calculate FP/FN rates by class
            fp_by_class: Dict[int, int] = {}
            fn_by_class: Dict[int, int] = {}
            
            for fp in self.fp_buffer:
                cls = fp.get("predicted_class", 0)
                fp_by_class[cls] = fp_by_class.get(cls, 0) + 1
            
            for fn in self.fn_buffer:
                cls = fn.get("correct_class", 0)
                fn_by_class[cls] = fn_by_class.get(cls, 0) + 1
            
            # Adjust thresholds
            for cls in set(list(fp_by_class.keys()) + list(fn_by_class.keys())):
                fp_rate = fp_by_class.get(cls, 0) / max(len(self.fp_buffer), 1)
                fn_rate = fn_by_class.get(cls, 0) / max(len(self.fn_buffer), 1)
                
                current = self.confidence_thresholds.get(cls, self.default_threshold)
                
                # More FPs -> raise threshold
                # More FNs -> lower threshold
                adjustment = (fp_rate - fn_rate) * 0.1
                new_threshold = max(0.1, min(0.9, current + adjustment))
                
                self.confidence_thresholds[cls] = new_threshold
                
                if abs(adjustment) > 0.01:
                    print(f"   Class {cls}: threshold {current:.2f} -> {new_threshold:.2f}")
    
    def _generate_suppression_rules(self):
        """Generate suppression rules from FP patterns"""
        with self._lock:
            # Analyze FP patterns
            if len(self.fp_buffer) < 10:
                return
            
            # Find common patterns in FPs
            # (This is simplified - real implementation would use clustering)
            fp_classes = [fp.get("predicted_class", 0) for fp in self.fp_buffer]
            
            from collections import Counter
            class_counts = Counter(fp_classes)
            
            for cls, count in class_counts.most_common(3):
                if count >= 5:  # At least 5 FPs from this class
                    rule = {
                        "type": "high_fp_class",
                        "class": cls,
                        "reason": f"High FP rate for class {cls}",
                        "confidence_boost": -0.2  # Lower confidence for this class
                    }
                    
                    # Add if not already present
                    if not any(r["class"] == cls for r in self.suppression_rules if r.get("type") == "high_fp_class"):
                        self.suppression_rules.append(rule)
                        print(f"   Added suppression rule for class {cls}")
    
    def _matches_suppression_rule(self, prediction: Dict, rule: Dict) -> bool:
        """Check if prediction matches a suppression rule"""
        if rule.get("type") == "high_fp_class":
            return prediction.get("predicted_class") == rule.get("class")
        return False
    
    def _fine_tune_model(self):
        """Fine-tune model with recent feedback (placeholder)"""
        # In real implementation, this would:
        # 1. Convert feedback to training samples
        # 2. Fine-tune model with low learning rate
        # 3. Validate on held-out set
        # 4. Update model if improved
        
        print("   Fine-tuning model with feedback samples...")
        print(f"   Samples available: {len(self.feedback_buffer)}")
    
    def _record_performance(self):
        """Record current performance metrics"""
        with self._lock:
            total = self.current_metrics["total_predictions"]
            if total == 0:
                return
            
            tp = self.current_metrics["true_positives"]
            fp = self.current_metrics["false_positives"]
            fn = self.current_metrics["false_negatives"]
            
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 0.001)
            
            record = {
                "timestamp": datetime.now().isoformat(),
                "total_predictions": total,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "thresholds": dict(self.confidence_thresholds)
            }
            
            self.performance_history.append(record)
            print(f"   Performance: P={precision:.2f}, R={recall:.2f}, F1={f1:.2f}")
    
    def get_threshold(self, class_id: int) -> float:
        """Get confidence threshold for a class"""
        return self.confidence_thresholds.get(class_id, self.default_threshold)
    
    def set_threshold(self, class_id: int, threshold: float):
        """Manually set threshold for a class"""
        with self._lock:
            self.confidence_thresholds[class_id] = max(0.1, min(0.99, threshold))
    
    def add_suppression_rule(self, rule: Dict):
        """Manually add a suppression rule"""
        with self._lock:
            self.suppression_rules.append(rule)
    
    def get_stats(self) -> Dict:
        """Get retuner statistics"""
        with self._lock:
            return {
                "feedback_count": len(self.feedback_buffer),
                "fp_count": len(self.fp_buffer),
                "fn_count": len(self.fn_buffer),
                "suppression_rules": len(self.suppression_rules),
                "class_thresholds": dict(self.confidence_thresholds),
                "metrics": self.current_metrics,
                "performance_records": len(self.performance_history)
            }
    
    def export_config(self, path: str):
        """Export retuner configuration"""
        config = {
            "confidence_thresholds": dict(self.confidence_thresholds),
            "suppression_rules": self.suppression_rules,
            "performance_history": self.performance_history[-10:],
            "exported_at": datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"   Exported config to {path}")
    
    def import_config(self, path: str):
        """Import retuner configuration"""
        with open(path, 'r') as f:
            config = json.load(f)
        
        with self._lock:
            self.confidence_thresholds = config.get("confidence_thresholds", {})
            self.confidence_thresholds = {int(k): v for k, v in self.confidence_thresholds.items()}
            self.suppression_rules = config.get("suppression_rules", [])
        
        print(f"   Imported config from {path}")


# Singleton
_online_retuner = None

def get_online_retuner() -> OnlineRetuner:
    global _online_retuner
    if _online_retuner is None:
        _online_retuner = OnlineRetuner()
    return _online_retuner


if __name__ == "__main__":
    retuner = OnlineRetuner()
    
    print("\n🔄 Online Retuner Test\n")
    
    # Simulate predictions and feedback
    for i in range(20):
        retuner.add_prediction({
            "prediction_id": f"pred_{i}",
            "predicted_class": i % 5,
            "confidence": 0.6 + (i % 4) * 0.1
        })
        
        if i % 3 == 0:
            retuner.add_feedback(
                prediction_id=f"pred_{i}",
                correct_class=i % 5 if i % 2 == 0 else (i + 1) % 5,
                feedback_type="false_positive" if i % 4 == 0 else "true_positive"
            )
    
    print(f"\n📊 Stats: {retuner.get_stats()}")
