"""
Explainable AI (XAI) Module for PCDS
Based on survey paper recommendations for transparent ML decisions

Features:
- SHAP explanations for feature importance
- LIME for local interpretability
- Feature attribution visualization
- Decision explanations for analysts
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json

# Try importing XAI libraries
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠️ SHAP not installed")

try:
    import lime
    import lime.lime_tabular
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("⚠️ LIME not installed")


@dataclass
class FeatureExplanation:
    """Explanation for a single feature's contribution"""
    feature_name: str
    feature_value: float
    contribution: float  # SHAP value or LIME weight
    direction: str  # "increases" or "decreases" risk


@dataclass
class PredictionExplanation:
    """Full explanation for a model prediction"""
    prediction_id: str
    predicted_class: int
    class_name: str
    confidence: float
    top_features: List[FeatureExplanation]
    explanation_text: str
    timestamp: str
    method: str  # "shap" or "lime"


# Feature names for CICIDS2017
FEATURE_NAMES = [
    "Destination_Port", "Flow_Duration", "Total_Fwd_Packets", "Total_Backward_Packets",
    "Total_Length_of_Fwd_Packets", "Total_Length_of_Bwd_Packets", "Fwd_Packet_Length_Max",
    "Fwd_Packet_Length_Min", "Fwd_Packet_Length_Mean", "Fwd_Packet_Length_Std",
    "Bwd_Packet_Length_Max", "Bwd_Packet_Length_Min", "Bwd_Packet_Length_Mean",
    "Bwd_Packet_Length_Std", "Flow_Bytes_s", "Flow_Packets_s", "Flow_IAT_Mean",
    "Flow_IAT_Std", "Flow_IAT_Max", "Flow_IAT_Min", "Fwd_IAT_Total", "Fwd_IAT_Mean",
    "Fwd_IAT_Std", "Fwd_IAT_Max", "Fwd_IAT_Min", "Bwd_IAT_Total", "Bwd_IAT_Mean",
    "Bwd_IAT_Std", "Bwd_IAT_Max", "Bwd_IAT_Min", "Fwd_PSH_Flags", "Bwd_PSH_Flags",
    "Fwd_URG_Flags", "Bwd_URG_Flags", "Fwd_Header_Length", "Bwd_Header_Length",
    "Fwd_Packets_s", "Bwd_Packets_s", "Min_Packet_Length", "Max_Packet_Length",
    "Packet_Length_Mean", "Packet_Length_Std", "Packet_Length_Variance", "FIN_Flag_Count",
    "SYN_Flag_Count", "RST_Flag_Count", "PSH_Flag_Count", "ACK_Flag_Count",
    "URG_Flag_Count", "CWE_Flag_Count", "ECE_Flag_Count", "Down_Up_Ratio",
    "Average_Packet_Size", "Avg_Fwd_Segment_Size", "Avg_Bwd_Segment_Size",
    "Fwd_Avg_Bytes_Bulk", "Fwd_Avg_Packets_Bulk", "Fwd_Avg_Bulk_Rate",
    "Bwd_Avg_Bytes_Bulk", "Bwd_Avg_Packets_Bulk", "Bwd_Avg_Bulk_Rate",
    "Subflow_Fwd_Packets", "Subflow_Fwd_Bytes", "Subflow_Bwd_Packets",
    "Subflow_Bwd_Bytes", "Init_Win_bytes_forward", "Init_Win_bytes_backward",
    "act_data_pkt_fwd", "min_seg_size_forward", "Active_Mean", "Active_Std",
    "Active_Max", "Active_Min", "Idle_Mean", "Idle_Std", "Idle_Max", "Idle_Min"
]

# Attack class mapping
ATTACK_CLASSES = {
    0: "Normal", 1: "Bot", 2: "DDoS", 3: "DoS GoldenEye", 4: "DoS Hulk",
    5: "DoS Slowhttptest", 6: "DoS Slowloris", 7: "FTP-Patator", 8: "Heartbleed",
    9: "Infiltration", 10: "PortScan", 11: "SSH-Patator", 
    12: "Web Attack - Brute Force", 13: "Web Attack - SQL Injection",
    14: "Web Attack - XSS"
}


class ExplainableAI:
    """
    Explainable AI module for PCDS
    
    Provides interpretable explanations for ML model predictions
    using SHAP and LIME techniques.
    """
    
    def __init__(self, model=None, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or FEATURE_NAMES[:78]
        self.shap_explainer = None
        self.lime_explainer = None
        self.background_data = None
        
        print("🔍 Explainable AI module initialized")
    
    def set_model(self, model, background_data: np.ndarray = None):
        """Set the model to explain and create explainers"""
        self.model = model
        self.background_data = background_data
        
        # Create SHAP explainer
        if HAS_SHAP and model is not None:
            try:
                if background_data is not None:
                    # Use TreeExplainer for tree-based models
                    self.shap_explainer = shap.TreeExplainer(model)
                    print("  ✅ SHAP TreeExplainer created")
                else:
                    # Fallback to KernelExplainer
                    self.shap_explainer = shap.Explainer(model)
                    print("  ✅ SHAP Explainer created")
            except Exception as e:
                print(f"  ⚠️ SHAP explainer failed: {e}")
        
        # Create LIME explainer
        if HAS_LIME and background_data is not None:
            try:
                self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    training_data=background_data[:1000],  # Sample for efficiency
                    feature_names=self.feature_names[:background_data.shape[1]],
                    class_names=list(ATTACK_CLASSES.values()),
                    mode='classification'
                )
                print("  ✅ LIME explainer created")
            except Exception as e:
                print(f"  ⚠️ LIME explainer failed: {e}")
    
    def explain_prediction_shap(self, features: np.ndarray, 
                                prediction: int,
                                confidence: float,
                                top_k: int = 10) -> PredictionExplanation:
        """
        Generate SHAP-based explanation for a prediction
        """
        import uuid
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        explanations = []
        
        if self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(features)
                
                # Handle multi-class output
                if isinstance(shap_values, list):
                    values = shap_values[prediction][0]
                else:
                    values = shap_values[0]
                
                # Get top contributing features
                indices = np.argsort(np.abs(values))[::-1][:top_k]
                
                for idx in indices:
                    feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
                    contribution = float(values[idx])
                    
                    explanations.append(FeatureExplanation(
                        feature_name=feat_name,
                        feature_value=float(features[0, idx]),
                        contribution=contribution,
                        direction="increases" if contribution > 0 else "decreases"
                    ))
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
        
        # Generate natural language explanation
        explanation_text = self._generate_explanation_text(
            prediction, confidence, explanations
        )
        
        return PredictionExplanation(
            prediction_id=str(uuid.uuid4())[:8],
            predicted_class=prediction,
            class_name=ATTACK_CLASSES.get(prediction, f"Class_{prediction}"),
            confidence=confidence,
            top_features=explanations,
            explanation_text=explanation_text,
            timestamp=datetime.utcnow().isoformat(),
            method="shap"
        )
    
    def explain_prediction_lime(self, features: np.ndarray,
                                prediction: int,
                                confidence: float,
                                top_k: int = 10) -> PredictionExplanation:
        """
        Generate LIME-based explanation for a prediction
        """
        import uuid
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        explanations = []
        
        if self.lime_explainer is not None and self.model is not None:
            try:
                exp = self.lime_explainer.explain_instance(
                    features[0],
                    self.model.predict_proba,
                    num_features=top_k,
                    labels=[prediction]
                )
                
                for feat_idx, weight in exp.as_list(label=prediction):
                    # Parse feature name from LIME output
                    feat_name = feat_idx.split()[0] if isinstance(feat_idx, str) else f"Feature_{feat_idx}"
                    
                    explanations.append(FeatureExplanation(
                        feature_name=feat_name,
                        feature_value=0.0,  # LIME doesn't provide this directly
                        contribution=float(weight),
                        direction="increases" if weight > 0 else "decreases"
                    ))
            except Exception as e:
                print(f"LIME explanation failed: {e}")
        
        explanation_text = self._generate_explanation_text(
            prediction, confidence, explanations
        )
        
        return PredictionExplanation(
            prediction_id=str(uuid.uuid4())[:8],
            predicted_class=prediction,
            class_name=ATTACK_CLASSES.get(prediction, f"Class_{prediction}"),
            confidence=confidence,
            top_features=explanations,
            explanation_text=explanation_text,
            timestamp=datetime.utcnow().isoformat(),
            method="lime"
        )
    
    def _generate_explanation_text(self, prediction: int, confidence: float,
                                   features: List[FeatureExplanation]) -> str:
        """Generate human-readable explanation"""
        class_name = ATTACK_CLASSES.get(prediction, f"Class_{prediction}")
        
        if prediction == 0:
            base = f"This traffic is classified as **Normal** with {confidence*100:.1f}% confidence."
        else:
            base = f"⚠️ This traffic is classified as **{class_name}** attack with {confidence*100:.1f}% confidence."
        
        if not features:
            return base
        
        # Top contributing features
        top_3 = features[:3]
        factors = []
        for f in top_3:
            direction = "high" if f.direction == "increases" else "low"
            factors.append(f"{direction} {f.feature_name}")
        
        explanation = f"{base}\n\n**Key factors:** {', '.join(factors)}"
        
        return explanation
    
    def get_feature_importance(self, X: np.ndarray, 
                               sample_size: int = 100) -> Dict[str, float]:
        """
        Get global feature importance using SHAP
        """
        if self.shap_explainer is None:
            return {}
        
        try:
            # Sample for efficiency
            if len(X) > sample_size:
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            shap_values = self.shap_explainer.shap_values(X_sample)
            
            # Average absolute SHAP values
            if isinstance(shap_values, list):
                mean_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                mean_importance = np.abs(shap_values).mean(axis=0)
            
            # Create importance dict
            importance = {}
            for i, imp in enumerate(mean_importance):
                feat_name = self.feature_names[i] if i < len(self.feature_names) else f"Feature_{i}"
                importance[feat_name] = float(imp)
            
            # Sort by importance
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        
        except Exception as e:
            print(f"Feature importance failed: {e}")
            return {}
    
    def to_dict(self, explanation: PredictionExplanation) -> Dict:
        """Convert explanation to dictionary for API response"""
        result = asdict(explanation)
        result['top_features'] = [asdict(f) for f in explanation.top_features]
        return result


# Global instance
_xai: Optional[ExplainableAI] = None


def get_xai() -> ExplainableAI:
    """Get or create XAI instance"""
    global _xai
    if _xai is None:
        _xai = ExplainableAI()
    return _xai


def explain_prediction(features: np.ndarray, prediction: int, 
                       confidence: float, method: str = "shap") -> Dict:
    """
    Convenience function to explain a prediction
    """
    xai = get_xai()
    
    if method == "lime":
        exp = xai.explain_prediction_lime(features, prediction, confidence)
    else:
        exp = xai.explain_prediction_shap(features, prediction, confidence)
    
    return xai.to_dict(exp)
