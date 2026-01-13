"""
Explainable AI API Endpoints
Provides interpretable explanations for ML predictions
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

router = APIRouter(prefix="/xai", tags=["Explainable AI"])


class ExplainRequest(BaseModel):
    """Request to explain a prediction"""
    features: List[float]
    prediction: int
    confidence: float
    method: str = "shap"  # "shap" or "lime"


class FeatureContribution(BaseModel):
    """Feature contribution to prediction"""
    feature_name: str
    feature_value: float
    contribution: float
    direction: str


class ExplanationResponse(BaseModel):
    """Explanation response"""
    prediction_id: str
    predicted_class: int
    class_name: str
    confidence: float
    top_features: List[FeatureContribution]
    explanation_text: str
    method: str


@router.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplainRequest):
    """
    Generate explainable AI interpretation for a prediction
    
    Uses SHAP or LIME to explain why the model made a specific prediction.
    """
    from ml.explainable_ai import get_xai, ATTACK_CLASSES
    import uuid
    
    xai = get_xai()
    
    features = np.array(request.features).reshape(1, -1)
    
    if request.method == "lime" and xai.lime_explainer:
        exp = xai.explain_prediction_lime(features, request.prediction, request.confidence)
    elif xai.shap_explainer:
        exp = xai.explain_prediction_shap(features, request.prediction, request.confidence)
    else:
        # Fallback: generate basic explanation without explainers
        class_name = ATTACK_CLASSES.get(request.prediction, f"Class_{request.prediction}")
        
        return ExplanationResponse(
            prediction_id=str(uuid.uuid4())[:8],
            predicted_class=request.prediction,
            class_name=class_name,
            confidence=request.confidence,
            top_features=[],
            explanation_text=f"Classified as {class_name} with {request.confidence*100:.1f}% confidence.",
            method="basic"
        )
    
    return ExplanationResponse(
        prediction_id=exp.prediction_id,
        predicted_class=exp.predicted_class,
        class_name=exp.class_name,
        confidence=exp.confidence,
        top_features=[
            FeatureContribution(
                feature_name=f.feature_name,
                feature_value=f.feature_value,
                contribution=f.contribution,
                direction=f.direction
            ) for f in exp.top_features
        ],
        explanation_text=exp.explanation_text,
        method=exp.method
    )


@router.get("/feature-importance")
async def get_feature_importance(sample_size: int = 100):
    """
    Get global feature importance from the model
    
    Returns ranked list of features by their importance to predictions.
    """
    from ml.explainable_ai import get_xai
    
    xai = get_xai()
    
    # If we have background data, compute importance
    if xai.background_data is not None and xai.shap_explainer is not None:
        importance = xai.get_feature_importance(xai.background_data, sample_size)
        return {
            "status": "success",
            "feature_count": len(importance),
            "importance": importance
        }
    
    # Return default importance based on domain knowledge
    default_importance = {
        "Flow_Duration": 0.15,
        "Total_Fwd_Packets": 0.12,
        "Flow_Bytes_s": 0.11,
        "Destination_Port": 0.10,
        "Fwd_Packet_Length_Mean": 0.09,
        "SYN_Flag_Count": 0.08,
        "Flow_IAT_Mean": 0.07,
        "ACK_Flag_Count": 0.06,
        "Bwd_Packet_Length_Mean": 0.05,
        "PSH_Flag_Count": 0.04
    }
    
    return {
        "status": "default",
        "message": "Using default feature importance (model not loaded)",
        "feature_count": len(default_importance),
        "importance": default_importance
    }


@router.post("/initialize")
async def initialize_xai(model_path: str = "ml/models/ensemble_xgb.json"):
    """
    Initialize XAI with a trained model
    """
    from ml.explainable_ai import get_xai
    import xgboost as xgb
    from pathlib import Path
    
    xai = get_xai()
    
    model_file = Path(model_path)
    if not model_file.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    
    try:
        model = xgb.XGBClassifier()
        model.load_model(str(model_file))
        
        # Load some background data for explainers
        import pandas as pd
        data_path = Path("ml/datasets/cicids2017/CIC-IDS-2017-V2.csv")
        if data_path.exists():
            df = pd.read_csv(data_path, nrows=1000)
            background = df.select_dtypes(include=['number']).values
            xai.set_model(model, background)
        else:
            xai.set_model(model)
        
        return {
            "status": "success",
            "model": model_path,
            "shap_ready": xai.shap_explainer is not None,
            "lime_ready": xai.lime_explainer is not None
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_xai_status():
    """Get XAI module status"""
    from ml.explainable_ai import get_xai, HAS_SHAP, HAS_LIME
    
    xai = get_xai()
    
    return {
        "shap_installed": HAS_SHAP,
        "lime_installed": HAS_LIME,
        "shap_explainer_ready": xai.shap_explainer is not None,
        "lime_explainer_ready": xai.lime_explainer is not None,
        "model_loaded": xai.model is not None,
        "feature_count": len(xai.feature_names)
    }
