"""
Adversarial ML Defense API
Protects ML models against evasion attacks
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import numpy as np

router = APIRouter(prefix="/adversarial", tags=["Adversarial Defense"])


class CheckRequest(BaseModel):
    """Request to check for adversarial input"""
    features: List[float]
    predictions: Dict[str, List[float]] = None  # Optional ensemble predictions


class SanitizeRequest(BaseModel):
    """Request to sanitize input"""
    features: List[float]


class DefenseConfigUpdate(BaseModel):
    """Update defense configuration"""
    enable_input_validation: bool = None
    enable_statistical_check: bool = None
    enable_feature_squeezing: bool = None
    enable_ensemble_check: bool = None
    detection_threshold: float = None


@router.post("/check")
async def check_adversarial(request: CheckRequest):
    """
    Check if input appears to be adversarial
    
    Uses multiple detection techniques:
    - Input validation
    - Statistical anomaly detection
    - Feature squeezing
    - Ensemble disagreement
    """
    from ml.adversarial_defense import get_adversarial_defense
    
    defense = get_adversarial_defense()
    features = np.array(request.features).reshape(1, -1)
    
    # Convert predictions if provided
    predictions = None
    if request.predictions:
        predictions = {k: np.array(v) for k, v in request.predictions.items()}
    
    result = defense.detect_adversarial(features, predictions)
    
    return {
        "is_adversarial": result.is_adversarial,
        "confidence": result.confidence,
        "detection_method": result.detection_method,
        "anomaly_scores": result.anomaly_scores,
        "recommendation": result.recommendation
    }


@router.post("/sanitize")
async def sanitize_input(request: SanitizeRequest):
    """
    Sanitize input to remove potential adversarial perturbations
    
    Applies:
    - NaN/Inf replacement
    - Extreme value clipping
    - Feature squeezing
    """
    from ml.adversarial_defense import get_adversarial_defense
    
    defense = get_adversarial_defense()
    features = np.array(request.features).reshape(1, -1)
    
    sanitized = defense.sanitize_input(features)
    
    return {
        "original": request.features,
        "sanitized": sanitized[0].tolist(),
        "changes_made": int(np.sum(features != sanitized))
    }


@router.post("/learn-statistics")
async def learn_statistics(sample_size: int = 10000):
    """
    Learn feature statistics from training data
    
    This improves adversarial detection by understanding normal data distribution.
    """
    from ml.adversarial_defense import get_adversarial_defense
    import pandas as pd
    from pathlib import Path
    
    defense = get_adversarial_defense()
    
    data_path = Path("ml/datasets/cicids2017/CIC-IDS-2017-V2.csv")
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Training data not found")
    
    try:
        df = pd.read_csv(data_path, nrows=sample_size)
        X = df.select_dtypes(include=['number']).values
        
        defense.learn_feature_statistics(X)
        
        return {
            "status": "success",
            "samples_used": len(X),
            "features_learned": len(defense.feature_stats)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_defense_status():
    """Get adversarial defense module status"""
    from ml.adversarial_defense import get_adversarial_defense
    
    defense = get_adversarial_defense()
    return defense.get_stats()


@router.post("/configure")
async def configure_defense(config: DefenseConfigUpdate):
    """Update defense configuration"""
    from ml.adversarial_defense import get_adversarial_defense
    
    defense = get_adversarial_defense()
    
    if config.enable_input_validation is not None:
        defense.config["enable_input_validation"] = config.enable_input_validation
    if config.enable_statistical_check is not None:
        defense.config["enable_statistical_check"] = config.enable_statistical_check
    if config.enable_feature_squeezing is not None:
        defense.config["enable_feature_squeezing"] = config.enable_feature_squeezing
    if config.enable_ensemble_check is not None:
        defense.config["enable_ensemble_check"] = config.enable_ensemble_check
    if config.detection_threshold is not None:
        defense.detection_threshold = config.detection_threshold
    
    return {
        "status": "updated",
        "config": defense.config,
        "detection_threshold": defense.detection_threshold
    }
