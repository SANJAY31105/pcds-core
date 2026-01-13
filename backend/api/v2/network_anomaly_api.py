"""
Network Anomaly Detection API
SVM, Change Point Detection, Tukey Outliers, and t-SNE Visualization
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np

router = APIRouter(prefix="/network-anomaly", tags=["Network Anomaly Detection"])


class NetworkMetrics(BaseModel):
    """Network performance metrics"""
    throughput: float = 0.0
    latency: float = 0.0
    jitter: float = 0.0
    packet_loss: float = 0.0
    congestion: float = 0.0


class AnalyzeRequest(BaseModel):
    """Request to analyze network metrics"""
    metrics: NetworkMetrics


class TrainSVMRequest(BaseModel):
    """Request to train SVM classifier"""
    features: List[List[float]]
    labels: List[int]


class TSNERequest(BaseModel):
    """Request for t-SNE visualization"""
    features: List[List[float]]
    labels: Optional[List[int]] = None
    feature_names: Optional[List[str]] = None
    perplexity: int = 30


@router.post("/analyze")
async def analyze_metrics(request: AnalyzeRequest):
    """
    Analyze network metrics for anomalies
    
    Uses multiple techniques:
    - SVM classification (if trained)
    - Change Point Detection
    - Tukey IQR outlier detection
    """
    from ml.network_anomaly import get_network_anomaly_detector
    from dataclasses import asdict
    
    detector = get_network_anomaly_detector()
    metrics = request.metrics.dict()
    
    result = detector.analyze(metrics)
    
    # Convert to JSON-serializable format
    response = {
        "anomaly_id": result.anomaly_id,
        "timestamp": result.timestamp,
        "is_anomaly": result.is_anomaly,
        "confidence": result.confidence,
        "detection_methods": result.detection_methods,
        "metrics": result.metrics,
        "svm_prediction": result.svm_prediction,
        "change_points": [
            {
                "metric": cp.metric,
                "index": cp.index,
                "before_mean": cp.before_mean,
                "after_mean": cp.after_mean,
                "change_magnitude": cp.change_magnitude,
                "significance": cp.significance
            } for cp in result.change_points
        ],
        "outliers": [
            {
                "metric": o.metric,
                "value": o.value,
                "is_outlier": o.is_outlier,
                "lower_bound": o.lower_bound,
                "upper_bound": o.upper_bound
            } for o in result.outliers
        ]
    }
    
    return response


@router.post("/train-svm")
async def train_svm(request: TrainSVMRequest):
    """
    Train SVM classifier on labeled network data
    
    Paper achieved 96.5% accuracy with SVM
    """
    from ml.network_anomaly import get_network_anomaly_detector
    
    detector = get_network_anomaly_detector()
    
    X = np.array(request.features)
    y = np.array(request.labels)
    
    if len(X) < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 samples for training")
    
    accuracy = detector.train_svm(X, y)
    
    return {
        "status": "trained",
        "accuracy": accuracy,
        "samples_used": len(X)
    }


@router.post("/learn-baseline")
async def learn_baseline(sample_size: int = 1000):
    """
    Learn baseline statistics from CICIDS2017 dataset
    """
    from ml.network_anomaly import get_network_anomaly_detector
    import pandas as pd
    from pathlib import Path
    
    detector = get_network_anomaly_detector()
    
    data_path = Path("ml/datasets/cicids2017/CIC-IDS-2017-V2.csv")
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        df = pd.read_csv(data_path, nrows=sample_size)
        
        # Map columns to network metrics
        metric_mapping = {
            "throughput": "Flow_Bytes_s",
            "latency": "Flow_Duration",
            "jitter": "Flow_IAT_Std",
            "packet_loss": "Packet_Loss_Rate" if "Packet_Loss_Rate" in df.columns else None,
            "congestion": "Bwd_Packets_s"
        }
        
        data = {}
        for metric, col in metric_mapping.items():
            if col and col in df.columns:
                values = df[col].replace([np.inf, -np.inf], np.nan).dropna().values
                if len(values) > 0:
                    data[metric] = values[:sample_size]
        
        detector.learn_baseline(data)
        
        return {
            "status": "success",
            "metrics_learned": list(data.keys()),
            "samples_used": sample_size
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_detector_stats():
    """Get network anomaly detector statistics"""
    from ml.network_anomaly import get_network_anomaly_detector
    
    detector = get_network_anomaly_detector()
    return detector.get_stats()


@router.post("/tsne")
async def generate_tsne(request: TSNERequest):
    """
    Generate t-SNE visualization for network data
    
    Reduces high-dimensional data to 2D for visualization
    """
    from ml.tsne_visualizer import get_tsne_visualizer
    
    visualizer = get_tsne_visualizer()
    
    X = np.array(request.features)
    labels = np.array(request.labels) if request.labels else None
    
    if len(X) < 5:
        raise HTTPException(status_code=400, detail="Need at least 5 samples for t-SNE")
    
    # Adjust perplexity
    perplexity = min(request.perplexity, len(X) - 1)
    visualizer.perplexity = perplexity
    
    result = visualizer.fit_transform(
        X, 
        labels=labels,
        feature_names=request.feature_names
    )
    
    return {
        "status": "success",
        "points": result.points,
        "perplexity": result.perplexity,
        "n_iter": result.n_iter,
        "point_count": len(result.points)
    }


@router.post("/tsne/from-dataset")
async def tsne_from_dataset(sample_size: int = 500, perplexity: int = 30):
    """
    Generate t-SNE visualization from CICIDS2017 dataset
    """
    from ml.tsne_visualizer import get_tsne_visualizer
    import pandas as pd
    from pathlib import Path
    from sklearn.preprocessing import LabelEncoder
    
    data_path = Path("ml/datasets/cicids2017/CIC-IDS-2017-V2.csv")
    if not data_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        df = pd.read_csv(data_path, nrows=sample_size)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        
        # Get features and labels
        X = df.select_dtypes(include=['number']).values
        
        # Get labels
        label_col = None
        for col in ['label', 'Label', ' Label']:
            if col in df.columns:
                label_col = col
                break
        
        if label_col:
            le = LabelEncoder()
            labels = le.fit_transform(df[label_col].values)
        else:
            labels = np.zeros(len(X))
        
        # Run t-SNE
        visualizer = get_tsne_visualizer()
        visualizer.perplexity = min(perplexity, len(X) - 1)
        
        feature_names = list(df.select_dtypes(include=['number']).columns[:10])
        
        result = visualizer.fit_transform(
            X[:, :10],  # Use first 10 features for speed
            labels=labels,
            feature_names=feature_names
        )
        
        return {
            "status": "success",
            "points": result.points,
            "perplexity": result.perplexity,
            "point_count": len(result.points),
            "class_distribution": {
                str(k): int(v) for k, v in 
                zip(*np.unique(labels, return_counts=True))
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/change-points")
async def detect_change_points(data: List[float], metric_name: str = "metric"):
    """
    Detect change points in time series data
    """
    from ml.network_anomaly import ChangePointDetector
    
    detector = ChangePointDetector()
    data_arr = np.array(data)
    
    if len(data_arr) < 20:
        raise HTTPException(status_code=400, detail="Need at least 20 data points")
    
    change_points = detector.detect(data_arr)
    
    results = []
    for cp_idx in change_points:
        cp = detector.analyze_changepoint(data_arr, cp_idx, metric_name)
        results.append({
            "index": cp.index,
            "metric": cp.metric,
            "before_mean": cp.before_mean,
            "after_mean": cp.after_mean,
            "change_magnitude": cp.change_magnitude,
            "significance": cp.significance
        })
    
    return {
        "change_points": results,
        "total_found": len(results)
    }


@router.post("/tukey-outliers")
async def detect_tukey_outliers(data: List[float], metric_name: str = "metric", k: float = 1.5):
    """
    Detect outliers using Tukey's IQR method
    """
    from ml.network_anomaly import TukeyOutlierDetector
    
    detector = TukeyOutlierDetector(k=k)
    data_arr = np.array(data)
    
    # Learn statistics
    stats = detector.learn_statistics(data_arr, metric_name)
    
    # Detect outliers
    outliers = detector.detect_batch(data_arr, metric_name)
    
    outlier_indices = [i for i, o in enumerate(outliers) if o.is_outlier]
    
    return {
        "statistics": stats,
        "outlier_count": len(outlier_indices),
        "outlier_indices": outlier_indices,
        "outlier_values": [data[i] for i in outlier_indices],
        "total_samples": len(data)
    }
