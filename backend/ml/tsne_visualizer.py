"""
t-SNE Visualization Module for Network Data
Based on Paper 3: Used for visualizing high-dimensional network data
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

try:
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False
    print("⚠️ t-SNE not available")


@dataclass
class TSNEResult:
    """t-SNE visualization result"""
    points: List[Dict]  # [{x, y, label, metrics...}]
    perplexity: int
    n_iter: int
    feature_names: List[str]


class TSNEVisualizer:
    """
    t-SNE Visualization for Network Anomaly Data
    
    Reduces high-dimensional network metrics to 2D for visualization
    """
    
    def __init__(self, perplexity: int = 30, n_iter: int = 1000, random_state: int = 42):
        self.perplexity = perplexity
        self.n_iter = n_iter
        self.random_state = random_state
        self.scaler = StandardScaler() if HAS_TSNE else None
        self.last_result: Optional[TSNEResult] = None
        
    def fit_transform(self, X: np.ndarray, labels: np.ndarray = None,
                     feature_names: List[str] = None,
                     metadata: List[Dict] = None) -> TSNEResult:
        """
        Apply t-SNE to reduce dimensions and prepare for visualization
        
        Args:
            X: Feature matrix (n_samples, n_features)
            labels: Optional labels for coloring points
            feature_names: Names of features
            metadata: Additional metadata per point
        """
        if not HAS_TSNE:
            return TSNEResult(points=[], perplexity=self.perplexity, 
                            n_iter=self.n_iter, feature_names=[])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Adjust perplexity if needed
        perplexity = min(self.perplexity, len(X) - 1)
        
        # Apply t-SNE
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=self.n_iter,
            random_state=self.random_state,
            init='pca'
        )
        
        embedding = tsne.fit_transform(X_scaled)
        
        # Prepare points for visualization
        points = []
        for i, (x, y) in enumerate(embedding):
            point = {
                "x": float(x),
                "y": float(y),
                "index": i
            }
            
            if labels is not None:
                point["label"] = int(labels[i]) if hasattr(labels[i], 'item') else labels[i]
            
            if metadata is not None and i < len(metadata):
                point.update(metadata[i])
            
            # Add original feature values
            if feature_names:
                for j, name in enumerate(feature_names[:X.shape[1]]):
                    point[name] = float(X[i, j])
            
            points.append(point)
        
        result = TSNEResult(
            points=points,
            perplexity=perplexity,
            n_iter=self.n_iter,
            feature_names=feature_names or []
        )
        
        self.last_result = result
        return result
    
    def to_json(self, result: TSNEResult = None) -> str:
        """Convert result to JSON for frontend visualization"""
        if result is None:
            result = self.last_result
        
        if result is None:
            return json.dumps({"error": "No t-SNE result available"})
        
        return json.dumps({
            "points": result.points,
            "perplexity": result.perplexity,
            "n_iter": result.n_iter,
            "feature_names": result.feature_names,
            "point_count": len(result.points)
        })


# Global instance
_visualizer: Optional[TSNEVisualizer] = None


def get_tsne_visualizer() -> TSNEVisualizer:
    """Get or create t-SNE visualizer"""
    global _visualizer
    if _visualizer is None:
        _visualizer = TSNEVisualizer()
    return _visualizer
