"""
Advanced Network Anomaly Detection Module
Based on: Schummer et al. (2024) - ML-Based Network Anomaly Detection

Features:
- SVM Classifier (96.5% accuracy in paper)
- Change Point Detection for temporal anomalies
- Tukey IQR-based outlier detection
- Network metrics analysis (latency, jitter, throughput, packet loss)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import deque
import pickle

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class ChangePoint:
    """Detected change point in time series"""
    timestamp: str
    index: int
    metric: str
    before_mean: float
    after_mean: float
    change_magnitude: float
    significance: float


@dataclass
class TukeyOutlier:
    """Tukey method outlier detection result"""
    value: float
    metric: str
    is_outlier: bool
    lower_bound: float
    upper_bound: float
    iqr: float


@dataclass
class NetworkAnomaly:
    """Complete network anomaly detection result"""
    anomaly_id: str
    timestamp: str
    is_anomaly: bool
    confidence: float
    detection_methods: List[str]
    metrics: Dict[str, float]
    change_points: List[ChangePoint]
    outliers: List[TukeyOutlier]
    svm_prediction: Optional[int] = None


class SVMClassifier:
    """
    SVM-based anomaly classifier
    Paper achieved 96.5% accuracy with SVM
    """
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.is_trained = False
        
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
        """Train SVM classifier"""
        if not HAS_SKLEARN:
            print("⚠️ scikit-learn not available")
            return None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Train SVM
        self.model = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            probability=True,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
        accuracy = self.model.score(X_test, y_test)
        self.is_trained = True
        
        print(f"  ✅ SVM trained: {accuracy*100:.2f}% accuracy")
        return accuracy
    
    def predict(self, X: np.ndarray) -> Tuple[int, float]:
        """Predict anomaly with confidence"""
        if not self.is_trained or self.model is None:
            return 0, 0.5
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        proba = self.model.predict_proba(X_scaled)[0]
        confidence = float(max(proba))
        
        return int(prediction), confidence
    
    def save(self, path: str):
        """Save trained model"""
        if self.is_trained:
            with open(path, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def load(self, path: str):
        """Load trained model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.is_trained = True


class ChangePointDetector:
    """
    Change Point Detection using dynamic programming
    Identifies significant shifts in time series data
    """
    
    def __init__(self, min_segment_length: int = 10, penalty: float = 3.0):
        self.min_segment_length = min_segment_length
        self.penalty = penalty  # Higher = fewer change points
        
    def detect(self, data: np.ndarray, max_changepoints: int = 5) -> List[int]:
        """
        Detect change points in time series using PELT-like algorithm
        """
        n = len(data)
        if n < 2 * self.min_segment_length:
            return []
        
        # Use CUSUM-based detection
        change_points = []
        mean = np.mean(data)
        cumsum = np.cumsum(data - mean)
        
        # Find points where cumulative sum deviates significantly
        threshold = self.penalty * np.std(data)
        
        # Sliding window approach
        window_size = self.min_segment_length
        for i in range(window_size, n - window_size):
            left_mean = np.mean(data[i-window_size:i])
            right_mean = np.mean(data[i:i+window_size])
            
            change = abs(right_mean - left_mean)
            
            if change > threshold:
                # Check if this is a local maximum
                is_local_max = True
                for cp in change_points:
                    if abs(i - cp) < self.min_segment_length:
                        is_local_max = False
                        break
                
                if is_local_max:
                    change_points.append(i)
        
        # Keep only top N change points
        if len(change_points) > max_changepoints:
            # Sort by magnitude of change
            magnitudes = []
            for cp in change_points:
                left = np.mean(data[max(0, cp-window_size):cp])
                right = np.mean(data[cp:min(n, cp+window_size)])
                magnitudes.append((cp, abs(right - left)))
            
            magnitudes.sort(key=lambda x: x[1], reverse=True)
            change_points = [cp for cp, _ in magnitudes[:max_changepoints]]
        
        return sorted(change_points)
    
    def analyze_changepoint(self, data: np.ndarray, index: int, 
                           metric_name: str) -> ChangePoint:
        """Analyze a detected change point"""
        window = min(20, len(data) // 4)
        
        before = data[max(0, index-window):index]
        after = data[index:min(len(data), index+window)]
        
        before_mean = float(np.mean(before)) if len(before) > 0 else 0
        after_mean = float(np.mean(after)) if len(after) > 0 else 0
        
        # Calculate significance (z-score of change)
        overall_std = np.std(data)
        if overall_std > 0:
            significance = abs(after_mean - before_mean) / overall_std
        else:
            significance = 0
        
        return ChangePoint(
            timestamp=datetime.utcnow().isoformat(),
            index=index,
            metric=metric_name,
            before_mean=before_mean,
            after_mean=after_mean,
            change_magnitude=abs(after_mean - before_mean),
            significance=float(significance)
        )


class TukeyOutlierDetector:
    """
    Tukey's method for outlier detection
    Uses IQR (Interquartile Range) to identify statistical outliers
    """
    
    def __init__(self, k: float = 1.5):
        """
        Args:
            k: Threshold multiplier (1.5 = standard outliers, 3.0 = extreme outliers)
        """
        self.k = k
        self.learned_stats: Dict[str, Dict] = {}
    
    def learn_statistics(self, data: np.ndarray, metric_name: str):
        """Learn IQR statistics from training data"""
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        self.learned_stats[metric_name] = {
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(iqr),
            "lower_bound": float(q1 - self.k * iqr),
            "upper_bound": float(q3 + self.k * iqr),
            "mean": float(np.mean(data)),
            "std": float(np.std(data))
        }
        
        return self.learned_stats[metric_name]
    
    def detect(self, value: float, metric_name: str) -> TukeyOutlier:
        """Detect if value is an outlier"""
        if metric_name not in self.learned_stats:
            # Use default thresholds
            return TukeyOutlier(
                value=value,
                metric=metric_name,
                is_outlier=False,
                lower_bound=float('-inf'),
                upper_bound=float('inf'),
                iqr=0
            )
        
        stats = self.learned_stats[metric_name]
        is_outlier = value < stats["lower_bound"] or value > stats["upper_bound"]
        
        return TukeyOutlier(
            value=value,
            metric=metric_name,
            is_outlier=is_outlier,
            lower_bound=stats["lower_bound"],
            upper_bound=stats["upper_bound"],
            iqr=stats["iqr"]
        )
    
    def detect_batch(self, data: np.ndarray, metric_name: str) -> List[TukeyOutlier]:
        """Detect outliers in batch"""
        return [self.detect(v, metric_name) for v in data]


class NetworkAnomalyDetector:
    """
    Comprehensive Network Anomaly Detection System
    Combines SVM, Change Point Detection, and Tukey Outliers
    """
    
    def __init__(self):
        self.svm = SVMClassifier()
        self.change_detector = ChangePointDetector()
        self.tukey_detector = TukeyOutlierDetector()
        
        # Time series buffers for each metric
        self.metric_buffers: Dict[str, deque] = {
            "throughput": deque(maxlen=1000),
            "latency": deque(maxlen=1000),
            "jitter": deque(maxlen=1000),
            "packet_loss": deque(maxlen=1000),
            "congestion": deque(maxlen=1000)
        }
        
        # Stats
        self.stats = {
            "total_analyzed": 0,
            "anomalies_detected": 0,
            "change_points_found": 0,
            "outliers_found": 0
        }
        
        print("🔍 Network Anomaly Detector initialized")
    
    def add_metrics(self, metrics: Dict[str, float]):
        """Add new metrics to buffers"""
        for name, value in metrics.items():
            if name in self.metric_buffers:
                self.metric_buffers[name].append(value)
    
    def analyze(self, metrics: Dict[str, float]) -> NetworkAnomaly:
        """
        Comprehensive anomaly analysis
        """
        import uuid
        
        self.stats["total_analyzed"] += 1
        self.add_metrics(metrics)
        
        detection_methods = []
        change_points = []
        outliers = []
        
        # 1. Tukey outlier detection for each metric
        for name, value in metrics.items():
            if name in self.metric_buffers and len(self.metric_buffers[name]) > 20:
                # Learn stats if not already
                if name not in self.tukey_detector.learned_stats:
                    self.tukey_detector.learn_statistics(
                        np.array(list(self.metric_buffers[name])), name
                    )
                
                outlier = self.tukey_detector.detect(value, name)
                if outlier.is_outlier:
                    outliers.append(outlier)
                    self.stats["outliers_found"] += 1
        
        if outliers:
            detection_methods.append("tukey_outlier")
        
        # 2. Change point detection on buffers
        for name, buffer in self.metric_buffers.items():
            if len(buffer) >= 50:
                data = np.array(list(buffer))
                cps = self.change_detector.detect(data)
                
                for cp_idx in cps[-3:]:  # Only check recent change points
                    if cp_idx > len(data) - 10:  # Recent change point
                        cp = self.change_detector.analyze_changepoint(data, cp_idx, name)
                        if cp.significance > 2.0:  # Significant change
                            change_points.append(cp)
                            self.stats["change_points_found"] += 1
        
        if change_points:
            detection_methods.append("change_point")
        
        # 3. SVM prediction (if trained)
        svm_pred = None
        svm_conf = 0.5
        if self.svm.is_trained:
            features = np.array([metrics.get(m, 0) for m in self.metric_buffers.keys()])
            svm_pred, svm_conf = self.svm.predict(features)
            if svm_pred == 1:  # Anomaly
                detection_methods.append("svm")
        
        # Determine final anomaly status
        is_anomaly = len(detection_methods) > 0
        
        # Calculate overall confidence
        if is_anomaly:
            confidence = max(
                len(outliers) / max(len(metrics), 1),  # Outlier ratio
                max([cp.significance / 5.0 for cp in change_points], default=0),  # Change significance
                svm_conf if svm_pred == 1 else 0
            )
            confidence = min(confidence, 1.0)
            self.stats["anomalies_detected"] += 1
        else:
            confidence = 1.0 - svm_conf if self.svm.is_trained else 0.1
        
        return NetworkAnomaly(
            anomaly_id=str(uuid.uuid4())[:8],
            timestamp=datetime.utcnow().isoformat(),
            is_anomaly=is_anomaly,
            confidence=float(confidence),
            detection_methods=detection_methods,
            metrics=metrics,
            change_points=change_points,
            outliers=outliers,
            svm_prediction=svm_pred
        )
    
    def train_svm(self, X: np.ndarray, y: np.ndarray):
        """Train the SVM classifier"""
        return self.svm.train(X, y)
    
    def learn_baseline(self, data: Dict[str, np.ndarray]):
        """Learn baseline statistics for all metrics"""
        for name, values in data.items():
            if name in self.metric_buffers:
                self.tukey_detector.learn_statistics(values, name)
                for v in values[:100]:  # Add first 100 to buffer
                    self.metric_buffers[name].append(v)
        
        print(f"  ✅ Learned baseline for {len(data)} metrics")
    
    def get_stats(self) -> Dict:
        """Get detector statistics"""
        return {
            **self.stats,
            "svm_trained": self.svm.is_trained,
            "metrics_monitored": list(self.metric_buffers.keys()),
            "tukey_metrics_learned": list(self.tukey_detector.learned_stats.keys()),
            "buffer_sizes": {k: len(v) for k, v in self.metric_buffers.items()}
        }


# Global instance
_detector: Optional[NetworkAnomalyDetector] = None


def get_network_anomaly_detector() -> NetworkAnomalyDetector:
    """Get or create network anomaly detector"""
    global _detector
    if _detector is None:
        _detector = NetworkAnomalyDetector()
    return _detector
