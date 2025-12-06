"""
PCDS Enterprise - Transformer Detector
State-of-the-art attention-based anomaly detection
"""

import numpy as np
from typing import List, Tuple, Optional, Dict
import math


class PositionalEncoding:
    """Positional encoding for sequence data"""
    
    def __init__(self, d_model: int, max_len: int = 100):
        self.d_model = d_model
        self.max_len = max_len
        self._encoding = self._create_encoding()
    
    def _create_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encoding"""
        pe = np.zeros((self.max_len, self.d_model))
        position = np.arange(0, self.max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe
    
    def encode(self, sequence_length: int) -> np.ndarray:
        return self._encoding[:sequence_length]


class MultiHeadAttention:
    """Multi-head self-attention mechanism"""
    
    def __init__(self, d_model: int = 64, num_heads: int = 4):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weights (simplified - in production use proper initialization)
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def attention(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scaled dot-product attention"""
        scores = np.matmul(Q, K.T) / math.sqrt(self.d_k)
        attention_weights = self._softmax(scores)
        output = np.matmul(attention_weights, V)
        return output, attention_weights
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through multi-head attention"""
        # Linear projections
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Apply attention
        attn_output, attn_weights = self.attention(Q, K, V)
        
        # Output projection
        output = np.matmul(attn_output, self.W_o)
        
        return output, attn_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class FeedForward:
    """Position-wise feed-forward network"""
    
    def __init__(self, d_model: int = 64, d_ff: int = 256):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with ReLU activation"""
        hidden = np.maximum(0, np.matmul(x, self.W1) + self.b1)  # ReLU
        output = np.matmul(hidden, self.W2) + self.b2
        return output


class TransformerEncoderLayer:
    """Single transformer encoder layer"""
    
    def __init__(self, d_model: int = 64, num_heads: int = 4, d_ff: int = 256):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.gamma1 = np.ones(d_model)
        self.beta1 = np.zeros(d_model)
        self.gamma2 = np.ones(d_model)
        self.beta2 = np.zeros(d_model)
    
    def layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + 1e-6
        return gamma * (x - mean) / std + beta
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass with residual connections"""
        # Self-attention with residual
        attn_output, attn_weights = self.attention.forward(x)
        x = self.layer_norm(x + attn_output, self.gamma1, self.beta1)
        
        # Feed-forward with residual
        ff_output = self.feed_forward.forward(x)
        x = self.layer_norm(x + ff_output, self.gamma2, self.beta2)
        
        return x, attn_weights


class TransformerDetector:
    """
    Transformer-based anomaly detector
    Uses self-attention to analyze sequences of security events
    """
    
    def __init__(self, input_dim: int = 32, d_model: int = 64, num_heads: int = 4, 
                 num_layers: int = 2, sequence_length: int = 10):
        self.input_dim = input_dim
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = np.random.randn(input_dim, d_model) * 0.1
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, sequence_length)
        
        # Transformer layers
        self.layers = [
            TransformerEncoderLayer(d_model, num_heads, d_model * 4)
            for _ in range(num_layers)
        ]
        
        # Output layers
        self.output_projection = np.random.randn(d_model, 1) * 0.1
        
        # Anomaly threshold
        self.threshold = 0.5
        
        # Attention weights for explainability
        self.last_attention_weights = None
        
        # Training state
        self.is_trained = False
        self.baseline_mean = None
        self.baseline_std = None
    
    def project_input(self, x: np.ndarray) -> np.ndarray:
        """Project input features to model dimension"""
        return np.matmul(x, self.input_projection)
    
    def forward(self, sequence: np.ndarray) -> Tuple[float, Dict]:
        """
        Forward pass through transformer
        
        Args:
            sequence: Shape (seq_len, input_dim) - sequence of feature vectors
            
        Returns:
            anomaly_score: 0-1 probability of anomaly
            attention_info: Attention weights for explainability
        """
        # Ensure sequence is right shape
        if len(sequence.shape) == 1:
            sequence = sequence.reshape(1, -1)
        
        seq_len = sequence.shape[0]
        
        # Project to model dimension
        x = self.project_input(sequence)
        
        # Add positional encoding
        x = x + self.pos_encoding.encode(seq_len)[:seq_len]
        
        # Pass through transformer layers
        all_attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer.forward(x)
            all_attention_weights.append(attn_weights)
        
        self.last_attention_weights = all_attention_weights
        
        # Global average pooling
        pooled = np.mean(x, axis=0)
        
        # Output projection
        logit = np.matmul(pooled, self.output_projection)[0]
        
        # Sigmoid activation
        anomaly_score = 1 / (1 + np.exp(-logit))
        
        # Clip to valid range
        anomaly_score = np.clip(anomaly_score, 0.0, 1.0)
        
        return float(anomaly_score), {
            'attention_weights': all_attention_weights,
            'pooled_representation': pooled.tolist()
        }
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float, float]:
        """
        Predict if input is anomalous
        
        Returns:
            is_anomaly: Boolean
            anomaly_score: 0-1 score
            confidence: Model confidence
        """
        anomaly_score, _ = self.forward(features)
        
        # Apply baseline normalization if available
        if self.baseline_mean is not None:
            # Deviation from baseline
            deviation = abs(anomaly_score - self.baseline_mean)
            normalized_score = min(deviation / (self.baseline_std + 0.01), 1.0)
            anomaly_score = normalized_score
        
        is_anomaly = anomaly_score > self.threshold
        confidence = anomaly_score if is_anomaly else (1 - anomaly_score)
        
        return is_anomaly, float(anomaly_score), float(confidence)
    
    def train_baseline(self, normal_sequences: List[np.ndarray]):
        """Train baseline from normal data"""
        scores = []
        for seq in normal_sequences:
            score, _ = self.forward(seq)
            scores.append(score)
        
        self.baseline_mean = np.mean(scores)
        self.baseline_std = np.std(scores)
        self.is_trained = True
        
        return {
            'baseline_mean': float(self.baseline_mean),
            'baseline_std': float(self.baseline_std),
            'samples': len(normal_sequences)
        }
    
    def get_attention_explanation(self) -> Dict:
        """Get attention-based explanation for last prediction"""
        if self.last_attention_weights is None:
            return {'explanation': 'No prediction made yet'}
        
        # Aggregate attention across layers
        avg_attention = np.mean([w for w in self.last_attention_weights], axis=0)
        
        # Find most attended positions
        attention_scores = np.mean(avg_attention, axis=0)
        top_positions = np.argsort(attention_scores)[-3:][::-1]
        
        return {
            'top_attended_positions': top_positions.tolist(),
            'attention_scores': attention_scores.tolist(),
            'interpretation': 'Higher attention = more influential in decision'
        }


# Create global instance
transformer_detector = TransformerDetector(
    input_dim=32,
    d_model=64,
    num_heads=4,
    num_layers=2,
    sequence_length=10
)
