"""
PCDS Enterprise - Bidirectional LSTM Detector
Enhanced temporal pattern detection with attention
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class LSTMCell:
    """Single LSTM cell implementation"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights (Xavier initialization)
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        
        # Gates: input, forget, cell, output
        self.W_i = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.b_i = np.zeros(hidden_size)
        
        self.W_f = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.b_f = np.ones(hidden_size)  # Initialize forget gate bias to 1
        
        self.W_c = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.b_c = np.zeros(hidden_size)
        
        self.W_o = np.random.randn(input_size + hidden_size, hidden_size) * scale
        self.b_o = np.zeros(hidden_size)
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass through LSTM cell"""
        # Concatenate input and previous hidden state
        combined = np.concatenate([x, h_prev])
        
        # Input gate
        i = self.sigmoid(np.dot(combined, self.W_i) + self.b_i)
        
        # Forget gate
        f = self.sigmoid(np.dot(combined, self.W_f) + self.b_f)
        
        # Cell candidate
        c_tilde = np.tanh(np.dot(combined, self.W_c) + self.b_c)
        
        # New cell state
        c = f * c_prev + i * c_tilde
        
        # Output gate
        o = self.sigmoid(np.dot(combined, self.W_o) + self.b_o)
        
        # New hidden state
        h = o * np.tanh(c)
        
        return h, c


class BidirectionalLSTM:
    """Bidirectional LSTM layer"""
    
    def __init__(self, input_size: int, hidden_size: int):
        self.hidden_size = hidden_size
        self.forward_cell = LSTMCell(input_size, hidden_size)
        self.backward_cell = LSTMCell(input_size, hidden_size)
    
    def forward(self, sequence: np.ndarray) -> np.ndarray:
        """Process sequence in both directions"""
        seq_len = sequence.shape[0]
        
        # Forward pass
        h_forward = np.zeros(self.hidden_size)
        c_forward = np.zeros(self.hidden_size)
        forward_outputs = []
        
        for t in range(seq_len):
            h_forward, c_forward = self.forward_cell.forward(
                sequence[t], h_forward, c_forward
            )
            forward_outputs.append(h_forward)
        
        # Backward pass
        h_backward = np.zeros(self.hidden_size)
        c_backward = np.zeros(self.hidden_size)
        backward_outputs = []
        
        for t in range(seq_len - 1, -1, -1):
            h_backward, c_backward = self.backward_cell.forward(
                sequence[t], h_backward, c_backward
            )
            backward_outputs.insert(0, h_backward)
        
        # Concatenate forward and backward
        outputs = np.array([
            np.concatenate([forward_outputs[t], backward_outputs[t]])
            for t in range(seq_len)
        ])
        
        return outputs


class AttentionLayer:
    """Self-attention for LSTM outputs"""
    
    def __init__(self, hidden_size: int):
        self.W_attention = np.random.randn(hidden_size, hidden_size) * 0.1
        self.v_attention = np.random.randn(hidden_size) * 0.1
    
    def forward(self, lstm_outputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply attention to LSTM outputs"""
        # Compute attention scores
        scores = np.tanh(np.dot(lstm_outputs, self.W_attention))
        scores = np.dot(scores, self.v_attention)
        
        # Softmax
        attention_weights = np.exp(scores - np.max(scores))
        attention_weights = attention_weights / np.sum(attention_weights)
        
        # Weighted sum
        context = np.sum(lstm_outputs * attention_weights.reshape(-1, 1), axis=0)
        
        return context, attention_weights


class TemporalLSTMDetector:
    """
    Bidirectional LSTM with attention for temporal anomaly detection
    Specialized for detecting time-series patterns in security events
    """
    
    def __init__(self, input_dim: int = 32, hidden_size: int = 64, num_layers: int = 2):
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Stacked bidirectional LSTMs
        self.lstm_layers = []
        current_input_size = input_dim
        for i in range(num_layers):
            layer = BidirectionalLSTM(current_input_size, hidden_size)
            self.lstm_layers.append(layer)
            current_input_size = hidden_size * 2  # Bidirectional doubles size
        
        # Attention layer
        self.attention = AttentionLayer(hidden_size * 2)
        
        # Output layers
        self.W_out = np.random.randn(hidden_size * 2, hidden_size) * 0.1
        self.b_out = np.zeros(hidden_size)
        self.W_final = np.random.randn(hidden_size, 1) * 0.1
        
        # Threshold
        self.threshold = 0.5
        
        # Pattern memory for temporal analysis
        self.pattern_memory = []
        self.max_patterns = 100
        
        # Attention weights for explainability
        self.last_attention_weights = None
    
    def forward(self, sequence: np.ndarray) -> Tuple[float, Dict]:
        """Forward pass through stacked BiLSTM with attention"""
        if len(sequence.shape) == 1:
            sequence = sequence.reshape(1, -1)
        
        # Pass through LSTM layers
        x = sequence
        for lstm_layer in self.lstm_layers:
            x = lstm_layer.forward(x)
        
        # Apply attention
        context, attention_weights = self.attention.forward(x)
        self.last_attention_weights = attention_weights
        
        # Output layers
        hidden = np.tanh(np.dot(context, self.W_out) + self.b_out)
        logit = np.dot(hidden, self.W_final)[0]
        
        # Sigmoid
        anomaly_score = 1 / (1 + np.exp(-np.clip(logit, -500, 500)))
        
        return float(anomaly_score), {
            'attention_weights': attention_weights.tolist(),
            'context_vector': context.tolist()
        }
    
    def predict(self, features: np.ndarray) -> Tuple[bool, float, float]:
        """Predict anomaly with temporal context"""
        # Add to pattern memory
        self.pattern_memory.append(features.copy())
        if len(self.pattern_memory) > self.max_patterns:
            self.pattern_memory.pop(0)
        
        # Use recent patterns as sequence
        if len(self.pattern_memory) >= 3:
            sequence = np.array(self.pattern_memory[-10:])  # Last 10 patterns
        else:
            sequence = features.reshape(1, -1)
        
        anomaly_score, _ = self.forward(sequence)
        is_anomaly = anomaly_score > self.threshold
        confidence = anomaly_score if is_anomaly else (1 - anomaly_score)
        
        return is_anomaly, float(anomaly_score), float(confidence)
    
    def detect_temporal_patterns(self, sequence: np.ndarray) -> Dict:
        """Analyze temporal patterns in sequence"""
        if len(sequence) < 3:
            return {'pattern': 'insufficient_data'}
        
        # Calculate temporal features
        diffs = np.diff(np.mean(sequence, axis=1))
        
        patterns = {
            'trend': 'increasing' if np.mean(diffs) > 0.1 else 
                     'decreasing' if np.mean(diffs) < -0.1 else 'stable',
            'volatility': float(np.std(diffs)),
            'burst_detected': bool(np.max(np.abs(diffs)) > 0.5),
            'periodic': bool(np.std(diffs) < 0.1 and len(diffs) > 5)
        }
        
        return patterns
    
    def get_temporal_explanation(self) -> Dict:
        """Get explanation based on temporal analysis"""
        if self.last_attention_weights is None:
            return {'explanation': 'No prediction made yet'}
        
        # Find most important time steps
        weights = np.array(self.last_attention_weights)
        top_steps = np.argsort(weights)[-3:][::-1]
        
        return {
            'most_important_timesteps': top_steps.tolist(),
            'attention_distribution': weights.tolist(),
            'interpretation': 'Higher attention = more anomalous time period'
        }


# Global instance
temporal_lstm_detector = TemporalLSTMDetector(
    input_dim=32,
    hidden_size=64,
    num_layers=2
)
