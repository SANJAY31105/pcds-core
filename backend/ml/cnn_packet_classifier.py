"""
CNN-based Deep Packet Inspection
Based on Paper 4: Convolutional Neural Networks for network packet analysis

Features:
- Treats packet bytes as 1D image for CNN processing
- Detects malicious patterns in raw packet payloads
- Binary classification (benign vs malicious)
- Can detect encrypted traffic anomalies
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@dataclass
class PacketAnalysis:
    """Result of CNN packet analysis"""
    is_malicious: bool
    confidence: float
    attack_class: str
    layer_activations: Optional[Dict] = None


class PacketCNN(nn.Module):
    """
    1D Convolutional Neural Network for packet classification
    
    Treats raw packet bytes as a 1D signal and applies convolutions
    to detect malicious patterns.
    """
    
    def __init__(self, input_size: int = 1500, n_classes: int = 2):
        super(PacketCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Calculate size after convolutions
        conv_output_size = input_size // 64  # After 3 pooling layers
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * conv_output_size, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, n_classes)
        
        self.n_classes = n_classes
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, packet_length)
        
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get class probabilities"""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)


class PacketFeatureExtractor:
    """
    Extract features from raw packet data for CNN input
    """
    
    def __init__(self, max_length: int = 1500):
        self.max_length = max_length
    
    def extract(self, packet_bytes: bytes) -> np.ndarray:
        """
        Convert raw packet bytes to normalized feature array
        """
        # Convert bytes to numpy array of integers
        byte_array = np.frombuffer(packet_bytes, dtype=np.uint8)
        
        # Normalize to [0, 1]
        normalized = byte_array.astype(np.float32) / 255.0
        
        # Pad or truncate to fixed length
        if len(normalized) >= self.max_length:
            return normalized[:self.max_length]
        else:
            padded = np.zeros(self.max_length, dtype=np.float32)
            padded[:len(normalized)] = normalized
            return padded
    
    def extract_headers(self, packet_bytes: bytes) -> Dict:
        """
        Extract header information from packet
        """
        if len(packet_bytes) < 20:
            return {"error": "Packet too short"}
        
        # Ethernet header (first 14 bytes if present)
        # IP header starts after Ethernet
        
        # For simplicity, extract basic features
        features = {
            "packet_length": len(packet_bytes),
            "header_length": min(40, len(packet_bytes)),
            "payload_length": max(0, len(packet_bytes) - 40),
            "first_bytes_entropy": self._calculate_entropy(packet_bytes[:64]),
            "payload_entropy": self._calculate_entropy(packet_bytes[40:]) if len(packet_bytes) > 40 else 0
        }
        
        return features
    
    def _calculate_entropy(self, data: bytes) -> float:
        """Calculate Shannon entropy of byte sequence"""
        if len(data) == 0:
            return 0.0
        
        byte_counts = np.zeros(256)
        for b in data:
            byte_counts[b] += 1
        
        probabilities = byte_counts / len(data)
        probabilities = probabilities[probabilities > 0]
        
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return float(entropy)


class DeepPacketInspector:
    """
    Deep Packet Inspection using CNN
    Analyzes raw packet payloads for malicious content
    """
    
    # Attack class names
    ATTACK_CLASSES = [
        "Benign",
        "DDoS",
        "PortScan",
        "Botnet",
        "Infiltration",
        "WebAttack",
        "Brute Force",
        "Malware"
    ]
    
    def __init__(self, device: str = None):
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.extractor = PacketFeatureExtractor()
        self.is_trained = False
        
        # Stats
        self.stats = {
            "packets_analyzed": 0,
            "malicious_detected": 0,
            "attack_classes": {c: 0 for c in self.ATTACK_CLASSES}
        }
        
        print(f"🔬 Deep Packet Inspector initialized on {self.device}")
    
    def initialize_model(self, n_classes: int = 2, input_size: int = 1500):
        """Initialize CNN model"""
        if not HAS_TORCH:
            print("⚠️ PyTorch not available")
            return
        
        self.model = PacketCNN(input_size=input_size, n_classes=n_classes)
        self.model.to(self.device)
        print(f"  ✅ CNN model initialized: {n_classes} classes")
    
    def analyze_packet(self, packet_bytes: bytes) -> PacketAnalysis:
        """
        Analyze a single packet for malicious content
        """
        self.stats["packets_analyzed"] += 1
        
        if self.model is None or not self.is_trained:
            # Use statistical analysis if model not trained
            features = self.extractor.extract_headers(packet_bytes)
            return self._statistical_analysis(packet_bytes, features)
        
        # Extract features
        features = self.extractor.extract(packet_bytes)
        
        # Prepare tensor
        x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        x = x.to(self.device)
        
        # Inference
        self.model.eval()
        with torch.no_grad():
            probs = self.model.predict_proba(x)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0, pred_class].item()
        
        is_malicious = pred_class > 0
        attack_class = self.ATTACK_CLASSES[min(pred_class, len(self.ATTACK_CLASSES) - 1)]
        
        if is_malicious:
            self.stats["malicious_detected"] += 1
            self.stats["attack_classes"][attack_class] += 1
        
        return PacketAnalysis(
            is_malicious=is_malicious,
            confidence=float(confidence),
            attack_class=attack_class
        )
    
    def _statistical_analysis(self, packet_bytes: bytes, 
                             features: Dict) -> PacketAnalysis:
        """
        Statistical analysis when CNN not available
        """
        risk_score = 0
        
        # High entropy in payload might indicate encrypted C2
        payload_entropy = features.get("payload_entropy", 0)
        if payload_entropy > 7.5:  # Near random
            risk_score += 30
        
        # Very small packets might be probes
        packet_length = features.get("packet_length", 0)
        if packet_length < 64:
            risk_score += 20
        
        # Very large packets might be exfiltration
        if packet_length > 1400:
            risk_score += 10
        
        # Low entropy header with high payload = encrypted tunnel
        header_entropy = features.get("first_bytes_entropy", 0)
        if header_entropy < 4 and payload_entropy > 7:
            risk_score += 25
        
        is_malicious = risk_score >= 40
        
        if is_malicious:
            self.stats["malicious_detected"] += 1
        
        return PacketAnalysis(
            is_malicious=is_malicious,
            confidence=min(risk_score / 100, 1.0),
            attack_class="Suspicious" if is_malicious else "Benign"
        )
    
    def analyze_batch(self, packets: List[bytes]) -> List[PacketAnalysis]:
        """Analyze multiple packets"""
        return [self.analyze_packet(p) for p in packets]
    
    def train(self, packets: List[bytes], labels: List[int], 
              epochs: int = 10, batch_size: int = 32):
        """Train the CNN model"""
        if not HAS_TORCH:
            print("⚠️ PyTorch not available for training")
            return
        
        if self.model is None:
            n_classes = len(set(labels))
            self.initialize_model(n_classes=n_classes)
        
        # Prepare data
        X = np.array([self.extractor.extract(p) for p in packets])
        y = np.array(labels)
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            print(f"  Epoch {epoch+1}/{epochs}: Loss = {total_loss/len(loader):.4f}")
        
        self.is_trained = True
        print("  ✅ CNN training complete")
    
    def save_model(self, path: str):
        """Save model to disk"""
        if self.model is not None:
            torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str, n_classes: int = 2):
        """Load model from disk"""
        self.initialize_model(n_classes=n_classes)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.is_trained = True
    
    def get_stats(self) -> Dict:
        """Get inspector statistics"""
        return {
            **self.stats,
            "model_trained": self.is_trained,
            "device": str(self.device)
        }


# Global instance
_inspector: Optional[DeepPacketInspector] = None


def get_packet_inspector() -> DeepPacketInspector:
    """Get or create packet inspector"""
    global _inspector
    if _inspector is None:
        _inspector = DeepPacketInspector()
    return _inspector
