"""
ML Training Pipeline - Per-Attack-Category Classifiers
Individual models trained for each attack type.

Attack Categories:
- Brute Force (FTP, SSH, Web)
- DoS/DDoS
- Botnet/C2
- Reconnaissance/Scanning
- Data Exfiltration
- Web Attacks (XSS, SQLi)
- Malware/Exploit
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

# PyTorch imports (with fallback)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available. Using numpy-based models.")


# ==================== Attack Classifier Models ====================

class BaseAttackClassifier:
    """Base class for attack classifiers"""
    
    def __init__(self, name: str, input_dim: int = 64):
        self.name = name
        self.input_dim = input_dim
        self.trained = False
        self.metrics = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict:
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def save(self, path: str):
        raise NotImplementedError
    
    def load(self, path: str):
        raise NotImplementedError


if TORCH_AVAILABLE:
    
    class AttackClassifierNN(nn.Module):
        """Neural network for binary attack classification"""
        
        def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 64, 32]):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
            
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    
    class TorchAttackClassifier(BaseAttackClassifier):
        """PyTorch-based attack classifier"""
        
        def __init__(self, name: str, input_dim: int = 64, 
                     hidden_dims: List[int] = [128, 64, 32]):
            super().__init__(name, input_dim)
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = AttackClassifierNN(input_dim, hidden_dims).to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            self.criterion = nn.BCELoss()
            self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        def train(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray = None, y_val: np.ndarray = None,
                  epochs: int = 50, batch_size: int = 256) -> Dict:
            """Train the classifier"""
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train.reshape(-1, 1))
            )
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            if X_val is not None:
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.FloatTensor(y_val.reshape(-1, 1))
                )
                val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Training loop
            self.model.train()
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                
                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    self.optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_train_loss = epoch_loss / len(train_loader)
                self.history['train_loss'].append(avg_train_loss)
                
                # Validation
                if X_val is not None:
                    val_loss, val_acc = self._evaluate(val_loader)
                    self.history['val_loss'].append(val_loss)
                    self.history['val_acc'].append(val_acc)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                    
                    if (epoch + 1) % 10 == 0:
                        print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - "
                              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
            
            self.trained = True
            
            # Calculate final metrics
            self.metrics = {
                'train_loss': self.history['train_loss'][-1],
                'val_loss': self.history['val_loss'][-1] if X_val is not None else None,
                'val_acc': self.history['val_acc'][-1] if X_val is not None else None,
                'epochs': epochs
            }
            
            return self.metrics
        
        def _evaluate(self, val_loader) -> Tuple[float, float]:
            """Evaluate on validation set"""
            self.model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)
                    total_loss += loss.item()
                    
                    predicted = (outputs > 0.5).float()
                    correct += (predicted == y_batch).sum().item()
                    total += y_batch.size(0)
            
            self.model.train()
            return total_loss / len(val_loader), correct / total
        
        def predict(self, X: np.ndarray) -> np.ndarray:
            """Predict binary labels"""
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                probs = self.model(X_tensor).cpu().numpy()
                return (probs > 0.5).astype(int).flatten()
        
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            """Predict probabilities"""
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).to(self.device)
                return self.model(X_tensor).cpu().numpy().flatten()
        
        def save(self, path: str):
            """Save model weights"""
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'metrics': self.metrics,
                'history': self.history,
                'name': self.name,
                'input_dim': self.input_dim
            }, path)
            print(f"✅ Saved {self.name} to {path}")
        
        def load(self, path: str):
            """Load model weights"""
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.metrics = checkpoint.get('metrics', {})
            self.history = checkpoint.get('history', {})
            self.trained = True
            print(f"✅ Loaded {self.name} from {path}")


# Fallback numpy-based classifier
class NumpyAttackClassifier(BaseAttackClassifier):
    """Simple numpy-based classifier (logistic regression style)"""
    
    def __init__(self, name: str, input_dim: int = 64):
        super().__init__(name, input_dim)
        self.weights = np.random.randn(input_dim) * 0.01
        self.bias = 0.0
        self.learning_rate = 0.01
    
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 100, batch_size: int = 256) -> Dict:
        """Train using gradient descent"""
        
        for epoch in range(epochs):
            # Forward pass
            z = np.dot(X_train, self.weights) + self.bias
            predictions = self._sigmoid(z)
            
            # Compute gradients
            error = predictions - y_train
            dw = np.dot(X_train.T, error) / len(y_train)
            db = np.mean(error)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if (epoch + 1) % 20 == 0:
                loss = -np.mean(y_train * np.log(predictions + 1e-8) + 
                               (1 - y_train) * np.log(1 - predictions + 1e-8))
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")
        
        self.trained = True
        
        # Validation accuracy
        if X_val is not None:
            val_preds = self.predict(X_val)
            val_acc = np.mean(val_preds == y_val)
            self.metrics = {'val_acc': val_acc}
        
        return self.metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def save(self, path: str):
        np.savez(path, weights=self.weights, bias=self.bias, 
                 metrics=self.metrics, name=self.name)
    
    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self.weights = data['weights']
        self.bias = data['bias']
        self.metrics = data['metrics'].item()
        self.trained = True


# ==================== Attack Category Manager ====================

class AttackCategoryManager:
    """
    Manages per-attack-category classifiers
    
    Categories:
    - brute_force: FTP/SSH/Web brute force
    - dos: DoS attacks
    - ddos: DDoS attacks
    - botnet: Botnet/C2 traffic
    - recon: Port scanning, reconnaissance
    - exfiltration: Data exfiltration
    - web_attack: XSS, SQLi
    - exploit: Exploits, shellcode
    - malware: Malware, worms
    """
    
    ATTACK_CATEGORIES = [
        'brute_force', 'dos', 'ddos', 'botnet', 'recon',
        'exfiltration', 'web_attack', 'exploit', 'malware'
    ]
    
    def __init__(self, input_dim: int = 64, model_dir: str = "ml/models/trained"):
        self.input_dim = input_dim
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.classifiers: Dict[str, BaseAttackClassifier] = {}
        self._initialize_classifiers()
    
    def _initialize_classifiers(self):
        """Initialize classifier for each attack category"""
        ClassifierClass = TorchAttackClassifier if TORCH_AVAILABLE else NumpyAttackClassifier
        
        for category in self.ATTACK_CATEGORIES:
            self.classifiers[category] = ClassifierClass(
                name=f"{category}_classifier",
                input_dim=self.input_dim
            )
        
        print(f"✅ Initialized {len(self.classifiers)} attack classifiers")
    
    def train_category(self, category: str, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None,
                       epochs: int = 50) -> Dict:
        """Train a specific attack category classifier"""
        
        if category not in self.classifiers:
            raise ValueError(f"Unknown category: {category}")
        
        print(f"\n🎓 Training {category} classifier...")
        print(f"   Training samples: {len(X_train)} (positive: {sum(y_train)})")
        
        metrics = self.classifiers[category].train(
            X_train, y_train, X_val, y_val, epochs=epochs
        )
        
        print(f"   ✅ {category} trained - Val Acc: {metrics.get('val_acc', 'N/A')}")
        
        return metrics
    
    def train_all(self, datasets: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
                  epochs: int = 50) -> Dict[str, Dict]:
        """
        Train all classifiers using provided datasets
        
        Args:
            datasets: Dict mapping attack category to (X, y_binary, y_category) tuples
            epochs: Training epochs per classifier
        
        Returns:
            Dict of training metrics per category
        """
        all_metrics = {}
        
        for category in self.ATTACK_CATEGORIES:
            if category in datasets:
                X, y_binary, _ = datasets[category]
                
                # Split for validation
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y_binary[:split_idx], y_binary[split_idx:]
                
                metrics = self.train_category(
                    category, X_train, y_train, X_val, y_val, epochs
                )
                all_metrics[category] = metrics
            else:
                print(f"⚠️ No data for {category}")
        
        return all_metrics
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from all classifiers"""
        predictions = {}
        
        for category, classifier in self.classifiers.items():
            if classifier.trained:
                predictions[category] = classifier.predict_proba(X)
            else:
                predictions[category] = np.zeros(len(X))
        
        return predictions
    
    def classify(self, X: np.ndarray, threshold: float = 0.5) -> List[str]:
        """
        Classify samples to attack categories
        
        Returns most likely attack category for each sample
        """
        predictions = self.predict(X)
        
        # Stack predictions
        pred_matrix = np.array([predictions[cat] for cat in self.ATTACK_CATEGORIES])
        
        # Get max prediction per sample
        max_indices = np.argmax(pred_matrix, axis=0)
        max_probs = np.max(pred_matrix, axis=0)
        
        # Classify
        results = []
        for i, (idx, prob) in enumerate(zip(max_indices, max_probs)):
            if prob >= threshold:
                results.append(self.ATTACK_CATEGORIES[idx])
            else:
                results.append('benign')
        
        return results
    
    def save_all(self):
        """Save all trained classifiers"""
        for category, classifier in self.classifiers.items():
            if classifier.trained:
                ext = '.pt' if TORCH_AVAILABLE else '.npz'
                path = self.model_dir / f"{category}_classifier{ext}"
                classifier.save(str(path))
        
        # Save metadata
        metadata = {
            'categories': self.ATTACK_CATEGORIES,
            'input_dim': self.input_dim,
            'saved_at': datetime.now().isoformat(),
            'torch_available': TORCH_AVAILABLE
        }
        with open(self.model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved all classifiers to {self.model_dir}")
    
    def load_all(self):
        """Load all trained classifiers"""
        ext = '.pt' if TORCH_AVAILABLE else '.npz'
        
        for category in self.ATTACK_CATEGORIES:
            path = self.model_dir / f"{category}_classifier{ext}"
            if path.exists():
                self.classifiers[category].load(str(path))
        
        print(f"✅ Loaded classifiers from {self.model_dir}")


# ==================== Factory ====================

def create_attack_classifier(name: str, input_dim: int = 64) -> BaseAttackClassifier:
    """Create a single attack classifier"""
    if TORCH_AVAILABLE:
        return TorchAttackClassifier(name, input_dim)
    return NumpyAttackClassifier(name, input_dim)


def create_category_manager(input_dim: int = 64, model_dir: str = "ml/models/trained") -> AttackCategoryManager:
    """Create attack category manager with all classifiers"""
    return AttackCategoryManager(input_dim, model_dir)


if __name__ == "__main__":
    print("Attack Classifiers Test")
    print("=" * 50)
    
    # Create test data
    np.random.seed(42)
    X_train = np.random.randn(1000, 64).astype(np.float32)
    y_train = (np.random.rand(1000) > 0.7).astype(np.float32)
    
    X_val = np.random.randn(200, 64).astype(np.float32)
    y_val = (np.random.rand(200) > 0.7).astype(np.float32)
    
    # Test single classifier
    classifier = create_attack_classifier("test_brute_force")
    metrics = classifier.train(X_train, y_train, X_val, y_val, epochs=20)
    
    print(f"\n✅ Training complete")
    print(f"   Metrics: {metrics}")
    
    # Test predictions
    predictions = classifier.predict(X_val[:10])
    probs = classifier.predict_proba(X_val[:10])
    
    print(f"   Sample predictions: {predictions}")
    print(f"   Sample probabilities: {probs[:5]}")
