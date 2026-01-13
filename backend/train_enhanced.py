"""
Enhanced Training for 95%+ Accuracy
- 2M synthetic samples
- Focal Loss for imbalanced classes
- Label Smoothing regularization
- Larger architecture (512 hidden)
- 100 epochs with warmup
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple
from datetime import datetime
import json
import random


# ============================================
# Focal Loss for Imbalanced Classes
# ============================================

class FocalLoss(nn.Module):
    """
    Focal Loss - focuses on hard examples
    γ (gamma) = focusing parameter
    """
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', 
                                  label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


# ============================================
# Enhanced Model Architecture
# ============================================

class EnhancedClassifier(nn.Module):
    """
    Enhanced classifier with:
    - Larger hidden layers (512)
    - More residual blocks
    - GELU activation
    - Layer normalization
    """
    def __init__(self, input_dim=40, hidden_dim=512, n_classes=11):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            self._make_block(hidden_dim) for _ in range(4)
        ])
        
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def _make_block(self, dim):
        return nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = x + block(x)  # Residual connection
        
        return self.head(x)


# ============================================
# Enhanced Data Generation
# ============================================

def generate_enhanced_dataset(n_samples: int = 2000000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate 2M high-quality synthetic samples"""
    
    print(f"🔧 Generating {n_samples:,} enhanced samples...")
    
    n_features = 40
    n_classes = 11
    
    # Class distribution (slightly imbalanced, realistic)
    class_ratios = [0.35, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.015, 0.005]
    
    X_list = []
    y_list = []
    
    for class_id, ratio in enumerate(class_ratios):
        n_class = int(n_samples * ratio)
        print(f"   Class {class_id}: {n_class:,} samples")
        
        # Generate class-specific features
        X_class = generate_class_features(class_id, n_class, n_features)
        y_class = np.full(n_class, class_id, dtype=np.int64)
        
        X_list.append(X_class)
        y_list.append(y_class)
    
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"✅ Generated {len(X):,} samples")
    
    return X.astype(np.float32), y


def generate_class_features(class_id: int, n_samples: int, n_features: int) -> np.ndarray:
    """Generate features for specific attack class"""
    
    X = np.zeros((n_samples, n_features), dtype=np.float32)
    
    # Class-specific feature patterns
    patterns = {
        0: {"mean": 0.15, "std": 0.1, "peaks": []},  # Normal
        1: {"mean": 0.6, "std": 0.2, "peaks": [0, 1, 2]},  # DoS
        2: {"mean": 0.4, "std": 0.15, "peaks": [3, 4, 5]},  # Scan
        3: {"mean": 0.5, "std": 0.2, "peaks": [6, 7, 8]},  # Brute Force
        4: {"mean": 0.55, "std": 0.2, "peaks": [9, 10, 11]},  # Web Attack
        5: {"mean": 0.45, "std": 0.15, "peaks": [12, 13, 14]},  # Infiltration
        6: {"mean": 0.5, "std": 0.18, "peaks": [15, 16, 17]},  # Botnet
        7: {"mean": 0.52, "std": 0.2, "peaks": [18, 19, 20]},  # Backdoor
        8: {"mean": 0.48, "std": 0.18, "peaks": [21, 22, 23]},  # Worms
        9: {"mean": 0.55, "std": 0.22, "peaks": [24, 25, 26]},  # Fuzzers
        10: {"mean": 0.45, "std": 0.2, "peaks": [27, 28, 29]},  # Other
    }
    
    pattern = patterns.get(class_id, patterns[0])
    
    # Base features
    X = np.random.normal(pattern["mean"], pattern["std"], (n_samples, n_features))
    
    # Add peak features for attack classes
    for peak_idx in pattern["peaks"]:
        if peak_idx < n_features:
            X[:, peak_idx] = np.random.uniform(0.6, 1.0, n_samples)
    
    # Add some noise and variations
    noise = np.random.normal(0, 0.05, X.shape)
    X = X + noise
    
    # Add inter-sample variations
    for i in range(n_samples):
        # Random feature spikes
        n_spikes = random.randint(0, 3)
        for _ in range(n_spikes):
            spike_idx = random.randint(0, n_features - 1)
            X[i, spike_idx] = random.uniform(0.5, 1.0) if class_id > 0 else random.uniform(0.0, 0.3)
    
    # Normalize to [0, 1]
    X = np.clip(X, 0.0, 1.0)
    
    return X


# ============================================
# Mixup Data Augmentation
# ============================================

def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================
# Training Function
# ============================================

def train_enhanced():
    """Enhanced training for 95%+ accuracy"""
    
    print("=" * 70)
    print("🚀 ENHANCED TRAINING FOR 95%+ ACCURACY")
    print("   2M samples | Focal Loss | 100 epochs")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Generate data
    X, y = generate_enhanced_dataset(n_samples=2000000)
    
    # Split
    split_idx = int(0.85 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\n📊 Training: {len(X_train):,} | Validation: {len(X_val):,}")
    
    # DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, 
                             num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, 
                           num_workers=0, pin_memory=True)
    
    # Model
    model = EnhancedClassifier(hidden_dim=512).to(device)
    print(f"   Model params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Focal Loss with label smoothing
    criterion = FocalLoss(gamma=2.0, label_smoothing=0.1)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    warmup_epochs = 5
    total_epochs = 100
    
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print("\n" + "=" * 70)
    print("🚀 TRAINING")
    print("=" * 70)
    
    best_acc = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(total_epochs):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Mixup augmentation (50% of batches)
            use_mixup = random.random() < 0.5
            
            if use_mixup:
                batch_X, y_a, y_b, lam = mixup_data(batch_X, batch_y, alpha=0.4)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            if use_mixup:
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                loss = criterion(outputs, batch_y)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        lr = scheduler.get_last_lr()[0]
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), "ml/models/trained/enhanced_classifier.pt")
            marker = " ★ BEST"
            
            # Also save if we hit target
            if val_acc >= 95.0:
                torch.save(model.state_dict(), "ml/models/trained/target_95_classifier.pt")
                marker += " 🎯 TARGET!"
        else:
            patience_counter += 1
            marker = ""
        
        # Print every 5 epochs or on new best
        if epoch % 5 == 0 or val_acc > best_acc - 0.1 or val_acc >= 95:
            print(f"Epoch {epoch+1:3d}/{total_epochs} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {lr:.6f}{marker}")
        
        # Early stopping
        if patience_counter >= patience and epoch > 50:
            print(f"\n⏹️ Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            break
    
    print("\n" + "=" * 70)
    print(f"✅ TRAINING COMPLETE")
    print(f"   Best Accuracy: {best_acc:.2f}%")
    print(f"   Model: ml/models/trained/enhanced_classifier.pt")
    if best_acc >= 95:
        print(f"   🎯 TARGET ACHIEVED!")
    print("=" * 70)
    
    # Save results
    results = {
        "best_accuracy": best_acc,
        "samples": len(X),
        "epochs_trained": epoch + 1,
        "target_reached": best_acc >= 95,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("ml/models/trained/enhanced_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return best_acc


if __name__ == "__main__":
    train_enhanced()
