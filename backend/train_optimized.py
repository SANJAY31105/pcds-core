"""
OPTIMIZED ML Training Script - Best Possible Model
- Class weights (attacks weighted 2.5x)
- Deeper network (512→256→128→64)
- Learning rate scheduler
- 100 epochs with early stopping
- GPU acceleration (RTX 4060)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from ml.training import CICIDS2017Loader, UNSWNB15Loader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import json

# Deeper network architecture
class OptimizedAttackClassifier(nn.Module):
    """Deeper neural network for better attack detection"""
    
    def __init__(self, input_dim: int = 38):
        super().__init__()
        
        self.network = nn.Sequential(
            # Layer 1: 512 neurons
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            # Layer 2: 256 neurons
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 3: 128 neurons
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 4: 64 neurons
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)


def main():
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Using CPU")
    
    print("=" * 60)
    print("🚀 OPTIMIZED ML Training - Best Possible Model")
    print("=" * 60)
    
    # Load datasets
    print("\n📥 Loading datasets...")
    try:
        cic_loader = CICIDS2017Loader()
        X_cic, y_cic, _ = cic_loader.load()
    except:
        X_cic, y_cic = None, None
    
    try:
        unsw_loader = UNSWNB15Loader()
        X_unsw, y_unsw, _ = unsw_loader.load()
    except:
        X_unsw, y_unsw = None, None
    
    # Combine datasets
    if X_cic is not None and X_unsw is not None:
        max_features = max(X_cic.shape[1], X_unsw.shape[1])
        if X_cic.shape[1] < max_features:
            X_cic = np.pad(X_cic, ((0,0), (0, max_features - X_cic.shape[1])))
        if X_unsw.shape[1] < max_features:
            X_unsw = np.pad(X_unsw, ((0,0), (0, max_features - X_unsw.shape[1])))
        X = np.vstack([X_cic, X_unsw])
        y = np.concatenate([y_cic, y_unsw])
    elif X_cic is not None:
        X, y = X_cic, y_cic
    else:
        X, y = X_unsw, y_unsw
    
    print(f"📊 Dataset: {X.shape[0]:,} samples, {X.shape[1]} features")
    print(f"   Attacks: {sum(y):,} ({100*sum(y)/len(y):.1f}%)")
    print(f"   Benign: {len(y)-sum(y):,} ({100*(len(y)-sum(y))/len(y):.1f}%)")
    
    # Calculate class weights (weight attacks more to improve recall)
    n_attacks = sum(y)
    n_benign = len(y) - n_attacks
    weight_attack = n_benign / n_attacks  # ~2.5x
    print(f"\n⚖️ Class weight for attacks: {weight_attack:.2f}x")
    
    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )
    print(f"📊 Split: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    
    # Create DataLoaders
    batch_size = 4096  # Larger batch for GPU
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create model
    model = OptimizedAttackClassifier(input_dim=X.shape[1]).to(device)
    print(f"\n🧠 Model: Optimized (512→256→128→64)")
    
    # Loss with class weights
    pos_weight = torch.tensor([weight_attack]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer with learning rate scheduler
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training parameters
    epochs = 100
    patience = 15
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    print(f"\n🎓 Training for {epochs} epochs (early stopping patience={patience})...")
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            output = model(X_batch).squeeze()
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)
                val_loss += loss.item()
                val_preds.extend((torch.sigmoid(output) > 0.5).cpu().numpy())
                val_labels.extend(y_batch.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        
        # Learning rate scheduler
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f} | Recall: {val_recall:.4f} | LR: {lr:.6f}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    print("\n📊 Evaluating on Test Set...")
    model.eval()
    test_preds = []
    test_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            output = model(X_batch).squeeze()
            probs = torch.sigmoid(output).cpu().numpy()
            test_probs.extend(probs)
            test_preds.extend((probs > 0.5).astype(int))
    
    # Metrics
    accuracy = accuracy_score(y_test, test_preds)
    precision = precision_score(y_test, test_preds, zero_division=0)
    recall = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    cm = confusion_matrix(y_test, test_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print("\n" + "=" * 60)
    print("📈 FINAL RESULTS - OPTIMIZED MODEL")
    print("=" * 60)
    print(f"   Accuracy:           {accuracy:.4f}")
    print(f"   Precision:          {precision:.4f}")
    print(f"   Recall:             {recall:.4f} ← KEY METRIC")
    print(f"   F1 Score:           {f1:.4f}")
    print(f"   False Positive Rate: {fpr:.4f}")
    print(f"   False Negative Rate: {fnr:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {tn:,}  FP: {fp:,}")
    print(f"   FN: {fn:,}  TP: {tp:,}")
    
    # Save model
    model_dir = Path("ml/models/trained")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in format compatible with ml_detector.py
    torch.save({
        'model_state_dict': best_model_state,
        'input_dim': X.shape[1],
        'architecture': 'optimized_512_256_128_64'
    }, model_dir / "attack_detector.pt")
    
    # Save metrics
    results = {
        "samples": int(len(X)),
        "features": int(X.shape[1]),
        "architecture": "optimized_512_256_128_64",
        "class_weight": float(weight_attack),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }
    
    with open(model_dir / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Model saved to {model_dir / 'attack_detector.pt'}")
    print(f"✅ Results saved to {model_dir / 'training_results.json'}")
    
    return results


if __name__ == "__main__":
    main()
