"""
ULTIMATE ML Training Script - Maximum GPU Utilization
RTX 4060 8GB VRAM optimization

Features:
- Large batches (8192) for 100% GPU usage
- Balanced class weight (1.5x - optimal for both recall & precision)
- Very deep network with residual connections
- 200 epochs with early stopping
- Mixed precision training (FP16) for speed
- Best model checkpoint saving
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from ml.training import CICIDS2017Loader, UNSWNB15Loader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import json
import time

# Ultimate network architecture with residual connections
class UltimateAttackClassifier(nn.Module):
    """Deep residual network for maximum attack detection accuracy"""
    
    def __init__(self, input_dim: int = 38):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # Residual Block 1: 512 -> 512
        self.res1 = self._make_residual_block(512, 512)
        
        # Residual Block 2: 512 -> 256
        self.res2 = self._make_residual_block(512, 256)
        
        # Residual Block 3: 256 -> 128
        self.res3 = self._make_residual_block(256, 128)
        
        # Residual Block 4: 128 -> 64
        self.res4 = self._make_residual_block(128, 64)
        
        # Output layers
        self.output = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
    
    def _make_residual_block(self, in_dim, out_dim):
        return nn.ModuleDict({
            'main': nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(out_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            ),
            'skip': nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity(),
            'relu': nn.ReLU()
        })
    
    def _apply_residual(self, block, x):
        return block['relu'](block['main'](x) + block['skip'](x))
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self._apply_residual(self.res1, x)
        x = self._apply_residual(self.res2, x)
        x = self._apply_residual(self.res3, x)
        x = self._apply_residual(self.res4, x)
        return self.output(x)


def main():
    start_time = time.time()
    
    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🚀 GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    print("=" * 60)
    print("🏆 ULTIMATE ML Training - Maximum GPU Utilization")
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
    
    # Optimal class weight (balanced between recall and precision)
    weight_attack = 1.5  # Balanced - not too aggressive
    print(f"\n⚖️ Class weight for attacks: {weight_attack}x (balanced)")
    
    # Split data
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )
    print(f"📊 Split: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")
    
    # Create DataLoaders with LARGE batch size for max GPU
    batch_size = 8192  # Maximum for 8GB VRAM
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=True)
    
    # Create model
    model = UltimateAttackClassifier(input_dim=X.shape[1]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: Ultimate Residual Network ({total_params:,} parameters)")
    
    # Loss with balanced class weights
    pos_weight = torch.tensor([weight_attack]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer with OneCycleLR
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Mixed precision training for speed
    scaler = GradScaler()
    
    # Training parameters
    epochs = 200
    patience = 20
    best_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=len(train_loader)
    )
    
    print(f"\n🎓 Training for {epochs} epochs (batch={batch_size}, mixed precision)...")
    
    for epoch in range(epochs):
        # Training with mixed precision
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                with autocast():
                    output = model(X_batch).squeeze()
                val_preds.extend((torch.sigmoid(output) > 0.5).cpu().numpy())
                val_labels.extend(y_batch.numpy())
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        # Save best model based on F1 (balance of precision and recall)
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            patience_counter += 1
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f}")
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1} (best F1: {best_f1:.4f} at epoch {best_epoch})")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    model.to(device)
    
    # Evaluate on test set
    print("\n📊 Evaluating on Test Set...")
    model.eval()
    test_preds = []
    test_probs = []
    with torch.no_grad():
        for i in range(0, len(X_test), batch_size):
            X_batch = torch.FloatTensor(X_test[i:i+batch_size]).to(device)
            with autocast():
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
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("🏆 ULTIMATE MODEL - FINAL RESULTS")
    print("=" * 60)
    print(f"   Training Time:      {elapsed/60:.1f} minutes")
    print(f"   Best Epoch:         {best_epoch}")
    print(f"   Accuracy:           {accuracy:.4f}")
    print(f"   Precision:          {precision:.4f}")
    print(f"   Recall:             {recall:.4f}")
    print(f"   F1 Score:           {f1:.4f} ← OPTIMIZED FOR THIS")
    print(f"   False Positive Rate: {fpr:.4f}")
    print(f"   False Negative Rate: {fnr:.4f}")
    print(f"\n   Confusion Matrix:")
    print(f"   TN: {tn:,}  FP: {fp:,}")
    print(f"   FN: {fn:,}  TP: {tp:,}")
    
    # Save model
    model_dir = Path("ml/models/trained")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        'model_state_dict': best_model_state,
        'input_dim': X.shape[1],
        'architecture': 'ultimate_residual_network'
    }, model_dir / "attack_detector.pt")
    
    # Save metrics
    results = {
        "samples": int(len(X)),
        "features": int(X.shape[1]),
        "architecture": "ultimate_residual_network",
        "class_weight": float(weight_attack),
        "training_time_minutes": round(elapsed/60, 1),
        "best_epoch": best_epoch,
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
