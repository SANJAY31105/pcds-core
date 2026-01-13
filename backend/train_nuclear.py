"""
NUCLEAR GPU Training - 100% GPU, 0% CPU
Entire dataset loaded to GPU VRAM - bypasses CPU bottleneck completely

Features:
- ALL data in GPU VRAM (no CPU data loading)
- 100% GPU utilization
- Ultra-fast training (no CPU wait)
- Mixed precision (FP16)
- Best model checkpoint
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from ml.training import CICIDS2017Loader, UNSWNB15Loader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pathlib import Path
import json
import time

# Ultimate network architecture with residual connections
class UltimateAttackClassifier(nn.Module):
    def __init__(self, input_dim: int = 38):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        self.res1 = self._make_residual_block(512, 512)
        self.res2 = self._make_residual_block(512, 256)
        self.res3 = self._make_residual_block(256, 128)
        self.res4 = self._make_residual_block(128, 64)
        
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
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return
    
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    print(f"🚀 GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
    print("=" * 60)
    print("💥 NUCLEAR GPU Training - 100% GPU, 0% CPU")
    print("=" * 60)
    
    # Load datasets (CPU - one time only)
    print("\n📥 Loading datasets to RAM...")
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
    
    # Split data (CPU - one time only)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )
    
    # ===========================================
    # 🚀 NUCLEAR: MOVE ENTIRE DATASET TO GPU VRAM
    # ===========================================
    print("\n💥 MOVING ENTIRE DATASET TO GPU VRAM (Bypassing CPU)...")
    
    X_train_gpu = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_gpu = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_val_gpu = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_gpu = torch.tensor(y_val, dtype=torch.float32, device=device)
    X_test_gpu = torch.tensor(X_test, dtype=torch.float32, device=device)
    
    # Calculate VRAM usage
    vram_used = (X_train_gpu.numel() + y_train_gpu.numel() + X_val_gpu.numel() + 
                 y_val_gpu.numel() + X_test_gpu.numel()) * 4 / 1e9
    print(f"✅ Data loaded to VRAM: {vram_used:.2f} GB")
    print(f"✅ CPU is now FREE - 100% GPU training!")
    
    # Free CPU memory
    del X_train, X_val, X_test, X_trainval, X, X_cic, X_unsw
    
    print(f"📊 Split: Train={len(X_train_gpu):,}, Val={len(X_val_gpu):,}, Test={len(X_test_gpu):,}")
    
    # Create model
    model = UltimateAttackClassifier(input_dim=X_train_gpu.shape[1]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: Ultimate Residual Network ({total_params:,} parameters)")
    
    # Loss with balanced class weights
    weight_attack = 1.5
    pos_weight = torch.tensor([weight_attack], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scaler = GradScaler()
    
    # Training parameters
    epochs = 200
    batch_size = 8192
    patience = 20
    best_f1 = 0
    patience_counter = 0
    best_model_state = None
    best_epoch = 0
    
    num_samples = X_train_gpu.shape[0]
    num_batches = int(np.ceil(num_samples / batch_size))
    
    # OneCycleLR scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=epochs, steps_per_epoch=num_batches
    )
    
    print(f"\n🎓 Training for {epochs} epochs (batch={batch_size}, 100% GPU)...")
    
    for epoch in range(epochs):
        # ===== TRAINING (100% GPU) =====
        model.train()
        train_loss = 0
        
        # Shuffle indices ON GPU (no CPU)
        indices = torch.randperm(num_samples, device=device)
        
        for i in range(num_batches):
            start = i * batch_size
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]
            
            # Direct GPU slicing (ULTRA FAST)
            X_batch = X_train_gpu[batch_idx]
            y_batch = y_train_gpu[batch_idx]
            
            optimizer.zero_grad()
            
            with autocast():
                output = model(X_batch).squeeze()
                loss = criterion(output, y_batch)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
        
        train_loss /= num_batches
        
        # ===== VALIDATION (100% GPU) =====
        model.eval()
        with torch.no_grad():
            with autocast():
                val_output = model(X_val_gpu).squeeze()
            val_preds = (torch.sigmoid(val_output) > 0.5).cpu().numpy()
            val_labels = y_val_gpu.cpu().numpy()
        
        val_acc = accuracy_score(val_labels, val_preds)
        val_recall = recall_score(val_labels, val_preds, zero_division=0)
        val_precision = precision_score(val_labels, val_preds, zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        # Save best model based on F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
        else:
            patience_counter += 1
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Loss: {train_loss:.4f} | Acc: {val_acc:.4f} | Prec: {val_precision:.4f} | Rec: {val_recall:.4f} | F1: {val_f1:.4f}")
        
        if patience_counter >= patience:
            print(f"\n⏹️ Early stopping at epoch {epoch+1} (best F1: {best_f1:.4f} at epoch {best_epoch})")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    model.to(device)
    
    # ===== TEST EVALUATION (100% GPU) =====
    print("\n📊 Evaluating on Test Set...")
    model.eval()
    with torch.no_grad():
        with autocast():
            test_output = model(X_test_gpu).squeeze()
        test_probs = torch.sigmoid(test_output).cpu().numpy()
        test_preds = (test_probs > 0.5).astype(int)
    
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
    print("💥 NUCLEAR GPU MODEL - FINAL RESULTS")
    print("=" * 60)
    print(f"   Training Time:      {elapsed/60:.1f} minutes")
    print(f"   Best Epoch:         {best_epoch}")
    print(f"   Accuracy:           {accuracy:.4f}")
    print(f"   Precision:          {precision:.4f}")
    print(f"   Recall:             {recall:.4f}")
    print(f"   F1 Score:           {f1:.4f}")
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
        'input_dim': X_train_gpu.shape[1],
        'architecture': 'nuclear_gpu_residual'
    }, model_dir / "attack_detector.pt")
    
    results = {
        "samples": int(len(X_train_gpu) + len(X_val_gpu) + len(X_test_gpu)),
        "features": int(X_train_gpu.shape[1]),
        "architecture": "nuclear_gpu_residual",
        "training_mode": "100% GPU",
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
