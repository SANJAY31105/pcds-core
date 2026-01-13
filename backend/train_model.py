"""
GPU-Accelerated ML Training Script
Trains on CIC-IDS 2017 + UNSW-NB15 datasets with:
- GPU acceleration (RTX 4060)
- Class weights for imbalanced data (improves recall)
- 50 epochs (vs 30 previously)
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import torch
from ml.training import CICIDS2017Loader, UNSWNB15Loader
from ml.training.attack_classifiers import create_attack_classifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import json

def main():
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU Available: {gpu_name}")
    else:
        device = torch.device("cpu")
        print("⚠️ GPU not available, using CPU")
    
    print("=" * 60)
    print("🚀 PCDS ML Model Training (GPU-Accelerated)")
    print("=" * 60)
    
    # Load datasets
    print("\n📥 Loading CIC-IDS 2017...")
    try:
        cic_loader = CICIDS2017Loader()
        X_cic, y_cic, _ = cic_loader.load()
        print(f"   Loaded: {X_cic.shape}")
    except Exception as e:
        print(f"   ⚠️ CIC-IDS error: {e}")
        X_cic, y_cic = None, None
    
    print("\n📥 Loading UNSW-NB15...")
    try:
        unsw_loader = UNSWNB15Loader()
        X_unsw, y_unsw, _ = unsw_loader.load()
        print(f"   Loaded: {X_unsw.shape}")
    except Exception as e:
        print(f"   ⚠️ UNSW error: {e}")
        X_unsw, y_unsw = None, None
    
    # Combine datasets (match feature dimensions)
    if X_cic is not None and X_unsw is not None:
        # Pad smaller to match larger feature count
        max_features = max(X_cic.shape[1], X_unsw.shape[1])
        
        if X_cic.shape[1] < max_features:
            X_cic = np.pad(X_cic, ((0,0), (0, max_features - X_cic.shape[1])))
        if X_unsw.shape[1] < max_features:
            X_unsw = np.pad(X_unsw, ((0,0), (0, max_features - X_unsw.shape[1])))
        
        X = np.vstack([X_cic, X_unsw])
        y = np.concatenate([y_cic, y_unsw])
    elif X_cic is not None:
        X, y = X_cic, y_cic
    elif X_unsw is not None:
        X, y = X_unsw, y_unsw
    else:
        print("❌ No datasets loaded!")
        return
    
    print(f"\n📊 Combined Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Attack samples: {sum(y)} ({100*sum(y)/len(y):.1f}%)")
    print(f"   Benign samples: {len(y) - sum(y)} ({100*(len(y)-sum(y))/len(y):.1f}%)")
    
    # Split data
    print("\n📊 Creating train/val/test splits (60/20/20)...")
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )
    
    print(f"   Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train classifier
    print("\n🎓 Training Binary Attack Classifier...")
    classifier = create_attack_classifier("attack_detector", input_dim=X.shape[1])
    
    metrics = classifier.train(X_train, y_train, X_val, y_val, epochs=50)
    
    # Evaluate
    print("\n📊 Evaluating on Test Set...")
    y_pred = classifier.predict(X_test)
    y_proba = classifier.predict_proba(X_test)
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print("\n" + "=" * 60)
    print("📈 TRAINING RESULTS")
    print("=" * 60)
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
    
    classifier.save(str(model_dir / "attack_detector.pt"))
    
    # Save metrics
    results = {
        "samples": int(len(X)),
        "features": int(X.shape[1]),
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
