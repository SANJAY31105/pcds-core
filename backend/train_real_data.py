"""
Train on Real CICIDS + UNSW Datasets
Uses actual labeled attack data for production-grade model
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Tuple
from datetime import datetime
import json


def load_cicids_real(data_dir: str = "ml/datasets/cicids2017") -> Tuple[np.ndarray, np.ndarray]:
    """Load real CICIDS 2017 dataset"""
    print("📂 Loading CICIDS 2017...")
    
    csv_path = Path(data_dir) / "CIC-IDS-2017-V2.csv"
    
    if not csv_path.exists():
        print(f"   ⚠️ File not found: {csv_path}")
        return None, None
    
    try:
        # Read CSV
        df = pd.read_csv(csv_path, encoding='latin-1', low_memory=False)
        print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
        
        # Find label column
        label_col = None
        for col in df.columns:
            if 'label' in col.lower():
                label_col = col
                break
        
        if not label_col:
            print("   ⚠️ No label column found")
            return None, None
        
        # Get labels
        labels = df[label_col].values
        unique_labels = np.unique(labels)
        print(f"   Labels: {len(unique_labels)} classes")
        
        # Create label mapping
        label_map = {
            'BENIGN': 0, 'Normal': 0,
            'DoS': 1, 'DDoS': 1, 'DoS Hulk': 1, 'DoS GoldenEye': 1, 
            'DoS slowloris': 1, 'DoS Slowhttptest': 1, 'Heartbleed': 1,
            'PortScan': 2, 'Portscan': 2,
            'FTP-Patator': 3, 'SSH-Patator': 3, 'Brute Force': 3,
            'Web Attack': 4, 'Web Attack – Brute Force': 4,
            'Web Attack – XSS': 4, 'Web Attack – Sql Injection': 4,
            'Infiltration': 5,
            'Bot': 6, 'Botnet': 6,
        }
        
        # Map labels to class IDs
        y = np.array([label_map.get(str(l).strip(), 10) for l in labels], dtype=np.int64)
        
        # Get numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:40]
        
        if len(numeric_cols) < 40:
            # Pad with zeros
            X = df[numeric_cols].values
            X = np.hstack([X, np.zeros((len(X), 40 - len(numeric_cols)))])
        else:
            X = df[numeric_cols[:40]].values
        
        # Clean and normalize
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Normalize per feature
        for i in range(X.shape[1]):
            col = X[:, i]
            min_val, max_val = col.min(), col.max()
            if max_val > min_val:
                X[:, i] = (col - min_val) / (max_val - min_val)
            else:
                X[:, i] = 0.0
        
        X = np.clip(X, 0, 1).astype(np.float32)
        
        print(f"   ✅ Processed: {X.shape[0]:,} samples, {X.shape[1]} features")
        
        # Class distribution
        unique, counts = np.unique(y, return_counts=True)
        print("   Class distribution:")
        class_names = ["Normal", "DoS/DDoS", "Scan", "Brute Force", "Web Attack", 
                       "Infiltration", "Botnet", "Backdoor", "Worms", "Fuzzers", "Other"]
        for cls, count in zip(unique, counts):
            name = class_names[cls] if cls < len(class_names) else "Other"
            print(f"      {name}: {count:,}")
        
        return X, y
        
    except Exception as e:
        print(f"   ⚠️ Error loading CICIDS: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def load_unsw_real(data_dir: str = "ml/datasets/unsw-nb15") -> Tuple[np.ndarray, np.ndarray]:
    """Load real UNSW-NB15 dataset"""
    print("\n📂 Loading UNSW-NB15...")
    
    data_path = Path(data_dir)
    
    # Try training set first
    train_file = data_path / "UNSW_NB15_training-set.csv"
    
    if train_file.exists():
        try:
            df = pd.read_csv(train_file, low_memory=False)
            print(f"   Loaded training set: {len(df):,} rows")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            return None, None
    else:
        # Try main files
        csv_files = list(data_path.glob("UNSW-NB15_*.csv"))
        if not csv_files:
            print(f"   ⚠️ No UNSW files found")
            return None, None
        
        dfs = []
        for f in csv_files[:2]:  # Limit to first 2 files
            try:
                df_part = pd.read_csv(f, low_memory=False)
                dfs.append(df_part)
                print(f"   Loaded {f.name}: {len(df_part):,} rows")
            except:
                pass
        
        if not dfs:
            return None, None
        
        df = pd.concat(dfs, ignore_index=True)
    
    # Find label column
    label_col = None
    for col in df.columns:
        if 'attack_cat' in col.lower():
            label_col = col
            break
        elif 'label' in col.lower():
            label_col = col
    
    if not label_col:
        # Use binary label
        if 'Label' in df.columns:
            label_col = 'Label'
        else:
            print("   ⚠️ No label column found")
            return None, None
    
    # Label mapping for UNSW
    attack_map = {
        'Normal': 0, 'normal': 0, 0: 0,
        'DoS': 1, 'dos': 1,
        'Reconnaissance': 2, 'reconnaissance': 2, 'Analysis': 2,
        'Shellcode': 4, 'shellcode': 4, 'Exploits': 4, 'exploits': 4,
        'Generic': 5, 'generic': 5,
        'Fuzzers': 9, 'fuzzers': 9,
        'Worms': 8, 'worms': 8,
        'Backdoor': 7, 'backdoor': 7, 'Backdoors': 7,
    }
    
    labels = df[label_col].values
    y = np.array([attack_map.get(str(l).strip(), 10) for l in labels], dtype=np.int64)
    
    # Get numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Remove label columns
    numeric_cols = [c for c in numeric_cols if 'label' not in c.lower() and 'id' not in c.lower()][:40]
    
    if len(numeric_cols) < 40:
        X = df[numeric_cols].values
        X = np.hstack([X, np.zeros((len(X), 40 - len(numeric_cols)))])
    else:
        X = df[numeric_cols[:40]].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Normalize
    for i in range(X.shape[1]):
        col = X[:, i]
        min_val, max_val = col.min(), col.max()
        if max_val > min_val:
            X[:, i] = (col - min_val) / (max_val - min_val)
        else:
            X[:, i] = 0.0
    
    X = np.clip(X, 0, 1).astype(np.float32)
    
    print(f"   ✅ Processed: {X.shape[0]:,} samples, {X.shape[1]} features")
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("   Class distribution:")
    class_names = ["Normal", "DoS/DDoS", "Scan", "Brute Force", "Web Attack", 
                   "Infiltration", "Botnet", "Backdoor", "Worms", "Fuzzers", "Other"]
    for cls, count in zip(unique, counts):
        name = class_names[cls] if cls < len(class_names) else "Other"
        print(f"      {name}: {count:,}")
    
    return X, y


class RealDataClassifier(nn.Module):
    """Classifier optimized for real attack data"""
    
    def __init__(self, input_dim=40, hidden_dim=256, n_classes=11):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_on_real_data():
    """Train on real CICIDS + UNSW datasets"""
    
    print("=" * 70)
    print("🧠 TRAINING ON REAL DATASETS")
    print("   CICIDS 2017 + UNSW-NB15")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load real data
    X_cicids, y_cicids = load_cicids_real()
    X_unsw, y_unsw = load_unsw_real()
    
    # Combine available data
    X_list = []
    y_list = []
    
    if X_cicids is not None:
        X_list.append(X_cicids)
        y_list.append(y_cicids)
    
    if X_unsw is not None:
        X_list.append(X_unsw)
        y_list.append(y_unsw)
    
    if not X_list:
        print("\n⚠️ No real data available!")
        return 0
    
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    
    print(f"\n📊 TOTAL REAL DATA: {len(X):,} samples")
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"   Training:   {len(X_train):,}")
    print(f"   Validation: {len(X_val):,}")
    
    # DataLoaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)
    
    # Model
    model = RealDataClassifier().to(device)
    
    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=11).astype(float)
    class_counts[class_counts == 0] = 1
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * len(weights)
    class_weights = torch.FloatTensor(weights).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    print("\n" + "=" * 70)
    print("🚀 TRAINING")
    print("=" * 70)
    
    best_acc = 0
    epochs = 50
    
    for epoch in range(epochs):
        model.train()
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
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
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "ml/models/trained/real_data_classifier.pt")
            marker = " ★ BEST"
        else:
            marker = ""
        
        if epoch % 5 == 0 or val_acc > best_acc - 0.1:
            print(f"Epoch {epoch+1:2d}/{epochs} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%{marker}")
    
    print("\n" + "=" * 70)
    print(f"✅ TRAINING COMPLETE")
    print(f"   Best Accuracy: {best_acc:.2f}%")
    print(f"   Model: ml/models/trained/real_data_classifier.pt")
    print("=" * 70)
    
    # Save results
    results = {
        "best_accuracy": best_acc,
        "samples": len(X),
        "cicids_samples": len(X_cicids) if X_cicids is not None else 0,
        "unsw_samples": len(X_unsw) if X_unsw is not None else 0,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("ml/models/trained/real_data_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return best_acc


if __name__ == "__main__":
    train_on_real_data()
