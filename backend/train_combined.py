"""
Combined Dataset Training
Train multi-class classifier on CICIDS 2017 + UNSW-NB15

Expected improvement: More attack types, modern attack patterns
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from pathlib import Path
import json
from datetime import datetime

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# Unified attack class mapping
UNIFIED_CLASSES = {
    # Normal
    "normal": 0, "benign": 0, "BENIGN": 0, "Normal": 0,
    
    # DoS/DDoS
    "dos": 1, "DoS": 1, "ddos": 1, "DDoS": 1,
    "DoS Hulk": 1, "DoS GoldenEye": 1, "DoS slowloris": 1, "DoS Slowhttptest": 1,
    
    # Port Scan / Recon
    "portscan": 2, "PortScan": 2, "Reconnaissance": 2, "reconnaissance": 2,
    "Analysis": 2,
    
    # Brute Force
    "brute_force": 3, "FTP-Patator": 3, "SSH-Patator": 3,
    
    # Web Attack / Exploits
    "web_attack": 4, "Web Attack – Brute Force": 4, "Web Attack – XSS": 4,
    "Web Attack – Sql Injection": 4, "Exploits": 4,
    
    # Infiltration
    "infiltration": 5, "Infiltration": 5,
    
    # Botnet
    "botnet": 6, "Bot": 6,
    
    # Backdoor / Shellcode
    "backdoor": 7, "Backdoor": 7, "Backdoors": 7, "Shellcode": 7,
    
    # Worms
    "worms": 8, "Worms": 8,
    
    # Fuzzers
    "fuzzers": 9, "Fuzzers": 9,
    
    # Generic/Other
    "generic": 10, "Generic": 10, "other": 10,
}

CLASS_NAMES = [
    "Normal", "DoS/DDoS", "Recon/Scan", "Brute Force",
    "Web/Exploit", "Infiltration", "Botnet", "Backdoor",
    "Worms", "Fuzzers", "Other"
]


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
        )
        self.skip = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.net(x) + self.skip(x))


class CombinedClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 11, hidden_dim: int = 256):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim * 2),
            ResidualBlock(hidden_dim * 2, hidden_dim * 2),
            ResidualBlock(hidden_dim * 2, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim // 2),
        )
        
        self.output = nn.Linear(hidden_dim // 2, num_classes)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output(x)


def load_cicids2017():
    """Load CICIDS 2017"""
    path = Path("ml/datasets/cicids2017/CIC-IDS-2017-V2.csv")
    
    if not path.exists():
        print("⚠️ CICIDS 2017 not found")
        return None, None
    
    print("📂 Loading CICIDS 2017...")
    df = pd.read_csv(path, encoding='latin-1', low_memory=False)
    print(f"   Loaded {len(df)} samples")
    
    # Find label column
    label_col = [c for c in df.columns if 'label' in c.lower()][0]
    
    # Map to unified classes
    df['attack_class'] = df[label_col].apply(lambda x: UNIFIED_CLASSES.get(str(x).strip(), 10))
    
    # Get numeric features
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c != 'attack_class']
    
    X = df[feature_cols].values
    y = df['attack_class'].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return X, y, feature_cols


def load_unsw_nb15():
    """Load UNSW-NB15"""
    unsw_dir = Path("ml/datasets/unsw-nb15")
    
    # Try different file names
    possible_files = [
        "UNSW_NB15_training-set.csv",
        "UNSW-NB15_1.csv",
        "UNSW-NB15.csv",
    ]
    
    df = None
    for fname in possible_files:
        path = unsw_dir / fname
        if path.exists():
            print(f"📂 Loading UNSW-NB15 from {fname}...")
            try:
                df = pd.read_csv(path, encoding='latin-1', low_memory=False)
                print(f"   Loaded {len(df)} samples")
                break
            except:
                pass
    
    # Also try to find any CSV in the directory
    if df is None:
        csvs = list(unsw_dir.glob("*.csv"))
        for csv_path in csvs:
            if 'test' not in csv_path.name.lower():
                try:
                    df = pd.read_csv(csv_path, encoding='latin-1', low_memory=False)
                    print(f"📂 Loading UNSW-NB15 from {csv_path.name}...")
                    print(f"   Loaded {len(df)} samples")
                    break
                except:
                    pass
    
    if df is None:
        print("⚠️ UNSW-NB15 not found")
        return None, None, None
    
    # Find label column
    label_cols = [c for c in df.columns if 'attack_cat' in c.lower() or 'label' in c.lower()]
    if label_cols:
        label_col = label_cols[0]
        df['attack_class'] = df[label_col].apply(lambda x: UNIFIED_CLASSES.get(str(x).strip(), 10))
    else:
        df['attack_class'] = 10
    
    # Get numeric features
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in ['attack_class', 'label', 'Label', 'id']]
    
    X = df[feature_cols].values
    y = df['attack_class'].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return X, y, feature_cols


def train_combined():
    """Train on combined dataset"""
    print("=" * 60)
    print("🎯 COMBINED DATASET TRAINING")
    print("=" * 60)
    
    # Load datasets
    X_cicids, y_cicids, _ = load_cicids2017()
    X_unsw, y_unsw, _ = load_unsw_nb15()
    
    # Combine
    datasets = []
    if X_cicids is not None:
        datasets.append((X_cicids, y_cicids, "CICIDS2017"))
    if X_unsw is not None:
        datasets.append((X_unsw, y_unsw, "UNSW-NB15"))
    
    if not datasets:
        print("❌ No datasets found!")
        return
    
    if len(datasets) == 1:
        X, y = datasets[0][0], datasets[0][1]
        print(f"\n📊 Using {datasets[0][2]} only: {len(X)} samples")
    else:
        # Align features to minimum
        min_features = min(d[0].shape[1] for d in datasets)
        print(f"\n📊 Aligning to {min_features} features")
        
        X_list = [d[0][:, :min_features] for d in datasets]
        y_list = [d[1] for d in datasets]
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        print(f"📊 Combined: {len(X)} samples, {X.shape[1]} features")
        for name, (x, _, _) in zip([d[2] for d in datasets], datasets):
            print(f"   - {name}: {len(x)} samples")
    
    # Class distribution
    print("\n📊 Class Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        if cls < len(CLASS_NAMES):
            print(f"   {CLASS_NAMES[cls]}: {cnt:,}")
    
    # Get number of actual classes present
    num_classes = len(unique)
    print(f"\n   Active classes: {num_classes}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    print(f"\n📊 Train: {len(X_train_t)}, Test: {len(X_test_t)}")
    
    # Class weights
    class_counts = np.bincount(y_train, minlength=11)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 11
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Model
    model = CombinedClassifier(
        input_dim=X_train.shape[1],
        num_classes=11,
        hidden_dim=256
    ).to(device)
    
    print(f"🧠 Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Training
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    epochs = 50
    batch_size = 2048
    best_acc = 0
    
    print(f"\n🚀 Training for {epochs} epochs...")
    print("-" * 60)
    
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_t)
            test_preds = test_outputs.argmax(dim=1)
            accuracy = (test_preds == y_test_t).float().mean().item()
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_mean': scaler.mean_,
                'scaler_scale': scaler.scale_,
                'input_dim': X_train.shape[1],
                'num_classes': 11,
                'class_names': CLASS_NAMES
            }, 'ml/models/trained/combined_classifier.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d} | Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Acc: {accuracy*100:.2f}% | Best: {best_acc*100:.2f}%")
    
    print("-" * 60)
    print(f"\n✅ Model saved to ml/models/trained/combined_classifier.pt")
    print(f"   Best accuracy: {best_acc*100:.2f}%")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": float(best_acc),
        "datasets": [d[2] for d in datasets],
        "total_samples": len(X),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X_train.shape[1]
    }
    
    with open('ml/models/trained/combined_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, best_acc


if __name__ == "__main__":
    train_combined()
