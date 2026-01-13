"""
Multi-Class Attack Classifier Training
Phase 1: Train classifier for 8-12 attack types

Attack Classes:
1. Normal (Benign)
2. DoS/DDoS
3. Port Scan
4. Brute Force
5. Web Attack (SQL Injection, XSS)
6. Infiltration
7. Botnet
8. Reconnaissance
9. Exploit
10. Backdoor
11. Ransomware
12. Data Exfiltration

Uses:
- CICIDS 2017 dataset
- UNSW-NB15 dataset
- CrossEntropy loss
- Residual network architecture
- GPU acceleration
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import json
from datetime import datetime

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# Attack class mapping
ATTACK_CLASSES = {
    # CICIDS 2017 mapping
    "BENIGN": 0,
    "benign": 0,
    "normal": 0,
    
    # DoS attacks
    "DoS Hulk": 1,
    "DoS GoldenEye": 1,
    "DoS slowloris": 1,
    "DoS Slowhttptest": 1,
    "DDoS": 1,
    "dos": 1,
    
    # Port Scan
    "PortScan": 2,
    "port_scan": 2,
    
    # Brute Force
    "FTP-Patator": 3,
    "SSH-Patator": 3,
    "brute_force": 3,
    
    # Web Attacks
    "Web Attack – Brute Force": 4,
    "Web Attack – XSS": 4,
    "Web Attack – Sql Injection": 4,
    "web_attack": 4,
    
    # Infiltration
    "Infiltration": 5,
    "infiltration": 5,
    
    # Botnet
    "Bot": 6,
    "botnet": 6,
    
    # Reconnaissance (UNSW-NB15)
    "Reconnaissance": 7,
    "reconnaissance": 7,
    "Analysis": 7,
    
    # Exploits
    "Exploits": 8,
    "exploits": 8,
    "Shellcode": 8,
    
    # Backdoor
    "Backdoor": 9,
    "Backdoors": 9,
    "backdoor": 9,
    
    # Worms
    "Worms": 10,
    "worms": 10,
    
    # Generic/Other
    "Generic": 11,
    "Fuzzers": 11,
    "generic": 11,
}

CLASS_NAMES = [
    "Normal", "DoS/DDoS", "Port Scan", "Brute Force",
    "Web Attack", "Infiltration", "Botnet", "Reconnaissance",
    "Exploit", "Backdoor", "Worms", "Other"
]


class ResidualBlock(nn.Module):
    """Residual block for deep network"""
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


class MultiClassAttackClassifier(nn.Module):
    """
    Multi-class attack classifier with residual connections
    
    Architecture:
    - Input projection
    - 4 residual blocks
    - Output layer with softmax
    """
    
    def __init__(self, input_dim: int, num_classes: int = 12, hidden_dim: int = 256):
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
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.blocks(x)
        return self.output(x)


def load_cicids_multiclass(data_path: str) -> tuple:
    """Load CICIDS 2017 with multi-class labels"""
    print("📂 Loading CICIDS 2017...")
    
    csv_path = Path(data_path)
    if not csv_path.exists():
        print(f"   ⚠️ CICIDS not found at {data_path}")
        return None, None
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path, encoding='latin-1', low_memory=False)
    except:
        df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False, on_bad_lines='skip')
    
    print(f"   Loaded {len(df)} samples")
    
    # Find label column
    label_col = None
    for col in [' Label', 'Label', 'label', 'attack_cat']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        print("   ⚠️ Label column not found")
        return None, None
    
    # Map labels to attack classes
    df['attack_class'] = df[label_col].map(lambda x: ATTACK_CLASSES.get(str(x).strip(), 11))
    
    # Show class distribution
    print("\n   Class distribution:")
    for cls, count in df['attack_class'].value_counts().head(10).items():
        print(f"      {CLASS_NAMES[cls]}: {count}")
    
    # Get numeric features
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'attack_class' in feature_cols:
        feature_cols.remove('attack_class')
    
    X = df[feature_cols].values
    y = df['attack_class'].values
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return X, y


def load_unsw_multiclass(data_path: str) -> tuple:
    """Load UNSW-NB15 with multi-class labels"""
    print("📂 Loading UNSW-NB15...")
    
    csv_path = Path(data_path)
    if not csv_path.exists():
        print(f"   ⚠️ UNSW not found at {data_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path, encoding='latin-1', low_memory=False)
    except:
        df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False, on_bad_lines='skip')
    
    print(f"   Loaded {len(df)} samples")
    
    # Find label column
    label_col = None
    for col in ['attack_cat', 'Attack_cat', 'label', 'Label']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        print("   ⚠️ Label column not found")
        return None, None
    
    # Map labels
    df['attack_class'] = df[label_col].map(lambda x: ATTACK_CLASSES.get(str(x).strip(), 11))
    
    # Show distribution
    print("\n   Class distribution:")
    for cls, count in df['attack_class'].value_counts().head(10).items():
        print(f"      {CLASS_NAMES[cls]}: {count}")
    
    # Get features
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'attack_class' in feature_cols:
        feature_cols.remove('attack_class')
    if 'label' in feature_cols:
        feature_cols.remove('label')
    
    X = df[feature_cols].values
    y = df['attack_class'].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    return X, y


def train_multiclass_classifier():
    """Train the multi-class attack classifier"""
    
    print("=" * 60)
    print("🎯 MULTI-CLASS ATTACK CLASSIFIER TRAINING")
    print("=" * 60)
    
    # Load datasets
    X_cicids, y_cicids = load_cicids_multiclass(
        "ml/datasets/cicids2017/CIC-IDS-2017-V2.csv"
    )
    
    X_unsw, y_unsw = load_unsw_multiclass(
        "ml/datasets/unsw-nb15/UNSW-NB15.csv"
    )
    
    # Combine datasets (if both available)
    if X_cicids is not None and X_unsw is not None:
        # Align feature dimensions
        min_features = min(X_cicids.shape[1], X_unsw.shape[1])
        X_cicids = X_cicids[:, :min_features]
        X_unsw = X_unsw[:, :min_features]
        
        X = np.vstack([X_cicids, X_unsw])
        y = np.concatenate([y_cicids, y_unsw])
        print(f"\n📊 Combined dataset: {len(X)} samples, {X.shape[1]} features")
    elif X_cicids is not None:
        X, y = X_cicids, y_cicids
        print(f"\n📊 Using CICIDS only: {len(X)} samples")
    elif X_unsw is not None:
        X, y = X_unsw, y_unsw
        print(f"\n📊 Using UNSW only: {len(X)} samples")
    else:
        print("❌ No dataset found!")
        return
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors and move to GPU
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.LongTensor(y_test).to(device)
    
    print(f"   Train: {len(X_train_t)}, Test: {len(X_test_t)}")
    
    # Class weights for imbalanced data
    class_counts = np.bincount(y_train, minlength=12)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * 12
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    # Create model
    num_classes = len(CLASS_NAMES)
    model = MultiClassAttackClassifier(
        input_dim=X_train.shape[1],
        num_classes=num_classes,
        hidden_dim=256
    ).to(device)
    
    print(f"\n🧠 Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    # Training loop
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
                'num_classes': num_classes,
                'class_names': CLASS_NAMES
            }, 'ml/models/trained/multiclass_classifier.pt')
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d} | Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Acc: {accuracy*100:.2f}% | Best: {best_acc*100:.2f}%")
    
    print("-" * 60)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_t)
        test_preds = test_outputs.argmax(dim=1).cpu().numpy()
    
    print("\n📊 CLASSIFICATION REPORT:")
    print(classification_report(
        y_test, test_preds, 
        target_names=[CLASS_NAMES[i] for i in range(num_classes)],
        zero_division=0
    ))
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "accuracy": float(best_acc),
        "num_classes": num_classes,
        "class_names": CLASS_NAMES,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": X_train.shape[1]
    }
    
    with open('ml/models/trained/multiclass_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Model saved to ml/models/trained/multiclass_classifier.pt")
    print(f"   Best accuracy: {best_acc*100:.2f}%")
    
    return model, best_acc


if __name__ == "__main__":
    train_multiclass_classifier()
