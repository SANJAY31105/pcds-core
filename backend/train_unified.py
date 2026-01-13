"""
Unified ML Training System
Combines Real (60%) + Synthetic (30%) + Online Learning (10%)

Real Data Sources:
- CICIDS 2017
- UNSW-NB15  
- Sysmon logs
- Network traffic logs
- Malware samples (features)

Synthetic Data:
- AI-generated attack variations
- Adversarial examples (perturbations)
- Edge-case attacks
- Novel AI-driven malware behaviors

Online Learning:
- Real environment adaptation
- False positive suppression
- Continuous improvement
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from datetime import datetime
import json


# ============================================
# Data Sources Configuration
# ============================================

DATA_CONFIG = {
    "real_weight": 0.60,      # 60% real data
    "synthetic_weight": 0.30,  # 30% synthetic
    "online_weight": 0.10,     # 10% online learning
    
    "n_features": 40,
    "n_classes": 11,
    "batch_size": 512,
    "epochs": 50,
    
    "class_names": [
        "Normal", "DoS/DDoS", "Recon/Scan", "Brute Force", 
        "Web/Exploit", "Infiltration", "Botnet", "Backdoor",
        "Worms", "Fuzzers", "Other"
    ]
}


# ============================================
# Real Data Loaders
# ============================================

def load_cicids_data(data_dir: str = "ml/datasets/cicids") -> Tuple[np.ndarray, np.ndarray]:
    """Load CICIDS 2017 dataset"""
    print("📂 Loading CICIDS 2017...")
    
    try:
        # Try to load preprocessed
        X = np.load(f"{data_dir}/X_train.npy")
        y = np.load(f"{data_dir}/y_train.npy")
        print(f"   ✅ Loaded {len(X):,} samples")
        return X, y
    except:
        pass
    
    # Try to load from CSV
    csv_path = Path(data_dir)
    csv_files = list(csv_path.glob("*.csv"))
    
    if not csv_files:
        print("   ⚠️ CICIDS not found, generating synthetic equivalent")
        return generate_cicids_synthetic(50000)
    
    dfs = []
    for f in csv_files[:5]:  # Limit files
        try:
            df = pd.read_csv(f, encoding='latin-1', low_memory=False)
            dfs.append(df)
        except:
            pass
    
    if dfs:
        df = pd.concat(dfs, ignore_index=True)
        # Extract features and labels
        X, y = process_cicids_df(df)
        return X, y
    
    return generate_cicids_synthetic(50000)


def load_unsw_data(data_dir: str = "ml/datasets/unsw-nb15") -> Tuple[np.ndarray, np.ndarray]:
    """Load UNSW-NB15 dataset"""
    print("📂 Loading UNSW-NB15...")
    
    csv_path = Path(data_dir)
    csv_files = list(csv_path.glob("*.csv"))
    
    if not csv_files:
        print("   ⚠️ UNSW not found, generating synthetic equivalent")
        return generate_unsw_synthetic(30000)
    
    try:
        df = pd.read_csv(csv_files[0], low_memory=False)
        X, y = process_unsw_df(df)
        print(f"   ✅ Loaded {len(X):,} samples")
        return X, y
    except Exception as e:
        print(f"   ⚠️ Error loading UNSW: {e}")
        return generate_unsw_synthetic(30000)


def load_sysmon_data(data_dir: str = "ml/datasets/real_logs") -> Tuple[np.ndarray, np.ndarray]:
    """Load Sysmon log data"""
    print("📂 Loading Sysmon logs...")
    
    log_path = Path(data_dir)
    json_files = list(log_path.glob("*.json"))
    
    if not json_files:
        print("   ⚠️ Sysmon logs not found, generating synthetic")
        return generate_sysmon_synthetic(10000)
    
    try:
        all_events = []
        for f in json_files:
            with open(f, 'r') as fp:
                data = json.load(fp)
                all_events.extend(data.get("events", []))
        
        X, y = process_sysmon_events(all_events)
        print(f"   ✅ Loaded {len(X):,} events")
        return X, y
    except Exception as e:
        print(f"   ⚠️ Error loading Sysmon: {e}")
        return generate_sysmon_synthetic(10000)


def load_network_data(data_dir: str = "ml/datasets/network") -> Tuple[np.ndarray, np.ndarray]:
    """Load network traffic logs"""
    print("📂 Loading network traffic...")
    
    # Try to find pcap/netflow data
    net_path = Path(data_dir)
    
    if not net_path.exists():
        print("   ⚠️ Network data not found, generating synthetic")
        return generate_network_synthetic(20000)
    
    csv_files = list(net_path.glob("*.csv"))
    if csv_files:
        try:
            df = pd.read_csv(csv_files[0])
            X, y = process_network_df(df)
            print(f"   ✅ Loaded {len(X):,} flows")
            return X, y
        except:
            pass
    
    return generate_network_synthetic(20000)


# ============================================
# Synthetic Data Generators
# ============================================

def generate_cicids_synthetic(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate CICIDS-style traffic data"""
    X = []
    y = []
    
    # Attack type distributions
    types = [
        (0, 0.5, "Normal"),
        (1, 0.15, "DoS"),
        (2, 0.1, "PortScan"),
        (3, 0.1, "Brute Force"),
        (4, 0.05, "Web Attack"),
        (5, 0.05, "Botnet"),
        (7, 0.05, "Infiltration"),
    ]
    
    for class_id, ratio, _ in types:
        n = int(n_samples * ratio)
        for _ in range(n):
            features = generate_traffic_features(class_id)
            X.append(features)
            y.append(class_id)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def generate_unsw_synthetic(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate UNSW-style attack data"""
    X = []
    y = []
    
    types = [
        (0, 0.4, "Normal"),
        (9, 0.15, "Fuzzers"),
        (2, 0.1, "Analysis"),
        (7, 0.1, "Backdoor"),
        (1, 0.1, "DoS"),
        (4, 0.05, "Exploits"),
        (5, 0.05, "Generic"),
        (6, 0.05, "Reconnaissance"),
    ]
    
    for class_id, ratio, _ in types:
        n = int(n_samples * ratio)
        for _ in range(n):
            features = generate_unsw_features(class_id)
            X.append(features)
            y.append(class_id)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def generate_sysmon_synthetic(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate Sysmon event features"""
    X = []
    y = []
    
    for _ in range(n_samples):
        is_attack = random.random() > 0.7
        features = generate_sysmon_features(is_attack)
        X.append(features)
        y.append(4 if is_attack else 0)  # Web/Exploit or Normal
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def generate_network_synthetic(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate network flow features"""
    X = []
    y = []
    
    for _ in range(n_samples):
        is_attack = random.random() > 0.6
        features = generate_flow_features(is_attack)
        X.append(features)
        y.append(random.choice([1, 2, 5, 6]) if is_attack else 0)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def generate_traffic_features(class_id: int) -> np.ndarray:
    """Generate realistic traffic features"""
    features = np.zeros(DATA_CONFIG["n_features"], dtype=np.float32)
    
    if class_id == 0:  # Normal
        features[0] = random.uniform(0.0, 0.2)  # duration
        features[1] = random.uniform(0.0, 0.1)  # packet count
        features[2] = random.uniform(0.0, 0.1)  # bytes
        features[3] = random.choice([80/65535, 443/65535, 53/65535])  # port
    elif class_id == 1:  # DoS
        features[0] = random.uniform(0.0, 0.01)  # short
        features[1] = random.uniform(0.8, 1.0)  # high packets
        features[2] = random.uniform(0.3, 0.7)  # medium bytes
        features[3] = random.uniform(0.0, 0.1)  # low ports
    elif class_id == 2:  # Scan
        features[0] = random.uniform(0.0, 0.001)  # very short
        features[1] = random.uniform(0.0, 0.1)  # few packets
        features[4] = random.uniform(0.7, 1.0)  # many unique ports
    elif class_id == 3:  # Brute Force
        features[0] = random.uniform(0.3, 0.8)  # longer
        features[1] = random.uniform(0.3, 0.6)  # medium packets
        features[5] = random.uniform(0.7, 1.0)  # many attempts
    else:
        features = np.random.uniform(0.2, 0.6, DATA_CONFIG["n_features"]).astype(np.float32)
    
    # Add noise
    features += np.random.normal(0, 0.05, DATA_CONFIG["n_features"]).astype(np.float32)
    features = np.clip(features, 0.0, 1.0)
    
    return features


def generate_unsw_features(class_id: int) -> np.ndarray:
    """Generate UNSW-style features"""
    features = np.zeros(DATA_CONFIG["n_features"], dtype=np.float32)
    
    if class_id == 9:  # Fuzzers
        features[0] = random.uniform(0.5, 1.0)  # High entropy input
        features[1] = random.uniform(0.3, 0.8)  # Varying lengths
        features[2] = random.uniform(0.0, 0.2)  # Low success rate
    elif class_id == 7:  # Backdoor
        features[3] = random.uniform(0.7, 1.0)  # Hidden port
        features[4] = random.uniform(0.4, 0.7)  # Periodic
        features[5] = random.uniform(0.3, 0.6)  # Small data
    else:
        features = generate_traffic_features(class_id)
    
    return features


def generate_sysmon_features(is_attack: bool) -> np.ndarray:
    """Generate Sysmon event features"""
    features = np.zeros(DATA_CONFIG["n_features"], dtype=np.float32)
    
    if is_attack:
        features[0] = random.uniform(0.5, 1.0)  # Suspicious parent
        features[1] = random.uniform(0.6, 1.0)  # Long cmdline
        features[2] = random.uniform(0.4, 0.9)  # Encoded content
        features[3] = random.uniform(0.3, 0.7)  # Unusual path
    else:
        features[0] = random.uniform(0.0, 0.3)
        features[1] = random.uniform(0.0, 0.2)
        features[2] = random.uniform(0.0, 0.1)
    
    features += np.random.normal(0, 0.03, DATA_CONFIG["n_features"]).astype(np.float32)
    return np.clip(features, 0.0, 1.0)


def generate_flow_features(is_attack: bool) -> np.ndarray:
    """Generate network flow features"""
    features = np.zeros(DATA_CONFIG["n_features"], dtype=np.float32)
    
    if is_attack:
        attack_type = random.choice(["c2", "exfil", "scan", "dos"])
        if attack_type == "c2":
            features[0] = random.uniform(0.3, 0.5)  # Periodic
            features[1] = random.uniform(0.1, 0.3)  # Small payloads
            features[6] = random.uniform(0.7, 1.0)  # Beacon pattern
        elif attack_type == "exfil":
            features[2] = random.uniform(0.7, 1.0)  # Large upload
            features[3] = random.uniform(0.4, 0.8)  # Encrypted
        elif attack_type == "scan":
            features[4] = random.uniform(0.8, 1.0)  # Many targets
            features[5] = random.uniform(0.0, 0.1)  # Short connections
        else:  # dos
            features[1] = random.uniform(0.8, 1.0)  # High volume
            features[7] = random.uniform(0.7, 1.0)  # Fast rate
    else:
        features[:5] = np.random.uniform(0.0, 0.2, 5).astype(np.float32)
    
    return np.clip(features, 0.0, 1.0)


# ============================================
# AI-Generated & Adversarial Data
# ============================================

def generate_adversarial_examples(X: np.ndarray, y: np.ndarray, 
                                  epsilon: float = 0.1,
                                  n_samples: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate adversarial examples with perturbations"""
    print(f"🔧 Generating {n_samples:,} adversarial examples...")
    
    X_adv = []
    y_adv = []
    
    attack_indices = np.where(y > 0)[0]
    
    for _ in range(n_samples):
        idx = random.choice(attack_indices)
        x = X[idx].copy()
        
        # Random perturbation
        perturbation = np.random.uniform(-epsilon, epsilon, x.shape)
        x_perturbed = np.clip(x + perturbation, 0, 1)
        
        X_adv.append(x_perturbed)
        y_adv.append(y[idx])
    
    return np.array(X_adv, dtype=np.float32), np.array(y_adv, dtype=np.int64)


def generate_edge_case_attacks(n_samples: int = 20000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate edge-case attack patterns"""
    print(f"🔧 Generating {n_samples:,} edge-case attacks...")
    
    X = []
    y = []
    
    edge_cases = [
        # Low-and-slow attacks
        {"type": "slow_scan", "class": 2, "features": {"rate": 0.01, "packets": 0.05}},
        {"type": "slow_brute", "class": 3, "features": {"rate": 0.02, "attempts": 0.1}},
        
        # Encrypted/obfuscated
        {"type": "encrypted_c2", "class": 6, "features": {"entropy": 0.95, "size": 0.3}},
        {"type": "encoded_malware", "class": 7, "features": {"base64": 0.9, "length": 0.8}},
        
        # Mimicry attacks (look like normal traffic)
        {"type": "http_mimicry", "class": 4, "features": {"port": 80/65535, "normal_pattern": 0.7}},
        {"type": "dns_mimicry", "class": 6, "features": {"port": 53/65535, "tunnel": 0.8}},
        
        # Polymorphic patterns
        {"type": "polymorphic_scan", "class": 2, "features": {"variation": 0.9}},
        {"type": "polymorphic_dos", "class": 1, "features": {"variation": 0.85}},
    ]
    
    samples_per_case = n_samples // len(edge_cases)
    
    for case in edge_cases:
        for _ in range(samples_per_case):
            features = np.random.uniform(0.1, 0.4, DATA_CONFIG["n_features"]).astype(np.float32)
            
            # Apply case-specific features
            idx = 0
            for name, value in case["features"].items():
                features[idx] = value + random.uniform(-0.1, 0.1)
                idx += 1
            
            features = np.clip(features, 0.0, 1.0)
            X.append(features)
            y.append(case["class"])
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def generate_novel_malware_behaviors(n_samples: int = 15000) -> Tuple[np.ndarray, np.ndarray]:
    """Generate novel AI-driven malware behaviors"""
    print(f"🔧 Generating {n_samples:,} novel malware patterns...")
    
    X = []
    y = []
    
    novel_behaviors = [
        # AI-evading patterns
        ("adaptive_evasion", 7, {"adapt_rate": 0.8, "detect_avoid": 0.9}),
        ("ml_poisoning", 10, {"gradual_drift": 0.7, "feature_shift": 0.6}),
        
        # Advanced persistence
        ("fileless_persist", 7, {"memory_only": 0.95, "registry_minimal": 0.3}),
        ("living_off_land", 4, {"lolbins": 0.9, "normal_tools": 0.85}),
        
        # AI-generated C2
        ("ai_beacon", 6, {"adaptive_timing": 0.8, "traffic_blend": 0.75}),
        ("domain_gen", 6, {"dga_entropy": 0.85, "random_subdomains": 0.9}),
        
        # Supply chain patterns
        ("supply_chain", 5, {"trusted_source": 0.7, "delayed_payload": 0.6}),
        ("dependency_confusion", 5, {"package_spoof": 0.8, "typosquat": 0.7}),
    ]
    
    samples_per_behavior = n_samples // len(novel_behaviors)
    
    for name, class_id, params in novel_behaviors:
        for _ in range(samples_per_behavior):
            features = np.random.uniform(0.2, 0.5, DATA_CONFIG["n_features"]).astype(np.float32)
            
            idx = 0
            for param, value in params.items():
                features[idx] = value + random.gauss(0, 0.1)
                idx += 1
            
            features = np.clip(features, 0.0, 1.0)
            X.append(features)
            y.append(class_id)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ============================================
# Online Learning Component
# ============================================

class OnlineLearningAdapter:
    """Adapts model based on real environment feedback"""
    
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.fp_buffer = []  # False positive buffer
        self.fn_buffer = []  # False negative buffer
        self.adaptation_threshold = 100
    
    def record_false_positive(self, features: np.ndarray, predicted_class: int):
        """Record a false positive for adaptation"""
        self.fp_buffer.append((features, 0))  # Should be normal
        
        if len(self.fp_buffer) >= self.adaptation_threshold:
            self._adapt_model()
    
    def record_false_negative(self, features: np.ndarray, true_class: int):
        """Record a false negative for adaptation"""
        self.fn_buffer.append((features, true_class))
        
        if len(self.fn_buffer) >= self.adaptation_threshold:
            self._adapt_model()
    
    def _adapt_model(self):
        """Perform one adaptation step"""
        if not self.fp_buffer and not self.fn_buffer:
            return
        
        # Combine buffers
        X_adapt = []
        y_adapt = []
        
        for features, label in self.fp_buffer + self.fn_buffer:
            X_adapt.append(features)
            y_adapt.append(label)
        
        if len(X_adapt) < 10:
            return
        
        X_adapt = torch.FloatTensor(np.array(X_adapt)).to(self.device)
        y_adapt = torch.LongTensor(np.array(y_adapt)).to(self.device)
        
        # Fine-tune with low learning rate
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for _ in range(5):  # Few adaptation steps
            optimizer.zero_grad()
            outputs = self.model(X_adapt)
            loss = criterion(outputs, y_adapt)
            loss.backward()
            optimizer.step()
        
        # Clear buffers
        self.fp_buffer = []
        self.fn_buffer = []
        
        print(f"   ✅ Model adapted with {len(X_adapt)} samples")
    
    def generate_adaptation_samples(self, n_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate samples that simulate online learning scenarios"""
        X = []
        y = []
        
        # FP-like samples (attacks that look normal)
        for _ in range(n_samples // 2):
            features = np.random.uniform(0.0, 0.3, DATA_CONFIG["n_features"]).astype(np.float32)
            features[random.randint(0, 5)] = random.uniform(0.4, 0.6)  # Slight anomaly
            X.append(features)
            y.append(0)  # Normal
        
        # FN-like samples (normal that looks like attacks)
        for _ in range(n_samples // 2):
            features = np.random.uniform(0.3, 0.6, DATA_CONFIG["n_features"]).astype(np.float32)
            features[random.randint(0, 5)] = random.uniform(0.0, 0.2)  # Normal indicator
            X.append(features)
            y.append(random.randint(1, 10))  # Attack
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


# ============================================
# Helper Functions for Real Data Processing
# ============================================

def process_cicids_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Process CICIDS DataFrame"""
    n_features = DATA_CONFIG["n_features"]
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:n_features]
    
    if len(numeric_cols) < n_features:
        # Pad with zeros
        X = df[numeric_cols].values
        X = np.hstack([X, np.zeros((len(X), n_features - len(numeric_cols)))])
    else:
        X = df[numeric_cols[:n_features]].values
    
    # Normalize
    X = np.nan_to_num(X, 0)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    X = np.clip(X, 0, 1).astype(np.float32)
    
    # Labels
    label_col = [c for c in df.columns if 'label' in c.lower()]
    if label_col:
        y = (df[label_col[0]] != 'BENIGN').astype(int).values
    else:
        y = np.zeros(len(X), dtype=np.int64)
    
    return X, y


def process_unsw_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Process UNSW DataFrame"""
    n_features = DATA_CONFIG["n_features"]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:n_features]
    
    if len(numeric_cols) < n_features:
        X = df[numeric_cols].values
        X = np.hstack([X, np.zeros((len(X), n_features - len(numeric_cols)))])
    else:
        X = df[numeric_cols[:n_features]].values
    
    X = np.nan_to_num(X, 0)
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    X = np.clip(X, 0, 1).astype(np.float32)
    
    # Labels
    if 'label' in df.columns:
        y = df['label'].values.astype(np.int64)
    elif 'attack_cat' in df.columns:
        y = (df['attack_cat'] != 'Normal').astype(int).values
    else:
        y = np.zeros(len(X), dtype=np.int64)
    
    return X, y


def process_sysmon_events(events: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
    """Process Sysmon events to features"""
    n_features = DATA_CONFIG["n_features"]
    X = []
    y = []
    
    for event in events:
        features = np.zeros(n_features, dtype=np.float32)
        
        # Extract features from event
        if 'process_name' in event:
            features[0] = 1.0 if 'powershell' in event['process_name'].lower() else 0.0
            features[1] = 1.0 if 'cmd' in event['process_name'].lower() else 0.0
        
        if 'command_line' in event:
            cmdline = event.get('command_line', '')
            features[2] = min(len(cmdline) / 1000, 1.0)
            features[3] = 1.0 if '-enc' in cmdline.lower() else 0.0
        
        X.append(features)
        y.append(0)  # Default to normal, would need labeled data
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def process_network_df(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Process network traffic DataFrame"""
    return process_cicids_df(df)  # Similar processing


# ============================================
# Unified Model Architecture
# ============================================

class UnifiedAttackClassifier(nn.Module):
    """Unified classifier for all attack types"""
    
    def __init__(self, input_dim: int = 40, hidden_dim: int = 256, n_classes: int = 11):
        super().__init__()
        
        # Feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Residual block
        self.residual = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes)
        )
    
    def forward(self, x):
        features = self.feature_net(x)
        residual = self.residual(features)
        features = features + residual  # Skip connection
        return self.classifier(features)


# ============================================
# Main Training Function
# ============================================

def train_unified_model():
    """Train the unified model with all data sources"""
    
    print("=" * 70)
    print("🧠 UNIFIED ML TRAINING SYSTEM")
    print("   Real (60%) + Synthetic (30%) + Online Learning (10%)")
    print("=" * 70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # ========================================
    # Load Real Data (60%)
    # ========================================
    print("\n" + "=" * 50)
    print("📊 LOADING REAL DATA (60%)")
    print("=" * 50)
    
    X_cicids, y_cicids = load_cicids_data()
    X_unsw, y_unsw = load_unsw_data()
    X_sysmon, y_sysmon = load_sysmon_data()
    X_network, y_network = load_network_data()
    
    # Combine real data
    X_real = np.concatenate([X_cicids, X_unsw, X_sysmon, X_network])
    y_real = np.concatenate([y_cicids, y_unsw, y_sysmon, y_network])
    
    print(f"\n📊 Total Real Data: {len(X_real):,} samples")
    
    # ========================================
    # Generate Synthetic Data (30%)
    # ========================================
    print("\n" + "=" * 50)
    print("🔧 GENERATING SYNTHETIC DATA (30%)")
    print("=" * 50)
    
    n_synthetic = int(len(X_real) * DATA_CONFIG["synthetic_weight"] / DATA_CONFIG["real_weight"])
    
    # Generate adversarial examples
    X_adv, y_adv = generate_adversarial_examples(X_real, y_real, n_samples=n_synthetic // 3)
    
    # Generate edge cases
    X_edge, y_edge = generate_edge_case_attacks(n_samples=n_synthetic // 3)
    
    # Generate novel malware
    X_novel, y_novel = generate_novel_malware_behaviors(n_samples=n_synthetic // 3)
    
    # Combine synthetic
    X_synthetic = np.concatenate([X_adv, X_edge, X_novel])
    y_synthetic = np.concatenate([y_adv, y_edge, y_novel])
    
    print(f"\n📊 Total Synthetic Data: {len(X_synthetic):,} samples")
    
    # ========================================
    # Generate Online Learning Data (10%)
    # ========================================
    print("\n" + "=" * 50)
    print("🔄 GENERATING ONLINE LEARNING DATA (10%)")
    print("=" * 50)
    
    n_online = int(len(X_real) * DATA_CONFIG["online_weight"] / DATA_CONFIG["real_weight"])
    
    adapter = OnlineLearningAdapter(None, device)
    X_online, y_online = adapter.generate_adaptation_samples(n_samples=n_online)
    
    print(f"\n📊 Total Online Learning Data: {len(X_online):,} samples")
    
    # ========================================
    # Combine All Data
    # ========================================
    print("\n" + "=" * 50)
    print("🔗 COMBINING ALL DATA")
    print("=" * 50)
    
    X_all = np.concatenate([X_real, X_synthetic, X_online])
    y_all = np.concatenate([y_real, y_synthetic, y_online])
    
    print(f"\n📊 DATASET COMPOSITION:")
    print(f"   Real Data:     {len(X_real):,} ({100*len(X_real)/len(X_all):.1f}%)")
    print(f"   Synthetic:     {len(X_synthetic):,} ({100*len(X_synthetic)/len(X_all):.1f}%)")
    print(f"   Online Learn:  {len(X_online):,} ({100*len(X_online)/len(X_all):.1f}%)")
    print(f"   TOTAL:         {len(X_all):,} samples")
    
    # Shuffle
    indices = np.random.permutation(len(X_all))
    X_all = X_all[indices]
    y_all = y_all[indices]
    
    # Class distribution
    print(f"\n📊 Class Distribution:")
    unique, counts = np.unique(y_all, return_counts=True)
    for cls, count in zip(unique, counts):
        name = DATA_CONFIG["class_names"][cls] if cls < len(DATA_CONFIG["class_names"]) else "Other"
        print(f"   {name}: {count:,} ({100*count/len(y_all):.1f}%)")
    
    # ========================================
    # Create DataLoaders
    # ========================================
    split_idx = int(0.8 * len(X_all))
    X_train, X_val = X_all[:split_idx], X_all[split_idx:]
    y_train, y_val = y_all[:split_idx], y_all[split_idx:]
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=DATA_CONFIG["batch_size"], 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG["batch_size"], 
                           shuffle=False, num_workers=0)
    
    print(f"\n📊 Training: {len(X_train):,} | Validation: {len(X_val):,}")
    
    # ========================================
    # Create Model
    # ========================================
    model = UnifiedAttackClassifier(
        input_dim=DATA_CONFIG["n_features"],
        n_classes=DATA_CONFIG["n_classes"]
    ).to(device)
    
    # Try to load pretrained weights
    try:
        checkpoint = torch.load("ml/models/trained/synthetic_classifier.pt", 
                               map_location=device, weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        print("   ✅ Loaded pretrained weights from synthetic_classifier.pt")
    except:
        print("   ⚠️ Training from scratch")
    
    # ========================================
    # Training
    # ========================================
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DATA_CONFIG["epochs"])
    
    print("\n" + "=" * 70)
    print("🚀 TRAINING")
    print("=" * 70)
    
    best_acc = 0
    
    for epoch in range(DATA_CONFIG["epochs"]):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
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
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "ml/models/trained/unified_classifier.pt")
            marker = " ★ BEST"
        else:
            marker = ""
        
        if epoch % 5 == 0 or val_acc > best_acc - 0.01:
            print(f"Epoch {epoch+1:2d}/{DATA_CONFIG['epochs']} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%{marker}")
    
    print("\n" + "=" * 70)
    print(f"✅ TRAINING COMPLETE")
    print(f"   Best Validation Accuracy: {best_acc:.2f}%")
    print(f"   Model saved: ml/models/trained/unified_classifier.pt")
    print("=" * 70)
    
    # Save training results
    results = {
        "best_accuracy": best_acc,
        "total_samples": len(X_all),
        "real_samples": len(X_real),
        "synthetic_samples": len(X_synthetic),
        "online_samples": len(X_online),
        "epochs": DATA_CONFIG["epochs"],
        "timestamp": datetime.now().isoformat()
    }
    
    with open("ml/models/trained/unified_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return best_acc


if __name__ == "__main__":
    train_unified_model()
