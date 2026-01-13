"""
Synthetic Attack Data Generator
Generates realistic attack patterns for ML training

Attack Types:
- APT Kill Chains (multi-stage attacks)
- Evasion Techniques (encoded, obfuscated)
- Modern Attack Vectors (supply chain, cloud)
- MITRE ATT&CK coverage
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random
from datetime import datetime


@dataclass
class AttackPattern:
    """Attack pattern definition"""
    name: str
    category: str
    mitre_tactics: List[str]
    feature_ranges: Dict[str, Tuple[float, float]]


# Attack pattern definitions
ATTACK_PATTERNS = [
    # APT Kill Chain - Reconnaissance
    AttackPattern(
        name="port_scan_aggressive",
        category="Recon/Scan",
        mitre_tactics=["TA0043"],
        feature_ranges={
            "dst_port": (1, 1024),
            "duration": (0.001, 0.1),
            "packets": (1, 5),
            "bytes": (40, 200),
        }
    ),
    AttackPattern(
        name="service_enumeration",
        category="Recon/Scan",
        mitre_tactics=["TA0043"],
        feature_ranges={
            "dst_port": (20, 445),
            "duration": (0.1, 2.0),
            "packets": (5, 50),
        }
    ),
    
    # APT Kill Chain - Initial Access
    AttackPattern(
        name="brute_force_ssh",
        category="Brute Force",
        mitre_tactics=["TA0001", "TA0006"],
        feature_ranges={
            "dst_port": (22, 22),
            "duration": (0.5, 30),
            "packets": (100, 10000),
            "failed_logins": (10, 1000),
        }
    ),
    AttackPattern(
        name="brute_force_rdp",
        category="Brute Force",
        mitre_tactics=["TA0001", "TA0006"],
        feature_ranges={
            "dst_port": (3389, 3389),
            "duration": (1.0, 60),
            "packets": (200, 20000),
        }
    ),
    AttackPattern(
        name="credential_stuffing",
        category="Brute Force",
        mitre_tactics=["TA0006"],
        feature_ranges={
            "dst_port": (80, 443),
            "http_requests": (100, 5000),
            "unique_users": (50, 500),
        }
    ),
    
    # APT Kill Chain - Execution
    AttackPattern(
        name="powershell_encoded",
        category="Web/Exploit",
        mitre_tactics=["TA0002"],
        feature_ranges={
            "cmdline_length": (200, 5000),
            "entropy": (4.5, 6.0),
            "base64_ratio": (0.7, 1.0),
        }
    ),
    AttackPattern(
        name="dll_injection",
        category="Infiltration",
        mitre_tactics=["TA0002", "TA0005"],
        feature_ranges={
            "parent_child_mismatch": (0.8, 1.0),
            "suspicious_api_calls": (5, 50),
        }
    ),
    
    # APT Kill Chain - Persistence
    AttackPattern(
        name="registry_persistence",
        category="Backdoor",
        mitre_tactics=["TA0003"],
        feature_ranges={
            "registry_writes": (1, 10),
            "run_key_modified": (0.9, 1.0),
        }
    ),
    AttackPattern(
        name="scheduled_task",
        category="Backdoor",
        mitre_tactics=["TA0003"],
        feature_ranges={
            "task_created": (1, 1),
            "suspicious_trigger": (0.7, 1.0),
        }
    ),
    
    # APT Kill Chain - C2
    AttackPattern(
        name="dns_tunneling",
        category="Botnet",
        mitre_tactics=["TA0011"],
        feature_ranges={
            "dns_query_length": (50, 253),
            "dns_requests_per_min": (10, 1000),
            "unique_subdomains": (100, 10000),
        }
    ),
    AttackPattern(
        name="http_beacon",
        category="Botnet",
        mitre_tactics=["TA0011"],
        feature_ranges={
            "beacon_interval": (30, 300),
            "jitter": (0.1, 0.3),
            "c2_connections": (10, 1000),
        }
    ),
    
    # DoS Attacks
    AttackPattern(
        name="syn_flood",
        category="DoS/DDoS",
        mitre_tactics=["TA0040"],
        feature_ranges={
            "syn_packets": (10000, 1000000),
            "src_ips": (1, 10000),
            "duration": (1, 600),
        }
    ),
    AttackPattern(
        name="http_flood",
        category="DoS/DDoS",
        mitre_tactics=["TA0040"],
        feature_ranges={
            "http_requests": (1000, 100000),
            "request_rate": (100, 10000),
        }
    ),
    AttackPattern(
        name="slowloris",
        category="DoS/DDoS",
        mitre_tactics=["TA0040"],
        feature_ranges={
            "connections": (100, 10000),
            "partial_requests": (0.8, 1.0),
            "duration": (60, 3600),
        }
    ),
    
    # Lateral Movement
    AttackPattern(
        name="smb_lateral",
        category="Worms",
        mitre_tactics=["TA0008"],
        feature_ranges={
            "smb_connections": (5, 100),
            "dst_port": (445, 445),
            "internal_targets": (2, 50),
        }
    ),
    AttackPattern(
        name="wmi_execution",
        category="Worms",
        mitre_tactics=["TA0008"],
        feature_ranges={
            "wmi_calls": (10, 100),
            "remote_hosts": (2, 20),
        }
    ),
    
    # Data Exfiltration
    AttackPattern(
        name="large_upload",
        category="Infiltration",
        mitre_tactics=["TA0010"],
        feature_ranges={
            "upload_bytes": (10000000, 1000000000),
            "dst_port": (443, 443),
            "duration": (60, 3600),
        }
    ),
    AttackPattern(
        name="icmp_exfil",
        category="Infiltration",
        mitre_tactics=["TA0010"],
        feature_ranges={
            "icmp_data_bytes": (1000, 100000),
            "icmp_packets": (100, 10000),
        }
    ),
    
    # Malware Patterns
    AttackPattern(
        name="ransomware_behavior",
        category="Backdoor",
        mitre_tactics=["TA0040"],
        feature_ranges={
            "file_encryptions": (100, 100000),
            "entropy_change": (0.3, 0.5),
            "extension_changes": (100, 100000),
        }
    ),
    AttackPattern(
        name="cryptominer",
        category="Backdoor",
        mitre_tactics=["TA0040"],
        feature_ranges={
            "cpu_usage": (0.7, 1.0),
            "mining_pool_connections": (1, 10),
        }
    ),
]


class SyntheticAttackGenerator:
    """Generate synthetic attack data for ML training"""
    
    def __init__(self, n_features: int = 40):
        self.n_features = n_features
        self.patterns = ATTACK_PATTERNS
        
        # Class mapping (same as training)
        self.class_map = {
            "Normal": 0,
            "DoS/DDoS": 1,
            "Recon/Scan": 2,
            "Brute Force": 3,
            "Web/Exploit": 4,
            "Infiltration": 5,
            "Botnet": 6,
            "Backdoor": 7,
            "Worms": 8,
            "Fuzzers": 9,
            "Other": 10,
        }
        
        # Reverse mapping
        self.class_names = {v: k for k, v in self.class_map.items()}
    
    def generate_normal_sample(self) -> np.ndarray:
        """Generate normal traffic sample"""
        features = np.zeros(self.n_features)
        
        # Normal traffic characteristics
        features[0] = random.uniform(0.0, 0.3)  # Low duration
        features[1] = random.uniform(0.0, 0.2)  # Low packet count
        features[2] = random.uniform(0.0, 0.2)  # Low byte count
        features[3] = random.uniform(0.0, 0.1)  # Low entropy
        
        # Common ports (80, 443, 53)
        features[4] = random.choice([80/65535, 443/65535, 53/65535, 8080/65535])
        
        # Add some noise
        for i in range(5, self.n_features):
            features[i] = random.uniform(0.0, 0.15)
        
        return features
    
    def generate_attack_sample(self, pattern: AttackPattern) -> np.ndarray:
        """Generate attack sample based on pattern"""
        features = np.zeros(self.n_features)
        
        # Base features from pattern
        idx = 0
        for name, (low, high) in pattern.feature_ranges.items():
            if idx < self.n_features:
                # Normalize to 0-1 range
                max_val = max(high, 1.0)
                features[idx] = random.uniform(low, high) / max_val
                idx += 1
        
        # Add pattern-specific features
        if "DoS" in pattern.category:
            features[idx % self.n_features] = random.uniform(0.7, 1.0)  # High volume
            features[(idx+1) % self.n_features] = random.uniform(0.5, 0.9)  # Fast rate
        
        if "Brute" in pattern.category:
            features[idx % self.n_features] = random.uniform(0.6, 1.0)  # Many attempts
            features[(idx+1) % self.n_features] = random.uniform(0.3, 0.7)  # Medium duration
        
        if "Recon" in pattern.category:
            features[idx % self.n_features] = random.uniform(0.4, 0.8)  # Many targets
            features[(idx+1) % self.n_features] = random.uniform(0.1, 0.3)  # Short connections
        
        if "Botnet" in pattern.category or "C2" in pattern.name:
            features[idx % self.n_features] = random.uniform(0.3, 0.6)  # Periodic
            features[(idx+1) % self.n_features] = random.uniform(0.4, 0.8)  # Beaconing
        
        # Fill remaining with noise
        for i in range(idx+2, self.n_features):
            features[i] = random.uniform(0.1, 0.5) + random.gauss(0, 0.1)
        
        # Clamp to valid range
        features = np.clip(features, 0.0, 1.0)
        
        return features
    
    def generate_dataset(self, 
                        n_samples: int = 100000,
                        attack_ratio: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate complete dataset with balanced attack types
        
        Args:
            n_samples: Total samples to generate
            attack_ratio: Ratio of attack samples (0.4 = 40% attacks)
        
        Returns:
            X: Feature matrix
            y: Labels
        """
        print(f"🔧 Generating {n_samples:,} synthetic samples...")
        
        X = []
        y = []
        
        n_attacks = int(n_samples * attack_ratio)
        n_normal = n_samples - n_attacks
        
        # Generate normal traffic
        print(f"   Generating {n_normal:,} normal samples...")
        for _ in range(n_normal):
            X.append(self.generate_normal_sample())
            y.append(0)  # Normal
        
        # Generate attacks (balanced across patterns)
        print(f"   Generating {n_attacks:,} attack samples...")
        attacks_per_pattern = n_attacks // len(self.patterns)
        
        for pattern in self.patterns:
            class_id = self.class_map.get(pattern.category, 10)
            
            for _ in range(attacks_per_pattern):
                X.append(self.generate_attack_sample(pattern))
                y.append(class_id)
        
        # Remaining attacks (random patterns)
        remaining = n_attacks - (attacks_per_pattern * len(self.patterns))
        for _ in range(remaining):
            pattern = random.choice(self.patterns)
            class_id = self.class_map.get(pattern.category, 10)
            X.append(self.generate_attack_sample(pattern))
            y.append(class_id)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        # Class distribution
        print(f"\n📊 Class Distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            name = self.class_names.get(cls, "Unknown")
            print(f"   {name}: {count:,} ({100*count/len(y):.1f}%)")
        
        return X, y
    
    def generate_apt_sequence(self, n_steps: int = 10) -> List[np.ndarray]:
        """Generate multi-stage APT attack sequence"""
        sequence = []
        
        # APT stages in order
        apt_stages = [
            ("Recon/Scan", ["port_scan_aggressive", "service_enumeration"]),
            ("Brute Force", ["brute_force_ssh", "credential_stuffing"]),
            ("Web/Exploit", ["powershell_encoded", "dll_injection"]),
            ("Backdoor", ["registry_persistence", "scheduled_task"]),
            ("Botnet", ["http_beacon", "dns_tunneling"]),
            ("Worms", ["smb_lateral", "wmi_execution"]),
            ("Infiltration", ["large_upload"]),
        ]
        
        for stage_category, pattern_names in apt_stages:
            # Find matching patterns
            matching = [p for p in self.patterns if p.name in pattern_names]
            if matching:
                pattern = random.choice(matching)
                sequence.append(self.generate_attack_sample(pattern))
            
            if len(sequence) >= n_steps:
                break
        
        return sequence


def generate_and_train():
    """Generate synthetic data and train the classifier"""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    print("=" * 60)
    print("🔧 SYNTHETIC ATTACK TRAINING")
    print("=" * 60)
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️ Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Generate synthetic data
    generator = SyntheticAttackGenerator(n_features=40)
    X_syn, y_syn = generator.generate_dataset(n_samples=500000, attack_ratio=0.5)
    
    print(f"\n📊 Synthetic Dataset: {X_syn.shape[0]:,} samples, {X_syn.shape[1]} features")
    
    # Load existing real data if available
    try:
        import sys
        sys.path.insert(0, '.')
        from train_combined import create_combined_dataloader, AttackClassifier
        
        print("\n📂 Loading real dataset for augmentation...")
        real_loader = create_combined_dataloader(batch_size=1024)
        
        # Extract real data
        X_real_list = []
        y_real_list = []
        for batch_X, batch_y in real_loader:
            X_real_list.append(batch_X.numpy())
            y_real_list.append(batch_y.numpy())
        
        X_real = np.concatenate(X_real_list, axis=0)
        y_real = np.concatenate(y_real_list, axis=0)
        
        print(f"   Real data: {X_real.shape[0]:,} samples")
        
        # Combine with synthetic
        X_combined = np.concatenate([X_real, X_syn], axis=0)
        y_combined = np.concatenate([y_real, y_syn], axis=0)
        
        print(f"   Combined: {X_combined.shape[0]:,} samples")
        
    except Exception as e:
        print(f"   ⚠️ Could not load real data: {e}")
        print("   Using synthetic data only")
        X_combined = X_syn
        y_combined = y_syn
    
    # Shuffle
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    
    # Train/val split
    split_idx = int(0.8 * len(X_combined))
    X_train, X_val = X_combined[:split_idx], X_combined[split_idx:]
    y_train, y_val = y_combined[:split_idx], y_combined[split_idx:]
    
    print(f"\n📊 Training: {len(X_train):,} samples")
    print(f"   Validation: {len(X_val):,} samples")
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=0)
    
    # Create model
    n_classes = len(generator.class_map)
    
    class ImprovedClassifier(nn.Module):
        def __init__(self, input_dim=40, hidden_dim=256, n_classes=11):
            super().__init__()
            
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                
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
    
    model = ImprovedClassifier(input_dim=40, n_classes=n_classes).to(device)
    
    # Try to load existing weights
    try:
        checkpoint = torch.load("ml/models/trained/combined_classifier.pt", map_location=device, weights_only=False)
        # Load compatible weights
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() 
                         if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"   ✅ Loaded {len(pretrained_dict)} pretrained weights")
    except Exception as e:
        print(f"   ⚠️ Training from scratch: {e}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training loop
    print("\n" + "=" * 60)
    print("🚀 TRAINING")
    print("=" * 60)
    
    best_acc = 0
    epochs = 30
    
    for epoch in range(epochs):
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
            torch.save(model.state_dict(), "ml/models/trained/synthetic_classifier.pt")
            marker = " ← BEST"
        else:
            marker = ""
        
        print(f"Epoch {epoch+1:2d}/{epochs} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%{marker}")
    
    print("\n" + "=" * 60)
    print(f"✅ TRAINING COMPLETE")
    print(f"   Best Accuracy: {best_acc:.2f}%")
    print(f"   Model saved: ml/models/trained/synthetic_classifier.pt")
    print("=" * 60)
    
    return best_acc


if __name__ == "__main__":
    generate_and_train()
