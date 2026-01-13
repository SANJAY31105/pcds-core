"""
Attack Sequence Learning with Transformer
Phase 2: Learn multi-stage intrusion patterns

Architecture:
- Positional encoding for temporal sequences
- Transformer encoder for pattern recognition
- Sequence classification head

Input: Sequence of events (process, network, file, etc.)
Output: Attack stage classification or attack campaign prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import math

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using device: {device}")


# Attack stages (Kill Chain)
ATTACK_STAGES = [
    "normal",           # 0
    "reconnaissance",   # 1
    "initial_access",   # 2
    "execution",        # 3
    "persistence",      # 4
    "privilege_esc",    # 5
    "defense_evasion",  # 6
    "credential_access",# 7
    "discovery",        # 8
    "lateral_movement", # 9
    "collection",       # 10
    "exfiltration",     # 11
    "impact"           # 12
]


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences"""
    
    def __init__(self, d_model: int, max_len: int = 1000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AttackSequenceTransformer(nn.Module):
    """
    Transformer for attack sequence classification
    
    Learns temporal patterns in security events to:
    1. Classify current attack stage
    2. Predict next attack stage
    3. Identify multi-stage intrusions
    """
    
    def __init__(self, 
                 input_dim: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 3,
                 num_classes: int = 13,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head - current stage
        self.stage_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Prediction head - next stage
        self.next_stage_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
        # Multi-stage intrusion detector
        self.intrusion_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: [batch, seq_len, input_dim] - sequence of events
            mask: Optional attention mask
        
        Returns:
            stage_logits: Current stage classification
            next_stage_logits: Next stage prediction
            intrusion_prob: Probability of multi-stage intrusion
        """
        # Project to model dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch, seq_len, d_model]
        
        # Transformer encoding
        encoded = self.transformer(x, mask)
        
        # Use last token for classification
        last_hidden = encoded[:, -1, :]
        
        # Classifications
        stage_logits = self.stage_classifier(last_hidden)
        next_stage_logits = self.next_stage_predictor(last_hidden)
        intrusion_prob = self.intrusion_detector(last_hidden)
        
        return stage_logits, next_stage_logits, intrusion_prob


class AttackSequenceDataset(Dataset):
    """Dataset for attack sequences"""
    
    def __init__(self, sequences, labels, next_labels=None, intrusion_labels=None):
        self.sequences = sequences
        self.labels = labels
        self.next_labels = next_labels if next_labels is not None else labels
        self.intrusion_labels = intrusion_labels if intrusion_labels is not None else np.zeros(len(labels))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.LongTensor([self.labels[idx]])[0],
            torch.LongTensor([self.next_labels[idx]])[0],
            torch.FloatTensor([self.intrusion_labels[idx]])[0]
        )


def generate_synthetic_sequences(num_sequences: int = 10000, 
                                  seq_length: int = 20,
                                  feature_dim: int = 32) -> tuple:
    """Generate synthetic attack sequences for training"""
    
    print("📝 Generating synthetic attack sequences...")
    
    sequences = []
    labels = []
    next_labels = []
    intrusion_labels = []
    
    # Attack patterns (stage progressions)
    attack_patterns = [
        [1, 2, 3, 4, 5],          # Recon -> Access -> Execute -> Persist -> Priv Esc
        [1, 2, 3, 7, 9],          # Recon -> Access -> Execute -> Cred -> Lateral
        [2, 3, 8, 9, 10, 11],     # Access -> Execute -> Discovery -> Lateral -> Collect -> Exfil
        [1, 8, 9, 10, 11, 12],    # Recon -> Discovery -> Lateral -> Collect -> Exfil -> Impact
        [2, 3, 4, 6, 12],         # Access -> Execute -> Persist -> Evasion -> Impact (Ransomware)
    ]
    
    for i in range(num_sequences):
        if i % 3 == 0:
            # Normal traffic
            seq = np.random.randn(seq_length, feature_dim) * 0.5
            stage = 0
            next_stage = 0
            is_intrusion = 0
        else:
            # Attack sequence
            pattern = attack_patterns[i % len(attack_patterns)]
            position = (i % len(pattern))
            stage = pattern[position]
            next_stage = pattern[min(position + 1, len(pattern) - 1)]
            
            # Generate sequence with attack characteristics
            seq = np.random.randn(seq_length, feature_dim)
            
            # Add attack stage signature
            seq[:, stage % feature_dim] += 2.0
            seq[:, :5] += stage * 0.3  # Temporal pattern
            
            is_intrusion = 1 if position > 0 else 0
        
        sequences.append(seq.astype(np.float32))
        labels.append(stage)
        next_labels.append(next_stage)
        intrusion_labels.append(is_intrusion)
    
    print(f"   Generated {num_sequences} sequences")
    
    return np.array(sequences), np.array(labels), np.array(next_labels), np.array(intrusion_labels)


def train_sequence_model():
    """Train the attack sequence transformer"""
    
    print("=" * 60)
    print("🔄 ATTACK SEQUENCE TRANSFORMER TRAINING")
    print("=" * 60)
    
    # Generate training data
    sequences, labels, next_labels, intrusion_labels = generate_synthetic_sequences(
        num_sequences=20000,
        seq_length=20,
        feature_dim=32
    )
    
    # Split
    split_idx = int(len(sequences) * 0.8)
    train_dataset = AttackSequenceDataset(
        sequences[:split_idx], labels[:split_idx],
        next_labels[:split_idx], intrusion_labels[:split_idx]
    )
    test_dataset = AttackSequenceDataset(
        sequences[split_idx:], labels[split_idx:],
        next_labels[split_idx:], intrusion_labels[split_idx:]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128)
    
    print(f"📊 Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    model = AttackSequenceTransformer(
        input_dim=32,
        d_model=128,
        nhead=4,
        num_layers=3,
        num_classes=len(ATTACK_STAGES),
        dropout=0.1
    ).to(device)
    
    print(f"🧠 Model: {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss functions
    stage_criterion = nn.CrossEntropyLoss()
    next_criterion = nn.CrossEntropyLoss()
    intrusion_criterion = nn.BCELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    # Training
    epochs = 30
    best_acc = 0
    
    print(f"\n🚀 Training for {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for batch_seq, batch_stage, batch_next, batch_intrusion in train_loader:
            batch_seq = batch_seq.to(device)
            batch_stage = batch_stage.to(device)
            batch_next = batch_next.to(device)
            batch_intrusion = batch_intrusion.to(device)
            
            optimizer.zero_grad()
            
            stage_logits, next_logits, intrusion_prob = model(batch_seq)
            
            # Combined loss
            loss = (
                stage_criterion(stage_logits, batch_stage) +
                0.5 * next_criterion(next_logits, batch_next) +
                0.5 * intrusion_criterion(intrusion_prob.squeeze(), batch_intrusion)
            )
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_seq, batch_stage, _, _ in test_loader:
                batch_seq = batch_seq.to(device)
                batch_stage = batch_stage.to(device)
                
                stage_logits, _, _ = model(batch_seq)
                preds = stage_logits.argmax(dim=1)
                correct += (preds == batch_stage).sum().item()
                total += len(batch_stage)
        
        accuracy = correct / total
        
        if accuracy > best_acc:
            best_acc = accuracy
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': 32,
                'd_model': 128,
                'num_classes': len(ATTACK_STAGES),
                'attack_stages': ATTACK_STAGES
            }, 'ml/models/trained/sequence_transformer.pt')
        
        if (epoch + 1) % 5 == 0:
            print(f"   Epoch {epoch+1:3d} | Loss: {epoch_loss/len(train_loader):.4f} | "
                  f"Acc: {accuracy*100:.2f}% | Best: {best_acc*100:.2f}%")
    
    print("-" * 60)
    print(f"\n✅ Model saved to ml/models/trained/sequence_transformer.pt")
    print(f"   Best accuracy: {best_acc*100:.2f}%")
    
    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "stage_accuracy": float(best_acc),
        "attack_stages": ATTACK_STAGES,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset)
    }
    
    with open('ml/models/trained/sequence_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return model, best_acc


if __name__ == "__main__":
    train_sequence_model()
