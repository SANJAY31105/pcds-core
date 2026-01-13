"""
UEBA Training Module
Phase 4: Behavioral model training per entity

Features:
- Train per-entity baselines from Windows event logs
- Normal login time modeling
- Process baseline learning
- Network baseline learning
- Deviation scoring

Uses:
- Autoencoder for anomaly detection
- Statistical models for baselines
- Incremental learning for updates
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import statistics

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EntityAutoencoder(nn.Module):
    """
    Autoencoder for entity behavior anomaly detection
    
    Learns normal behavior pattern and flags deviations
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def get_reconstruction_error(self, x):
        """Get reconstruction error as anomaly score"""
        decoded, _ = self.forward(x)
        return ((x - decoded) ** 2).mean(dim=1)


class EntityProfile:
    """Statistical profile for an entity"""
    
    def __init__(self, entity_id: str):
        self.entity_id = entity_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Login time statistics
        self.login_hours: List[int] = []
        self.login_days: List[int] = []  # 0=Mon, 6=Sun
        
        # Process statistics
        self.processes: Dict[str, int] = defaultdict(int)  # process -> count
        self.process_counts: List[int] = []  # daily process counts
        
        # Network statistics
        self.connections: Dict[str, int] = defaultdict(int)  # dest -> count
        self.connection_counts: List[int] = []
        self.bytes_sent: List[int] = []
        self.bytes_received: List[int] = []
        
        # File access statistics
        self.file_paths: Dict[str, int] = defaultdict(int)
        self.file_access_counts: List[int] = []
        
        # Computed baselines
        self.baselines_computed = False
        self.login_hour_mean = 12.0
        self.login_hour_std = 4.0
        self.process_count_mean = 50.0
        self.process_count_std = 20.0
        self.connection_count_mean = 100.0
        self.connection_count_std = 50.0
    
    def add_login_event(self, timestamp: datetime):
        """Record login event"""
        self.login_hours.append(timestamp.hour)
        self.login_days.append(timestamp.weekday())
        self.updated_at = datetime.now()
        
        # Keep last 90 days
        if len(self.login_hours) > 1000:
            self.login_hours = self.login_hours[-1000:]
            self.login_days = self.login_days[-1000:]
    
    def add_process_event(self, process_name: str):
        """Record process creation"""
        self.processes[process_name] += 1
        self.updated_at = datetime.now()
    
    def add_connection_event(self, destination: str, bytes_sent: int = 0, bytes_received: int = 0):
        """Record network connection"""
        self.connections[destination] += 1
        self.bytes_sent.append(bytes_sent)
        self.bytes_received.append(bytes_received)
        self.updated_at = datetime.now()
        
        if len(self.bytes_sent) > 1000:
            self.bytes_sent = self.bytes_sent[-1000:]
            self.bytes_received = self.bytes_received[-1000:]
    
    def compute_baselines(self):
        """Compute statistical baselines from accumulated data"""
        if len(self.login_hours) >= 10:
            self.login_hour_mean = statistics.mean(self.login_hours)
            self.login_hour_std = statistics.stdev(self.login_hours) if len(self.login_hours) > 1 else 4.0
        
        if len(self.process_counts) >= 5:
            self.process_count_mean = statistics.mean(self.process_counts)
            self.process_count_std = statistics.stdev(self.process_counts) if len(self.process_counts) > 1 else 20.0
        
        if len(self.connection_counts) >= 5:
            self.connection_count_mean = statistics.mean(self.connection_counts)
            self.connection_count_std = statistics.stdev(self.connection_counts) if len(self.connection_counts) > 1 else 50.0
        
        self.baselines_computed = True
    
    def get_login_time_deviation(self, hour: int) -> float:
        """Get z-score for login hour"""
        if self.login_hour_std == 0:
            return 0.0
        return abs(hour - self.login_hour_mean) / self.login_hour_std
    
    def get_process_count_deviation(self, count: int) -> float:
        """Get z-score for process count"""
        if self.process_count_std == 0:
            return 0.0
        return abs(count - self.process_count_mean) / self.process_count_std
    
    def is_unusual_process(self, process_name: str) -> bool:
        """Check if process is unusual for this entity"""
        return process_name not in self.processes
    
    def to_feature_vector(self) -> np.ndarray:
        """Convert profile to feature vector for autoencoder"""
        features = []
        
        # Login time features
        features.append(self.login_hour_mean / 24)
        features.append(self.login_hour_std / 12)
        features.append(len(set(self.login_days)) / 7)  # day variety
        
        # Process features
        features.append(min(len(self.processes) / 100, 1.0))  # unique processes
        features.append(self.process_count_mean / 200)
        
        # Network features
        features.append(min(len(self.connections) / 50, 1.0))  # unique connections
        features.append(self.connection_count_mean / 500)
        
        # Bytes features
        avg_sent = statistics.mean(self.bytes_sent) if self.bytes_sent else 0
        avg_recv = statistics.mean(self.bytes_received) if self.bytes_received else 0
        features.append(min(avg_sent / 1e6, 1.0))
        features.append(min(avg_recv / 1e6, 1.0))
        
        # Pad to fixed size
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32], dtype=np.float32)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "entity_id": self.entity_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "login_hours_count": len(self.login_hours),
            "unique_processes": len(self.processes),
            "unique_connections": len(self.connections),
            "baselines_computed": self.baselines_computed,
            "login_hour_mean": self.login_hour_mean,
            "process_count_mean": self.process_count_mean
        }


class UEBATrainer:
    """
    UEBA (User and Entity Behavior Analytics) Training System
    
    Capabilities:
    - Build behavioral profiles from Windows event logs
    - Train autoencoder for anomaly detection
    - Compute statistical baselines per entity
    - Continuous learning from new events
    """
    
    def __init__(self, model_save_path: str = "ml/models/trained/ueba"):
        self.model_save_path = Path(model_save_path)
        self.model_save_path.mkdir(parents=True, exist_ok=True)
        
        # Entity profiles
        self.profiles: Dict[str, EntityProfile] = {}
        
        # Autoencoder model
        self.autoencoder: EntityAutoencoder = None
        
        # Training data
        self.training_samples: List[np.ndarray] = []
        
        # Anomaly threshold (MSE)
        self.anomaly_threshold = 0.1
        
        # Stats
        self.stats = {
            "entities_profiled": 0,
            "events_processed": 0,
            "anomalies_detected": 0,
            "model_trained": False
        }
        
        print("📊 UEBA Trainer initialized")
    
    def process_login_event(self, entity_id: str, timestamp: datetime):
        """Process a login event"""
        profile = self._get_or_create_profile(entity_id)
        profile.add_login_event(timestamp)
        self.stats["events_processed"] += 1
    
    def process_process_event(self, entity_id: str, process_name: str):
        """Process a process creation event"""
        profile = self._get_or_create_profile(entity_id)
        profile.add_process_event(process_name)
        self.stats["events_processed"] += 1
    
    def process_connection_event(self, entity_id: str, destination: str,
                                  bytes_sent: int = 0, bytes_received: int = 0):
        """Process a network connection event"""
        profile = self._get_or_create_profile(entity_id)
        profile.add_connection_event(destination, bytes_sent, bytes_received)
        self.stats["events_processed"] += 1
    
    def _get_or_create_profile(self, entity_id: str) -> EntityProfile:
        """Get or create entity profile"""
        if entity_id not in self.profiles:
            self.profiles[entity_id] = EntityProfile(entity_id)
            self.stats["entities_profiled"] += 1
        return self.profiles[entity_id]
    
    def train_autoencoder(self, epochs: int = 50, batch_size: int = 32):
        """Train autoencoder on all entity profiles"""
        print("\n🧠 Training UEBA Autoencoder...")
        
        # Compute baselines for all profiles
        for profile in self.profiles.values():
            profile.compute_baselines()
        
        # Create training data
        X = []
        for profile in self.profiles.values():
            X.append(profile.to_feature_vector())
        
        if len(X) < 10:
            print("   ⚠️ Not enough profiles for training (need at least 10)")
            return
        
        X = np.array(X)
        print(f"   Training samples: {len(X)}")
        
        # Create model
        self.autoencoder = EntityAutoencoder(input_dim=32, latent_dim=16).to(device)
        
        # Training
        X_tensor = torch.FloatTensor(X).to(device)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, in loader:
                optimizer.zero_grad()
                decoded, _ = self.autoencoder(batch_x)
                loss = criterion(decoded, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(loader):.6f}")
        
        # Set anomaly threshold (95th percentile of training errors)
        with torch.no_grad():
            errors = self.autoencoder.get_reconstruction_error(X_tensor).cpu().numpy()
            self.anomaly_threshold = np.percentile(errors, 95)
        
        print(f"   Anomaly threshold: {self.anomaly_threshold:.6f}")
        
        # Save model
        torch.save({
            'model_state_dict': self.autoencoder.state_dict(),
            'anomaly_threshold': self.anomaly_threshold
        }, self.model_save_path / "autoencoder.pt")
        
        self.stats["model_trained"] = True
        print("   ✅ UEBA autoencoder trained and saved")
    
    def detect_anomaly(self, entity_id: str) -> Tuple[bool, float]:
        """
        Detect if entity behavior is anomalous
        
        Returns:
            (is_anomalous, anomaly_score)
        """
        if entity_id not in self.profiles:
            return False, 0.0
        
        if self.autoencoder is None:
            return False, 0.0
        
        profile = self.profiles[entity_id]
        features = torch.FloatTensor(profile.to_feature_vector()).unsqueeze(0).to(device)
        
        with torch.no_grad():
            error = self.autoencoder.get_reconstruction_error(features).item()
        
        is_anomalous = error > self.anomaly_threshold
        
        if is_anomalous:
            self.stats["anomalies_detected"] += 1
        
        return is_anomalous, error
    
    def get_entity_risk_score(self, entity_id: str) -> float:
        """Get overall risk score for an entity"""
        if entity_id not in self.profiles:
            return 0.0
        
        profile = self.profiles[entity_id]
        
        # Combine multiple factors
        is_anomalous, anomaly_score = self.detect_anomaly(entity_id)
        
        # Normalize to 0-1
        normalized_score = min(anomaly_score / (self.anomaly_threshold * 2), 1.0)
        
        return normalized_score
    
    def save_profiles(self, path: str = None):
        """Save all entity profiles"""
        path = path or str(self.model_save_path / "profiles.json")
        
        profiles_data = {
            entity_id: profile.to_dict()
            for entity_id, profile in self.profiles.items()
        }
        
        with open(path, 'w') as f:
            json.dump(profiles_data, f, indent=2)
        
        print(f"   Saved {len(profiles_data)} profiles to {path}")
    
    def get_stats(self) -> Dict:
        """Get trainer statistics"""
        return self.stats


# Singleton
_ueba_trainer = None

def get_ueba_trainer() -> UEBATrainer:
    global _ueba_trainer
    if _ueba_trainer is None:
        _ueba_trainer = UEBATrainer()
    return _ueba_trainer


def train_from_synthetic_data():
    """Train UEBA from synthetic event data"""
    print("=" * 60)
    print("📊 UEBA TRAINING FROM SYNTHETIC DATA")
    print("=" * 60)
    
    trainer = UEBATrainer()
    
    # Generate synthetic entities
    print("\n📝 Generating synthetic entity data...")
    
    for i in range(100):  # 100 users
        entity_id = f"user_{i:03d}"
        
        # Simulate login events
        base_hour = 8 + (i % 4)  # Different typical login times
        for j in range(30):  # 30 login events
            hour = base_hour + np.random.randint(-2, 3)
            timestamp = datetime.now() - timedelta(days=30-j, hours=24-hour)
            trainer.process_login_event(entity_id, timestamp)
        
        # Simulate process events
        num_processes = 20 + (i % 30)  # Different levels of activity
        for j in range(num_processes):
            process = f"process_{j % 10}"
            trainer.process_process_event(entity_id, process)
        
        # Simulate network events
        num_connections = 50 + (i % 100)
        for j in range(num_connections):
            dest = f"server_{j % 20}.internal"
            trainer.process_connection_event(
                entity_id, dest,
                bytes_sent=np.random.randint(100, 10000),
                bytes_received=np.random.randint(1000, 100000)
            )
    
    print(f"   Entities: {trainer.stats['entities_profiled']}")
    print(f"   Events: {trainer.stats['events_processed']}")
    
    # Train autoencoder
    trainer.train_autoencoder(epochs=30)
    
    # Test anomaly detection
    print("\n📍 Testing anomaly detection...")
    
    # Create an anomalous entity
    anomalous_id = "user_anomalous"
    
    # Unusual login time
    trainer.process_login_event(anomalous_id, 
                                 datetime.now().replace(hour=3))  # 3 AM login
    
    # Many unique processes
    for i in range(100):
        trainer.process_process_event(anomalous_id, f"unusual_process_{i}")
    
    # Many external connections
    for i in range(200):
        trainer.process_connection_event(
            anomalous_id, f"external_{i}.suspicious.com",
            bytes_sent=100000
        )
    
    trainer.profiles[anomalous_id].compute_baselines()
    
    is_anomalous, score = trainer.detect_anomaly(anomalous_id)
    print(f"   Anomalous entity detected: {is_anomalous} (score: {score:.6f})")
    
    # Test normal entity
    is_anomalous_normal, score_normal = trainer.detect_anomaly("user_001")
    print(f"   Normal entity: {is_anomalous_normal} (score: {score_normal:.6f})")
    
    # Save
    trainer.save_profiles()
    
    print(f"\n✅ UEBA training complete")
    print(f"   Stats: {trainer.get_stats()}")
    
    return trainer


if __name__ == "__main__":
    train_from_synthetic_data()
