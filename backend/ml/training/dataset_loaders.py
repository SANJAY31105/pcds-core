"""
ML Training Pipeline - Dataset Loaders
Supports: CIC-IDS 2017, UNSW-NB15, CTU-13

Downloads:
- CIC-IDS 2017: https://www.unb.ca/cic/datasets/ids-2017.html
- UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- CTU-13: https://www.stratosphereips.org/datasets-ctu13
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class DatasetLoader:
    """Base class for dataset loading"""
    
    def __init__(self, data_dir: str = "ml/datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and return (features, labels, attack_types)"""
        raise NotImplementedError
    
    def get_attack_categories(self) -> Dict[str, List[str]]:
        """Return mapping of attack categories to labels"""
        raise NotImplementedError


class CICIDS2017Loader(DatasetLoader):
    """
    CIC-IDS 2017 Dataset Loader
    
    Dataset contains:
    - Benign traffic
    - Brute Force (FTP, SSH)
    - DoS (Hulk, GoldenEye, Slowloris, Slowhttptest)
    - Heartbleed
    - Web Attack (XSS, SQL Injection, Brute Force)
    - Infiltration
    - Botnet
    - PortScan
    - DDoS
    
    Features: 78 flow-based features
    """
    
    ATTACK_MAPPING = {
        'BENIGN': 'benign',
        'FTP-Patator': 'brute_force',
        'SSH-Patator': 'brute_force',
        'DoS Hulk': 'dos',
        'DoS GoldenEye': 'dos',
        'DoS slowloris': 'dos',
        'DoS Slowhttptest': 'dos',
        'Heartbleed': 'exploit',
        'Web Attack – Brute Force': 'brute_force',
        'Web Attack – XSS': 'web_attack',
        'Web Attack – Sql Injection': 'web_attack',
        'Infiltration': 'infiltration',
        'Bot': 'botnet',
        'PortScan': 'recon',
        'DDoS': 'ddos'
    }
    
    # Key features to extract (most important for ML)
    KEY_FEATURES = [
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Fwd Packet Length Mean', 'Bwd Packet Length Mean',
        'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
        'Fwd IAT Mean', 'Bwd IAT Mean', 'Fwd PSH Flags',
        'SYN Flag Count', 'RST Flag Count', 'ACK Flag Count',
        'URG Flag Count', 'Average Packet Size', 'Avg Fwd Segment Size',
        'Avg Bwd Segment Size', 'Init_Win_bytes_forward',
        'Init_Win_bytes_backward', 'Subflow Fwd Packets',
        'Subflow Bwd Packets', 'Subflow Fwd Bytes', 'Subflow Bwd Bytes'
    ]
    
    def __init__(self, data_dir: str = "ml/datasets"):
        super().__init__(data_dir)
        self.dataset_path = self.data_dir / "cicids2017"
    
    def load(self, file_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load CIC-IDS 2017 dataset
        
        Args:
            file_path: Path to CSV file (downloads contain multiple day files)
        
        Returns:
            features, labels, attack_types
        """
        if file_path:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', low_memory=False)
        else:
            # Try to find any CSV in dataset directory
            csv_files = list(self.dataset_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(
                    f"No CSV files found in {self.dataset_path}. "
                    "Download CIC-IDS 2017 from: https://www.unb.ca/cic/datasets/ids-2017.html"
                )
            dfs = []
            for f in csv_files:
                try:
                    dfs.append(pd.read_csv(f, encoding='utf-8', on_bad_lines='skip', low_memory=False))
                except:
                    try:
                        dfs.append(pd.read_csv(f, encoding='latin-1', on_bad_lines='skip', low_memory=False))
                    except Exception as e:
                        print(f"⚠️ Skipping {f.name}: {e}")
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Get label column
        label_col = 'Label' if 'Label' in df.columns else df.columns[-1]
        
        # Map to attack categories
        df['attack_category'] = df[label_col].map(self.ATTACK_MAPPING).fillna('unknown')
        
        # Extract features (handle missing columns gracefully)
        available_features = [f for f in self.KEY_FEATURES if f in df.columns]
        if not available_features:
            # Use numeric columns if key features not found
            available_features = df.select_dtypes(include=[np.number]).columns.tolist()
            available_features = [f for f in available_features if f != label_col]
        
        X = df[available_features].values
        
        # Handle inf and nan
        X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
        
        # Encode labels
        y_category = self.label_encoder.fit_transform(df['attack_category'])
        y_binary = (df['attack_category'] != 'benign').astype(int).values
        
        # Normalize features
        X = self.scaler.fit_transform(X)
        
        print(f"✅ Loaded CIC-IDS 2017: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Attack distribution: {df['attack_category'].value_counts().to_dict()}")
        
        return X, y_binary, y_category
    
    def get_attack_categories(self) -> Dict[str, List[str]]:
        return {
            'brute_force': ['FTP-Patator', 'SSH-Patator', 'Web Attack – Brute Force'],
            'dos': ['DoS Hulk', 'DoS GoldenEye', 'DoS slowloris', 'DoS Slowhttptest'],
            'ddos': ['DDoS'],
            'botnet': ['Bot'],
            'recon': ['PortScan'],
            'web_attack': ['Web Attack – XSS', 'Web Attack – Sql Injection'],
            'infiltration': ['Infiltration'],
            'exploit': ['Heartbleed']
        }


class UNSWNB15Loader(DatasetLoader):
    """
    UNSW-NB15 Dataset Loader
    
    Dataset contains:
    - Normal traffic
    - Fuzzers
    - Analysis
    - Backdoors
    - DoS
    - Exploits
    - Generic
    - Reconnaissance
    - Shellcode
    - Worms
    
    Features: 49 features
    """
    
    ATTACK_MAPPING = {
        'Normal': 'benign',
        'Fuzzers': 'fuzzing',
        'Analysis': 'analysis',
        'Backdoors': 'backdoor',
        'DoS': 'dos',
        'Exploits': 'exploit',
        'Generic': 'generic',
        'Reconnaissance': 'recon',
        'Shellcode': 'shellcode',
        'Worms': 'worm'
    }
    
    KEY_FEATURES = [
        'dur', 'spkts', 'dpkts', 'sbytes', 'dbytes',
        'rate', 'sttl', 'dttl', 'sload', 'dload',
        'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit',
        'djit', 'swin', 'stcpb', 'dtcpb', 'dwin',
        'tcprtt', 'synack', 'ackdat', 'smean', 'dmean',
        'trans_depth', 'response_body_len', 'ct_srv_src',
        'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm',
        'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'is_ftp_login',
        'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst'
    ]
    
    def __init__(self, data_dir: str = "ml/datasets"):
        super().__init__(data_dir)
        self.dataset_path = self.data_dir / "unsw-nb15"
    
    def load(self, file_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load UNSW-NB15 dataset"""
        if file_path:
            df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip', low_memory=False)
        else:
            csv_files = list(self.dataset_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(
                    f"No CSV files found in {self.dataset_path}. "
                    "Download UNSW-NB15 from: https://research.unsw.edu.au/projects/unsw-nb15-dataset"
                )
            # Prefer training/testing sets if available
            train_test_files = [f for f in csv_files if 'training' in f.name.lower() or 'testing' in f.name.lower()]
            files_to_load = train_test_files if train_test_files else csv_files[:2]  # Limit to avoid memory issues
            
            dfs = []
            for f in files_to_load:
                try:
                    print(f"   Loading {f.name}...")
                    dfs.append(pd.read_csv(f, encoding='utf-8', on_bad_lines='skip', low_memory=False))
                except:
                    try:
                        dfs.append(pd.read_csv(f, encoding='latin-1', on_bad_lines='skip', low_memory=False))
                    except Exception as e:
                        print(f"⚠️ Skipping {f.name}: {e}")
            df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        
        # Clean column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Get attack category
        attack_col = 'attack_cat' if 'attack_cat' in df.columns else 'label'
        df['attack_category'] = df[attack_col].map(
            {k.lower(): v for k, v in self.ATTACK_MAPPING.items()}
        ).fillna('unknown')
        
        # Extract features
        available_features = [f for f in self.KEY_FEATURES if f in df.columns]
        if not available_features:
            available_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        X = df[available_features].values
        X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
        
        y_category = self.label_encoder.fit_transform(df['attack_category'])
        y_binary = (df['attack_category'] != 'benign').astype(int).values
        
        X = self.scaler.fit_transform(X)
        
        print(f"✅ Loaded UNSW-NB15: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Attack distribution: {df['attack_category'].value_counts().to_dict()}")
        
        return X, y_binary, y_category
    
    def get_attack_categories(self) -> Dict[str, List[str]]:
        return {
            'dos': ['DoS'],
            'exploit': ['Exploits', 'Shellcode'],
            'recon': ['Reconnaissance', 'Analysis'],
            'backdoor': ['Backdoors', 'Worms'],
            'fuzzing': ['Fuzzers'],
            'generic': ['Generic']
        }


class CTU13Loader(DatasetLoader):
    """
    CTU-13 Botnet Dataset Loader
    
    Dataset contains 13 scenarios of botnet traffic:
    - Neris, Rbot, Virut, Menti, Sogou, Murlo, NSIS.ay
    - Background (normal) traffic
    
    Features: NetFlow-based features
    """
    
    ATTACK_MAPPING = {
        'Background': 'benign',
        'Normal': 'benign',
        'Botnet': 'botnet',
        'C&C': 'c2'
    }
    
    def __init__(self, data_dir: str = "ml/datasets"):
        super().__init__(data_dir)
        self.dataset_path = self.data_dir / "ctu13"
    
    def load(self, file_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load CTU-13 dataset"""
        if file_path:
            df = pd.read_csv(file_path)
        else:
            csv_files = list(self.dataset_path.glob("*.csv"))
            if not csv_files:
                raise FileNotFoundError(
                    f"No CSV files found in {self.dataset_path}. "
                    "Download CTU-13 from: https://www.stratosphereips.org/datasets-ctu13"
                )
            df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
        
        # Clean and standardize
        df.columns = df.columns.str.strip().str.lower()
        
        # Determine label column
        label_col = next((c for c in df.columns if 'label' in c.lower()), None)
        if label_col:
            df['attack_category'] = df[label_col].apply(
                lambda x: 'botnet' if 'botnet' in str(x).lower() else 'benign'
            )
        else:
            df['attack_category'] = 'unknown'
        
        # Use numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        X = df[numeric_cols].values
        X = np.nan_to_num(X, nan=0, posinf=1e10, neginf=-1e10)
        
        y_category = self.label_encoder.fit_transform(df['attack_category'])
        y_binary = (df['attack_category'] != 'benign').astype(int).values
        
        X = self.scaler.fit_transform(X)
        
        print(f"✅ Loaded CTU-13: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   Attack distribution: {df['attack_category'].value_counts().to_dict()}")
        
        return X, y_binary, y_category
    
    def get_attack_categories(self) -> Dict[str, List[str]]:
        return {
            'botnet': ['Botnet', 'C&C'],
        }


class UnifiedDatasetLoader:
    """
    Unified loader that can load and combine multiple datasets
    """
    
    def __init__(self, data_dir: str = "ml/datasets"):
        self.data_dir = Path(data_dir)
        self.loaders = {
            'cicids2017': CICIDS2017Loader(data_dir),
            'unsw-nb15': UNSWNB15Loader(data_dir),
            'ctu13': CTU13Loader(data_dir)
        }
        self.scaler = StandardScaler()
    
    def load_all(self) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Load all available datasets"""
        datasets = {}
        
        for name, loader in self.loaders.items():
            try:
                X, y_binary, y_category = loader.load()
                datasets[name] = (X, y_binary, y_category)
            except FileNotFoundError as e:
                print(f"⚠️ {name}: {e}")
        
        return datasets
    
    def create_training_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        val_size: float = 0.2
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Create train/validation/test split (60/20/20)
        """
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_size_adjusted, stratify=y_trainval, random_state=42
        )
        
        print(f"📊 Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }


# Convenience function
def load_dataset(name: str, file_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a specific dataset by name
    
    Args:
        name: 'cicids2017', 'unsw-nb15', or 'ctu13'
        file_path: Optional path to specific CSV file
    
    Returns:
        (features, binary_labels, category_labels)
    """
    loaders = {
        'cicids2017': CICIDS2017Loader,
        'unsw-nb15': UNSWNB15Loader,
        'ctu13': CTU13Loader
    }
    
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    
    loader = loaders[name]()
    return loader.load(file_path)


if __name__ == "__main__":
    print("ML Dataset Loaders - Test")
    print("=" * 50)
    
    # Test unified loader
    unified = UnifiedDatasetLoader()
    datasets = unified.load_all()
    
    print(f"\n✅ Loaded {len(datasets)} datasets")
    for name, (X, y_bin, y_cat) in datasets.items():
        print(f"   {name}: {X.shape}")
