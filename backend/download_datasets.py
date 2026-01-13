"""
Dataset Downloader for PCDS ML Training
Downloads and prepares:
- UNSW-NB15
- CSE-CIC-IDS 2018
- EMBER (Malware)
"""

import os
import sys
import requests
from pathlib import Path
import zipfile
import tarfile
import shutil

# Dataset paths
DATASETS_DIR = Path("ml/datasets")

# Dataset URLs
DATASETS = {
    "unsw-nb15": {
        "urls": [
            "https://cloudstor.aarnet.edu.au/plus/s/2DhnLGDdEECo4ys/download",  # UNSW-NB15_1.csv
        ],
        "description": "UNSW-NB15 Network Intrusion Dataset",
        "size": "~700MB"
    },
    "ember": {
        "url": "https://github.com/elastic/ember/raw/master/ember2018.tar.bz2",
        "description": "EMBER Malware Dataset",
        "size": "~2GB"
    }
}


def download_file(url: str, dest: Path, desc: str = "") -> bool:
    """Download file with progress"""
    try:
        print(f"📥 Downloading {desc}...")
        print(f"   URL: {url[:50]}...")
        
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest, 'wb') as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        pct = downloaded / total_size * 100
                        print(f"\r   Progress: {pct:.1f}% ({downloaded//1024//1024}MB)", end="")
        
        print(f"\n   ✅ Downloaded to {dest}")
        return True
        
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return False


def download_unsw_nb15():
    """Download UNSW-NB15 dataset"""
    print("\n" + "=" * 60)
    print("📦 UNSW-NB15 Dataset")
    print("=" * 60)
    
    dest_dir = DATASETS_DIR / "unsw-nb15"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Use kagglehub or direct download alternative
    print("ℹ️ UNSW-NB15 requires manual download from:")
    print("   https://research.unsw.edu.au/projects/unsw-nb15-dataset")
    print("   or https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15")
    print()
    print("   Please download and place files in:")
    print(f"   {dest_dir.absolute()}")
    
    # Create README with instructions
    readme = dest_dir / "README.md"
    readme.write_text("""# UNSW-NB15 Dataset

## Download Instructions:
1. Go to https://research.unsw.edu.au/projects/unsw-nb15-dataset
2. Download the CSV files
3. Place them in this directory

## Expected Files:
- UNSW-NB15_1.csv
- UNSW-NB15_2.csv
- UNSW-NB15_3.csv
- UNSW-NB15_4.csv

## Or use Kaggle:
```bash
kaggle datasets download -d mrwellsdavid/unsw-nb15
```
""")
    
    print("   📝 Created README with download instructions")
    return False


def download_cicids_2018():
    """Download CSE-CIC-IDS 2018"""
    print("\n" + "=" * 60)
    print("📦 CSE-CIC-IDS 2018 Dataset")
    print("=" * 60)
    
    dest_dir = DATASETS_DIR / "cicids2018"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("ℹ️ CSE-CIC-IDS 2018 requires manual download (~7GB)")
    print("   https://www.unb.ca/cic/datasets/ids-2018.html")
    print()
    print("   AWS CLI command:")
    print("   aws s3 sync --no-sign-request s3://cse-cic-ids2018 .")
    print()
    print(f"   Place files in: {dest_dir.absolute()}")
    
    readme = dest_dir / "README.md"
    readme.write_text("""# CSE-CIC-IDS 2018 Dataset

## Download Instructions:

### Option 1: AWS CLI (Recommended)
```bash
aws s3 sync --no-sign-request s3://cse-cic-ids2018/Processed\\ Traffic\\ Data\\ for\\ ML\\ Algorithms/ .
```

### Option 2: Direct Download
1. Go to https://www.unb.ca/cic/datasets/ids-2018.html
2. Download the processed CSV files
3. Place them in this directory

## Expected Size: ~7GB
""")
    
    print("   📝 Created README with download instructions")
    return False


def download_ember():
    """Download EMBER malware dataset"""
    print("\n" + "=" * 60)
    print("📦 EMBER Malware Dataset")
    print("=" * 60)
    
    dest_dir = DATASETS_DIR / "ember"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Try pip install ember
    print("ℹ️ EMBER can be installed via pip:")
    print()
    print("   pip install lief ember")
    print()
    print("   Then download dataset:")
    print("   ember-download 2018")
    print()
    print("   Or manual download from:")
    print("   https://github.com/elastic/ember")
    print(f"   Place files in: {dest_dir.absolute()}")
    
    readme = dest_dir / "README.md"
    readme.write_text("""# EMBER Malware Dataset

## Download Instructions:

### Option 1: Python Package (Recommended)
```bash
pip install lief ember
ember-download 2018
```

### Option 2: Manual Download
1. Go to https://github.com/elastic/ember
2. Download ember2018.tar.bz2
3. Extract to this directory

## Dataset Info:
- 1.1M training samples
- 100K test samples
- PE file features (no actual malware binaries)
""")
    
    print("   📝 Created README with download instructions")
    return False


def create_combined_loader():
    """Create a combined data loader script"""
    print("\n" + "=" * 60)
    print("📝 Creating Combined Data Loader")
    print("=" * 60)
    
    loader_code = '''"""
Combined Dataset Loader
Loads and combines multiple security datasets for training
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Optional

DATASETS_DIR = Path(__file__).parent.parent / "datasets"

# Unified attack class mapping
UNIFIED_CLASSES = {
    # Normal
    "normal": 0, "benign": 0, "BENIGN": 0,
    
    # DoS/DDoS
    "dos": 1, "DoS": 1, "ddos": 1, "DDoS": 1,
    "DoS Hulk": 1, "DoS GoldenEye": 1, "DoS slowloris": 1,
    
    # Port Scan / Recon
    "portscan": 2, "PortScan": 2, "Reconnaissance": 2, "reconnaissance": 2,
    "Analysis": 2,
    
    # Brute Force
    "brute_force": 3, "FTP-Patator": 3, "SSH-Patator": 3,
    
    # Web Attack
    "web_attack": 4, "Web Attack": 4, "Exploits": 4,
    
    # Infiltration
    "infiltration": 5, "Infiltration": 5,
    
    # Botnet
    "botnet": 6, "Bot": 6,
    
    # Backdoor
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
    "Web Attack", "Infiltration", "Botnet", "Backdoor",
    "Worms", "Fuzzers", "Other"
]


def load_cicids2017(sample_frac: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load CICIDS 2017 dataset"""
    path = DATASETS_DIR / "cicids2017" / "CIC-IDS-2017-V2.csv"
    
    if not path.exists():
        print(f"⚠️ CICIDS 2017 not found at {path}")
        return None, None
    
    print("📂 Loading CICIDS 2017...")
    df = pd.read_csv(path, encoding='latin-1', low_memory=False)
    
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    
    # Find label column
    label_col = [c for c in df.columns if 'label' in c.lower()][0]
    
    # Map to unified classes
    df['attack_class'] = df[label_col].map(lambda x: UNIFIED_CLASSES.get(str(x).strip(), 10))
    
    # Get numeric features
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c != 'attack_class']
    
    X = df[feature_cols].values
    y = df['attack_class'].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    print(f"   Loaded {len(X)} samples, {X.shape[1]} features")
    return X, y


def load_unsw_nb15(sample_frac: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load UNSW-NB15 dataset"""
    unsw_dir = DATASETS_DIR / "unsw-nb15"
    
    files = list(unsw_dir.glob("UNSW-NB15*.csv"))
    if not files:
        print(f"⚠️ UNSW-NB15 not found at {unsw_dir}")
        return None, None
    
    print("📂 Loading UNSW-NB15...")
    
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding='latin-1', low_memory=False)
            dfs.append(df)
        except:
            pass
    
    if not dfs:
        return None, None
    
    df = pd.concat(dfs, ignore_index=True)
    
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42)
    
    # Find label column
    label_cols = [c for c in df.columns if 'attack_cat' in c.lower() or 'label' in c.lower()]
    if not label_cols:
        return None, None
    
    label_col = label_cols[0]
    df['attack_class'] = df[label_col].map(lambda x: UNIFIED_CLASSES.get(str(x).strip(), 10))
    
    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in feature_cols if c not in ['attack_class', 'label', 'Label']]
    
    X = df[feature_cols].values
    y = df['attack_class'].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    
    print(f"   Loaded {len(X)} samples, {X.shape[1]} features")
    return X, y


def load_combined_datasets(sample_frac: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Load and combine all available datasets"""
    print("=" * 60)
    print("🔄 Loading Combined Datasets")
    print("=" * 60)
    
    datasets = []
    
    # Load CICIDS 2017
    X_cicids, y_cicids = load_cicids2017(sample_frac)
    if X_cicids is not None:
        datasets.append((X_cicids, y_cicids, "CICIDS2017"))
    
    # Load UNSW-NB15
    X_unsw, y_unsw = load_unsw_nb15(sample_frac)
    if X_unsw is not None:
        datasets.append((X_unsw, y_unsw, "UNSW-NB15"))
    
    if not datasets:
        print("❌ No datasets found!")
        return None, None
    
    # Combine datasets (align features)
    if len(datasets) == 1:
        X, y = datasets[0][0], datasets[0][1]
    else:
        # Find minimum feature count
        min_features = min(d[0].shape[1] for d in datasets)
        
        X_list = [d[0][:, :min_features] for d in datasets]
        y_list = [d[1] for d in datasets]
        
        X = np.vstack(X_list)
        y = np.concatenate(y_list)
        
        print(f"\\n📊 Combined: {len(X)} samples, {X.shape[1]} features")
    
    # Class distribution
    print("\\nClass Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in sorted(zip(unique, counts), key=lambda x: -x[1]):
        print(f"   {CLASS_NAMES[cls]}: {cnt:,}")
    
    return X, y


if __name__ == "__main__":
    X, y = load_combined_datasets(sample_frac=0.1)
    if X is not None:
        print(f"\\nFinal: {X.shape}")
'''
    
    loader_path = DATASETS_DIR.parent / "combined_loader.py"
    loader_path.write_text(loader_code)
    print(f"   ✅ Created {loader_path}")
    return True


def main():
    print("=" * 60)
    print("📦 PCDS Dataset Downloader")
    print("=" * 60)
    
    # Create directories
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download each dataset
    download_unsw_nb15()
    download_cicids_2018()
    download_ember()
    
    # Create combined loader
    create_combined_loader()
    
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    print("""
Due to dataset hosting policies, most datasets require manual download.

📥 Next Steps:

1. UNSW-NB15 (~700MB):
   - Go to: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
   - Download and extract to: ml/datasets/unsw-nb15/

2. CSE-CIC-IDS 2018 (~7GB):
   - Run: aws s3 sync --no-sign-request s3://cse-cic-ids2018 ml/datasets/cicids2018/

3. EMBER (~2GB):
   - Run: pip install ember && ember-download 2018

After downloading, run train_combined.py to train on all data!
""")


if __name__ == "__main__":
    main()
