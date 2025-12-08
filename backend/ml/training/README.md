# ML Training Pipeline

## Overview
Complete ML training pipeline for training attack detection models on labeled datasets.

## Supported Datasets

| Dataset | Download | Features |
|---------|----------|----------|
| CIC-IDS 2017 | [UNB](https://www.unb.ca/cic/datasets/ids-2017.html) | Brute Force, DoS, DDoS, Botnet, PortScan |
| UNSW-NB15 | [UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | Exploits, Shellcode, Worms, Recon |
| CTU-13 | [Stratosphere](https://www.stratosphereips.org/datasets-ctu13) | Botnet C&C traffic |

## Directory Structure
```
ml/
├── training/
│   ├── __init__.py
│   ├── dataset_loaders.py      # Load CIC-IDS, UNSW, CTU-13
│   ├── feature_extraction.py   # Extract ML features
│   ├── attack_classifiers.py   # Per-attack models
│   └── training_pipeline.py    # Main training orchestration
├── datasets/
│   ├── cicids2017/             # Put CIC-IDS CSV files here
│   ├── unsw-nb15/              # Put UNSW-NB15 CSV files here
│   └── ctu13/                  # Put CTU-13 CSV files here
└── models/
    └── trained/                # Saved model weights
```

## Quick Start

```python
from ml.training import TrainingPipeline

# Initialize
pipeline = TrainingPipeline()

# Train on CIC-IDS 2017
pipeline.run_full_pipeline(['cicids2017'], epochs=50)

# Or train on all available datasets
pipeline.run_full_pipeline(epochs=100)
```

## Training Flow

1. **Load datasets** → CSV files with labeled attack traffic
2. **Extract features** → 64-dimensional feature vectors
3. **Balance classes** → SMOTE oversampling + undersampling
4. **Train classifiers** → Per-attack-category models
5. **Validate** → Test accuracy, F1, FPR, FNR
6. **Export** → Save weights for PCDS engine

## Attack Categories

- `brute_force` - FTP/SSH/Web brute force
- `dos` - Denial of Service
- `ddos` - Distributed DoS
- `botnet` - Botnet/C2 traffic
- `recon` - Port scanning, reconnaissance
- `exfiltration` - Data exfiltration
- `web_attack` - XSS, SQLi
- `exploit` - Exploits, shellcode
- `malware` - Malware, worms

## Requirements

```bash
pip install torch pandas numpy scikit-learn imbalanced-learn
```
