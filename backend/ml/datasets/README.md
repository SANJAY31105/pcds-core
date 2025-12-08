# PCDS ML Datasets

## Directory Structure
```
datasets/
├── cicids2017/      ← Put CIC-IDS 2017 CSV files here
│   └── README.md
├── unsw-nb15/       ← Put UNSW-NB15 CSV files here
│   └── README.md
├── ctu13/           ← Put CTU-13 files here
│   └── README.md
└── README.md        ← This file
```

## Quick Download Links

| Dataset | Size | Link |
|---------|------|------|
| CIC-IDS 2017 | ~2.8 GB | [Download](https://www.unb.ca/cic/datasets/ids-2017.html) |
| UNSW-NB15 | ~700 MB | [Download](https://research.unsw.edu.au/projects/unsw-nb15-dataset) |
| CTU-13 | ~200 MB | [Download](https://www.stratosphereips.org/datasets-ctu13) |

## After Downloading

1. Extract the ZIP files
2. Place CSV files in the respective folders
3. Run training:

```python
from ml.training import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.run_full_pipeline(epochs=50)
```

## What Each Dataset Provides

| Dataset | Best For |
|---------|----------|
| CIC-IDS 2017 | DoS, DDoS, Brute Force, Web Attacks |
| UNSW-NB15 | Exploits, Shellcode, Worms |
| CTU-13 | Botnet, C2 Detection |
