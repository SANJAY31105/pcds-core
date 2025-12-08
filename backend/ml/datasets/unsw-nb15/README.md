# UNSW-NB15 Dataset

## Download
**URL:** https://research.unsw.edu.au/projects/unsw-nb15-dataset

Or direct link:
https://cloudstor.aarnet.edu.au/plus/index.php/s/2DhnLGDdEECo4ys

## Files to Place Here
Download and place:
- `UNSW-NB15_1.csv`
- `UNSW-NB15_2.csv`
- `UNSW-NB15_3.csv`
- `UNSW-NB15_4.csv`
- `UNSW_NB15_training-set.csv`
- `UNSW_NB15_testing-set.csv`

## Attack Types Included
| Category | Description |
|----------|-------------|
| Fuzzers | Random data injection |
| Analysis | Port scan, spam |
| Backdoors | Bypass authentication |
| DoS | Denial of Service |
| Exploits | Vulnerability exploitation |
| Generic | Known attack techniques |
| Reconnaissance | Info gathering |
| Shellcode | Code injection |
| Worms | Self-replicating malware |

## Features (49 total)
- Flow features: duration, packets, bytes
- Connection features: src/dst IPs, ports
- Time features: inter-arrival time
- Protocol features: TCP flags, HTTP methods

## Usage
```python
from ml.training import UNSWNB15Loader

loader = UNSWNB15Loader()
X, y_binary, y_category = loader.load()
```
