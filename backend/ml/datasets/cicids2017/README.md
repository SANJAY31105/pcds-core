# CIC-IDS 2017 Dataset

## Download
**URL:** https://www.unb.ca/cic/datasets/ids-2017.html

Click on "CSV Files" to download the labeled network traffic data.

## Files to Place Here
After downloading, extract and place these CSV files:
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Monday-WorkingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`

## Attack Types Included
| Attack Type | Day |
|-------------|-----|
| Brute Force (FTP, SSH) | Tuesday |
| DoS (Hulk, GoldenEye, Slowloris) | Wednesday |
| Web Attacks (XSS, SQL Injection) | Thursday Morning |
| Infiltration | Thursday Afternoon |
| Botnet | Friday |
| PortScan | Friday |
| DDoS | Friday |

## Usage
```python
from ml.training import CICIDS2017Loader

loader = CICIDS2017Loader()
X, y_binary, y_category = loader.load()
```
