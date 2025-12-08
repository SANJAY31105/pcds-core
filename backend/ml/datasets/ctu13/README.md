# CTU-13 Botnet Dataset

## Download
**URL:** https://www.stratosphereips.org/datasets-ctu13

## Description
13 different botnet scenarios captured from real malware:
- Neris
- Rbot
- Virut
- Menti
- Sogou
- Murlo
- NSIS.ay

## Files to Place Here
Download and place the CSV/binetflow files:
- `capture20110810.binetflow` (Scenario 1)
- `capture20110811.binetflow` (Scenario 2)
- ... etc up to Scenario 13

Or the processed CSV versions.

## What This Dataset Contains
- **Botnet C2 traffic:** Command and control communication
- **Background traffic:** Normal user activity
- **Labeled flows:** Each flow is labeled as botnet or normal

## Features
- Flow duration
- Protocol
- Source/Dest IPs and ports
- Packets sent/received
- Bytes transferred

## Usage
```python
from ml.training import CTU13Loader

loader = CTU13Loader()
X, y_binary, y_category = loader.load()
```
