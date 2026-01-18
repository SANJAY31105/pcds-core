# PCDS Network Agent

## Overview
Enterprise-grade network traffic capture and threat detection agent.

## Files
| File | Description |
|------|-------------|
| `capture.py` | Packet capture via Scapy/Npcap |
| `features.py` | ML feature extraction (18 features) |
| `analyzer.py` | Local ML inference (sklearn/PyTorch/rules) |
| `sender.py` | Cloud API sender with batching |
| `parsers.py` | HTTP/DNS/TLS protocol parsing |
| `demo.py` | Full local demo with colored output |
| `simulator.py` | Attack simulation for testing |
| `main.py` | Production CLI runner |

## Quick Start

### List Network Interfaces
```bash
python list_interfaces.py
```

### Test Packet Capture
```bash
python test_capture.py
```

### Run Local Demo
```bash
python demo.py
```

### Run Attack Simulator (separate terminal)
```bash
python simulator.py -t 127.0.0.1 -a all
```

### Run Production Agent
```bash
python -m agent.main -i "{INTERFACE-GUID}" -k "your-api-key"
```

## Requirements
- Python 3.10+
- Npcap (Windows) or libpcap (Linux)
- scapy, requests, numpy

```bash
pip install scapy requests numpy
```

## Features
- ✅ Real-time packet capture
- ✅ Flow aggregation
- ✅ 18 ML features extracted
- ✅ Local threat detection (no cloud needed)
- ✅ MITRE ATT&CK mapping
- ✅ HTTP/DNS/TLS deep inspection
- ✅ Attack simulation for testing
- ✅ Colored terminal output
