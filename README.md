# 🛡️ PCDS - Predictive Cyber Defence System

<div align="center">

**AI-Powered Predictive Cybersecurity Platform**

*Stop attacks before they start*

[![Imagine Cup 2026](https://img.shields.io/badge/Imagine%20Cup-2026-00a2ed?style=for-the-badge&logo=microsoft)](https://imaginecup.microsoft.com/)
[![Azure OpenAI](https://img.shields.io/badge/Azure%20OpenAI-GPT--4o-0078d4?style=for-the-badge&logo=microsoft-azure)](https://azure.microsoft.com/products/ai-services/openai-service)
[![Accuracy](https://img.shields.io/badge/Accuracy-88.3%25-10a37f?style=for-the-badge)](/)

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-00a393?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.1.0-000000?style=flat-square&logo=next.js)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3.3-3178c6?style=flat-square&logo=typescript)](https://www.typescriptlang.org/)

</div>

---

## 💡 The Problem

> **Every 39 seconds**, a business loses **$4.45 million** to a cyberattack they never saw coming.

Traditional security tools (SIEMs, EDRs) are **reactive** — they alert after attacks begin. SOC analysts drown in 10,000+ daily alerts, with 95% being false positives.

## 🚀 Our Solution

**PCDS is a predictive decision-intelligence layer** that sits above SIEMs, giving security teams **hours to days of warning** instead of minutes.

| Traditional Tools | PCDS |
|-------------------|------|
| Alert after attack | Predict before attack |
| 15% false positives | 2.8% false positives |
| Manual investigation | AI-powered explanations |
| Reactive response | Proactive prevention |

---

## 📊 ML Performance

*Tested on 5.5M+ samples from two industry-standard datasets:*
- **UNSW-NB15** (Australia) - 2.95M network intrusion samples
- **CICIDS 2017** (Canada) - 2.8M attack scenario samples

| Metric | PCDS | Industry Avg |
|--------|------|--------------|
| **Accuracy** | 88.3% | 78% |
| **Precision** | 90.7% | 75% |
| **False Positive Rate** | 2.8% | 15% |
| **Detection Latency** | 1.9ms | 50ms+ |

### 5-Model Ensemble
- 🧠 LSTM Sequence Detector (temporal patterns)
- 🌲 Random Forest Classifier (feature-based)
- 🔍 Isolation Forest (anomaly detection)
- 👤 Behavioral Analyzer (UEBA)
- 🌐 DGA Detector CNN (malicious domains)

---

## ☁️ Azure Integration

PCDS is built natively on **Microsoft Azure**:

| Azure Service | Purpose |
|---------------|---------|
| **Azure OpenAI (GPT-4o)** | Natural language threat explanations |
| **Azure Machine Learning** | Scalable model training & deployment |
| **Azure Cognitive Services** | Intelligent threat analysis |

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🔮 **Predictive Timeline** | See attacks developing 72+ hours before execution |
| 🤖 **AI Copilot** | Ask questions in natural language, get expert answers |
| ⚡ **Kill Chain Visualizer** | Track attack progression in real-time |
| 🎯 **MITRE ATT&CK** | Full tactics & techniques mapping |
| 🛡️ **SOAR Automation** | Automated response playbooks with human approval |
| 📊 **ML Transparency** | Explainable AI with confidence scores |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              PCDS Enterprise Architecture                    │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Next.js 14)                                       │
│  ├── Dashboard with Prediction Timeline                     │
│  ├── AI Copilot (Azure OpenAI GPT-4o)                       │
│  ├── Kill Chain Visualizer                                   │
│  └── 25+ Feature Pages                                       │
├─────────────────────────────────────────────────────────────┤
│  Backend (FastAPI)                                           │
│  ├── REST API (25+ endpoints)                               │
│  ├── WebSocket (real-time updates)                          │
│  ├── Authentication (JWT)                                    │
│  └── SOAR Automation                                         │
├─────────────────────────────────────────────────────────────┤
│  ML Engine (PyTorch)                                         │
│  ├── 5-Model Ensemble Detector                              │
│  ├── LSTM Sequence Analyzer                                  │
│  ├── Behavioral Analytics (UEBA)                            │
│  └── Azure OpenAI Integration                                │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                  │
│  ├── SQLite (detections, entities, MITRE)                   │
│  └── Real-time Event Streaming                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+
- Azure OpenAI API key (optional, has fallback)

### Installation

```bash
# Clone repository
git clone https://github.com/SANJAY31105/pcds-core.git
cd pcds-core

# Backend setup
cd backend
pip install -r requirements.txt

# Configure Azure OpenAI (optional)
# Create .env file with:
# AZURE_OPENAI_ENDPOINT=your-endpoint
# AZURE_OPENAI_KEY=your-key
# AZURE_OPENAI_DEPLOYMENT=your-deployment

python main_v2.py

# Frontend setup (new terminal)
cd frontend
npm install
npm run dev
```

### Access
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Demo Login
- Email: `admin@pcds.com`
- Password: `admin123`

---

## 📁 Project Structure

```
pcds-core/
├── backend/
│   ├── main_v2.py              # FastAPI application
│   ├── api/v2/                 # REST API endpoints
│   ├── ml/                     # PyTorch models & Azure AI
│   │   ├── ensemble_detector.py
│   │   ├── lstm_detector.py
│   │   └── azure_ai_service.py
│   ├── detections/             # Detection engine
│   └── config/                 # Settings
│
├── frontend/
│   ├── app/                    # Next.js pages (25+)
│   ├── components/             # React components
│   └── lib/                    # API client
│
├── PITCH_DECK_15_SLIDES.md     # Pitch content
├── DEMO_VIDEO_SCRIPT.md        # Demo narration
└── PITCH_VIDEO_SCRIPT.md       # Pitch narration
```

---

## 🎯 Demo Highlights

1. **Dashboard** → Prediction timeline showing threats before they execute
2. **AI Copilot** → "Explain this threat" with GPT-4o response
3. **Live Feed** → Real-time attack detection (phishing, C2, ransomware)
4. **ML Metrics** → 88.3% accuracy with full transparency
5. **MITRE Matrix** → Complete attack technique coverage

---

## 👥 Team

**Keshav Memorial Institute of Technology (KMIT), Hyderabad**

Computer Science & Engineering students focused on:
- 🤖 Machine Learning & AI
- 🔐 Cybersecurity Research
- ☁️ Cloud-Native Systems

---

## 🏆 Imagine Cup 2026

This project is our submission for **Microsoft Imagine Cup 2026**.

> *"PCDS is a predictive decision-intelligence layer that sits above SIEMs, giving security teams hours to days of warning instead of minutes."*

---

## 📝 License

MIT License

---

<div align="center">

**PCDS** - Predictive Cyber Defence System

*Transforming cybersecurity from reactive detection to proactive prevention*

Built with ❤️ using Microsoft Azure | 2024-2025

</div>
   
 