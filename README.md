# 🛡️ PCDS - Predictive Cyber Defence System

<div align="center">

**AI-Powered Cybersecurity Platform for Real-Time Threat Detection and Prevention**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-00a393?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.1.0-000000?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3.3-3178c6?style=for-the-badge&logo=typescript)](https://www.typescriptlang.org/)

</div>

## 🌟 Overview

PCDS is an **enterprise-grade cybersecurity platform** that transforms threat detection from reactive to proactive. Using cutting-edge AI/ML techniques, real-time data analysis, and predictive algorithms, PCDS identifies and mitigates cyber threats before they can cause damage.

### ✨ Key Features

- **🤖 AI-Powered Detection**: PyTorch LSTM models for real-time anomaly detection
- **⚡ Blazing Fast**: Sub-100ms threat detection with WebSocket real-time streaming
- **📊 Premium Dashboard**: Next.js 14 with stunning Tailwind CSS UI and Framer Motion animations
- **🎯 Predictive Analytics**: Forecast vulnerabilities before breaches occur
- **🔄 Real-Time Monitoring**: Live network traffic analysis and alerts
- **📈 Advanced Visualizations**: D3.js and Recharts for threat intelligence
- **🐳 Docker Ready**: One-command deployment with Docker Compose
- **🔒 Enterprise Security**: JWT authentication and RBAC support

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Next.js 14 Frontend                   │
│   TypeScript │ Tailwind CSS │ Framer Motion │ Recharts  │
└────────────────────┬────────────────────────────────────┘
                     │ WebSocket + REST API
┌────────────────────┴────────────────────────────────────┐
│                   FastAPI Backend                        │
│         Async │ WebSocket │ Background Tasks             │
├──────────────┬──────────────┬──────────────────────────┤
│  PyTorch ML  │ Threat Engine│   Redis   │  PostgreSQL  │
│     LSTM     │  Detection   │  Caching  │   Storage    │
└──────────────┴──────────────┴───────────┴──────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+**
- **Node.js 20+**
- **Docker & Docker Compose** (recommended)

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/your username/pcds-core.git
cd pcds-core

# Start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Option 2: Manual Setup

**Backend Setup:**

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run backend
python main.py
```

**Frontend Setup:**

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

## 📁 Project Structure

```
pcds-core/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── models.py               # Pydantic models
│   ├── database.py             # PostgreSQL configuration
│   ├── redis_client.py         # Redis client
│   ├── websocket_manager.py    # WebSocket handler
│   ├── threat_engine.py        # Threat detection engine
│   ├── data_generator.py       # Data simulation
│   ├── ml/
│   │   └── anomaly_detector.py # PyTorch LSTM model
│   ├── requirements.txt        # Python dependencies
│   └── Dockerfile
├── frontend/
│   ├── app/
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Main page
│   │   └── globals.css         # Global styles
│   ├── components/
│   │   ├── Dashboard.tsx       # Main dashboard
│   │   ├── ThreatCard.tsx      # Threat display
│   │   ├── AlertPanel.tsx      # Live alerts
│   │   ├── StatsCard.tsx       # Stats widget
│   │   └── charts/
│   │       └── NetworkChart.tsx
│   ├── hooks/
│   │   └── useWebSocket.ts     # WebSocket hook
│   ├── lib/
│   │   └── api.ts              # API client
│   ├── types/
│   │   └── index.ts            # TypeScript types
│   ├── package.json
│   ├── tsconfig.json
│   ├── tailwind.config.ts
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

## 🎨 UI Features

### Premium Cyber-Themed Design

- **Glassmorphism Effects**: Modern frosted-glass UI components
- **Dynamic Animations**: Framer Motion for smooth 60fps transitions
- **Color System**: 
  - Cyber Blue (`#00f0ff`) - Primary actions
  - Cyber Purple (`#b300ff`) - Accents  
  - Cyber Green (`#00ff85`) - Success states
  - Threat Colors - Severity-based (Critical, High, Medium, Low)
- **Responsive Design**: Mobile-first approach
- **Dark Mode**: Cyber-optimized dark theme
- **Real-Time Updates**: Live WebSocket data streaming

## 🤖 ML Model

### LSTM Anomaly Detector

- **Architecture**: 2-layer LSTM with 64 hidden units
- **Input Features**: 10-dimensional network event vectors
- **Training**: Simulated network traffic patterns
- **Inference**: < 100ms per prediction
- **Accuracy**: Configurable threshold (default: 0.7)

### Feature Extraction

- Packet size (normalized)
- Port number (normalized)
- Protocol encoding
- Source/Destination IP hashes
- Temporal features
- Session patterns

## 📊 API Endpoints

### REST API

- `GET /` - Root endpoint
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /api/v1/dashboard/stats` - Dashboard statistics
- `GET /api/v1/threats` - List threats
- `GET /api/v1/threats/{id}` - Get specific threat
- `GET /api/v1/countermeasures/{id}` - Get countermeasures
- `GET /api/v1/alerts` - List alerts
- `GET /api/v1/metrics/system` - System metrics
- `POST /api/v1/analyze` - Analyze network event

### WebSocket

- `WS /ws` - Real-time updates stream

**Message Types:**
- `connected` - Connection established
- `threat_detected` - New threat detected
- `alert` - New alert notification
- `system_metrics` - System health update
- `heartbeat` - Keep-alive ping

## 🔧 Configuration

### Environment Variables

**Backend (.env):**
```env
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/pcds_db
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key
```

**Frontend (.env.local):**
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
```

## 📈 Monitoring & Metrics

- **Prometheus Metrics**: `/metrics` endpoint
- **Health Checks**: `/health` endpoint  
- **Structured Logging**: JSON-formatted logs
- **System Metrics**: CPU, Memory, Network throughput
- **Threat Metrics**: Detection rate, risk scores

## 🛠️ Development

### Backend Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with hot reload
uvicorn main:app --reload

# Access API docs
open http://localhost:8000/docs
```

### Frontend Development

```bash
# Install dependencies
npm install

# Run development server
npm run dev

# Build for production
npm run build
```

## 🚢 Deployment

### Production Deployment

**Docker Compose:**
```bash
docker-compose -f docker-compose.prod.yml up -d
```

**Kubernetes:**
```bash
kubectl apply -f k8s/
```

### Environment Considerations

- Use strong `SECRET_KEY` in production
- Configure CORS for your domain
- Set up SSL/TLS certificates
- Enable database connection pooling
- Configure Redis persistence
- Set up log aggregation
- Enable Prometheus monitoring

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests  
cd frontend
npm test

# E2E tests
npm run test:e2e
```

## 📚 Tech Stack

### Backend
- **Framework**: FastAPI 0.109.0
- **ML**: PyTorch 2.1.2
- **Database**: PostgreSQL 15+ (SQLAlchemy async)
- **Cache**: Redis 7+ 
- **Monitoring**: Prometheus
- **Validation**: Pydantic v2

### Frontend
- **Framework**: Next.js 14.1.0 (React 18)
- **Language**: TypeScript 5.3.3
- **Styling**: Tailwind CSS 3.4.1
- **Animations**: Framer Motion 11.0.3
- **Charts**: Recharts 2.10.3, D3.js 7.8.5
- **Icons**: Lucide React

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with ❤️ using cutting-edge technologies
- Inspired by modern SOC (Security Operations Center) platforms
- Designed for maximum performance and user experience

---

<div align="center">

**Built to disrupt the cybersecurity market** 🚀

Made with 💙 by the PCDS Team

</div>
