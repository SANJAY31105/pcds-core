# 🛡️ PCDS Enterprise - Network Detection & Response

<div align="center">

**AI-Powered Attack Signal Intelligence Platform**

[![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-00a393?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.1.0-000000?style=for-the-badge&logo=next.js)](https://nextjs.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3.3-3178c6?style=for-the-badge&logo=typescript)](https://www.typescriptlang.org/)

</div>

---

## 🌟 Overview

PCDS Enterprise is an **advanced Network Detection & Response (NDR)** platform that uses AI-powered threat detection, MITRE ATT&CK mapping, and automated response capabilities to protect enterprise networks.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🤖 **AI Detection** | PyTorch LSTM models for real-time anomaly detection |
| ⚡ **Real-Time** | Sub-100ms threat detection with WebSocket streaming |
| 🎯 **MITRE ATT&CK** | Full tactics & techniques mapping with coverage analytics |
| 🔍 **Entity Scoring** | AI-driven urgency assessment for hosts, IPs, and users |
| 📊 **Threat Hunting** | Built-in hunt queries for proactive threat discovery |
| 🤖 **Playbooks** | Automated response with approval workflows |
| 📈 **Executive Reports** | Compliance, metrics, and trend analysis |
| 🔐 **Enterprise Auth** | JWT authentication with role-based access |

---

## 🖥️ UI Design

### Clean Professional Theme
- **Background**: `#0a0a0a` (pure dark)
- **Cards/Panels**: `#141414` (subtle elevation)
- **Accent Color**: `#10a37f` (professional green)
- **Borders**: `#2a2a2a` (minimal)
- **Typography**: Inter font, clean hierarchy

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Global Search |
| `ESC` | Close modals |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Next.js 14 Frontend (Port 3000)            │
│   TypeScript │ Tailwind CSS │ Recharts │ Lucide Icons   │
└────────────────────┬────────────────────────────────────┘
                     │ REST API + WebSocket
┌────────────────────┴────────────────────────────────────┐
│              FastAPI Backend (Port 8000)                │
│         Async │ WebSocket │ Background Tasks            │
├──────────────┬──────────────┬──────────────────────────┤
│  PyTorch ML  │ Detection    │  SQLite    │   Redis     │
│     LSTM     │  Engine      │  Database  │   Cache     │
└──────────────┴──────────────┴────────────┴─────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 20+

### Installation

```bash
# Clone repository
git clone https://github.com/SANJAY31105/pcds-core.git
cd pcds-core

# Backend
cd backend
pip install -r requirements.txt
python main_v2.py

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

### Access
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### Default Login
- Email: `admin@pcds.com`
- Password: `admin123`

---

## 📁 Project Structure

```
pcds-core/
├── backend/
│   ├── main_v2.py              # FastAPI application
│   ├── api/v2/                 # API endpoints
│   │   ├── auth.py             # Authentication
│   │   ├── entities.py         # Entity management
│   │   ├── detections.py       # Threat detections
│   │   ├── playbooks.py        # Response playbooks
│   │   └── reports.py          # Executive reports
│   ├── detections/             # Detection engine
│   ├── engine/                 # Scoring & correlation
│   ├── ml/                     # PyTorch models
│   └── config/                 # Database config
│
└── frontend/
    ├── app/
    │   ├── page.tsx            # Dashboard
    │   ├── entities/           # Entity pages
    │   ├── detections/         # Detection pages
    │   ├── hunt/               # Threat hunting
    │   ├── mitre/              # MITRE ATT&CK  
    │   ├── playbooks/          # Response playbooks
    │   ├── reports/            # Reports
    │   └── live/               # Live feed
    ├── components/
    │   ├── Navigation.tsx      # Sidebar nav
    │   ├── GlobalSearch.tsx    # Ctrl+K search
    │   ├── ToastProvider.tsx   # Notifications
    │   └── Skeleton.tsx        # Loading states
    └── lib/
        └── api.ts              # API client
```

---

## 📊 Pages Overview

| Page | Features |
|------|----------|
| **Dashboard** | KPIs, severity distribution, recent detections, quick actions |
| **Entities** | Entity list with urgency scores, search, filtering |
| **Detections** | Real-time threat detections with MITRE mapping |
| **Approvals** | Pending response actions requiring approval |
| **Timeline** | Chronological attack progression view |
| **Investigations** | Active security investigations |
| **Playbooks** | Automated response playbooks |
| **Hunt** | Proactive threat hunting queries |
| **MITRE** | ATT&CK tactics/techniques coverage grid |
| **Live Feed** | Real-time event stream with pause/resume |
| **Reports** | Executive dashboards and metrics |

---

## 🔧 API Endpoints

### Authentication
- `POST /api/v2/auth/login` - User login
- `POST /api/v2/auth/register` - User registration

### Entities
- `GET /api/v2/entities` - List entities
- `GET /api/v2/entities/{id}` - Entity details
- `GET /api/v2/entities/stats` - Entity statistics

### Detections
- `GET /api/v2/detections` - List detections
- `GET /api/v2/detections/{id}` - Detection details

### WebSocket
- `WS /ws` - Real-time event stream

---

## 📚 Tech Stack

### Backend
- FastAPI 0.109.0
- PyTorch 2.1.2 (LSTM anomaly detection)
- SQLAlchemy (async)
- Pydantic v2
- JWT Authentication

### Frontend
- Next.js 14.1.0
- TypeScript 5.3.3
- Tailwind CSS 3.4.1
- Recharts
- Lucide Icons

---

## 📝 License

MIT License

---

<div align="center">

**PCDS Enterprise** - Predictive Cyber Defence System

Built by Sanjay | 2024

</div>
