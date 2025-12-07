# PCDS Enterprise - Administrator Guide v2.0

## Quick Start

### Starting the System

```bash
# Terminal 1: Backend API
cd backend
python main_v2.py

# Terminal 2: Frontend Dashboard
cd frontend
npm run dev
```

**Access Dashboard:** http://localhost:3000
**Default Login:** admin / admin123

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PCDS Enterprise                          │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Next.js)          │  Backend (FastAPI)           │
│  - Dashboard                 │  - ML Engine v3.0            │
│  - Detections                │  - Decision Engine           │
│  - Approvals                 │  - Playbook Engine           │
│  - Timeline                  │  - MITRE Mapper              │
│  - Playbooks                 │  - Email Notifications       │
└─────────────────────────────────────────────────────────────┘
                              ↓
           SQLite Database (pcds_enterprise.db)
```

---

## Key Features

### 1. Detection Engine
- Analyzes network metadata (IPs, ports, protocols)
- ML-based anomaly detection
- MITRE ATT&CK technique mapping (155 techniques)

### 2. Decision Engine (SOAR)
- Policy-based automated responses
- Human-in-the-loop approval workflow
- All enforcement actions require approval by default

### 3. Response Playbooks
- 7 pre-built enterprise playbooks
- Ransomware, C2, Lateral Movement handlers
- Customizable action sequences

### 4. Alert Notifications
- Email alerts for critical detections
- Approval request notifications
- Configure SMTP in environment variables

---

## Configuration

### Environment Variables

```bash
# Email Notifications
PCDS_SMTP_HOST=smtp.gmail.com
PCDS_SMTP_PORT=587
PCDS_SMTP_USER=your-email@gmail.com
PCDS_SMTP_PASS=your-app-password
PCDS_ALERT_EMAIL=security-team@college.edu

# Database
PCDS_DB_PATH=./pcds_enterprise.db

# Security
PCDS_SECRET_KEY=your-secret-key-here
PCDS_JWT_EXPIRY=24
```

### Decision Engine Policies

Edit `backend/automation/decision_engine.py` to modify response policies:

```python
ResponsePolicy(
    name="Ransomware Auto-Response",
    threat_types=["ransomware"],
    severity_threshold="high",
    confidence_threshold=0.7,
    actions=["isolate_host", "kill_process"],
    require_approval=True  # Set False for auto-execute
)
```

---

## Daily Operations

### Morning Checklist
1. Check dashboard for overnight alerts
2. Review pending approvals
3. Verify system health status

### Running Attack Simulation (Demo)
```bash
cd backend
python demo_attack.py
```

### Database Maintenance
```bash
# Clean old data
cd backend
python cleanup_db.py
```

### View Logs
```bash
# Backend logs appear in terminal
# Check for ERROR or WARNING messages
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Dashboard shows "No detections" | Run `python demo_attack.py` to generate test data |
| Login fails | Restart backend, check admin password |
| API errors | Check if backend is running on port 8000 |
| Frontend not loading | Run `npm run dev` in frontend folder |

---

## Security Best Practices

1. **Change default password** after first login
2. **Enable email notifications** for critical alerts
3. **Review approval queue** at least twice daily
4. **Backup database** weekly (use `ops/backup.sh`)
5. **Update MITRE mappings** quarterly

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v2/dashboard/overview` | GET | Dashboard metrics |
| `/api/v2/detections/` | GET | List detections |
| `/api/v2/response/pending` | GET | Pending approvals |
| `/api/v2/response/approve/{id}` | POST | Approve action |
| `/api/v2/playbooks/` | GET | List playbooks |

---

## Contact & Support

**Repository:** https://github.com/SANJAY31105/pcds-core
**Version:** 2.0.0
**Last Updated:** December 2025
