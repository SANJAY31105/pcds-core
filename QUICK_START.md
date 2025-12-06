# PCDS Enterprise - Quick Start Deployment Guide

## 🚀 Deploy in Under 30 Minutes

---

## Prerequisites

**Required**:
- Python 3.9+ installed
- Node.js 18+ installed  
- Git installed
- 4GB RAM minimum
- 10GB disk space

---

## Step 1: Clone & Setup (5 minutes)

```bash
# Clone repository
git clone <your-repo-url>
cd pcds-core

# Backend setup
cd backend
pip install -r requirements.txt

# Frontend setup
cd ../frontend
npm install
```

---

## Step 2: Database Initialization (5 minutes)

```bash
cd backend

# Initialize database
python init_database.py

# (Optional) Load sample data
python ai_attack_simulation.py
# This will create 10,000-200,000 sample attacks
# Press Ctrl+C to stop at any time
```

---

## Step 3: Start Services (2 minutes)

**Terminal 1 - Backend**:
```bash
cd backend
python main_v2.py
```

**Terminal 2 - Frontend**:
```bash
cd frontend
npm run dev
```

---

## Step 4: Access System

**Open browser**: `http://localhost:3000`

**Default credentials** (if auth enabled):
- Username: `admin`
- Password: Check `backend/.env` file

---

## 🎯 Verification

**Check backend**: `http://localhost:8000/docs` (API documentation)  
**Check frontend**: `http://localhost:3000` (Dashboard)

**Quick test**:
```bash
cd backend
python check_status.py
```

Should show detections and entities count.

---

## 🔧 Configuration

**Backend** (`backend/.env`):
```env
DATABASE_PATH=./data/pcds.db
LOG_LEVEL=INFO
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
```

**Frontend** (`frontend/.env.local`):
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## 📊 Performance Optimization

**For production** (100K+ records):
```bash
cd backend
python optimize_db.py  # Adds database indexes
```

This improves query speed from 100ms → 2-4ms.

---

## 🚨 Troubleshooting

**Backend won't start**:
- Check Python version: `python --version` (need 3.9+)
- Check dependencies: `pip install -r requirements.txt`
- Check port 8000: `netstat -ano | findstr :8000`

**Frontend won't start**:
- Check Node version: `node --version` (need 18+)
- Delete `node_modules`, run `npm install` again
- Check port 3000: `netstat -ano | findstr :3000`

**No data showing**:
- Run `check_status.py` to verify database
- Check browser console (F12) for errors
- Verify backend API: `http://localhost:8000/api/v2/dashboard/overview`

---

## 📚 Next Steps

**After deployment**:
1. Review `DEMO_SCRIPT.md` for presentation
2. Check `TESTING_CHECKLIST.md` for verification
3. Read `COLLEGE_IT_PROPOSAL.md` for deployment details
4. See `DEPLOYMENT_CHECKLIST.md` for production setup

---

## 🎓 College Network Deployment

**For college IT infrastructure**:
1. Request server with specs:
   - 8GB RAM
   - 50GB storage
   - Ubuntu 20.04 or Windows Server
2. Deploy using Docker (see `DOCKER_DEPLOYMENT.md`)
3. Configure reverse proxy (see `PRODUCTION_DEPLOYMENT.md`)
4. Enable HTTPS with SSL certificate

---

**Questions?** Check documentation or run `python main_v2.py --help`

**Ready to impress your college!** 🚀
