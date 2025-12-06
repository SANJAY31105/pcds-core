# PCDS Enterprise - Share with Friends Guide
## How to Let Your Friends Test the Application

You have **3 options** - choose based on how technical you want to get:

---

## 🚀 Option 1: Quick & Easy (ngrok - Recommended for Testing)

**Best for:** Quick sharing, no configuration needed  
**Time:** 5 minutes  
**Cost:** Free

### Steps:

1. **Install ngrok**
   ```bash
   # Download from https://ngrok.com/download
   # Or use winget on Windows:
   winget install ngrok
   ```

2. **Start your backend** (if not already running)
   ```bash
   cd backend
   python main_v2.py
   ```

3. **Start your frontend** (in another terminal)
   ```bash
   cd frontend
   npm run dev
   ```

4. **Expose backend with ngrok**
   ```bash
   # Open a new terminal
   ngrok http 8000
   ```
   
   You'll see something like:
   ```
   Forwarding: https://abc123.ngrok.io -> http://localhost:8000
   ```

5. **Expose frontend with ngrok** (another terminal)
   ```bash
   ngrok http 3000
   ```
   
   You'll get:
   ```
   Forwarding: https://def456.ngrok.io -> http://localhost:3000
   ```

6. **Update frontend API URL**
   - Edit `frontend/lib/api.ts`
   - Change `http://localhost:8000` to your ngrok backend URL
   - Example: `https://abc123.ngrok.io`

7. **Update backend CORS**
   - Edit `backend/config/settings.py`
   - Add your ngrok frontend URL to CORS_ORIGINS
   - Example: `["http://localhost:3000", "https://def456.ngrok.io"]`

8. **Restart both servers**
   - Stop both (Ctrl+C)
   - Start again

9. **Share the frontend ngrok URL** with friends!
   - Send them: `https://def456.ngrok.io`
   - Login: `admin@pcds.local` / `admin123`

**Pros:** Super easy, works instantly  
**Cons:** URLs change every time you restart ngrok (unless you have paid account)

---

## 🏠 Option 2: Local Network (Same WiFi Required)

**Best for:** Testing with friends in same location  
**Time:** 10 minutes  
**Cost:** Free

### Steps:

1. **Find your local IP address**
   ```bash
   # Windows: Open Command Prompt
   ipconfig
   
   # Look for "IPv4 Address" under your WiFi adapter
   # Example: 192.168.1.100
   ```

2. **Update backend to listen on all interfaces**
   - Already configured: `HOST = "0.0.0.0"` in settings

3. **Allow firewall access**
   ```bash
   # Windows Firewall
   # Allow ports 3000 and 8000 through Windows Firewall
   # Go to: Control Panel > Windows Defender Firewall > Advanced Settings
   # Create Inbound Rule: Allow TCP ports 3000, 8000
   ```

4. **Update frontend API URL**
   - Edit `frontend/lib/api.ts`
   - Change `localhost` to your IP
   - Example: `http://192.168.1.100:8000`

5. **Update backend CORS**
   - Edit `backend/config/settings.py`
   - Add: `http://192.168.1.100:3000`

6. **Share your URLs** with friends on same WiFi:
   - Frontend: `http://192.168.1.100:3000`
   - Backend: `http://192.168.1.100:8000`

**Pros:** No external services, full speed  
**Cons:** Only works on same network, firewall setup needed

---

## ☁️ Option 3: Cloud Deployment (Production-Ready)

**Best for:** Permanent access, real testing environment  
**Time:** 30-60 minutes  
**Cost:** $5-10/month (or free tier)

### A. Using Railway (Easiest Cloud Option)

1. **Create account** at https://railway.app (free tier available)

2. **Install Railway CLI**
   ```bash
   npm install -g @railway/cli
   railway login
   ```

3. **Initialize project**
   ```bash
   cd pcds-core
   railway init
   ```

4. **Deploy backend**
   ```bash
   cd backend
   railway up
   ```

5. **Deploy frontend**
   ```bash
   cd ../frontend
   railway up
   ```

6. Railway will give you permanent URLs like:
   - Backend: `https://pcds-backend-production.up.railway.app`
   - Frontend: `https://pcds-frontend-production.up.railway.app`

### B. Using Vercel (For Frontend) + Railway (For Backend)

**Frontend on Vercel:**
```bash
cd frontend
npx vercel
# Follow prompts
```

**Backend on Railway:**
```bash
cd backend
railway up
```

### C. Using Docker + DigitalOcean/AWS

See `DOCKER_DEPLOYMENT.md` for full instructions.

---

## 🔒 Important Security Notes

**Before sharing with friends:**

1. **Change admin password**
   ```sql
   -- Connect to your database and update the admin password
   -- Use a strong password, not "admin123"
   ```

2. **Add their user accounts**
   - Create separate accounts for each friend
   - Don't share the admin account

3. **Use HTTPS**
   - ngrok provides HTTPS automatically ✅
   - For cloud: Use platform's built-in SSL ✅
   - For local network: Consider self-signed certificate

4. **Set environment variables**
   - Create `.env` file with strong SECRET_KEY
   - Don't commit `.env` to git

---

## 📝 Quick Start Script (Recommended)

Create `start-for-friends.bat` on Windows:

```batch
@echo off
echo ===================================
echo  PCDS Enterprise - Starting...
echo ===================================

:: Start backend
cd backend
start cmd /k "python main_v2.py"
timeout /t 5

:: Start frontend
cd ../frontend
start cmd /k "npm run dev"
timeout /t 10

:: Start ngrok for backend
start cmd /k "ngrok http 8000"
timeout /t 5

:: Start ngrok for frontend
start cmd /k "ngrok http 3000"

echo.
echo ===================================
echo  All services started!
echo  Check ngrok terminals for URLs
echo ===================================
pause
```

Run: Double-click `start-for-friends.bat`

---

## 🎯 Recommended Approach

**For quick testing (this weekend):**
→ Use **Option 1 (ngrok)** - 5 minutes setup

**For ongoing testing (next few weeks):**
→ Use **Option 3A (Railway)** - Free tier, permanent URLs

**For demo to investors/customers:**
→ Use **Option 3B (Vercel + Railway)** - Professional, fast, reliable

---

## 🐛 Troubleshooting

**Friends can't connect:**
- Check Windows Firewall settings
- Verify they're on same WiFi (Option 2)
- Verify ngrok URLs are correct (Option 1)
- Check CORS settings in backend

**Login not working:**
- Verify database initialized (admin user exists)
- Check browser console for errors
- Verify backend is responding: `http://your-url/api/docs`

**Slow performance:**
- ngrok free tier has some latency (normal)
- Consider upgrading ngrok or using cloud deployment

---

## 📞 Support

If you need help:
1. Check backend logs (terminal running `python main_v2.py`)
2. Check frontend logs (browser console F12)
3. Check ngrok dashboard: https://dashboard.ngrok.com

---

**Ready to share?** Pick your option and follow the steps!

**Pro Tip:** Start with Option 1 (ngrok) - it's the fastest way to get your friends testing within 5 minutes!
