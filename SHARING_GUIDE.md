# PCDS Enterprise - Sharing Guide for Real-World Testing

## 🚀 How to Share PCDS for Real-Life Testing

This guide covers **5 different methods** to share your PCDS Enterprise platform for real-world testing, from quickest (5 minutes) to most professional (production deployment).

---

## Method 1: Quick Share with Ngrok (⚡ 5 Minutes)

**Best For:** Quick demos, testing with friends, proof-of-concept  
**Cost:** FREE  
**Requires:** Active local servers

### Step 1: Install Ngrok

**Windows:**
```powershell
# Download from https://ngrok.com/download
# Or use chocolatey:
choco install ngrok
```

**After installation, sign up at https://ngrok.com and get your authtoken**

### Step 2: Configure Ngrok

```powershell
ngrok authtoken YOUR_AUTH_TOKEN_HERE
```

### Step 3: Expose Your PCDS Dashboard

```powershell
# Terminal 1 - Keep backend running
cd backend
python main_v2.py

# Terminal 2 - Keep frontend running  
cd frontend
npm run dev

# Terminal 3 - Expose frontend to internet
ngrok http 3000
```

### Step 4: Share the URL

Ngrok will give you a URL like:
```
Forwarding: https://abc123.ngrok-free.app -> http://localhost:3000
```

**Share this URL with anyone!** They can access your PCDS dashboard from anywhere in the world.

### ⚠️ Important Notes:
- URL changes every time you restart ngrok
- Free tier has session limits
- Anyone with the URL can access (add authentication!)
- Tunnel closes when you close terminal

### 🎯 **Perfect For:**
- Quick demos to clients
- Testing with remote team members
- Portfolio showcasing
- Getting feedback

---

## Method 2: Desktop App with Electron (⚡ 30 Minutes)

**Best For:** Distributable desktop application  
**Cost:** FREE  
**Platform:** Windows, Mac, Linux

### Step 1: Create Electron Wrapper

Create `electron-app/package.json`:

```json
{
  "name": "pcds-enterprise-desktop",
  "version": "2.0.0",
  "description": "PCDS Enterprise Desktop Application",
  "main": "main.js",
  "scripts": {
    "start": "electron .",
    "build-win": "electron-builder --win",
    "build-mac": "electron-builder --mac",
    "build-linux": "electron-builder --linux"
  },
  "build": {
    "appId": "com.pcds.enterprise",
    "productName": "PCDS Enterprise",
    "directories": {
      "output": "dist"
    },
    "win": {
      "target": "nsis",
      "icon": "assets/icon.ico"
    },
    "mac": {
      "target": "dmg",
      "icon": "assets/icon.icns"
    }
  },
  "dependencies": {
    "electron": "^28.0.0"
  },
  "devDependencies": {
    "electron-builder": "^24.9.1"
  }
}
```

### Step 2: Create Main Process

Create `electron-app/main.js`:

```javascript
const { app, BrowserWindow } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let backendProcess;

function startBackend() {
  // Start Python backend
  backendProcess = spawn('python', ['main_v2.py'], {
    cwd: path.join(__dirname, '../backend')
  });
  
  backendProcess.stdout.on('data', (data) => {
    console.log(`Backend: ${data}`);
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    },
    icon: path.join(__dirname, 'assets/icon.png')
  });

  // Wait for backend to start, then load frontend
  setTimeout(() => {
    mainWindow.loadURL('http://localhost:3000');
  }, 3000);

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.on('ready', () => {
  startBackend();
  createWindow();
});

app.on('window-all-closed', () => {
  if (backendProcess) {
    backendProcess.kill();
  }
  app.quit();
});
```

### Step 3: Build Desktop App

```powershell
cd electron-app
npm install
npm run build-win  # Creates .exe installer
```

### Step 4: Share the Installer

The installer will be in `electron-app/dist/PCDS Enterprise Setup.exe`

**Share this file!** Users just double-click to install.

### 🎯 **Perfect For:**
- Enterprise clients who want desktop installation
- Offline demos
- Trade shows
- Limited internet environments

---

## Method 3: Cloud Deployment (⚡ 1-2 Hours)

**Best For:** Professional, always-online access  
**Cost:** $5-50/month  
**Platform:** AWS, Azure, GCP, Heroku, DigitalOcean

### Option A: Deploy to Heroku (Easiest)

**Step 1: Install Heroku CLI**
```powershell
# Download from https://devcenter.heroku.com/articles/heroku-cli
```

**Step 2: Create Heroku Apps**

```powershell
# Login
heroku login

# Create backend app
heroku create pcds-backend

# Create frontend app  
heroku create pcds-frontend

# Add PostgreSQL database
heroku addons:create heroku-postgresql:mini -a pcds-backend

# Add Redis
heroku addons:create heroku-redis:mini -a pcds-backend
```

**Step 3: Deploy Backend**

```powershell
cd backend

# Create Procfile
echo "web: python main_v2.py" > Procfile

# Create runtime.txt
echo "python-3.11.0" > runtime.txt

# Deploy
git init
heroku git:remote -a pcds-backend
git add .
git commit -m "Deploy PCDS backend"
git push heroku master
```

**Step 4: Deploy Frontend**

```powershell
cd frontend

# Create Procfile
echo "web: npm start" > Procfile

# Deploy
git init
heroku git:remote -a pcds-frontend
git add .
git commit -m "Deploy PCDS frontend"
git push heroku master
```

**Your PCDS is now at:**
- Backend: `https://pcds-backend.herokuapp.com`
- Frontend: `https://pcds-frontend.herokuapp.com`

### Option B: Deploy to DigitalOcean (Recommended)

**Step 1: Create Droplet**
- Go to digitalocean.com
- Create Droplet (Ubuntu 22.04)
- Size: $12/month (2GB RAM, 1 CPU)

**Step 2: SSH and Setup**

```bash
# SSH to your droplet
ssh root@YOUR_DROPLET_IP

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
apt install docker-compose

# Clone your code
git clone YOUR_REPO_URL
cd pcds-core

# Start services
docker-compose up -d
```

**Step 3: Configure Domain (Optional)**

```bash
# Point domain to droplet IP
# Then configure nginx for HTTPS
apt install nginx certbot python3-certbot-nginx

# Get SSL certificate
certbot --nginx -d yourdomain.com
```

**Access at:** `https://yourdomain.com`

### Option C: AWS Elastic Beanstalk

**Step 1: Install EB CLI**
```powershell
pip install awsebcli
```

**Step 2: Initialize and Deploy**
```powershell
cd backend
eb init -p python-3.11 pcds-backend
eb create pcds-prod
eb open
```

### 🎯 **Perfect For:**
- Professional deployment
- 24/7 availability
- Multiple users
- Enterprise clients
- Portfolio showcase

---

## Method 4: Docker Image Distribution (⚡ 20 Minutes)

**Best For:** Easy installation for technical users  
**Cost:** FREE  
**Requires:** Docker Desktop

### Step 1: Build Docker Images

```powershell
# Build backend image
cd backend
docker build -t pcds-backend:latest .

# Build frontend image
cd ../frontend
docker build -t pcds-frontend:latest .
```

### Step 2: Save Images to File

```powershell
# Export images
docker save pcds-backend:latest | gzip > pcds-backend.tar.gz
docker save pcds-frontend:latest | gzip > pcds-frontend.tar.gz
```

### Step 3: Create Distribution Package

Create `run-pcds.bat` for Windows:

```batch
@echo off
echo Starting PCDS Enterprise...

REM Load Docker images
docker load < pcds-backend.tar.gz
docker load < pcds-frontend.tar.gz

REM Start services
docker-compose up -d

echo PCDS Enterprise is running!
echo.
echo Dashboard: http://localhost:3000
echo API: http://localhost:8000
echo.
echo Press Ctrl+C to stop
pause
```

### Step 4: Share the Package

Zip these files:
- `pcds-backend.tar.gz`
- `pcds-frontend.tar.gz`
- `docker-compose.yml`
- `run-pcds.bat`

**Share the ZIP!** Users just:
1. Install Docker Desktop
2. Extract ZIP
3. Run `run-pcds.bat`

### 🎯 **Perfect For:**
- Technical audience
- Offline installations
- Enterprise IT departments
- Reproducible environments

---

## Method 5: Mobile-Friendly PWA (⚡ 15 Minutes)

**Best For:** Mobile device testing  
**Cost:** FREE  
**Platform:** Any device with browser

### Step 1: Make Frontend a PWA

Create `frontend/public/manifest.json`:

```json
{
  "name": "PCDS Enterprise",
  "short_name": "PCDS",
  "description": "Predictive Cyber Defence System",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#0f172a",
  "theme_color": "#0ea5e9",
  "icons": [
    {
      "src": "/icon-192.png",
      "sizes": "192x192",
      "type": "image/png"
    },
    {
      "src": "/icon-512.png",
      "sizes": "512x512",
      "type": "image/png"
    }
  ]
}
```

### Step 2: Add Service Worker

The PWA can be installed on phones/tablets!

### Step 3: Deploy to Vercel (Free)

```powershell
cd frontend

# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Follow prompts, get URL like:
# https://pcds-enterprise.vercel.app
```

### 🎯 **Perfect For:**
- Mobile testing
- On-the-go demos
- Tablet interfaces
- Field testing

---

## 📊 Comparison Table

| Method | Setup Time | Cost | Best For | Accessibility |
|--------|-----------|------|----------|---------------|
| Ngrok | 5 min | FREE | Quick demos | Temporary URL |
| Desktop App | 30 min | FREE | Offline use | Installer file |
| Cloud Deploy | 1-2 hrs | $5-50/mo | Production | Always online |
| Docker Image | 20 min | FREE | Tech users | Portable |
| PWA | 15 min | FREE | Mobile | App-like |

---

## 🎯 Recommended Path

### For Quick Testing (Today):
1. **Use Ngrok** - Get sharing in 5 minutes
2. Share URL with friends/colleagues
3. Get immediate feedback

### For Professional Showcase (This Week):
1. **Deploy to DigitalOcean** - $12/month
2. Get custom domain
3. Add HTTPS
4. Share professional URL

### For Enterprise Clients (This Month):
1. **Create Desktop App** - Looks professional
2. **Docker Package** - Easy IT deployment
3. **Cloud Production** - 24/7 availability

---

## 🔒 Security Reminder

Before sharing publicly:

```powershell
# 1. Change default passwords
# Edit .env file

# 2. Enable HTTPS (for cloud deployments)
# Use Let's Encrypt / Certbot

# 3. Add rate limiting
# Configure in backend

# 4. Set up monitoring
# Use UptimeRobot, etc.

# 5. Regular backups
# Database export cron job
```

---

## 💡 Quick Start Recommendation

**For RIGHT NOW (next 5 minutes):**

```powershell
# Terminal 1 - Backend (already running)
cd backend
python main_v2.py

# Terminal 2 - Frontend (already running)
cd frontend
npm run dev

# Terminal 3 - Share with Ngrok
ngrok http 3000
```

**Copy the `https://` URL from ngrok and share it!**

Anyone can access your PCDS dashboard immediately! 🚀

---

## 📱 Testing Checklist

Once shared, have testers check:

- [ ] Can they access the dashboard?
- [ ] Can they see the 10,000 detections?
- [ ] Do filters work (Critical/High/Medium)?
- [ ] Can they view entities?
- [ ] Does MITRE mapping display?
- [ ] Can they create investigations?
- [ ] Mobile-friendly on phones?
- [ ] Performance acceptable?

---

## 🎓 Next Steps

1. **Choose your method** (Ngrok for quick start)
2. **Share the URL/installer**
3. **Collect feedback**
4. **Iterate and improve**
5. **Deploy to production** when ready

---

**Your PCDS is ready to share with the world!** 🌍🚀
