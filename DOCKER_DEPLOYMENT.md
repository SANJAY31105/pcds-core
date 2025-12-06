# PCDS Enterprise - Docker Deployment Guide

## 🐳 Containerized Deployment

---

## Step 1: Create Dockerfile for Backend

**File**: `backend/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main_v2.py"]
```

---

## Step 2: Create Dockerfile for Frontend

**File**: `frontend/Dockerfile`

```dockerfile
FROM node:18-alpine AS builder

WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm ci

# Copy source
COPY . .

# Build
RUN npm run build

# Production image
FROM node:18-alpine

WORKDIR /app

COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/public ./public

EXPOSE 3000

CMD ["npm", "start"]
```

---

## Step 3: Create Docker Compose

**File**: `docker-compose.yml` (in root directory)

```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_PATH=/data/pcds.db
      - CORS_ORIGINS=http://localhost:3000,http://frontend:3000
    volumes:
      - ./data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    depends_on:
      - backend
    restart: unless-stopped

volumes:
  data:
```

---

## Step 4: Deploy

```bash
# Build and start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

---

## Step 5: Initialize Data (One-time)

```bash
# Access backend container
docker-compose exec backend bash

# Initialize database
python init_database.py

# (Optional) Load sample attacks
python ai_attack_simulation.py

# Exit container
exit
```

---

## 🚀 Access Application

- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`

---

## 🔧 Production Configuration

### Environment Variables

**Backend** (`.env.production`):
```env
DATABASE_PATH=/data/pcds.db
LOG_LEVEL=WARNING
CORS_ORIGINS=https://your-domain.com
SECRET_KEY=your-generated-secret-key
```

**Frontend** (`.env.production`):
```env
NEXT_PUBLIC_API_URL=https://api.your-domain.com
```

### Update docker-compose.yml

```yaml
services:
  backend:
    env_file:
      - ./backend/.env.production
    
  frontend:
    env_file:
      - ./frontend/.env.production
```

---

## 🔒 Security Hardening

### 1. Use Non-Root User

Add to `backend/Dockerfile`:
```dockerfile
RUN adduser --disabled-password --gecos '' appuser
USER appuser
```

### 2. Enable HTTPS

Use Nginx reverse proxy:

```yaml
# Add to docker-compose.yml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
    - "443:443"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
    - ./ssl:/etc/nginx/ssl
  depends_on:
    - backend
    - frontend
```

### 3. Resource Limits

```yaml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
```

---

## 📊 Health Monitoring

### Backend Health Endpoint

Add to `backend/main_v2.py`:
```python
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}
```

### Docker Health Check

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

---

## 🔄 Updates & Maintenance

### Update Application

```bash
# Pull latest changes
git pull

# Rebuild containers
docker-compose up -d --build

# Check logs
docker-compose logs -f
```

### Database Backup

```bash
# Backup database
docker-compose exec backend cp /data/pcds.db /data/backup_$(date +%Y%m%d).db

# Copy to host
docker cp pcds_backend_1:/data/backup_20250205.db ./backups/
```

---

## 📈 Scaling

### Horizontal Scaling

```yaml
services:
  backend:
    deploy:
      replicas: 3
    
  nginx:
    # Load balancer configuration
```

### Database

For production scale (1M+ events):
- Consider PostgreSQL instead of SQLite
- Add Redis for caching
- Implement database sharding

---

## 🚨 Troubleshooting

**Container won't start**:
```bash
docker-compose logs backend
docker-compose logs frontend
```

**Port conflicts**:
```bash
# Change ports in docker-compose.yml
ports:
  - "8001:8000"  # Backend
  - "3001:3000"  # Frontend
```

**Reset everything**:
```bash
docker-compose down -v
docker-compose up -d --build
```

---

## 🎓 College Deployment

**For college servers**:
1. Request Docker installation from IT
2. Clone repo to server
3. Configure environment variables
4. Run `docker-compose up -d`
5. Access via college network

**Recommended specs**:
- 8GB RAM
- 4 CPU cores
- 50GB storage
- Ubuntu 20.04 or Docker-enabled Windows Server

---

**Docker deployment complete!** Your PCDS Enterprise is now containerized and production-ready! 🐳🚀
