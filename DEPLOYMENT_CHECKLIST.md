# PCDS Enterprise - Step-by-Step Deployment Guide
## From Approval to Fully Operational System

**Estimated Total Time:** 3-4 weeks  
**Your Role:** Project Lead & System Administrator

---

## 📅 WEEK 1: PREPARATION & ACCESS

### Day 1: Post-Approval Setup

**☑️ Step 1.1: Get Written Approval**
```
□ Request formal approval email from IT Director/CISO
□ Get copy of approval for your records
□ Confirm deployment timeline agreed upon
□ Get emergency contact information
```

**☑️ Step 1.2: Schedule Kickoff Meeting**
```
□ Schedule meeting with:
  - IT Network Administrator
  - Server Administrator  
  - Security Team Lead
□ Agenda:
  - Technical requirements review
  - Server access discussion
  - Network SPAN port setup
  - Timeline confirmation
```

**☑️ Step 1.3: Create Project Documentation**
```
□ Create shared folder/drive for project
□ Upload all documentation:
  - PCDS_Deployment_Proposal.pdf
  - Technical Architecture Diagram
  - Installation Guide
  - Training Materials
□ Share access with IT team
```

---

### Day 2-3: Server Provisioning

**☑️ Step 2.1: Request Server Access**

**Email Template to Server Admin:**
```
Subject: Server Access Request for PCDS Deployment

Hi [Server Admin Name],

Following approval for PCDS Enterprise deployment, I need access to 
the designated server for installation.

Required Access:
- Server: [Server name/IP from IT]
- OS: Ubuntu 22.04 LTS (preferred) or RHEL/CentOS
- Access Type: SSH with sudo privileges
- Username: [Your username request]

Server Specifications Needed:
- CPU: 8 cores minimum
- RAM: 16GB minimum (32GB recommended)
- Storage: 500GB SSD
- Network: 2 NICs (1 for management, 1 for SPAN port monitoring)

Please provide:
- SSH access credentials
- Server IP address
- Firewall rules needed

Timeline: Need access by [Date] to meet deployment schedule.

Thanks,
[Your Name]
```

**☑️ Step 2.2: Verify Server Specs**

Once you get access, SSH to server and check:

```bash
# SSH to server
ssh your_username@server_ip

# Check CPU
lscpu | grep "^CPU(s)"
# Should show: CPU(s): 8 or more

# Check RAM
free -h
# Should show: 16GB or more

# Check Disk
df -h
# Should show: 500GB+ available

# Check Network Interfaces
ip addr show
# Should show: 2+ network interfaces

# Check OS version
lsb_release -a
# Should show: Ubuntu 22.04 or RHEL 8+

# Document everything
echo "Server validation complete" > ~/pcds-server-check.txt
date >> ~/pcds-server-check.txt
lscpu >> ~/pcds-server-check.txt
free -h >> ~/pcds-server-check.txt
df -h >> ~/pcds-server-check.txt
```

---

### Day 4-5: Network Configuration

**☑️ Step 3.1: Request SPAN Port Configuration**

**Email to Network Administrator:**
```
Subject: SPAN Port Configuration Request for PCDS

Hi [Network Admin Name],

For PCDS network monitoring, I need a SPAN/mirror port configured 
to copy all network traffic to our monitoring server.

Configuration Needed:

Source: Core switch ports (all campus traffic)
Destination: Port connected to PCDS server NIC #2
Type: SPAN session (read-only monitor)

Server Details:
- Server IP: [From Step 2]
- Monitoring NIC: eth1 (or interface name)
- MAC Address: [Get from: ip link show eth1]

This is passive monitoring only - no traffic modification.

Can we schedule this for [Preferred Date/Time]?

Thanks,
[Your Name]
```

**☑️ Step 3.2: Get Firewall Rules Configured**

**Ports to Open:**
```
On Server Firewall:

Inbound:
- TCP 22 (SSH) - From IT admin IPs only
- TCP 8000 (Backend API) - From IT security team subnet only
- TCP 3000 (Frontend) - From IT security team subnet only

Outbound:
- TCP 443 (HTTPS) - For updates
- TCP 80 (HTTP) - For package downloads
- DNS (UDP 53) - For name resolution

Internal (localhost only):
- TCP 5432 (PostgreSQL)
- TCP 6379 (Redis)
```

**Request from Network Admin:**
```
Please configure these firewall rules for PCDS server [IP]:
[Paste rules above]
```

---

## 📅 WEEK 2: INSTALLATION

### Day 6: Prepare Server Environment

**☑️ Step 4.1: Update System**

```bash
# SSH to server
ssh your_username@server_ip

# Update package lists
sudo apt update

# Upgrade system (may require reboot)
sudo apt upgrade -y

# Reboot if kernel updated
sudo reboot

# Reconnect after reboot
ssh your_username@server_ip
```

**☑️ Step 4.2: Install Prerequisites**

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group
sudo usermod -aG docker $USER

# Logout and login again for group to take effect
exit
ssh your_username@server_ip

# Verify Docker installed
docker --version
# Should show: Docker version 24.x.x or higher

# Install Docker Compose
sudo apt install docker-compose -y

# Verify
docker-compose --version
# Should show: docker-compose version 1.29.x or higher

# Install Git
sudo apt install git -y

# Install Python (if not already installed)
sudo apt install python3 python3-pip -y

# Verify
python3 --version
pip3 --version
```

**☑️ Step 4.3: Create Directory Structure**

```bash
# Create main directory
sudo mkdir -p /opt/pcds-enterprise
sudo chown $USER:$USER /opt/pcds-enterprise
cd /opt/pcds-enterprise

# Create subdirectories
mkdir -p logs backups config data

# Set permissions
chmod 750 /opt/pcds-enterprise
```

---

### Day 7: Deploy PCDS Code

**☑️ Step 5.1: Upload PCDS Code**

**Option A: From Your Laptop (If GitHub)**
```bash
# On server
cd /opt/pcds-enterprise
git clone https://github.com/YOUR_USERNAME/pcds-core.git .

# Verify files
ls -la
# Should see: backend/ frontend/ docker-compose.yml etc.
```

**Option B: Direct Upload (If no GitHub)**
```powershell
# On your laptop
cd C:\Users\sanja\OneDrive\Desktop\pcds-core

# Use SCP to upload
scp -r * your_username@server_ip:/opt/pcds-enterprise/

# Or use WinSCP GUI tool
```

**☑️ Step 5.2: Configure Environment**

```bash
# On server
cd /opt/pcds-enterprise

# Copy environment template
cp .env.example .env

# Edit environment file
nano .env
```

**Edit .env file with production values:**
```bash
# Database Configuration
DATABASE_URL=postgresql://pcds_admin:CHANGE_THIS_SECURE_PASSWORD@postgres:5432/pcds_production
POSTGRES_PASSWORD=CHANGE_THIS_SECURE_PASSWORD

# Security
SECRET_KEY=GENERATE_RANDOM_64_CHAR_STRING_HERE
JWT_SECRET_KEY=GENERATE_ANOTHER_RANDOM_STRING

# Environment
ENVIRONMENT=production
DEBUG=false

# Network
CORS_ORIGINS=https://security.yourcollege.edu,http://collegeserver.local:3000

# Redis
REDIS_URL=redis://redis:6379

# Email (for alerts)
SMTP_SERVER=smtp.yourcollege.edu
SMTP_PORT=587
SMTP_USER=pcds@yourcollege.edu
SMTP_PASSWORD=EMAIL_PASSWORD_HERE
ALERT_EMAIL=security-team@yourcollege.edu
```

**Generate secure passwords:**
```bash
# Generate SECRET_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(64))"

# Generate JWT_SECRET
python3 -c "import secrets; print(secrets.token_urlsafe(64))"

# Generate DB password
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Save the file:**
```
Ctrl+O (save)
Enter (confirm)
Ctrl+X (exit)
```

---

### Day 8-9: Start Services

**☑️ Step 6.1: Start Docker Containers**

```bash
# Make sure you're in the right directory
cd /opt/pcds-enterprise

# Pull required Docker images
docker-compose pull

# Start all services
docker-compose up -d

# This will start:
# - PostgreSQL database
# - Redis cache
# - PCDS backend API
# - PCDS frontend

# Check status
docker-compose ps
```

**Expected Output:**
```
NAME                  STATUS              PORTS
pcds-postgres         Up 30 seconds      0.0.0.0:5432->5432/tcp
pcds-redis            Up 30 seconds      0.0.0.0:6379->6379/tcp
pcds-backend          Up 30 seconds      0.0.0.0:8000->8000/tcp
pcds-frontend         Up 30 seconds      0.0.0.0:3000->3000/tcp
```

**☑️ Step 6.2: Initialize Database**

```bash
# Run database schema creation
docker exec -it pcds-backend python init_database.py

# Create admin user
docker exec -it pcds-backend python create_admin_user.py

# When prompted, enter:
# Email: admin@yourcollege.edu
# Password: [Choose strong password]
# Name: IT Security Admin
```

**☑️ Step 6.3: Verify Services**

```bash
# Check backend health
curl http://localhost:8000/health
# Should return: {"status": "healthy"}

# Check frontend
curl http://localhost:3000
# Should return HTML page

# Check database
docker exec -it pcds-postgres psql -U pcds_admin -d pcds_production -c "SELECT COUNT(*) FROM users;"
# Should show: 1 (the admin user)

# Check logs
docker-compose logs backend | tail -50
docker-compose logs frontend | tail -50
```

---

### Day 10: Network Monitoring Setup

**☑️ Step 7.1: Verify SPAN Port Active**

```bash
# Check if monitoring interface is receiving traffic
sudo tcpdump -i eth1 -c 10
# Should show network packets (if SPAN port is configured)

# If no packets, contact network admin - SPAN port not configured yet
```

**☑️ Step 7.2: Install Network Monitor**

```bash
# Install packet capture tools
sudo apt install tcpdump python3-pyshark libpcap-dev -y

# Create network monitor script
nano /opt/pcds-enterprise/network_monitor.py
```

**Paste this script:**
```python
#!/usr/bin/env python3
"""
PCDS Network Monitor - Captures college network traffic
"""

import pyshark
import requests
import json
from datetime import datetime

PCDS_API = "http://localhost:8000/api/v2/detections"
MONITORING_INTERFACE = "eth1"  # Change to your SPAN port interface

def monitor_network():
    print(f"Starting network monitor on {MONITORING_INTERFACE}...")
    
    capture = pyshark.LiveCapture(
        interface=MONITORING_INTERFACE,
        bpf_filter='ip'  # Only capture IP traffic
    )
    
    for packet in capture.sniff_continuously():
        try:
            if hasattr(packet, 'ip'):
                analyze_packet(packet)
        except Exception as e:
            continue

def analyze_packet(packet):
    """Analyze packet for threats"""
    # Extract basic info
    src_ip = packet.ip.src
    dst_ip = packet.ip.dst
    
    # Detect suspicious patterns
    suspicious = False
    detection_type = "normal_traffic"
    
    # Check for suspicious ports
    if hasattr(packet, 'tcp'):
        dst_port = int(packet.tcp.dstport)
        if dst_port in [4444, 5555, 6666]:  # C2 ports
            suspicious = True
            detection_type = "c2_communication"
    
    # Send to PCDS if suspicious
    if suspicious:
        send_detection(src_ip, dst_ip, detection_type)

def send_detection(src_ip, dst_ip, det_type):
    """Send detection to PCDS"""
    try:
        data = {
            'detection_type': det_type,
            'severity': 'high',
            'source_ip': src_ip,
            'destination_ip': dst_ip,
            'detected_at': datetime.utcnow().isoformat()
        }
        requests.post(PCDS_API, json=data, timeout=1)
        print(f"Alert: {src_ip} -> {dst_ip} ({det_type})")
    except:
        pass

if __name__ == "__main__":
    monitor_network()
```

**Make it executable:**
```bash
chmod +x /opt/pcds-enterprise/network_monitor.py
```

**☑️ Step 7.3: Create System Service**

```bash
# Create systemd service file
sudo nano /etc/systemd/system/pcds-monitor.service
```

**Paste this:**
```ini
[Unit]
Description=PCDS Network Monitor
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=root
WorkingDirectory=/opt/pcds-enterprise
ExecStart=/usr/bin/python3 /opt/pcds-enterprise/network_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable and start:**
```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable pcds-monitor

# Start service
sudo systemctl start pcds-monitor

# Check status
sudo systemctl status pcds-monitor
# Should show: active (running)

# View logs
sudo journalctl -u pcds-monitor -f
# Should show network traffic being monitored
```

---

## 📅 WEEK 3: TESTING & VALIDATION

### Day 11-12: System Testing

**☑️ Step 8.1: Access Dashboard**

```bash
# Get server IP
hostname -I

# From your laptop, access:
http://SERVER_IP:3000

# Login with:
# Email: admin@yourcollege.edu
# Password: [Password you set]
```

**☑️ Step 8.2: Verify Detection Flow**

```bash
# On server, check if detections are being created
docker exec -it pcds-postgres psql -U pcds_admin -d pcds_production -c "SELECT COUNT(*) FROM detections;"

# Should show increasing count

# Check recent detections
docker exec -it pcds-postgres psql -U pcds_admin -d pcds_production -c "SELECT id, detection_type, severity, detected_at FROM detections ORDER BY detected_at DESC LIMIT 10;"
```

**☑️ Step 8.3: Performance Testing**

```bash
# Check CPU usage
top

# Check memory
free -h

# Check disk space
df -h

# Check network traffic
sudo iftop -i eth1

# All should be within normal limits
```

---

### Day 13-14: IT Team Training

**☑️ Step 9.1: Schedule Training Session**

```
Invite: IT Security Team, Network Admins
Duration: 2 hours
Location: Conference room with projector
Access: PCDS dashboard (bring laptop)
```

**Training Agenda:**

**Hour 1: Platform Overview (60 min)**
```
1. Login & Dashboard (10 min)
   - How to access
   - Overview metrics
   - System health

2. Detections Page (15 min)
   - Viewing alerts
   - Filtering by severity
   - Understanding detection types

3. MITRE Mapping (10 min)
   - What is MITRE ATT&CK
   - How detections are classified
   - Tactic/technique breakdown

4. Entity Tracking (10 min)
   - What are entities
   - Risk scoring
   - Historical activity

5. Investigations (15 min)
   - Creating investigations
   - Adding evidence
   - Resolution workflow
```

**Hour 2: Operations (60 min)**
```
6. Alert Response (20 min)
   - How to investigate alerts
   - False positive handling
   - Escalation procedures

7. Reporting (15 min)
   - Generating reports
   - Metrics to track
   - Monthly summaries

8. System Maintenance (15 min)
   - Checking system health
   - Log review
   - Backup procedures

9. Q&A (10 min)
```

**☑️ Step 9.2: Create Training Materials**

```bash
# Create training folder
mkdir /opt/pcds-enterprise/training

# Create quick reference guide
nano /opt/pcds-enterprise/training/Quick_Reference.md
```

**Quick Reference Content:**
```markdown
# PCDS Quick Reference Guide

## Daily Tasks
1. Check dashboard for critical alerts
2. Review new detections
3. Investigate high-priority incidents

## Weekly Tasks
1. Generate weekly summary report
2. Review system performance
3. Update threat signatures

## Monthly Tasks
1. Executive security report
2. System backup verification
3. Performance optimization

## Emergency Contacts
- System Admin: [Your contact]
- Faculty Advisor: [If applicable]
- IT Director: [Director contact]

## Common Tasks

### Investigate an Alert
1. Click detection in list
2. Review MITRE context
3. Check source/destination IPs
4. Create investigation if needed

### Generate Report
1. Go to Reports page
2. Select date range
3. Click "Generate"
4. Download PDF

### Check System Health
1. Dashboard -> System Status
2. All indicators should be green
3. If red, contact admin
```

---

### Day 15: Go-Live Preparation

**☑️ Step 10.1: Final Pre-Launch Checklist**

```
□ All services running stable for 72+ hours
□ Network monitor capturing traffic
□ Detections being created correctly
□ Dashboard accessible to IT team
□ IT team trained
□ Documentation complete
□ Backup system configured
□ Emergency contacts established
□ Go-live date confirmed
```

**☑️ Step 10.2: Configure Backups**

```bash
# Create backup script
nano /opt/pcds-enterprise/backup.sh
```

```bash
#!/bin/bash
# PCDS Database Backup Script

BACKUP_DIR="/opt/pcds-enterprise/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
docker exec pcds-postgres pg_dump -U pcds_admin pcds_production > \
  $BACKUP_DIR/pcds_backup_$DATE.sql

# Remove backups older than 30 days
find $BACKUP_DIR -name "pcds_backup_*.sql" -mtime +30 -delete

echo "Backup completed: pcds_backup_$DATE.sql"
```

**Make executable and schedule:**
```bash
chmod +x /opt/pcds-enterprise/backup.sh

# Add to crontab (daily at 2 AM)
crontab -e
```

Add line:
```
0 2 * * * /opt/pcds-enterprise/backup.sh >> /opt/pcds-enterprise/logs/backup.log 2>&1
```

---

## 📅 WEEK 4: GO-LIVE & OPERATIONS

### Day 16: Official Go-Live

**☑️ Step 11.1: Announce Go-Live**

**Email to IT Team:**
```
Subject: PCDS Enterprise Now Live - Network Monitoring Active

Team,

PCDS Enterprise is now officially live and monitoring our campus network 24/7.

Access: http://[SERVER_IP]:3000
Login: Use credentials provided

Key Features:
- Real-time threat detection
- MITRE ATT&CK framework integration
- Automated alert generation
- Investigation workflow

Please check the dashboard daily and escalate critical alerts immediately.

Training materials: \\server\pcds\training\

Support: [Your contact info]

Thanks,
[Your Name]
```

**☑️ Step 11.2: Start Monitoring Log**

Create monitoring journal:
```bash
nano /opt/pcds-enterprise/logs/operations_log.txt
```

```
PCDS OPERATIONS LOG
Started: [Date]

Day 1:
- System went live at [time]
- [X] detections generated
- [Y] reviewed by security team
- Status: Normal operations

[Update daily]
```

---

### Day 17-21: First Week Operations

**Daily Checklist:**

**Morning (9 AM):**
```
□ Check system status (all green?)
□ Review overnight detections
□ Check critical alerts (if any)
□ Verify network monitor running
□ Check disk space
```

**Evening (5 PM):**
```
□ Daily detection summary
□ Update operations log
□ Backup verification
□ Plan for next day
```

**☑️ Step 12.1: Generate First Weekly Report**

```bash
# After 7 days, generate report
# Login to dashboard
# Go to Reports
# Select "Weekly Summary"
# Download PDF
```

**Share with stakeholders:**
```
To: IT Director, CISO, Faculty Advisor
Subject: PCDS Enterprise - Week 1 Report

Attached is the first weekly security report from PCDS Enterprise.

Highlights:
- Network traffic monitored: [X] TB
- Detections generated: [Y]
- Critical threats: [Z]
- System uptime: 99.9%

All detections were investigated and no false positives reported.

Next steps:
- Continue monitoring
- Monthly executive report (due [date])
- Quarterly system review

Full report attached.

[Your Name]
```

---

### Day 22-30: Ongoing Operations

**☑️ Step 13.1: Monthly Tasks**

**Week 4:**
```
□ Generate monthly report
□ Review system performance metrics
□ Check for software updates
□ Backup verification
□ Schedule review meeting with IT Director
```

**☑️ Step 13.2: Monthly Review Meeting**

**Meeting Agenda:**
```
1. System Status (5 min)
   - Uptime: X%
   - Detections: Y
   - Alerts: Z

2. Threat Summary (10 min)
   - Top threats detected
   - Trends observed
   - Incidents prevented

3. System Performance (5 min)
   - Resource usage
   - Detection accuracy
   - Response times

4. Recommendations (5 min)
   - Security improvements
   - System optimizations
   - Future enhancements

5. Q&A (5 min)
```

---

## 🎯 SUCCESS CRITERIA

**After 30 Days, You Should Have:**

✅ **System Running:**
- 99%+ uptime
- 24/7 monitoring
- Automated detections

✅ **Value Demonstrated:**
- Threats detected
- Network visibility
- Security insights

✅ **Team Adoption:**
- IT team using dashboard daily
- Investigations created
- Reports generated

✅ **Documentation:**
- Operations log maintained
- Monthly reports delivered
- Training materials created

---

## 🚨 TROUBLESHOOTING

**Common Issues & Fixes:**

**Issue: Docker containers won't start**
```bash
# Check logs
docker-compose logs

# Restart services
docker-compose down
docker-compose up -d
```

**Issue: No detections appearing**
```bash
# Check network monitor
sudo systemctl status pcds-monitor

# Restart monitor
sudo systemctl restart pcds-monitor

# Check SPAN port
sudo tcpdump -i eth1 -c 10
```

**Issue: Dashboard not accessible**
```bash
# Check frontend service
docker logs pcds-frontend

# Restart frontend
docker restart pcds-frontend
```

**Issue: High resource usage**
```bash
# Check resource usage
docker stats

# Adjust resource limits in docker-compose.yml
```

---

## 📞 SUPPORT ESCALATION

**Level 1: Self-Help**
- Check documentation
- Review logs
- Restart services

**Level 2: Your Support**
- Email: [Your email]
- Phone: [Your phone]
- Response: < 4 hours

**Level 3: Faculty/Expert**
- Escalate complex issues
- Performance optimization
- Feature requests

---

## ✅ DEPLOYMENT COMPLETE!

**Congratulations! Your PCDS Enterprise is now protecting your college network!**

**What You've Achieved:**
- ✅ Enterprise-grade security monitoring
- ✅ Real-time threat detection
- ✅ 24/7 network visibility
- ✅ Professional deployment
- ✅ First enterprise client!

**This is now your portfolio piece and potential business!** 🎓🚀

---

*Keep this guide for reference throughout the deployment!*
