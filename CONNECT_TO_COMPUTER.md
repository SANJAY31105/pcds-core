# PCDS Enterprise - Connect to Your Computer/Network
## How to Monitor Real Traffic & Systems

Currently, PCDS is using **simulated data** (generated for testing). To monitor your **actual computer/network**, you have several options:

---

## 🖥️ Option 1: Monitor Your Windows Computer (Easiest)

### What You'll Monitor:
- Process execution
- Network connections
- File access
- Registry changes
- User logins

### Steps:

**1. Install Sysmon (Microsoft System Monitor)**
```powershell
# Download Sysmon
# Visit: https://learn.microsoft.com/en-us/sysinternals/downloads/sysmon

# Install with recommended config
sysmon64.exe -accepteula -i sysmonconfig.xml
```

**2. Configure Log Collection**

Create `backend/integrations/sysmon_collector.py`:
```python
import win32evtlog
import time
from datetime import datetime

def collect_sysmon_events():
    """Collect Windows Sysmon events"""
    server = 'localhost'
    logtype = 'Microsoft-Windows-Sysmon/Operational'
    hand = win32evtlog.OpenEventLog(server, logtype)
    
    flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
    
    events = win32evtlog.ReadEventLog(hand, flags, 0)
    
    for event in events:
        # Parse event and send to detection engine
        event_data = {
            'timestamp': event.TimeGenerated,
            'event_id': event.EventID,
            'source': 'sysmon',
            'data': event.StringInserts
        }
        # Send to detection engine
        analyze_event(event_data)
```

**3. Start Monitoring**
```bash
cd backend
python start_monitoring.py
```

---

## 🌐 Option 2: Monitor Your Network Traffic

### What You'll Monitor:
- All network connections
- DNS queries
- HTTP/HTTPS traffic (with SSL inspection)
- Suspicious connections

### Steps:

**1. Install Wireshark/tcpdump**
```powershell
# Install Wireshark (includes command-line tools)
winget install WiresharkFoundation.Wireshark
```

**2. Capture Network Traffic**

Create `backend/integrations/network_capture.py`:
```python
import pyshark

def monitor_network():
    """Capture and analyze network traffic"""
    capture = pyshark.LiveCapture(interface='WiFi')  # or 'Ethernet'
    
    for packet in capture.sniff_continuously():
        if 'IP' in packet:
            analyze_packet({
                'src_ip': packet.ip.src,
                'dst_ip': packet.ip.dst,
                'protocol': packet.transport_layer,
                'timestamp': packet.sniff_time
            })
```

**3. Run Network Monitor**
```bash
python network_capture.py
```

---

## 📁 Option 3: Monitor File System & Registry

### What You'll Monitor:
- File creations/deletions
- Registry modifications
- Suspicious file operations

Create `backend/integrations/fs_monitor.py`:
```python
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ThreatFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        # Check for suspicious file creations
        if event.src_path.endswith('.exe'):
            analyze_file_creation(event.src_path)

observer = Observer()
observer.schedule(ThreatFileHandler(), path='C:\\', recursive=True)
observer.start()
```

---

## 🔄 Option 4: Import Existing Security Logs

### From Windows Event Viewer:
```python
# Export Windows Security logs
wevtutil epl Security c:\logs\security.evtx

# Import to PCDS
python import_logs.py --source windows --file c:\logs\security.evtx
```

### From Firewall:
```python
# Import firewall logs
python import_logs.py --source firewall --file c:\logs\firewall.log
```

---

## ⚡ Quick Start: Monitor This Computer Now

**Simplest option - Monitor network connections:**

1. **Install pyshark**
```bash
pip install pyshark
```

2. **Create and run this script** (`monitor_now.py`):
```python
import subprocess
import json
import requests
from datetime import datetime

def get_active_connections():
    """Get active network connections"""
    output = subprocess.check_output('netstat -ano', shell=True).decode()
    
    for line in output.split('\n'):
        if 'ESTABLISHED' in line:
            parts = line.split()
            if len(parts) >= 5:
                local = parts[1]
                remote = parts[2]
                
                # Send to PCDS API
                send_to_pcds({
                    'type': 'network_connection',
                    'local_addr': local,
                    'remote_addr': remote,
                    'timestamp': datetime.utcnow().isoformat()
                })

def send_to_pcds(event):
    """Send event to PCDS detection engine"""
    try:
        requests.post('http://localhost:8000/api/v2/events/ingest', json=event)
        print(f"✅ Sent: {event['type']}")
    except:
        print(f"❌ Failed to send event")

# Monitor every 10 seconds
import time
while True:
    get_active_connections()
    time.sleep(10)
```

3. **Run it:**
```bash
python monitor_now.py
```

---

## 🎯 Recommended Approach

For **testing on your computer right now**:

1. **Monitor Network Connections** (Option 4 script above)
   - No installation needed
   - See real connections immediately
   - PCDS will analyze for suspicious IPs, beaconing, etc.

2. **Optional: Install Sysmon**
   - More comprehensive monitoring
   - See process execution, file changes
   - Industry-standard tool

3. **Check PCDS Dashboard**
   - Go to http://localhost:3000
   - Watch detections appear in real-time
   - See your computer's activity analyzed

---

## ⚠️ Important Notes

**Performance:**
- Network monitoring can be resource-intensive
- Start with limited scope (just your network connections)
- Scale up gradually

**Privacy:**
- PCDS will see all monitored activity
- Data stays local (nothing sent to cloud)
- You control what to monitor

**Testing:**
- Run some "suspicious" activities to test detection:
  - Download a file from internet
  - Connect to different IPs
  - Run PowerShell commands
- PCDS should flag anomalies

---

## 🚀 Want Me To Set This Up?

I can create a ready-to-run monitoring script that:
1. Captures your network connections
2. Monitors running processes
3. Sends data to PCDS detection engine
4. Shows results in dashboard in real-time

Should I create this for you?

---

## Alternative: Test with Simulated "Attack"

If you just want to see PCDS in action, I can create a script that simulates attack traffic on your system:

```python
# simulate_attack.py
# Simulates credential dumping, lateral movement, etc.
# PCDS will detect and alert on these
```

Which would you prefer?
1. 📊 **Monitor real activity** (see your actual network/system)
2. 🎯 **Simulate attacks** (test detection without real threats)
3. ⏸️ **Keep using test data** (current setup)
