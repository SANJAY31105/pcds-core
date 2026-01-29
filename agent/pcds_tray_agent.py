#!/usr/bin/env python3
"""
PCDS Agent - System Tray Application with Active Defense
Runs as a background tray app monitoring network traffic and can kill malicious processes.

Requirements:
    pip install psutil requests pystray pillow
"""

import threading
import socket
import time
import requests
import psutil
import configparser
import os
import sys
import webbrowser
from datetime import datetime
from typing import List, Dict, Set
from PIL import Image, ImageDraw

try:
    import pystray
    from pystray import MenuItem as item
except ImportError:
    print("pystray not found. Install with: pip install pystray pillow")
    sys.exit(1)

# Configuration
PCDS_API_URL = "https://pcds-backend-production.up.railway.app/api/v2/ingest"
PCDS_DASHBOARD_URL = "https://pcdsai.app/dashboard"
SEND_INTERVAL = 10  # seconds
APP_NAME = "PCDS Agent"

# ============= MALWARE SIGNATURE DATABASE =============
# Comprehensive list of known malicious tools, ransomware, and attack tools

# CREDENTIAL THEFT / DUMPERS
CREDENTIAL_TOOLS = {
    'mimikatz', 'lazagne', 'wce', 'gsecdump', 'pwdump', 'fgdump',
    'secretsdump', 'pypykatz', 'nanodump', 'handlekatz', 'kekeo',
    'safetykatz', 'sharpkatz', 'dumppert', 'dumpert', 'lsassy'
}

# REMOTE ACCESS TROJANS (RATs)
RATS = {
    'cobaltstrike', 'beacon', 'meterpreter', 'covenant', 'sliver',
    'bruteratel', 'havoc', 'poshc2', 'empire', 'nishang',
    'quasarrat', 'asyncrat', 'njrat', 'darkcomet', 'remcosrat',
    'orcusrat', 'netwire', 'poisonivy', 'blackshades', 'warzone'
}

# LATERAL MOVEMENT TOOLS
LATERAL_MOVEMENT = {
    'psexec', 'paexec', 'wmiexec', 'smbexec', 'dcomexec', 'atexec',
    'winrm', 'crackmapexec', 'impacket', 'sharphound', 'bloodhound',
    'adexplorer', 'pingcastle', 'rubeus', 'kerbrute', 'spray'
}

# RANSOMWARE FAMILIES
RANSOMWARE = {
    'wannacry', 'petya', 'notpetya', 'ryuk', 'conti', 'lockbit',
    'revil', 'sodinokibi', 'maze', 'egregor', 'darkside', 'blackcat',
    'hive', 'blackbasta', 'royal', 'akira', 'play', 'clop',
    'phobos', 'dharma', 'stop', 'djvu', 'locky', 'cryptolocker'
}

# NETWORK TOOLS (Often Abused)
NETWORK_TOOLS = {
    'nc', 'nc.exe', 'ncat', 'netcat', 'nmap', 'masscan', 'angry',
    'tcpdump', 'wireshark', 'responder', 'inveigh', 'ntlmrelay',
    'mitm6', 'bettercap', 'mitmproxy', 'ettercap'
}

# LOADERS / STAGERS
LOADERS = {
    'donut', 'shellcode', 'msfvenom', 'shellter', 'veil', 'unicorn',
    'powercat', 'invoke-obfuscation', 'amsi', 'sharpshooter'
}

# ROOTKITS / EVASION
EVASION = {
    'gdriverload', 'kdmapper', 'dsefix', 'mimifree', 'processhider',
    'rootkit', 'bootkit', 'turla', 'equation', 'uroburos'
}

# Combined malicious set (for auto-kill when enabled)
MALICIOUS_PROCESSES = (
    CREDENTIAL_TOOLS | RATS | LATERAL_MOVEMENT | 
    RANSOMWARE | NETWORK_TOOLS | LOADERS | EVASION
)

# Suspicious command-line patterns (warn but configurable kill)
SUSPICIOUS_PATTERNS = [
    # Encoded/Obfuscated PowerShell
    'powershell -e', 'powershell -enc', 'powershell -encodedcommand',
    'powershell -w hidden', 'powershell -windowstyle hidden',
    'powershell -nop', 'powershell -noprofile', 'powershell iex',
    'invoke-expression', 'invoke-command', 'invoke-webrequest',
    'downloadstring', 'downloadfile', 'system.net.webclient',
    
    # CMD abuse
    'cmd /c', 'cmd /k', 'cmd.exe /c', 'wmic process',
    'wmic shadowcopy delete', 'vssadmin delete shadows',
    
    # LOLBins (Living Off the Land Binaries)
    'certutil -urlcache', 'certutil -decode', 'certutil -encode',
    'bitsadmin /transfer', 'bitsadmin /create',
    'mshta vbscript', 'mshta javascript',
    'regsvr32 /s /n /u', 'regsvr32 scrobj',
    'rundll32 javascript', 'rundll32 shell32',
    'cscript', 'wscript',
    
    # Disable security
    'set-mppreference', 'disable-windowsoptionalfeature',
    'netsh advfirewall set', 'netsh firewall',
    'sc stop', 'sc delete',
    
    # Persistence
    'schtasks /create', 'at \\\\', 'reg add.*\\run',
    'wmic startup', 'startup folder'
]

# High-confidence malicious (immediate action)
CRITICAL_THREATS = {
    'mimikatz', 'cobaltstrike', 'meterpreter', 'ryuk', 'conti',
    'lockbit', 'wannacry', 'petya', 'revil', 'darkside'
}

class ActiveDefense:
    """Active Defense module - can terminate malicious processes"""
    
    def __init__(self):
        self.enabled = True
        self.killed_processes: Set[str] = set()
        self.blocked_ips: Set[str] = set()
        self.auto_kill = False  # Requires explicit enable
        
    def check_process(self, process_name: str, pid: int) -> Dict:
        """Check if a process is malicious"""
        result = {
            "is_malicious": False,
            "threat_level": "safe",
            "action_taken": None
        }
        
        name_lower = process_name.lower()
        
        # Check against known malicious
        for mal in MALICIOUS_PROCESSES:
            if mal in name_lower:
                result["is_malicious"] = True
                result["threat_level"] = "critical"
                
                if self.auto_kill and self.enabled:
                    killed = self.kill_process(pid, process_name)
                    if killed:
                        result["action_taken"] = "terminated"
                        self.killed_processes.add(f"{process_name}:{pid}")
                break
        
        return result
    
    def kill_process(self, pid: int, reason: str = "") -> bool:
        """Terminate a process by PID"""
        try:
            proc = psutil.Process(pid)
            proc.terminate()
            proc.wait(timeout=5)
            print(f"[ACTIVE DEFENSE] Killed process {pid}: {reason}")
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            return False
    
    def block_ip(self, ip: str) -> bool:
        """Add IP to blocked list (firewall integration placeholder)"""
        self.blocked_ips.add(ip)
        print(f"[ACTIVE DEFENSE] Blocked IP: {ip}")
        return True


class ProcessMonitor:
    """Monitor running processes for suspicious behavior"""
    
    def __init__(self):
        self.process_baseline: Dict[int, Dict] = {}
        self.suspicious_activities: List[Dict] = []
        
    def get_process_info(self) -> List[Dict]:
        """Get detailed process information"""
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cmdline', 'create_time']):
            try:
                info = proc.info
                processes.append({
                    "pid": info['pid'],
                    "name": info['name'],
                    "user": info['username'],
                    "cmdline": ' '.join(info['cmdline'] or []),
                    "created": datetime.fromtimestamp(info['create_time']).isoformat()
                })
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return processes
    
    def detect_suspicious(self, processes: List[Dict]) -> List[Dict]:
        """Detect suspicious process activity"""
        suspicious = []
        
        for proc in processes:
            cmdline = (proc.get('cmdline') or '').lower()
            name = (proc.get('name') or '').lower()
            
            # Check suspicious patterns
            for pattern in SUSPICIOUS_PATTERNS:
                if pattern in cmdline:
                    suspicious.append({
                        "type": "suspicious_command",
                        "process": proc['name'],
                        "pid": proc['pid'],
                        "detail": f"Suspicious pattern: {pattern}",
                        "timestamp": datetime.now().isoformat()
                    })
                    break
            
            # Detect encoded PowerShell
            if 'powershell' in name and ('-e ' in cmdline or '-enc' in cmdline):
                suspicious.append({
                    "type": "encoded_powershell",
                    "process": proc['name'],
                    "pid": proc['pid'],
                    "detail": "Encoded PowerShell detected",
                    "timestamp": datetime.now().isoformat()
                })
        
        return suspicious


class PCDSAgent:
    def __init__(self):
        self.running = False
        self.api_key = None
        self.api_url = PCDS_API_URL
        self.hostname = socket.gethostname()
        self.events_sent = 0
        self.threats_blocked = 0
        self.last_status = "Ready"
        self.monitor_thread = None
        self.icon = None
        
        # Active Defense
        self.active_defense = ActiveDefense()
        self.process_monitor = ProcessMonitor()
        
        # Load config
        self.load_config()
    
    def get_config_path(self):
        """Get config.ini path"""
        if getattr(sys, 'frozen', False):
            return os.path.join(os.path.dirname(sys.executable), 'config.ini')
        else:
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.ini')
    
    def load_config(self):
        """Load configuration from config.ini"""
        config = configparser.ConfigParser()
        config_path = self.get_config_path()
        
        if os.path.exists(config_path):
            config.read(config_path)
            if 'PCDS' in config:
                self.api_key = config['PCDS'].get('api_key')
                self.api_url = config['PCDS'].get('url', PCDS_API_URL)
                # Load active defense setting
                self.active_defense.auto_kill = config['PCDS'].getboolean('auto_kill', False)
    
    def save_config(self, api_key: str):
        """Save API key to config"""
        config = configparser.ConfigParser()
        config['PCDS'] = {
            'api_key': api_key,
            'url': self.api_url,
            'auto_kill': str(self.active_defense.auto_kill)
        }
        with open(self.get_config_path(), 'w') as f:
            config.write(f)
        self.api_key = api_key
    
    def create_icon_image(self, color='green'):
        """Create a simple icon for the tray"""
        size = 64
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        colors = {
            'green': (34, 197, 94),    # Protected
            'yellow': (234, 179, 8),   # Warning
            'red': (239, 68, 68),      # Threat detected
            'gray': (100, 100, 100)    # Inactive
        }
        fill_color = colors.get(color, colors['gray'])
        
        draw.ellipse([4, 4, size-4, size-4], fill=fill_color)
        draw.text((size//4, size//4), "P", fill="white")
        
        return image
    
    def get_network_connections(self) -> List[Dict]:
        """Get current network connections with threat detection"""
        connections = []
        
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status in ['ESTABLISHED', 'LISTEN']:
                    process_name = "unknown"
                    is_malicious = False
                    action_taken = None
                    
                    if conn.pid:
                        try:
                            process = psutil.Process(conn.pid)
                            process_name = process.name()
                            
                            # Active Defense check
                            defense_result = self.active_defense.check_process(process_name, conn.pid)
                            is_malicious = defense_result["is_malicious"]
                            action_taken = defense_result["action_taken"]
                            
                            if is_malicious:
                                self.threats_blocked += 1
                                if self.icon:
                                    self.icon.icon = self.create_icon_image('red')
                                    
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
                    event = {
                        "source_ip": conn.laddr.ip if conn.laddr else "0.0.0.0",
                        "dest_ip": conn.raddr.ip if conn.raddr else None,
                        "dest_port": conn.raddr.port if conn.raddr else None,
                        "protocol": "TCP" if conn.type == socket.SOCK_STREAM else "UDP",
                        "process_name": process_name,
                        "status": conn.status,
                        "is_threat": is_malicious,
                        "action_taken": action_taken,
                        "timestamp": datetime.now().isoformat()
                    }
                    connections.append(event)
        except psutil.AccessDenied:
            pass
        
        return connections
    
    def send_to_pcds(self, events: List[Dict]) -> bool:
        """Send events to PCDS backend"""
        if not events or not self.api_key:
            return False
        
        # Include process monitoring data
        process_alerts = self.process_monitor.detect_suspicious(
            self.process_monitor.get_process_info()[:50]  # Top 50 processes
        )
        
        payload = {
            "api_key": self.api_key,
            "hostname": self.hostname,
            "events": events,
            "process_alerts": process_alerts,  # NEW: process monitoring
            "threats_blocked": self.threats_blocked
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.events_sent += data.get('events_received', 0)
                status = f"Protected - {self.events_sent} events"
                if self.threats_blocked > 0:
                    status += f", {self.threats_blocked} threats blocked"
                self.last_status = status
                return True
            elif response.status_code == 401:
                self.last_status = "Invalid API Key"
                return False
            else:
                self.last_status = f"Error: {response.status_code}"
                return False
        except requests.exceptions.RequestException:
            self.last_status = "Connection Error"
            return False
    
    def monitor_loop(self):
        """Main monitoring loop with process monitoring"""
        while self.running:
            if self.api_key:
                events = self.get_network_connections()
                if events:
                    self.send_to_pcds(events)
                    
                    # Update icon color based on status
                    if self.threats_blocked > 0:
                        self.icon.icon = self.create_icon_image('yellow')
                    else:
                        self.icon.icon = self.create_icon_image('green')
            
            time.sleep(SEND_INTERVAL)
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.last_status = "Monitoring started"
            if self.icon:
                self.icon.icon = self.create_icon_image('green')
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        self.running = False
        self.last_status = "Monitoring stopped"
        if self.icon:
            self.icon.icon = self.create_icon_image('gray')
    
    def toggle_auto_kill(self):
        """Toggle auto-kill feature"""
        self.active_defense.auto_kill = not self.active_defense.auto_kill
        status = "enabled" if self.active_defense.auto_kill else "disabled"
        print(f"[ACTIVE DEFENSE] Auto-kill {status}")
    
    def open_dashboard(self):
        """Open PCDS dashboard in browser"""
        webbrowser.open(PCDS_DASHBOARD_URL)
    
    def quit_app(self, icon):
        """Quit the application"""
        self.stop_monitoring()
        icon.stop()
    
    def create_menu(self):
        """Create the tray menu with Active Defense options"""
        return pystray.Menu(
            item(lambda text: f"PCDS Agent - {self.hostname}", lambda: None, enabled=False),
            item(lambda text: f"Status: {self.last_status}", lambda: None, enabled=False),
            item(lambda text: f"Threats Blocked: {self.threats_blocked}", lambda: None, enabled=False),
            pystray.Menu.SEPARATOR,
            item("Open Dashboard", lambda: self.open_dashboard()),
            item("Start Monitoring", lambda: self.start_monitoring()),
            item("Stop Monitoring", lambda: self.stop_monitoring()),
            pystray.Menu.SEPARATOR,
            item(
                lambda text: f"Active Defense: {'ON' if self.active_defense.auto_kill else 'OFF'}",
                lambda: self.toggle_auto_kill()
            ),
            pystray.Menu.SEPARATOR,
            item("Quit", self.quit_app)
        )
    
    def run(self):
        """Run the tray application"""
        if not self.api_key:
            import tkinter as tk
            from tkinter import simpledialog, messagebox
            
            root = tk.Tk()
            root.withdraw()
            
            api_key = simpledialog.askstring(
                "PCDS Setup",
                "Enter your API Key:\n\n(Get it from pcdsai.app/settings/api-keys)",
                parent=root
            )
            
            if api_key:
                self.save_config(api_key)
            else:
                messagebox.showwarning("PCDS Agent", "No API key provided. Agent will not send data.")
            
            root.destroy()
        
        self.icon = pystray.Icon(
            APP_NAME,
            self.create_icon_image('gray'),
            APP_NAME,
            self.create_menu()
        )
        
        if self.api_key:
            self.start_monitoring()
        
        self.icon.run()


def add_to_startup():
    """Add agent to Windows startup"""
    import winreg
    
    if getattr(sys, 'frozen', False):
        exe_path = sys.executable
    else:
        exe_path = f'pythonw.exe "{os.path.abspath(__file__)}"'
    
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, "PCDS Agent", 0, winreg.REG_SZ, exe_path)
        winreg.CloseKey(key)
        return True
    except Exception as e:
        print(f"Failed to add to startup: {e}")
        return False


def main():
    add_to_startup()
    agent = PCDSAgent()
    agent.run()


if __name__ == "__main__":
    main()
