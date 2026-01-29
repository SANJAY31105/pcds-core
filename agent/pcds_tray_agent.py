#!/usr/bin/env python3
"""
PCDS Agent - System Tray Application
Runs as a background tray app monitoring network traffic.

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
from typing import List, Dict
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

class PCDSAgent:
    def __init__(self):
        self.running = False
        self.api_key = None
        self.api_url = PCDS_API_URL
        self.hostname = socket.gethostname()
        self.events_sent = 0
        self.last_status = "Ready"
        self.monitor_thread = None
        self.icon = None
        
        # Load config
        self.load_config()
    
    def get_config_path(self):
        """Get config.ini path"""
        if getattr(sys, 'frozen', False):
            # Running as compiled exe
            return os.path.join(os.path.dirname(sys.executable), 'config.ini')
        else:
            # Running as script
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
    
    def save_config(self, api_key: str):
        """Save API key to config"""
        config = configparser.ConfigParser()
        config['PCDS'] = {
            'api_key': api_key,
            'url': self.api_url
        }
        with open(self.get_config_path(), 'w') as f:
            config.write(f)
        self.api_key = api_key
    
    def create_icon_image(self, color='green'):
        """Create a simple icon for the tray"""
        size = 64
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        
        # Shield shape
        if color == 'green':
            fill_color = (34, 197, 94)  # Green - protected
        elif color == 'yellow':
            fill_color = (234, 179, 8)  # Yellow - warning
        elif color == 'red':
            fill_color = (239, 68, 68)  # Red - error
        else:
            fill_color = (100, 100, 100)  # Gray - inactive
        
        # Draw a shield
        draw.ellipse([4, 4, size-4, size-4], fill=fill_color)
        draw.text((size//4, size//4), "P", fill="white")
        
        return image
    
    def get_network_connections(self) -> List[Dict]:
        """Get current network connections"""
        connections = []
        
        try:
            for conn in psutil.net_connections(kind='inet'):
                if conn.status in ['ESTABLISHED', 'LISTEN']:
                    process_name = "unknown"
                    if conn.pid:
                        try:
                            process = psutil.Process(conn.pid)
                            process_name = process.name()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            pass
                    
                    event = {
                        "source_ip": conn.laddr.ip if conn.laddr else "0.0.0.0",
                        "dest_ip": conn.raddr.ip if conn.raddr else None,
                        "dest_port": conn.raddr.port if conn.raddr else None,
                        "protocol": "TCP" if conn.type == socket.SOCK_STREAM else "UDP",
                        "process_name": process_name,
                        "status": conn.status,
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
        
        payload = {
            "api_key": self.api_key,
            "hostname": self.hostname,
            "events": events
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=10)
            if response.status_code == 200:
                data = response.json()
                self.events_sent += data.get('events_received', 0)
                self.last_status = f"Protected - {self.events_sent} events sent"
                return True
            elif response.status_code == 401:
                self.last_status = "Invalid API Key"
                return False
            else:
                self.last_status = f"Error: {response.status_code}"
                return False
        except requests.exceptions.RequestException as e:
            self.last_status = "Connection Error"
            return False
    
    def monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            if self.api_key:
                events = self.get_network_connections()
                if events:
                    self.send_to_pcds(events)
                    # Update icon to green when working
                    if self.icon:
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
    
    def open_dashboard(self):
        """Open PCDS dashboard in browser"""
        webbrowser.open(PCDS_DASHBOARD_URL)
    
    def get_status_text(self, icon):
        """Get current status for menu"""
        return f"Status: {self.last_status}"
    
    def quit_app(self, icon):
        """Quit the application"""
        self.stop_monitoring()
        icon.stop()
    
    def create_menu(self):
        """Create the tray menu"""
        return pystray.Menu(
            item(lambda text: f"PCDS Agent - {self.hostname}", lambda: None, enabled=False),
            item(lambda text: f"Status: {self.last_status}", lambda: None, enabled=False),
            pystray.Menu.SEPARATOR,
            item("Open Dashboard", lambda: self.open_dashboard()),
            item("Start Monitoring", lambda: self.start_monitoring()),
            item("Stop Monitoring", lambda: self.stop_monitoring()),
            pystray.Menu.SEPARATOR,
            item("Quit", self.quit_app)
        )
    
    def run(self):
        """Run the tray application"""
        # Check for API key
        if not self.api_key:
            # Show simple dialog to get API key
            import tkinter as tk
            from tkinter import simpledialog, messagebox
            
            root = tk.Tk()
            root.withdraw()
            
            api_key = simpledialog.askstring(
                "PCDS Setup",
                "Enter your API Key:\n\n(Get it from pcdsai.app/dashboard)",
                parent=root
            )
            
            if api_key:
                self.save_config(api_key)
            else:
                messagebox.showwarning("PCDS Agent", "No API key provided. Agent will not send data.")
            
            root.destroy()
        
        # Create tray icon
        self.icon = pystray.Icon(
            APP_NAME,
            self.create_icon_image('gray'),
            APP_NAME,
            self.create_menu()
        )
        
        # Auto-start monitoring if we have an API key
        if self.api_key:
            self.start_monitoring()
        
        # Run the tray icon (blocks until quit)
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
    # Add to startup on first run
    add_to_startup()
    
    # Run the agent
    agent = PCDSAgent()
    agent.run()

if __name__ == "__main__":
    main()
