"""
PCDS Enterprise Setup
Modern installer for PCDS System Tray Agent
"""

import tkinter as tk
from tkinter import messagebox, ttk
import sys
import os
import shutil
import subprocess
import configparser
import ctypes
import winreg

# Constants
INSTALL_DIR = os.path.join(os.environ.get('LOCALAPPDATA', 'C:\\'), 'PCDS')
EXE_NAME = "pcds_tray_agent.exe"
API_URL = "https://pcds-backend-production.up.railway.app/api/v2/ingest"
DASHBOARD_URL = "https://pcdsai.app/dashboard"
APP_NAME = "PCDS Agent"

def is_admin():
    """Check if running with admin privileges (not required for tray app)"""
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and PyInstaller"""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def add_to_startup(exe_path):
    """Add application to Windows startup"""
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE
        )
        winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, f'"{exe_path}"')
        winreg.CloseKey(key)
        return True
    except Exception as e:
        print(f"Startup registry error: {e}")
        return False

def create_desktop_shortcut(exe_path):
    """Create a desktop shortcut"""
    try:
        desktop = os.path.join(os.path.expanduser("~"), "Desktop")
        shortcut_path = os.path.join(desktop, "PCDS Agent.lnk")
        
        # Use PowerShell to create shortcut
        ps_script = f'''
        $WshShell = New-Object -comObject WScript.Shell
        $Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
        $Shortcut.TargetPath = "{exe_path}"
        $Shortcut.WorkingDirectory = "{INSTALL_DIR}"
        $Shortcut.Description = "PCDS Enterprise Threat Monitor"
        $Shortcut.Save()
        '''
        subprocess.run(['powershell', '-Command', ps_script], capture_output=True)
        return True
    except:
        return False

class InstallerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("PCDS Setup")
        self.root.geometry("450x400")
        self.root.resizable(False, False)
        self.root.configure(bg="#0f0f0f")
        
        # Center the window
        self.center_window()
        
        self.setup_ui()
    
    def center_window(self):
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{450}x{400}+{x}+{y}')
    
    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#0f0f0f")
        header.pack(pady=30)
        
        tk.Label(
            header, 
            text="🛡️", 
            font=("Segoe UI Emoji", 36),
            bg="#0f0f0f"
        ).pack()
        
        tk.Label(
            header,
            text="PCDS Enterprise",
            font=("Segoe UI", 20, "bold"),
            fg="#22c55e",
            bg="#0f0f0f"
        ).pack()
        
        tk.Label(
            header,
            text="AI-Powered Threat Detection",
            font=("Segoe UI", 10),
            fg="#888888",
            bg="#0f0f0f"
        ).pack(pady=5)
        
        # API Key Input
        input_frame = tk.Frame(self.root, bg="#0f0f0f")
        input_frame.pack(pady=20, padx=40, fill="x")
        
        tk.Label(
            input_frame,
            text="License Key",
            font=("Segoe UI", 9, "bold"),
            fg="#cccccc",
            bg="#0f0f0f"
        ).pack(anchor="w")
        
        self.key_entry = tk.Entry(
            input_frame,
            font=("Consolas", 11),
            bg="#1a1a1a",
            fg="white",
            insertbackground="white",
            relief="flat",
            highlightthickness=1,
            highlightcolor="#22c55e",
            highlightbackground="#333333"
        )
        self.key_entry.pack(fill="x", pady=5, ipady=8)
        
        # Bind Enter key to install
        self.key_entry.bind('<Return>', lambda e: self.install())
        self.key_entry.focus_set()  # Auto-focus the input field
        
        tk.Label(
            input_frame,
            text="Get your key at pcdsai.app/dashboard",
            font=("Segoe UI", 8),
            fg="#666666",
            bg="#0f0f0f"
        ).pack(anchor="w")
        
        # Options
        options_frame = tk.Frame(self.root, bg="#0f0f0f")
        options_frame.pack(pady=10, padx=40, fill="x")
        
        self.startup_var = tk.BooleanVar(value=True)
        self.shortcut_var = tk.BooleanVar(value=True)
        
        tk.Checkbutton(
            options_frame,
            text="Start with Windows",
            variable=self.startup_var,
            font=("Segoe UI", 9),
            fg="#cccccc",
            bg="#0f0f0f",
            activebackground="#0f0f0f",
            activeforeground="white",
            selectcolor="#1a1a1a"
        ).pack(anchor="w")
        
        tk.Checkbutton(
            options_frame,
            text="Create Desktop Shortcut",
            variable=self.shortcut_var,
            font=("Segoe UI", 9),
            fg="#cccccc",
            bg="#0f0f0f",
            activebackground="#0f0f0f",
            activeforeground="white",
            selectcolor="#1a1a1a"
        ).pack(anchor="w")
        
        # Install Button
        self.install_btn = tk.Button(
            self.root,
            text="Install & Start Protection",
            command=self.install,
            font=("Segoe UI", 11, "bold"),
            bg="#22c55e",
            fg="white",
            activebackground="#16a34a",
            activeforeground="white",
            relief="flat",
            cursor="hand2",
            padx=20,
            pady=10
        )
        self.install_btn.pack(pady=15)
        
        # Status
        self.status_label = tk.Label(
            self.root,
            text="Ready to install",
            font=("Segoe UI", 9),
            fg="#666666",
            bg="#0f0f0f"
        )
        self.status_label.pack(side="bottom", pady=15)
    
    def update_status(self, text):
        self.status_label.config(text=text)
        self.root.update()
    
    def install(self):
        api_key = self.key_entry.get().strip()
        if not api_key:
            messagebox.showerror("Error", "Please enter your License Key")
            return
        
        self.install_btn.config(state="disabled", text="Installing...")
        
        try:
            # 1. Create install directory
            self.update_status("Creating directories...")
            os.makedirs(INSTALL_DIR, exist_ok=True)
            
            # 2. Copy executable
            self.update_status("Copying files...")
            source_exe = get_resource_path(EXE_NAME)
            dest_exe = os.path.join(INSTALL_DIR, EXE_NAME)
            
            if not os.path.exists(source_exe):
                messagebox.showerror("Error", f"Installer corrupted. Missing {EXE_NAME}")
                self.reset_button()
                return
            
            shutil.copy2(source_exe, dest_exe)
            
            # 3. Create config
            self.update_status("Saving configuration...")
            config = configparser.ConfigParser()
            config['PCDS'] = {
                'api_key': api_key,
                'url': API_URL
            }
            config_path = os.path.join(INSTALL_DIR, 'config.ini')
            with open(config_path, 'w') as f:
                config.write(f)
            
            # 4. Add to startup
            if self.startup_var.get():
                self.update_status("Configuring startup...")
                add_to_startup(dest_exe)
            
            # 5. Create shortcut
            if self.shortcut_var.get():
                self.update_status("Creating shortcut...")
                create_desktop_shortcut(dest_exe)
            
            # 6. Launch the agent
            self.update_status("Starting agent...")
            subprocess.Popen([dest_exe], cwd=INSTALL_DIR)
            
            # Success!
            self.update_status("Installation complete!")
            messagebox.showinfo(
                "Success",
                "PCDS Agent installed successfully!\n\n"
                "The agent is now running in your system tray.\n"
                "Look for the shield icon near your clock."
            )
            self.root.destroy()
            
        except Exception as e:
            messagebox.showerror("Installation Failed", f"Error: {str(e)}")
            self.reset_button()
    
    def reset_button(self):
        self.install_btn.config(state="normal", text="Install & Start Protection")
        self.update_status("Ready to install")

def main():
    root = tk.Tk()
    app = InstallerApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
