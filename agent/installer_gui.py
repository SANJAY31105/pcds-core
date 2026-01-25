import tkinter as tk
from tkinter import messagebox, ttk
import sys
import os
import shutil
import subprocess
import configparser
import ctypes

# Constants
INSTALL_DIR = r"C:\Program Files\PCDS"
EXE_NAME = "pcds_agent.exe"
SERVICE_NAME = "PCDSAgent"
API_URL = "https://pcds-backend-production.up.railway.app/api/v2/ingest"

def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def install_agent():
    api_key = key_entry.get().strip()
    if not api_key:
        messagebox.showerror("Error", "Please enter a valid License Key")
        return

    install_btn.config(state="disabled", text="Installing...")
    status_label.config(text="Status: Preparing installation...")
    root.update()

    try:
        # 1. Create Directory
        if not os.path.exists(INSTALL_DIR):
            os.makedirs(INSTALL_DIR)
        
        # 2. Copy Executable
        source_exe = get_resource_path(EXE_NAME)
        dest_exe = os.path.join(INSTALL_DIR, EXE_NAME)
        
        if not os.path.exists(source_exe):
             messagebox.showerror("Error", f"Installer corrupted. Could not find {EXE_NAME}")
             return

        status_label.config(text="Status: Copying files...")
        root.update()
        shutil.copy2(source_exe, dest_exe)

        # 3. Create Config
        config = configparser.ConfigParser()
        config['PCDS'] = {
            'api_key': api_key,
            'url': API_URL
        }
        with open(os.path.join(INSTALL_DIR, 'config.ini'), 'w') as configfile:
            config.write(configfile)

        # 4. Stop/Delete Existing Service
        status_label.config(text="Status: Configuring service...")
        root.update()
        subprocess.run(['sc', 'stop', SERVICE_NAME], capture_output=True)
        subprocess.run(['sc', 'delete', SERVICE_NAME], capture_output=True)

        # 5. Create Service
        # Note: sc create requires a space after binPath= 
        cmd = ['sc', 'create', SERVICE_NAME, f'binPath= "{dest_exe}"', 'start=', 'auto', 'DisplayName=', 'PCDS Threat Monitor']
        result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True)
        
        if result.returncode != 0 and "1073" not in result.stdout: # 1073 is "service exists" which we handled but just in case
             print(f"Service Error: {result.stderr}")

        subprocess.run(['sc', 'description', SERVICE_NAME, 'Monitors network traffic for cyber threats (PCDS AI)'], shell=True)

        # 6. Start Service
        status_label.config(text="Status: Starting agent...")
        root.update()
        subprocess.run(['sc', 'start', SERVICE_NAME], capture_output=True)

        status_label.config(text="Status: Complete!")
        messagebox.showinfo("Success", "PCDS Enterprise Agent installed successfully!\n\nYour device is now protected.")
        root.destroy()

    except Exception as e:
        messagebox.showerror("Installation Failed", f"Error: {str(e)}")
        install_btn.config(state="normal", text="Install Agent")
        status_label.config(text="Status: Failed.")

# --- UI Setup ---
if not is_admin():
    # Re-run the program with admin rights
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    sys.exit()

root = tk.Tk()
root.title("PCDS Setup")
root.geometry("400x300")
root.resizable(False, False)
root.configure(bg="#0a0a0a")

# Style
style = ttk.Style()
style.theme_use('clam')
style.configure("TLabel", background="#0a0a0a", foreground="white")
style.configure("TButton", font=('Segoe UI', 10, 'bold'))

# Header
header_frame = tk.Frame(root, bg="#0a0a0a", pady=20)
header_frame.pack()

try:
    # Use logo if available, otherwise text
    # logo_img = tk.PhotoImage(file=get_resource_path("logo.png"))
    # tk.Label(header_frame, image=logo_img, bg="#0a0a0a").pack()
    pass
except:
    pass

tk.Label(header_frame, text="PCDS Enterprise", font=("Segoe UI", 16, "bold"), fg="#22c55e", bg="#0a0a0a").pack()
tk.Label(header_frame, text="Agent Installer", font=("Segoe UI", 10), fg="#a1a1a1", bg="#0a0a0a").pack()

# Input
input_frame = tk.Frame(root, bg="#0a0a0a", pady=20)
input_frame.pack()

tk.Label(input_frame, text="Enter License Key / API Key:", font=("Segoe UI", 9), fg="white", bg="#0a0a0a").pack(anchor="w", padx=40)
key_entry = ttk.Entry(input_frame, width=35, font=("Segoe UI", 10))
key_entry.pack(pady=5)

# Buttons
install_btn = tk.Button(root, text="Install Agent", command=install_agent, 
                       bg="#22c55e", fg="white", font=("Segoe UI", 10, "bold"), 
                       relief="flat", padx=20, pady=5, cursor="hand2")
install_btn.pack(pady=10)

status_label = tk.Label(root, text="Ready to install", font=("Segoe UI", 8), fg="#666", bg="#0a0a0a")
status_label.pack(side="bottom", pady=10)

root.mainloop()
