"""
Response Actions - EDR Response Layer
Automated threat response capabilities

Actions:
- Kill process
- Quarantine file
- Block IP
- Isolate host
"""

from typing import Dict, Any, Optional
import psutil
import subprocess
import shutil
import os
from datetime import datetime
from pathlib import Path


class ResponseActions:
    """
    EDR Response Actions
    
    Provides automated response capabilities:
    - Kill malicious processes
    - Quarantine suspicious files
    - Block malicious IPs
    - Isolate compromised hosts
    """
    
    def __init__(self):
        self.quarantine_dir = Path("C:/ProgramData/PCDS/quarantine")
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        self.action_log = []
        
        print("🛡️ Response Actions initialized")
    
    def kill_process(self, pid: int, force: bool = False) -> Dict:
        """
        Kill a malicious process
        
        Args:
            pid: Process ID to kill
            force: Force kill even if protected
        
        Returns:
            Result with success status
        """
        try:
            process = psutil.Process(pid)
            process_name = process.name()
            
            # Safety check - don't kill critical processes
            critical_processes = ["system", "smss.exe", "csrss.exe", "wininit.exe", 
                                  "services.exe", "lsass.exe", "winlogon.exe", "explorer.exe"]
            
            if process_name.lower() in critical_processes and not force:
                return {
                    "success": False,
                    "action": "kill_process",
                    "pid": pid,
                    "error": f"Protected process: {process_name}"
                }
            
            # Kill process and children
            children = process.children(recursive=True)
            for child in children:
                try:
                    child.kill()
                except:
                    pass
            
            process.kill()
            
            result = {
                "success": True,
                "action": "kill_process",
                "pid": pid,
                "process_name": process_name,
                "children_killed": len(children),
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_action(result)
            print(f"🔴 KILLED: {process_name} (PID: {pid})")
            
            return result
            
        except psutil.NoSuchProcess:
            return {"success": False, "action": "kill_process", "pid": pid, "error": "Process not found"}
        except psutil.AccessDenied:
            return {"success": False, "action": "kill_process", "pid": pid, "error": "Access denied"}
        except Exception as e:
            return {"success": False, "action": "kill_process", "pid": pid, "error": str(e)}
    
    def quarantine_file(self, filepath: str) -> Dict:
        """
        Move suspicious file to quarantine
        
        Args:
            filepath: Path to file to quarantine
        
        Returns:
            Result with success status
        """
        try:
            source = Path(filepath)
            
            if not source.exists():
                return {"success": False, "action": "quarantine", "filepath": filepath, "error": "File not found"}
            
            # Create unique quarantine name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            quarantine_name = f"{timestamp}_{source.name}"
            destination = self.quarantine_dir / quarantine_name
            
            # Move file
            shutil.move(str(source), str(destination))
            
            # Save metadata
            metadata = {
                "original_path": filepath,
                "quarantine_path": str(destination),
                "timestamp": datetime.now().isoformat(),
                "file_size": destination.stat().st_size if destination.exists() else 0
            }
            
            metadata_file = destination.with_suffix(destination.suffix + ".meta")
            with open(metadata_file, "w") as f:
                import json
                json.dump(metadata, f, indent=2)
            
            result = {
                "success": True,
                "action": "quarantine",
                "original_path": filepath,
                "quarantine_path": str(destination),
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_action(result)
            print(f"🔒 QUARANTINED: {source.name}")
            
            return result
            
        except Exception as e:
            return {"success": False, "action": "quarantine", "filepath": filepath, "error": str(e)}
    
    def block_ip(self, ip_address: str, direction: str = "both") -> Dict:
        """
        Block IP address using Windows Firewall
        
        Args:
            ip_address: IP to block
            direction: "in", "out", or "both"
        
        Returns:
            Result with success status
        """
        try:
            rule_name = f"PCDS_Block_{ip_address.replace('.', '_')}"
            
            # Block inbound
            if direction in ["in", "both"]:
                cmd_in = f'netsh advfirewall firewall add rule name="{rule_name}_IN" dir=in action=block remoteip={ip_address}'
                subprocess.run(cmd_in, shell=True, capture_output=True, timeout=10)
            
            # Block outbound
            if direction in ["out", "both"]:
                cmd_out = f'netsh advfirewall firewall add rule name="{rule_name}_OUT" dir=out action=block remoteip={ip_address}'
                subprocess.run(cmd_out, shell=True, capture_output=True, timeout=10)
            
            result = {
                "success": True,
                "action": "block_ip",
                "ip_address": ip_address,
                "direction": direction,
                "rule_name": rule_name,
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_action(result)
            print(f"🚫 BLOCKED IP: {ip_address}")
            
            return result
            
        except Exception as e:
            return {"success": False, "action": "block_ip", "ip_address": ip_address, "error": str(e)}
    
    def isolate_host(self, allow_ip: str = None) -> Dict:
        """
        Isolate host from network (DANGEROUS - use with caution)
        
        Args:
            allow_ip: Optional IP to still allow (e.g., management server)
        
        Returns:
            Result with success status
        """
        try:
            # Create isolation rules
            rules_created = []
            
            # Block all outbound
            cmd_out = 'netsh advfirewall firewall add rule name="PCDS_Isolate_OUT" dir=out action=block'
            result = subprocess.run(cmd_out, shell=True, capture_output=True, timeout=10)
            if result.returncode == 0:
                rules_created.append("PCDS_Isolate_OUT")
            
            # Block all inbound
            cmd_in = 'netsh advfirewall firewall add rule name="PCDS_Isolate_IN" dir=in action=block'
            result = subprocess.run(cmd_in, shell=True, capture_output=True, timeout=10)
            if result.returncode == 0:
                rules_created.append("PCDS_Isolate_IN")
            
            # Allow specific IP if provided
            if allow_ip:
                cmd_allow_out = f'netsh advfirewall firewall add rule name="PCDS_Allow_Management_OUT" dir=out action=allow remoteip={allow_ip}'
                cmd_allow_in = f'netsh advfirewall firewall add rule name="PCDS_Allow_Management_IN" dir=in action=allow remoteip={allow_ip}'
                subprocess.run(cmd_allow_out, shell=True, capture_output=True, timeout=10)
                subprocess.run(cmd_allow_in, shell=True, capture_output=True, timeout=10)
                rules_created.extend(["PCDS_Allow_Management_OUT", "PCDS_Allow_Management_IN"])
            
            result = {
                "success": True,
                "action": "isolate_host",
                "rules_created": rules_created,
                "allow_ip": allow_ip,
                "timestamp": datetime.now().isoformat(),
                "warning": "Host is now isolated from network!"
            }
            
            self._log_action(result)
            print(f"🔒 HOST ISOLATED - Network blocked except: {allow_ip or 'none'}")
            
            return result
            
        except Exception as e:
            return {"success": False, "action": "isolate_host", "error": str(e)}
    
    def remove_isolation(self) -> Dict:
        """Remove host isolation"""
        try:
            # Remove isolation rules
            rules = ["PCDS_Isolate_OUT", "PCDS_Isolate_IN", "PCDS_Allow_Management_OUT", "PCDS_Allow_Management_IN"]
            
            for rule in rules:
                cmd = f'netsh advfirewall firewall delete rule name="{rule}"'
                subprocess.run(cmd, shell=True, capture_output=True, timeout=10)
            
            result = {
                "success": True,
                "action": "remove_isolation",
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_action(result)
            print("🔓 HOST ISOLATION REMOVED")
            
            return result
            
        except Exception as e:
            return {"success": False, "action": "remove_isolation", "error": str(e)}
    
    def unblock_ip(self, ip_address: str) -> Dict:
        """Remove IP block"""
        try:
            rule_name = f"PCDS_Block_{ip_address.replace('.', '_')}"
            
            cmd_in = f'netsh advfirewall firewall delete rule name="{rule_name}_IN"'
            cmd_out = f'netsh advfirewall firewall delete rule name="{rule_name}_OUT"'
            
            subprocess.run(cmd_in, shell=True, capture_output=True, timeout=10)
            subprocess.run(cmd_out, shell=True, capture_output=True, timeout=10)
            
            result = {
                "success": True,
                "action": "unblock_ip",
                "ip_address": ip_address,
                "timestamp": datetime.now().isoformat()
            }
            
            self._log_action(result)
            print(f"✅ UNBLOCKED IP: {ip_address}")
            
            return result
            
        except Exception as e:
            return {"success": False, "action": "unblock_ip", "ip_address": ip_address, "error": str(e)}
    
    def _log_action(self, result: Dict):
        """Log action for audit"""
        self.action_log.append(result)
    
    def get_action_log(self) -> list:
        """Get action history"""
        return self.action_log


# Confidence-based auto response
def auto_respond(detection: Dict, confidence: float, actions: ResponseActions) -> Optional[Dict]:
    """
    Automated response based on confidence level
    
    Confidence Levels:
    - 0.90+: Automatic containment
    - 0.75-0.90: Alert + recommend action
    - 0.50-0.75: Log only
    - <0.50: Ignore
    """
    
    if confidence >= 0.90:
        # CRITICAL - Automatic containment
        if detection.get("type") == "process":
            pid = detection.get("data", {}).get("pid")
            if pid:
                return actions.kill_process(pid)
        
        elif detection.get("type") == "file":
            filepath = detection.get("data", {}).get("filepath")
            if filepath:
                return actions.quarantine_file(filepath)
        
        elif detection.get("type") == "network":
            ip = detection.get("data", {}).get("remote_ip")
            if ip:
                return actions.block_ip(ip)
    
    elif confidence >= 0.75:
        # HIGH - Alert but don't auto-respond
        print(f"⚠️ HIGH CONFIDENCE ({confidence:.2f}): Manual review recommended")
        return {
            "action": "alert",
            "confidence": confidence,
            "recommendation": "Manual review required",
            "detection": detection
        }
    
    # Below 0.75 - log only
    return None


# Singleton
_response_actions = None

def get_response_actions() -> ResponseActions:
    global _response_actions
    if _response_actions is None:
        _response_actions = ResponseActions()
    return _response_actions
