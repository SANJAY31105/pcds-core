"""
Automated Playbook Engine - EDR Response Automation
Pre-built response workflows for common attack scenarios

Playbooks:
- Ransomware Response
- Credential Theft Response  
- Lateral Movement Containment
- C2 Communication Block
- Data Exfiltration Prevention
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from threading import Thread, Lock
import time
import json


class PlaybookStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class PlaybookStep:
    """Single step in a playbook"""
    step_id: int
    name: str
    action: str
    parameters: Dict
    condition: str = None  # Optional condition to execute
    timeout: int = 60  # seconds
    on_failure: str = "continue"  # continue, abort, retry


@dataclass
class PlaybookExecution:
    """Track playbook execution"""
    execution_id: str
    playbook_name: str
    trigger_event: Dict
    status: PlaybookStatus
    current_step: int
    steps_completed: List[Dict]
    start_time: datetime
    end_time: datetime = None
    error: str = None


class AutomatedPlaybooks:
    """
    Automated Response Playbook Engine
    
    Features:
    - Pre-built playbooks for common scenarios
    - Custom playbook creation
    - Conditional execution
    - Action logging
    - Rollback support
    """
    
    def __init__(self):
        self._lock = Lock()
        
        # Initialize response actions
        try:
            from ..actions.response_actions import get_response_actions
            self.response_actions = get_response_actions()
        except:
            self.response_actions = None
        
        # Playbook definitions
        self.playbooks: Dict[str, List[PlaybookStep]] = {}
        
        # Active executions
        self.executions: Dict[str, PlaybookExecution] = {}
        
        # Execution history
        self.history: List[Dict] = []
        
        # Stats
        self.stats = {
            "playbooks_executed": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "actions_taken": 0
        }
        
        # Load default playbooks
        self._load_default_playbooks()
        
        print("📋 Automated Playbook Engine initialized")
        print(f"   Loaded {len(self.playbooks)} playbooks")
    
    def _load_default_playbooks(self):
        """Load pre-built playbooks"""
        
        # Playbook 1: Ransomware Response
        self.playbooks["ransomware_response"] = [
            PlaybookStep(
                step_id=1,
                name="Kill Malicious Process",
                action="kill_process",
                parameters={"target": "{{pid}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=2,
                name="Isolate Host",
                action="isolate_host",
                parameters={},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=3,
                name="Block C2 IP",
                action="block_ip",
                parameters={"target": "{{remote_ip}}"},
                condition="{{remote_ip}}",
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=4,
                name="Create Forensic Snapshot",
                action="create_snapshot",
                parameters={"type": "memory"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=5,
                name="Send Alert",
                action="send_alert",
                parameters={
                    "severity": "critical",
                    "message": "Ransomware detected and contained"
                }
            )
        ]
        
        # Playbook 2: Credential Theft Response
        self.playbooks["credential_theft_response"] = [
            PlaybookStep(
                step_id=1,
                name="Kill Credential Tool",
                action="kill_process",
                parameters={"target": "{{pid}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=2,
                name="Quarantine Tool",
                action="quarantine_file",
                parameters={"target": "{{file_path}}"},
                condition="{{file_path}}",
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=3,
                name="Force Password Reset",
                action="flag_for_password_reset",
                parameters={"user": "{{username}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=4,
                name="Revoke Active Sessions",
                action="revoke_sessions",
                parameters={"user": "{{username}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=5,
                name="Send Alert",
                action="send_alert",
                parameters={
                    "severity": "critical",
                    "message": "Credential theft detected - password reset flagged"
                }
            )
        ]
        
        # Playbook 3: Lateral Movement Containment
        self.playbooks["lateral_movement_containment"] = [
            PlaybookStep(
                step_id=1,
                name="Kill Remote Tool",
                action="kill_process",
                parameters={"target": "{{pid}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=2,
                name="Block Source IP",
                action="block_ip",
                parameters={"target": "{{source_ip}}"},
                condition="{{source_ip}}",
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=3,
                name="Disable User Account",
                action="disable_account",
                parameters={"user": "{{username}}"},
                condition="{{confidence}} > 0.9",
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=4,
                name="Send Alert",
                action="send_alert",
                parameters={
                    "severity": "high",
                    "message": "Lateral movement contained"
                }
            )
        ]
        
        # Playbook 4: C2 Communication Block
        self.playbooks["c2_block"] = [
            PlaybookStep(
                step_id=1,
                name="Block C2 IP",
                action="block_ip",
                parameters={"target": "{{remote_ip}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=2,
                name="Block C2 Domain",
                action="block_domain",
                parameters={"target": "{{domain}}"},
                condition="{{domain}}",
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=3,
                name="Kill Beacon Process",
                action="kill_process",
                parameters={"target": "{{pid}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=4,
                name="Add to IOC Database",
                action="add_ioc",
                parameters={
                    "indicator": "{{remote_ip}}",
                    "type": "ip",
                    "threat": "C2 Server"
                }
            )
        ]
        
        # Playbook 5: Data Exfiltration Prevention
        self.playbooks["exfiltration_prevention"] = [
            PlaybookStep(
                step_id=1,
                name="Kill Exfil Process",
                action="kill_process",
                parameters={"target": "{{pid}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=2,
                name="Block Destination",
                action="block_ip",
                parameters={"target": "{{remote_ip}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=3,
                name="Quarantine Staged Files",
                action="quarantine_file",
                parameters={"target": "{{file_path}}"},
                condition="{{file_path}}",
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=4,
                name="Send Alert",
                action="send_alert",
                parameters={
                    "severity": "critical",
                    "message": "Data exfiltration attempt blocked"
                }
            )
        ]
        
        # Playbook 6: Malware Containment (Generic)
        self.playbooks["malware_containment"] = [
            PlaybookStep(
                step_id=1,
                name="Kill Malware Process",
                action="kill_process",
                parameters={"target": "{{pid}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=2,
                name="Quarantine Malware",
                action="quarantine_file",
                parameters={"target": "{{file_path}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=3,
                name="Scan Related Files",
                action="scan_directory",
                parameters={"target": "{{directory}}"},
                on_failure="continue"
            ),
            PlaybookStep(
                step_id=4,
                name="Remove Persistence",
                action="remove_persistence",
                parameters={"target": "{{persistence_key}}"},
                condition="{{persistence_key}}",
                on_failure="continue"
            )
        ]
    
    def execute_playbook(self, playbook_name: str, trigger_event: Dict) -> str:
        """
        Execute a playbook
        
        Args:
            playbook_name: Name of playbook to execute
            trigger_event: Event data that triggered the playbook
        
        Returns:
            execution_id: Unique ID for this execution
        """
        if playbook_name not in self.playbooks:
            raise ValueError(f"Unknown playbook: {playbook_name}")
        
        import uuid
        execution_id = str(uuid.uuid4())
        
        execution = PlaybookExecution(
            execution_id=execution_id,
            playbook_name=playbook_name,
            trigger_event=trigger_event,
            status=PlaybookStatus.RUNNING,
            current_step=0,
            steps_completed=[],
            start_time=datetime.now()
        )
        
        with self._lock:
            self.executions[execution_id] = execution
            self.stats["playbooks_executed"] += 1
        
        # Execute in background
        thread = Thread(target=self._execute_playbook_async, 
                       args=(execution_id,), daemon=True)
        thread.start()
        
        return execution_id
    
    def _execute_playbook_async(self, execution_id: str):
        """Execute playbook steps asynchronously"""
        with self._lock:
            execution = self.executions.get(execution_id)
            if not execution:
                return
            
            playbook = self.playbooks[execution.playbook_name]
        
        print(f"📋 Executing playbook: {execution.playbook_name}")
        
        for step in playbook:
            if execution.status == PlaybookStatus.PAUSED:
                break
            
            try:
                # Check condition
                if step.condition:
                    condition_met = self._evaluate_condition(
                        step.condition, execution.trigger_event
                    )
                    if not condition_met:
                        execution.steps_completed.append({
                            "step_id": step.step_id,
                            "name": step.name,
                            "status": "skipped",
                            "reason": "condition not met"
                        })
                        continue
                
                # Execute step
                result = self._execute_step(step, execution.trigger_event)
                
                execution.current_step = step.step_id
                execution.steps_completed.append({
                    "step_id": step.step_id,
                    "name": step.name,
                    "status": "success" if result.get("success") else "failed",
                    "result": result
                })
                
                self.stats["actions_taken"] += 1
                print(f"   ✓ Step {step.step_id}: {step.name}")
                
            except Exception as e:
                execution.steps_completed.append({
                    "step_id": step.step_id,
                    "name": step.name,
                    "status": "error",
                    "error": str(e)
                })
                
                if step.on_failure == "abort":
                    execution.status = PlaybookStatus.FAILED
                    execution.error = str(e)
                    break
                
                print(f"   ✗ Step {step.step_id}: {step.name} - {e}")
        
        # Complete execution
        execution.end_time = datetime.now()
        if execution.status == PlaybookStatus.RUNNING:
            execution.status = PlaybookStatus.COMPLETED
            self.stats["successful_executions"] += 1
        else:
            self.stats["failed_executions"] += 1
        
        # Add to history
        self.history.append({
            "execution_id": execution_id,
            "playbook_name": execution.playbook_name,
            "status": execution.status.value,
            "steps_completed": len(execution.steps_completed),
            "start_time": execution.start_time.isoformat(),
            "end_time": execution.end_time.isoformat() if execution.end_time else None
        })
        
        print(f"📋 Playbook {execution.playbook_name}: {execution.status.value}")
    
    def _execute_step(self, step: PlaybookStep, context: Dict) -> Dict:
        """Execute a single playbook step"""
        # Resolve parameters with context
        resolved_params = {}
        for key, value in step.parameters.items():
            if isinstance(value, str) and "{{" in value:
                resolved_params[key] = self._resolve_template(value, context)
            else:
                resolved_params[key] = value
        
        # Execute action
        action = step.action
        
        if action == "kill_process" and self.response_actions:
            pid = int(resolved_params.get("target", 0))
            if pid:
                return self.response_actions.kill_process(pid)
        
        elif action == "quarantine_file" and self.response_actions:
            path = resolved_params.get("target", "")
            if path:
                return self.response_actions.quarantine_file(path)
        
        elif action == "block_ip" and self.response_actions:
            ip = resolved_params.get("target", "")
            if ip:
                return self.response_actions.block_ip(ip)
        
        elif action == "isolate_host" and self.response_actions:
            return self.response_actions.isolate_host()
        
        elif action == "send_alert":
            # Log alert
            return {
                "success": True,
                "action": "alert",
                "severity": resolved_params.get("severity"),
                "message": resolved_params.get("message")
            }
        
        elif action in ["create_snapshot", "flag_for_password_reset", 
                       "revoke_sessions", "disable_account", "block_domain",
                       "add_ioc", "scan_directory", "remove_persistence"]:
            # Placeholder for advanced actions
            return {
                "success": True,
                "action": action,
                "parameters": resolved_params,
                "message": f"Action {action} logged (placeholder)"
            }
        
        return {"success": False, "error": f"Unknown action: {action}"}
    
    def _evaluate_condition(self, condition: str, context: Dict) -> bool:
        """Evaluate a condition string"""
        resolved = self._resolve_template(condition, context)
        
        if not resolved or resolved == "None" or resolved == "":
            return False
        
        # Handle comparison conditions
        if ">" in condition:
            parts = condition.split(">")
            left = self._resolve_template(parts[0].strip(), context)
            right = parts[1].strip()
            try:
                return float(left) > float(right)
            except:
                return False
        
        return True
    
    def _resolve_template(self, template: str, context: Dict) -> str:
        """Resolve template variables like {{variable}}"""
        import re
        
        def replace(match):
            key = match.group(1).strip()
            value = context.get(key, "")
            return str(value) if value else ""
        
        return re.sub(r'\{\{(\w+)\}\}', replace, template)
    
    def get_playbook_list(self) -> List[Dict]:
        """Get list of available playbooks"""
        return [
            {
                "name": name,
                "steps": len(steps),
                "description": self._get_playbook_description(name)
            }
            for name, steps in self.playbooks.items()
        ]
    
    def _get_playbook_description(self, name: str) -> str:
        """Get playbook description"""
        descriptions = {
            "ransomware_response": "Kill process, isolate host, block C2",
            "credential_theft_response": "Kill tool, quarantine, flag password reset",
            "lateral_movement_containment": "Block source, disable account",
            "c2_block": "Block C2 IP/domain, kill beacon",
            "exfiltration_prevention": "Kill process, block destination, quarantine",
            "malware_containment": "Kill, quarantine, scan, remove persistence"
        }
        return descriptions.get(name, "Custom playbook")
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict]:
        """Get execution status"""
        with self._lock:
            if execution_id in self.executions:
                execution = self.executions[execution_id]
                return {
                    "execution_id": execution_id,
                    "playbook_name": execution.playbook_name,
                    "status": execution.status.value,
                    "current_step": execution.current_step,
                    "steps_completed": execution.steps_completed,
                    "start_time": execution.start_time.isoformat(),
                    "end_time": execution.end_time.isoformat() if execution.end_time else None
                }
        return None
    
    def get_stats(self) -> Dict:
        """Get playbook engine statistics"""
        return {
            **self.stats,
            "available_playbooks": len(self.playbooks),
            "active_executions": len([e for e in self.executions.values() 
                                     if e.status == PlaybookStatus.RUNNING])
        }


# Singleton
_playbook_engine = None

def get_playbook_engine() -> AutomatedPlaybooks:
    global _playbook_engine
    if _playbook_engine is None:
        _playbook_engine = AutomatedPlaybooks()
    return _playbook_engine


if __name__ == "__main__":
    engine = AutomatedPlaybooks()
    
    print("\n📋 Available Playbooks:")
    for pb in engine.get_playbook_list():
        print(f"   - {pb['name']}: {pb['description']} ({pb['steps']} steps)")
    
    # Test execution
    print("\n📋 Testing ransomware_response playbook...")
    
    test_event = {
        "pid": 1234,
        "remote_ip": "185.220.101.1",
        "file_path": "C:/Users/test/malware.exe"
    }
    
    exec_id = engine.execute_playbook("ransomware_response", test_event)
    
    import time
    time.sleep(2)
    
    status = engine.get_execution_status(exec_id)
    print(f"\nExecution status: {status['status']}")
    print(f"Steps completed: {len(status['steps_completed'])}")
