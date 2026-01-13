"""
EDR API Endpoints
Exposes EDR agent functionality via REST API
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

router = APIRouter(prefix="/api/v2/edr", tags=["EDR"])

# EDR Agent instance
_edr_agent = None

def get_agent():
    """Get or create EDR agent"""
    global _edr_agent
    if _edr_agent is None:
        try:
            from edr import get_edr_agent
            _edr_agent = get_edr_agent(auto_response=True, response_threshold=0.90)
        except Exception as e:
            print(f"⚠️ Failed to initialize EDR agent: {e}")
            return None
    return _edr_agent


class EDRConfigRequest(BaseModel):
    """EDR configuration request"""
    auto_response: Optional[bool] = None
    response_threshold: Optional[float] = None


class ResponseActionRequest(BaseModel):
    """Response action request"""
    action: str  # kill_process, quarantine, block_ip, isolate_host
    target: str  # PID, filepath, IP, etc.


@router.get("/status")
async def get_edr_status():
    """Get EDR agent status and stats"""
    agent = get_agent()
    
    if agent is None:
        return {
            "status": "unavailable",
            "message": "EDR agent not initialized"
        }
    
    return {
        "status": "running" if agent._running else "stopped",
        "auto_response": agent.auto_response,
        "response_threshold": agent.response_threshold,
        "stats": agent.get_stats()
    }


@router.post("/start")
async def start_edr():
    """Start EDR agent"""
    agent = get_agent()
    
    if agent is None:
        raise HTTPException(status_code=500, detail="EDR agent not available")
    
    if agent._running:
        return {"status": "already_running"}
    
    agent.start()
    return {"status": "started"}


@router.post("/stop")
async def stop_edr():
    """Stop EDR agent"""
    agent = get_agent()
    
    if agent is None:
        raise HTTPException(status_code=500, detail="EDR agent not available")
    
    if not agent._running:
        return {"status": "already_stopped"}
    
    agent.stop()
    return {"status": "stopped"}


@router.put("/config")
async def update_edr_config(config: EDRConfigRequest):
    """Update EDR configuration"""
    agent = get_agent()
    
    if agent is None:
        raise HTTPException(status_code=500, detail="EDR agent not available")
    
    if config.auto_response is not None:
        agent.set_auto_response(config.auto_response)
    
    if config.response_threshold is not None:
        agent.set_response_threshold(config.response_threshold)
    
    return {
        "status": "updated",
        "auto_response": agent.auto_response,
        "response_threshold": agent.response_threshold
    }


@router.get("/detections")
async def get_recent_detections(count: int = 50):
    """Get recent EDR detections"""
    agent = get_agent()
    
    if agent is None:
        return {"detections": []}
    
    return {"detections": agent.get_recent_detections(count)}


@router.get("/monitors")
async def get_monitor_stats():
    """Get individual monitor statistics"""
    agent = get_agent()
    
    if agent is None:
        return {"monitors": {}}
    
    stats = agent.get_stats()
    
    return {
        "monitors": {
            "process": stats.get("process_monitor", {}),
            "file": stats.get("file_monitor", {}),
            "registry": stats.get("registry_monitor", {}),
            "network": stats.get("network_monitor", {})
        }
    }


@router.post("/response")
async def execute_response_action(request: ResponseActionRequest):
    """Execute a response action"""
    agent = get_agent()
    
    if agent is None:
        raise HTTPException(status_code=500, detail="EDR agent not available")
    
    actions = agent.response_actions
    
    result = None
    
    if request.action == "kill_process":
        try:
            pid = int(request.target)
            result = actions.kill_process(pid)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid PID")
    
    elif request.action == "quarantine":
        result = actions.quarantine_file(request.target)
    
    elif request.action == "block_ip":
        result = actions.block_ip(request.target)
    
    elif request.action == "isolate_host":
        result = actions.isolate_host()
    
    elif request.action == "remove_isolation":
        result = actions.remove_isolation()
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
    
    return result


@router.get("/response/log")
async def get_response_log():
    """Get response action log"""
    agent = get_agent()
    
    if agent is None:
        return {"actions": []}
    
    return {"actions": agent.response_actions.get_action_log()}
