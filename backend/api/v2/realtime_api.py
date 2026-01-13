"""
Real-time ML Pipeline API
Control and monitor the real-time ML prediction pipeline
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import asyncio

router = APIRouter(tags=["Real-time Pipeline"])

# WebSocket connections
active_connections: Set[WebSocket] = set()


async def broadcast_to_clients(message: Dict):
    """Broadcast message to all connected WebSocket clients"""
    disconnected = set()
    for ws in active_connections:
        try:
            await ws.send_json(message)
        except:
            disconnected.add(ws)
    
    for ws in disconnected:
        active_connections.discard(ws)


@router.get("/status")
async def get_pipeline_status() -> Dict:
    """Get real-time pipeline status"""
    from ml.realtime_pipeline import get_realtime_pipeline
    
    pipeline = get_realtime_pipeline()
    return {
        "status": "running" if pipeline.running else "stopped",
        "stats": pipeline.get_stats()
    }


@router.post("/start")
async def start_pipeline() -> Dict:
    """Start the real-time ML pipeline"""
    from ml.realtime_pipeline import get_realtime_pipeline
    
    pipeline = get_realtime_pipeline()
    
    if not pipeline.running:
        # Set WebSocket broadcast callback
        pipeline.set_ws_callback(broadcast_to_clients)
        await pipeline.start()
        
        return {"status": "started", "message": "Real-time pipeline started"}
    
    return {"status": "already_running", "message": "Pipeline already running"}


@router.post("/stop")
async def stop_pipeline() -> Dict:
    """Stop the real-time ML pipeline"""
    from ml.realtime_pipeline import get_realtime_pipeline
    
    pipeline = get_realtime_pipeline()
    await pipeline.stop()
    
    return {"status": "stopped", "message": "Pipeline stopped"}


@router.post("/simulate/start")
async def start_simulation(events_per_second: float = 2.0) -> Dict:
    """Start event simulation for testing"""
    from ml.realtime_pipeline import get_realtime_pipeline, get_event_simulator
    
    pipeline = get_realtime_pipeline()
    simulator = get_event_simulator()
    
    # Start pipeline if not running
    if not pipeline.running:
        pipeline.set_ws_callback(broadcast_to_clients)
        await pipeline.start()
    
    # Start simulator in background
    asyncio.create_task(simulator.start(events_per_second))
    
    return {
        "status": "started",
        "events_per_second": events_per_second,
        "message": "Event simulation started"
    }


@router.post("/simulate/stop")
async def stop_simulation() -> Dict:
    """Stop event simulation"""
    from ml.realtime_pipeline import get_event_simulator
    
    simulator = get_event_simulator()
    simulator.stop()
    
    return {"status": "stopped", "message": "Simulation stopped"}


@router.websocket("/predictions")
async def websocket_predictions(websocket: WebSocket):
    """WebSocket endpoint for real-time predictions"""
    await websocket.accept()
    active_connections.add(websocket)
    
    try:
        # Send initial status
        from ml.realtime_pipeline import get_realtime_pipeline
        pipeline = get_realtime_pipeline()
        
        await websocket.send_json({
            "type": "connected",
            "pipeline_running": pipeline.running,
            "stats": pipeline.get_stats()
        })
        
        # Keep connection alive
        while True:
            try:
                # Wait for client messages (ping/pong)
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
                elif data == "stats":
                    await websocket.send_json({
                        "type": "stats", 
                        "data": pipeline.get_stats()
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        pass
    finally:
        active_connections.discard(websocket)
