"""
WebSocket connection manager for real-time communication
"""
from fastapi import WebSocket
from typing import List, Dict, Set
import json
import asyncio
from datetime import datetime


class ConnectionManager:
    """Manages WebSocket connections and broadcasting"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.rooms: Dict[str, Set[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, room: str = "default"):
        """Accept and register new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        
        if room not in self.rooms:
            self.rooms[room] = set()
        self.rooms[room].add(websocket)
        
        print(f"✅ Client connected to room '{room}'. Total connections: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket, room: str = "default"):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            
        if room in self.rooms and websocket in self.rooms[room]:
            self.rooms[room].remove(websocket)
            
        print(f"❌ Client disconnected from room '{room}'. Total connections: {len(self.active_connections)}")
        
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending personal message: {e}")
            
    async def broadcast(self, message: dict, room: str = "default"):
        """Broadcast message to all clients in room"""
        if room not in self.rooms:
            return
            
        message['timestamp'] = datetime.utcnow().isoformat()
        
        disconnected = []
        for connection in self.rooms[room]:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
                
        # Clean up disconnected clients
        for conn in disconnected:
            self.disconnect(conn, room)
            
    async def broadcast_to_all(self, message: dict):
        """Broadcast message to all connected clients"""
        message['timestamp'] = datetime.utcnow().isoformat()
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to all: {e}")
                disconnected.append(connection)
                
        # Clean up disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
                
    async def heartbeat(self):
        """Send periodic heartbeat to keep connections alive"""
        while True:
            await asyncio.sleep(30)
            await self.broadcast_to_all({"type": "heartbeat", "status": "alive"})
            
    def get_connection_count(self, room: str = None) -> int:
        """Get number of active connections"""
        if room:
            return len(self.rooms.get(room, set()))
        return len(self.active_connections)


# Global connection manager instance
manager = ConnectionManager()
