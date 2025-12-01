from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import logging
import sqlite3
import os
import sys
from datetime import datetime
from fpdf import FPDF

# --- STARTUP FIX: Add parent directory to path so we can import ai_engine ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ai_engine.ml_engine import engine

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PCDS_Core")

app = FastAPI(title="PCDS Enterprise Core")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory Firewall ---
BLOCKED_IPS = set()

# --- Database Setup ---
DB_FILE = "pcds.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS alerts 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  timestamp TEXT, 
                  ip TEXT, 
                  type TEXT, 
                  confidence REAL, 
                  status TEXT)''')
    conn.commit()
    conn.close()

def log_alert_to_db(ip, threat_type, confidence):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute("INSERT INTO alerts (timestamp, ip, type, confidence, status) VALUES (?, ?, ?, ?, ?)",
              (timestamp, ip, threat_type, confidence, "Detected"))
    conn.commit()
    conn.close()

@app.on_event("startup")
async def startup_event():
    init_db()
    engine.load_or_train()

# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass

manager = ConnectionManager()

# --- Analysis Core ---
async def analyze_and_broadcast(packet_info, src_ip):
    if src_ip in BLOCKED_IPS:
        return # Drop packet

    analysis = engine.predict(packet_info)
    is_anomaly = analysis['prediction'] != 'Normal'

    if is_anomaly:
        log_alert_to_db(src_ip, analysis['prediction'], analysis['confidence'])

    msg = {
        "type": "traffic_update",
        "data": {
            "ip": src_ip,
            "threat_type": analysis['prediction'],
            "confidence": analysis['confidence'],
            "is_anomaly": is_anomaly,
            "packet_info": packet_info
        }
    }
    await manager.broadcast(json.dumps(msg))

# --- API Endpoints ---
class SensorData(BaseModel):
    size: int
    protocol: str
    port: int
    rate: int
    source_ip: str

@app.post("/api/sensor_input")
async def sensor_input(data: SensorData):
    packet_info = {'size': data.size, 'protocol': data.protocol, 'port': data.port, 'rate': data.rate}
    await analyze_and_broadcast(packet_info, data.source_ip)
    return {"status": "processed"}

class BlockRequest(BaseModel):
    ip: str

@app.post("/api/block_ip")
async def block_ip(req: BlockRequest):
    BLOCKED_IPS.add(req.ip)
    return {"status": "blocked", "ip": req.ip}

@app.get("/api/blocked_list")
async def get_blocked_list():
    return list(BLOCKED_IPS)

@app.get("/api/history")
async def get_history():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 50")
    rows = c.fetchall()
    conn.close()
    history = []
    for row in rows:
        history.append({"id": row[0], "timestamp": row[1], "ip": row[2], "type": row[3], "confidence": row[4], "status": row[5]})
    return history

@app.get("/api/report")
async def generate_pdf_report():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM alerts ORDER BY id DESC LIMIT 100")
    data = c.fetchall()
    conn.close()

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="PCDS Security Incident Report", ln=1, align='C')
    pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=2, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=10)
    pdf.cell(40, 10, "Time", 1)
    pdf.cell(40, 10, "IP", 1)
    pdf.cell(40, 10, "Threat", 1)
    pdf.cell(30, 10, "Conf%", 1)
    pdf.ln()

    for row in data:
        pdf.cell(40, 10, str(row[1]), 1)
        pdf.cell(40, 10, str(row[2]), 1)
        pdf.cell(40, 10, str(row[3]), 1)
        pdf.cell(30, 10, f"{row[4]}%", 1)
        pdf.ln()

    report_file = "security_report.pdf"
    pdf.output(report_file)
    return FileResponse(report_file, media_type='application/pdf', filename=report_file)

@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        manager.disconnect(websocket)