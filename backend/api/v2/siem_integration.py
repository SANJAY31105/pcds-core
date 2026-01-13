"""
SIEM Integration API
Export alerts to Splunk, Elastic, QRadar and other SIEM platforms

Formats:
- Syslog (RFC 5424)
- CEF (Common Event Format)
- LEEF (Log Event Extended Format)
- JSON (Generic)

Transport:
- Syslog UDP/TCP
- Webhook (HTTP POST)
- Splunk HEC (HTTP Event Collector)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime
from enum import Enum
import socket
import json
import httpx
from threading import Lock
import asyncio


router = APIRouter(prefix="/siem", tags=["SIEM Integration"])


class SIEMFormat(str, Enum):
    SYSLOG = "syslog"
    CEF = "cef"
    LEEF = "leef"
    JSON = "json"


class SIEMTransport(str, Enum):
    SYSLOG_UDP = "syslog_udp"
    SYSLOG_TCP = "syslog_tcp"
    WEBHOOK = "webhook"
    SPLUNK_HEC = "splunk_hec"


class SIEMConfig(BaseModel):
    """SIEM destination configuration"""
    name: str
    enabled: bool = True
    transport: SIEMTransport
    format: SIEMFormat
    host: str
    port: int = 514
    token: Optional[str] = None  # For Splunk HEC
    webhook_url: Optional[str] = None
    include_raw: bool = False
    min_severity: str = "low"  # low, medium, high, critical


class Alert(BaseModel):
    """Alert to send to SIEM"""
    id: str
    timestamp: str
    severity: str  # low, medium, high, critical
    title: str
    description: str
    source_ip: Optional[str] = None
    dest_ip: Optional[str] = None
    user: Optional[str] = None
    process: Optional[str] = None
    mitre_tactic: Optional[str] = None
    mitre_technique: Optional[str] = None
    raw_data: Optional[Dict] = None


# SIEM configurations
siem_configs: Dict[str, SIEMConfig] = {}
siem_stats: Dict[str, Dict] = {}
_lock = Lock()


# Severity levels
SEVERITY_MAP = {
    "low": 1,
    "medium": 5,
    "high": 8,
    "critical": 10
}


def format_syslog(alert: Alert, config: SIEMConfig) -> str:
    """
    Format alert as RFC 5424 Syslog
    
    Format: <PRI>VERSION TIMESTAMP HOSTNAME APP-NAME PROCID MSGID STRUCTURED-DATA MSG
    """
    severity = SEVERITY_MAP.get(alert.severity, 5)
    facility = 1  # user-level
    pri = facility * 8 + min(severity, 7)
    
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    hostname = socket.gethostname()
    app_name = "PCDS"
    proc_id = "-"
    msg_id = alert.id
    
    # Structured data
    sd = f'[pcds@1 severity="{alert.severity}" title="{alert.title}"'
    if alert.source_ip:
        sd += f' src="{alert.source_ip}"'
    if alert.dest_ip:
        sd += f' dst="{alert.dest_ip}"'
    if alert.mitre_technique:
        sd += f' mitre="{alert.mitre_technique}"'
    sd += ']'
    
    msg = alert.description
    
    return f"<{pri}>1 {timestamp} {hostname} {app_name} {proc_id} {msg_id} {sd} {msg}"


def format_cef(alert: Alert, config: SIEMConfig) -> str:
    """
    Format alert as CEF (Common Event Format)
    
    Format: CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
    """
    severity = SEVERITY_MAP.get(alert.severity, 5)
    
    # Extension fields
    extensions = []
    if alert.source_ip:
        extensions.append(f"src={alert.source_ip}")
    if alert.dest_ip:
        extensions.append(f"dst={alert.dest_ip}")
    if alert.user:
        extensions.append(f"suser={alert.user}")
    if alert.process:
        extensions.append(f"sproc={alert.process}")
    if alert.mitre_tactic:
        extensions.append(f"cs1={alert.mitre_tactic}")
        extensions.append("cs1Label=MITRE Tactic")
    if alert.mitre_technique:
        extensions.append(f"cs2={alert.mitre_technique}")
        extensions.append("cs2Label=MITRE Technique")
    
    extensions.append(f"msg={alert.description}")
    extensions.append(f"rt={datetime.now().strftime('%b %d %Y %H:%M:%S')}")
    
    ext_str = " ".join(extensions)
    
    return f"CEF:0|PCDS|Enterprise NDR|2.0|{alert.id}|{alert.title}|{severity}|{ext_str}"


def format_leef(alert: Alert, config: SIEMConfig) -> str:
    """
    Format alert as LEEF (Log Event Extended Format) for QRadar
    
    Format: LEEF:Version|Vendor|Product|Version|EventID|Key=Value pairs
    """
    # LEEF attributes
    attrs = []
    attrs.append(f"cat={alert.severity}")
    attrs.append(f"devTime={datetime.now().strftime('%b %d %Y %H:%M:%S')}")
    
    if alert.source_ip:
        attrs.append(f"src={alert.source_ip}")
    if alert.dest_ip:
        attrs.append(f"dst={alert.dest_ip}")
    if alert.user:
        attrs.append(f"usrName={alert.user}")
    if alert.process:
        attrs.append(f"eventDesc={alert.process}")
    
    attrs.append(f"msg={alert.description}")
    
    attr_str = "\t".join(attrs)
    
    return f"LEEF:2.0|PCDS|Enterprise NDR|2.0|{alert.id}|{attr_str}"


def format_json(alert: Alert, config: SIEMConfig) -> str:
    """Format alert as JSON"""
    data = {
        "timestamp": alert.timestamp,
        "id": alert.id,
        "severity": alert.severity,
        "title": alert.title,
        "description": alert.description,
        "source": {
            "ip": alert.source_ip,
            "user": alert.user,
            "process": alert.process
        },
        "destination": {
            "ip": alert.dest_ip
        },
        "mitre": {
            "tactic": alert.mitre_tactic,
            "technique": alert.mitre_technique
        },
        "product": "PCDS Enterprise NDR",
        "product_version": "2.0"
    }
    
    if config.include_raw and alert.raw_data:
        data["raw"] = alert.raw_data
    
    return json.dumps(data)


async def send_syslog_udp(message: str, host: str, port: int):
    """Send message via Syslog UDP"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(message.encode('utf-8'), (host, port))
        sock.close()
        return True
    except Exception as e:
        print(f"Syslog UDP error: {e}")
        return False


async def send_syslog_tcp(message: str, host: str, port: int):
    """Send message via Syslog TCP"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        sock.connect((host, port))
        sock.send(f"{message}\n".encode('utf-8'))
        sock.close()
        return True
    except Exception as e:
        print(f"Syslog TCP error: {e}")
        return False


async def send_webhook(message: str, url: str, format_type: str):
    """Send alert via webhook"""
    try:
        async with httpx.AsyncClient() as client:
            if format_type == "json":
                response = await client.post(
                    url,
                    json=json.loads(message),
                    timeout=10
                )
            else:
                response = await client.post(
                    url,
                    content=message,
                    headers={"Content-Type": "text/plain"},
                    timeout=10
                )
            return response.status_code < 400
    except Exception as e:
        print(f"Webhook error: {e}")
        return False


async def send_splunk_hec(message: str, host: str, port: int, token: str):
    """Send alert to Splunk HTTP Event Collector"""
    try:
        url = f"https://{host}:{port}/services/collector/event"
        
        event_data = {
            "event": json.loads(message) if message.startswith("{") else message,
            "sourcetype": "pcds:alert",
            "source": "pcds_ndr"
        }
        
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(
                url,
                json=event_data,
                headers={
                    "Authorization": f"Splunk {token}",
                    "Content-Type": "application/json"
                },
                timeout=10
            )
            return response.status_code < 400
    except Exception as e:
        print(f"Splunk HEC error: {e}")
        return False


async def send_alert_to_siem(alert: Alert, config: SIEMConfig) -> bool:
    """Send alert to configured SIEM"""
    # Check severity threshold
    alert_level = SEVERITY_MAP.get(alert.severity, 0)
    min_level = SEVERITY_MAP.get(config.min_severity, 0)    
    if alert_level < min_level:
        return False
    
    # Format message
    if config.format == SIEMFormat.SYSLOG:
        message = format_syslog(alert, config)
    elif config.format == SIEMFormat.CEF:
        message = format_cef(alert, config)
    elif config.format == SIEMFormat.LEEF:
        message = format_leef(alert, config)
    else:
        message = format_json(alert, config)
    
    # Send via transport
    success = False
    if config.transport == SIEMTransport.SYSLOG_UDP:
        success = await send_syslog_udp(message, config.host, config.port)
    elif config.transport == SIEMTransport.SYSLOG_TCP:
        success = await send_syslog_tcp(message, config.host, config.port)
    elif config.transport == SIEMTransport.WEBHOOK:
        success = await send_webhook(message, config.webhook_url, config.format)
    elif config.transport == SIEMTransport.SPLUNK_HEC:
        success = await send_splunk_hec(message, config.host, config.port, config.token)
    
    # Update stats
    with _lock:
        if config.name not in siem_stats:
            siem_stats[config.name] = {"sent": 0, "failed": 0, "last_sent": None}
        
        if success:
            siem_stats[config.name]["sent"] += 1
            siem_stats[config.name]["last_sent"] = datetime.now().isoformat()
        else:
            siem_stats[config.name]["failed"] += 1
    
    return success


@router.post("/config")
async def add_siem_config(config: SIEMConfig) -> Dict:
    """Add or update SIEM configuration"""
    siem_configs[config.name] = config
    siem_stats[config.name] = {"sent": 0, "failed": 0, "last_sent": None}
    
    return {"added": True, "name": config.name}


@router.get("/config")
async def get_siem_configs() -> Dict:
    """Get all SIEM configurations"""
    return {
        "configs": [
            {
                **config.dict(),
                "token": "***" if config.token else None,  # Mask token
                "stats": siem_stats.get(config.name, {})
            }
            for config in siem_configs.values()
        ],
        "count": len(siem_configs)
    }


@router.delete("/config/{name}")
async def delete_siem_config(name: str) -> Dict:
    """Delete SIEM configuration"""
    if name in siem_configs:
        del siem_configs[name]
        if name in siem_stats:
            del siem_stats[name]
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Config not found")


@router.post("/send")
async def send_alert(alert: Alert, background_tasks: BackgroundTasks) -> Dict:
    """
    Send alert to all configured SIEMs
    
    Sends asynchronously in background
    """
    sent_to = []
    
    for name, config in siem_configs.items():
        if config.enabled:
            background_tasks.add_task(send_alert_to_siem, alert, config)
            sent_to.append(name)
    
    return {
        "queued": True,
        "sent_to": sent_to,
        "alert_id": alert.id
    }


@router.post("/test/{name}")
async def test_siem_connection(name: str) -> Dict:
    """Test SIEM connection with a test alert"""
    if name not in siem_configs:
        raise HTTPException(status_code=404, detail="Config not found")
    
    config = siem_configs[name]
    
    test_alert = Alert(
        id="TEST-001",
        timestamp=datetime.now().isoformat(),
        severity="low",
        title="PCDS Test Alert",
        description="This is a test alert from PCDS to verify SIEM connectivity",
        source_ip="192.168.1.100",
        dest_ip="10.0.0.1"
    )
    
    success = await send_alert_to_siem(test_alert, config)
    
    return {
        "success": success,
        "config_name": name,
        "transport": config.transport,
        "format": config.format
    }


@router.get("/stats")
async def get_siem_stats() -> Dict:
    """Get SIEM export statistics"""
    total_sent = sum(s.get("sent", 0) for s in siem_stats.values())
    total_failed = sum(s.get("failed", 0) for s in siem_stats.values())
    
    return {
        "total_sent": total_sent,
        "total_failed": total_failed,
        "success_rate": total_sent / max(total_sent + total_failed, 1),
        "destinations": len(siem_configs),
        "enabled": len([c for c in siem_configs.values() if c.enabled]),
        "per_destination": siem_stats
    }


@router.get("/formats")
async def get_supported_formats() -> Dict:
    """Get supported SIEM formats and transports"""
    return {
        "formats": [
            {"id": "syslog", "name": "Syslog (RFC 5424)", "description": "Standard syslog format"},
            {"id": "cef", "name": "CEF", "description": "ArcSight Common Event Format"},
            {"id": "leef", "name": "LEEF", "description": "QRadar Log Event Extended Format"},
            {"id": "json", "name": "JSON", "description": "Generic JSON format"}
        ],
        "transports": [
            {"id": "syslog_udp", "name": "Syslog UDP", "port": 514},
            {"id": "syslog_tcp", "name": "Syslog TCP", "port": 514},
            {"id": "webhook", "name": "Webhook", "description": "HTTP POST"},
            {"id": "splunk_hec", "name": "Splunk HEC", "port": 8088}
        ],
        "severities": ["low", "medium", "high", "critical"]
    }


# Pre-configure some example SIEMs for demo
def init_demo_configs():
    """Initialize demo SIEM configurations"""
    demo_configs = [
        SIEMConfig(
            name="local_syslog",
            enabled=False,
            transport=SIEMTransport.SYSLOG_UDP,
            format=SIEMFormat.SYSLOG,
            host="127.0.0.1",
            port=514,
            min_severity="medium"
        ),
        SIEMConfig(
            name="splunk_demo",
            enabled=False,
            transport=SIEMTransport.SPLUNK_HEC,
            format=SIEMFormat.JSON,
            host="splunk.local",
            port=8088,
            token="demo-token",
            min_severity="high"
        ),
    ]
    
    for config in demo_configs:
        siem_configs[config.name] = config
        siem_stats[config.name] = {"sent": 0, "failed": 0, "last_sent": None}


init_demo_configs()
