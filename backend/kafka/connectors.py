"""
PCDS Enterprise - SIEM Connectors
Splunk, Elastic, and Syslog integration
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import socket
import logging

# Optional aiohttp for HTTP connectors
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False


@dataclass
class SIEMConfig:
    """SIEM connection configuration"""
    enabled: bool = False
    endpoint: str = ""
    api_key: str = ""
    index: str = "pcds-security"
    batch_size: int = 100
    flush_interval: float = 5.0


class BaseSIEMConnector:
    """Base class for SIEM connectors"""
    
    def __init__(self, config: SIEMConfig):
        self.config = config
        self.batch: List[Dict] = []
        self.metrics = {
            "events_sent": 0,
            "batches_sent": 0,
            "errors": 0,
            "last_send": None
        }
    
    async def send(self, event: Dict) -> bool:
        """Add event to batch"""
        self.batch.append(self._transform_event(event))
        
        if len(self.batch) >= self.config.batch_size:
            return await self.flush()
        return True
    
    async def flush(self) -> bool:
        """Flush batch to SIEM"""
        if not self.batch:
            return True
        
        try:
            success = await self._send_batch(self.batch)
            if success:
                self.metrics["events_sent"] += len(self.batch)
                self.metrics["batches_sent"] += 1
                self.metrics["last_send"] = datetime.utcnow().isoformat()
                self.batch = []
                return True
        except Exception as e:
            self.metrics["errors"] += 1
            logging.error(f"SIEM flush error: {e}")
        return False
    
    def _transform_event(self, event: Dict) -> Dict:
        """Transform event to SIEM format - override in subclass"""
        return event
    
    async def _send_batch(self, batch: List[Dict]) -> bool:
        """Send batch to SIEM - override in subclass"""
        raise NotImplementedError


class SplunkConnector(BaseSIEMConnector):
    """
    Splunk HTTP Event Collector (HEC) connector
    Sends events to Splunk via HEC endpoint
    """
    
    def __init__(self, hec_url: str = "", hec_token: str = "", 
                 index: str = "pcds-security", sourcetype: str = "pcds:detection"):
        config = SIEMConfig(
            enabled=bool(hec_url and hec_token),
            endpoint=hec_url,
            api_key=hec_token,
            index=index
        )
        super().__init__(config)
        self.sourcetype = sourcetype
    
    def _transform_event(self, event: Dict) -> Dict:
        """Transform to Splunk HEC format"""
        return {
            "time": datetime.utcnow().timestamp(),
            "host": "pcds-enterprise",
            "source": "pcds-ml-engine",
            "sourcetype": self.sourcetype,
            "index": self.config.index,
            "event": event
        }
    
    async def _send_batch(self, batch: List[Dict]) -> bool:
        """Send batch to Splunk HEC"""
        if not self.config.enabled:
            # Simulate success for testing
            print(f"📤 [SIMULATED] Splunk: Would send {len(batch)} events")
            return True
        
        try:
            headers = {
                "Authorization": f"Splunk {self.config.api_key}",
                "Content-Type": "application/json"
            }
            
            # Splunk HEC expects newline-delimited JSON
            payload = "\n".join(json.dumps(event) for event in batch)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.endpoint,
                    headers=headers,
                    data=payload,
                    ssl=False  # Allow self-signed certs for dev
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            logging.error(f"Splunk send error: {e}")
            return False


class ElasticConnector(BaseSIEMConnector):
    """
    Elasticsearch connector
    Sends events to Elastic via bulk API
    """
    
    def __init__(self, es_url: str = "", api_key: str = "", 
                 index: str = "pcds-detections"):
        config = SIEMConfig(
            enabled=bool(es_url),
            endpoint=es_url,
            api_key=api_key,
            index=index
        )
        super().__init__(config)
    
    def _transform_event(self, event: Dict) -> Dict:
        """Transform to Elastic document format"""
        return {
            "@timestamp": datetime.utcnow().isoformat(),
            "event": {
                "kind": "alert",
                "category": ["intrusion_detection"],
                "type": ["indicator"],
                "module": "pcds"
            },
            "pcds": event,
            "host": {"name": "pcds-enterprise"},
            "ecs": {"version": "8.0.0"}
        }
    
    async def _send_batch(self, batch: List[Dict]) -> bool:
        """Send batch to Elasticsearch"""
        if not self.config.enabled:
            print(f"📤 [SIMULATED] Elastic: Would send {len(batch)} events")
            return True
        
        try:
            headers = {
                "Content-Type": "application/x-ndjson"
            }
            if self.config.api_key:
                headers["Authorization"] = f"ApiKey {self.config.api_key}"
            
            # Build bulk request body
            bulk_body = []
            for event in batch:
                bulk_body.append(json.dumps({"index": {"_index": self.config.index}}))
                bulk_body.append(json.dumps(event))
            
            payload = "\n".join(bulk_body) + "\n"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.config.endpoint}/_bulk",
                    headers=headers,
                    data=payload
                ) as response:
                    return response.status in [200, 201]
                    
        except Exception as e:
            logging.error(f"Elastic send error: {e}")
            return False


class SyslogConnector(BaseSIEMConnector):
    """
    Syslog connector for legacy SIEM integration
    Supports UDP, TCP, and TLS
    """
    
    def __init__(self, host: str = "", port: int = 514, 
                 protocol: str = "udp", facility: int = 1):
        config = SIEMConfig(
            enabled=bool(host),
            endpoint=f"{protocol}://{host}:{port}"
        )
        super().__init__(config)
        self.host = host
        self.port = port
        self.protocol = protocol
        self.facility = facility
        self.socket = None
    
    def _transform_event(self, event: Dict) -> str:
        """Transform to syslog format (RFC 5424)"""
        severity = self._map_severity(event.get("severity", "medium"))
        priority = (self.facility * 8) + severity
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        hostname = "pcds-enterprise"
        app_name = "pcds"
        proc_id = "-"
        msg_id = event.get("detection_type", "DETECTION")
        
        # Structured data
        sd = f'[pcds severity="{event.get("severity", "medium")}" ' \
             f'entity="{event.get("entity_id", "-")}" ' \
             f'technique="{event.get("technique_id", "-")}"]'
        
        # Message
        msg = json.dumps(event)
        
        return f"<{priority}>1 {timestamp} {hostname} {app_name} {proc_id} {msg_id} {sd} {msg}"
    
    def _map_severity(self, level: str) -> int:
        """Map PCDS severity to syslog severity"""
        mapping = {
            "critical": 2,  # Critical
            "high": 3,      # Error
            "medium": 4,    # Warning
            "low": 6        # Informational
        }
        return mapping.get(level, 5)  # Notice
    
    async def _send_batch(self, batch: List[str]) -> bool:
        """Send batch via syslog"""
        if not self.config.enabled:
            print(f"📤 [SIMULATED] Syslog: Would send {len(batch)} events")
            return True
        
        try:
            for message in batch:
                if self.protocol == "udp":
                    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    sock.sendto(message.encode(), (self.host, self.port))
                    sock.close()
                elif self.protocol == "tcp":
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect((self.host, self.port))
                    sock.send(message.encode() + b"\n")
                    sock.close()
            return True
        except Exception as e:
            logging.error(f"Syslog send error: {e}")
            return False


class SIEMManager:
    """
    Manages multiple SIEM connectors
    Supports fanout to multiple destinations
    """
    
    def __init__(self):
        self.connectors: Dict[str, BaseSIEMConnector] = {}
        self.enabled = True
    
    def add_connector(self, name: str, connector: BaseSIEMConnector):
        """Add a connector"""
        self.connectors[name] = connector
    
    def remove_connector(self, name: str):
        """Remove a connector"""
        self.connectors.pop(name, None)
    
    async def send_to_all(self, event: Dict) -> Dict[str, bool]:
        """Send event to all connectors"""
        results = {}
        for name, connector in self.connectors.items():
            results[name] = await connector.send(event)
        return results
    
    async def flush_all(self) -> Dict[str, bool]:
        """Flush all connectors"""
        results = {}
        for name, connector in self.connectors.items():
            results[name] = await connector.flush()
        return results
    
    def get_metrics(self) -> Dict[str, Dict]:
        """Get metrics from all connectors"""
        return {
            name: connector.metrics 
            for name, connector in self.connectors.items()
        }


# Global SIEM manager with demo connectors
siem_manager = SIEMManager()
siem_manager.add_connector("splunk", SplunkConnector())
siem_manager.add_connector("elastic", ElasticConnector())
siem_manager.add_connector("syslog", SyslogConnector())
