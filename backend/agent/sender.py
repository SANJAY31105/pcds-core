"""
PCDS Cloud Sender
Securely sends captured data to PCDS Cloud API
"""

import logging
import threading
import queue
import json
import time
import gzip
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import hmac

try:
    import aiohttp
    import asyncio
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from .features import FlowFeatures
except ImportError:
    from features import FlowFeatures

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Agent configuration"""
    agent_id: str
    api_key: str
    api_endpoint: str = "https://api.pcds.app/v1"
    organization_id: str = ""
    batch_size: int = 100
    send_interval: float = 5.0  # seconds
    compress: bool = True
    retry_attempts: int = 3
    timeout: float = 30.0


class CloudSender:
    """
    Securely sends flow data to PCDS Cloud
    Features:
    - Batching for efficiency
    - Compression for bandwidth
    - Retry logic for reliability
    - Async sending for performance
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        
        # Queue for pending data
        self.send_queue: queue.Queue = queue.Queue(maxsize=10000)
        
        # Statistics
        self.stats = {
            "batches_sent": 0,
            "batches_failed": 0,
            "flows_sent": 0,
            "bytes_sent": 0,
            "last_send_time": None,
            "errors": []
        }
        
        # Control
        self._running = False
        self._sender_thread: Optional[threading.Thread] = None
        
        logger.info(f"CloudSender initialized for agent {config.agent_id}")
    
    def _sign_request(self, payload: bytes, timestamp: str) -> str:
        """Generate HMAC signature for request authentication"""
        message = f"{timestamp}:{payload.decode()}"
        signature = hmac.new(
            self.config.api_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _compress_payload(self, data: bytes) -> bytes:
        """Compress payload using gzip"""
        return gzip.compress(data)
    
    def _prepare_batch(self, features: List[FlowFeatures]) -> Dict[str, Any]:
        """Prepare batch payload for API"""
        return {
            "agent_id": self.config.agent_id,
            "organization_id": self.config.organization_id,
            "timestamp": datetime.utcnow().isoformat(),
            "batch_size": len(features),
            "flows": [f.to_dict() for f in features]
        }
    
    def _send_batch_sync(self, batch: Dict[str, Any]) -> bool:
        """Send batch synchronously using requests"""
        if not REQUESTS_AVAILABLE:
            logger.error("requests library not available")
            return False
        
        try:
            # Prepare payload
            payload = json.dumps(batch).encode()
            
            if self.config.compress:
                payload = self._compress_payload(payload)
            
            # Sign request
            timestamp = datetime.utcnow().isoformat()
            signature = self._sign_request(payload, timestamp)
            
            # Send
            headers = {
                "Content-Type": "application/json",
                "X-Agent-ID": self.config.agent_id,
                "X-Timestamp": timestamp,
                "X-Signature": signature,
                "Authorization": f"Bearer {self.config.api_key}"
            }
            
            if self.config.compress:
                headers["Content-Encoding"] = "gzip"
            
            response = requests.post(
                f"{self.config.api_endpoint}/ingest/flows",
                data=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                self.stats["batches_sent"] += 1
                self.stats["flows_sent"] += batch["batch_size"]
                self.stats["bytes_sent"] += len(payload)
                self.stats["last_send_time"] = datetime.now().isoformat()
                logger.info(f"Batch sent: {batch['batch_size']} flows")
                return True
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                self.stats["errors"].append({
                    "time": datetime.now().isoformat(),
                    "error": f"HTTP {response.status_code}"
                })
                return False
                
        except requests.Timeout:
            logger.error("Request timeout")
            self.stats["batches_failed"] += 1
            return False
        except Exception as e:
            logger.error(f"Send error: {e}")
            self.stats["batches_failed"] += 1
            self.stats["errors"].append({
                "time": datetime.now().isoformat(),
                "error": str(e)
            })
            return False
    
    def queue_features(self, features: FlowFeatures):
        """Queue flow features for sending"""
        try:
            self.send_queue.put_nowait(features)
        except queue.Full:
            logger.warning("Send queue full, dropping flow")
    
    def queue_batch(self, features: List[FlowFeatures]):
        """Queue multiple flow features"""
        for f in features:
            self.queue_features(f)
    
    def _sender_loop(self):
        """Main sender loop"""
        logger.info("Sender loop started")
        batch: List[FlowFeatures] = []
        last_send = time.time()
        
        while self._running or not self.send_queue.empty():
            try:
                # Collect items for batch
                while len(batch) < self.config.batch_size:
                    try:
                        item = self.send_queue.get(timeout=0.1)
                        batch.append(item)
                    except queue.Empty:
                        break
                
                # Check if we should send
                should_send = (
                    len(batch) >= self.config.batch_size or
                    (len(batch) > 0 and time.time() - last_send >= self.config.send_interval)
                )
                
                if should_send and batch:
                    payload = self._prepare_batch(batch)
                    
                    # Retry logic
                    for attempt in range(self.config.retry_attempts):
                        if self._send_batch_sync(payload):
                            break
                        logger.warning(f"Retry {attempt + 1}/{self.config.retry_attempts}")
                        time.sleep(1.0 * (attempt + 1))
                    
                    batch = []
                    last_send = time.time()
                    
            except Exception as e:
                logger.error(f"Sender loop error: {e}")
                time.sleep(1.0)
        
        # Send remaining
        if batch:
            payload = self._prepare_batch(batch)
            self._send_batch_sync(payload)
        
        logger.info("Sender loop stopped")
    
    def start(self):
        """Start sender thread"""
        if self._running:
            return
        
        self._running = True
        self._sender_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._sender_thread.start()
        logger.info("CloudSender started")
    
    def stop(self):
        """Stop sender thread"""
        logger.info("Stopping CloudSender...")
        self._running = False
        
        if self._sender_thread:
            self._sender_thread.join(timeout=30.0)
        
        logger.info("CloudSender stopped")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get sender statistics"""
        return {
            **self.stats,
            "queue_size": self.send_queue.qsize()
        }


class LocalCache:
    """
    Local cache for offline operation
    Stores flows when cloud is unreachable
    """
    
    def __init__(self, cache_file: str = "pcds_cache.json"):
        self.cache_file = cache_file
        self.cache: List[Dict] = []
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from disk"""
        try:
            with open(self.cache_file, 'r') as f:
                self.cache = json.load(f)
            logger.info(f"Loaded {len(self.cache)} cached items")
        except FileNotFoundError:
            self.cache = []
        except Exception as e:
            logger.error(f"Cache load error: {e}")
            self.cache = []
    
    def save(self, features: FlowFeatures):
        """Save features to cache"""
        self.cache.append(features.to_dict())
        self._flush()
    
    def _flush(self):
        """Write cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Cache flush error: {e}")
    
    def get_all(self) -> List[Dict]:
        """Get all cached items"""
        return self.cache
    
    def clear(self):
        """Clear cache"""
        self.cache = []
        self._flush()
