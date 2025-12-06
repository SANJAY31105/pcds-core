"""
PCDS Enterprise - Kafka Configuration
Enterprise-grade event streaming settings
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TopicConfig:
    """Kafka topic configuration"""
    name: str
    partitions: int = 6
    replication_factor: int = 1
    retention_ms: int = 604800000  # 7 days
    cleanup_policy: str = "delete"


# Topic Definitions
TOPICS = {
    "raw_events": TopicConfig(
        name="pcds.raw-events",
        partitions=12,
        retention_ms=604800000,  # 7 days
    ),
    "detections": TopicConfig(
        name="pcds.detections",
        partitions=6,
        retention_ms=2592000000,  # 30 days
    ),
    "alerts": TopicConfig(
        name="pcds.alerts",
        partitions=3,
        retention_ms=7776000000,  # 90 days
    ),
    "dead_letter": TopicConfig(
        name="pcds.dead-letter",
        partitions=3,
        retention_ms=1209600000,  # 14 days
    ),
    "audit": TopicConfig(
        name="pcds.audit",
        partitions=3,
        retention_ms=31536000000,  # 365 days
        cleanup_policy="compact",
    ),
}


@dataclass
class KafkaSettings:
    """Kafka connection settings"""
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "pcds-enterprise"
    group_id: str = "pcds-consumers"
    
    # Producer settings
    acks: str = "all"
    retries: int = 3
    batch_size: int = 16384
    linger_ms: int = 5
    compression_type: str = "gzip"
    
    # Consumer settings
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    session_timeout_ms: int = 30000
    
    # Security (optional)
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: Optional[str] = None
    sasl_username: Optional[str] = None
    sasl_password: Optional[str] = None
    
    def to_producer_config(self) -> Dict:
        """Get producer configuration"""
        config = {
            "bootstrap.servers": self.bootstrap_servers,
            "client.id": f"{self.client_id}-producer",
            "acks": self.acks,
            "retries": self.retries,
            "batch.size": self.batch_size,
            "linger.ms": self.linger_ms,
            "compression.type": self.compression_type,
        }
        return config
    
    def to_consumer_config(self, group_suffix: str = "") -> Dict:
        """Get consumer configuration"""
        group = f"{self.group_id}-{group_suffix}" if group_suffix else self.group_id
        config = {
            "bootstrap.servers": self.bootstrap_servers,
            "group.id": group,
            "client.id": f"{self.client_id}-consumer",
            "auto.offset.reset": self.auto_offset_reset,
            "enable.auto.commit": self.enable_auto_commit,
            "max.poll.records": self.max_poll_records,
            "session.timeout.ms": self.session_timeout_ms,
        }
        return config


# Global settings instance
kafka_settings = KafkaSettings()
