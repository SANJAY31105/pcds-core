"""
PCDS Enterprise Settings Management
Environment-based configuration with validation
"""

from pydantic_settings import BaseSettings
from typing import Optional
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "PCDS Enterprise NDR"
    APP_VERSION: str = "2.0.0"
    API_V2_PREFIX: str = "/api/v2"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # Server
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    WORKERS: int = 4
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./pcds_enterprise.db")
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10
    DATABASE_ECHO: bool = False  # Set to True for SQL query logging
    
    # Redis (for caching and real-time features)
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = False  # Enable when Redis is available
    
    # Security - Load from environment variables (CRITICAL!)
    SECRET_KEY: str = os.getenv("SECRET_KEY", "dev-secret-key-change-immediately")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # CORS - Production ready (no "null" origin)
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list = ["*"]
    CORS_ALLOW_HEADERS: list = ["*"]
    
    # Scoring Engine
    URGENCY_RECALC_INTERVAL_SECONDS: int = 60  # How often to recalculate entity scores
    MAX_DETECTIONS_PER_ENTITY: int = 1000  # Keep last N detections per entity
    
    # Detection Engine
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5  # Minimum confidence to create detection
    AUTO_CREATE_CAMPAIGNS: bool = True  # Automatically correlate into campaigns
    CAMPAIGN_TIME_WINDOW_HOURS: int = 24  # Max time between detections in same campaign
    
    # Threat Hunting
    HUNT_RESULT_TTL_DAYS: int = 30  # Keep hunt results for N days
    MAX_HUNT_RESULTS: int = 1000  # Maximum findings per hunt
    
    # ML Model
    ML_MODEL_PATH: Optional[str] = None  # Path to trained PyTorch model
    ML_ENABLED: bool = False  # Enable ML predictions
    
    # Logging
    LOG_LEVEL: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    MITRE_DATA_FILE: Path = DATA_DIR / "mitre_attack_full.json"
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30  # Seconds between heartbeats
    WS_MESSAGE_QUEUE_SIZE: int = 100
    
    # Investigation
    EVIDENCE_UPLOAD_DIR: Path = BASE_DIR / "uploads" / "evidence"
    MAX_EVIDENCE_SIZE_MB: int = 100
    ALLOWED_EVIDENCE_TYPES: list = [
        "application/pdf",
        "image/png", 
        "image/jpeg",
        "text/plain",
        "application/json"
    ]
    
    # Performance
    ENTITY_CACHE_TTL_SECONDS: int = 300  # Cache entity data for 5 minutes
    DETECTION_BATCH_SIZE: int = 100  # Process detections in batches
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True


# Global settings instance
settings = Settings()


# Ensure required directories exist
settings.EVIDENCE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
