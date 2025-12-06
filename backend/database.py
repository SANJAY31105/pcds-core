"""
Database configuration and connection management
"""
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, DateTime, Float, Integer, Boolean, JSON
import os
from datetime import datetime

# Database URL from environment or default
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://pcds_user:pcds_password@localhost:5432/pcds_db"
)

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    pool_size=10,
    max_overflow=20,
)

# Create async session factory
AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Base class for ORM models
Base = declarative_base()


class ThreatRecord(Base):
    """Database model for threat records"""
    __tablename__ = "threats"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    severity = Column(String, index=True)
    category = Column(String, index=True)
    title = Column(String)
    description = Column(String)
    source_ip = Column(String, index=True)
    destination_ip = Column(String)
    risk_score = Column(Float)
    confidence = Column(Float)
    indicators = Column(JSON)
    affected_systems = Column(JSON)
    resolved = Column(Boolean, default=False)
    

class AlertRecord(Base):
    """Database model for alerts"""
    __tablename__ = "alerts"
    
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    severity = Column(String)
    message = Column(String)
    threat_id = Column(String)
    acknowledged = Column(Boolean, default=False)
    

class MetricsRecord(Base):
    """Database model for system metrics"""
    __tablename__ = "metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    network_throughput = Column(Float)
    active_connections = Column(Integer)
    threats_detected_today = Column(Integer)
    threats_blocked_today = Column(Integer)


async def get_db():
    """Dependency for getting database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db():
    """Initialize database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
