"""
Redis client for caching and pub/sub (optional - graceful fallback)
"""
import json
from typing import Any, Optional


class RedisClient:
    """Redis client wrapper with graceful fallback"""
    
    def __init__(self):
        self.client = None
        self.pubsub = None
        self.enabled = False
        
    async def connect(self):
        """Attempt Redis connection"""
        try:
            import redis.asyncio as redis
            import os
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.client = redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
            await self.client.ping()
            self.enabled = True
            print("✅ Connected to Redis")
        except Exception as e:
            print(f"⚠️ Redis not available (running without caching): {e}")
            self.enabled = False
        
    async def disconnect(self):
        """Close Redis connection"""
        if self.client:
            if self.pubsub:
                await self.pubsub.close()
            await self.client.close()
        
    async def set_cache(self, key: str, value: Any, expiry: int = 3600):
        """Cache a value with expiry (seconds)"""
        if not self.enabled or not self.client:
            return
        try:
            await self.client.setex(key, expiry, json.dumps(value))
        except Exception as e:
            print(f"Redis cache set error: {e}")
        
    async def get_cache(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if not self.enabled or not self.client:
            return None
        try:
            value = await self.client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            print(f"Redis cache get error: {e}")
            return None
        
    async def delete_cache(self, key: str):
        """Delete cached value"""
        if not self.enabled or not self.client:
            return
        try:
            await self.client.delete(key)
        except Exception as e:
            print(f"Redis delete error: {e}")
        
    async def publish(self, channel: str, message: dict):
        """Publish message to channel"""
        if not self.enabled or not self.client:
            return
        try:
            await self.client.publish(channel, json.dumps(message))
        except Exception as e:
            print(f"Redis publish error: {e}")
        
    async def subscribe(self, channel: str):
        """Subscribe to channel"""
        if not self.enabled or not self.client:
            return None
        try:
            if not self.pubsub:
                self.pubsub = self.client.pubsub()
            await self.pubsub.subscribe(channel)
            return self.pubsub
        except Exception as e:
            print(f"Redis subscribe error: {e}")
            return None


# Global Redis client instance
redis_client = RedisClient()
