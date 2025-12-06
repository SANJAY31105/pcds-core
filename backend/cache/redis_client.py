"""
Redis Cache Client for PCDS Enterprise
Provides caching, session management, and rate limiting
"""
from redis import Redis
from redis.asyncio import Redis as AsyncRedis
import json
from functools import wraps
from typing import Any, Optional, Callable
import hashlib
import logging

logger = logging.getLogger(__name__)


class CacheClient:
    """Redis-based caching client with async support"""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0):
        self.redis = Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_keepalive=True,
            socket_connect_timeout=5
        )
        self.async_redis = None
        logger.info(f"Redis cache client initialized: {host}:{port}")
    
    async def get_async_redis(self):
        """Get async Redis connection"""
        if self.async_redis is None:
            self.async_redis = await AsyncRedis(
                host='localhost',
                port=6379,
                decode_responses=True
            )
        return self.async_redis
    
    def _make_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from function args"""
        key_parts = [str(arg) for arg in args]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        key_str = ":".join(key_parts)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:8]
        return f"{prefix}:{key_hash}"
    
    def cache(self, ttl: int = 300, prefix: str = None):
        """
        Decorator for caching function results
        
        Args:
            ttl: Time to live in seconds (default 5 minutes)
            prefix: Cache key prefix (defaults to function name)
        """
        def decorator(func: Callable):
            cache_prefix = prefix or func.__name__
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._make_key(cache_prefix, *args, **kwargs)
                
                try:
                    # Try to get from cache
                    cached = self.redis.get(cache_key)
                    if cached:
                        logger.debug(f"Cache HIT: {cache_key}")
                        return json.loads(cached)
                except Exception as e:
                    logger.warning(f"Cache get error: {e}")
                
                # Execute function
                logger.debug(f"Cache MISS: {cache_key}")
                result = await func(*args, **kwargs)
                
                try:
                    # Store in cache
                    self.redis.setex(
                        cache_key,
                        ttl,
                        json.dumps(result, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Cache set error: {e}")
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._make_key(cache_prefix, *args, **kwargs)
                
                try:
                    # Try to get from cache
                    cached = self.redis.get(cache_key)
                    if cached:
                        logger.debug(f"Cache HIT: {cache_key}")
                        return json.loads(cached)
                except Exception as e:
                    logger.warning(f"Cache get error: {e}")
                
                # Execute function
                logger.debug(f"Cache MISS: {cache_key}")
                result = func(*args, **kwargs)
                
                try:
                    # Store in cache
                    self.redis.setex(
                        cache_key,
                        ttl,
                        json.dumps(result, default=str)
                    )
                except Exception as e:
                    logger.warning(f"Cache set error: {e}")
                
                return result
            
            # Return appropriate wrapper based on function type
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def invalidate(self, pattern: str):
        """Invalidate cache keys matching pattern"""
        try:
            keys = self.redis.keys(f"{pattern}*")
            if keys:
                self.redis.delete(*keys)
                logger.info(f"Invalidated {len(keys)} cache keys: {pattern}")
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        try:
            value = self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: int = 300):
        """Set value in cache"""
        try:
            self.redis.setex(key, ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def delete(self, key: str):
        """Delete key from cache"""
        try:
            self.redis.delete(key)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
    
    def ping(self) -> bool:
        """Check Redis connection"""
        try:
            return self.redis.ping()
        except Exception:
            return False


# Global cache client instance
cache_client = CacheClient()
