"""
PCDS Enterprise - Production-Grade In-Memory Cache
Fallback caching system when Redis is unavailable
Thread-safe, TTL-based, enterprise-ready
"""

import time
import hashlib
import json
from typing import Any, Optional, Callable
from threading import Lock
from datetime import datetime, timedelta
from functools import wraps

class InMemoryCache:
    """
    Thread-safe in-memory cache with TTL support
    Used as fallback when Redis is unavailable
    """
    
    def __init__(self):
        self._cache = {}
        self._lock = Lock()
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function signature"""
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if time.time() < expiry:
                    self._hits += 1
                    return value
                else:
                    # Expired, remove it
                    del self._cache[key]
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: int = 60):
        """Set value in cache with TTL (seconds)"""
        with self._lock:
            expiry = time.time() + ttl
            self._cache[key] = (value, expiry)
    
    def delete(self, key: str):
        """Delete key from cache"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
    
    def clear(self):
        """Clear all cache"""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def stats(self) -> dict:
        """Get cache statistics"""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2),
            "cache_size": len(self._cache)
        }
    
    def cleanup_expired(self):
        """Remove expired entries"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if current_time >= expiry
            ]
            for key in expired_keys:
                del self._cache[key]
    
    def cache_function(self, ttl: int = 60):
        """
        Decorator to cache function results
        
        Usage:
            @memory_cache.cache_function(ttl=120)
            async def get_dashboard_data():
                return expensive_computation()
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self._generate_key(func.__name__, args, kwargs)
                
                # Try to get from cache
                cached_value = self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Cache miss, execute function
                result = await func(*args, **kwargs)
                
                # Store in cache
                self.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator


# Global cache instance
memory_cache = InMemoryCache()


# Utility function for manual caching
def cache_with_ttl(key: str, ttl: int = 60):
    """
    Decorator for manual cache key specification
    
    Usage:
        @cache_with_ttl("dashboard:overview", ttl=60)
        async def get_overview():
            return data
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cached_value = memory_cache.get(key)
            if cached_value is not None:
                return cached_value
            
            result = await func(*args, **kwargs)
            memory_cache.set(key, result, ttl)
            return result
        return wrapper
    return decorator
