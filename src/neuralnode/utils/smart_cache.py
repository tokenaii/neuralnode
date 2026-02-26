"""Advanced caching with smart invalidation and monitoring."""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading

from ..core.types import LLMResponse, Message


@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata."""
    key: str
    response: LLMResponse
    timestamp: float
    ttl: Optional[float] = None
    tags: Set[str] = None
    access_count: int = 0
    last_accessed: float = 0
    dependencies: Set[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
        if self.dependencies is None:
            self.dependencies = set()
        if self.last_accessed == 0:
            self.last_accessed = self.timestamp
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def touch(self) -> None:
        """Update access metrics."""
        self.access_count += 1
        self.last_accessed = time.time()


class SmartCache:
    """Smart cache with invalidation rules and monitoring."""
    
    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._lock = threading.RLock()
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> set of keys
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
        }
    
    def get(
        self,
        key: str,
        check_dependencies: bool = True,
    ) -> Optional[LLMResponse]:
        """Get from cache with smart checks."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            # Check TTL
            if entry.is_expired():
                self._remove_entry(key)
                self._stats["misses"] += 1
                return None
            
            # Check dependencies if requested
            if check_dependencies and entry.dependencies:
                for dep_key in entry.dependencies:
                    if dep_key not in self._cache:
                        # Dependency invalidated, invalidate this entry too
                        self._remove_entry(key)
                        self._stats["misses"] += 1
                        return None
            
            # Cache hit
            entry.touch()
            self._stats["hits"] += 1
            return entry.response
    
    def set(
        self,
        key: str,
        response: LLMResponse,
        ttl: Optional[float] = None,
        tags: Optional[Set[str]] = None,
        dependencies: Optional[Set[str]] = None,
    ) -> None:
        """Set cache entry with smart features."""
        with self._lock:
            ttl = ttl or self._default_ttl
            
            # Evict if needed
            if len(self._cache) >= self._max_size:
                self._evict_lru()
            
            entry = CacheEntry(
                key=key,
                response=response,
                timestamp=time.time(),
                ttl=ttl,
                tags=tags or set(),
                dependencies=dependencies or set(),
            )
            
            self._cache[key] = entry
            
            # Update tag index
            for tag in entry.tags:
                if tag not in self._tag_index:
                    self._tag_index[tag] = set()
                self._tag_index[tag].add(key)
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a specific entry."""
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                self._stats["invalidations"] += 1
                return True
            return False
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate all entries with a tag."""
        with self._lock:
            keys = self._tag_index.get(tag, set()).copy()
            for key in keys:
                self._remove_entry(key)
            self._stats["invalidations"] += len(keys)
            return len(keys)
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate keys matching pattern."""
        import re
        with self._lock:
            keys_to_remove = [
                key for key in self._cache.keys()
                if re.search(pattern, key)
            ]
            for key in keys_to_remove:
                self._remove_entry(key)
            self._stats["invalidations"] += len(keys_to_remove)
            return len(keys_to_remove)
    
    def invalidate_all(self) -> None:
        """Clear all cache."""
        with self._lock:
            self._cache.clear()
            self._tag_index.clear()
            self._stats["invalidations"] += 1
    
    def _remove_entry(self, key: str) -> None:
        """Remove entry and update indexes."""
        if key not in self._cache:
            return
        
        entry = self._cache[key]
        
        # Remove from tag index
        for tag in entry.tags:
            if tag in self._tag_index:
                self._tag_index[tag].discard(key)
                if not self._tag_index[tag]:
                    del self._tag_index[tag]
        
        del self._cache[key]
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return
        
        # Find oldest by last_accessed
        lru_key = min(self._cache.keys(), key=lambda k: self._cache[k].last_accessed)
        self._remove_entry(lru_key)
        self._stats["evictions"] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = len(self._cache)
            total_accesses = self._stats["hits"] + self._stats["misses"]
            hit_rate = self._stats["hits"] / total_accesses if total_accesses > 0 else 0
            
            # Calculate average entry age
            now = time.time()
            ages = [now - e.timestamp for e in self._cache.values()]
            avg_age = sum(ages) / len(ages) if ages else 0
            
            return {
                "total_entries": total,
                "max_size": self._max_size,
                "hit_rate": round(hit_rate, 4),
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "invalidations": self._stats["invalidations"],
                "avg_entry_age_seconds": round(avg_age, 2),
                "tags": len(self._tag_index),
            }
    
    def get_entry_info(self, key: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a cache entry."""
        with self._lock:
            entry = self._cache.get(key)
            if not entry:
                return None
            
            now = time.time()
            return {
                "key": key,
                "age_seconds": round(now - entry.timestamp, 2),
                "access_count": entry.access_count,
                "last_accessed_seconds_ago": round(now - entry.last_accessed, 2),
                "tags": list(entry.tags),
                "dependencies": list(entry.dependencies),
                "is_expired": entry.is_expired(),
            }
    
    def add_dependency(self, key: str, dependency_key: str) -> bool:
        """Add a dependency to an entry."""
        with self._lock:
            if key in self._cache:
                self._cache[key].dependencies.add(dependency_key)
                return True
            return False


class CacheWarmer:
    """Pre-warm cache with common queries."""
    
    def __init__(self, cache: SmartCache, llm_func: Callable):
        self.cache = cache
        self.llm_func = llm_func
    
    def warm(self, common_queries: List[str], tags: Optional[Set[str]] = None) -> None:
        """Pre-populate cache with common queries."""
        for query in common_queries:
            key = self._generate_key(query)
            
            # Check if already cached
            if self.cache.get(key) is not None:
                continue
            
            # Generate and cache
            try:
                response = self.llm_func(query)
                self.cache.set(key, response, tags=tags)
            except Exception as e:
                print(f"Failed to warm cache for query '{query}': {e}")
    
    def _generate_key(self, query: str) -> str:
        """Generate cache key."""
        return hashlib.sha256(query.encode()).hexdigest()


class CacheDecorator:
    """Decorator factory for caching."""
    
    def __init__(self, cache: SmartCache, ttl: Optional[float] = None):
        self.cache = cache
        self.ttl = ttl
    
    def __call__(self, func: Callable) -> Callable:
        """Apply caching decorator."""
        def wrapper(*args, **kwargs):
            # Generate key from function and arguments
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": kwargs,
            }
            key = hashlib.sha256(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Try cache
            cached = self.cache.get(key)
            if cached is not None:
                return cached
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache if LLMResponse
            if isinstance(result, LLMResponse):
                self.cache.set(key, result, ttl=self.ttl)
            
            return result
        
        return wrapper


# Global smart cache instance
_global_smart_cache = SmartCache()


def get_global_cache() -> SmartCache:
    """Get global smart cache instance."""
    return _global_smart_cache


def cached(
    ttl: Optional[float] = None,
    tags: Optional[Set[str]] = None,
    dependencies: Optional[Set[str]] = None,
):
    """Decorator for smart caching."""
    cache = _global_smart_cache
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate key
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": kwargs,
            }
            key = hashlib.sha256(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Try cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Call function
            result = func(*args, **kwargs)
            
            # Cache result
            if isinstance(result, LLMResponse):
                cache.set(key, result, ttl=ttl, tags=tags, dependencies=dependencies)
            
            return result
        
        return wrapper
    return decorator
