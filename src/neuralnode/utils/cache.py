"""Caching layer for LLM responses to save costs and improve performance."""

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path

from ..core.types import LLMResponse, Message, StreamingChunk


@dataclass
class CacheEntry:
    """A cache entry for LLM responses."""
    key: str
    response: LLMResponse
    timestamp: float
    ttl: Optional[float] = None  # Time-to-live in seconds
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class BaseCache(ABC):
    """Abstract base class for caches."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[LLMResponse]:
        """Get a cached response."""
        pass
    
    @abstractmethod
    def set(
        self,
        key: str,
        response: LLMResponse,
        ttl: Optional[float] = None,
    ) -> None:
        """Cache a response."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached entries."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete a specific cache entry."""
        pass
    
    def _generate_key(
        self,
        messages: List[Message],
        model: str,
        **kwargs: Any,
    ) -> str:
        """Generate a cache key from request parameters."""
        # Create a stable representation
        key_data = {
            "messages": [m.to_dict() for m in messages],
            "model": model,
            **kwargs,
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()


class InMemoryCache(BaseCache):
    """In-memory cache with optional TTL."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[str, CacheEntry] = {}
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[LLMResponse]:
        """Get a cached response."""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        if entry.is_expired():
            del self._cache[key]
            return None
        
        return entry.response
    
    def set(
        self,
        key: str,
        response: LLMResponse,
        ttl: Optional[float] = None,
    ) -> None:
        """Cache a response."""
        # Evict oldest entries if at capacity
        if len(self._cache) >= self._max_size:
            # Remove expired entries first
            expired = [k for k, v in self._cache.items() if v.is_expired()]
            for k in expired:
                del self._cache[k]
            
            # If still at capacity, remove oldest
            if len(self._cache) >= self._max_size:
                oldest = min(self._cache.keys(), key=lambda k: self._cache[k].timestamp)
                del self._cache[oldest]
        
        entry = CacheEntry(
            key=key,
            response=response,
            timestamp=time.time(),
            ttl=ttl,
        )
        self._cache[key] = entry
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
    
    def delete(self, key: str) -> bool:
        """Delete a specific cache entry."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = len(self._cache)
        expired = sum(1 for e in self._cache.values() if e.is_expired())
        return {
            "total_entries": total,
            "expired_entries": expired,
            "active_entries": total - expired,
            "max_size": self._max_size,
        }


class DiskCache(BaseCache):
    """Disk-based cache for persistence across sessions."""
    
    def __init__(self, cache_dir: Union[str, Path] = ".nn_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.cache_dir / "index.json"
        self._index: Dict[str, Dict[str, Any]] = self._load_index()
    
    def _load_index(self) -> Dict[str, Any]:
        """Load cache index from disk."""
        if self._index_file.exists():
            try:
                with open(self._index_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return {}
        return {}
    
    def _save_index(self) -> None:
        """Save cache index to disk."""
        with open(self._index_file, "w") as f:
            json.dump(self._index, f)
    
    def _get_cache_file(self, key: str) -> Path:
        """Get path to cache file for a key."""
        return self.cache_dir / f"{key}.json"
    
    def get(self, key: str) -> Optional[LLMResponse]:
        """Get a cached response."""
        if key not in self._index:
            return None
        
        entry_meta = self._index[key]
        
        # Check TTL
        if entry_meta.get("ttl"):
            if time.time() - entry_meta["timestamp"] > entry_meta["ttl"]:
                self.delete(key)
                return None
        
        # Load from disk
        cache_file = self._get_cache_file(key)
        if not cache_file.exists():
            del self._index[key]
            self._save_index()
            return None
        
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            
            response = LLMResponse(
                text=data["text"],
                model=data["model"],
                provider=data["provider"],
                usage=data.get("usage"),
                finish_reason=data.get("finish_reason"),
                metadata=data.get("metadata", {}),
            )
            return response
        except (json.JSONDecodeError, KeyError, IOError):
            self.delete(key)
            return None
    
    def set(
        self,
        key: str,
        response: LLMResponse,
        ttl: Optional[float] = None,
    ) -> None:
        """Cache a response."""
        cache_file = self._get_cache_file(key)
        
        data = {
            "text": response.text,
            "model": response.model,
            "provider": response.provider.value if hasattr(response.provider, "value") else str(response.provider),
            "usage": asdict(response.usage) if response.usage else None,
            "finish_reason": response.finish_reason,
            "metadata": response.metadata,
        }
        
        with open(cache_file, "w") as f:
            json.dump(data, f)
        
        self._index[key] = {
            "timestamp": time.time(),
            "ttl": ttl,
            "file": str(cache_file),
        }
        self._save_index()
    
    def clear(self) -> None:
        """Clear all cached entries."""
        for key in list(self._index.keys()):
            self.delete(key)
    
    def delete(self, key: str) -> bool:
        """Delete a specific cache entry."""
        if key not in self._index:
            return False
        
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            cache_file.unlink()
        
        del self._index[key]
        self._save_index()
        return True
    
    def cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        removed = 0
        current_time = time.time()
        
        for key in list(self._index.keys()):
            entry = self._index[key]
            if entry.get("ttl"):
                if current_time - entry["timestamp"] > entry["ttl"]:
                    self.delete(key)
                    removed += 1
        
        return removed
    
    def get_size(self) -> int:
        """Get total cache size in bytes."""
        total_size = 0
        for cache_file in self.cache_dir.glob("*.json"):
            total_size += cache_file.stat().st_size
        return total_size


class SemanticCache(BaseCache):
    """Cache based on semantic similarity of queries."""
    
    def __init__(
        self,
        embedding_function: Any,
        similarity_threshold: float = 0.95,
        max_entries: int = 1000,
    ):
        self.embedding_function = embedding_function
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        self._entries: List[Dict[str, Any]] = []
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self.embedding_function:
            return self.embedding_function(text)
        # Fallback: use simple hash-based embedding
        return [hash(text) % 1000 / 1000.0]
    
    def _similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity."""
        import math
        
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot / (norm_a * norm_b)
    
    def get(self, key: str) -> Optional[LLMResponse]:
        """Get a semantically similar cached response."""
        query_embedding = self._get_embedding(key)
        
        best_match = None
        best_similarity = 0
        
        for entry in self._entries:
            if entry.get("is_expired", lambda: False)():
                continue
            
            similarity = self._similarity(query_embedding, entry["embedding"])
            if similarity > self.similarity_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = entry
        
        if best_match:
            return best_match["response"]
        
        return None
    
    def set(
        self,
        key: str,
        response: LLMResponse,
        ttl: Optional[float] = None,
    ) -> None:
        """Cache a response with semantic embedding."""
        # Evict oldest if at capacity
        if len(self._entries) >= self.max_entries:
            self._entries.pop(0)
        
        embedding = self._get_embedding(key)
        
        entry = {
            "key": key,
            "embedding": embedding,
            "response": response,
            "timestamp": time.time(),
            "ttl": ttl,
            "is_expired": lambda: ttl is not None and time.time() - entry.get("timestamp", 0) > ttl,
        }
        
        self._entries.append(entry)
    
    def clear(self) -> None:
        """Clear all cached entries."""
        self._entries.clear()
    
    def delete(self, key: str) -> bool:
        """Delete a specific cache entry."""
        for i, entry in enumerate(self._entries):
            if entry["key"] == key:
                self._entries.pop(i)
                return True
        return False


# Global cache instances
_default_memory_cache = InMemoryCache()


def get_default_cache() -> BaseCache:
    """Get the default in-memory cache."""
    return _default_memory_cache


def cached(
    cache: Optional[BaseCache] = None,
    ttl: Optional[float] = None,
):
    """
    Decorator to cache LLM function results.
    
    Example:
        @cached(ttl=3600)  # Cache for 1 hour
        def get_completion(prompt):
            return ai.chat(prompt)
    """
    cache_instance = cache or _default_memory_cache
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key from function args
            key_data = {
                "func": func.__name__,
                "args": args,
                "kwargs": kwargs,
            }
            key = hashlib.sha256(
                json.dumps(key_data, sort_keys=True, default=str).encode()
            ).hexdigest()
            
            # Try to get from cache
            cached_response = cache_instance.get(key)
            if cached_response is not None:
                return cached_response
            
            # Call function and cache result
            result = func(*args, **kwargs)
            
            if isinstance(result, LLMResponse):
                cache_instance.set(key, result, ttl)
            
            return result
        
        return wrapper
    return decorator
