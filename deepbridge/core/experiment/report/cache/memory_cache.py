"""
In-memory cache with LRU eviction (Phase 3 Sprint 17-18).

Provides fast in-memory caching with:
- LRU eviction policy
- Configurable max size
- TTL support
- Thread-safe operations
"""

from typing import Any, Optional, Dict
from datetime import datetime
from collections import OrderedDict
import threading
import sys

from .base import CacheStrategy, CacheEntry


class MemoryCache(CacheStrategy):
    """
    In-memory cache with LRU eviction.

    Features:
    - LRU (Least Recently Used) eviction
    - Thread-safe
    - Configurable TTL
    - Memory limits
    - Statistics tracking

    Example:
        >>> cache = MemoryCache(max_size=100, default_ttl=3600)
        >>> cache.set('key', 'value')
        >>> value = cache.get('key')
        >>> stats = cache.stats()
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[int] = None
    ):
        """
        Initialize memory cache.

        Args:
            max_size: Maximum number of entries (LRU eviction when exceeded)
            default_ttl: Default TTL in seconds (None = no expiration)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl

        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Thread-safe. Moves accessed entry to end (most recent).

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check expiration
            if entry.is_expired:
                del self._cache[key]
                self._misses += 1
                return None

            # Update stats and move to end (most recent)
            entry.hits += 1
            self._hits += 1
            self._cache.move_to_end(key)

            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Store value in cache.

        Thread-safe. Evicts LRU entry if max_size exceeded.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = use default)
        """
        with self._lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl

            # Create entry
            entry = CacheEntry(
                value=value,
                created_at=datetime.now(),
                ttl=ttl,
                size_bytes=sys.getsizeof(value),
            )

            # Store entry
            self._cache[key] = entry
            self._cache.move_to_end(key)

            # Evict LRU if needed
            while len(self._cache) > self.max_size:
                # Remove oldest (first) entry
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Thread-safe.

        Args:
            key: Cache key

        Returns:
            True if key was found and deleted
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """
        Clear all entries from cache.

        Thread-safe. Resets statistics.
        """
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Thread-safe.

        Returns:
            Dictionary with statistics:
            - size: Number of entries
            - max_size: Maximum capacity
            - hits: Total cache hits
            - misses: Total cache misses
            - hit_rate: Hit rate percentage
            - memory_bytes: Approximate memory usage
            - entries: List of entry info (key, age, hits)
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0

            # Calculate memory usage
            memory_bytes = sum(
                entry.size_bytes or 0
                for entry in self._cache.values()
            )

            # Collect entry info
            entries = []
            for key, entry in self._cache.items():
                entries.append({
                    'key': key,
                    'age_seconds': entry.age_seconds,
                    'hits': entry.hits,
                    'size_bytes': entry.size_bytes,
                    'ttl': entry.ttl,
                })

            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hits': self._hits,
                'misses': self._misses,
                'hit_rate': hit_rate,
                'memory_bytes': memory_bytes,
                'entries': entries,
            }

    def evict_expired(self) -> int:
        """
        Manually evict all expired entries.

        Thread-safe.

        Returns:
            Number of entries evicted

        Example:
            >>> evicted = cache.evict_expired()
            >>> print(f"Evicted {evicted} expired entries")
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]

            for key in expired_keys:
                del self._cache[key]

            return len(expired_keys)

    def resize(self, new_max_size: int):
        """
        Change maximum cache size.

        Evicts LRU entries if new size is smaller.

        Args:
            new_max_size: New maximum size

        Example:
            >>> cache.resize(50)  # Reduce to 50 entries
        """
        with self._lock:
            self.max_size = new_max_size

            # Evict if needed
            while len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

    def keys(self):
        """
        Get all cache keys.

        Thread-safe.

        Returns:
            List of cache keys

        Example:
            >>> all_keys = cache.keys()
            >>> print(f"Cache contains {len(all_keys)} entries")
        """
        with self._lock:
            return list(self._cache.keys())
