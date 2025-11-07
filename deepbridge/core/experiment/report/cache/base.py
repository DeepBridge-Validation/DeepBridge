"""
Base cache interface (Phase 3 Sprint 17-18).

Defines the contract for all cache implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass


@dataclass
class CacheEntry:
    """
    Cache entry with value and metadata.

    Attributes:
        value: The cached value
        created_at: When entry was created
        ttl: Time to live in seconds (None = no expiration)
        hits: Number of times this entry was accessed
        size_bytes: Approximate size in bytes (optional)
    """
    value: Any
    created_at: datetime
    ttl: Optional[int] = None
    hits: int = 0
    size_bytes: Optional[int] = None

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl

    @property
    def age_seconds(self) -> float:
        """Get age of entry in seconds."""
        return (datetime.now() - self.created_at).total_seconds()


class CacheStrategy(ABC):
    """
    Abstract base class for cache implementations.

    Implementations must provide:
    - get(key) - Retrieve value
    - set(key, value, ttl) - Store value
    - delete(key) - Remove value
    - clear() - Clear all entries
    - stats() - Get cache statistics

    Example:
        class MyCache(CacheStrategy):
            def get(self, key: str) -> Optional[Any]:
                # Implementation
                pass

            def set(self, key: str, value: Any, ttl: Optional[int] = None):
                # Implementation
                pass
    """

    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired

        Example:
            >>> value = cache.get('chart:coverage:abc123')
            >>> if value is None:
            ...     value = generate_value()
            ...     cache.set('chart:coverage:abc123', value)
        """
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """
        Store value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None = use default)

        Example:
            >>> cache.set('chart:coverage:abc123', chart_data, ttl=3600)
        """
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if key was found and deleted, False otherwise

        Example:
            >>> cache.delete('chart:coverage:abc123')
        """
        pass

    @abstractmethod
    def clear(self):
        """
        Clear all entries from cache.

        Example:
            >>> cache.clear()  # Remove everything
        """
        pass

    @abstractmethod
    def stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with statistics:
            - size: Number of entries
            - hits: Total cache hits
            - misses: Total cache misses
            - hit_rate: Hit rate percentage
            - memory_bytes: Approximate memory usage

        Example:
            >>> stats = cache.stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.1f}%")
        """
        pass

    def has(self, key: str) -> bool:
        """
        Check if key exists in cache (and is not expired).

        Args:
            key: Cache key

        Returns:
            True if key exists and is valid

        Example:
            >>> if cache.has('chart:coverage:abc123'):
            ...     print("Chart is cached")
        """
        return self.get(key) is not None

    def get_or_set(self, key: str, factory, ttl: Optional[int] = None) -> Any:
        """
        Get value from cache or compute and cache it.

        Args:
            key: Cache key
            factory: Callable to generate value if not cached
            ttl: Time to live in seconds

        Returns:
            Cached or freshly computed value

        Example:
            >>> def generate_chart():
            ...     return expensive_operation()
            >>>
            >>> chart = cache.get_or_set('chart:coverage:abc123', generate_chart, ttl=3600)
        """
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl)
        return value
