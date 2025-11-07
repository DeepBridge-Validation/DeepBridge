"""
No-op cache (disabled caching) (Phase 3 Sprint 17-18).

Provides a cache implementation that does nothing.
Useful for disabling caching or testing.
"""

from typing import Any, Optional, Dict
from .base import CacheStrategy


class NoOpCache(CacheStrategy):
    """
    No-op cache implementation.

    Does nothing - all operations are pass-through.
    Useful for:
    - Disabling caching
    - Testing without cache
    - Development mode

    Example:
        >>> cache = NoOpCache()
        >>> cache.set('key', 'value')  # Does nothing
        >>> value = cache.get('key')  # Always returns None
        >>> assert value is None
    """

    def get(self, key: str) -> Optional[Any]:
        """Always returns None (cache miss)."""
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Does nothing."""
        pass

    def delete(self, key: str) -> bool:
        """Always returns False (nothing to delete)."""
        return False

    def clear(self):
        """Does nothing."""
        pass

    def stats(self) -> Dict[str, Any]:
        """
        Returns empty statistics.

        Returns:
            Dictionary with zero statistics
        """
        return {
            'size': 0,
            'hits': 0,
            'misses': 0,
            'hit_rate': 0.0,
            'memory_bytes': 0,
        }
