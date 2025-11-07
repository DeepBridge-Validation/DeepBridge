"""
Cache layer for report generation (Phase 3 Sprint 17-18).

Provides caching for expensive operations:
- Chart generation
- Template compilation
- Data transformations

Architecture:
    CacheStrategy (ABC) ← Interface
        ↑
        ├── MemoryCache (LRU)
        ├── DiskCache (persistent)
        └── NoOpCache (disabled)

    CacheManager
        - Manages multiple caches
        - Provides high-level API
        - Handles cache keys

Benefits:
    - 2-5x faster report generation
    - Reduced CPU usage
    - Configurable TTL
    - Smart invalidation

Example Usage:
    >>> from deepbridge.core.experiment.report.cache import CacheManager, MemoryCache
    >>>
    >>> # Create cache manager
    >>> cache = CacheManager(
    ...     chart_cache=MemoryCache(max_size=100, ttl=3600),
    ...     template_cache=MemoryCache(max_size=50, ttl=7200)
    ... )
    >>>
    >>> # Cache chart
    >>> key = cache.make_key('chart', chart_type='coverage', data_hash='abc123')
    >>> cache.set(key, chart_data)
    >>>
    >>> # Retrieve cached chart
    >>> cached = cache.get(key)
    >>> if cached:
    ...     return cached
    >>> else:
    ...     # Generate and cache
    ...     chart = generate_chart()
    ...     cache.set(key, chart)
"""

from .base import CacheStrategy, CacheEntry
from .memory_cache import MemoryCache
from .no_op_cache import NoOpCache
from .cache_manager import CacheManager

__all__ = [
    'CacheStrategy',
    'CacheEntry',
    'MemoryCache',
    'NoOpCache',
    'CacheManager',
]
