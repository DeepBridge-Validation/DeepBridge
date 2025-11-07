"""
Cache manager for coordinating multiple caches (Phase 3 Sprint 17-18).

Provides high-level API for caching in report generation.
"""

from typing import Any, Optional, Dict
import hashlib
import json

from .base import CacheStrategy
from .memory_cache import MemoryCache
from .no_op_cache import NoOpCache


class CacheManager:
    """
    Manages multiple caches for different purposes.

    Features:
    - Separate caches for charts, templates, data
    - Smart cache key generation
    - Unified API
    - Statistics aggregation

    Example:
        >>> manager = CacheManager(
        ...     chart_cache=MemoryCache(max_size=100, default_ttl=3600),
        ...     template_cache=MemoryCache(max_size=50, default_ttl=7200),
        ... )
        >>>
        >>> # Cache a chart
        >>> key = manager.make_chart_key('coverage', {'alphas': [...], 'coverage': [...]})
        >>> manager.cache_chart(key, chart_data)
        >>>
        >>> # Get cached chart
        >>> cached = manager.get_chart(key)
    """

    def __init__(
        self,
        chart_cache: Optional[CacheStrategy] = None,
        template_cache: Optional[CacheStrategy] = None,
        data_cache: Optional[CacheStrategy] = None,
        enabled: bool = True,
    ):
        """
        Initialize cache manager.

        Args:
            chart_cache: Cache for generated charts (default: MemoryCache)
            template_cache: Cache for compiled templates (default: MemoryCache)
            data_cache: Cache for transformed data (default: MemoryCache)
            enabled: Enable/disable all caching (default: True)
        """
        if not enabled:
            # Disable all caching
            self.chart_cache = NoOpCache()
            self.template_cache = NoOpCache()
            self.data_cache = NoOpCache()
        else:
            # Use provided caches or defaults
            self.chart_cache = chart_cache or MemoryCache(
                max_size=100,
                default_ttl=3600  # 1 hour
            )
            self.template_cache = template_cache or MemoryCache(
                max_size=50,
                default_ttl=7200  # 2 hours
            )
            self.data_cache = data_cache or MemoryCache(
                max_size=200,
                default_ttl=1800  # 30 minutes
            )

        self.enabled = enabled

    # =========================================================================
    # Chart Caching
    # =========================================================================

    def make_chart_key(self, chart_type: str, data: Dict[str, Any]) -> str:
        """
        Generate cache key for chart.

        Args:
            chart_type: Type of chart (e.g., 'coverage', 'calibration')
            data: Chart data dictionary

        Returns:
            Cache key (format: "chart:{type}:{data_hash}")

        Example:
            >>> key = manager.make_chart_key('coverage', {'alphas': [0.1, 0.2], 'coverage': [0.9, 0.8]})
            >>> # Returns: "chart:coverage:a1b2c3d4..."
        """
        data_hash = self._hash_dict(data)
        return f"chart:{chart_type}:{data_hash}"

    def get_chart(self, key: str) -> Optional[Any]:
        """Get cached chart."""
        return self.chart_cache.get(key)

    def cache_chart(self, key: str, value: Any, ttl: Optional[int] = None):
        """Cache generated chart."""
        self.chart_cache.set(key, value, ttl)

    def get_or_generate_chart(
        self,
        chart_type: str,
        data: Dict[str, Any],
        generator,
        ttl: Optional[int] = None
    ) -> Any:
        """
        Get cached chart or generate and cache it.

        Args:
            chart_type: Type of chart
            data: Chart data
            generator: Callable that generates chart
            ttl: Time to live in seconds

        Returns:
            Cached or freshly generated chart

        Example:
            >>> def generate():
            ...     return expensive_chart_generation()
            >>>
            >>> chart = manager.get_or_generate_chart('coverage', data, generate, ttl=3600)
        """
        key = self.make_chart_key(chart_type, data)
        return self.chart_cache.get_or_set(key, generator, ttl)

    # =========================================================================
    # Template Caching
    # =========================================================================

    def make_template_key(self, template_path: str) -> str:
        """
        Generate cache key for template.

        Args:
            template_path: Path to template file

        Returns:
            Cache key (format: "template:{path_hash}")

        Example:
            >>> key = manager.make_template_key('/templates/report.html')
            >>> # Returns: "template:e5f6g7h8..."
        """
        path_hash = hashlib.md5(template_path.encode()).hexdigest()[:16]
        return f"template:{path_hash}"

    def get_template(self, key: str) -> Optional[Any]:
        """Get cached template."""
        return self.template_cache.get(key)

    def cache_template(self, key: str, value: Any, ttl: Optional[int] = None):
        """Cache compiled template."""
        self.template_cache.set(key, value, ttl)

    # =========================================================================
    # Data Caching
    # =========================================================================

    def make_data_key(self, data_type: str, identifier: str) -> str:
        """
        Generate cache key for transformed data.

        Args:
            data_type: Type of data (e.g., 'metrics', 'results')
            identifier: Unique identifier for this data

        Returns:
            Cache key (format: "data:{type}:{id}")

        Example:
            >>> key = manager.make_data_key('metrics', 'model_123')
            >>> # Returns: "data:metrics:model_123"
        """
        return f"data:{data_type}:{identifier}"

    def get_data(self, key: str) -> Optional[Any]:
        """Get cached data."""
        return self.data_cache.get(key)

    def cache_data(self, key: str, value: Any, ttl: Optional[int] = None):
        """Cache transformed data."""
        self.data_cache.set(key, value, ttl)

    # =========================================================================
    # Management
    # =========================================================================

    def clear_all(self):
        """Clear all caches."""
        self.chart_cache.clear()
        self.template_cache.clear()
        self.data_cache.clear()

    def stats(self) -> Dict[str, Any]:
        """
        Get statistics for all caches.

        Returns:
            Dictionary with statistics for each cache

        Example:
            >>> stats = manager.stats()
            >>> print(f"Chart cache hit rate: {stats['chart']['hit_rate']:.1f}%")
        """
        return {
            'chart': self.chart_cache.stats(),
            'template': self.template_cache.stats(),
            'data': self.data_cache.stats(),
            'enabled': self.enabled,
        }

    def enable(self):
        """
        Enable caching.

        Note: Replaces NoOpCaches with MemoryCaches.
        """
        if not self.enabled:
            self.chart_cache = MemoryCache(max_size=100, default_ttl=3600)
            self.template_cache = MemoryCache(max_size=50, default_ttl=7200)
            self.data_cache = MemoryCache(max_size=200, default_ttl=1800)
            self.enabled = True

    def disable(self):
        """
        Disable caching.

        Note: Replaces all caches with NoOpCaches.
        """
        if self.enabled:
            self.chart_cache = NoOpCache()
            self.template_cache = NoOpCache()
            self.data_cache = NoOpCache()
            self.enabled = False

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _hash_dict(data: Dict[str, Any]) -> str:
        """
        Generate hash for dictionary.

        Args:
            data: Dictionary to hash

        Returns:
            MD5 hash (first 16 characters)

        Example:
            >>> hash_val = CacheManager._hash_dict({'a': 1, 'b': 2})
        """
        # Sort keys for consistent hashing
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(json_str.encode()).hexdigest()[:16]
