"""
Tests for cache system (Phase 3 Sprint 17-18).

Tests all cache implementations and CacheManager.
"""

import time
from datetime import datetime, timedelta

import pytest

from deepbridge.core.experiment.report.cache import (
    CacheEntry,
    CacheManager,
    CacheStrategy,
    MemoryCache,
    NoOpCache,
)

# =============================================================================
# CacheEntry Tests
# =============================================================================

class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_cache_entry_creation(self):
        """Test creating cache entry."""
        entry = CacheEntry(
            value="test_value",
            created_at=datetime.now(),
            ttl=3600,
        )

        assert entry.value == "test_value"
        assert entry.ttl == 3600
        assert entry.hits == 0

    def test_cache_entry_expiration(self):
        """Test entry expiration."""
        # Not expired (1 hour TTL)
        entry = CacheEntry(
            value="test",
            created_at=datetime.now(),
            ttl=3600,
        )
        assert not entry.is_expired

        # Expired (created 2 hours ago, 1 hour TTL)
        old_entry = CacheEntry(
            value="test",
            created_at=datetime.now() - timedelta(hours=2),
            ttl=3600,
        )
        assert old_entry.is_expired

    def test_cache_entry_no_expiration(self):
        """Test entry with no TTL never expires."""
        entry = CacheEntry(
            value="test",
            created_at=datetime.now() - timedelta(days=365),
            ttl=None,
        )
        assert not entry.is_expired

    def test_cache_entry_age(self):
        """Test entry age calculation."""
        entry = CacheEntry(
            value="test",
            created_at=datetime.now() - timedelta(seconds=10),
            ttl=None,
        )

        # Age should be approximately 10 seconds
        assert 9 < entry.age_seconds < 11


# =============================================================================
# MemoryCache Tests
# =============================================================================

class TestMemoryCache:
    """Tests for MemoryCache."""

    def test_memory_cache_initialization(self):
        """Test memory cache initialization."""
        cache = MemoryCache(max_size=100, default_ttl=3600)

        assert cache.max_size == 100
        assert cache.default_ttl == 3600

        stats = cache.stats()
        assert stats['size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0

    def test_memory_cache_set_and_get(self):
        """Test basic set and get operations."""
        cache = MemoryCache()

        cache.set('key1', 'value1')
        value = cache.get('key1')

        assert value == 'value1'

    def test_memory_cache_get_miss(self):
        """Test cache miss."""
        cache = MemoryCache()

        value = cache.get('nonexistent')

        assert value is None

    def test_memory_cache_ttl_expiration(self):
        """Test TTL expiration."""
        cache = MemoryCache()

        # Set with 1 second TTL
        cache.set('key1', 'value1', ttl=1)

        # Should exist initially
        assert cache.get('key1') == 'value1'

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert cache.get('key1') is None

    def test_memory_cache_lru_eviction(self):
        """Test LRU eviction when max_size exceeded."""
        cache = MemoryCache(max_size=3)

        # Add 4 entries (should evict oldest)
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')
        cache.set('key4', 'value4')  # Should evict key1

        # key1 should be evicted
        assert cache.get('key1') is None

        # Others should exist
        assert cache.get('key2') == 'value2'
        assert cache.get('key3') == 'value3'
        assert cache.get('key4') == 'value4'

    def test_memory_cache_lru_access_order(self):
        """Test that accessing entries updates LRU order."""
        cache = MemoryCache(max_size=3)

        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')

        # Access key1 (makes it most recent)
        cache.get('key1')

        # Add key4 (should evict key2, not key1)
        cache.set('key4', 'value4')

        # key2 should be evicted (least recently used)
        assert cache.get('key2') is None

        # key1 should still exist (was accessed recently)
        assert cache.get('key1') == 'value1'

    def test_memory_cache_delete(self):
        """Test deleting entries."""
        cache = MemoryCache()

        cache.set('key1', 'value1')
        assert cache.delete('key1') is True
        assert cache.get('key1') is None

        # Deleting nonexistent key
        assert cache.delete('nonexistent') is False

    def test_memory_cache_clear(self):
        """Test clearing cache."""
        cache = MemoryCache()

        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        cache.clear()

        assert cache.get('key1') is None
        assert cache.get('key2') is None
        assert cache.stats()['size'] == 0

    def test_memory_cache_stats(self):
        """Test cache statistics."""
        cache = MemoryCache()

        cache.set('key1', 'value1')

        # Hit
        cache.get('key1')

        # Miss
        cache.get('key2')

        stats = cache.stats()

        assert stats['size'] == 1
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 50.0

    def test_memory_cache_has(self):
        """Test has() method."""
        cache = MemoryCache()

        cache.set('key1', 'value1')

        assert cache.has('key1') is True
        assert cache.has('nonexistent') is False

    def test_memory_cache_get_or_set(self):
        """Test get_or_set() method."""
        cache = MemoryCache()

        # Factory function
        call_count = [0]

        def factory():
            call_count[0] += 1
            return 'generated_value'

        # First call should generate
        value1 = cache.get_or_set('key1', factory)
        assert value1 == 'generated_value'
        assert call_count[0] == 1

        # Second call should use cache
        value2 = cache.get_or_set('key1', factory)
        assert value2 == 'generated_value'
        assert call_count[0] == 1  # Factory not called again

    def test_memory_cache_evict_expired(self):
        """Test manual eviction of expired entries."""
        cache = MemoryCache()

        cache.set('key1', 'value1', ttl=1)
        cache.set('key2', 'value2', ttl=None)

        # Wait for key1 to expire
        time.sleep(1.1)

        evicted = cache.evict_expired()

        assert evicted == 1
        assert cache.get('key1') is None
        assert cache.get('key2') == 'value2'

    def test_memory_cache_resize(self):
        """Test resizing cache."""
        cache = MemoryCache(max_size=5)

        for i in range(5):
            cache.set(f'key{i}', f'value{i}')

        # Resize to 3 (should evict 2 oldest)
        cache.resize(3)

        assert cache.stats()['size'] == 3
        assert cache.stats()['max_size'] == 3

    def test_memory_cache_keys(self):
        """Test getting all keys."""
        cache = MemoryCache()

        cache.set('key1', 'value1')
        cache.set('key2', 'value2')

        keys = cache.keys()

        assert len(keys) == 2
        assert 'key1' in keys
        assert 'key2' in keys


# =============================================================================
# NoOpCache Tests
# =============================================================================

class TestNoOpCache:
    """Tests for NoOpCache."""

    def test_no_op_cache_get(self):
        """Test get always returns None."""
        cache = NoOpCache()

        cache.set('key', 'value')
        value = cache.get('key')

        assert value is None

    def test_no_op_cache_set(self):
        """Test set does nothing."""
        cache = NoOpCache()

        # Should not raise
        cache.set('key', 'value')

    def test_no_op_cache_delete(self):
        """Test delete returns False."""
        cache = NoOpCache()

        result = cache.delete('key')

        assert result is False

    def test_no_op_cache_clear(self):
        """Test clear does nothing."""
        cache = NoOpCache()

        # Should not raise
        cache.clear()

    def test_no_op_cache_stats(self):
        """Test stats returns zeros."""
        cache = NoOpCache()

        stats = cache.stats()

        assert stats['size'] == 0
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['hit_rate'] == 0.0


# =============================================================================
# CacheManager Tests
# =============================================================================

class TestCacheManager:
    """Tests for CacheManager."""

    def test_cache_manager_initialization(self):
        """Test cache manager initialization."""
        manager = CacheManager()

        assert manager.enabled is True
        assert manager.chart_cache is not None
        assert manager.template_cache is not None
        assert manager.data_cache is not None

    def test_cache_manager_disabled(self):
        """Test cache manager with caching disabled."""
        manager = CacheManager(enabled=False)

        assert manager.enabled is False
        assert isinstance(manager.chart_cache, NoOpCache)

    def test_cache_manager_chart_key_generation(self):
        """Test chart cache key generation."""
        manager = CacheManager()

        key = manager.make_chart_key('coverage', {'alphas': [0.1, 0.2], 'coverage': [0.9, 0.8]})

        assert key.startswith('chart:coverage:')
        assert len(key) > len('chart:coverage:')

    def test_cache_manager_chart_caching(self):
        """Test caching and retrieving charts."""
        manager = CacheManager()

        key = manager.make_chart_key('coverage', {'data': 'test'})
        chart_data = {'plotly': 'json'}

        # Cache chart
        manager.cache_chart(key, chart_data)

        # Retrieve chart
        cached = manager.get_chart(key)

        assert cached == chart_data

    def test_cache_manager_get_or_generate_chart(self):
        """Test get_or_generate_chart method."""
        manager = CacheManager()

        call_count = [0]

        def generator():
            call_count[0] += 1
            return {'generated': 'chart'}

        # First call generates
        chart1 = manager.get_or_generate_chart('coverage', {'data': 'test'}, generator)
        assert chart1 == {'generated': 'chart'}
        assert call_count[0] == 1

        # Second call uses cache
        chart2 = manager.get_or_generate_chart('coverage', {'data': 'test'}, generator)
        assert chart2 == {'generated': 'chart'}
        assert call_count[0] == 1  # Not called again

    def test_cache_manager_template_key_generation(self):
        """Test template cache key generation."""
        manager = CacheManager()

        key = manager.make_template_key('/templates/report.html')

        assert key.startswith('template:')

    def test_cache_manager_template_caching(self):
        """Test caching and retrieving templates."""
        manager = CacheManager()

        key = manager.make_template_key('/templates/report.html')
        template = "compiled template"

        manager.cache_template(key, template)
        cached = manager.get_template(key)

        assert cached == template

    def test_cache_manager_data_key_generation(self):
        """Test data cache key generation."""
        manager = CacheManager()

        key = manager.make_data_key('metrics', 'model_123')

        assert key == 'data:metrics:model_123'

    def test_cache_manager_data_caching(self):
        """Test caching and retrieving data."""
        manager = CacheManager()

        key = manager.make_data_key('metrics', 'model_123')
        data = {'accuracy': 0.95}

        manager.cache_data(key, data)
        cached = manager.get_data(key)

        assert cached == data

    def test_cache_manager_clear_all(self):
        """Test clearing all caches."""
        manager = CacheManager()

        # Add to all caches
        chart_key = manager.make_chart_key('coverage', {'data': 'test'})
        template_key = manager.make_template_key('/templates/report.html')
        data_key = manager.make_data_key('metrics', 'model_123')

        manager.cache_chart(chart_key, 'chart')
        manager.cache_template(template_key, 'template')
        manager.cache_data(data_key, 'data')

        # Clear all
        manager.clear_all()

        assert manager.get_chart(chart_key) is None
        assert manager.get_template(template_key) is None
        assert manager.get_data(data_key) is None

    def test_cache_manager_stats(self):
        """Test getting statistics from all caches."""
        manager = CacheManager()

        stats = manager.stats()

        assert 'chart' in stats
        assert 'template' in stats
        assert 'data' in stats
        assert 'enabled' in stats
        assert stats['enabled'] is True

    def test_cache_manager_enable_disable(self):
        """Test enabling and disabling caching."""
        manager = CacheManager()

        # Disable
        manager.disable()
        assert manager.enabled is False
        assert isinstance(manager.chart_cache, NoOpCache)

        # Enable
        manager.enable()
        assert manager.enabled is True
        assert isinstance(manager.chart_cache, MemoryCache)

    def test_cache_manager_hash_dict_consistency(self):
        """Test that hash_dict generates consistent hashes."""
        manager = CacheManager()

        data1 = {'a': 1, 'b': 2, 'c': 3}
        data2 = {'c': 3, 'a': 1, 'b': 2}  # Different order

        hash1 = manager._hash_dict(data1)
        hash2 = manager._hash_dict(data2)

        # Should be the same (order-independent)
        assert hash1 == hash2


# =============================================================================
# Integration Tests
# =============================================================================

class TestCacheIntegration:
    """Integration tests for cache system."""

    def test_cache_with_multiple_charts(self):
        """Test caching multiple charts."""
        manager = CacheManager()

        # Cache different charts
        for i in range(10):
            key = manager.make_chart_key(f'chart_{i}', {'data': i})
            manager.cache_chart(key, f'chart_data_{i}')

        # Verify all cached
        for i in range(10):
            key = manager.make_chart_key(f'chart_{i}', {'data': i})
            assert manager.get_chart(key) == f'chart_data_{i}'

    def test_cache_performance_improvement(self):
        """Test that cache improves performance."""
        manager = CacheManager()

        call_count = [0]

        def expensive_operation():
            call_count[0] += 1
            time.sleep(0.01)  # Simulate expensive operation
            return 'result'

        # First call (not cached)
        start = time.time()
        result1 = manager.get_or_generate_chart('test', {'data': 'test'}, expensive_operation)
        first_duration = time.time() - start

        # Second call (cached)
        start = time.time()
        result2 = manager.get_or_generate_chart('test', {'data': 'test'}, expensive_operation)
        second_duration = time.time() - start

        # Cached call should be much faster
        assert second_duration < first_duration / 2
        assert call_count[0] == 1  # Only called once

    def test_cache_with_custom_caches(self):
        """Test cache manager with custom cache implementations."""
        chart_cache = MemoryCache(max_size=50, default_ttl=1800)
        template_cache = MemoryCache(max_size=25, default_ttl=3600)

        manager = CacheManager(
            chart_cache=chart_cache,
            template_cache=template_cache
        )

        # Use custom caches
        key = manager.make_chart_key('test', {'data': 'test'})
        manager.cache_chart(key, 'data')

        assert manager.get_chart(key) == 'data'
        assert chart_cache.stats()['size'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
