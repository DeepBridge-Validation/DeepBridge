"""
Intelligent Cache System for HPM-KD

This module implements an intelligent caching system that eliminates redundant
computations by storing and reusing teacher predictions, features, and attention maps.
"""

import hashlib
import pickle
import json
import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union, Callable, Tuple
from collections import OrderedDict
from functools import wraps
import time
import logging
import gc

# Try to import psutil, but make it optional
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Least Recently Used (LRU) cache implementation with size limit.
    """

    def __init__(self, max_size_bytes: int):
        """
        Initialize LRU cache.

        Args:
            max_size_bytes: Maximum cache size in bytes
        """
        self.max_size_bytes = max_size_bytes
        self.cache = OrderedDict()
        self.size_bytes = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """
        Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]['data']

        self.misses += 1
        return None

    def put(self, key: str, value: Any, size_bytes: int):
        """
        Add item to cache.

        Args:
            key: Cache key
            value: Value to cache
            size_bytes: Size of value in bytes
        """
        # Remove old entry if exists
        if key in self.cache:
            self.size_bytes -= self.cache[key]['size']
            del self.cache[key]

        # Evict items if needed
        while self.size_bytes + size_bytes > self.max_size_bytes and self.cache:
            evicted_key, evicted_value = self.cache.popitem(last=False)
            self.size_bytes -= evicted_value['size']

        # Add new item
        self.cache[key] = {'data': value, 'size': size_bytes}
        self.size_bytes += size_bytes

    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.size_bytes = 0
        self.hits = 0
        self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'size_bytes': self.size_bytes,
            'size_mb': self.size_bytes / (1024 * 1024),
            'num_items': len(self.cache),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / max(1, self.hits + self.misses)
        }


class IntelligentCache:
    """
    Intelligent caching system for knowledge distillation computations.

    This system caches:
    - Teacher model predictions
    - Intermediate features
    - Attention maps
    - Computed metrics
    """

    def __init__(
        self,
        max_memory_gb: float = 2.0,
        teacher_ratio: float = 0.5,
        feature_ratio: float = 0.3,
        attention_ratio: float = 0.2
    ):
        """
        Initialize the intelligent cache system.

        Args:
            max_memory_gb: Maximum memory to use for caching (GB)
            teacher_ratio: Proportion of cache for teacher predictions
            feature_ratio: Proportion of cache for features
            attention_ratio: Proportion of cache for attention maps
        """
        self.max_memory_gb = max_memory_gb
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)

        # Initialize separate caches for different data types
        self.teacher_cache = LRUCache(int(self.max_memory_bytes * teacher_ratio))
        self.feature_cache = LRUCache(int(self.max_memory_bytes * feature_ratio))
        self.attention_cache = LRUCache(int(self.max_memory_bytes * attention_ratio))

        # Computation timing statistics
        self.timing_stats = {
            'cache_hits_time': 0.0,
            'cache_misses_time': 0.0,
            'total_computations': 0
        }

        # Monitor memory usage
        self._check_memory_usage()

    def get_or_compute(
        self,
        key: Any,
        compute_fn: Callable,
        cache_type: str = 'teacher',
        force_recompute: bool = False
    ) -> Any:
        """
        Get value from cache or compute it.

        Args:
            key: Cache key (will be hashed)
            compute_fn: Function to compute value if not cached
            cache_type: Type of cache to use ('teacher', 'feature', 'attention')
            force_recompute: Force recomputation even if cached

        Returns:
            Cached or computed value
        """
        start_time = time.time()

        # Generate hash key
        hash_key = self._generate_hash_key(key)

        # Select appropriate cache
        cache = self._get_cache(cache_type)

        # Try to get from cache
        if not force_recompute:
            cached_value = cache.get(hash_key)
            if cached_value is not None:
                elapsed = time.time() - start_time
                self.timing_stats['cache_hits_time'] += elapsed
                logger.debug(f"Cache hit for {cache_type} (key: {hash_key[:8]}...)")
                return cached_value

        # Compute value
        logger.debug(f"Cache miss for {cache_type}, computing...")
        computed_value = compute_fn()

        # Estimate size and cache
        size_bytes = self._estimate_size(computed_value)
        cache.put(hash_key, computed_value, size_bytes)

        elapsed = time.time() - start_time
        self.timing_stats['cache_misses_time'] += elapsed
        self.timing_stats['total_computations'] += 1

        return computed_value

    def cache_teacher_predictions(
        self,
        teacher_model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        Cache teacher model predictions.

        Args:
            teacher_model: Teacher model
            X: Input data
            temperature: Temperature for softmax scaling

        Returns:
            Teacher predictions
        """
        # Create cache key
        cache_key = {
            'model_id': id(teacher_model),
            'X_shape': X.shape,
            'X_sample': self._get_data_sample(X),
            'temperature': temperature
        }

        def compute_predictions():
            # Get raw predictions
            if hasattr(teacher_model, 'predict_proba'):
                probs = teacher_model.predict_proba(X)
            else:
                # For models without predict_proba
                predictions = teacher_model.predict(X)
                probs = np.eye(len(np.unique(predictions)))[predictions]

            # Apply temperature scaling
            if temperature != 1.0:
                probs = self._apply_temperature(probs, temperature)

            return probs

        return self.get_or_compute(
            key=cache_key,
            compute_fn=compute_predictions,
            cache_type='teacher'
        )

    def cache_features(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        layer_name: Optional[str] = None
    ) -> np.ndarray:
        """
        Cache intermediate features from a model.

        Args:
            model: Model to extract features from
            X: Input data
            layer_name: Optional specific layer to extract

        Returns:
            Extracted features
        """
        cache_key = {
            'model_id': id(model),
            'X_shape': X.shape,
            'X_sample': self._get_data_sample(X),
            'layer': layer_name
        }

        def compute_features():
            # Extract features based on model type
            if hasattr(model, 'transform'):
                # For models with transform method
                return model.transform(X)

            elif hasattr(model, 'decision_function'):
                # For models with decision function
                return model.decision_function(X)

            elif hasattr(model, 'predict'):
                # Fallback to predictions
                return model.predict(X)

            else:
                raise ValueError(f"Cannot extract features from model type: {type(model)}")

        return self.get_or_compute(
            key=cache_key,
            compute_fn=compute_features,
            cache_type='feature'
        )

    def cache_attention_maps(
        self,
        model: Any,
        X: Union[np.ndarray, pd.DataFrame],
        method: str = 'gradient'
    ) -> np.ndarray:
        """
        Cache attention or importance maps.

        Args:
            model: Model to extract attention from
            X: Input data
            method: Method for attention extraction

        Returns:
            Attention maps
        """
        cache_key = {
            'model_id': id(model),
            'X_shape': X.shape,
            'X_sample': self._get_data_sample(X),
            'method': method
        }

        def compute_attention():
            # Extract attention based on model type
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
                # Broadcast to match data shape
                attention = np.tile(importances, (X.shape[0], 1))
                return attention

            elif hasattr(model, 'coef_'):
                # Linear models
                coef = model.coef_
                if coef.ndim == 1:
                    coef = coef.reshape(1, -1)
                # Use absolute coefficients as attention
                attention = np.abs(coef)
                # Broadcast to match data shape
                attention = np.tile(attention, (X.shape[0], 1))
                return attention

            else:
                # Fallback: uniform attention
                return np.ones((X.shape[0], X.shape[1])) / X.shape[1]

        return self.get_or_compute(
            key=cache_key,
            compute_fn=compute_attention,
            cache_type='attention'
        )

    def _generate_hash_key(self, key: Any) -> str:
        """
        Generate a hash key from arbitrary input.

        Args:
            key: Input to hash

        Returns:
            Hash string
        """
        # Convert to string representation
        if isinstance(key, dict):
            key_str = json.dumps(key, sort_keys=True, default=str)
        elif isinstance(key, (list, tuple)):
            key_str = str(key)
        elif isinstance(key, np.ndarray):
            key_str = f"{key.shape}_{key.dtype}_{hash(key.tobytes())}"
        else:
            key_str = str(key)

        # Generate MD5 hash
        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_data_sample(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        sample_size: int = 100
    ) -> str:
        """
        Get a representative sample of data for cache key.

        Args:
            X: Input data
            sample_size: Number of elements to sample

        Returns:
            String representation of data sample
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Sample first and last elements
        flat = X.flatten()
        if len(flat) > sample_size:
            indices = np.linspace(0, len(flat) - 1, sample_size, dtype=int)
            sample = flat[indices]
        else:
            sample = flat

        # Round to reduce minor differences
        sample = np.round(sample, 6)

        return str(sample.tolist())

    def _apply_temperature(self, probs: np.ndarray, temperature: float) -> np.ndarray:
        """
        Apply temperature scaling to probabilities.

        Args:
            probs: Probability array
            temperature: Temperature value

        Returns:
            Scaled probabilities
        """
        # Apply temperature to logits
        logits = np.log(probs + 1e-10)
        scaled_logits = logits / temperature

        # Convert back to probabilities
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=-1, keepdims=True))
        scaled_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return scaled_probs

    def _estimate_size(self, obj: Any) -> int:
        """
        Estimate size of object in bytes.

        Args:
            obj: Object to estimate

        Returns:
            Estimated size in bytes
        """
        if isinstance(obj, np.ndarray):
            return obj.nbytes

        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()

        # Fallback: serialize and measure
        try:
            serialized = pickle.dumps(obj)
            return len(serialized)
        except:
            # Rough estimate
            return 1024  # 1 KB default

    def _get_cache(self, cache_type: str) -> LRUCache:
        """
        Get the appropriate cache based on type.

        Args:
            cache_type: Type of cache

        Returns:
            Cache object
        """
        if cache_type == 'teacher':
            return self.teacher_cache
        elif cache_type == 'feature':
            return self.feature_cache
        elif cache_type == 'attention':
            return self.attention_cache
        else:
            raise ValueError(f"Unknown cache type: {cache_type}")

    def _check_memory_usage(self):
        """
        Check system memory usage and adjust cache if needed.
        """
        if HAS_PSUTIL:
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            if memory_usage_percent > 90:
                logger.warning(f"High memory usage detected ({memory_usage_percent}%)")
                self.reduce_cache_size(0.5)
        else:
            # Without psutil, do a simple size check
            total_size = (self.teacher_cache.size_bytes +
                         self.feature_cache.size_bytes +
                         self.attention_cache.size_bytes)
            if total_size > self.max_memory_bytes * 0.9:
                logger.warning("Cache size approaching limit")
                self.reduce_cache_size(0.5)

    def reduce_cache_size(self, factor: float = 0.5):
        """
        Reduce cache size by a factor.

        Args:
            factor: Factor to reduce by (0.5 = half size)
        """
        logger.info(f"Reducing cache size by factor {factor}")

        # Clear portions of each cache
        for cache in [self.teacher_cache, self.feature_cache, self.attention_cache]:
            items_to_remove = int(len(cache.cache) * (1 - factor))
            for _ in range(items_to_remove):
                if cache.cache:
                    cache.cache.popitem(last=False)

        # Force garbage collection
        gc.collect()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        stats = {
            'teacher_cache': self.teacher_cache.get_stats(),
            'feature_cache': self.feature_cache.get_stats(),
            'attention_cache': self.attention_cache.get_stats(),
            'timing': self.timing_stats,
            'total_size_mb': (
                self.teacher_cache.size_bytes +
                self.feature_cache.size_bytes +
                self.attention_cache.size_bytes
            ) / (1024 * 1024),
            'memory_usage_percent': psutil.virtual_memory().percent if HAS_PSUTIL else -1
        }

        # Calculate time saved
        if self.timing_stats['total_computations'] > 0:
            avg_computation_time = (
                self.timing_stats['cache_misses_time'] /
                max(1, self.teacher_cache.misses + self.feature_cache.misses + self.attention_cache.misses)
            )
            total_hits = self.teacher_cache.hits + self.feature_cache.hits + self.attention_cache.hits
            stats['time_saved_seconds'] = avg_computation_time * total_hits

        return stats

    def clear_all(self):
        """
        Clear all caches.
        """
        self.teacher_cache.clear()
        self.feature_cache.clear()
        self.attention_cache.clear()

        # Reset timing stats
        self.timing_stats = {
            'cache_hits_time': 0.0,
            'cache_misses_time': 0.0,
            'total_computations': 0
        }

        # Force garbage collection
        gc.collect()

        logger.info("All caches cleared")


# Decorator for automatic caching
def cached_computation(cache: IntelligentCache, cache_type: str = 'teacher'):
    """
    Decorator for automatic caching of function results.

    Args:
        cache: IntelligentCache instance
        cache_type: Type of cache to use

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function arguments
            cache_key = {
                'func': func.__name__,
                'args': str(args),
                'kwargs': str(kwargs)
            }

            return cache.get_or_compute(
                key=cache_key,
                compute_fn=lambda: func(*args, **kwargs),
                cache_type=cache_type
            )

        return wrapper
    return decorator