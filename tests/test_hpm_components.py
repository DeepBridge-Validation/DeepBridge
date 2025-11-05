"""
Unit tests for HPM-KD base components
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from deepbridge.distillation.techniques.hpm import (
    AdaptiveConfigurationManager,
    SharedOptimizationMemory,
    IntelligentCache
)
from deepbridge.utils.model_registry import ModelType


class TestAdaptiveConfigurationManager(unittest.TestCase):
    """Test suite for AdaptiveConfigurationManager"""

    def setUp(self):
        """Set up test fixtures"""
        self.manager = AdaptiveConfigurationManager(
            max_configs=16,
            initial_samples=4,
            exploration_ratio=0.3,
            random_state=42
        )

        self.model_types = [
            ModelType.LOGISTIC_REGRESSION,
            ModelType.DECISION_TREE,
            ModelType.GBM,
            ModelType.XGB
        ]
        self.temperatures = [0.5, 1.0, 2.0, 3.0]
        self.alphas = [0.3, 0.5, 0.7, 0.9]

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.manager.max_configs, 16)
        self.assertEqual(self.manager.initial_samples, 4)
        self.assertEqual(self.manager.exploration_ratio, 0.3)
        self.assertIsNotNone(self.manager.gp_model)

    def test_config_reduction(self):
        """Test configuration reduction from 64 to 16"""
        configs = self.manager.select_promising_configs(
            model_types=self.model_types,
            temperatures=self.temperatures,
            alphas=self.alphas
        )

        # Should reduce to max_configs
        self.assertEqual(len(configs), 16)

        # Each config should have required keys
        for config in configs:
            self.assertIn('model_type', config)
            self.assertIn('temperature', config)
            self.assertIn('alpha', config)

    def test_small_config_space(self):
        """Test when config space is smaller than max_configs"""
        configs = self.manager.select_promising_configs(
            model_types=[ModelType.LOGISTIC_REGRESSION],
            temperatures=[1.0, 2.0],
            alphas=[0.5, 0.7]
        )

        # Should return all 4 configurations (1 * 2 * 2)
        self.assertEqual(len(configs), 4)

    def test_stratified_sampling(self):
        """Test stratified sampling for diversity"""
        configs = self.manager._stratified_sampling(
            list(zip(self.model_types * 4,
                    [1.0] * 16,
                    [0.5] * 16)),
            n_samples=8
        )

        # Should have samples from different model types
        model_types_sampled = set(c[0] for c in configs)
        self.assertGreater(len(model_types_sampled), 1)

    def test_dataset_features_influence(self):
        """Test that dataset features influence configuration selection"""
        # Small dataset features
        small_dataset = {
            'n_samples': 500,
            'n_features': 10
        }

        configs_small = self.manager.select_promising_configs(
            model_types=self.model_types,
            temperatures=self.temperatures,
            alphas=self.alphas,
            dataset_features=small_dataset
        )

        # Large dataset features
        large_dataset = {
            'n_samples': 50000,
            'n_features': 100
        }

        configs_large = self.manager.select_promising_configs(
            model_types=self.model_types,
            temperatures=self.temperatures,
            alphas=self.alphas,
            dataset_features=large_dataset
        )

        # Configurations should be different
        configs_small_set = set(
            (c['model_type'], c['temperature'], c['alpha'])
            for c in configs_small
        )
        configs_large_set = set(
            (c['model_type'], c['temperature'], c['alpha'])
            for c in configs_large
        )

        # At least some differences expected
        self.assertNotEqual(configs_small_set, configs_large_set)

    def test_update_history(self):
        """Test updating configuration history"""
        config = {
            'model_type': ModelType.XGB,
            'temperature': 2.0,
            'alpha': 0.5
        }

        self.manager.update_history(config, 0.85)

        self.assertEqual(len(self.manager.config_history), 1)
        self.assertEqual(len(self.manager.performance_history), 1)
        self.assertEqual(self.manager.performance_history[0], 0.85)


class TestSharedOptimizationMemory(unittest.TestCase):
    """Test suite for SharedOptimizationMemory"""

    def setUp(self):
        """Set up test fixtures"""
        self.memory = SharedOptimizationMemory(
            cache_size=10,
            similarity_threshold=0.8,
            min_reuse_score=0.5
        )

    def test_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.memory.cache_size, 10)
        self.assertEqual(self.memory.similarity_threshold, 0.8)
        self.assertEqual(len(self.memory.param_cache), 0)

    def test_add_and_retrieve_result(self):
        """Test adding and retrieving optimization results"""
        # Add a result
        self.memory.add_result(
            model_type=ModelType.XGB,
            temperature=2.0,
            alpha=0.5,
            best_params={'max_depth': 5, 'n_estimators': 100},
            best_score=0.85,
            n_trials=10
        )

        # Try to retrieve similar config
        similar = self.memory.get_similar_configs(
            model_type=ModelType.XGB,
            temperature=2.1,  # Slightly different
            alpha=0.5
        )

        self.assertEqual(len(similar), 1)
        self.assertEqual(similar[0].best_score, 0.85)

    def test_similarity_calculation(self):
        """Test similarity calculation between configurations"""
        # Add a result
        self.memory.add_result(
            model_type=ModelType.GBM,
            temperature=1.0,
            alpha=0.7,
            best_params={'max_depth': 3},
            best_score=0.8,
            n_trials=5
        )

        # Exact match should be found
        similar = self.memory.get_similar_configs(
            model_type=ModelType.GBM,
            temperature=1.0,
            alpha=0.7
        )
        self.assertEqual(len(similar), 1)

        # Different model type should not be found
        similar = self.memory.get_similar_configs(
            model_type=ModelType.XGB,
            temperature=1.0,
            alpha=0.7
        )
        self.assertEqual(len(similar), 0)

    def test_warm_start_study(self):
        """Test warm-starting an Optuna study"""
        # Add some results
        for i in range(3):
            self.memory.add_result(
                model_type=ModelType.XGB,
                temperature=2.0 + i * 0.1,
                alpha=0.5,
                best_params={'max_depth': 5 + i, 'n_estimators': 100},
                best_score=0.8 + i * 0.02,
                n_trials=10
            )

        similar = self.memory.get_similar_configs(
            model_type=ModelType.XGB,
            temperature=2.0,
            alpha=0.5
        )

        # Create warm-started study
        study = self.memory.warm_start_study(
            model_type=ModelType.XGB,
            temperature=2.0,
            alpha=0.5,
            similar_configs=similar,
            n_trials=10
        )

        self.assertIsNotNone(study)
        # Should have enqueued trials
        self.assertGreater(self.memory.stats['trials_saved'], 0)

    def test_cache_eviction(self):
        """Test LRU eviction when cache is full"""
        # Fill cache beyond capacity
        for i in range(15):
            self.memory.add_result(
                model_type=ModelType.LOGISTIC_REGRESSION,
                temperature=1.0 + i * 0.1,
                alpha=0.5,
                best_params={'C': 1.0},
                best_score=0.7 + i * 0.01,
                n_trials=5
            )

        # Cache should not exceed max size
        self.assertLessEqual(len(self.memory.param_cache), 10)

    def test_save_and_load(self):
        """Test saving and loading cache to/from disk"""
        # Add some results
        self.memory.add_result(
            model_type=ModelType.XGB,
            temperature=2.0,
            alpha=0.5,
            best_params={'max_depth': 5},
            best_score=0.85,
            n_trials=10
        )

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self.memory.save_to_disk(tmp_path)

            # Create new memory and load
            new_memory = SharedOptimizationMemory()
            new_memory.load_from_disk(tmp_path)

            # Should have same data
            self.assertEqual(len(new_memory.param_cache), 1)
            self.assertEqual(new_memory.param_cache[0].best_score, 0.85)

        finally:
            os.unlink(tmp_path)

    def test_stats_tracking(self):
        """Test statistics tracking"""
        # Add result
        self.memory.add_result(
            model_type=ModelType.GBM,
            temperature=1.0,
            alpha=0.5,
            best_params={},
            best_score=0.75,
            n_trials=5
        )

        # Cache hit
        self.memory.get_similar_configs(
            model_type=ModelType.GBM,
            temperature=1.0,
            alpha=0.5
        )

        # Cache miss
        self.memory.get_similar_configs(
            model_type=ModelType.XGB,
            temperature=5.0,
            alpha=0.1
        )

        stats = self.memory.get_stats()
        self.assertEqual(stats['cache_hits'], 1)
        self.assertEqual(stats['cache_misses'], 1)
        self.assertGreater(stats['hit_rate'], 0)


class TestIntelligentCache(unittest.TestCase):
    """Test suite for IntelligentCache"""

    def setUp(self):
        """Set up test fixtures"""
        self.cache = IntelligentCache(
            max_memory_gb=0.1,  # Small size for testing
            teacher_ratio=0.5,
            feature_ratio=0.3,
            attention_ratio=0.2
        )

        # Create mock model
        self.mock_model = Mock()
        self.mock_model.predict_proba = Mock(
            return_value=np.array([[0.3, 0.7], [0.6, 0.4]])
        )

        # Test data
        self.X = np.array([[1, 2, 3], [4, 5, 6]])
        self.y = np.array([1, 0])

    def test_initialization(self):
        """Test proper initialization"""
        self.assertIsNotNone(self.cache.teacher_cache)
        self.assertIsNotNone(self.cache.feature_cache)
        self.assertIsNotNone(self.cache.attention_cache)

    def test_cache_teacher_predictions(self):
        """Test caching teacher predictions"""
        # First call should compute
        preds1 = self.cache.cache_teacher_predictions(
            self.mock_model,
            self.X,
            temperature=1.0
        )

        # Second call should hit cache
        preds2 = self.cache.cache_teacher_predictions(
            self.mock_model,
            self.X,
            temperature=1.0
        )

        np.testing.assert_array_equal(preds1, preds2)
        self.assertEqual(self.cache.teacher_cache.hits, 1)

    def test_temperature_scaling(self):
        """Test temperature scaling of predictions"""
        probs = np.array([[0.3, 0.7], [0.6, 0.4]])

        # High temperature should smooth probabilities
        scaled_high = self.cache._apply_temperature(probs, temperature=5.0)
        self.assertLess(np.max(scaled_high), np.max(probs))

        # Low temperature should sharpen probabilities
        scaled_low = self.cache._apply_temperature(probs, temperature=0.5)
        self.assertGreater(np.max(scaled_low), np.max(probs))

    def test_get_or_compute(self):
        """Test get_or_compute functionality"""
        compute_count = 0

        def expensive_computation():
            nonlocal compute_count
            compute_count += 1
            return np.random.randn(10, 10)

        # First call
        result1 = self.cache.get_or_compute(
            key={'test': 'key'},
            compute_fn=expensive_computation,
            cache_type='teacher'
        )

        # Second call with same key
        result2 = self.cache.get_or_compute(
            key={'test': 'key'},
            compute_fn=expensive_computation,
            cache_type='teacher'
        )

        # Should only compute once
        self.assertEqual(compute_count, 1)
        np.testing.assert_array_equal(result1, result2)

    def test_cache_features(self):
        """Test caching intermediate features"""
        mock_model = Mock()
        mock_model.transform = Mock(return_value=np.array([[1, 2], [3, 4]]))

        features = self.cache.cache_features(mock_model, self.X)

        # Should call transform
        mock_model.transform.assert_called_once()

        # Second call should use cache
        features2 = self.cache.cache_features(mock_model, self.X)
        np.testing.assert_array_equal(features, features2)

        # Transform should still be called only once
        mock_model.transform.assert_called_once()

    def test_cache_attention_maps(self):
        """Test caching attention maps"""
        # Mock tree-based model
        mock_model = Mock()
        mock_model.feature_importances_ = np.array([0.5, 0.3, 0.2])

        attention = self.cache.cache_attention_maps(mock_model, self.X)

        # Should have correct shape
        self.assertEqual(attention.shape, self.X.shape)

        # Feature importances should be broadcast
        np.testing.assert_array_equal(attention[0], mock_model.feature_importances_)

    def test_cache_size_management(self):
        """Test cache size limits and eviction"""
        # Add items until cache is full
        for i in range(20):
            self.cache.get_or_compute(
                key={'index': i},
                compute_fn=lambda: np.random.randn(100, 100),
                cache_type='teacher'
            )

        # Cache should respect size limits
        stats = self.cache.teacher_cache.get_stats()
        self.assertLessEqual(
            stats['size_bytes'],
            self.cache.teacher_cache.max_size_bytes
        )

    def test_clear_all(self):
        """Test clearing all caches"""
        # Add some data to caches
        self.cache.cache_teacher_predictions(self.mock_model, self.X)

        # Clear all
        self.cache.clear_all()

        # Caches should be empty
        self.assertEqual(len(self.cache.teacher_cache.cache), 0)
        self.assertEqual(len(self.cache.feature_cache.cache), 0)
        self.assertEqual(len(self.cache.attention_cache.cache), 0)

    def test_get_stats(self):
        """Test getting comprehensive statistics"""
        # Perform some operations
        self.cache.cache_teacher_predictions(self.mock_model, self.X)
        self.cache.cache_teacher_predictions(self.mock_model, self.X)  # Hit

        stats = self.cache.get_stats()

        self.assertIn('teacher_cache', stats)
        self.assertIn('total_size_mb', stats)
        self.assertIn('memory_usage_percent', stats)

        # Should have recorded hits
        self.assertGreater(stats['teacher_cache']['hits'], 0)


if __name__ == '__main__':
    unittest.main()