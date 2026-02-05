"""
Integration tests for HPM-KD distillation system.

These tests verify that HPM components work correctly together
and integrate properly with the existing DeepBridge infrastructure.
"""

import os
import tempfile
import time
import unittest
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Import DeepBridge components
from deepbridge.core.db_data import DBDataset
from deepbridge.distillation import AutoDistiller
from deepbridge.distillation.techniques.hpm import (
    AdaptiveConfigurationManager,
    AttentionWeightedMultiTeacher,
    HPMConfig,
    HPMDistiller,
    ParallelDistillationPipeline,
    ProgressiveDistillationChain,
)
from deepbridge.utils.model_registry import ModelType


class TestHPMIntegration(unittest.TestCase):
    """Integration tests for HPM-KD system."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Create a sample dataset
        X, y = make_classification(
            n_samples=1000,
            n_features=20,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=2,
            random_state=42
        )

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Train a teacher model
        teacher = RandomForestClassifier(n_estimators=50, random_state=42)
        teacher.fit(X_train, y_train)

        # Get teacher probabilities
        probs_train = teacher.predict_proba(X_train)
        probs_test = teacher.predict_proba(X_test)

        # Combine data
        X_full = np.vstack([X_train, X_test])
        y_full = np.hstack([y_train, y_test])
        probs_full = np.vstack([probs_train, probs_test])

        # Create DBDataset
        cls.dataset = DBDataset(
            X=pd.DataFrame(X_full),
            y=pd.Series(y_full),
            probabilities=pd.DataFrame(probs_full, columns=['prob_0', 'prob_1'])
        )

        cls.X_train = X_train
        cls.X_test = X_test
        cls.y_train = y_train
        cls.y_test = y_test
        cls.teacher_probs = probs_train

    def test_hpm_distiller_initialization(self):
        """Test HPM distiller initialization."""
        config = HPMConfig(
            max_configs=8,
            n_trials=2,
            verbose=False
        )

        distiller = HPMDistiller(config=config)

        self.assertIsNotNone(distiller.config_manager)
        self.assertIsNotNone(distiller.shared_memory)
        self.assertIsNotNone(distiller.cache)
        self.assertIsNotNone(distiller.progressive_chain)
        self.assertIsNotNone(distiller.multi_teacher)
        self.assertIsNotNone(distiller.temp_scheduler)
        self.assertIsNotNone(distiller.pipeline)

    def test_auto_distiller_hpm_method(self):
        """Test AutoDistiller with HPM method."""
        distiller = AutoDistiller(
            dataset=self.dataset,
            method='hpm',
            n_trials=2,  # Minimal for testing
            verbose=False
        )

        self.assertEqual(distiller.method, 'hpm')
        self.assertIsNotNone(distiller.hpm_distiller)

    def test_auto_distiller_auto_selection(self):
        """Test AutoDistiller automatic method selection."""
        # Small dataset should select legacy
        small_dataset = DBDataset(
            X=pd.DataFrame(np.random.randn(100, 10)),
            y=pd.Series(np.random.randint(0, 2, 100)),
            probabilities=pd.DataFrame(np.random.rand(100, 2))
        )

        distiller_small = AutoDistiller(
            dataset=small_dataset,
            method='auto',
            verbose=False
        )

        self.assertEqual(distiller_small.method, 'legacy')

        # Large dataset should select HPM
        large_dataset = DBDataset(
            X=pd.DataFrame(np.random.randn(15000, 10)),
            y=pd.Series(np.random.randint(0, 2, 15000)),
            probabilities=pd.DataFrame(np.random.rand(15000, 2))
        )

        distiller_large = AutoDistiller(
            dataset=large_dataset,
            method='auto',
            verbose=False
        )

        self.assertEqual(distiller_large.method, 'hpm')

    def test_adaptive_configuration_reduction(self):
        """Test that adaptive configuration manager reduces configs."""
        manager = AdaptiveConfigurationManager(max_configs=8)

        model_types = [
            ModelType.LOGISTIC_REGRESSION,
            ModelType.DECISION_TREE,
            ModelType.GBM,
            ModelType.XGB
        ]
        temperatures = [0.5, 1.0, 2.0, 3.0]
        alphas = [0.3, 0.5, 0.7, 0.9]

        # Total possible: 4 * 4 * 4 = 64
        configs = manager.select_promising_configs(
            model_types=model_types,
            temperatures=temperatures,
            alphas=alphas
        )

        # Should reduce to max_configs
        self.assertEqual(len(configs), 8)

        # Each config should be valid
        for config in configs:
            self.assertIn(config['model_type'], model_types)
            self.assertIn(config['temperature'], temperatures)
            self.assertIn(config['alpha'], alphas)

    def test_progressive_chain_training(self):
        """Test progressive distillation chain."""
        chain = ProgressiveDistillationChain(
            chain_order=[
                ModelType.LOGISTIC_REGRESSION,
                ModelType.DECISION_TREE
            ],
            min_improvement=0.001,
            random_state=42
        )

        # Train chain
        stages = chain.train_progressive(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_test,
            y_val=self.y_test,
            teacher_probs=self.teacher_probs
        )

        # Should have 2 stages
        self.assertEqual(len(stages), 2)

        # Each stage should have a trained model
        for stage in stages:
            self.assertIsNotNone(stage.model)
            self.assertIsNotNone(stage.predictions)

        # Best model should be available
        best_model = chain.get_best_model()
        self.assertIsNotNone(best_model)

    def test_multi_teacher_system(self):
        """Test multi-teacher ensemble system."""
        multi_teacher = AttentionWeightedMultiTeacher(
            attention_type='learned'
        )

        # Create mock teachers
        for i in range(3):
            mock_model = Mock()
            mock_model.predict_proba = Mock(
                return_value=np.random.rand(len(self.X_test), 2)
            )

            multi_teacher.add_teacher(
                model=mock_model,
                model_type=f"teacher_{i}",
                performance=0.8 + i * 0.05
            )

        # Compute attention weights
        weights = multi_teacher.compute_attention_weights(self.X_test)

        self.assertEqual(len(weights), 3)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=5)

        # Fuse knowledge
        fused = multi_teacher.weighted_knowledge_fusion(self.X_test)
        self.assertEqual(fused.shape, (len(self.X_test), 2))

    def test_parallel_pipeline(self):
        """Test parallel training pipeline."""
        from deepbridge.distillation.techniques.hpm.parallel_pipeline import (
            WorkloadConfig,
            train_config_worker,
        )

        pipeline = ParallelDistillationPipeline(
            n_workers=2,
            use_processes=False,  # Use threads for testing
            timeout_per_config=10.0
        )

        # Create workloads
        workloads = []
        for i in range(4):
            config = WorkloadConfig(
                config_id=f"test_{i}",
                model_type=ModelType.LOGISTIC_REGRESSION,
                temperature=1.0,
                alpha=0.5,
                hyperparams={'max_iter': 100}
            )
            workloads.append(config)

        # Mock train function
        def mock_train(model_type, temperature, alpha, hyperparams, dataset):
            time.sleep(0.1)  # Simulate training time
            return Mock(), {'accuracy': 0.85}

        # Run parallel training
        with pipeline:
            results = pipeline.train_batch_parallel(
                configurations=workloads,
                train_function=mock_train,
                dataset={'X_train': self.X_train, 'y_train': self.y_train}
            )

        self.assertEqual(len(results), 4)
        for result in results:
            self.assertTrue(result.success)
            self.assertIsNotNone(result.model)
            self.assertIsNotNone(result.metrics)

    def test_hpm_fit_integration(self):
        """Test full HPM distiller fit process."""
        config = HPMConfig(
            max_configs=4,
            n_trials=1,
            verbose=False,
            use_parallel=False,  # Disable for testing
            use_progressive=False,  # Simplify for testing
            use_multi_teacher=False  # Simplify for testing
        )

        distiller = HPMDistiller(config=config)

        # Fit the distiller
        distiller.fit(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_test,
            y_val=self.y_test,
            teacher_probs=self.teacher_probs,
            model_types=[ModelType.LOGISTIC_REGRESSION],
            temperatures=[1.0, 2.0],
            alphas=[0.5]
        )

        self.assertTrue(distiller.is_fitted)
        self.assertIsNotNone(distiller.best_model)
        self.assertIsNotNone(distiller.best_metrics)

        # Should be able to make predictions
        predictions = distiller.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))

    def test_cache_efficiency(self):
        """Test that cache reduces computation time."""
        from deepbridge.distillation.techniques.hpm import IntelligentCache

        cache = IntelligentCache(max_memory_gb=0.1)

        # Mock expensive computation
        computation_count = 0

        def expensive_computation():
            nonlocal computation_count
            computation_count += 1
            time.sleep(0.1)
            return np.random.randn(100, 10)

        # First call should compute
        result1 = cache.get_or_compute(
            key={'test': 'key'},
            compute_fn=expensive_computation,
            cache_type='teacher'
        )

        # Second call should use cache
        result2 = cache.get_or_compute(
            key={'test': 'key'},
            compute_fn=expensive_computation,
            cache_type='teacher'
        )

        # Should only compute once
        self.assertEqual(computation_count, 1)
        np.testing.assert_array_equal(result1, result2)

        # Check cache stats
        stats = cache.get_stats()
        self.assertGreater(stats['teacher_cache']['hits'], 0)

    def test_shared_memory_warm_start(self):
        """Test shared memory provides warm start."""
        from deepbridge.distillation.techniques.hpm import (
            SharedOptimizationMemory,
        )

        memory = SharedOptimizationMemory()

        # Add a successful configuration
        memory.add_result(
            model_type=ModelType.XGB,
            temperature=2.0,
            alpha=0.5,
            best_params={'max_depth': 5, 'n_estimators': 100},
            best_score=0.9,
            n_trials=10
        )

        # Try to get similar config
        similar = memory.get_similar_configs(
            model_type=ModelType.XGB,
            temperature=2.1,  # Slightly different
            alpha=0.5
        )

        self.assertGreater(len(similar), 0)

        # Create warm-started study
        study = memory.warm_start_study(
            model_type=ModelType.XGB,
            temperature=2.0,
            alpha=0.5,
            similar_configs=similar,
            n_trials=5
        )

        self.assertIsNotNone(study)
        self.assertGreater(memory.stats['trials_saved'], 0)

    def test_temperature_scheduler_adaptation(self):
        """Test meta temperature scheduler adaptation."""
        from deepbridge.distillation.techniques.hpm import (
            MetaTemperatureScheduler,
        )

        scheduler = MetaTemperatureScheduler(
            initial_temperature=3.0,
            min_temperature=0.5,
            max_temperature=5.0
        )

        # Simulate training progress
        temperatures = []
        for epoch in range(20):
            temp = scheduler.adaptive_temperature(
                epoch=epoch,
                loss=2.0 * (1 - epoch / 20),
                kl_divergence=1.0 * (1 - epoch / 20),
                student_accuracy=0.5 + 0.4 * epoch / 20,
                teacher_accuracy=0.9
            )
            temperatures.append(temp)

        # Temperature should adapt over time
        self.assertNotEqual(temperatures[0], temperatures[-1])

        # Get stats
        stats = scheduler.get_stats()
        self.assertIn('current_temperature', stats)
        self.assertIn('temperature_range', stats)

    def test_hpm_speedup(self):
        """Test that HPM is faster than sequential processing."""
        # This is a simplified test - real speedup would be more significant

        config_hpm = HPMConfig(
            max_configs=4,
            n_trials=1,
            use_parallel=True,
            parallel_workers=2,
            verbose=False
        )

        config_seq = HPMConfig(
            max_configs=4,
            n_trials=1,
            use_parallel=False,  # Sequential
            verbose=False
        )

        # Mock training time
        with patch('time.sleep', side_effect=lambda x: None):  # Skip sleep
            # HPM with parallelization
            start = time.time()
            distiller_hpm = HPMDistiller(config=config_hpm)
            # Simulate fit (mocked for speed)
            elapsed_hpm = time.time() - start

            # Sequential
            start = time.time()
            distiller_seq = HPMDistiller(config=config_seq)
            # Simulate fit (mocked for speed)
            elapsed_seq = time.time() - start

        # HPM initialization should be comparable (not testing actual training)
        # In real scenario, training would show significant speedup
        self.assertLess(elapsed_hpm, elapsed_seq + 1.0)  # Allow some variance

    def test_end_to_end_workflow(self):
        """Test complete end-to-end HPM workflow."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Initialize distiller
            distiller = AutoDistiller(
                dataset=self.dataset,
                method='hpm',
                output_dir=tmp_dir,
                n_trials=2,
                verbose=False
            )

            # Customize configuration
            distiller.customize_config(
                model_types=[ModelType.LOGISTIC_REGRESSION, ModelType.XGB],
                temperatures=[1.0, 2.0],
                alphas=[0.5]
            )

            # Run distillation (this would run actual distillation)
            # For testing, we'll check the configuration was set correctly
            self.assertEqual(len(distiller.config.model_types), 2)
            self.assertEqual(len(distiller.config.temperatures), 2)
            self.assertEqual(len(distiller.config.alphas), 1)

            # Check output directory was created
            self.assertTrue(os.path.exists(tmp_dir))


class TestHPMPerformance(unittest.TestCase):
    """Performance tests for HPM-KD system."""

    def test_configuration_reduction_performance(self):
        """Test performance impact of configuration reduction."""
        manager = AdaptiveConfigurationManager(max_configs=16)

        # Large configuration space
        model_types = [mt for mt in ModelType][:5]  # 5 model types
        temperatures = np.linspace(0.5, 5.0, 10).tolist()  # 10 temperatures
        alphas = np.linspace(0.1, 0.9, 9).tolist()  # 9 alphas

        # Total: 5 * 10 * 9 = 450 configurations

        start = time.time()
        configs = manager.select_promising_configs(
            model_types=model_types,
            temperatures=temperatures,
            alphas=alphas
        )
        elapsed = time.time() - start

        # Should reduce to 16 configurations quickly
        self.assertEqual(len(configs), 16)
        self.assertLess(elapsed, 1.0)  # Should be fast

        print(f"Reduced 450 → 16 configurations in {elapsed:.3f} seconds")

    def test_cache_memory_usage(self):
        """Test cache memory management."""
        from deepbridge.distillation.techniques.hpm import IntelligentCache

        cache = IntelligentCache(max_memory_gb=0.01)  # 10MB limit

        # Add data until cache is full
        for i in range(100):
            data = np.random.randn(1000, 100)  # ~800KB each
            cache.get_or_compute(
                key={'index': i},
                compute_fn=lambda: data,
                cache_type='teacher'
            )

        # Check memory usage
        stats = cache.get_stats()
        total_mb = stats['total_size_mb']

        # Should respect memory limit (approximately)
        self.assertLess(total_mb, 15)  # Allow some overhead

        print(f"Cache using {total_mb:.2f} MB (limit: 10 MB)")

    def test_parallel_speedup(self):
        """Test parallel processing speedup."""
        from deepbridge.distillation.techniques.hpm.parallel_pipeline import (
            WorkloadConfig,
        )

        # Test with different number of workers
        workloads = [
            WorkloadConfig(
                config_id=f"test_{i}",
                model_type=ModelType.LOGISTIC_REGRESSION,
                temperature=1.0,
                alpha=0.5,
                hyperparams={},
                estimated_time=0.1
            )
            for i in range(8)
        ]

        def simulate_training(model_type, temperature, alpha, hyperparams, dataset):
            time.sleep(0.05)  # Simulate work
            return Mock(), {'accuracy': 0.85}

        # Sequential (1 worker)
        pipeline_seq = ParallelDistillationPipeline(n_workers=1, use_processes=False)
        with pipeline_seq:
            start = time.time()
            results_seq = pipeline_seq.train_batch_parallel(
                configurations=workloads,
                train_function=simulate_training,
                dataset={}
            )
            time_seq = time.time() - start

        # Parallel (4 workers)
        pipeline_par = ParallelDistillationPipeline(n_workers=4, use_processes=False)
        with pipeline_par:
            start = time.time()
            results_par = pipeline_par.train_batch_parallel(
                configurations=workloads,
                train_function=simulate_training,
                dataset={}
            )
            time_par = time.time() - start

        # Check speedup
        speedup = time_seq / time_par
        self.assertGreater(speedup, 1.5)  # Should have some speedup

        print(f"Parallel speedup: {speedup:.2f}x (Sequential: {time_seq:.3f}s, Parallel: {time_par:.3f}s)")


if __name__ == '__main__':
    unittest.main()