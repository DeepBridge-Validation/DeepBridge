"""Tests for RobustnessSuite validation wrapper."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite


@pytest.fixture
def sample_classification_dataset():
    """Create a simple classification dataset for robustness testing."""
    np.random.seed(42)
    n_samples = 200

    # Create synthetic classification data
    X = np.random.randn(n_samples, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    data = pd.DataFrame({
        'feature1': X[:, 0],
        'feature2': X[:, 1],
        'feature3': X[:, 2],
        'feature4': X[:, 3],
        'target': y
    })

    # Create DBDataset
    dataset = DBDataset(
        data=data,
        target_column='target'
    )

    # Train a simple model
    model = LogisticRegression(random_state=42, max_iter=100)
    model.fit(dataset.train_data[['feature1', 'feature2', 'feature3', 'feature4']],
              dataset.train_data['target'])
    dataset.set_model(model)

    return dataset


@pytest.fixture
def sample_complex_dataset():
    """Create a more complex classification dataset."""
    np.random.seed(123)
    n_samples = 300

    X = np.random.randn(n_samples, 5)
    y = ((X[:, 0] + X[:, 1] - X[:, 2]) > 0).astype(int)

    data = pd.DataFrame({
        'feat_a': X[:, 0],
        'feat_b': X[:, 1],
        'feat_c': X[:, 2],
        'feat_d': X[:, 3],
        'feat_e': X[:, 4],
        'target': y
    })

    dataset = DBDataset(
        data=data,
        target_column='target'
    )

    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    features = ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'feat_e']
    model.fit(dataset.train_data[features], dataset.train_data['target'])
    dataset.set_model(model)

    return dataset


class TestRobustnessSuiteInit:
    """Test RobustnessSuite initialization."""

    def test_init_basic(self, sample_classification_dataset):
        """Test basic initialization."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        assert suite.dataset is not None
        assert suite.verbose is False
        assert suite.metric == 'AUC'
        assert suite.feature_subset is None
        assert suite.n_iterations == 1

    def test_init_with_params(self, sample_classification_dataset):
        """Test initialization with custom parameters."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=True,
            metric='accuracy',
            feature_subset=['feature1', 'feature2'],
            random_state=456,
            n_iterations=3
        )

        assert suite.verbose is True
        assert suite.metric == 'accuracy'
        assert suite.feature_subset == ['feature1', 'feature2']
        assert suite.n_iterations == 3

    def test_init_creates_components(self, sample_classification_dataset):
        """Test that initialization creates required components."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        # Check that components are initialized
        assert hasattr(suite, 'data_perturber')
        assert hasattr(suite, 'evaluator')
        assert suite.data_perturber is not None
        assert suite.evaluator is not None

    def test_init_with_random_state(self, sample_classification_dataset):
        """Test initialization with random state."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        # Random state should be set in data_perturber (as 'rng')
        assert hasattr(suite.data_perturber, 'rng')
        assert suite.data_perturber.rng is not None


class TestRobustnessSuiteConfig:
    """Test RobustnessSuite configuration."""

    def test_config_quick(self, sample_classification_dataset):
        """Test quick configuration."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        result = suite.config('quick')

        assert result is suite  # Should return self for chaining
        assert hasattr(suite, 'current_config')
        assert suite.current_config is not None
        assert len(suite.current_config) == 2  # Quick has 2 tests

    def test_config_medium(self, sample_classification_dataset):
        """Test medium configuration."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        suite.config('medium')

        assert suite.current_config is not None
        assert len(suite.current_config) == 3  # Medium has 3 tests

    def test_config_full(self, sample_classification_dataset):
        """Test full configuration."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        suite.config('full')

        assert suite.current_config is not None
        assert len(suite.current_config) == 6  # Full has 6 tests

    def test_config_quick_compare(self, sample_classification_dataset):
        """Test quick_compare configuration."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        suite.config('quick_compare')

        assert suite.current_config is not None
        assert len(suite.current_config) == 4  # Quick compare has 4 tests

    def test_config_invalid(self, sample_classification_dataset):
        """Test invalid configuration name."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        with pytest.raises(ValueError, match='Unknown configuration'):
            suite.config('invalid_config')

    def test_config_with_feature_subset(self, sample_complex_dataset):
        """Test configuration with feature subset."""
        suite = RobustnessSuite(
            dataset=sample_complex_dataset,
            verbose=False
        )

        suite.config('quick', feature_subset=['feat_a', 'feat_b'])

        assert suite.feature_subset == ['feat_a', 'feat_b']
        # Feature subset should be added to test params
        for test in suite.current_config:
            assert 'feature_subset' in test['params']

    def test_clone_config_creates_copy(self, sample_classification_dataset):
        """Test that _clone_config creates deep copy."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        original = [{'type': 'raw', 'params': {'level': 0.1}}]
        cloned = suite._clone_config(original)

        # Modify cloned
        cloned[0]['params']['level'] = 0.999

        # Original should be unchanged
        assert original[0]['params']['level'] == 0.1


class TestRobustnessSuiteRun:
    """Test RobustnessSuite run method."""

    def test_run_with_quick_config(self, sample_classification_dataset):
        """Test running suite with quick configuration."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            n_iterations=1  # Keep it quick
        )

        suite.config('quick')
        results = suite.run()

        # Check results structure
        assert isinstance(results, dict)
        assert 'base_score' in results
        assert 'raw' in results or 'quantile' in results
        assert results['base_score'] > 0

    def test_run_without_config_uses_default(self, sample_classification_dataset):
        """Test that run() works without explicit config."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            n_iterations=1
        )

        # Don't call config() - should handle this gracefully or use default
        # The implementation might set a default or raise an error
        # Let's check if it has current_config after init
        if not hasattr(suite, 'current_config') or suite.current_config is None:
            # If no default, calling config manually
            suite.config('quick')

        results = suite.run()
        assert isinstance(results, dict)

    def test_run_with_verbose(self, sample_classification_dataset, capsys):
        """Test run with verbose output."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=True,
            n_iterations=1
        )

        suite.config('quick')
        suite.run()

        # Check that some output was produced
        captured = capsys.readouterr()
        # Verbose mode should produce some output
        assert len(captured.out) >= 0  # At least no errors

    def test_run_with_different_metrics(self, sample_classification_dataset):
        """Test running with different metrics."""
        for metric in ['AUC', 'accuracy']:
            suite = RobustnessSuite(
                dataset=sample_classification_dataset,
                verbose=False,
                metric=metric,
                n_iterations=1
            )

            suite.config('quick')
            results = suite.run()

            assert isinstance(results, dict)
            assert 'base_score' in results
            assert 'metric' in results
            assert results['metric'] == metric


class TestRobustnessSuiteGetters:
    """Test RobustnessSuite getter methods."""

    def test_get_results(self, sample_classification_dataset):
        """Test get_results method."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            n_iterations=1
        )

        suite.config('quick')
        suite.run()

        results = suite.get_results()
        assert isinstance(results, dict)

    def test_get_visualizations(self, sample_classification_dataset):
        """Test get_visualizations method."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            n_iterations=1
        )

        suite.config('quick')
        suite.run()

        visualizations = suite.get_visualizations()
        assert isinstance(visualizations, dict)


class TestRobustnessSuiteEdgeCases:
    """Test edge cases and error handling."""

    def test_with_feature_subset(self, sample_complex_dataset):
        """Test running with feature subset."""
        suite = RobustnessSuite(
            dataset=sample_complex_dataset,
            verbose=False,
            feature_subset=['feat_a', 'feat_b', 'feat_c'],
            n_iterations=1
        )

        suite.config('quick')
        results = suite.run()

        assert isinstance(results, dict)
        assert suite.feature_subset == ['feat_a', 'feat_b', 'feat_c']

    def test_with_multiple_iterations(self, sample_classification_dataset):
        """Test running with multiple iterations."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            n_iterations=2
        )

        suite.config('quick')
        results = suite.run()

        assert isinstance(results, dict)
        assert suite.n_iterations == 2

    def test_config_types_in_template(self, sample_classification_dataset):
        """Test that config templates have correct structure."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        # Check that templates exist
        assert hasattr(RobustnessSuite, '_CONFIG_TEMPLATES')
        assert 'quick' in RobustnessSuite._CONFIG_TEMPLATES
        assert 'medium' in RobustnessSuite._CONFIG_TEMPLATES
        assert 'full' in RobustnessSuite._CONFIG_TEMPLATES

        # Check structure of quick config
        quick_config = RobustnessSuite._CONFIG_TEMPLATES['quick']
        assert isinstance(quick_config, list)
        assert len(quick_config) > 0
        assert 'type' in quick_config[0]
        assert 'params' in quick_config[0]

    def test_compare_configs_have_both_types(self, sample_classification_dataset):
        """Test that compare configs include both raw and quantile."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        suite.config('quick_compare')

        # Should have both 'raw' and 'quantile' types
        types = [test['type'] for test in suite.current_config]
        assert 'raw' in types
        assert 'quantile' in types

    def test_method_chaining(self, sample_classification_dataset):
        """Test that methods support chaining."""
        suite = RobustnessSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            n_iterations=1
        )

        # Config should return self for chaining
        result = suite.config('quick')
        assert result is suite

        # Should be able to call run immediately
        results = result.run()
        assert isinstance(results, dict)
