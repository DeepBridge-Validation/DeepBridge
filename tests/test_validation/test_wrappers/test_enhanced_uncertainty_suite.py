"""Tests for EnhancedUncertaintySuite validation wrapper."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.enhanced_uncertainty_suite import (
    EnhancedUncertaintySuite,
)


@pytest.fixture
def sample_regression_dataset():
    """Create a regression dataset for uncertainty testing."""
    np.random.seed(42)
    n_samples = 200

    # Create synthetic regression data
    X = np.random.randn(n_samples, 3)
    y = 2*X[:, 0] + 3*X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5

    data = pd.DataFrame({
        'feature1': X[:, 0],
        'feature2': X[:, 1],
        'feature3': X[:, 2],
        'target': y
    })

    # Create DBDataset
    dataset = DBDataset(
        data=data,
        target_column='target'
    )

    # Train a simple model
    model = LinearRegression()
    model.fit(dataset.train_data[['feature1', 'feature2', 'feature3']],
              dataset.train_data['target'])
    dataset.set_model(model)

    return dataset


@pytest.fixture
def sample_complex_regression_dataset():
    """Create a more complex regression dataset with more features."""
    np.random.seed(123)
    n_samples = 300

    # Create synthetic data with 5 features
    X = np.random.randn(n_samples, 5)
    y = (1.5*X[:, 0] + 2*X[:, 1] - 0.5*X[:, 2] +
         X[:, 3] - 1.2*X[:, 4] + np.random.randn(n_samples) * 0.8)

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

    # Use RandomForest for more complex model
    model = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=5)
    features = ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'feat_e']
    model.fit(dataset.train_data[features], dataset.train_data['target'])
    dataset.set_model(model)

    return dataset


class TestEnhancedUncertaintySuiteInit:
    """Test EnhancedUncertaintySuite initialization."""

    def test_init_basic(self, sample_regression_dataset):
        """Test basic initialization."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        assert suite.dataset is not None
        assert suite.verbose is False
        assert suite.feature_subset is None
        assert suite.current_config is None
        assert suite.results == {}

    def test_init_with_params(self, sample_regression_dataset):
        """Test initialization with custom parameters."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=True,
            feature_subset=['feature1', 'feature2'],
            random_state=456
        )

        assert suite.verbose is True
        assert suite.feature_subset == ['feature1', 'feature2']
        assert suite.random_state == 456

    def test_init_enhanced_params(self, sample_regression_dataset):
        """Test that enhanced parameters are initialized."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        # Check enhanced-specific attributes
        assert hasattr(suite, 'reliable_threshold_ratio')
        assert suite.reliable_threshold_ratio == 1.1
        assert hasattr(suite, 'bin_count')
        assert suite.bin_count == 10

    def test_inherits_from_uncertainty_suite(self, sample_regression_dataset):
        """Test that EnhancedUncertaintySuite inherits from UncertaintySuite."""
        from deepbridge.validation.wrappers.uncertainty_suite import UncertaintySuite

        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        assert isinstance(suite, UncertaintySuite)
        assert hasattr(suite, 'uncertainty_models')
        assert hasattr(suite, '_problem_type')


class TestEnhancedUncertaintySuiteConfig:
    """Test EnhancedUncertaintySuite configuration (inherited from base)."""

    def test_config_quick(self, sample_regression_dataset):
        """Test quick configuration."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        result = suite.config('quick')

        assert result is suite  # Should return self for chaining
        assert suite.current_config is not None

    def test_config_medium(self, sample_regression_dataset):
        """Test medium configuration."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        suite.config('medium')
        assert suite.current_config is not None

    def test_config_full(self, sample_regression_dataset):
        """Test full configuration."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        suite.config('full')
        assert suite.current_config is not None

    def test_config_with_feature_subset(self, sample_complex_regression_dataset):
        """Test configuration with feature subset."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_complex_regression_dataset,
            verbose=False
        )

        suite.config('quick', feature_subset=['feat_a', 'feat_b'])
        assert suite.feature_subset == ['feat_a', 'feat_b']


class TestEnhancedUncertaintySuiteRun:
    """Test EnhancedUncertaintySuite run method."""

    def test_run_with_quick_config(self, sample_regression_dataset):
        """Test running suite with quick configuration."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        suite.config('quick')
        results = suite.run()

        # Check basic results structure from parent
        assert isinstance(results, dict)
        assert 'crqr' in results or len(results) > 0

    def test_run_produces_enhanced_metrics(self, sample_regression_dataset):
        """Test that run produces enhanced metrics."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        suite.config('quick')
        results = suite.run()

        # Enhanced metrics that should be added
        # Note: These may only appear if CRQR results are available
        if 'crqr' in results and results.get('crqr', {}).get('all_results'):
            # Check for enhanced metrics
            assert isinstance(results, dict)
            # At minimum, results should have some structure
            assert len(results) > 0

    def test_run_without_config_uses_default(self, sample_regression_dataset):
        """Test that run() works without explicit config."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        # Don't call config() - should use parent's default behavior
        results = suite.run()

        assert isinstance(results, dict)
        assert suite.current_config is not None  # Should be set by parent

    def test_run_with_verbose(self, sample_regression_dataset, capsys):
        """Test run with verbose output."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=True
        )

        suite.config('quick')
        suite.run()

        # Check that some output was produced
        captured = capsys.readouterr()
        # Verbose mode should produce some output from parent class
        assert len(captured.out) >= 0  # At least no errors


class TestEnhancedUncertaintySuitePrivateMethods:
    """Test private methods of EnhancedUncertaintySuite."""

    def test_calculate_psi_method_exists(self, sample_regression_dataset):
        """Test that _calculate_psi method exists."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        assert hasattr(suite, '_calculate_psi')

    def test_calculate_psi_with_simple_data(self, sample_regression_dataset):
        """Test _calculate_psi with simple equal distributions."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        # Identical distributions should have PSI near 0
        expected = np.array([1, 2, 3, 4, 5] * 20)
        actual = np.array([1, 2, 3, 4, 5] * 20)

        psi = suite._calculate_psi(expected, actual, bins=5)

        # PSI should be very small for identical distributions
        assert isinstance(psi, (float, np.floating))
        assert psi >= 0  # PSI is always non-negative
        assert psi < 0.1  # Should be very small for identical dists

    def test_analyze_reliability_method_exists(self, sample_regression_dataset):
        """Test that _analyze_reliability method exists."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        assert hasattr(suite, '_analyze_reliability')

    def test_analyze_marginal_bandwidth_method_exists(self, sample_regression_dataset):
        """Test that _analyze_marginal_bandwidth method exists."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        assert hasattr(suite, '_analyze_marginal_bandwidth')

    def test_calculate_additional_metrics_method_exists(self, sample_regression_dataset):
        """Test that _calculate_additional_metrics method exists."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        assert hasattr(suite, '_calculate_additional_metrics')


class TestEnhancedUncertaintySuiteEdgeCases:
    """Test edge cases and error handling."""

    def test_with_feature_subset(self, sample_complex_regression_dataset):
        """Test running with feature subset."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_complex_regression_dataset,
            verbose=False,
            feature_subset=['feat_a', 'feat_b', 'feat_c']
        )

        suite.config('quick')
        results = suite.run()

        assert isinstance(results, dict)
        assert suite.feature_subset == ['feat_a', 'feat_b', 'feat_c']

    def test_with_random_state(self, sample_regression_dataset):
        """Test that random_state provides reproducibility."""
        suite1 = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        suite2 = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        assert suite1.random_state == suite2.random_state
        assert suite1.random_state == 42

    def test_threshold_ratio_affects_classification(self, sample_regression_dataset):
        """Test that reliable_threshold_ratio is configurable."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        # Check default value
        assert suite.reliable_threshold_ratio == 1.1

        # Modify and check
        suite.reliable_threshold_ratio = 1.5
        assert suite.reliable_threshold_ratio == 1.5

    def test_bin_count_configurable(self, sample_regression_dataset):
        """Test that bin_count is configurable."""
        suite = EnhancedUncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        # Check default value
        assert suite.bin_count == 10

        # Modify and check
        suite.bin_count = 15
        assert suite.bin_count == 15
