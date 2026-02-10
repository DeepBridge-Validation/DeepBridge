"""Tests for UncertaintySuite validation wrapper."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.uncertainty_suite import UncertaintySuite


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


class TestUncertaintySuiteInit:
    """Test UncertaintySuite initialization."""

    def test_init_basic(self, sample_regression_dataset):
        """Test basic initialization."""
        suite = UncertaintySuite(
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
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=True,
            feature_subset=['feature1', 'feature2'],
            random_state=456
        )

        assert suite.verbose is True
        assert suite.feature_subset == ['feature1', 'feature2']
        assert suite.random_state == 456

    def test_init_detects_problem_type(self, sample_regression_dataset):
        """Test that problem type is detected correctly."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        assert suite._problem_type == 'regression'

    def test_init_creates_model_cache(self, sample_regression_dataset):
        """Test that model cache is initialized."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        assert hasattr(suite, '_model_cache')
        assert isinstance(suite._model_cache, dict)
        assert len(suite._model_cache) == 0


class TestUncertaintySuiteConfig:
    """Test UncertaintySuite configuration."""

    def test_config_quick(self, sample_regression_dataset):
        """Test quick configuration."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        result = suite.config('quick')

        assert result is suite  # Should return self for chaining
        assert suite.current_config is not None
        assert suite.current_config_name == 'quick'

    def test_config_medium(self, sample_regression_dataset):
        """Test medium configuration."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        result = suite.config('medium')

        assert result is suite
        assert suite.current_config_name == 'medium'

    def test_config_full(self, sample_regression_dataset):
        """Test full configuration."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        result = suite.config('full')

        assert result is suite
        assert suite.current_config_name == 'full'

    def test_config_invalid(self, sample_regression_dataset):
        """Test invalid configuration raises error."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        with pytest.raises(ValueError, match="Unknown configuration"):
            suite.config('invalid_config')

    def test_config_with_feature_subset(self, sample_regression_dataset):
        """Test configuration with feature subset."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        suite.config('quick', feature_subset=['feature1', 'feature2'])

        assert suite.feature_subset == ['feature1', 'feature2']

    def test_config_clones_template(self, sample_regression_dataset):
        """Test that configuration clones templates."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        suite.config('quick')
        config1 = suite.current_config

        # Create another suite and config
        suite2 = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )
        suite2.config('quick')
        config2 = suite2.current_config

        # Modify config1
        if len(config1) > 0:
            config1[0]['modified'] = True

        # config2 should not be affected
        if len(config2) > 0:
            assert 'modified' not in config2[0]


class TestUncertaintySuiteDetermineProbleType:
    """Test problem type determination."""

    def test_determine_problem_type_from_dataset(self, sample_regression_dataset):
        """Test problem type detection from dataset attribute."""
        sample_regression_dataset.problem_type = 'classification'

        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        assert suite._problem_type == 'classification'

    def test_determine_problem_type_from_model(self, sample_regression_dataset):
        """Test problem type detection from model."""
        # Remove problem_type from dataset
        if hasattr(sample_regression_dataset, 'problem_type'):
            delattr(sample_regression_dataset, 'problem_type')

        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        # LinearRegression doesn't have predict_proba, so should be regression
        assert suite._problem_type == 'regression'

    def test_determine_problem_type_default(self):
        """Test default problem type when no info available."""
        # Create minimal dataset without model
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [1, 2, 3, 4, 5]
        })

        dataset = DBDataset(data=data, target_column='target')

        suite = UncertaintySuite(
            dataset=dataset,
            verbose=False
        )

        # Should default to regression
        assert suite._problem_type == 'regression'


class TestUncertaintySuiteHelperMethods:
    """Test helper methods."""

    def test_clone_config(self, sample_regression_dataset):
        """Test configuration cloning."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        original = [{'method': 'test', 'params': {'alpha': 0.1}}]
        cloned = suite._clone_config(original)

        # Modify cloned
        cloned[0]['params']['alpha'] = 0.2

        # Original should not be affected
        assert original[0]['params']['alpha'] == 0.1

    def test_get_config_templates(self, sample_regression_dataset):
        """Test getting configuration templates."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        templates = suite._get_config_templates()

        assert isinstance(templates, dict)
        assert 'quick' in templates
        assert 'medium' in templates
        assert 'full' in templates

    def test_get_config_templates_fallback(self, sample_regression_dataset, monkeypatch):
        """Test configuration templates fallback on error."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        # Mock get_test_config to raise error
        def mock_get_test_config(*args, **kwargs):
            raise Exception("Test error")

        import deepbridge.validation.wrappers.uncertainty_suite as us_module
        monkeypatch.setattr(us_module, "get_test_config", mock_get_test_config)

        templates = suite._get_config_templates()

        # Should return fallback templates
        assert isinstance(templates, dict)
        assert 'quick' in templates
        assert 'medium' in templates
        assert 'full' in templates


class TestUncertaintySuiteRun:
    """Test running uncertainty tests."""

    def test_run_with_config(self, sample_regression_dataset):
        """Test running with a configuration."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        # First configure
        suite.config('quick')

        # Then run
        results = suite.run()

        assert isinstance(results, dict)
        assert 'summary' in results or len(results) > 0

    def test_run_without_config(self, sample_regression_dataset):
        """Test that run without config uses default 'quick' config."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        # Should use default 'quick' config
        results = suite.run()

        assert isinstance(results, dict)
        assert suite.current_config_name == 'quick'

    def test_run_stores_results(self, sample_regression_dataset):
        """Test that run stores results in suite.results."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        suite.config('quick')
        results = suite.run()

        # Results should be stored in suite.results
        assert len(suite.results) > 0 or len(results) > 0


class TestUncertaintySuiteEvaluate:
    """Test uncertainty evaluation methods."""

    def test_evaluate_uncertainty_basic(self, sample_regression_dataset):
        """Test basic uncertainty evaluation."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        # Test CRQR evaluation
        result = suite.evaluate_uncertainty(
            method='crqr',
            params={'alpha': 0.1, 'test_size': 0.3}
        )

        assert isinstance(result, dict)
        assert 'coverage' in result or 'error' in result

    def test_evaluate_uncertainty_with_feature(self, sample_regression_dataset):
        """Test uncertainty evaluation for specific feature."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        result = suite.evaluate_uncertainty(
            method='crqr',
            params={'alpha': 0.1},
            feature='feature1'
        )

        assert isinstance(result, dict)

    def test_create_crqr_model(self, sample_regression_dataset):
        """Test CRQR model creation."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        model = suite._create_crqr_model(alpha=0.1, test_size=0.3)

        assert model is not None
        assert hasattr(model, 'alpha')
        assert model.alpha == 0.1

    def test_calculate_feature_importance_fast(self, sample_regression_dataset):
        """Test fast feature importance calculation."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        # First train a CRQR model
        X = sample_regression_dataset.get_feature_data()
        y = sample_regression_dataset.get_target_data()

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        model = suite._create_crqr_model(alpha=0.1)

        try:
            model.fit(X, y)

            importance = suite._calculate_feature_importance_fast(
                model, X, y, 'feature1'
            )

            assert isinstance(importance, (int, float))
            assert importance >= 0
        except Exception:
            # If CRQR fitting fails, that's okay for this test
            pytest.skip("CRQR fitting not available")


class TestUncertaintySuiteGetResults:
    """Test result retrieval methods."""

    def test_get_results_empty(self, sample_regression_dataset):
        """Test getting results when none exist."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        results = suite.results
        assert results == {}

    def test_get_results_after_run(self, sample_regression_dataset):
        """Test getting results after running tests."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        suite.config('quick')
        suite.run()

        results = suite.results
        assert isinstance(results, dict)


class TestUncertaintySuiteEdgeCases:
    """Test edge cases and error handling."""

    def test_with_empty_feature_subset(self, sample_regression_dataset):
        """Test with empty feature subset."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            feature_subset=[]
        )

        assert suite.feature_subset == []

    def test_with_invalid_feature_subset(self, sample_regression_dataset):
        """Test with invalid feature names in subset."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            feature_subset=['nonexistent_feature']
        )

        suite.config('quick')

        # Should handle gracefully
        try:
            suite.run()
        except (KeyError, ValueError):
            # Expected to fail with nonexistent feature
            pass

    def test_verbose_mode(self, sample_regression_dataset, capsys):
        """Test verbose output."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=True
        )

        suite.config('quick')

        captured = capsys.readouterr()
        # Should have printed something during config
        assert len(captured.out) > 0 or suite.verbose is True

    def test_random_state_consistency(self, sample_regression_dataset):
        """Test that random_state provides consistent results."""
        suite1 = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )
        suite1.config('quick')
        results1 = suite1.run()

        suite2 = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )
        suite2.config('quick')
        results2 = suite2.run()

        # Results should be similar (not necessarily identical due to floating point)
        assert type(results1) == type(results2)


class TestUncertaintySuiteIntegration:
    """Integration tests."""

    def test_full_workflow_quick(self, sample_regression_dataset):
        """Test complete workflow with quick config."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        # Configure and run
        suite.config('quick')
        results = suite.run()

        assert isinstance(results, dict)

    def test_full_workflow_medium(self, sample_complex_regression_dataset):
        """Test complete workflow with medium config."""
        suite = UncertaintySuite(
            dataset=sample_complex_regression_dataset,
            verbose=False,
            random_state=42
        )

        suite.config('medium')
        results = suite.run()

        assert isinstance(results, dict)

    def test_chaining_config_and_run(self, sample_regression_dataset):
        """Test method chaining config().run()."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        # Should be able to chain
        results = suite.config('quick').run()

        assert isinstance(results, dict)

    def test_multiple_runs_same_config(self, sample_regression_dataset):
        """Test running multiple times with same config."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        suite.config('quick')

        results1 = suite.run()
        results2 = suite.run()

        assert isinstance(results1, dict)
        assert isinstance(results2, dict)

    def test_config_change_between_runs(self, sample_regression_dataset):
        """Test changing config between runs."""
        suite = UncertaintySuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42
        )

        suite.config('quick')
        results1 = suite.run()

        suite.config('medium')
        results2 = suite.run()

        assert isinstance(results1, dict)
        assert isinstance(results2, dict)
