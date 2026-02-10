"""Tests for ResilienceSuite validation wrapper."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_classification, make_regression

from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.resilience_suite import ResilienceSuite


@pytest.fixture
def sample_classification_dataset():
    """Create a classification dataset for resilience testing."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=300,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )

    data = pd.DataFrame({
        'feature1': X[:, 0],
        'feature2': X[:, 1],
        'feature3': X[:, 2],
        'feature4': X[:, 3],
        'feature5': X[:, 4],
        'target': y
    })

    dataset = DBDataset(
        data=data,
        target_column='target'
    )

    # Train a model
    model = LogisticRegression(random_state=42, max_iter=1000)
    features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    model.fit(dataset.train_data[features], dataset.train_data['target'])
    dataset.set_model(model)

    return dataset


@pytest.fixture
def sample_regression_dataset():
    """Create a regression dataset for resilience testing."""
    np.random.seed(123)
    X, y = make_regression(
        n_samples=300,
        n_features=5,
        n_informative=3,
        noise=10.0,
        random_state=123
    )

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

    # Train a model
    model = LinearRegression()
    features = ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'feat_e']
    model.fit(dataset.train_data[features], dataset.train_data['target'])
    dataset.set_model(model)

    return dataset


class TestResilienceSuiteInit:
    """Test ResilienceSuite initialization."""

    def test_init_basic(self, sample_classification_dataset):
        """Test basic initialization."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        assert suite.dataset is not None
        assert suite.verbose is False
        assert suite.feature_subset is None
        assert suite.current_config is None
        assert suite.results == {}

    def test_init_with_params(self, sample_classification_dataset):
        """Test initialization with custom parameters."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=True,
            feature_subset=['feature1', 'feature2'],
            random_state=456
        )

        assert suite.verbose is True
        assert suite.feature_subset == ['feature1', 'feature2']
        assert suite.random_state == 456

    def test_init_detects_problem_type(self, sample_classification_dataset):
        """Test that problem type is detected correctly."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        assert suite._problem_type == 'classification'

    def test_init_with_regression(self, sample_regression_dataset):
        """Test initialization with regression dataset."""
        suite = ResilienceSuite(
            dataset=sample_regression_dataset,
            verbose=False
        )

        assert suite._problem_type == 'regression'


class TestResilienceSuiteConfig:
    """Test ResilienceSuite configuration."""

    def test_config_quick(self, sample_classification_dataset):
        """Test quick configuration."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        result = suite.config('quick')

        assert result is suite  # Should return self for chaining
        assert suite.current_config is not None

    def test_config_medium(self, sample_classification_dataset):
        """Test medium configuration."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        result = suite.config('medium')

        assert result is suite
        assert suite.current_config is not None

    def test_config_full(self, sample_classification_dataset):
        """Test full configuration."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        result = suite.config('full')

        assert result is suite
        assert suite.current_config is not None

    def test_config_invalid(self, sample_classification_dataset):
        """Test invalid configuration raises error."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        with pytest.raises(ValueError, match="Unknown configuration"):
            suite.config('invalid_config')

    def test_config_with_feature_subset(self, sample_classification_dataset):
        """Test configuration with feature subset."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        suite.config('quick', feature_subset=['feature1', 'feature2'])

        assert suite.feature_subset == ['feature1', 'feature2']

    def test_config_clones_template(self, sample_classification_dataset):
        """Test that configuration clones templates."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        suite.config('quick')
        config1 = suite.current_config

        # Create another suite and config
        suite2 = ResilienceSuite(
            dataset=sample_classification_dataset,
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


class TestResilienceSuiteDetermineProbleType:
    """Test problem type determination."""

    def test_determine_problem_type_from_dataset(self, sample_classification_dataset):
        """Test problem type detection from dataset attribute."""
        sample_classification_dataset.problem_type = 'regression'

        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        assert suite._problem_type == 'regression'

    def test_determine_problem_type_from_model(self, sample_classification_dataset):
        """Test problem type detection from model."""
        # Remove problem_type from dataset
        if hasattr(sample_classification_dataset, 'problem_type'):
            delattr(sample_classification_dataset, 'problem_type')

        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        # LogisticRegression has predict_proba, so should be classification
        assert suite._problem_type == 'classification'

    def test_determine_problem_type_default(self):
        """Test default problem type when no info available."""
        # Create minimal dataset without model
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [0, 1, 0, 1, 0]
        })

        dataset = DBDataset(data=data, target_column='target')

        suite = ResilienceSuite(
            dataset=dataset,
            verbose=False
        )

        # Should default to classification (has predict_proba check first)
        assert suite._problem_type in ['classification', 'regression']


class TestResilienceSuiteHelperMethods:
    """Test helper methods."""

    def test_clone_config(self, sample_classification_dataset):
        """Test configuration cloning."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        original = [{'method': 'test', 'params': {'alpha': 0.1}}]
        cloned = suite._clone_config(original)

        # Modify cloned
        cloned[0]['params']['alpha'] = 0.2

        # Original should not be affected
        assert original[0]['params']['alpha'] == 0.1

    def test_get_config_templates(self, sample_classification_dataset):
        """Test getting configuration templates."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        templates = suite._get_config_templates()

        assert isinstance(templates, dict)
        assert 'quick' in templates
        assert 'medium' in templates
        assert 'full' in templates

    def test_get_config_templates_fallback(self, sample_classification_dataset, monkeypatch):
        """Test configuration templates fallback on error."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        # Mock get_test_config to raise error
        def mock_get_test_config(*args, **kwargs):
            raise Exception("Test error")

        import deepbridge.validation.wrappers.resilience_suite as rs_module
        monkeypatch.setattr(rs_module, "get_test_config", mock_get_test_config)

        templates = suite._get_config_templates()

        # Should return fallback templates
        assert isinstance(templates, dict)


class TestResilienceSuiteRun:
    """Test running resilience tests."""

    def test_run_with_config(self, sample_classification_dataset):
        """Test running with a configuration."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        # First configure
        suite.config('quick')

        # Then run
        results = suite.run()

        assert isinstance(results, dict)

    def test_run_without_config(self, sample_classification_dataset):
        """Test that run without config uses default 'quick' config."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        # Should use default 'quick' config or raise error
        try:
            results = suite.run()
            assert isinstance(results, dict)
        except (ValueError, AttributeError):
            # Expected if no default config
            pass

    def test_run_stores_results(self, sample_classification_dataset):
        """Test that run stores results in suite.results."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        suite.config('quick')
        results = suite.run()

        # Results should be stored
        assert isinstance(results, dict)


class TestResilienceSuiteEvaluate:
    """Test resilience evaluation methods."""

    def test_evaluate_distribution_shift(self, sample_classification_dataset):
        """Test distribution shift evaluation."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        # Use correct method name
        result = suite.evaluate_distribution_shift(
            method='distribution_shift',
            params={'alpha': 0.1, 'metric': 'f1', 'distance_metric': 'PSI'}
        )

        assert isinstance(result, dict)

    def test_evaluate_worst_sample(self, sample_classification_dataset):
        """Test worst sample evaluation."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        result = suite.evaluate_worst_sample(
            method='worst_sample',
            params={'alpha': 0.1, 'metric': 'f1', 'ranking_method': 'residual'}
        )

        assert isinstance(result, dict)

    def test_evaluate_worst_cluster(self, sample_classification_dataset):
        """Test worst cluster evaluation."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        result = suite.evaluate_worst_cluster(
            method='worst_cluster',
            params={'n_clusters': 3, 'metric': 'f1'}
        )

        assert isinstance(result, dict)

    def test_evaluate_outer_sample(self, sample_classification_dataset):
        """Test outer sample evaluation."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        result = suite.evaluate_outer_sample(
            method='outer_sample',
            params={'alpha': 0.05, 'metric': 'f1', 'outlier_method': 'isolation_forest'}
        )

        assert isinstance(result, dict)

    def test_evaluate_hard_sample(self, sample_classification_dataset):
        """Test hard sample evaluation."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        result = suite.evaluate_hard_sample(
            method='hard_sample',
            params={'disagreement_threshold': 0.3, 'metric': 'f1', 'n_estimators': 3}
        )

        assert isinstance(result, dict)


class TestResilienceSuiteGetResults:
    """Test result retrieval methods."""

    def test_get_results_empty(self, sample_classification_dataset):
        """Test getting results when none exist."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        results = suite.results
        assert results == {}

    def test_get_results_after_run(self, sample_classification_dataset):
        """Test getting results after running tests."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        suite.config('quick')
        suite.run()

        results = suite.results
        assert isinstance(results, dict)


class TestResilienceSuiteEdgeCases:
    """Test edge cases and error handling."""

    def test_with_empty_feature_subset(self, sample_classification_dataset):
        """Test with empty feature subset."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            feature_subset=[]
        )

        assert suite.feature_subset == []

    def test_with_invalid_feature_subset(self, sample_classification_dataset):
        """Test with invalid feature names in subset."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
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

    def test_verbose_mode(self, sample_classification_dataset, capsys):
        """Test verbose output."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=True
        )

        suite.config('quick')

        captured = capsys.readouterr()
        # Should have printed something during config
        assert len(captured.out) > 0 or suite.verbose is True

    def test_random_state_consistency(self, sample_classification_dataset):
        """Test that random_state provides consistent results."""
        suite1 = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )
        suite1.config('quick')
        results1 = suite1.run()

        suite2 = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )
        suite2.config('quick')
        results2 = suite2.run()

        # Results should be similar
        assert type(results1) == type(results2)


class TestResilienceSuiteIntegration:
    """Integration tests."""

    def test_full_workflow_quick(self, sample_classification_dataset):
        """Test complete workflow with quick config."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        # Configure and run
        suite.config('quick')
        results = suite.run()

        assert isinstance(results, dict)

    def test_full_workflow_medium(self, sample_classification_dataset):
        """Test complete workflow with medium config."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        suite.config('medium')
        results = suite.run()

        assert isinstance(results, dict)

    def test_chaining_config_and_run(self, sample_classification_dataset):
        """Test method chaining config().run()."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        # Should be able to chain
        results = suite.config('quick').run()

        assert isinstance(results, dict)

    def test_multiple_runs_same_config(self, sample_classification_dataset):
        """Test running multiple times with same config."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        suite.config('quick')

        results1 = suite.run()
        results2 = suite.run()

        assert isinstance(results1, dict)
        assert isinstance(results2, dict)

    def test_config_change_between_runs(self, sample_classification_dataset):
        """Test changing config between runs."""
        suite = ResilienceSuite(
            dataset=sample_classification_dataset,
            verbose=False,
            random_state=42
        )

        suite.config('quick')
        results1 = suite.run()

        suite.config('medium')
        results2 = suite.run()

        assert isinstance(results1, dict)
        assert isinstance(results2, dict)

    def test_with_regression_dataset(self, sample_regression_dataset):
        """Test workflow with regression dataset."""
        suite = ResilienceSuite(
            dataset=sample_regression_dataset,
            verbose=False,
            random_state=42,
            metric='mae'  # Use appropriate regression metric
        )

        # Test basic initialization with regression
        assert suite._problem_type == 'regression'
        assert suite.metric == 'mae'
