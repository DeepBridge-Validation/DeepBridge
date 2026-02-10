"""Tests for HyperparameterSuite validation wrapper."""

import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression

from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.hyperparameter_suite import (
    HyperparameterSuite,
)


@pytest.fixture
def sample_classification_dataset():
    """Create a simple classification dataset for testing."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })

    # Create DBDataset
    dataset = DBDataset(
        data=data,
        target_column='target'
    )

    # Set a simple model
    model = LogisticRegression(random_state=42, max_iter=100)
    model.fit(dataset.train_data[['feature1', 'feature2', 'feature3']],
              dataset.train_data['target'])
    dataset.set_model(model)

    return dataset


@pytest.fixture
def sample_regression_dataset():
    """Create a simple regression dataset for testing."""
    np.random.seed(42)
    n_samples = 100

    data = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'target': np.random.randn(n_samples)
    })

    dataset = DBDataset(
        data=data,
        target_column='target'
    )

    model = LinearRegression()
    model.fit(dataset.train_data[['feature1', 'feature2']],
              dataset.train_data['target'])
    dataset.set_model(model)

    return dataset


class TestHyperparameterSuiteInit:
    """Test HyperparameterSuite initialization."""

    def test_init_basic(self, sample_classification_dataset):
        """Test basic initialization."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        assert suite.dataset is not None
        assert suite.verbose is False
        assert suite.metric == 'accuracy'
        assert suite.current_config is None
        assert suite.results == {}

    def test_init_with_params(self, sample_classification_dataset):
        """Test initialization with custom parameters."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=True,
            random_state=123,
            metric='auc',
            feature_subset=['feature1', 'feature2']
        )

        assert suite.verbose is True
        assert suite.random_state == 123
        assert suite.metric == 'auc'
        assert suite.feature_subset == ['feature1', 'feature2']

    def test_problem_type_detection_classification(self, sample_classification_dataset):
        """Test problem type detection for classification."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        # Should detect classification
        assert suite._problem_type in ['classification', 'binary', 'multiclass']

    def test_problem_type_detection_regression(self, sample_regression_dataset):
        """Test problem type detection for regression."""
        suite = HyperparameterSuite(
            dataset=sample_regression_dataset,
            verbose=False,
            metric='mse'
        )

        # Should detect regression
        assert suite._problem_type == 'regression'


class TestHyperparameterSuiteConfig:
    """Test HyperparameterSuite configuration."""

    def test_config_quick(self, sample_classification_dataset):
        """Test quick configuration."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        result = suite.config('quick')

        assert result is suite  # Should return self for chaining
        assert suite.current_config is not None
        assert len(suite.current_config) == 1
        assert suite.current_config[0]['method'] == 'importance'

    def test_config_medium(self, sample_classification_dataset):
        """Test medium configuration."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        suite.config('medium')

        assert suite.current_config is not None
        assert len(suite.current_config) == 1
        assert suite.current_config[0]['params']['cv'] == 5
        assert suite.current_config[0]['params']['n_subsamples'] == 10

    def test_config_full(self, sample_classification_dataset):
        """Test full configuration."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        suite.config('full')

        assert suite.current_config is not None
        assert len(suite.current_config) == 2  # Full has 2 test configurations

    def test_config_invalid(self, sample_classification_dataset):
        """Test invalid configuration name."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        with pytest.raises(ValueError, match='Unknown configuration'):
            suite.config('invalid_config')

    def test_config_with_feature_subset(self, sample_classification_dataset):
        """Test configuration with feature subset."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        suite.config('quick', feature_subset=['feature1'])

        assert suite.feature_subset == ['feature1']
        assert suite.current_config[0]['params']['feature_subset'] == ['feature1']


class TestHyperparameterSuiteRun:
    """Test HyperparameterSuite run method."""

    def test_run_with_quick_config(self, sample_classification_dataset):
        """Test running suite with quick configuration."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        suite.config('quick')
        results = suite.run()

        # Check results structure
        assert 'importance' in results
        assert 'importance_scores' in results
        assert 'sorted_importance' in results
        assert 'tuning_order' in results

        # Check that we got some importance scores
        assert len(results['importance_scores']) > 0
        assert isinstance(results['tuning_order'], list)

    def test_run_without_config_uses_default(self, sample_classification_dataset):
        """Test that run() uses 'quick' config by default if none set."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        # Don't call config() - run should use default
        results = suite.run()

        # Should still work with default config
        assert 'importance_scores' in results
        assert suite.current_config is not None

    def test_run_with_verbose(self, sample_classification_dataset, capsys):
        """Test run with verbose output."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=True
        )

        suite.config('quick')
        suite.run()

        # Check that some output was printed
        captured = capsys.readouterr()
        assert 'Running hyperparameter importance' in captured.out or \
               'Test suite completed' in captured.out


class TestHyperparameterSuiteEdgeCases:
    """Test edge cases and error handling."""

    def test_with_different_metrics_classification(self, sample_classification_dataset):
        """Test with different classification metrics."""
        for metric in ['accuracy', 'auc', 'f1']:
            suite = HyperparameterSuite(
                dataset=sample_classification_dataset,
                verbose=False,
                metric=metric
            )

            assert suite.metric == metric

    def test_with_different_metrics_regression(self, sample_regression_dataset):
        """Test with different regression metrics."""
        for metric in ['mse', 'mae', 'r2']:
            suite = HyperparameterSuite(
                dataset=sample_regression_dataset,
                verbose=False,
                metric=metric
            )

            assert suite.metric == metric

    def test_clone_config_creates_copy(self, sample_classification_dataset):
        """Test that _clone_config creates deep copy."""
        suite = HyperparameterSuite(
            dataset=sample_classification_dataset,
            verbose=False
        )

        original = [{'method': 'importance', 'params': {'cv': 3}}]
        cloned = suite._clone_config(original)

        # Modify cloned
        cloned[0]['params']['cv'] = 999

        # Original should be unchanged
        assert original[0]['params']['cv'] == 3
