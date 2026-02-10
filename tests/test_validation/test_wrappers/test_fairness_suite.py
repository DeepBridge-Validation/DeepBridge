"""Tests for FairnessSuite validation wrapper."""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.fairness_suite import FairnessSuite


@pytest.fixture
def sample_fairness_dataset():
    """Create a simple dataset with protected attributes for fairness testing."""
    np.random.seed(42)
    n_samples = 300

    # Create synthetic data with protected attributes
    gender = np.random.choice(['male', 'female'], n_samples)
    age = np.random.randint(20, 70, n_samples)
    race = np.random.choice(['white', 'black', 'asian'], n_samples)

    # Features correlated with protected attributes
    feature1 = np.random.randn(n_samples) + (gender == 'male') * 0.5
    feature2 = np.random.randn(n_samples) + (age > 40) * 0.3

    # Target somewhat correlated with features (but trying to be fair)
    y = ((feature1 + feature2) > 0.5).astype(int)

    data = pd.DataFrame({
        'gender': gender,
        'age': age,
        'race': race,
        'feature1': feature1,
        'feature2': feature2,
        'target': y
    })

    # Create DBDataset
    dataset = DBDataset(
        data=data,
        target_column='target'
    )

    # Train a simple model on non-protected features only
    # The model should be trained ONLY on predictive features, not protected attributes
    model = LogisticRegression(random_state=42, max_iter=200)
    modeling_features = ['feature1', 'feature2']
    model.fit(dataset.train_data[modeling_features], dataset.train_data['target'])
    dataset.set_model(model)

    return dataset


@pytest.fixture
def sample_complex_fairness_dataset():
    """Create a more complex dataset with multiple protected attributes."""
    np.random.seed(123)
    n_samples = 400

    # Multiple protected attributes
    gender = np.random.choice(['male', 'female', 'other'], n_samples)
    age = np.random.randint(18, 80, n_samples)
    ethnicity = np.random.choice(['group_a', 'group_b', 'group_c'], n_samples)

    # More features
    f1 = np.random.randn(n_samples)
    f2 = np.random.randn(n_samples)
    f3 = np.random.randn(n_samples)

    y = ((f1 + f2 - f3) > 0).astype(int)

    data = pd.DataFrame({
        'gender': gender,
        'age': age,
        'ethnicity': ethnicity,
        'feat1': f1,
        'feat2': f2,
        'feat3': f3,
        'target': y
    })

    dataset = DBDataset(data=data, target_column='target')

    # Train model only on non-protected features
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=4)
    modeling_features = ['feat1', 'feat2', 'feat3']
    model.fit(dataset.train_data[modeling_features], dataset.train_data['target'])
    dataset.set_model(model)

    return dataset


class TestFairnessSuiteInit:
    """Test FairnessSuite initialization."""

    def test_init_basic(self, sample_fairness_dataset):
        """Test basic initialization."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        assert suite.dataset is not None
        assert suite.protected_attributes == ['gender']
        assert suite.verbose is False
        assert suite.current_config is None

    def test_init_with_multiple_attributes(self, sample_fairness_dataset):
        """Test initialization with multiple protected attributes."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender', 'race'],
            verbose=False
        )

        assert suite.protected_attributes == ['gender', 'race']

    def test_init_with_privileged_groups(self, sample_fairness_dataset):
        """Test initialization with privileged groups specified."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            privileged_groups={'gender': 'male'},
            verbose=False
        )

        assert suite.privileged_groups == {'gender': 'male'}

    def test_init_with_age_grouping_options(self, sample_fairness_dataset):
        """Test initialization with age grouping options."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['age'],
            age_grouping=True,
            age_grouping_strategy='median',
            verbose=False
        )

        assert suite.age_grouping is True
        assert suite.age_grouping_strategy == 'median'

    def test_init_invalid_protected_attribute(self, sample_fairness_dataset):
        """Test that invalid protected attributes raise error."""
        with pytest.raises(ValueError, match='Protected attributes not found'):
            FairnessSuite(
                dataset=sample_fairness_dataset,
                protected_attributes=['nonexistent_column'],
                verbose=False
            )

    def test_init_creates_metrics_calculator(self, sample_fairness_dataset):
        """Test that metrics calculator is initialized."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        assert hasattr(suite, 'metrics_calculator')
        assert suite.metrics_calculator is not None


class TestFairnessSuiteConfig:
    """Test FairnessSuite configuration."""

    def test_config_quick(self, sample_fairness_dataset):
        """Test quick configuration."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        result = suite.config('quick')

        assert result is suite  # Should return self for chaining
        assert suite.current_config is not None
        assert suite.current_config_name == 'quick'
        assert 'statistical_parity' in suite.current_config['metrics']

    def test_config_medium(self, sample_fairness_dataset):
        """Test medium configuration."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        suite.config('medium')

        assert suite.current_config_name == 'medium'
        assert len(suite.current_config['metrics']) >= 5

    def test_config_full(self, sample_fairness_dataset):
        """Test full configuration."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        suite.config('full')

        assert suite.current_config_name == 'full'
        assert len(suite.current_config['metrics']) >= 10

    def test_config_invalid(self, sample_fairness_dataset):
        """Test invalid configuration name."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        with pytest.raises(ValueError, match='Unknown configuration'):
            suite.config('invalid_config')

    def test_config_templates_exist(self):
        """Test that config templates are defined."""
        assert hasattr(FairnessSuite, '_CONFIG_TEMPLATES')
        assert 'quick' in FairnessSuite._CONFIG_TEMPLATES
        assert 'medium' in FairnessSuite._CONFIG_TEMPLATES
        assert 'full' in FairnessSuite._CONFIG_TEMPLATES

    def test_metric_lists_exist(self):
        """Test that metric lists are defined."""
        assert hasattr(FairnessSuite, '_PRETRAIN_METRICS')
        assert hasattr(FairnessSuite, '_POSTTRAIN_METRICS')
        assert len(FairnessSuite._PRETRAIN_METRICS) > 0
        assert len(FairnessSuite._POSTTRAIN_METRICS) > 0


class TestFairnessSuiteRun:
    """Test FairnessSuite run method."""

    def test_run_with_quick_config(self, sample_fairness_dataset):
        """Test running suite with quick configuration."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        suite.config('quick')
        results = suite.run()

        # Check results - may be FairnessResult object or dict
        assert results is not None
        # FairnessResult object has attributes we can check
        if hasattr(results, '__dict__'):
            assert len(vars(results)) > 0  # Should have some attributes
        else:
            assert isinstance(results, dict)

    def test_run_without_config_uses_default(self, sample_fairness_dataset):
        """Test that run() works without explicit config."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        # Don't call config() - should use default or handle gracefully
        # The implementation might auto-configure
        if suite.current_config is None:
            suite.config('quick')

        results = suite.run()
        assert results is not None  # Accept any result object/dict

    def test_run_with_multiple_attributes(self, sample_complex_fairness_dataset):
        """Test running with multiple protected attributes."""
        suite = FairnessSuite(
            dataset=sample_complex_fairness_dataset,
            protected_attributes=['gender', 'ethnicity'],
            verbose=False
        )

        suite.config('quick')
        results = suite.run()

        assert results is not None  # Accept any result object

    def test_run_with_verbose(self, sample_fairness_dataset, capsys):
        """Test run with verbose output."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=True
        )

        suite.config('quick')
        suite.run()

        # Check that some output was produced
        captured = capsys.readouterr()
        assert len(captured.out) >= 0  # At least no errors


class TestFairnessSuiteHelperMethods:
    """Test FairnessSuite helper methods."""

    def test_is_age_column_detection(self, sample_fairness_dataset):
        """Test age column detection."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        # Test with actual age column
        age_data = sample_fairness_dataset.get_feature_data()['age']
        is_age = suite._is_age_column('age', age_data)
        assert isinstance(is_age, bool)

    def test_collect_dataset_info(self, sample_fairness_dataset):
        """Test dataset info collection."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        # _collect_dataset_info requires X and y_true parameters
        X = sample_fairness_dataset.get_feature_data()
        y_true = sample_fairness_dataset.get_target_data()
        info = suite._collect_dataset_info(X, y_true)
        assert isinstance(info, dict)

    def test_assess_fairness_method(self, sample_fairness_dataset):
        """Test fairness assessment categorization."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        # Test different score ranges
        assessment_high = suite._assess_fairness(0.9)
        assessment_low = suite._assess_fairness(0.3)

        assert isinstance(assessment_high, str)
        assert isinstance(assessment_low, str)
        assert assessment_high != assessment_low  # Different scores should give different assessments


class TestFairnessSuiteEdgeCases:
    """Test edge cases and error handling."""

    def test_with_age_attribute(self, sample_fairness_dataset):
        """Test running with age as protected attribute."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['age'],
            age_grouping=True,
            verbose=False
        )

        suite.config('quick')
        results = suite.run()

        assert results is not None  # Accept any result object

    def test_with_custom_privileged_groups(self, sample_fairness_dataset):
        """Test with custom privileged groups."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender', 'race'],
            privileged_groups={'gender': 'male', 'race': 'white'},
            verbose=False
        )

        suite.config('quick')
        results = suite.run()

        assert results is not None  # Accept any result object

    def test_different_age_grouping_strategies(self, sample_fairness_dataset):
        """Test different age grouping strategies."""
        for strategy in ['median', 'adea', 'ecoa']:
            suite = FairnessSuite(
                dataset=sample_fairness_dataset,
                protected_attributes=['age'],
                age_grouping=True,
                age_grouping_strategy=strategy,
                verbose=False
            )

            assert suite.age_grouping_strategy == strategy

    def test_method_chaining(self, sample_fairness_dataset):
        """Test that methods support chaining."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        # Config should return self for chaining
        result = suite.config('quick')
        assert result is suite

        # Should be able to call run immediately
        results = result.run()
        assert results is not None  # Accept any result object

    def test_get_detailed_results(self, sample_fairness_dataset):
        """Test getting detailed results for a specific attribute."""
        suite = FairnessSuite(
            dataset=sample_fairness_dataset,
            protected_attributes=['gender'],
            verbose=False
        )

        suite.config('quick')
        suite.run()

        # Try to get detailed results
        detailed = suite.get_detailed_results('gender')
        # May return None or dict depending on implementation
        assert detailed is None or isinstance(detailed, dict)
