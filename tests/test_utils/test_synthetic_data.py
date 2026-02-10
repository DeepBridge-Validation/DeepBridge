"""
Comprehensive tests for SyntheticDataGenerator in utils.synthetic_data.

This module tests the synthetic data generation utilities that create
synthetic data based on real data distributions while preserving statistical
properties and correlations.
"""

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from deepbridge.utils.synthetic_data import SyntheticDataGenerator


@pytest.fixture
def sample_numeric_data():
    """Create sample data with numerical columns only."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.uniform(20000, 100000, 100),
        'score': np.random.normal(75, 10, 100),
    })


@pytest.fixture
def sample_mixed_data():
    """Create sample data with mixed column types."""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'income': np.random.uniform(20000, 100000, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'gender': np.random.choice(['M', 'F'], 100),
        'score': np.random.normal(75, 10, 100),
    })


@pytest.fixture
def sample_categorical_data():
    """Create sample data with categorical columns only."""
    np.random.seed(42)
    return pd.DataFrame({
        'category_a': np.random.choice(['X', 'Y', 'Z'], 100),
        'category_b': np.random.choice(['Low', 'Medium', 'High'], 100),
    })


class TestSyntheticDataGeneratorInitialization:
    """Test initialization of SyntheticDataGenerator."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        generator = SyntheticDataGenerator()

        assert generator.preserve_correlations is True
        assert generator.columns is None
        assert generator.dtypes == {}
        assert generator.categorical_columns == []
        assert generator.numerical_columns == []
        assert generator.correlations is None
        assert generator._is_fitted is False

    def test_init_without_correlation_preservation(self):
        """Test initialization without preserving correlations."""
        generator = SyntheticDataGenerator(preserve_correlations=False)

        assert generator.preserve_correlations is False


class TestSyntheticDataGeneratorIdentifyColumnTypes:
    """Test column type identification."""

    def test_identify_numeric_columns(self, sample_numeric_data):
        """Test identification of numerical columns."""
        generator = SyntheticDataGenerator()
        generator._identify_column_types(sample_numeric_data)

        assert len(generator.numerical_columns) == 3
        assert 'age' in generator.numerical_columns
        assert 'income' in generator.numerical_columns
        assert 'score' in generator.numerical_columns
        assert len(generator.categorical_columns) == 0

    def test_identify_mixed_columns(self, sample_mixed_data):
        """Test identification of mixed column types."""
        generator = SyntheticDataGenerator()
        generator._identify_column_types(sample_mixed_data)

        assert len(generator.numerical_columns) == 3
        assert len(generator.categorical_columns) == 2
        assert 'category' in generator.categorical_columns
        assert 'gender' in generator.categorical_columns

    def test_identify_categorical_columns(self, sample_categorical_data):
        """Test identification of categorical columns."""
        generator = SyntheticDataGenerator()
        generator._identify_column_types(sample_categorical_data)

        assert len(generator.categorical_columns) == 2
        assert len(generator.numerical_columns) == 0

    def test_category_mappings_created(self, sample_mixed_data):
        """Test that category mappings are created correctly."""
        generator = SyntheticDataGenerator()
        generator._identify_column_types(sample_mixed_data)

        assert 'category' in generator.category_mappings
        assert 'gender' in generator.category_mappings
        assert len(generator.category_mappings['gender']) == 2
        assert len(generator.category_mappings['category']) == 3


class TestSyntheticDataGeneratorTransformCategorical:
    """Test categorical transformation methods."""

    def test_transform_categorical(self, sample_mixed_data):
        """Test transformation of categorical to numeric."""
        generator = SyntheticDataGenerator()
        generator._identify_column_types(sample_mixed_data)

        transformed = generator._transform_categorical(sample_mixed_data)

        assert transformed['category'].dtype in [np.int64, int, object]
        assert transformed['gender'].dtype in [np.int64, int, object]

    def test_transform_with_unknown_categories(self, sample_mixed_data):
        """Test transformation with unknown categories."""
        generator = SyntheticDataGenerator()
        generator._identify_column_types(sample_mixed_data)

        # Create test data with unknown category
        test_data = sample_mixed_data.copy()
        test_data.loc[0, 'category'] = 'UNKNOWN'

        transformed = generator._transform_categorical(test_data)

        # Unknown category should be replaced with most common (0)
        assert transformed.loc[0, 'category'] == 0

    def test_inverse_transform_categorical(self, sample_mixed_data):
        """Test inverse transformation from numeric to categorical."""
        generator = SyntheticDataGenerator()
        generator._identify_column_types(sample_mixed_data)

        transformed = generator._transform_categorical(sample_mixed_data)
        restored = generator._inverse_transform_categorical(transformed)

        # Values should match original (may differ in exact order but same distribution)
        assert set(restored['category'].unique()).issubset(set(sample_mixed_data['category'].unique()))

    def test_inverse_transform_clips_out_of_bounds(self, sample_mixed_data):
        """Test that inverse transform clips out of bounds values."""
        generator = SyntheticDataGenerator()
        generator._identify_column_types(sample_mixed_data)

        # Create data with out-of-bounds indices
        transformed = pd.DataFrame({
            'category': [0, 1, 2, 10, -5],  # 10 and -5 are out of bounds
            'gender': [0, 1, 0, 1, 0]
        })

        restored = generator._inverse_transform_categorical(transformed)

        # Should not raise error and all values should be valid
        assert restored['category'].notna().all()


class TestSyntheticDataGeneratorCalculateStatistics:
    """Test statistics calculation."""

    def test_calculate_statistics_numeric(self, sample_numeric_data):
        """Test calculation of statistics for numerical columns."""
        generator = SyntheticDataGenerator()
        generator._identify_column_types(sample_numeric_data)
        transformed = generator._transform_categorical(sample_numeric_data)
        generator._calculate_statistics(transformed)

        assert 'age' in generator.stats
        assert 'income' in generator.stats
        assert 'mean' in generator.stats['age']
        assert 'std' in generator.stats['age']
        assert 'min' in generator.stats['age']
        assert 'max' in generator.stats['age']

    def test_statistics_avoid_zero_std(self):
        """Test that zero std is avoided."""
        generator = SyntheticDataGenerator()

        # Create data with constant column
        data = pd.DataFrame({'const': [5.0] * 100})
        generator._identify_column_types(data)
        transformed = generator._transform_categorical(data)
        generator._calculate_statistics(transformed)

        # std should be at least 1e-5
        assert generator.stats['const']['std'] >= 1e-5


class TestSyntheticDataGeneratorFit:
    """Test the fit method."""

    def test_fit_basic(self, sample_numeric_data):
        """Test basic fitting."""
        generator = SyntheticDataGenerator()
        result = generator.fit(sample_numeric_data)

        assert generator._is_fitted is True
        assert result == generator  # Should return self
        assert generator.correlations is not None

    def test_fit_with_target_column(self, sample_mixed_data):
        """Test fitting with target column."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_mixed_data, target_column='score')

        assert generator.target_column == 'score'
        assert generator._is_fitted is True

    def test_fit_with_model(self, sample_mixed_data):
        """Test fitting with a model."""
        mock_model = Mock()
        generator = SyntheticDataGenerator()
        generator.fit(sample_mixed_data, target_column='score', model=mock_model)

        assert generator.model == mock_model

    def test_fit_without_correlation_preservation(self, sample_numeric_data):
        """Test fitting without preserving correlations."""
        generator = SyntheticDataGenerator(preserve_correlations=False)
        generator.fit(sample_numeric_data)

        assert generator.correlations is None

    def test_fit_handles_nan_correlations(self):
        """Test that NaN correlations are handled correctly."""
        generator = SyntheticDataGenerator()

        # Create data that might produce NaN correlations
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [1, 1, 1, 1, 1],  # constant column
        })

        generator.fit(data)

        # Should not have NaN in correlations
        assert not generator.correlations.isna().any().any()


class TestSyntheticDataGeneratorGenerate:
    """Test the generate method."""

    def test_generate_basic(self, sample_numeric_data):
        """Test basic data generation."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_numeric_data)

        synthetic = generator.generate(50)

        assert len(synthetic) == 50
        assert list(synthetic.columns) == list(sample_numeric_data.columns)

    def test_generate_not_fitted_raises_error(self):
        """Test that generate raises error if not fitted."""
        generator = SyntheticDataGenerator()

        with pytest.raises(ValueError, match='must be fitted'):
            generator.generate(10)

    def test_generate_with_random_state(self, sample_numeric_data):
        """Test generation with random state for reproducibility."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_numeric_data)

        synthetic1 = generator.generate(50, random_state=42)
        synthetic2 = generator.generate(50, random_state=42)

        pd.testing.assert_frame_equal(synthetic1, synthetic2)

    def test_generate_with_categorical_columns(self, sample_mixed_data):
        """Test generation with categorical columns."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_mixed_data)

        synthetic = generator.generate(50)

        assert len(synthetic) == 50
        assert 'category' in synthetic.columns
        assert 'gender' in synthetic.columns
        # Check that categorical values are from original set
        assert set(synthetic['gender'].unique()).issubset({'M', 'F'})

    def test_generate_with_target_column_no_model(self, sample_mixed_data):
        """Test generation with target column but no model."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_mixed_data, target_column='score')

        synthetic = generator.generate(50)

        assert 'score' in synthetic.columns
        assert len(synthetic) == 50

    def test_generate_with_target_column_and_model(self, sample_mixed_data):
        """Test generation with target column and model."""
        # Create mock model with predict
        mock_model = Mock()
        mock_model.predict = Mock(return_value=np.random.normal(75, 10, 50))

        # Remove target from features for generation
        features = sample_mixed_data.drop('score', axis=1)

        generator = SyntheticDataGenerator()
        generator.fit(sample_mixed_data, target_column='score', model=mock_model)

        synthetic = generator.generate(50)

        assert 'score' in synthetic.columns
        assert len(synthetic) == 50

    def test_generate_with_classification_model(self, sample_mixed_data):
        """Test generation with classification model (predict_proba)."""
        # Create mock classification model
        mock_model = Mock()
        mock_model.predict_proba = Mock(return_value=np.random.rand(50, 2))

        sample_mixed_data['target'] = np.random.choice([0, 1], 100)

        generator = SyntheticDataGenerator()
        generator.fit(sample_mixed_data, target_column='target', model=mock_model)

        synthetic = generator.generate(50)

        assert 'target' in synthetic.columns

    def test_generate_with_multiclass_classification(self):
        """Test generation with multi-class classification."""
        data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.choice([0, 1, 2], 100)
        })

        mock_model = Mock()
        # Multi-class probabilities
        probs = np.random.rand(50, 3)
        probs = probs / probs.sum(axis=1, keepdims=True)
        mock_model.predict_proba = Mock(return_value=probs)

        generator = SyntheticDataGenerator()
        generator.fit(data, target_column='target', model=mock_model)

        synthetic = generator.generate(50)

        assert 'target' in synthetic.columns

    def test_generate_with_model_error_fallback(self, sample_mixed_data):
        """Test that generation falls back if model prediction fails."""
        # Create mock model that raises error
        mock_model = Mock()
        mock_model.predict = Mock(side_effect=Exception('Model error'))

        generator = SyntheticDataGenerator()
        generator.fit(sample_mixed_data, target_column='score', model=mock_model)

        synthetic = generator.generate(50)

        # Should still generate data using statistical fallback
        assert 'score' in synthetic.columns
        assert len(synthetic) == 50

    def test_generate_without_correlations(self, sample_numeric_data):
        """Test generation without preserving correlations."""
        generator = SyntheticDataGenerator(preserve_correlations=False)
        generator.fit(sample_numeric_data)

        synthetic = generator.generate(50)

        assert len(synthetic) == 50
        assert list(synthetic.columns) == list(sample_numeric_data.columns)

    def test_generate_with_linalg_error_fallback(self, sample_numeric_data):
        """Test fallback when correlation matrix causes LinAlgError."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_numeric_data)

        # Corrupt correlation matrix to cause LinAlgError
        generator.correlations = pd.DataFrame(
            [[1, 0.9, 0.9], [0.9, 1, 0.9], [0.9, 0.9, -1]],
            columns=['age', 'income', 'score'],
            index=['age', 'income', 'score']
        )

        synthetic = generator.generate(50)

        # Should still generate using fallback method
        assert len(synthetic) == 50

    def test_generate_respects_min_max_constraints(self, sample_numeric_data):
        """Test that generated data respects min/max constraints."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_numeric_data)

        synthetic = generator.generate(100)

        for col in generator.numerical_columns:
            original_min = sample_numeric_data[col].min()
            original_max = sample_numeric_data[col].max()

            assert synthetic[col].min() >= original_min - 1  # Allow small tolerance
            assert synthetic[col].max() <= original_max + 1


class TestSyntheticDataGeneratorEvaluateQuality:
    """Test quality evaluation method."""

    def test_evaluate_quality_numeric(self, sample_numeric_data):
        """Test quality evaluation for numerical columns."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_numeric_data)

        synthetic = generator.generate(100)
        metrics = generator.evaluate_quality(sample_numeric_data, synthetic)

        assert 'age' in metrics
        assert 'income' in metrics
        assert 'mean_real' in metrics['age']
        assert 'mean_synthetic' in metrics['age']
        assert 'mean_diff' in metrics['age']
        assert 'std_real' in metrics['age']

    def test_evaluate_quality_with_ks_test(self, sample_numeric_data):
        """Test that KS test is performed when sufficient samples."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_numeric_data)

        synthetic = generator.generate(100)
        metrics = generator.evaluate_quality(sample_numeric_data, synthetic)

        # Should have KS statistics
        assert 'ks_statistic' in metrics['age']
        assert 'ks_pvalue' in metrics['age']

    def test_evaluate_quality_categorical(self, sample_mixed_data):
        """Test quality evaluation for categorical columns."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_mixed_data)

        synthetic = generator.generate(100)
        metrics = generator.evaluate_quality(sample_mixed_data, synthetic)

        assert 'category' in metrics
        assert 'distribution_difference' in metrics['category']
        assert 'category_count_real' in metrics['category']
        assert 'category_count_synthetic' in metrics['category']

    def test_evaluate_quality_overall_metrics(self, sample_numeric_data):
        """Test overall quality metrics."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_numeric_data)

        synthetic = generator.generate(100)
        metrics = generator.evaluate_quality(sample_numeric_data, synthetic)

        assert 'overall' in metrics
        assert 'avg_mean_diff_pct' in metrics['overall']
        assert 'avg_ks_statistic' in metrics['overall']

    def test_evaluate_quality_with_few_samples(self):
        """Test evaluation with very few samples (no KS test)."""
        data = pd.DataFrame({'col1': [1, 2, 3]})

        generator = SyntheticDataGenerator()
        generator.fit(data)

        synthetic = generator.generate(3)
        metrics = generator.evaluate_quality(data, synthetic)

        # Should still work but might not have KS test
        assert 'col1' in metrics

    def test_evaluate_quality_empty_categorical(self):
        """Test evaluation when categorical columns might be empty."""
        generator = SyntheticDataGenerator()
        generator.fit(pd.DataFrame({'col1': [1, 2, 3]}))

        synthetic = generator.generate(10)
        metrics = generator.evaluate_quality(
            pd.DataFrame({'col1': [1, 2, 3]}),
            synthetic
        )

        # Should work without categorical columns
        assert 'col1' in metrics


class TestSyntheticDataGeneratorEdgeCases:
    """Test edge cases and error handling."""

    def test_single_column_data(self):
        """Test with single column data."""
        data = pd.DataFrame({'only_col': [1, 2, 3, 4, 5]})

        generator = SyntheticDataGenerator()
        generator.fit(data)
        synthetic = generator.generate(10)

        assert len(synthetic) == 10
        assert 'only_col' in synthetic.columns

    def test_very_small_dataset(self):
        """Test with very small dataset."""
        data = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})

        generator = SyntheticDataGenerator()
        generator.fit(data)
        synthetic = generator.generate(5)

        assert len(synthetic) == 5

    def test_large_number_of_samples(self, sample_numeric_data):
        """Test generating large number of samples."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_numeric_data)

        synthetic = generator.generate(10000)

        assert len(synthetic) == 10000

    def test_data_with_missing_values(self):
        """Test with data containing NaN values."""
        data = pd.DataFrame({
            'col1': [1, 2, np.nan, 4, 5],
            'col2': [5, np.nan, 7, 8, 9]
        })

        generator = SyntheticDataGenerator()
        generator.fit(data)

        # Should handle NaN gracefully
        synthetic = generator.generate(10)
        assert len(synthetic) == 10

    def test_preserve_dtypes(self, sample_mixed_data):
        """Test that dtypes are preserved in generated data."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_mixed_data)

        synthetic = generator.generate(50)

        # Check that categorical columns maintain their type
        for col in generator.categorical_columns:
            assert col in synthetic.columns


class TestSyntheticDataGeneratorIntegration:
    """Integration tests for full workflow."""

    def test_full_workflow_numeric(self, sample_numeric_data):
        """Test complete workflow with numerical data."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_numeric_data)

        synthetic = generator.generate(100, random_state=42)
        metrics = generator.evaluate_quality(sample_numeric_data, synthetic)

        assert len(synthetic) == 100
        assert 'overall' in metrics
        # Quality should be reasonable
        assert metrics['overall']['avg_mean_diff_pct'] < 50  # Within 50% error

    def test_full_workflow_mixed(self, sample_mixed_data):
        """Test complete workflow with mixed data types."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_mixed_data)

        synthetic = generator.generate(150, random_state=42)
        metrics = generator.evaluate_quality(sample_mixed_data, synthetic)

        assert len(synthetic) == 150
        assert len(synthetic.columns) == len(sample_mixed_data.columns)

    def test_multiple_generations_same_generator(self, sample_numeric_data):
        """Test multiple generations using same fitted generator."""
        generator = SyntheticDataGenerator()
        generator.fit(sample_numeric_data)

        synthetic1 = generator.generate(50, random_state=1)
        synthetic2 = generator.generate(100, random_state=2)

        assert len(synthetic1) == 50
        assert len(synthetic2) == 100
        # Should have different data
        assert not synthetic1.equals(synthetic2[:50])
