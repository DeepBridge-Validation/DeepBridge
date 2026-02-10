"""
Comprehensive tests for DataPerturber in validation.wrappers.robustness.data_perturber.

This module tests data perturbation methods used in robustness testing,
including raw and quantile-based perturbation strategies.
"""

import numpy as np
import pandas as pd
import pytest

from deepbridge.validation.wrappers.robustness.data_perturber import (
    DataPerturber,
)


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.uniform(0, 100, 50),
        'feature2': np.random.normal(50, 15, 50),
        'feature3': np.random.randint(1, 10, 50),
    })


@pytest.fixture
def sample_array():
    """Create a sample numpy array for testing."""
    np.random.seed(42)
    return np.random.rand(50, 3)


@pytest.fixture
def perturber():
    """Create a DataPerturber instance."""
    return DataPerturber()


class TestDataPerturberInitialization:
    """Test initialization of DataPerturber."""

    def test_init_creates_rng(self):
        """Test that initialization creates a random number generator."""
        perturber = DataPerturber()

        assert hasattr(perturber, 'rng')
        assert isinstance(perturber.rng, np.random.RandomState)

    def test_set_random_state(self, perturber):
        """Test setting random state for reproducibility."""
        perturber.set_random_state(42)

        assert isinstance(perturber.rng, np.random.RandomState)

        # Test reproducibility
        perturber.set_random_state(42)
        result1 = perturber.rng.rand(10)

        perturber.set_random_state(42)
        result2 = perturber.rng.rand(10)

        np.testing.assert_array_equal(result1, result2)

    def test_set_random_state_none(self, perturber):
        """Test setting random state with None."""
        perturber.set_random_state(None)

        assert isinstance(perturber.rng, np.random.RandomState)


class TestDataPerturberPerturbDataDataFrame:
    """Test perturb_data method with DataFrame input."""

    def test_perturb_data_raw_all_features(
        self, perturber, sample_dataframe
    ):
        """Test raw perturbation on all features."""
        perturber.set_random_state(42)

        original = sample_dataframe.copy()
        perturbed = perturber.perturb_data(
            sample_dataframe, perturb_method='raw', level=0.1
        )

        # Check that data was perturbed
        assert not perturbed.equals(original)
        assert perturbed.shape == original.shape
        assert list(perturbed.columns) == list(original.columns)

    def test_perturb_data_raw_specific_features(
        self, perturber, sample_dataframe
    ):
        """Test raw perturbation on specific features."""
        perturber.set_random_state(42)

        original = sample_dataframe.copy()
        perturbed = perturber.perturb_data(
            sample_dataframe,
            perturb_method='raw',
            level=0.1,
            perturb_features=['feature1'],
        )

        # feature1 should be perturbed
        assert not perturbed['feature1'].equals(original['feature1'])

        # Other features should remain unchanged
        pd.testing.assert_series_equal(
            perturbed['feature2'], original['feature2']
        )
        pd.testing.assert_series_equal(
            perturbed['feature3'], original['feature3']
        )

    def test_perturb_data_quantile_all_features(
        self, perturber, sample_dataframe
    ):
        """Test quantile perturbation on all features."""
        perturber.set_random_state(42)

        original = sample_dataframe.copy()
        perturbed = perturber.perturb_data(
            sample_dataframe, perturb_method='quantile', level=0.1
        )

        # Check that data was perturbed
        assert not perturbed.equals(original)
        assert perturbed.shape == original.shape

    def test_perturb_data_quantile_specific_features(
        self, perturber, sample_dataframe
    ):
        """Test quantile perturbation on specific features."""
        perturber.set_random_state(42)

        original = sample_dataframe.copy()
        perturbed = perturber.perturb_data(
            sample_dataframe,
            perturb_method='quantile',
            level=0.1,
            perturb_features=['feature2', 'feature3'],
        )

        # feature2 and feature3 should be perturbed
        assert not perturbed['feature2'].equals(original['feature2'])
        assert not perturbed['feature3'].equals(original['feature3'])

        # feature1 should remain unchanged
        pd.testing.assert_series_equal(
            perturbed['feature1'], original['feature1']
        )

    def test_perturb_data_preserves_original(
        self, perturber, sample_dataframe
    ):
        """Test that perturbation doesn't modify the original data."""
        perturber.set_random_state(42)

        original_copy = sample_dataframe.copy()
        _ = perturber.perturb_data(
            sample_dataframe, perturb_method='raw', level=0.1
        )

        # Original should remain unchanged
        pd.testing.assert_frame_equal(sample_dataframe, original_copy)

    def test_perturb_data_invalid_method_raises_error(
        self, perturber, sample_dataframe
    ):
        """Test that invalid perturbation method raises ValueError."""
        with pytest.raises(ValueError, match='Unknown perturbation method'):
            perturber.perturb_data(
                sample_dataframe, perturb_method='invalid_method', level=0.1
            )

    def test_perturb_data_nonexistent_feature(
        self, perturber, sample_dataframe
    ):
        """Test perturbation with nonexistent feature name."""
        perturber.set_random_state(42)

        original = sample_dataframe.copy()
        # Should not raise error, just skip the nonexistent feature
        perturbed = perturber.perturb_data(
            sample_dataframe,
            perturb_method='raw',
            level=0.1,
            perturb_features=['nonexistent_feature'],
        )

        # Data should remain unchanged since feature doesn't exist
        pd.testing.assert_frame_equal(perturbed, original)


class TestDataPerturberPerturbDataArray:
    """Test perturb_data method with numpy array input."""

    def test_perturb_data_raw_array(self, perturber, sample_array):
        """Test raw perturbation on numpy array."""
        perturber.set_random_state(42)

        original = sample_array.copy()
        perturbed = perturber.perturb_data(
            sample_array, perturb_method='raw', level=0.1
        )

        # Check that data was perturbed
        assert not np.array_equal(perturbed, original)
        assert perturbed.shape == original.shape

    def test_perturb_data_raw_array_specific_columns(
        self, perturber, sample_array
    ):
        """Test raw perturbation on specific columns of numpy array."""
        perturber.set_random_state(42)

        original = sample_array.copy()
        perturbed = perturber.perturb_data(
            sample_array,
            perturb_method='raw',
            level=0.1,
            perturb_features=[0, 2],
        )

        # Columns 0 and 2 should be perturbed
        assert not np.array_equal(perturbed[:, 0], original[:, 0])
        assert not np.array_equal(perturbed[:, 2], original[:, 2])

        # Column 1 should remain unchanged
        np.testing.assert_array_equal(perturbed[:, 1], original[:, 1])

    def test_perturb_data_quantile_array(self, perturber, sample_array):
        """Test quantile perturbation on numpy array."""
        perturber.set_random_state(42)

        original = sample_array.copy()
        perturbed = perturber.perturb_data(
            sample_array, perturb_method='quantile', level=0.1
        )

        # Check that data was perturbed
        assert not np.array_equal(perturbed, original)
        assert perturbed.shape == original.shape

    def test_perturb_data_array_out_of_bounds_index(
        self, perturber, sample_array
    ):
        """Test perturbation with out of bounds column index."""
        perturber.set_random_state(42)

        original = sample_array.copy()
        # Should not raise error, just skip the out of bounds index
        perturbed = perturber.perturb_data(
            sample_array,
            perturb_method='raw',
            level=0.1,
            perturb_features=[10, -1],  # Out of bounds indices
        )

        # Data should remain unchanged
        np.testing.assert_array_equal(perturbed, original)


class TestDataPerturberApplyRawPerturbation:
    """Test _apply_raw_perturbation method."""

    def test_raw_perturbation_adds_gaussian_noise(
        self, perturber, sample_dataframe
    ):
        """Test that raw perturbation adds Gaussian noise."""
        perturber.set_random_state(42)

        X_perturbed = sample_dataframe.copy()
        # Use column index 0 instead of name
        perturber._apply_raw_perturbation(
            sample_dataframe, X_perturbed, 0, 0.1
        )

        # Check that values changed
        assert not X_perturbed['feature1'].equals(sample_dataframe['feature1'])

        # The difference should be noise proportional to std
        expected_noise_std = 0.1 * sample_dataframe['feature1'].std()
        actual_noise = X_perturbed['feature1'] - sample_dataframe['feature1']

        # Noise should have approximately the expected scale
        assert abs(actual_noise.std() - expected_noise_std) < 2.0

    def test_raw_perturbation_integer_dtype(self, perturber):
        """Test raw perturbation with integer dtype columns."""
        df = pd.DataFrame({'int_col': [1, 2, 3, 4, 5]})
        perturber.set_random_state(42)

        X_perturbed = df.copy()
        # Use column index 0 instead of name
        perturber._apply_raw_perturbation(df, X_perturbed, 0, 0.1)

        # Should maintain integer dtype
        assert pd.api.types.is_integer_dtype(X_perturbed['int_col'].dtype)

    def test_raw_perturbation_array(self, perturber, sample_array):
        """Test raw perturbation on numpy array."""
        perturber.set_random_state(42)

        X_perturbed = sample_array.copy()
        perturber._apply_raw_perturbation(
            sample_array, X_perturbed, 0, 0.1
        )

        # Check that column 0 was perturbed
        assert not np.array_equal(X_perturbed[:, 0], sample_array[:, 0])

        # Other columns should remain unchanged
        np.testing.assert_array_equal(X_perturbed[:, 1:], sample_array[:, 1:])


class TestDataPerturberApplyQuantilePerturbation:
    """Test _apply_quantile_perturbation method."""

    def test_quantile_perturbation_uses_quantiles(
        self, perturber, sample_dataframe
    ):
        """Test that quantile perturbation uses data quantiles."""
        perturber.set_random_state(42)

        X_perturbed = sample_dataframe.copy()
        # Use column index 0 instead of name
        perturber._apply_quantile_perturbation(
            sample_dataframe, X_perturbed, 0, 0.1
        )

        # Check that values changed
        assert not X_perturbed['feature1'].equals(sample_dataframe['feature1'])

        # Values should be within expected range based on quantiles
        q25, q75 = sample_dataframe['feature1'].quantile([0.25, 0.75])
        expected_min = q25 * (1 - 0.1)
        expected_max = q75 * (1 + 0.1)

        assert X_perturbed['feature1'].min() >= expected_min - 1.0
        assert X_perturbed['feature1'].max() <= expected_max + 1.0

    def test_quantile_perturbation_integer_dtype(self, perturber):
        """Test quantile perturbation with integer dtype columns."""
        df = pd.DataFrame({'int_col': [10, 20, 30, 40, 50]})
        perturber.set_random_state(42)

        X_perturbed = df.copy()
        # Use column index 0 instead of name
        perturber._apply_quantile_perturbation(
            df, X_perturbed, 0, 0.1
        )

        # Should maintain integer dtype
        assert pd.api.types.is_integer_dtype(X_perturbed['int_col'].dtype)

    def test_quantile_perturbation_array(self, perturber, sample_array):
        """Test quantile perturbation on numpy array."""
        perturber.set_random_state(42)

        X_perturbed = sample_array.copy()
        perturber._apply_quantile_perturbation(
            sample_array, X_perturbed, 1, 0.1
        )

        # Check that column 1 was perturbed
        assert not np.array_equal(X_perturbed[:, 1], sample_array[:, 1])

        # Other columns should remain unchanged
        np.testing.assert_array_equal(X_perturbed[:, 0], sample_array[:, 0])
        np.testing.assert_array_equal(X_perturbed[:, 2], sample_array[:, 2])


class TestDataPerturberPerturbFeaturesIndividually:
    """Test perturb_features_individually method."""

    def test_perturb_features_individually_dataframe(
        self, perturber, sample_dataframe
    ):
        """Test perturbing features individually on DataFrame."""
        perturber.set_random_state(42)

        result = perturber.perturb_features_individually(
            sample_dataframe, perturb_method='raw', level=0.1
        )

        # Should return a dictionary with one entry per feature
        assert isinstance(result, dict)
        assert len(result) == 3
        assert 'feature1' in result
        assert 'feature2' in result
        assert 'feature3' in result

        # Each entry should be a perturbed version with only that feature changed
        for feature, perturbed in result.items():
            assert isinstance(perturbed, pd.DataFrame)
            assert perturbed.shape == sample_dataframe.shape

    def test_perturb_features_individually_array(
        self, perturber, sample_array
    ):
        """Test perturbing features individually on numpy array."""
        perturber.set_random_state(42)

        result = perturber.perturb_features_individually(
            sample_array, perturb_method='raw', level=0.1
        )

        # Should return a dictionary with one entry per column
        assert isinstance(result, dict)
        assert len(result) == 3

        # Each entry should be a perturbed version
        for col_idx, perturbed in result.items():
            assert isinstance(perturbed, np.ndarray)
            assert perturbed.shape == sample_array.shape

    def test_perturb_features_individually_with_subset(
        self, perturber, sample_dataframe
    ):
        """Test perturbing only a subset of features individually."""
        perturber.set_random_state(42)

        result = perturber.perturb_features_individually(
            sample_dataframe,
            perturb_method='raw',
            level=0.1,
            feature_subset=['feature1', 'feature2'],
        )

        # Should only contain the specified features
        assert len(result) == 2
        assert 'feature1' in result
        assert 'feature2' in result
        assert 'feature3' not in result

    def test_perturb_features_individually_quantile(
        self, perturber, sample_dataframe
    ):
        """Test perturbing features individually with quantile method."""
        perturber.set_random_state(42)

        result = perturber.perturb_features_individually(
            sample_dataframe, perturb_method='quantile', level=0.1
        )

        assert len(result) == 3
        for feature, perturbed in result.items():
            assert isinstance(perturbed, pd.DataFrame)

    def test_perturb_features_individually_preserves_other_features(
        self, perturber, sample_dataframe
    ):
        """Test that individual perturbation only affects the target feature."""
        perturber.set_random_state(42)

        result = perturber.perturb_features_individually(
            sample_dataframe, perturb_method='raw', level=0.1
        )

        # In the result for feature1, only feature1 should be different
        perturbed_feat1 = result['feature1']

        # feature1 should be different
        assert not perturbed_feat1['feature1'].equals(
            sample_dataframe['feature1']
        )

        # feature2 and feature3 should be unchanged
        pd.testing.assert_series_equal(
            perturbed_feat1['feature2'], sample_dataframe['feature2']
        )
        pd.testing.assert_series_equal(
            perturbed_feat1['feature3'], sample_dataframe['feature3']
        )


class TestDataPerturberEdgeCases:
    """Test edge cases and error handling."""

    def test_perturb_single_column_dataframe(self, perturber):
        """Test perturbation on single-column DataFrame."""
        df = pd.DataFrame({'only_col': [1.0, 2.0, 3.0, 4.0, 5.0]})
        perturber.set_random_state(42)

        perturbed = perturber.perturb_data(df, perturb_method='raw', level=0.1)

        assert perturbed.shape == df.shape
        assert not perturbed['only_col'].equals(df['only_col'])

    def test_perturb_single_column_array(self, perturber):
        """Test perturbation on single-column array."""
        arr = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        perturber.set_random_state(42)

        perturbed = perturber.perturb_data(
            arr, perturb_method='raw', level=0.1
        )

        assert perturbed.shape == arr.shape
        assert not np.array_equal(perturbed, arr)

    def test_perturb_with_zero_level(self, perturber, sample_dataframe):
        """Test perturbation with zero level (no perturbation)."""
        perturber.set_random_state(42)

        original = sample_dataframe.copy()
        perturbed = perturber.perturb_data(
            sample_dataframe, perturb_method='raw', level=0.0
        )

        # With zero level, raw perturbation adds zero noise
        # Values should be very close (may have small floating point differences)
        pd.testing.assert_frame_equal(perturbed, original, rtol=1e-10)

    def test_perturb_with_large_level(self, perturber, sample_dataframe):
        """Test perturbation with large level."""
        perturber.set_random_state(42)

        perturbed = perturber.perturb_data(
            sample_dataframe, perturb_method='raw', level=2.0
        )

        # Should still work, values should be significantly different
        assert not perturbed.equals(sample_dataframe)

    def test_perturb_empty_feature_list(self, perturber, sample_dataframe):
        """Test perturbation with empty feature list."""
        perturber.set_random_state(42)

        original = sample_dataframe.copy()
        perturbed = perturber.perturb_data(
            sample_dataframe,
            perturb_method='raw',
            level=0.1,
            perturb_features=[],
        )

        # With empty feature list, nothing should be perturbed
        pd.testing.assert_frame_equal(perturbed, original)

    def test_reproducibility_with_same_seed(
        self, perturber, sample_dataframe
    ):
        """Test that same seed produces same results."""
        perturber.set_random_state(42)
        perturbed1 = perturber.perturb_data(
            sample_dataframe, perturb_method='raw', level=0.1
        )

        perturber.set_random_state(42)
        perturbed2 = perturber.perturb_data(
            sample_dataframe, perturb_method='raw', level=0.1
        )

        pd.testing.assert_frame_equal(perturbed1, perturbed2)


class TestDataPerturberIntegration:
    """Integration tests for complete workflows."""

    def test_full_workflow_raw_perturbation(
        self, perturber, sample_dataframe
    ):
        """Test full workflow with raw perturbation."""
        perturber.set_random_state(42)

        # Perturb all features
        perturbed_all = perturber.perturb_data(
            sample_dataframe, perturb_method='raw', level=0.1
        )

        # Perturb features individually
        perturbed_individual = perturber.perturb_features_individually(
            sample_dataframe, perturb_method='raw', level=0.1
        )

        # Both should produce valid results
        assert perturbed_all.shape == sample_dataframe.shape
        assert len(perturbed_individual) == 3

    def test_full_workflow_quantile_perturbation(
        self, perturber, sample_dataframe
    ):
        """Test full workflow with quantile perturbation."""
        perturber.set_random_state(42)

        # Perturb specific features
        perturbed_subset = perturber.perturb_data(
            sample_dataframe,
            perturb_method='quantile',
            level=0.2,
            perturb_features=['feature1', 'feature3'],
        )

        # Perturb individually with subset
        perturbed_individual = perturber.perturb_features_individually(
            sample_dataframe,
            perturb_method='quantile',
            level=0.2,
            feature_subset=['feature1', 'feature3'],
        )

        assert perturbed_subset.shape == sample_dataframe.shape
        assert len(perturbed_individual) == 2

    def test_mixed_dtype_dataframe(self, perturber):
        """Test perturbation on DataFrame with mixed dtypes."""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'str_col': ['a', 'b', 'c', 'd', 'e'],
        })

        perturber.set_random_state(42)

        # Perturb only numeric columns
        perturbed = perturber.perturb_data(
            df,
            perturb_method='raw',
            level=0.1,
            perturb_features=['int_col', 'float_col'],
        )

        # Numeric columns should be perturbed
        assert not perturbed['int_col'].equals(df['int_col'])
        assert not perturbed['float_col'].equals(df['float_col'])

        # String column should remain unchanged
        pd.testing.assert_series_equal(perturbed['str_col'], df['str_col'])
