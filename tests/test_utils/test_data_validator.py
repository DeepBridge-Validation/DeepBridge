"""
Comprehensive tests for DataValidator utility class.

This test suite validates:
1. validate_data_input - data configuration validation
2. validate_features - feature validation and inference

Coverage Target: ~95%+
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from deepbridge.utils.data_validator import DataValidator


# ==================== Fixtures ====================


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing"""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [10, 20, 30, 40, 50],
        'feature3': [100, 200, 300, 400, 500],
        'target': [0, 1, 0, 1, 0],
        'prob_0': [0.6, 0.3, 0.7, 0.2, 0.8],
        'prob_1': [0.4, 0.7, 0.3, 0.8, 0.2],
    })


@pytest.fixture
def sklearn_dataset():
    """Create a mock scikit-learn dataset"""
    dataset = Mock()
    dataset.data = np.array([[1, 2], [3, 4], [5, 6]])
    dataset.target = np.array([0, 1, 0])
    dataset.feature_names = ['feature1', 'feature2']
    return dataset


# ==================== validate_data_input Tests ====================


class TestValidateDataInput:
    """Tests for validate_data_input static method"""

    def test_valid_unified_data_with_dataframe(self, sample_dataframe):
        """Test valid configuration with unified data (DataFrame)"""
        # Should not raise
        DataValidator.validate_data_input(
            data=sample_dataframe,
            train_data=None,
            test_data=None,
            target_column='target'
        )

    def test_valid_unified_data_with_sklearn_dataset(self, sklearn_dataset):
        """Test valid configuration with unified data (sklearn dataset)"""
        # Should not raise even without target_column (sklearn datasets)
        DataValidator.validate_data_input(
            data=sklearn_dataset,
            train_data=None,
            test_data=None,
            target_column=None
        )

    def test_valid_split_data_with_dataframes(self, sample_dataframe):
        """Test valid configuration with train/test split (DataFrames)"""
        train_df = sample_dataframe.iloc[:3]
        test_df = sample_dataframe.iloc[3:]

        # Should not raise
        DataValidator.validate_data_input(
            data=None,
            train_data=train_df,
            test_data=test_df,
            target_column='target'
        )

    def test_error_both_data_and_split_provided(self, sample_dataframe):
        """Test error when both data and train/test are provided"""
        with pytest.raises(ValueError, match='Cannot provide both data and train/test'):
            DataValidator.validate_data_input(
                data=sample_dataframe,
                train_data=sample_dataframe,
                test_data=None,
                target_column='target'
            )

    def test_error_both_data_and_train_provided(self, sample_dataframe):
        """Test error when both data and only train_data are provided"""
        with pytest.raises(ValueError, match='Cannot provide both data and train/test'):
            DataValidator.validate_data_input(
                data=sample_dataframe,
                train_data=sample_dataframe,
                test_data=None,
                target_column='target'
            )

    def test_error_both_data_and_test_provided(self, sample_dataframe):
        """Test error when both data and only test_data are provided"""
        with pytest.raises(ValueError, match='Cannot provide both data and train/test'):
            DataValidator.validate_data_input(
                data=sample_dataframe,
                train_data=None,
                test_data=sample_dataframe,
                target_column='target'
            )

    def test_error_no_data_provided(self):
        """Test error when no data is provided"""
        with pytest.raises(ValueError, match='Must provide either data or both train_data and test_data'):
            DataValidator.validate_data_input(
                data=None,
                train_data=None,
                test_data=None,
                target_column='target'
            )

    def test_error_only_train_data_provided(self, sample_dataframe):
        """Test error when only train_data is provided (missing test_data)"""
        with pytest.raises(ValueError, match='Must provide either data or both train_data and test_data'):
            DataValidator.validate_data_input(
                data=None,
                train_data=sample_dataframe,
                test_data=None,
                target_column='target'
            )

    def test_error_only_test_data_provided(self, sample_dataframe):
        """Test error when only test_data is provided (missing train_data)"""
        with pytest.raises(ValueError, match='Must provide either data or both train_data and test_data'):
            DataValidator.validate_data_input(
                data=None,
                train_data=None,
                test_data=sample_dataframe,
                target_column='target'
            )

    def test_error_dataframe_without_target_column(self, sample_dataframe):
        """Test error when DataFrame provided without target_column"""
        with pytest.raises(ValueError, match='target_column must be provided when using DataFrame'):
            DataValidator.validate_data_input(
                data=sample_dataframe,
                train_data=None,
                test_data=None,
                target_column=None
            )

    def test_error_train_dataframe_without_target_column(self, sample_dataframe):
        """Test error when train DataFrame provided without target_column"""
        with pytest.raises(ValueError, match='target_column must be provided when using DataFrame'):
            DataValidator.validate_data_input(
                data=None,
                train_data=sample_dataframe,
                test_data=sample_dataframe,
                target_column=None
            )

    def test_error_test_dataframe_without_target_column(self, sample_dataframe):
        """Test error when test DataFrame provided without target_column"""
        with pytest.raises(ValueError, match='target_column must be provided when using DataFrame'):
            DataValidator.validate_data_input(
                data=None,
                train_data=sample_dataframe,
                test_data=sample_dataframe,
                target_column=None
            )


# ==================== validate_features Tests ====================


class TestValidateFeatures:
    """Tests for validate_features static method"""

    def test_infer_features_from_dataframe(self, sample_dataframe):
        """Test inferring features from DataFrame (exclude target)"""
        features = DataValidator.validate_features(
            features=None,
            data=sample_dataframe,
            target_column='target',
            prob_cols=None
        )

        assert set(features) == {'feature1', 'feature2', 'feature3', 'prob_0', 'prob_1'}
        assert 'target' not in features

    def test_infer_features_excluding_prob_cols(self, sample_dataframe):
        """Test inferring features excluding probability columns"""
        features = DataValidator.validate_features(
            features=None,
            data=sample_dataframe,
            target_column='target',
            prob_cols=['prob_0', 'prob_1']
        )

        assert set(features) == {'feature1', 'feature2', 'feature3'}
        assert 'target' not in features
        assert 'prob_0' not in features
        assert 'prob_1' not in features

    def test_validate_provided_features_dataframe(self, sample_dataframe):
        """Test validating provided features against DataFrame"""
        features = DataValidator.validate_features(
            features=['feature1', 'feature2'],
            data=sample_dataframe,
            target_column='target',
            prob_cols=None
        )

        assert features == ['feature1', 'feature2']

    def test_error_missing_features_in_dataframe(self, sample_dataframe):
        """Test error when provided features don't exist in DataFrame"""
        with pytest.raises(ValueError, match="Features {'nonexistent'} not found in data"):
            DataValidator.validate_features(
                features=['feature1', 'nonexistent'],
                data=sample_dataframe,
                target_column='target',
                prob_cols=None
            )

    def test_error_multiple_missing_features(self, sample_dataframe):
        """Test error with multiple missing features"""
        with pytest.raises(ValueError, match='not found in data'):
            DataValidator.validate_features(
                features=['feature1', 'missing1', 'missing2'],
                data=sample_dataframe,
                target_column='target',
                prob_cols=None
            )

    def test_infer_features_from_sklearn_dataset(self, sklearn_dataset):
        """Test inferring features from sklearn dataset"""
        features = DataValidator.validate_features(
            features=None,
            data=sklearn_dataset,
            target_column='target',
            prob_cols=None
        )

        assert features == ['feature1', 'feature2']

    def test_validate_features_sklearn_dataset(self, sklearn_dataset):
        """Test validating provided features against sklearn dataset"""
        features = DataValidator.validate_features(
            features=['feature1'],
            data=sklearn_dataset,
            target_column='target',
            prob_cols=None
        )

        assert features == ['feature1']

    def test_error_missing_features_sklearn_dataset(self, sklearn_dataset):
        """Test error when features not in sklearn dataset"""
        with pytest.raises(ValueError, match="Features {'nonexistent'} not found in data"):
            DataValidator.validate_features(
                features=['feature1', 'nonexistent'],
                data=sklearn_dataset,
                target_column='target',
                prob_cols=None
            )

    def test_convert_sklearn_dataset_with_feature_names(self):
        """Test converting sklearn dataset with feature_names to DataFrame"""
        dataset = Mock()
        dataset.data = np.array([[1, 2], [3, 4]])
        dataset.feature_names = ['feat_a', 'feat_b']

        features = DataValidator.validate_features(
            features=None,
            data=dataset,
            target_column='target',
            prob_cols=None
        )

        # Should use feature_names from dataset
        assert features == ['feat_a', 'feat_b']

    def test_convert_sklearn_dataset_without_feature_names(self):
        """Test converting sklearn dataset without feature_names"""
        dataset = Mock(spec=['data'])  # Only has 'data' attribute
        dataset.data = np.array([[1, 2, 3], [4, 5, 6]])

        features = DataValidator.validate_features(
            features=None,
            data=dataset,
            target_column='target',
            prob_cols=None
        )

        # Should infer column names (0, 1, 2) except target
        assert 'target' not in features
        assert len(features) == 3  # All columns except target

    def test_convert_numpy_array_to_dataframe(self):
        """Test converting numpy array to DataFrame"""
        data_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        # When features not provided and data is raw numpy, should raise error
        # because target_column filtering doesn't work on numpy arrays properly
        with pytest.raises(ValueError, match='Cannot validate features'):
            features = DataValidator.validate_features(
                features=None,
                data=data_array,
                target_column=0,
                prob_cols=None
            )

    def test_error_invalid_data_type(self):
        """Test error when data cannot be converted"""
        invalid_data = "not a valid data type"

        with pytest.raises(ValueError, match='Cannot validate features'):
            DataValidator.validate_features(
                features=None,
                data=invalid_data,
                target_column='target',
                prob_cols=None
            )

    def test_validate_features_preserves_order(self, sample_dataframe):
        """Test that provided features preserve their order"""
        features = DataValidator.validate_features(
            features=['feature3', 'feature1', 'feature2'],
            data=sample_dataframe,
            target_column='target',
            prob_cols=None
        )

        assert features == ['feature3', 'feature1', 'feature2']

    def test_infer_features_empty_prob_cols_list(self, sample_dataframe):
        """Test with empty prob_cols list"""
        features = DataValidator.validate_features(
            features=None,
            data=sample_dataframe,
            target_column='target',
            prob_cols=[]  # Empty list
        )

        assert set(features) == {'feature1', 'feature2', 'feature3', 'prob_0', 'prob_1'}


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_single_feature_dataframe(self):
        """Test DataFrame with only one feature column"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [0, 1, 0]
        })

        features = DataValidator.validate_features(
            features=None,
            data=df,
            target_column='target',
            prob_cols=None
        )

        assert features == ['feature1']

    def test_all_columns_features(self):
        """Test DataFrame where all columns are features (target in data)"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })

        features = DataValidator.validate_features(
            features=None,
            data=df,
            target_column='nonexistent_target',  # Target not in data
            prob_cols=None
        )

        # All columns should be features since target doesn't exist
        assert set(features) == {'feature1', 'feature2'}

    def test_validate_data_input_with_none_for_all(self):
        """Test all None parameters"""
        with pytest.raises(ValueError, match='Must provide either data'):
            DataValidator.validate_data_input(
                data=None,
                train_data=None,
                test_data=None,
                target_column=None
            )
