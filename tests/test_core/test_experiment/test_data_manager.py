"""
Comprehensive tests for DataManager.

This test suite validates:
1. __init__ - initialization
2. prepare_data - train-test split functionality
3. get_dataset_split - retrieving splits
4. get_binary_predictions - converting probabilities to predictions
5. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock

from deepbridge.core.experiment.data_manager import DataManager


# ==================== Fixtures ====================


@pytest.fixture
def mock_dataset():
    """Create mock dataset"""
    dataset = Mock()
    dataset.X = pd.DataFrame({
        'feat1': range(100),
        'feat2': range(100, 200)
    })
    dataset.target = pd.Series(np.random.randint(0, 2, 100))
    dataset.original_prob = None
    return dataset


@pytest.fixture
def mock_dataset_with_prob():
    """Create mock dataset with probabilities"""
    dataset = Mock()
    dataset.X = pd.DataFrame({
        'feat1': range(100),
        'feat2': range(100, 200)
    })
    dataset.target = pd.Series(np.random.randint(0, 2, 100))
    dataset.original_prob = pd.DataFrame({
        'prob_0': np.random.rand(100),
        'prob_1': np.random.rand(100)
    })
    return dataset


@pytest.fixture
def data_manager(mock_dataset):
    """Create DataManager instance"""
    return DataManager(
        dataset=mock_dataset,
        test_size=0.2,
        random_state=42
    )


# ==================== Initialization Tests ====================


class TestInitialization:
    """Tests for __init__ method"""

    def test_init_stores_parameters(self, mock_dataset):
        """Test that initialization stores parameters"""
        dm = DataManager(
            dataset=mock_dataset,
            test_size=0.3,
            random_state=123
        )

        assert dm.dataset is mock_dataset
        assert dm.test_size == 0.3
        assert dm.random_state == 123

    def test_init_initializes_data_attributes_to_none(self, data_manager):
        """Test that data attributes are initialized to None"""
        assert data_manager.X_train is None
        assert data_manager.X_test is None
        assert data_manager.y_train is None
        assert data_manager.y_test is None
        assert data_manager.prob_train is None
        assert data_manager.prob_test is None


# ==================== Prepare Data Tests ====================


class TestPrepareData:
    """Tests for prepare_data method"""

    def test_prepare_data_creates_train_test_splits(self, data_manager):
        """Test that prepare_data creates train and test splits"""
        data_manager.prepare_data()

        assert data_manager.X_train is not None
        assert data_manager.X_test is not None
        assert data_manager.y_train is not None
        assert data_manager.y_test is not None

    def test_prepare_data_respects_test_size(self, mock_dataset):
        """Test that test_size is respected"""
        dm = DataManager(
            dataset=mock_dataset,
            test_size=0.2,
            random_state=42
        )
        dm.prepare_data()

        total_size = len(mock_dataset.X)
        test_size = len(dm.X_test)

        assert test_size == int(total_size * 0.2)

    def test_prepare_data_respects_random_state(self, mock_dataset):
        """Test that random_state produces reproducible splits"""
        dm1 = DataManager(mock_dataset, test_size=0.2, random_state=42)
        dm2 = DataManager(mock_dataset, test_size=0.2, random_state=42)

        dm1.prepare_data()
        dm2.prepare_data()

        pd.testing.assert_frame_equal(dm1.X_train, dm2.X_train)
        pd.testing.assert_frame_equal(dm1.X_test, dm2.X_test)

    def test_prepare_data_without_probabilities(self, data_manager):
        """Test prepare_data when dataset has no probabilities"""
        data_manager.prepare_data()

        assert data_manager.prob_train is None
        assert data_manager.prob_test is None

    def test_prepare_data_with_probabilities(self, mock_dataset_with_prob):
        """Test prepare_data when dataset has probabilities"""
        dm = DataManager(mock_dataset_with_prob, test_size=0.2, random_state=42)
        dm.prepare_data()

        assert dm.prob_train is not None
        assert dm.prob_test is not None
        assert len(dm.prob_train) == len(dm.X_train)
        assert len(dm.prob_test) == len(dm.X_test)

    def test_prepare_data_probability_indices_match(self, mock_dataset_with_prob):
        """Test that probability indices match X indices"""
        dm = DataManager(mock_dataset_with_prob, test_size=0.2, random_state=42)
        dm.prepare_data()

        assert list(dm.prob_train.index) == list(dm.X_train.index)
        assert list(dm.prob_test.index) == list(dm.X_test.index)


# ==================== Get Dataset Split Tests ====================


class TestGetDatasetSplit:
    """Tests for get_dataset_split method"""

    def test_get_train_split_returns_train_data(self, data_manager):
        """Test that get_dataset_split returns train data"""
        data_manager.prepare_data()

        X, y, prob = data_manager.get_dataset_split('train')

        assert X is data_manager.X_train
        assert y is data_manager.y_train
        assert prob is data_manager.prob_train

    def test_get_test_split_returns_test_data(self, data_manager):
        """Test that get_dataset_split returns test data"""
        data_manager.prepare_data()

        X, y, prob = data_manager.get_dataset_split('test')

        assert X is data_manager.X_test
        assert y is data_manager.y_test
        assert prob is data_manager.prob_test

    def test_get_dataset_split_raises_error_for_invalid_split(self, data_manager):
        """Test that invalid split name raises ValueError"""
        data_manager.prepare_data()

        with pytest.raises(ValueError, match="must be either 'train' or 'test'"):
            data_manager.get_dataset_split('invalid')

    def test_get_dataset_split_with_probabilities(self, mock_dataset_with_prob):
        """Test get_dataset_split when probabilities exist"""
        dm = DataManager(mock_dataset_with_prob, test_size=0.2, random_state=42)
        dm.prepare_data()

        X_train, y_train, prob_train = dm.get_dataset_split('train')

        assert prob_train is not None
        assert len(prob_train) == len(X_train)


# ==================== Get Binary Predictions Tests ====================


class TestGetBinaryPredictions:
    """Tests for get_binary_predictions method"""

    def test_get_binary_predictions_with_default_threshold(self, data_manager):
        """Test binary predictions with default threshold 0.5"""
        probs = pd.DataFrame({
            'prob_0': [0.3, 0.7, 0.4],
            'prob_1': [0.7, 0.3, 0.6]
        })

        predictions = data_manager.get_binary_predictions(probs)

        expected = pd.Series([1, 0, 1])
        pd.testing.assert_series_equal(predictions, expected, check_names=False)

    def test_get_binary_predictions_with_custom_threshold(self, data_manager):
        """Test binary predictions with custom threshold"""
        probs = pd.DataFrame({
            'prob_0': [0.2, 0.8],
            'prob_1': [0.8, 0.2]
        })

        predictions = data_manager.get_binary_predictions(probs, threshold=0.7)

        expected = pd.Series([1, 0])
        pd.testing.assert_series_equal(predictions, expected, check_names=False)

    def test_get_binary_predictions_with_single_column(self, data_manager):
        """Test binary predictions with single column probabilities"""
        probs = pd.DataFrame({
            'prob': [0.3, 0.7, 0.5, 0.6]
        })

        predictions = data_manager.get_binary_predictions(probs)

        expected = pd.Series([0, 1, 1, 1])
        pd.testing.assert_series_equal(predictions, expected, check_names=False)

    def test_get_binary_predictions_returns_int_type(self, data_manager):
        """Test that predictions are returned as integers"""
        probs = pd.DataFrame({'prob': [0.3, 0.7]})

        predictions = data_manager.get_binary_predictions(probs)

        assert predictions.dtype == int

    def test_get_binary_predictions_boundary_cases(self, data_manager):
        """Test binary predictions at threshold boundaries"""
        probs = pd.DataFrame({
            'prob_0': [0.5, 0.5],
            'prob_1': [0.5, 0.5]
        })

        predictions = data_manager.get_binary_predictions(probs, threshold=0.5)

        # At threshold, should be 1 (>=)
        expected = pd.Series([1, 1])
        pd.testing.assert_series_equal(predictions, expected, check_names=False)


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow_prepare_and_retrieve(self, data_manager):
        """Test complete workflow: prepare then retrieve splits"""
        # Prepare data
        data_manager.prepare_data()

        # Retrieve train split
        X_train, y_train, prob_train = data_manager.get_dataset_split('train')
        assert len(X_train) == len(y_train)

        # Retrieve test split
        X_test, y_test, prob_test = data_manager.get_dataset_split('test')
        assert len(X_test) == len(y_test)

        # Verify no overlap
        train_indices = set(X_train.index)
        test_indices = set(X_test.index)
        assert len(train_indices.intersection(test_indices)) == 0

    def test_workflow_with_probabilities_end_to_end(self, mock_dataset_with_prob):
        """Test end-to-end workflow with probabilities"""
        dm = DataManager(mock_dataset_with_prob, test_size=0.2, random_state=42)

        # Prepare
        dm.prepare_data()

        # Get test split
        X_test, y_test, prob_test = dm.get_dataset_split('test')

        # Generate predictions from probabilities
        predictions = dm.get_binary_predictions(prob_test)

        assert len(predictions) == len(y_test)


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_prepare_data_with_small_dataset(self):
        """Test with very small dataset"""
        dataset = Mock()
        dataset.X = pd.DataFrame({'feat': [1, 2, 3, 4, 5]})
        dataset.target = pd.Series([0, 1, 0, 1, 0])
        dataset.original_prob = None

        dm = DataManager(dataset, test_size=0.2, random_state=42)
        dm.prepare_data()

        assert len(dm.X_train) + len(dm.X_test) == 5

    def test_get_binary_predictions_with_all_zeros(self, data_manager):
        """Test predictions with all probabilities below threshold"""
        probs = pd.DataFrame({'prob': [0.1, 0.2, 0.3]})

        predictions = data_manager.get_binary_predictions(probs, threshold=0.5)

        assert all(predictions == 0)

    def test_get_binary_predictions_with_all_ones(self, data_manager):
        """Test predictions with all probabilities above threshold"""
        probs = pd.DataFrame({'prob': [0.9, 0.8, 0.7]})

        predictions = data_manager.get_binary_predictions(probs, threshold=0.5)

        assert all(predictions == 1)

    def test_prepare_data_with_different_test_sizes(self, mock_dataset):
        """Test different test_size values"""
        test_sizes = [0.1, 0.3, 0.5]

        for test_size in test_sizes:
            dm = DataManager(mock_dataset, test_size=test_size, random_state=42)
            dm.prepare_data()

            expected_test_size = int(len(mock_dataset.X) * test_size)
            assert len(dm.X_test) == expected_test_size

    def test_get_binary_predictions_extreme_thresholds(self, data_manager):
        """Test with extreme threshold values"""
        probs = pd.DataFrame({'prob': [0.5]})

        # Threshold = 0 should make everything 1
        pred_low = data_manager.get_binary_predictions(probs, threshold=0.0)
        assert pred_low[0] == 1

        # Threshold = 1 should make everything 0
        pred_high = data_manager.get_binary_predictions(probs, threshold=1.0)
        assert pred_high[0] == 0
