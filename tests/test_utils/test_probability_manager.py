"""
Comprehensive tests for DatabaseProbabilityManager.

This test suite validates:
1. Initialization and dataset validation
2. Train/test probability extraction
3. Binary probability extraction with various formats

Coverage Target: ~95%+
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from deepbridge.utils.probability_manager import DatabaseProbabilityManager


# ==================== Fixtures ====================


@pytest.fixture
def mock_dataset_with_probs():
    """Create mock dataset with probability information"""
    dataset = Mock(spec=['original_prob'])  # Only has original_prob
    dataset.original_prob = pd.DataFrame({
        'prob_class_0': [0.6, 0.3, 0.7, 0.2, 0.8],
        'prob_class_1': [0.4, 0.7, 0.3, 0.8, 0.2],
    })
    return dataset


@pytest.fixture
def mock_dataset_without_probs():
    """Create mock dataset without probability information"""
    dataset = Mock()
    dataset.original_prob = None
    return dataset


@pytest.fixture
def mock_dataset_split():
    """Create mock dataset with pre-split train/test data"""
    dataset = Mock()
    dataset.original_prob = pd.DataFrame({
        'prob_class_0': [0.6, 0.3, 0.7, 0.2, 0.8, 0.5, 0.9, 0.1],
        'prob_class_1': [0.4, 0.7, 0.3, 0.8, 0.2, 0.5, 0.1, 0.9],
    }, index=range(8))

    # Mock train and test data with specific indices
    dataset.train_data = pd.DataFrame({'feature': [1, 2, 3, 4, 5]}, index=[0, 1, 2, 3, 4])
    dataset.test_data = pd.DataFrame({'feature': [6, 7, 8]}, index=[5, 6, 7])
    return dataset


# ==================== Initialization Tests ====================


class TestInitialization:
    """Tests for DatabaseProbabilityManager initialization"""

    def test_init_with_probabilities_verbose(self, mock_dataset_with_probs, capsys):
        """Test initialization with probabilities and verbose=True"""
        manager = DatabaseProbabilityManager(mock_dataset_with_probs, verbose=True)

        assert manager.dataset == mock_dataset_with_probs
        assert manager.verbose is True

        # Check verbose output
        captured = capsys.readouterr()
        assert 'Dataset probability info' in captured.out
        assert 'Probability columns' in captured.out

    def test_init_without_probabilities_verbose(self, mock_dataset_without_probs, capsys):
        """Test initialization without probabilities and verbose=True"""
        manager = DatabaseProbabilityManager(mock_dataset_without_probs, verbose=True)

        captured = capsys.readouterr()
        assert 'WARNING: Dataset has no probability information' in captured.out

    def test_init_verbose_false(self, mock_dataset_with_probs, capsys):
        """Test initialization with verbose=False"""
        manager = DatabaseProbabilityManager(mock_dataset_with_probs, verbose=False)

        assert manager.verbose is False
        captured = capsys.readouterr()
        assert captured.out == ''  # No output

    def test_init_dataset_no_original_prob_attr(self, capsys):
        """Test initialization when dataset doesn't have original_prob attribute"""
        dataset = Mock(spec=[])  # No attributes
        manager = DatabaseProbabilityManager(dataset, verbose=True)

        captured = capsys.readouterr()
        assert 'WARNING: Dataset has no probability information' in captured.out


# ==================== get_train_test_probabilities Tests ====================


class TestGetTrainTestProbabilities:
    """Tests for get_train_test_probabilities method"""

    def test_no_probabilities_available(self, mock_dataset_without_probs, capsys):
        """Test when dataset has no probabilities"""
        manager = DatabaseProbabilityManager(mock_dataset_without_probs, verbose=True)

        train_probs, test_probs = manager.get_train_test_probabilities()

        assert train_probs is None
        assert test_probs is None
        captured = capsys.readouterr()
        assert 'No probabilities available' in captured.out

    def test_pre_split_train_test(self, mock_dataset_split, capsys):
        """Test with pre-split train/test data"""
        manager = DatabaseProbabilityManager(mock_dataset_split, verbose=True)

        train_probs, test_probs = manager.get_train_test_probabilities()

        assert train_probs.shape == (5, 2)  # 5 train samples, 2 classes
        assert test_probs.shape == (3, 2)   # 3 test samples, 2 classes

        # Verify correct indices
        assert list(train_probs.index) == [0, 1, 2, 3, 4]
        assert list(test_probs.index) == [5, 6, 7]

        captured = capsys.readouterr()
        assert 'Using pre-split train/test probabilities' in captured.out

    def test_split_with_test_indices(self, mock_dataset_with_probs, capsys):
        """Test splitting with provided test_indices"""
        manager = DatabaseProbabilityManager(mock_dataset_with_probs, verbose=True)

        test_indices = [3, 4]
        train_probs, test_probs = manager.get_train_test_probabilities(test_indices)

        assert len(train_probs) == 3  # Indices 0, 1, 2
        assert len(test_probs) == 2   # Indices 3, 4

        captured = capsys.readouterr()
        assert 'Splitting probabilities using provided test indices' in captured.out
        assert '(n=2)' in captured.out

    def test_no_split_info(self, mock_dataset_with_probs, capsys):
        """Test when no splitting information available"""
        manager = DatabaseProbabilityManager(mock_dataset_with_probs, verbose=True)

        train_probs, test_probs = manager.get_train_test_probabilities()

        assert len(train_probs) == 5  # All samples
        assert test_probs is None

        captured = capsys.readouterr()
        assert 'returning all probabilities as train' in captured.out

    def test_verbose_false_no_output(self, mock_dataset_with_probs, capsys):
        """Test that verbose=False produces no output"""
        manager = DatabaseProbabilityManager(mock_dataset_with_probs, verbose=False)

        train_probs, test_probs = manager.get_train_test_probabilities([3, 4])

        captured = capsys.readouterr()
        assert captured.out == ''


# ==================== extract_binary_probabilities Tests ====================


class TestExtractBinaryProbabilities:
    """Tests for extract_binary_probabilities static method"""

    def test_none_input(self):
        """Test with None input"""
        result = DatabaseProbabilityManager.extract_binary_probabilities(None)
        assert result is None

    def test_standard_prob_class_format(self):
        """Test standard prob_class_0, prob_class_1 format"""
        probs = pd.DataFrame({
            'prob_class_0': [0.6, 0.3, 0.7],
            'prob_class_1': [0.4, 0.7, 0.3],
        })

        result = DatabaseProbabilityManager.extract_binary_probabilities(probs)

        assert result.shape == (3, 2)
        np.testing.assert_array_almost_equal(result[0], [0.6, 0.4])
        np.testing.assert_array_almost_equal(result[1], [0.3, 0.7])

    def test_alternative_class_prob_format(self):
        """Test class_0_prob, class_1_prob format"""
        probs = pd.DataFrame({
            'class_0_prob': [0.2, 0.8],
            'class_1_prob': [0.8, 0.2],
        })

        result = DatabaseProbabilityManager.extract_binary_probabilities(probs)

        assert result.shape == (2, 2)
        np.testing.assert_array_almost_equal(result[0], [0.2, 0.8])

    def test_generic_prob_columns(self):
        """Test generic columns with 'prob' in name"""
        probs = pd.DataFrame({
            'prob_0': [0.1, 0.9],
            'prob_1': [0.9, 0.1],
        })

        result = DatabaseProbabilityManager.extract_binary_probabilities(probs)

        assert result.shape == (2, 2)
        assert result[0][0] in [0.1, 0.9]  # Should use the prob columns

    def test_single_probability_column(self):
        """Test with single probability column (positive class)"""
        probs = pd.DataFrame({
            'probability': [0.7, 0.3, 0.9],
        })

        result = DatabaseProbabilityManager.extract_binary_probabilities(probs)

        assert result.shape == (3, 2)
        # First column should be 1 - pos_prob
        np.testing.assert_array_almost_equal(result[0], [0.3, 0.7])
        np.testing.assert_array_almost_equal(result[1], [0.7, 0.3])
        np.testing.assert_array_almost_equal(result[2], [0.1, 0.9])

    def test_last_two_columns_fallback(self):
        """Test fallback to last two columns"""
        probs = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'col_a': [0.6, 0.2, 0.8],
            'col_b': [0.4, 0.8, 0.2],
        })

        result = DatabaseProbabilityManager.extract_binary_probabilities(probs)

        assert result.shape == (3, 2)
        # Should use last two columns (col_a, col_b)
        np.testing.assert_array_almost_equal(result[0], [0.6, 0.4])

    def test_single_column_fallback(self):
        """Test fallback with single column"""
        probs = pd.DataFrame({
            'value': [0.6, 0.3, 0.9],
        })

        result = DatabaseProbabilityManager.extract_binary_probabilities(probs)

        assert result.shape == (3, 2)
        # Should create complement probabilities
        np.testing.assert_array_almost_equal(result[0], [0.4, 0.6])
        np.testing.assert_array_almost_equal(result[1], [0.7, 0.3])

    def test_empty_dataframe(self):
        """Test with empty DataFrame"""
        probs = pd.DataFrame()

        result = DatabaseProbabilityManager.extract_binary_probabilities(probs)

        assert result is None


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow_with_split(self, mock_dataset_split):
        """Test complete workflow: init → get_probs → extract"""
        manager = DatabaseProbabilityManager(mock_dataset_split, verbose=False)

        train_probs, test_probs = manager.get_train_test_probabilities()

        # Extract binary probabilities
        train_binary = DatabaseProbabilityManager.extract_binary_probabilities(train_probs)
        test_binary = DatabaseProbabilityManager.extract_binary_probabilities(test_probs)

        assert train_binary.shape == (5, 2)
        assert test_binary.shape == (3, 2)

        # Verify probabilities sum to 1
        np.testing.assert_array_almost_equal(train_binary.sum(axis=1), np.ones(5))
        np.testing.assert_array_almost_equal(test_binary.sum(axis=1), np.ones(3))

    def test_workflow_with_custom_indices(self, mock_dataset_with_probs):
        """Test workflow with custom test indices"""
        manager = DatabaseProbabilityManager(mock_dataset_with_probs, verbose=False)

        test_indices = [1, 3]
        train_probs, test_probs = manager.get_train_test_probabilities(test_indices)

        train_binary = DatabaseProbabilityManager.extract_binary_probabilities(train_probs)
        test_binary = DatabaseProbabilityManager.extract_binary_probabilities(test_probs)

        assert len(train_binary) == 3  # 0, 2, 4
        assert len(test_binary) == 2   # 1, 3

    def test_workflow_no_probabilities(self, mock_dataset_without_probs):
        """Test workflow when dataset has no probabilities"""
        manager = DatabaseProbabilityManager(mock_dataset_without_probs, verbose=False)

        train_probs, test_probs = manager.get_train_test_probabilities()

        assert train_probs is None
        assert test_probs is None

        # extract should handle None gracefully
        result = DatabaseProbabilityManager.extract_binary_probabilities(train_probs)
        assert result is None


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_single_sample_dataset(self):
        """Test with dataset containing single sample"""
        dataset = Mock(spec=['original_prob'])
        dataset.original_prob = pd.DataFrame({
            'prob_class_0': [0.6],
            'prob_class_1': [0.4],
        })

        manager = DatabaseProbabilityManager(dataset, verbose=False)
        train_probs, test_probs = manager.get_train_test_probabilities()

        assert len(train_probs) == 1
        assert test_probs is None

    def test_extract_with_three_classes(self):
        """Test extraction with more than 2 probability columns"""
        probs = pd.DataFrame({
            'prob_0': [0.2, 0.3],
            'prob_1': [0.5, 0.4],
            'prob_2': [0.3, 0.3],
        })

        # Should use last two columns
        result = DatabaseProbabilityManager.extract_binary_probabilities(probs)

        assert result.shape == (2, 2)
        # Should extract last two columns
        np.testing.assert_array_almost_equal(result[0], [0.5, 0.3])

    def test_probabilities_with_mixed_naming(self):
        """Test with mixed naming convention"""
        probs = pd.DataFrame({
            'feature': [1, 2],
            'prob_negative': [0.6, 0.3],
            'prob_positive': [0.4, 0.7],
        })

        result = DatabaseProbabilityManager.extract_binary_probabilities(probs)

        # Should find columns with 'prob' in name
        assert result.shape == (2, 2)

    def test_verbose_output_shapes(self, mock_dataset_split, capsys):
        """Test that verbose output shows correct shapes"""
        manager = DatabaseProbabilityManager(mock_dataset_split, verbose=True)

        train_probs, test_probs = manager.get_train_test_probabilities()

        captured = capsys.readouterr()
        assert '(5, 2)' in captured.out  # Train shape
        assert '(3, 2)' in captured.out  # Test shape
