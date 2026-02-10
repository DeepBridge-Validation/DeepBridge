"""
Comprehensive tests for DBDatasetFactory.

Coverage Target: ~100%
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch

from deepbridge.utils.dataset_factory import DBDatasetFactory


@pytest.fixture
def sample_data():
    """Create sample train/test data"""
    train = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [4, 5, 6],
        'target': [0, 1, 0]
    })
    test = pd.DataFrame({
        'feat1': [7, 8],
        'feat2': [9, 10],
        'target': [1, 0]
    })
    return train, test


@pytest.fixture
def sample_predictions():
    """Create sample predictions"""
    train_pred = pd.DataFrame({
        'prob_0': [0.7, 0.3, 0.6],
        'prob_1': [0.3, 0.7, 0.4]
    })
    test_pred = pd.DataFrame({
        'prob_0': [0.2, 0.8],
        'prob_1': [0.8, 0.2]
    })
    return train_pred, test_pred


@pytest.fixture
def mock_model():
    """Create mock model"""
    return Mock()


@pytest.fixture
def mock_dataset():
    """Create mock DBDataset"""
    dataset = Mock()
    dataset.train_data = pd.DataFrame({'feat': [1, 2, 3], 'target': [0, 1, 0]})
    dataset.test_data = pd.DataFrame({'feat': [4, 5], 'target': [1, 0]})
    dataset.target_name = 'target'
    dataset.categorical_features = ['cat_feat']
    return dataset


class TestCreateFromModel:
    """Tests for create_from_model method"""

    @patch('deepbridge.utils.dataset_factory.DBDataset')
    def test_create_from_model_basic(self, mock_dbdataset, sample_data, mock_model):
        """Test basic dataset creation from model"""
        train, test = sample_data
        
        DBDatasetFactory.create_from_model(
            train_data=train,
            test_data=test,
            target_column='target',
            model=mock_model
        )
        
        mock_dbdataset.assert_called_once()
        call_kwargs = mock_dbdataset.call_args[1]
        assert call_kwargs['model'] is mock_model
        assert call_kwargs['target_column'] == 'target'

    @patch('deepbridge.utils.dataset_factory.DBDataset')
    def test_create_from_model_with_categorical(self, mock_dbdataset, sample_data, mock_model):
        """Test creation with categorical features"""
        train, test = sample_data
        
        DBDatasetFactory.create_from_model(
            train_data=train,
            test_data=test,
            target_column='target',
            model=mock_model,
            categorical_features=['feat1']
        )
        
        call_kwargs = mock_dbdataset.call_args[1]
        assert call_kwargs['categorical_features'] == ['feat1']

    @patch('deepbridge.utils.dataset_factory.DBDataset')
    def test_create_from_model_with_dataset_name(self, mock_dbdataset, sample_data, mock_model):
        """Test creation with dataset name"""
        train, test = sample_data
        
        DBDatasetFactory.create_from_model(
            train_data=train,
            test_data=test,
            target_column='target',
            model=mock_model,
            dataset_name='MyDataset'
        )
        
        call_kwargs = mock_dbdataset.call_args[1]
        assert call_kwargs['dataset_name'] == 'MyDataset'

    @patch('deepbridge.utils.dataset_factory.DBDataset')
    def test_create_from_model_with_kwargs(self, mock_dbdataset, sample_data, mock_model):
        """Test that additional kwargs are passed through"""
        train, test = sample_data
        
        DBDatasetFactory.create_from_model(
            train_data=train,
            test_data=test,
            target_column='target',
            model=mock_model,
            custom_param='value',
            another_param=123
        )
        
        call_kwargs = mock_dbdataset.call_args[1]
        assert call_kwargs['custom_param'] == 'value'
        assert call_kwargs['another_param'] == 123


class TestCreateFromProbabilities:
    """Tests for create_from_probabilities method"""

    @patch('deepbridge.utils.dataset_factory.DBDataset')
    def test_create_from_probabilities_basic(self, mock_dbdataset, sample_data, sample_predictions):
        """Test basic creation from probabilities"""
        train, test = sample_data
        train_pred, test_pred = sample_predictions
        
        DBDatasetFactory.create_from_probabilities(
            train_data=train,
            test_data=test,
            target_column='target',
            train_predictions=train_pred,
            test_predictions=test_pred
        )
        
        mock_dbdataset.assert_called_once()
        call_kwargs = mock_dbdataset.call_args[1]
        assert call_kwargs['target_column'] == 'target'

    @patch('deepbridge.utils.dataset_factory.DBDataset')
    def test_create_from_probabilities_with_prob_cols(self, mock_dbdataset, sample_data, sample_predictions):
        """Test creation with specified prob columns"""
        train, test = sample_data
        train_pred, _ = sample_predictions
        
        DBDatasetFactory.create_from_probabilities(
            train_data=train,
            test_data=test,
            target_column='target',
            train_predictions=train_pred,
            prob_cols=['prob_0', 'prob_1']
        )
        
        call_kwargs = mock_dbdataset.call_args[1]
        assert call_kwargs['prob_cols'] == ['prob_0', 'prob_1']

    @patch('deepbridge.utils.dataset_factory.DBDataset')
    def test_create_from_probabilities_without_test_pred(self, mock_dbdataset, sample_data, sample_predictions):
        """Test creation without test predictions"""
        train, test = sample_data
        train_pred, _ = sample_predictions
        
        DBDatasetFactory.create_from_probabilities(
            train_data=train,
            test_data=test,
            target_column='target',
            train_predictions=train_pred,
            test_predictions=None
        )
        
        call_kwargs = mock_dbdataset.call_args[1]
        assert call_kwargs['test_predictions'] is None


class TestCreateForAlternativeModel:
    """Tests for create_for_alternative_model method"""

    @patch('deepbridge.utils.dataset_factory.DBDataset')
    def test_create_for_alternative_model_basic(self, mock_dbdataset, mock_dataset, mock_model):
        """Test creation for alternative model"""
        DBDatasetFactory.create_for_alternative_model(
            original_dataset=mock_dataset,
            model=mock_model
        )
        
        mock_dbdataset.assert_called_once()
        call_kwargs = mock_dbdataset.call_args[1]
        assert call_kwargs['model'] is mock_model
        assert call_kwargs['target_column'] == 'target'

    @patch('deepbridge.utils.dataset_factory.DBDataset')
    def test_create_for_alternative_model_preserves_data(self, mock_dbdataset, mock_dataset, mock_model):
        """Test that original data is preserved"""
        DBDatasetFactory.create_for_alternative_model(
            original_dataset=mock_dataset,
            model=mock_model
        )
        
        call_kwargs = mock_dbdataset.call_args[1]
        assert call_kwargs['train_data'] is mock_dataset.train_data
        assert call_kwargs['test_data'] is mock_dataset.test_data

    @patch('deepbridge.utils.dataset_factory.DBDataset')
    def test_create_for_alternative_model_with_categorical(self, mock_dbdataset, mock_dataset, mock_model):
        """Test that categorical features are preserved"""
        DBDatasetFactory.create_for_alternative_model(
            original_dataset=mock_dataset,
            model=mock_model
        )
        
        call_kwargs = mock_dbdataset.call_args[1]
        assert call_kwargs['categorical_features'] == ['cat_feat']

    @patch('deepbridge.utils.dataset_factory.DBDataset')
    def test_create_for_alternative_model_with_kwargs(self, mock_dbdataset, mock_dataset, mock_model):
        """Test that kwargs are passed through"""
        DBDatasetFactory.create_for_alternative_model(
            original_dataset=mock_dataset,
            model=mock_model,
            custom_param='value'
        )
        
        call_kwargs = mock_dbdataset.call_args[1]
        assert call_kwargs['custom_param'] == 'value'
