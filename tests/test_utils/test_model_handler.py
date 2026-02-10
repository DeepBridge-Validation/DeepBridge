"""
Tests for ModelHandler class.

Coverage Target: 100%
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from pathlib import Path

from deepbridge.utils.model_handler import ModelHandler


@pytest.fixture
def handler():
    """Create ModelHandler instance"""
    return ModelHandler()


@pytest.fixture
def sample_data():
    """Create sample train/test data"""
    train_data = pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'feature2': [5, 4, 3, 2, 1],
        'target': [0, 1, 0, 1, 0]
    })
    test_data = pd.DataFrame({
        'feature1': [6, 7, 8],
        'feature2': [3, 2, 1],
        'target': [1, 0, 1]
    })
    return {'train': train_data, 'test': test_data}


@pytest.fixture
def mock_model():
    """Create mock model with predict_proba"""
    model = Mock()
    model.predict_proba = Mock(return_value=np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8]]))
    return model


class TestInitialization:
    """Tests for ModelHandler initialization"""

    def test_init_creates_instance(self):
        """Test ModelHandler can be instantiated"""
        handler = ModelHandler()
        assert handler is not None

    def test_init_sets_model_to_none(self, handler):
        """Test initial model is None"""
        assert handler._model is None

    def test_init_sets_predictions_to_none(self, handler):
        """Test initial predictions is None"""
        assert handler._predictions is None

    def test_init_sets_prob_cols_to_none(self, handler):
        """Test initial prob_cols is None"""
        assert handler._prob_cols is None

    def test_init_sets_initialize_predictions_to_false(self, handler):
        """Test initial _initialize_predictions is False"""
        assert handler._initialize_predictions is False

    def test_init_sets_original_predictions_to_none(self, handler):
        """Test initial _original_predictions is None"""
        assert handler._original_predictions is None


class TestModelProperty:
    """Tests for model property"""

    def test_model_getter_returns_model(self, handler):
        """Test model getter returns the model"""
        test_model = Mock()
        handler._model = test_model
        assert handler.model is test_model

    def test_model_setter_sets_model(self, handler):
        """Test model setter sets the model"""
        test_model = Mock()
        handler.model = test_model
        assert handler._model is test_model


class TestPredictionsProperty:
    """Tests for predictions property"""

    def test_predictions_getter_returns_none_initially(self, handler):
        """Test predictions getter returns None initially"""
        assert handler.predictions is None

    def test_predictions_getter_returns_predictions(self, handler):
        """Test predictions getter returns set predictions"""
        preds = pd.DataFrame({'prob_class_0': [0.3], 'prob_class_1': [0.7]})
        handler._predictions = preds
        assert handler.predictions is not None
        assert len(handler.predictions) == 1


class TestProbColsProperty:
    """Tests for prob_cols property"""

    def test_prob_cols_getter_returns_none_initially(self, handler):
        """Test prob_cols getter returns None initially"""
        assert handler.prob_cols is None

    def test_prob_cols_getter_returns_prob_cols(self, handler):
        """Test prob_cols getter returns set prob_cols"""
        cols = ['prob_class_0', 'prob_class_1']
        handler._prob_cols = cols
        assert handler.prob_cols == cols


class TestGeneratePredictions:
    """Tests for generate_predictions method"""

    def test_generate_predictions_raises_error_when_no_model(self, handler, sample_data):
        """Test generate_predictions raises error when no model"""
        with pytest.raises(ValueError, match='No model available'):
            handler.generate_predictions(sample_data, ['feature1', 'feature2'])

    def test_generate_predictions_skips_none_dataset(self, handler, mock_model):
        """Test generate_predictions skips None datasets"""
        handler._model = mock_model
        handler._prob_cols = ['prob_class_0', 'prob_class_1']  # Set prob_cols first
        data = {'train': None, 'test': pd.DataFrame()}

        # Should not raise error
        handler.generate_predictions(data, ['feature1'])

    def test_generate_predictions_skips_empty_dataset(self, handler, mock_model):
        """Test generate_predictions skips empty datasets"""
        handler._model = mock_model
        handler._prob_cols = ['prob_class_0', 'prob_class_1']  # Set prob_cols first
        data = {'train': pd.DataFrame(), 'test': pd.DataFrame()}

        # Should not raise error (will call set_predictions with None values)
        handler.generate_predictions(data, ['feature1'])

    def test_generate_predictions_raises_error_on_predict_failure(self, handler):
        """Test generate_predictions raises error when predict_proba fails"""
        model = Mock()
        model.predict_proba = Mock(side_effect=Exception("Prediction failed"))
        handler._model = model

        data = {'train': pd.DataFrame({'feature1': [1, 2, 3]})}

        with pytest.raises(ValueError, match='Failed to generate predictions for train data'):
            handler.generate_predictions(data, ['feature1'])


class TestLoadModel:
    """Tests for load_model method"""

    @patch('deepbridge.utils.model_handler.load')
    def test_load_model_loads_from_path(self, mock_load, handler):
        """Test load_model loads model from file"""
        mock_model = Mock()
        mock_load.return_value = mock_model

        handler.load_model('/path/to/model.pkl')

        mock_load.assert_called_once_with('/path/to/model.pkl')
        assert handler._model is mock_model

    @patch('deepbridge.utils.model_handler.load')
    def test_load_model_raises_error_on_load_failure(self, mock_load, handler):
        """Test load_model raises error when loading fails"""
        mock_load.side_effect = Exception("Load failed")

        with pytest.raises(ValueError, match='Failed to load model from'):
            handler.load_model('/path/to/model.pkl')


class TestSetPredictions:
    """Tests for set_predictions method"""

    def test_set_predictions_raises_error_when_no_prob_cols(self, handler):
        """Test set_predictions raises error when prob_cols not provided"""
        with pytest.raises(ValueError, match='prob_cols must be provided'):
            handler.set_predictions()

    def test_set_predictions_uses_existing_prob_cols(self, handler):
        """Test set_predictions uses existing prob_cols if not provided"""
        handler._prob_cols = ['prob_class_0', 'prob_class_1']

        # Should not raise error
        handler.set_predictions()

    def test_set_predictions_initializes_from_data_columns(self, handler):
        """Test set_predictions initializes from existing data columns"""
        prob_cols = ['prob_class_0', 'prob_class_1']
        train = pd.DataFrame({
            'feature1': [1, 2],
            'prob_class_0': [0.3, 0.6],
            'prob_class_1': [0.7, 0.4]
        })
        test = pd.DataFrame({
            'feature1': [3],
            'prob_class_0': [0.5],
            'prob_class_1': [0.5]
        })

        handler.set_predictions(train, test, None, None, prob_cols)

        assert handler._initialize_predictions is True
        assert not handler._predictions.empty


class TestValidatePredictions:
    """Tests for _validate_predictions method"""

    def test_validate_predictions_raises_error_for_non_dataframe(self):
        """Test _validate_predictions raises error for non-DataFrame"""
        with pytest.raises(ValueError, match='must be a pandas DataFrame'):
            ModelHandler._validate_predictions(
                "not a dataframe",
                pd.DataFrame(),
                ['prob_class_0'],
                'train'
            )

    def test_validate_predictions_raises_error_for_missing_columns(self):
        """Test _validate_predictions raises error for missing probability columns"""
        pred_df = pd.DataFrame({'wrong_col': [0.5]})
        data_df = pd.DataFrame({'feature1': [1]})

        with pytest.raises(ValueError, match='Probability columns .* not found'):
            ModelHandler._validate_predictions(
                pred_df,
                data_df,
                ['prob_class_0', 'prob_class_1'],
                'train'
            )

    def test_validate_predictions_raises_error_for_length_mismatch(self):
        """Test _validate_predictions raises error for length mismatch"""
        pred_df = pd.DataFrame({'prob_class_0': [0.3, 0.5]})
        data_df = pd.DataFrame({'feature1': [1]})

        with pytest.raises(ValueError, match='Length of .* must match'):
            ModelHandler._validate_predictions(
                pred_df,
                data_df,
                ['prob_class_0'],
                'train'
            )

    def test_validate_predictions_passes_for_valid_data(self):
        """Test _validate_predictions passes for valid data"""
        pred_df = pd.DataFrame({'prob_class_0': [0.3], 'prob_class_1': [0.7]})
        data_df = pd.DataFrame({'feature1': [1]})

        # Should not raise
        ModelHandler._validate_predictions(
            pred_df,
            data_df,
            ['prob_class_0', 'prob_class_1'],
            'train'
        )


class TestReset:
    """Tests for reset method"""

    def test_reset_clears_model(self, handler):
        """Test reset clears the model"""
        handler._model = Mock()
        handler.reset()
        assert handler._model is None

    def test_reset_clears_predictions(self, handler):
        """Test reset clears predictions"""
        handler._predictions = pd.DataFrame()
        handler.reset()
        assert handler._predictions is None

    def test_reset_clears_prob_cols(self, handler):
        """Test reset clears prob_cols"""
        handler._prob_cols = ['prob_class_0']
        handler.reset()
        assert handler._prob_cols is None

    def test_reset_clears_initialize_predictions_flag(self, handler):
        """Test reset clears initialize_predictions flag"""
        handler._initialize_predictions = True
        handler.reset()
        assert handler._initialize_predictions is False

    def test_reset_clears_original_predictions(self, handler):
        """Test reset clears original_predictions"""
        handler._original_predictions = pd.DataFrame()
        handler.reset()
        assert handler._original_predictions is None


class TestIntegration:
    """Integration tests for ModelHandler"""

    @patch('deepbridge.utils.model_handler.load')
    def test_full_workflow(self, mock_load, sample_data):
        """Test complete workflow: load model, generate predictions"""
        # Create handler
        handler = ModelHandler()

        # Create mock model
        model = Mock()
        # Return different predictions for train (5 rows) and test (3 rows)
        model.predict_proba = Mock(side_effect=[
            np.array([[0.3, 0.7], [0.6, 0.4], [0.2, 0.8], [0.5, 0.5], [0.1, 0.9]]),
            np.array([[0.4, 0.6], [0.7, 0.3], [0.3, 0.7]])
        ])
        mock_load.return_value = model

        # Load model and generate predictions
        handler.load_model('/path/to/model.pkl',
                          features=['feature1', 'feature2'],
                          data=sample_data)

        # Verify model was loaded
        assert handler.model is model

        # Verify predictions were generated
        assert handler.predictions is not None
        assert len(handler.predictions) == 8  # 5 train + 3 test

    def test_set_predictions_with_validation(self):
        """Test set_predictions with full validation"""
        handler = ModelHandler()

        train_data = pd.DataFrame({'feature1': [1, 2]})
        test_data = pd.DataFrame({'feature1': [3]})
        train_preds = pd.DataFrame({
            'prob_class_0': [0.3, 0.6],
            'prob_class_1': [0.7, 0.4]
        })
        test_preds = pd.DataFrame({
            'prob_class_0': [0.5],
            'prob_class_1': [0.5]
        })

        handler.set_predictions(
            train_data, test_data, train_preds, test_preds,
            ['prob_class_0', 'prob_class_1']
        )

        assert len(handler.predictions) == 3
        assert handler.prob_cols == ['prob_class_0', 'prob_class_1']
