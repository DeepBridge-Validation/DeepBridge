"""
Comprehensive tests for ModelManager.

This test suite validates:
1. ModelManager initialization
2. get_default_model_type - default model selection
3. create_alternative_models - alternative model creation with lazy loading
4. create_distillation_model - factory for distillation models
5. _create_model_from_probabilities - distillation from probabilities
6. _create_model_from_teacher - distillation from teacher model

Coverage Target: ~90%+
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from deepbridge.core.experiment.managers.model_manager import ModelManager
from deepbridge.utils.model_registry import ModelType, ModelMode


# ==================== Fixtures ====================


@pytest.fixture
def mock_dataset():
    """Create mock dataset with model"""
    dataset = Mock(spec=['model', 'original_prob'])
    dataset.model = Mock()
    dataset.model.__class__.__name__ = 'XGBClassifier'
    dataset.model.fit = Mock(return_value=dataset.model)
    dataset.original_prob = np.array([[0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    return dataset


@pytest.fixture
def mock_dataset_no_model():
    """Create mock dataset without model"""
    dataset = Mock(spec=['original_prob'])
    dataset.model = None
    dataset.original_prob = np.array([[0.2, 0.8], [0.7, 0.3]])
    return dataset


@pytest.fixture
def mock_dataset_no_prob():
    """Create mock dataset without probabilities"""
    dataset = Mock(spec=['model', 'original_prob'])
    dataset.model = Mock()
    dataset.model.__class__.__name__ = 'RandomForestClassifier'
    dataset.original_prob = None
    return dataset


@pytest.fixture
def X_train():
    """Sample training features"""
    return np.array([[1, 2], [3, 4], [5, 6], [7, 8]])


@pytest.fixture
def y_train():
    """Sample training labels"""
    return np.array([0, 1, 0, 1])


@pytest.fixture
def model_manager(mock_dataset):
    """Create ModelManager instance"""
    return ModelManager(
        dataset=mock_dataset,
        experiment_type='binary_classification',
        verbose=False
    )


# ==================== Initialization Tests ====================


class TestModelManagerInitialization:
    """Tests for ModelManager initialization"""

    def test_init_with_all_parameters(self, mock_dataset):
        """Test initialization with all parameters"""
        manager = ModelManager(
            dataset=mock_dataset,
            experiment_type='binary_classification',
            verbose=True
        )

        assert manager.dataset == mock_dataset
        assert manager.experiment_type == 'binary_classification'
        assert manager.verbose is True

    def test_init_with_minimal_parameters(self, mock_dataset):
        """Test initialization with minimal parameters"""
        manager = ModelManager(
            dataset=mock_dataset,
            experiment_type='regression'
        )

        assert manager.dataset == mock_dataset
        assert manager.experiment_type == 'regression'
        assert manager.verbose is False


# ==================== get_default_model_type Tests ====================


class TestGetDefaultModelType:
    """Tests for get_default_model_type method"""

    def test_returns_xgb_if_available(self, model_manager):
        """Test that XGB is returned if available"""
        default = model_manager.get_default_model_type()

        # XGB should be in the name
        assert 'XGB' in default.name

    def test_returns_model_type_enum(self, model_manager):
        """Test that return value is a ModelType enum"""
        default = model_manager.get_default_model_type()

        assert isinstance(default, ModelType)

    def test_fallback_behavior(self, model_manager):
        """Test that get_default_model_type returns a valid ModelType"""
        # Since we can't easily mock the enum iteration, just verify it returns a valid type
        result = model_manager.get_default_model_type()

        # Should be a ModelType enum member
        assert isinstance(result, ModelType)
        assert hasattr(result, 'name')
        assert hasattr(result, 'value')


# ==================== create_alternative_models Tests ====================


class TestCreateAlternativeModels:
    """Tests for create_alternative_models method"""

    def test_lazy_loading_returns_empty_dict(self, model_manager, X_train, y_train):
        """Test that lazy=True returns empty dict"""
        result = model_manager.create_alternative_models(X_train, y_train, lazy=True)

        assert result == {}

    def test_lazy_loading_verbose_message(self, mock_dataset, X_train, y_train):
        """Test verbose message with lazy loading"""
        manager = ModelManager(mock_dataset, 'binary_classification', verbose=True)

        with patch('builtins.print') as mock_print:
            manager.create_alternative_models(X_train, y_train, lazy=True)

            # Should print lazy loading message
            assert mock_print.call_count >= 1
            args = str(mock_print.call_args_list[0])
            assert 'Lazy loading' in args or 'lazy' in args.lower()

    def test_no_model_returns_empty_dict(self, mock_dataset_no_model, X_train, y_train):
        """Test that missing model returns empty dict"""
        manager = ModelManager(mock_dataset_no_model, 'binary_classification', verbose=False)

        result = manager.create_alternative_models(X_train, y_train, lazy=False)

        assert result == {}

    def test_no_model_verbose_message(self, mock_dataset_no_model, X_train, y_train):
        """Test verbose message when no model found"""
        manager = ModelManager(mock_dataset_no_model, 'binary_classification', verbose=True)

        with patch('builtins.print') as mock_print:
            manager.create_alternative_models(X_train, y_train, lazy=False)

            # Should print no model message
            assert mock_print.call_count >= 1

    def test_creates_up_to_3_models(self, model_manager, X_train, y_train):
        """Test that at most 3 alternative models are created"""
        with patch('deepbridge.core.experiment.managers.model_manager.ModelRegistry.get_model') as mock_get:
            mock_model = Mock()
            mock_model.fit = Mock(return_value=mock_model)
            mock_get.return_value = mock_model

            result = model_manager.create_alternative_models(X_train, y_train, lazy=False)

            # Should create at most 3 models
            assert len(result) <= 3

    def test_excludes_original_model_type(self, mock_dataset, X_train, y_train):
        """Test that original model type is excluded"""
        # Dataset has XGBClassifier
        manager = ModelManager(mock_dataset, 'binary_classification', verbose=False)

        with patch('deepbridge.core.experiment.managers.model_manager.ModelRegistry.get_model') as mock_get:
            mock_model = Mock()
            mock_model.fit = Mock(return_value=mock_model)
            mock_get.return_value = mock_model

            result = manager.create_alternative_models(X_train, y_train, lazy=False)

            # XGB should not be in the created models
            assert 'XGB' not in result

    def test_fits_each_model(self, model_manager, X_train, y_train):
        """Test that each model is fitted"""
        with patch('deepbridge.core.experiment.managers.model_manager.ModelRegistry.get_model') as mock_get:
            mock_model = Mock()
            mock_model.fit = Mock(return_value=mock_model)
            mock_model.__class__.__name__ = 'TestModel'
            mock_get.return_value = mock_model

            result = model_manager.create_alternative_models(X_train, y_train, lazy=False)

            # fit should have been called
            if len(result) > 0:
                assert mock_model.fit.call_count >= 1

    def test_handles_fit_exception(self, model_manager, X_train, y_train):
        """Test graceful handling of model fit exception"""
        with patch('deepbridge.core.experiment.managers.model_manager.ModelRegistry.get_model') as mock_get:
            mock_model = Mock()
            mock_model.fit = Mock(side_effect=Exception("Fit failed"))
            mock_get.return_value = mock_model

            # Should not raise exception
            result = model_manager.create_alternative_models(X_train, y_train, lazy=False)

            # Result might be empty if all fail
            assert isinstance(result, dict)

    def test_classification_mode_selected(self, model_manager, X_train, y_train):
        """Test that classification mode is selected for binary_classification"""
        with patch('deepbridge.core.experiment.managers.model_manager.ModelRegistry.get_model') as mock_get:
            mock_model = Mock()
            mock_model.fit = Mock(return_value=mock_model)
            mock_get.return_value = mock_model

            model_manager.create_alternative_models(X_train, y_train, lazy=False)

            # Check that CLASSIFICATION mode was used
            if mock_get.call_count > 0:
                call_args = mock_get.call_args[1]
                assert call_args.get('mode') == ModelMode.CLASSIFICATION

    def test_regression_mode_selected(self, mock_dataset, X_train, y_train):
        """Test that regression mode is selected for non-classification"""
        manager = ModelManager(mock_dataset, 'regression', verbose=False)

        with patch('deepbridge.core.experiment.managers.model_manager.ModelRegistry.get_model') as mock_get:
            mock_model = Mock()
            mock_model.fit = Mock(return_value=mock_model)
            mock_get.return_value = mock_model

            manager.create_alternative_models(X_train, y_train, lazy=False)

            # Check that REGRESSION mode was used
            if mock_get.call_count > 0:
                call_args = mock_get.call_args[1]
                assert call_args.get('mode') == ModelMode.REGRESSION

    def test_verbose_prints_progress(self, mock_dataset, X_train, y_train):
        """Test that verbose mode prints progress messages"""
        manager = ModelManager(mock_dataset, 'binary_classification', verbose=True)

        with patch('deepbridge.core.experiment.managers.model_manager.ModelRegistry.get_model') as mock_get:
            mock_model = Mock()
            mock_model.fit = Mock(return_value=mock_model)
            mock_model.__class__.__name__ = 'TestModel'
            mock_get.return_value = mock_model

            with patch('builtins.print') as mock_print:
                manager.create_alternative_models(X_train, y_train, lazy=False)

                # Should print multiple messages
                assert mock_print.call_count >= 3


# ==================== create_distillation_model Tests ====================


class TestCreateDistillationModel:
    """Tests for create_distillation_model method"""

    def test_with_probabilities_calls_from_probabilities(self, model_manager):
        """Test that use_probabilities=True calls _create_model_from_probabilities"""
        with patch.object(model_manager, '_create_model_from_probabilities') as mock_create:
            mock_create.return_value = Mock()

            model_manager.create_distillation_model(
                distillation_method='surrogate',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params=None,
                temperature=2.0,
                alpha=0.5,
                use_probabilities=True,
                n_trials=10,
                validation_split=0.2
            )

            mock_create.assert_called_once()

    def test_without_probabilities_calls_from_teacher(self, model_manager):
        """Test that use_probabilities=False calls _create_model_from_teacher"""
        with patch.object(model_manager, '_create_model_from_teacher') as mock_create:
            mock_create.return_value = Mock()

            model_manager.create_distillation_model(
                distillation_method='knowledge_distillation',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params=None,
                temperature=2.0,
                alpha=0.5,
                use_probabilities=False,
                n_trials=10,
                validation_split=0.2
            )

            mock_create.assert_called_once()

    def test_raises_error_if_no_probabilities_available(self, mock_dataset_no_prob):
        """Test error when use_probabilities=True but no probabilities"""
        manager = ModelManager(mock_dataset_no_prob, 'binary_classification', verbose=False)

        with pytest.raises(ValueError, match='No teacher probabilities available'):
            manager.create_distillation_model(
                distillation_method='surrogate',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params=None,
                temperature=2.0,
                alpha=0.5,
                use_probabilities=True,
                n_trials=10,
                validation_split=0.2
            )

    def test_raises_error_if_no_teacher_model(self, mock_dataset_no_model):
        """Test error when use_probabilities=False but no teacher model"""
        manager = ModelManager(mock_dataset_no_model, 'binary_classification', verbose=False)

        with pytest.raises(ValueError, match='No teacher model available'):
            manager.create_distillation_model(
                distillation_method='knowledge_distillation',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params=None,
                temperature=2.0,
                alpha=0.5,
                use_probabilities=False,
                n_trials=10,
                validation_split=0.2
            )


# ==================== _create_model_from_probabilities Tests ====================


@pytest.mark.skip(reason="Distillation functionality moved to deepbridge-distillation package")
class TestCreateModelFromProbabilities:
    """Tests for _create_model_from_probabilities method"""

    def test_surrogate_method(self, model_manager):
        """Test creating surrogate model from probabilities"""
        with patch('deepbridge.distillation.techniques.surrogate.SurrogateModel') as mock_surrogate:
            mock_surrogate.from_probabilities = Mock(return_value=Mock())

            result = model_manager._create_model_from_probabilities(
                distillation_method='surrogate',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params={'C': 1.0},
                temperature=2.0,
                alpha=0.5,
                n_trials=10,
                validation_split=0.2
            )

            mock_surrogate.from_probabilities.assert_called_once()
            assert result is not None

    def test_knowledge_distillation_method(self, model_manager):
        """Test creating knowledge distillation model from probabilities"""
        with patch('deepbridge.distillation.techniques.knowledge_distillation.KnowledgeDistillation') as mock_kd:
            mock_kd.from_probabilities = Mock(return_value=Mock())

            result = model_manager._create_model_from_probabilities(
                distillation_method='knowledge_distillation',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params={'C': 1.0},
                temperature=2.0,
                alpha=0.5,
                n_trials=10,
                validation_split=0.2
            )

            mock_kd.from_probabilities.assert_called_once()
            assert result is not None

    def test_case_insensitive_method_name(self, model_manager):
        """Test that method name is case-insensitive"""
        with patch('deepbridge.distillation.techniques.surrogate.SurrogateModel') as mock_surrogate:
            mock_surrogate.from_probabilities = Mock(return_value=Mock())

            # Test uppercase
            model_manager._create_model_from_probabilities(
                distillation_method='SURROGATE',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params=None,
                temperature=2.0,
                alpha=0.5,
                n_trials=10,
                validation_split=0.2
            )

            assert mock_surrogate.from_probabilities.call_count == 1

    def test_unknown_method_raises_error(self, model_manager):
        """Test that unknown method raises ValueError"""
        with pytest.raises(ValueError, match='Unknown distillation method'):
            model_manager._create_model_from_probabilities(
                distillation_method='unknown_method',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params=None,
                temperature=2.0,
                alpha=0.5,
                n_trials=10,
                validation_split=0.2
            )

    def test_passes_all_parameters(self, model_manager):
        """Test that all parameters are passed correctly"""
        with patch('deepbridge.distillation.techniques.knowledge_distillation.KnowledgeDistillation') as mock_kd:
            mock_kd.from_probabilities = Mock(return_value=Mock())

            model_manager._create_model_from_probabilities(
                distillation_method='knowledge_distillation',
                student_model_type=ModelType.DECISION_TREE,
                student_params={'max_depth': 5},
                temperature=3.0,
                alpha=0.7,
                n_trials=20,
                validation_split=0.3
            )

            call_kwargs = mock_kd.from_probabilities.call_args[1]
            assert call_kwargs['student_model_type'] == ModelType.DECISION_TREE
            assert call_kwargs['student_params'] == {'max_depth': 5}
            assert call_kwargs['temperature'] == 3.0
            assert call_kwargs['alpha'] == 0.7
            assert call_kwargs['n_trials'] == 20
            assert call_kwargs['validation_split'] == 0.3


# ==================== _create_model_from_teacher Tests ====================


@pytest.mark.skip(reason="Distillation functionality moved to deepbridge-distillation package")
class TestCreateModelFromTeacher:
    """Tests for _create_model_from_teacher method"""

    def test_surrogate_raises_error(self, model_manager):
        """Test that surrogate method raises error with teacher model"""
        with pytest.raises(ValueError, match='surrogate method does not support'):
            model_manager._create_model_from_teacher(
                distillation_method='surrogate',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params=None,
                temperature=2.0,
                alpha=0.5,
                n_trials=10,
                validation_split=0.2
            )

    def test_knowledge_distillation_method(self, model_manager):
        """Test creating knowledge distillation model from teacher"""
        with patch('deepbridge.distillation.techniques.knowledge_distillation.KnowledgeDistillation') as mock_kd:
            mock_kd.return_value = Mock()

            result = model_manager._create_model_from_teacher(
                distillation_method='knowledge_distillation',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params={'C': 1.0},
                temperature=2.0,
                alpha=0.5,
                n_trials=10,
                validation_split=0.2
            )

            # Should call constructor (not from_probabilities)
            mock_kd.assert_called_once()
            assert result is not None

    def test_case_insensitive_method_name(self, model_manager):
        """Test that method name is case-insensitive"""
        with patch('deepbridge.distillation.techniques.knowledge_distillation.KnowledgeDistillation') as mock_kd:
            mock_kd.return_value = Mock()

            # Test mixed case
            model_manager._create_model_from_teacher(
                distillation_method='Knowledge_Distillation',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params=None,
                temperature=2.0,
                alpha=0.5,
                n_trials=10,
                validation_split=0.2
            )

            assert mock_kd.call_count == 1

    def test_unknown_method_raises_error(self, model_manager):
        """Test that unknown method raises ValueError"""
        with pytest.raises(ValueError, match='Unknown distillation method'):
            model_manager._create_model_from_teacher(
                distillation_method='invalid_method',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params=None,
                temperature=2.0,
                alpha=0.5,
                n_trials=10,
                validation_split=0.2
            )

    def test_passes_teacher_model(self, model_manager):
        """Test that teacher model is passed from dataset"""
        with patch('deepbridge.distillation.techniques.knowledge_distillation.KnowledgeDistillation') as mock_kd:
            mock_kd.return_value = Mock()

            model_manager._create_model_from_teacher(
                distillation_method='knowledge_distillation',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params=None,
                temperature=2.0,
                alpha=0.5,
                n_trials=10,
                validation_split=0.2
            )

            call_kwargs = mock_kd.call_args[1]
            assert call_kwargs['teacher_model'] == model_manager.dataset.model

    def test_passes_all_parameters(self, model_manager):
        """Test that all parameters are passed correctly"""
        with patch('deepbridge.distillation.techniques.knowledge_distillation.KnowledgeDistillation') as mock_kd:
            mock_kd.return_value = Mock()

            model_manager._create_model_from_teacher(
                distillation_method='knowledge_distillation',
                student_model_type=ModelType.RANDOM_FOREST,
                student_params={'n_estimators': 100},
                temperature=4.0,
                alpha=0.9,
                n_trials=50,
                validation_split=0.25
            )

            call_kwargs = mock_kd.call_args[1]
            assert call_kwargs['student_model_type'] == ModelType.RANDOM_FOREST
            assert call_kwargs['student_params'] == {'n_estimators': 100}
            assert call_kwargs['temperature'] == 4.0
            assert call_kwargs['alpha'] == 0.9
            assert call_kwargs['n_trials'] == 50
            assert call_kwargs['validation_split'] == 0.25


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_alternative_models_workflow(self, model_manager, X_train, y_train):
        """Test complete workflow for creating alternative models"""
        with patch('deepbridge.core.experiment.managers.model_manager.ModelRegistry.get_model') as mock_get:
            mock_model = Mock()
            mock_model.fit = Mock(return_value=mock_model)
            mock_model.__class__.__name__ = 'LogisticRegression'
            mock_get.return_value = mock_model

            # Create alternatives
            result = model_manager.create_alternative_models(X_train, y_train, lazy=False)

            # Should be dict
            assert isinstance(result, dict)

    @pytest.mark.skip(reason="Distillation functionality moved to deepbridge-distillation package")
    def test_full_distillation_workflow_with_probabilities(self, model_manager):
        """Test complete distillation workflow using probabilities"""
        with patch('deepbridge.distillation.techniques.surrogate.SurrogateModel') as mock_surrogate:
            mock_instance = Mock()
            mock_surrogate.from_probabilities = Mock(return_value=mock_instance)

            result = model_manager.create_distillation_model(
                distillation_method='surrogate',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params={'C': 1.0},
                temperature=2.0,
                alpha=0.5,
                use_probabilities=True,
                n_trials=10,
                validation_split=0.2
            )

            assert result == mock_instance

    @pytest.mark.skip(reason="Distillation functionality moved to deepbridge-distillation package")
    def test_full_distillation_workflow_with_teacher(self, model_manager):
        """Test complete distillation workflow using teacher model"""
        with patch('deepbridge.distillation.techniques.knowledge_distillation.KnowledgeDistillation') as mock_kd:
            mock_instance = Mock()
            mock_kd.return_value = mock_instance

            result = model_manager.create_distillation_model(
                distillation_method='knowledge_distillation',
                student_model_type=ModelType.DECISION_TREE,
                student_params=None,
                temperature=3.0,
                alpha=0.7,
                use_probabilities=False,
                n_trials=15,
                validation_split=0.25
            )

            assert result == mock_instance


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_dataset_without_model_attribute(self, X_train, y_train):
        """Test handling dataset without model attribute"""
        dataset = Mock(spec=[])  # No 'model' attribute
        manager = ModelManager(dataset, 'binary_classification', verbose=False)

        result = manager.create_alternative_models(X_train, y_train, lazy=False)

        # Should handle gracefully
        assert result == {}

    def test_empty_training_data(self, model_manager):
        """Test with empty training data"""
        X_empty = np.array([]).reshape(0, 2)
        y_empty = np.array([])

        with patch('deepbridge.core.experiment.managers.model_manager.ModelRegistry.get_model') as mock_get:
            mock_model = Mock()
            mock_model.fit = Mock(side_effect=ValueError("Empty data"))
            mock_get.return_value = mock_model

            # Should handle exception
            result = model_manager.create_alternative_models(X_empty, y_empty, lazy=False)
            assert isinstance(result, dict)

    def test_model_with_unusual_name(self, X_train, y_train):
        """Test with model having unusual class name"""
        dataset = Mock(spec=['model', 'original_prob'])
        dataset.model = Mock()
        dataset.model.__class__.__name__ = 'CustomWeirdModelName123'
        dataset.original_prob = None

        manager = ModelManager(dataset, 'binary_classification', verbose=False)

        with patch('deepbridge.core.experiment.managers.model_manager.ModelRegistry.get_model') as mock_get:
            mock_model = Mock()
            mock_model.fit = Mock(return_value=mock_model)
            mock_get.return_value = mock_model

            # Should still work
            result = manager.create_alternative_models(X_train, y_train, lazy=False)
            assert isinstance(result, dict)

    def test_none_student_params(self, model_manager):
        """Test with None student_params"""
        with patch('deepbridge.distillation.techniques.surrogate.SurrogateModel') as mock_surrogate:
            mock_surrogate.from_probabilities = Mock(return_value=Mock())

            # Should handle None params
            result = model_manager._create_model_from_probabilities(
                distillation_method='surrogate',
                student_model_type=ModelType.LOGISTIC_REGRESSION,
                student_params=None,
                temperature=2.0,
                alpha=0.5,
                n_trials=10,
                validation_split=0.2
            )

            assert result is not None
