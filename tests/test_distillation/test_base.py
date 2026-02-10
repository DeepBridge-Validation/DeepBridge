"""
Comprehensive tests for base distillation classes.

This test suite validates:
1. BaseDistiller initialization
2. Concrete methods (predict_proba, evaluate, __repr__)
3. Abstract method enforcement
4. Evaluate with Classification metrics
5. Evaluate with fallback sklearn metrics

Coverage Target: ~95%+
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from sklearn.datasets import make_classification

from deepbridge.distillation.base import BaseDistiller
from deepbridge.utils.model_registry import ModelType


# ==================== Concrete Implementation for Testing ====================


class ConcreteDistiller(BaseDistiller):
    """Concrete implementation of BaseDistiller for testing purposes"""

    def fit(self, X, y, verbose=True):
        """Implement abstract fit method"""
        self.is_fitted = True
        self.best_params = {'learning_rate': 0.01, 'n_estimators': 100}
        return self

    def predict(self, X):
        """Implement abstract predict method - returns probabilities"""
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        # Return 2D probabilities for binary classification
        probs = np.random.RandomState(42).rand(n_samples, 2)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    @classmethod
    def from_probabilities(cls, probabilities, student_model_type=None, student_params=None, **kwargs):
        """Implement abstract from_probabilities method"""
        instance = cls(student_model_type=student_model_type, student_params=student_params)
        instance.probabilities = probabilities
        instance.is_fitted = True
        return instance


# ==================== Fixtures ====================


@pytest.fixture
def mock_teacher_model():
    """Create a mock teacher model"""
    model = Mock()
    model.predict_proba = Mock(return_value=np.array([[0.2, 0.8], [0.7, 0.3]]))
    return model


@pytest.fixture
def binary_classification_data():
    """Generate binary classification dataset"""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_classes=2,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    return X, y


# ==================== Initialization Tests ====================


class TestBaseDistillerInitialization:
    """Tests for BaseDistiller initialization"""

    def test_init_with_all_params(self, mock_teacher_model):
        """Test initialization with all parameters"""
        student_params = {'max_depth': 5, 'n_estimators': 50}
        distiller = ConcreteDistiller(
            teacher_model=mock_teacher_model,
            student_model_type=ModelType.RANDOM_FOREST,
            student_params=student_params,
            random_state=123
        )

        assert distiller.teacher_model == mock_teacher_model
        assert distiller.student_model_type == ModelType.RANDOM_FOREST
        assert distiller.student_params == student_params
        assert distiller.random_state == 123
        assert distiller.student_model is None
        assert distiller.is_fitted is False
        assert distiller.best_params is None

    def test_init_with_minimal_params(self):
        """Test initialization with minimal parameters"""
        distiller = ConcreteDistiller()

        assert distiller.teacher_model is None
        assert distiller.student_model_type is None
        assert distiller.student_params == {}
        assert distiller.random_state == 42  # default
        assert distiller.is_fitted is False

    def test_init_without_student_params(self, mock_teacher_model):
        """Test that student_params defaults to empty dict"""
        distiller = ConcreteDistiller(teacher_model=mock_teacher_model)

        assert distiller.student_params == {}

    def test_init_with_none_student_params(self):
        """Test explicit None for student_params becomes empty dict"""
        distiller = ConcreteDistiller(student_params=None)

        assert distiller.student_params == {}


# ==================== Abstract Method Tests ====================


class TestAbstractMethods:
    """Tests for abstract method enforcement"""

    def test_cannot_instantiate_base_distiller_directly(self):
        """Test that BaseDistiller cannot be instantiated directly"""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseDistiller()

    def test_concrete_class_must_implement_fit(self):
        """Test that concrete classes must implement fit"""
        class IncompleteDistiller(BaseDistiller):
            def predict(self, X):
                pass

            @classmethod
            def from_probabilities(cls, probabilities, **kwargs):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDistiller()

    def test_concrete_class_must_implement_predict(self):
        """Test that concrete classes must implement predict"""
        class IncompleteDistiller(BaseDistiller):
            def fit(self, X, y, verbose=True):
                pass

            @classmethod
            def from_probabilities(cls, probabilities, **kwargs):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDistiller()

    def test_concrete_class_must_implement_from_probabilities(self):
        """Test that concrete classes must implement from_probabilities"""
        class IncompleteDistiller(BaseDistiller):
            def fit(self, X, y, verbose=True):
                pass

            def predict(self, X):
                pass

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteDistiller()


# ==================== Concrete Method Tests ====================


class TestPredictProba:
    """Tests for predict_proba method"""

    def test_predict_proba_delegates_to_predict(self, binary_classification_data):
        """Test that predict_proba delegates to predict by default"""
        X, y = binary_classification_data
        distiller = ConcreteDistiller()
        distiller.fit(X, y)

        # predict_proba should return same as predict
        proba_result = distiller.predict_proba(X[:10])
        predict_result = distiller.predict(X[:10])

        np.testing.assert_array_equal(proba_result, predict_result)

    def test_predict_proba_shape(self, binary_classification_data):
        """Test that predict_proba returns correct shape"""
        X, y = binary_classification_data
        distiller = ConcreteDistiller()
        distiller.fit(X, y)

        result = distiller.predict_proba(X[:10])

        assert result.shape == (10, 2)  # (n_samples, n_classes)


class TestRepr:
    """Tests for __repr__ method"""

    def test_repr_not_fitted(self):
        """Test __repr__ for unfitted distiller"""
        distiller = ConcreteDistiller(student_model_type=ModelType.LOGISTIC_REGRESSION)

        repr_str = repr(distiller)

        assert 'ConcreteDistiller' in repr_str
        assert 'not fitted' in repr_str
        assert 'LOGISTIC_REGRESSION' in repr_str or 'LogisticRegression' in repr_str

    def test_repr_fitted(self, binary_classification_data):
        """Test __repr__ for fitted distiller"""
        X, y = binary_classification_data
        distiller = ConcreteDistiller(student_model_type=ModelType.RANDOM_FOREST)
        distiller.fit(X, y)

        repr_str = repr(distiller)

        assert 'ConcreteDistiller' in repr_str
        assert 'fitted' in repr_str
        assert 'not fitted' not in repr_str


# ==================== Evaluate Method Tests ====================


class TestEvaluateWithClassificationMetrics:
    """Tests for evaluate method using Classification metrics"""

    def test_evaluate_with_classification_module(self, binary_classification_data):
        """Test evaluate using deepbridge Classification metrics"""
        X, y = binary_classification_data
        distiller = ConcreteDistiller()
        distiller.fit(X, y)

        # Mock the Classification module
        with patch('deepbridge.metrics.classification.Classification') as mock_classification:
            mock_calculator = Mock()
            mock_metrics = {
                'accuracy': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85,
                'roc_auc': 0.90
            }
            mock_calculator.calculate_metrics.return_value = mock_metrics
            mock_classification.return_value = mock_calculator

            result = distiller.evaluate(X, y)

            # Verify Classification was used
            mock_classification.assert_called_once()
            mock_calculator.calculate_metrics.assert_called_once()

            # Verify metrics are returned
            assert result['accuracy'] == 0.85
            assert result['f1_score'] == 0.85
            assert 'best_params' in result
            assert result['best_params'] == {'learning_rate': 0.01, 'n_estimators': 100}

    def test_evaluate_includes_teacher_probs(self, binary_classification_data):
        """Test evaluate passes teacher probabilities to metrics calculator"""
        X, y = binary_classification_data
        distiller = ConcreteDistiller()
        distiller.fit(X, y)

        teacher_probs = np.random.rand(len(X))

        with patch('deepbridge.metrics.classification.Classification') as mock_classification:
            mock_calculator = Mock()
            mock_calculator.calculate_metrics.return_value = {'accuracy': 0.85}
            mock_classification.return_value = mock_calculator

            result = distiller.evaluate(X, y, teacher_probs=teacher_probs)

            # Verify teacher_prob was passed
            call_args = mock_calculator.calculate_metrics.call_args
            assert call_args[1]['teacher_prob'] is not None

    def test_evaluate_handles_2d_probabilities(self, binary_classification_data):
        """Test evaluate correctly extracts class 1 probabilities from 2D array"""
        X, y = binary_classification_data
        distiller = ConcreteDistiller()
        distiller.fit(X, y)

        with patch('deepbridge.metrics.classification.Classification') as mock_classification:
            mock_calculator = Mock()
            mock_calculator.calculate_metrics.return_value = {'accuracy': 0.85}
            mock_classification.return_value = mock_calculator

            result = distiller.evaluate(X, y)

            # Verify y_prob is 1D (class 1 probabilities)
            call_args = mock_calculator.calculate_metrics.call_args
            y_prob = call_args[1]['y_prob']
            assert y_prob.ndim == 1


class TestEvaluateWithSklearnFallback:
    """Tests for evaluate method using sklearn fallback metrics"""

    def test_evaluate_fallback_when_classification_not_available(self, binary_classification_data):
        """Test evaluate falls back to sklearn when Classification module unavailable"""
        X, y = binary_classification_data
        distiller = ConcreteDistiller()
        distiller.fit(X, y)

        # Mock ImportError for Classification module
        with patch('deepbridge.metrics.classification.Classification', side_effect=ImportError):
            result = distiller.evaluate(X, y)

            # Verify fallback metrics are present
            assert 'accuracy' in result
            assert 'precision' in result
            assert 'recall' in result
            assert 'f1_score' in result
            assert 'best_params' in result

            # Verify metrics are floats
            assert isinstance(result['accuracy'], float)
            assert isinstance(result['precision'], float)

    def test_evaluate_fallback_metrics_are_reasonable(self, binary_classification_data):
        """Test that fallback metrics produce reasonable values"""
        X, y = binary_classification_data
        distiller = ConcreteDistiller()
        distiller.fit(X, y)

        with patch('deepbridge.metrics.classification.Classification', side_effect=ImportError):
            result = distiller.evaluate(X, y)

            # Metrics should be between 0 and 1
            assert 0 <= result['accuracy'] <= 1
            assert 0 <= result['precision'] <= 1
            assert 0 <= result['recall'] <= 1
            assert 0 <= result['f1_score'] <= 1


class TestEvaluateErrorHandling:
    """Tests for evaluate method error handling"""

    def test_evaluate_raises_error_when_not_fitted(self, binary_classification_data):
        """Test evaluate raises ValueError when model not fitted"""
        X, y = binary_classification_data
        distiller = ConcreteDistiller()

        with pytest.raises(ValueError, match='Model must be fitted'):
            distiller.evaluate(X, y)

    def test_evaluate_without_best_params(self, binary_classification_data):
        """Test evaluate works when best_params is None"""
        X, y = binary_classification_data
        distiller = ConcreteDistiller()
        distiller.best_params = None
        distiller.fit(X, y)
        distiller.best_params = None  # Reset after fit

        with patch('deepbridge.metrics.classification.Classification', side_effect=ImportError):
            result = distiller.evaluate(X, y)

            # Should not have best_params in result
            assert 'best_params' not in result or result.get('best_params') is None


# ==================== From Probabilities Tests ====================


class TestFromProbabilities:
    """Tests for from_probabilities class method"""

    def test_from_probabilities_basic(self):
        """Test from_probabilities creates instance correctly"""
        probabilities = np.array([[0.3, 0.7], [0.8, 0.2], [0.5, 0.5]])

        distiller = ConcreteDistiller.from_probabilities(
            probabilities=probabilities,
            student_model_type=ModelType.DECISION_TREE,
            student_params={'max_depth': 3}
        )

        assert distiller.student_model_type == ModelType.DECISION_TREE
        assert distiller.student_params == {'max_depth': 3}
        assert distiller.is_fitted is True
        np.testing.assert_array_equal(distiller.probabilities, probabilities)

    def test_from_probabilities_with_dataframe(self):
        """Test from_probabilities works with DataFrame"""
        probabilities = pd.DataFrame({
            'class_0': [0.3, 0.8, 0.5],
            'class_1': [0.7, 0.2, 0.5]
        })

        distiller = ConcreteDistiller.from_probabilities(probabilities=probabilities)

        assert distiller.is_fitted is True
        pd.testing.assert_frame_equal(distiller.probabilities, probabilities)

    def test_from_probabilities_minimal(self):
        """Test from_probabilities with minimal parameters"""
        probabilities = np.array([[0.4, 0.6], [0.9, 0.1]])

        distiller = ConcreteDistiller.from_probabilities(probabilities=probabilities)

        assert distiller.student_model_type is None
        assert distiller.student_params is None or distiller.student_params == {}
        assert distiller.is_fitted is True


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow_fit_predict_evaluate(self, binary_classification_data):
        """Test complete workflow: fit -> predict -> evaluate"""
        X, y = binary_classification_data
        X_train, X_test = X[:80], X[80:]
        y_train, y_test = y[:80], y[80:]

        distiller = ConcreteDistiller(
            student_model_type=ModelType.LOGISTIC_REGRESSION,
            student_params={'C': 1.0},
            random_state=42
        )

        # Fit
        distiller.fit(X_train, y_train)
        assert distiller.is_fitted is True

        # Predict
        predictions = distiller.predict(X_test)
        assert predictions.shape[0] == len(X_test)

        # Evaluate
        with patch('deepbridge.metrics.classification.Classification', side_effect=ImportError):
            metrics = distiller.evaluate(X_test, y_test)
            assert 'accuracy' in metrics
            assert 'best_params' in metrics

    def test_workflow_from_probabilities(self):
        """Test workflow starting from pre-computed probabilities"""
        probabilities = np.random.rand(100, 2)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)

        distiller = ConcreteDistiller.from_probabilities(
            probabilities=probabilities,
            student_model_type=ModelType.RANDOM_FOREST
        )

        assert distiller.is_fitted is True
        assert distiller.student_model_type == ModelType.RANDOM_FOREST

        # Should be able to use directly without fit
        X_test = np.random.rand(20, 10)
        predictions = distiller.predict(X_test)
        assert predictions.shape[0] == 20
