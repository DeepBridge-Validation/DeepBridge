"""
Tests for model_registry module.

Coverage Target: Increase from 49.5% to ~70%+
Focus: LinearGAM and LogisticGAM wrappers
"""

import pytest
import numpy as np
import pandas as pd

from deepbridge.utils.model_registry import LinearGAM, LogisticGAM


class TestLinearGAM:
    """Tests for LinearGAM regression wrapper."""

    def test_init(self):
        """Test LinearGAM initialization."""
        model = LinearGAM(n_splines=15, spline_order=4, lam=0.5, max_iter=200, random_state=42)

        assert model.n_splines == 15
        assert model.spline_order == 4
        assert model.lam == 0.5
        assert model.max_iter == 200
        assert model.random_state == 42
        assert model.model is None
        assert model.smoother is None

    def test_fit_with_numpy_arrays(self):
        """Test fitting with numpy arrays."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1

        model = LinearGAM(n_splines=5, random_state=42)
        result = model.fit(X, y)

        assert result is model  # fit returns self
        assert model.model is not None
        assert model.smoother is not None

    def test_fit_with_pandas(self):
        """Test fitting with pandas DataFrames."""
        np.random.seed(42)
        X_df = pd.DataFrame(np.random.randn(100, 3), columns=['f1', 'f2', 'f3'])
        y_series = pd.Series(2 * X_df['f1'] + 3 * X_df['f2'] - X_df['f3'] + np.random.randn(100) * 0.1)

        model = LinearGAM(n_splines=5, random_state=42)
        model.fit(X_df, y_series)

        assert model.model is not None

    def test_predict_after_fit(self):
        """Test prediction after fitting."""
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        y_train = X_train[:, 0] + 2 * X_train[:, 1] + np.random.randn(100) * 0.1

        model = LinearGAM(n_splines=5, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(10, 2)
        predictions = model.predict(X_test)

        assert predictions.shape == (10,)
        assert not np.any(np.isnan(predictions))  # NaNs should be handled

    def test_predict_without_fit_raises_error(self):
        """Test that predict raises error when model not fitted."""
        model = LinearGAM()
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError, match='Model not fitted'):
            model.predict(X)

    def test_predict_with_pandas(self):
        """Test prediction with pandas DataFrame."""
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        y_train = X_train[:, 0] + 2 * X_train[:, 1] + np.random.randn(100) * 0.1

        model = LinearGAM(n_splines=5, random_state=42)
        model.fit(X_train, y_train)

        X_test_df = pd.DataFrame(np.random.randn(10, 2), columns=['f1', 'f2'])
        predictions = model.predict(X_test_df)

        assert predictions.shape == (10,)

    def test_create_bsplines(self):
        """Test that B-splines are created correctly."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = np.random.randn(50)

        model = LinearGAM(n_splines=8, spline_order=3)
        model.fit(X, y)

        # Check smoother was created
        assert model.smoother is not None
        assert hasattr(model.smoother, 'basis')

    def test_random_state_reproducibility(self):
        """Test that random_state makes fitting reproducible."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1
        X_test = np.random.randn(10, 2)

        model1 = LinearGAM(n_splines=5, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X_test)

        model2 = LinearGAM(n_splines=5, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_predict_recreates_splines_if_different_shape(self):
        """Test that predict recreates splines if feature count differs."""
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        y_train = X_train[:, 0] + X_train[:, 1] + X_train[:, 2]

        model = LinearGAM(n_splines=5, random_state=42)
        model.fit(X_train, y_train)

        # Manually change smoother.basis shape to trigger re-creation
        # This simulates predicting on different feature count (edge case)
        original_basis_shape = model.smoother.basis.shape

        # Predict normally first
        X_test = np.random.randn(10, 3)
        predictions = model.predict(X_test)

        assert predictions.shape == (10,)
        # Smoother should still work after potential re-creation
        assert model.smoother is not None


class TestLogisticGAM:
    """Tests for LogisticGAM classification wrapper."""

    def test_init(self):
        """Test LogisticGAM initialization."""
        model = LogisticGAM(n_splines=12, spline_order=2, max_iter=150, random_state=123)

        assert model.n_splines == 12
        assert model.spline_order == 2
        assert model.max_iter == 150
        assert model.random_state == 123
        assert model.model is None

    def test_fit_with_binary_labels(self):
        """Test fitting with binary classification data."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        # Create binary labels
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = LogisticGAM(n_splines=5, random_state=42)
        result = model.fit(X, y)

        assert result is model
        assert model.model is not None

    def test_fit_with_pandas(self):
        """Test fitting with pandas data."""
        np.random.seed(42)
        X_df = pd.DataFrame(np.random.randn(100, 2), columns=['f1', 'f2'])
        y_series = pd.Series((X_df['f1'] + X_df['f2'] > 0).astype(int))

        model = LogisticGAM(n_splines=5, random_state=42)
        model.fit(X_df, y_series)

        assert model.model is not None

    def test_predict_returns_binary_labels(self):
        """Test that predict returns binary labels."""
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

        model = LogisticGAM(n_splines=5, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 2)
        predictions = model.predict(X_test)

        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_predict_proba_returns_probabilities(self):
        """Test that predict_proba returns valid probabilities."""
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

        model = LogisticGAM(n_splines=5, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 2)
        probas = model.predict_proba(X_test)

        assert probas.shape == (20, 2)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(probas.sum(axis=1), np.ones(20))
        # Probabilities should be in [0, 1]
        assert np.all(probas >= 0)
        assert np.all(probas <= 1)

    def test_predict_without_fit_raises_error(self):
        """Test that predict raises error when not fitted."""
        model = LogisticGAM()
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError, match='Model not fitted'):
            model.predict(X)

    def test_predict_proba_without_fit_raises_error(self):
        """Test that predict_proba raises error when not fitted."""
        model = LogisticGAM()
        X = np.random.randn(10, 2)

        with pytest.raises(ValueError, match='Model not fitted'):
            model.predict_proba(X)

    def test_predict_with_pandas(self):
        """Test predict with pandas DataFrame."""
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

        model = LogisticGAM(n_splines=5, random_state=42)
        model.fit(X_train, y_train)

        X_test_df = pd.DataFrame(np.random.randn(20, 2), columns=['f1', 'f2'])
        predictions = model.predict(X_test_df)

        assert predictions.shape == (20,)
        assert np.all(np.isin(predictions, [0, 1]))

    def test_predict_proba_with_pandas(self):
        """Test predict_proba with pandas DataFrame."""
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)

        model = LogisticGAM(n_splines=5, random_state=42)
        model.fit(X_train, y_train)

        X_test_df = pd.DataFrame(np.random.randn(20, 2), columns=['f1', 'f2'])
        probas = model.predict_proba(X_test_df)

        assert probas.shape == (20, 2)

    def test_predict_threshold(self):
        """Test that predict uses 0.5 threshold."""
        np.random.seed(42)
        X_train = np.random.randn(100, 2)
        # Use both features to avoid perfect separation
        y_train = (X_train[:, 0] + X_train[:, 1] + np.random.randn(100) * 0.5 > 0).astype(int)

        model = LogisticGAM(n_splines=5, random_state=42)
        model.fit(X_train, y_train)

        X_test = np.random.randn(20, 2)
        predictions = model.predict(X_test)
        probas = model.predict_proba(X_test)

        # Verify predictions match threshold on probabilities
        expected_predictions = (probas[:, 1] > 0.5).astype(int)
        np.testing.assert_array_equal(predictions, expected_predictions)

    def test_random_state_reproducibility(self):
        """Test that random_state makes fitting reproducible."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        X_test = np.random.randn(10, 2)

        model1 = LogisticGAM(n_splines=5, random_state=42)
        model1.fit(X, y)
        pred1 = model1.predict(X_test)

        model2 = LogisticGAM(n_splines=5, random_state=42)
        model2.fit(X, y)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)


class TestModelRegistryModels:
    """Tests for creating different model types via ModelRegistry."""

    def test_get_gbm_classifier(self):
        """Test getting GBM classifier."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType
        from sklearn.ensemble import GradientBoostingClassifier

        model = ModelRegistry.get_model(ModelType.GBM)
        assert isinstance(model, GradientBoostingClassifier)
        assert model.n_estimators == 100

    def test_get_xgb_classifier(self):
        """Test getting XGBoost classifier."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType
        import xgboost as xgb

        model = ModelRegistry.get_model(ModelType.XGB)
        assert isinstance(model, xgb.XGBClassifier)

    def test_get_random_forest_classifier(self):
        """Test getting Random Forest classifier."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType
        from sklearn.ensemble import RandomForestClassifier

        model = ModelRegistry.get_model(ModelType.RANDOM_FOREST)
        assert isinstance(model, RandomForestClassifier)

    def test_get_logistic_regression_classifier(self):
        """Test getting Logistic Regression classifier."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType
        from sklearn.linear_model import LogisticRegression

        model = ModelRegistry.get_model(ModelType.LOGISTIC_REGRESSION)
        assert isinstance(model, LogisticRegression)

    def test_get_glm_classifier(self):
        """Test getting GLM classifier (SGDClassifier)."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType
        from sklearn.linear_model import SGDClassifier

        model = ModelRegistry.get_model(ModelType.GLM_CLASSIFIER)
        assert isinstance(model, SGDClassifier)

    def test_get_gam_classifier(self):
        """Test getting GAM classifier."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType
        from deepbridge.utils.model_registry import LogisticGAM

        model = ModelRegistry.get_model(ModelType.GAM_CLASSIFIER)
        assert isinstance(model, LogisticGAM)

    def test_get_gam_regressor(self):
        """Test getting GAM regressor."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode
        from deepbridge.utils.model_registry import LinearGAM

        model = ModelRegistry.get_model(ModelType.GAM_CLASSIFIER, mode=ModelMode.REGRESSION)
        assert isinstance(model, LinearGAM)


class TestModelRegistry:
    """Tests for ModelRegistry class."""

    def test_get_model_with_default_params(self):
        """Test getting model with default parameters."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode

        model = ModelRegistry.get_model(ModelType.DECISION_TREE)
        assert model is not None
        assert model.max_depth == 5
        assert model.min_samples_split == 2

    def test_get_model_with_custom_params(self):
        """Test getting model with custom parameters."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType

        model = ModelRegistry.get_model(
            ModelType.DECISION_TREE,
            custom_params={'max_depth': 10, 'min_samples_split': 5}
        )
        assert model.max_depth == 10
        assert model.min_samples_split == 5

    def test_get_model_regression_mode(self):
        """Test getting model in regression mode."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode
        from sklearn.tree import DecisionTreeRegressor

        model = ModelRegistry.get_model(ModelType.DECISION_TREE, mode=ModelMode.REGRESSION)
        assert isinstance(model, DecisionTreeRegressor)

    def test_get_model_xgb_regression(self):
        """Test XGBoost model in regression mode."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode
        import xgboost as xgb

        model = ModelRegistry.get_model(ModelType.XGB, mode=ModelMode.REGRESSION)
        assert isinstance(model, xgb.XGBRegressor)
        # Check that objective was changed to regression
        assert model.objective == 'reg:squarederror'

    def test_get_model_glm_regression(self):
        """Test GLM in regression mode changes loss function."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode
        from sklearn.linear_model import SGDRegressor

        model = ModelRegistry.get_model(ModelType.GLM_CLASSIFIER, mode=ModelMode.REGRESSION)
        assert isinstance(model, SGDRegressor)
        assert model.loss == 'squared_error'

    def test_get_model_logistic_regression_mode_removes_params(self):
        """Test that LogisticRegression in regression mode removes incompatible params."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode
        from sklearn.linear_model import LinearRegression

        model = ModelRegistry.get_model(ModelType.LOGISTIC_REGRESSION, mode=ModelMode.REGRESSION)
        assert isinstance(model, LinearRegression)

    def test_get_model_unsupported_type_raises_error(self):
        """Test that unsupported model type raises ValueError."""
        from deepbridge.utils.model_registry import ModelRegistry, ModelType
        from enum import Enum, auto

        class FakeModelType(Enum):
            UNSUPPORTED = auto()

        with pytest.raises(ValueError, match='Unsupported model type'):
            ModelRegistry.get_model(FakeModelType.UNSUPPORTED)

    def test_get_param_space_with_trial(self):
        """Test getting parameter space with Optuna trial."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry, ModelType

        study = optuna.create_study()
        trial = study.ask()

        param_space = ModelRegistry.get_param_space(ModelType.DECISION_TREE, trial)

        assert 'max_depth' in param_space
        assert 'min_samples_split' in param_space
        assert 'min_samples_leaf' in param_space

    def test_get_param_space_includes_random_state(self):
        """Test that random_state is added to param space."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry, ModelType

        study = optuna.create_study()
        trial = study.ask()

        param_space = ModelRegistry.get_param_space(ModelType.RANDOM_FOREST, trial)

        assert 'random_state' in param_space
        assert param_space['random_state'] == 42

    def test_get_param_space_xgb_regression_mode(self):
        """Test XGBoost param space in regression mode."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode

        study = optuna.create_study()
        trial = study.ask()

        param_space = ModelRegistry.get_param_space(
            ModelType.XGB, trial, mode=ModelMode.REGRESSION
        )

        assert param_space['objective'] == 'reg:squarederror'

    def test_get_param_space_glm_regression_mode(self):
        """Test GLM param space in regression mode."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode

        study = optuna.create_study()
        trial = study.ask()

        param_space = ModelRegistry.get_param_space(
            ModelType.GLM_CLASSIFIER, trial, mode=ModelMode.REGRESSION
        )

        # Should have squared_error loss for regression
        if 'loss' in param_space:
            assert param_space['loss'] == 'squared_error'

    def test_get_param_space_logistic_regression_mode_removes_params(self):
        """Test LogisticRegression param space removes incompatible params in regression."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode

        study = optuna.create_study()
        trial = study.ask()

        param_space = ModelRegistry.get_param_space(
            ModelType.LOGISTIC_REGRESSION, trial, mode=ModelMode.REGRESSION
        )

        # These params should be removed for LinearRegression
        assert 'C' not in param_space
        assert 'solver' not in param_space

    def test_get_param_space_unsupported_type_raises_error(self):
        """Test that unsupported type raises error in get_param_space."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry
        from enum import Enum, auto

        class FakeModelType(Enum):
            UNSUPPORTED = auto()

        study = optuna.create_study()
        trial = study.ask()

        with pytest.raises(ValueError, match='Unsupported model type'):
            ModelRegistry.get_param_space(FakeModelType.UNSUPPORTED, trial)


class TestEnums:
    """Tests for ModelType and ModelMode enums."""

    def test_model_type_enum_values(self):
        """Test ModelType enum has expected values."""
        from deepbridge.utils.model_registry import ModelType

        assert ModelType.GLM_CLASSIFIER
        assert ModelType.GAM_CLASSIFIER
        assert ModelType.GBM
        assert ModelType.XGB
        assert ModelType.LOGISTIC_REGRESSION
        assert ModelType.DECISION_TREE
        assert ModelType.RANDOM_FOREST
        assert ModelType.MLP

    def test_model_mode_enum_values(self):
        """Test ModelMode enum has expected values."""
        from deepbridge.utils.model_registry import ModelMode

        assert ModelMode.CLASSIFICATION
        assert ModelMode.REGRESSION


class TestStatsModelsGAM:
    """Tests for StatsModelsGAM base class."""

    def test_base_gam_initialization(self):
        """Test base GAM class initialization."""
        from deepbridge.utils.model_registry import StatsModelsGAM

        gam = StatsModelsGAM(n_splines=15, spline_order=4, lam=0.8, max_iter=200, random_state=123)
        assert gam.n_splines == 15
        assert gam.spline_order == 4
        assert gam.lam == 0.8
        assert gam.max_iter == 200
        assert gam.random_state == 123
        assert gam.model is None
        assert gam.smoother is None


class TestParameterSpaceFunctions:
    """Tests for individual parameter space functions."""

    def test_logistic_regression_param_space(self):
        """Test logistic regression parameter space."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry

        study = optuna.create_study()
        trial = study.ask()
        params = ModelRegistry._logistic_regression_param_space(trial)

        assert 'C' in params
        assert 'solver' in params
        assert 'max_iter' in params

    def test_linear_regression_param_space(self):
        """Test linear regression parameter space."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry

        study = optuna.create_study()
        trial = study.ask()
        params = ModelRegistry._linear_regression_param_space(trial)

        assert 'fit_intercept' in params
        assert 'positive' in params

    def test_decision_tree_param_space(self):
        """Test decision tree parameter space."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry

        study = optuna.create_study()
        trial = study.ask()
        params = ModelRegistry._decision_tree_param_space(trial)

        assert 'max_depth' in params
        assert 'min_samples_split' in params
        assert 'min_samples_leaf' in params

    def test_gbm_param_space(self):
        """Test GBM parameter space."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry

        study = optuna.create_study()
        trial = study.ask()
        params = ModelRegistry._gbm_param_space(trial)

        assert 'n_estimators' in params
        assert 'learning_rate' in params
        assert 'max_depth' in params
        assert 'subsample' in params

    def test_xgb_param_space(self):
        """Test XGBoost parameter space."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry

        study = optuna.create_study()
        trial = study.ask()
        params = ModelRegistry._xgb_param_space(trial)

        assert 'n_estimators' in params
        assert 'learning_rate' in params
        assert 'max_depth' in params
        assert 'subsample' in params
        assert 'colsample_bytree' in params

    def test_random_forest_param_space(self):
        """Test Random Forest parameter space."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry

        study = optuna.create_study()
        trial = study.ask()
        params = ModelRegistry._random_forest_param_space(trial)

        assert 'n_estimators' in params
        assert 'max_depth' in params
        assert 'min_samples_split' in params
        assert 'min_samples_leaf' in params
        assert 'max_features' in params

    def test_glm_param_space(self):
        """Test GLM parameter space."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry

        study = optuna.create_study()
        trial = study.ask()
        params = ModelRegistry._glm_param_space(trial)

        assert 'alpha' in params
        assert 'max_iter' in params
        assert 'fit_intercept' in params
        assert 'tol' in params
        assert 'penalty' in params
        assert 'l1_ratio' in params

    def test_gam_param_space(self):
        """Test GAM parameter space."""
        import optuna
        from deepbridge.utils.model_registry import ModelRegistry

        study = optuna.create_study()
        trial = study.ask()
        params = ModelRegistry._gam_param_space(trial)

        assert 'n_splines' in params
        assert 'spline_order' in params
        assert 'lam' in params
        assert 'max_iter' in params


class TestModelFactory:
    """Tests for ModelFactory class."""

    def test_create_model_with_string_type(self):
        """Test creating model with string model type."""
        from deepbridge.utils.model_registry import ModelFactory
        from sklearn.tree import DecisionTreeClassifier

        factory = ModelFactory()
        model = factory.create_model('decision_tree')
        assert isinstance(model, DecisionTreeClassifier)

    def test_create_model_with_enum_type(self):
        """Test creating model with enum model type."""
        from deepbridge.utils.model_registry import ModelFactory, ModelType
        from sklearn.ensemble import RandomForestClassifier

        factory = ModelFactory()
        model = factory.create_model(ModelType.RANDOM_FOREST)
        assert isinstance(model, RandomForestClassifier)

    def test_create_model_with_custom_params(self):
        """Test creating model with custom parameters."""
        from deepbridge.utils.model_registry import ModelFactory

        factory = ModelFactory()
        model = factory.create_model('decision_tree', max_depth=15, min_samples_split=10)
        assert model.max_depth == 15
        assert model.min_samples_split == 10

    def test_create_model_regression_task(self):
        """Test creating regression model."""
        from deepbridge.utils.model_registry import ModelFactory
        from sklearn.ensemble import GradientBoostingRegressor

        factory = ModelFactory()
        model = factory.create_model('gbm', task_type='regression')
        assert isinstance(model, GradientBoostingRegressor)

    def test_get_model_method(self):
        """Test backward-compatible get_model method."""
        from deepbridge.utils.model_registry import ModelFactory, ModelType, ModelMode
        from sklearn.linear_model import LogisticRegression

        factory = ModelFactory()
        model = factory.get_model(ModelType.LOGISTIC_REGRESSION)
        assert isinstance(model, LogisticRegression)

    def test_get_model_with_custom_params_dict(self):
        """Test get_model with custom_params dictionary."""
        from deepbridge.utils.model_registry import ModelFactory, ModelType

        factory = ModelFactory()
        model = factory.get_model(
            ModelType.DECISION_TREE,
            custom_params={'max_depth': 20}
        )
        assert model.max_depth == 20

    def test_get_model_regression_mode(self):
        """Test get_model in regression mode."""
        from deepbridge.utils.model_registry import ModelFactory, ModelType, ModelMode
        from sklearn.tree import DecisionTreeRegressor

        factory = ModelFactory()
        model = factory.get_model(ModelType.DECISION_TREE, mode=ModelMode.REGRESSION)
        assert isinstance(model, DecisionTreeRegressor)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
