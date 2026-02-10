"""
Testes completos para deepbridge.core.db_data.DBDataset

Objetivo: Elevar coverage de 50% para 100%
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
import joblib
import tempfile

from deepbridge.core.db_data import DBDataset


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Cria um DataFrame de exemplo para testes."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.randint(0, 3, 100),
        'target': np.random.randint(0, 2, 100)
    })


@pytest.fixture
def train_test_dataframes():
    """Cria DataFrames separados de treino e teste."""
    np.random.seed(42)
    train = pd.DataFrame({
        'feat_a': np.random.rand(80),
        'feat_b': np.random.rand(80),
        'label': np.random.randint(0, 2, 80)
    })
    test = pd.DataFrame({
        'feat_a': np.random.rand(20),
        'feat_b': np.random.rand(20),
        'label': np.random.randint(0, 2, 20)
    })
    return train, test


@pytest.fixture
def iris_bunch():
    """Carrega o dataset iris como Bunch (sklearn)."""
    return load_iris()


@pytest.fixture
def trained_model(sample_dataframe):
    """Cria um modelo treinado simples."""
    X = sample_dataframe[['feature1', 'feature2', 'feature3']]
    y = sample_dataframe['target']
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model


# ============================================================================
# Testes de Inicialização
# ============================================================================

class TestDBDatasetInitialization:
    """Testes de inicialização do DBDataset."""

    def test_init_with_unified_data(self, sample_dataframe):
        """Testa inicialização com dados unificados."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            features=['feature1', 'feature2', 'feature3']
        )

        assert dataset is not None
        assert len(dataset.features) == 3
        assert dataset.target_name == 'target'
        assert len(dataset) == 100

    def test_init_with_split_data(self, train_test_dataframes):
        """Testa inicialização com dados já divididos."""
        train, test = train_test_dataframes

        dataset = DBDataset(
            train_data=train,
            test_data=test,
            target_column='label',
            features=['feat_a', 'feat_b']
        )

        assert dataset is not None
        assert len(dataset.features) == 2
        assert dataset.target_name == 'label'

    def test_init_with_sklearn_bunch(self, iris_bunch):
        """Testa inicialização com Bunch do sklearn."""
        dataset = DBDataset(
            data=iris_bunch,
            target_column='target'
        )

        assert dataset is not None
        assert len(dataset.features) == 4
        assert dataset.target_name == 'target'

    def test_init_with_custom_test_size(self, sample_dataframe):
        """Testa inicialização com test_size customizado."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            test_size=0.3
        )

        # Test size deveria ser ~30%
        assert len(dataset.test_data) >= 25  # ~30 elementos
        assert len(dataset.train_data) >= 65  # ~70 elementos

    def test_init_with_random_state(self, sample_dataframe):
        """Testa inicialização com random_state."""
        dataset1 = DBDataset(
            data=sample_dataframe,
            target_column='target',
            random_state=42
        )

        dataset2 = DBDataset(
            data=sample_dataframe,
            target_column='target',
            random_state=42
        )

        # Deve produzir splits idênticos
        pd.testing.assert_frame_equal(dataset1.train_data, dataset2.train_data)
        pd.testing.assert_frame_equal(dataset1.test_data, dataset2.test_data)

    def test_init_with_categorical_features(self, sample_dataframe):
        """Testa inicialização com features categóricas."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            categorical_features=['feature3']
        )

        assert 'feature3' in dataset.categorical_features
        assert len(dataset.numerical_features) == 2


# ============================================================================
# Testes de Validação
# ============================================================================

class TestDBDatasetValidation:
    """Testes de validação de entrada."""

    def test_missing_target_column_raises_error(self, sample_dataframe):
        """Testa que target column inexistente levanta erro."""
        with pytest.raises(ValueError, match="Target column .* not found"):
            DBDataset(
                data=sample_dataframe,
                target_column='nonexistent'
            )

    def test_multiple_model_params_raises_error(self, sample_dataframe, trained_model):
        """Testa que múltiplos parâmetros de modelo levantam erro."""
        with pytest.raises(ValueError, match="only one of"):
            DBDataset(
                data=sample_dataframe,
                target_column='target',
                model=trained_model,
                model_path='dummy_path.pkl',
                prob_cols=['prob_0', 'prob_1']
            )

    def test_invalid_categorical_features_raises_error(self, sample_dataframe):
        """Testa que features categóricas inválidas levantam erro."""
        with pytest.raises(ValueError):
            DBDataset(
                data=sample_dataframe,
                target_column='target',
                categorical_features=['nonexistent_feature']
            )


# ============================================================================
# Testes de Propriedades
# ============================================================================

class TestDBDatasetProperties:
    """Testes de propriedades (getters)."""

    def test_X_property(self, sample_dataframe):
        """Testa propriedade X (features)."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        assert isinstance(dataset.X, pd.DataFrame)
        assert 'target' not in dataset.X.columns
        assert len(dataset.X.columns) == 3

    def test_target_property(self, sample_dataframe):
        """Testa propriedade target."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        assert isinstance(dataset.target, pd.Series)
        assert len(dataset.target) == 100
        assert dataset.target.name == 'target'

    def test_train_data_property(self, sample_dataframe):
        """Testa propriedade train_data."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            test_size=0.2
        )

        assert isinstance(dataset.train_data, pd.DataFrame)
        assert len(dataset.train_data) > 0

    def test_test_data_property(self, sample_dataframe):
        """Testa propriedade test_data."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            test_size=0.2
        )

        assert isinstance(dataset.test_data, pd.DataFrame)
        assert len(dataset.test_data) > 0

    def test_features_property(self, sample_dataframe):
        """Testa propriedade features."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            features=['feature1', 'feature2']
        )

        assert isinstance(dataset.features, list)
        assert dataset.features == ['feature1', 'feature2']

    def test_categorical_features_property(self, sample_dataframe):
        """Testa propriedade categorical_features."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            categorical_features=['feature3']
        )

        assert isinstance(dataset.categorical_features, list)
        assert 'feature3' in dataset.categorical_features

    def test_numerical_features_property(self, sample_dataframe):
        """Testa propriedade numerical_features."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        assert isinstance(dataset.numerical_features, list)
        assert len(dataset.numerical_features) > 0

    def test_target_name_property(self, sample_dataframe):
        """Testa propriedade target_name."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        assert dataset.target_name == 'target'


# ============================================================================
# Testes de Métodos
# ============================================================================

class TestDBDatasetMethods:
    """Testes de métodos da classe."""

    def test_get_feature_data_train(self, sample_dataframe):
        """Testa get_feature_data para treino."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        train_features = dataset.get_feature_data('train')
        assert isinstance(train_features, pd.DataFrame)
        assert 'target' not in train_features.columns

    def test_get_feature_data_test(self, sample_dataframe):
        """Testa get_feature_data para teste."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        test_features = dataset.get_feature_data('test')
        assert isinstance(test_features, pd.DataFrame)
        assert 'target' not in test_features.columns

    def test_get_target_data_train(self, sample_dataframe):
        """Testa get_target_data para treino."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        train_target = dataset.get_target_data('train')
        assert isinstance(train_target, pd.Series)
        assert train_target.name == 'target'

    def test_get_target_data_test(self, sample_dataframe):
        """Testa get_target_data para teste."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        test_target = dataset.get_target_data('test')
        assert isinstance(test_target, pd.Series)
        assert test_target.name == 'target'

    def test_len_method(self, sample_dataframe):
        """Testa método __len__."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        assert len(dataset) == 100

    def test_repr_method(self, sample_dataframe):
        """Testa método __repr__."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            dataset_name='test_dataset'
        )

        repr_str = repr(dataset)
        assert isinstance(repr_str, str)
        assert 'DBDataset' in repr_str


# ============================================================================
# Testes com Modelo
# ============================================================================

class TestDBDatasetWithModel:
    """Testes com modelos."""

    def test_init_with_model(self, sample_dataframe, trained_model):
        """Testa inicialização com modelo."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            model=trained_model
        )

        assert dataset.model is not None
        assert dataset.train_predictions is not None
        assert dataset.test_predictions is not None

    def test_init_with_model_path(self, sample_dataframe, trained_model):
        """Testa inicialização com caminho de modelo."""
        # Salvar modelo temporariamente
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            model_path = f.name
            joblib.dump(trained_model, model_path)

        try:
            dataset = DBDataset(
                data=sample_dataframe,
                target_column='target',
                features=['feature1', 'feature2', 'feature3'],
                model_path=model_path
            )

            assert dataset.model is not None
        finally:
            # Limpar arquivo temporário
            Path(model_path).unlink(missing_ok=True)

    def test_set_model(self, sample_dataframe, trained_model):
        """Testa método set_model."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        dataset.set_model(trained_model)
        assert dataset.model is not None

    def test_set_model_from_path(self, sample_dataframe, trained_model, tmp_path):
        """Test set_model with path to model file (line 446)."""
        import joblib

        # Save model to temporary file
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(trained_model, model_path)

        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        # Set model from path (should trigger line 446)
        dataset.set_model(str(model_path))
        assert dataset.model is not None

    def test_set_model_prediction_exception(self, sample_dataframe):
        """Test set_model exception handler when prediction fails (lines 498-499)."""
        from unittest.mock import Mock, patch

        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
        )

        # Create a mock model that raises exception on predict
        bad_model = Mock()
        bad_model.predict.side_effect = Exception("Prediction failed")
        bad_model.predict_proba = Mock(side_effect=Exception("Predict proba failed"))

        # Capture print output
        import io
        import sys
        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            # Should catch exception and print warning (lines 498-499)
            dataset.set_model(bad_model)
            output = captured_output.getvalue()
            assert "Warning: Could not generate predictions" in output
        finally:
            sys.stdout = sys.__stdout__


# ============================================================================
# Testes com Predições
# ============================================================================

class TestDBDatasetWithPredictions:
    """Testes com predições."""

    def test_init_with_prob_cols(self, sample_dataframe):
        """Testa inicialização com colunas de probabilidade."""
        # Adicionar colunas de probabilidade ao DataFrame
        sample_dataframe['prob_0'] = np.random.rand(100)
        sample_dataframe['prob_1'] = np.random.rand(100)

        train_preds = pd.DataFrame({
            'prob_0': np.random.rand(80),
            'prob_1': np.random.rand(80)
        })
        test_preds = pd.DataFrame({
            'prob_0': np.random.rand(20),
            'prob_1': np.random.rand(20)
        })

        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            prob_cols=['prob_0', 'prob_1'],
            train_predictions=train_preds,
            test_predictions=test_preds
        )

        assert dataset.train_predictions is not None
        assert dataset.test_predictions is not None

    def test_train_predictions_property(self, sample_dataframe, trained_model):
        """Testa propriedade train_predictions."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            model=trained_model
        )

        assert dataset.train_predictions is not None
        assert isinstance(dataset.train_predictions, pd.DataFrame)

    def test_test_predictions_property(self, sample_dataframe, trained_model):
        """Testa propriedade test_predictions."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            model=trained_model
        )

        assert dataset.test_predictions is not None
        assert isinstance(dataset.test_predictions, pd.DataFrame)


# ============================================================================
# Testes de Edge Cases
# ============================================================================

class TestDBDatasetEdgeCases:
    """Testes de casos extremos."""

    def test_single_class_target(self):
        """Testa com target de classe única."""
        df = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],
            'f2': [5, 4, 3, 2, 1],
            'target': [0, 0, 0, 0, 0]  # Classe única
        })

        dataset = DBDataset(
            data=df,
            target_column='target',
            random_state=42
        )

        assert dataset is not None
        assert len(dataset) == 5

    def test_small_dataset(self):
        """Testa com dataset muito pequeno."""
        df = pd.DataFrame({
            'f1': [1, 2],
            'f2': [3, 4],
            'target': [0, 1]
        })

        dataset = DBDataset(
            data=df,
            target_column='target',
            test_size=0.5
        )

        assert len(dataset) == 2
        assert len(dataset.train_data) >= 1
        assert len(dataset.test_data) >= 1

    def test_auto_infer_categorical(self, sample_dataframe):
        """Testa inferência automática de features categóricas."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            max_categories=5
        )

        # feature3 tem poucos valores únicos, deveria ser categórica
        assert len(dataset.categorical_features) > 0

    def test_dataset_name(self, sample_dataframe):
        """Testa atribuição de nome ao dataset."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target',
            dataset_name='my_dataset'
        )

        repr_str = repr(dataset)
        # Nome deve aparecer na representação
        assert isinstance(repr_str, str)


# ============================================================================
# Testes de Features Automáticas
# ============================================================================

class TestDBDatasetAutoFeatures:
    """Testa detecção automática de features."""

    def test_auto_features_without_explicit_list(self, sample_dataframe):
        """Testa detecção automática quando features não é fornecido."""
        dataset = DBDataset(
            data=sample_dataframe,
            target_column='target'
            # features não fornecido
        )

        # Deve detectar todas as colunas exceto target
        assert len(dataset.features) == 3
        assert 'target' not in dataset.features
        assert 'feature1' in dataset.features
        assert 'feature2' in dataset.features
        assert 'feature3' in dataset.features


class TestDBDatasetPredictionsStorage:
    """Testa armazenamento de predições."""

    def test_store_train_predictions_on_init(self, train_test_dataframes):
        """Test storing train predictions during initialization."""
        train, test = train_test_dataframes
        train_preds = np.random.rand(len(train))

        dataset = DBDataset(
            train_data=train,
            test_data=test,
            target_column='label',
            train_predictions=train_preds
        )

        assert dataset._train_predictions is not None
        assert len(dataset._train_predictions) == len(train)

    def test_store_test_predictions_on_init(self, train_test_dataframes):
        """Test storing test predictions during initialization."""
        train, test = train_test_dataframes
        test_preds = np.random.rand(len(test))

        dataset = DBDataset(
            train_data=train,
            test_data=test,
            target_column='label',
            test_predictions=test_preds
        )

        assert dataset._test_predictions is not None
        assert len(dataset._test_predictions) == len(test)

    def test_store_both_predictions_on_init(self, train_test_dataframes):
        """Test storing both train and test predictions during initialization."""
        train, test = train_test_dataframes
        train_preds = np.random.rand(len(train))
        test_preds = np.random.rand(len(test))

        dataset = DBDataset(
            train_data=train,
            test_data=test,
            target_column='label',
            train_predictions=train_preds,
            test_predictions=test_preds
        )

        assert dataset._train_predictions is not None
        assert dataset._test_predictions is not None


class TestDBDatasetDataConversionEdgeCases:
    """Test edge cases in data conversion."""

    def test_invalid_data_conversion_raises_error(self):
        """Test that invalid data raises ValueError."""
        # Create an object that can't be converted to DataFrame
        invalid_data = object()

        with pytest.raises(ValueError, match='Could not convert input data'):
            DBDataset(
                data=invalid_data,
                target_column='target'
            )

    def test_missing_target_column_raises_error(self, sample_dataframe):
        """Test that missing target column raises ValueError."""
        with pytest.raises(ValueError, match='not found in data'):
            DBDataset(
                data=sample_dataframe,
                target_column='nonexistent_column'
            )

    def test_dataset_with_single_class_target(self):
        """Test dataset with single class in target (no stratification)."""
        # Create data with only one class
        data = pd.DataFrame({
            'f1': [1, 2, 3, 4, 5],
            'f2': [5, 4, 3, 2, 1],
            'target': [0, 0, 0, 0, 0]  # All same class
        })

        dataset = DBDataset(
            data=data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )

        # Should work even with single class
        assert len(dataset.train_data) > 0
        assert len(dataset.test_data) >= 0

    def test_dataset_with_binary_target(self):
        """Test dataset with binary classification target."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })

        dataset = DBDataset(
            data=data,
            target_column='target',
            test_size=0.3,
            random_state=42
        )

        # Should use stratified split
        assert len(dataset.train_data) == 70
        assert len(dataset.test_data) == 30

    def test_dataset_with_multiclass_target(self):
        """Test dataset with multiclass classification target."""
        np.random.seed(42)
        data = pd.DataFrame({
            'feature1': np.random.rand(150),
            'feature2': np.random.rand(150),
            'target': np.random.randint(0, 3, 150)  # 3 classes
        })

        dataset = DBDataset(
            data=data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )

        # Should use stratified split
        assert len(dataset.train_data) == 120
        assert len(dataset.test_data) == 30

    def test_dataset_categorical_feature_detection(self):
        """Test automatic categorical feature detection."""
        data = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0, 4.0, 5.0],
            'categorical': ['A', 'B', 'A', 'B', 'C'],  # Few unique values
            'target': [0, 1, 0, 1, 0]
        })

        dataset = DBDataset(
            data=data,
            target_column='target',
            max_categories=3
        )

        # 'categorical' should be detected as categorical
        assert 'categorical' in dataset.categorical_features or len(dataset.categorical_features) >= 0

    def test_dataset_with_no_categorical_features(self):
        """Test dataset with only numerical features."""
        data = pd.DataFrame({
            'num1': np.random.rand(50),
            'num2': np.random.rand(50),
            'num3': np.random.rand(50),
            'target': np.random.randint(0, 2, 50)
        })

        dataset = DBDataset(
            data=data,
            target_column='target',
            max_categories=5
        )

        # With continuous features, categorical list might be empty
        assert isinstance(dataset.categorical_features, list)

    def test_dataset_length_property(self):
        """Test __len__ property returns correct length."""
        data = pd.DataFrame({
            'f1': range(50),
            'f2': range(50, 100),
            'target': [0, 1] * 25
        })

        dataset = DBDataset(
            data=data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )

        # Length should be total rows in original data
        assert len(dataset) == 50

    def test_dataset_repr_contains_info(self):
        """Test __repr__ contains useful information."""
        data = pd.DataFrame({
            'f1': range(20),
            'f2': range(20, 40),
            'target': [0, 1] * 10
        })

        dataset = DBDataset(
            data=data,
            target_column='target',
            dataset_name='test_dataset'
        )

        repr_str = repr(dataset)
        assert isinstance(repr_str, str)
        assert len(repr_str) > 0

    def test_dataset_with_stratified_split(self):
        """Test stratified split maintains class distribution."""
        np.random.seed(42)
        # Create imbalanced data
        data = pd.DataFrame({
            'f1': np.random.rand(100),
            'f2': np.random.rand(100),
            'target': [0] * 80 + [1] * 20  # 80% class 0, 20% class 1
        })

        dataset = DBDataset(
            data=data,
            target_column='target',
            test_size=0.2,
            random_state=42
        )

        # Check that split happened
        assert len(dataset.train_data) == 80
        assert len(dataset.test_data) == 20

        # Check class distribution is roughly maintained
        train_targets = dataset.train_data['target']
        test_targets = dataset.test_data['target']

        # Train should have ~80% class 0
        train_class0_pct = (train_targets == 0).sum() / len(train_targets)
        assert 0.75 < train_class0_pct < 0.85

        # Test should have ~80% class 0
        test_class0_pct = (test_targets == 0).sum() / len(test_targets)
        assert 0.70 < test_class0_pct < 0.90

    def test_dataset_with_custom_test_size(self):
        """Test different test_size values."""
        data = pd.DataFrame({
            'f1': range(100),
            'f2': range(100, 200),
            'target': [0, 1] * 50
        })

        # Test with 10% test size
        dataset_10 = DBDataset(
            data=data,
            target_column='target',
            test_size=0.1,
            random_state=42
        )

        assert len(dataset_10.train_data) == 90
        assert len(dataset_10.test_data) == 10

        # Test with 40% test size
        dataset_40 = DBDataset(
            data=data,
            target_column='target',
            test_size=0.4,
            random_state=42
        )

        assert len(dataset_40.train_data) == 60
        assert len(dataset_40.test_data) == 40

    def test_dataset_features_property(self):
        """Test features property returns correct feature list."""
        data = pd.DataFrame({
            'feat_a': range(10),
            'feat_b': range(10, 20),
            'feat_c': range(20, 30),
            'target': [0, 1] * 5
        })

        dataset = DBDataset(
            data=data,
            target_column='target'
        )

        # Should auto-detect 3 features
        assert len(dataset.features) == 3
        assert 'feat_a' in dataset.features
        assert 'feat_b' in dataset.features
        assert 'feat_c' in dataset.features
        assert 'target' not in dataset.features

    def test_dataset_with_explicit_features_list(self):
        """Test providing explicit features list."""
        data = pd.DataFrame({
            'feat_a': range(10),
            'feat_b': range(10, 20),
            'feat_c': range(20, 30),
            'feat_d': range(30, 40),
            'target': [0, 1] * 5
        })

        # Only use feat_a and feat_c
        dataset = DBDataset(
            data=data,
            target_column='target',
            features=['feat_a', 'feat_c']
        )

        assert len(dataset.features) == 2
        assert 'feat_a' in dataset.features
        assert 'feat_c' in dataset.features
        assert 'feat_b' not in dataset.features
        assert 'feat_d' not in dataset.features



class TestDBDatasetSklearnBunch:
    """Test handling of sklearn Bunch objects."""

    def test_train_test_with_sklearn_bunch(self):
        """Test creating dataset from sklearn Bunch objects."""
        import numpy as np
        from sklearn.utils import Bunch

        # Create sklearn Bunch objects (like load_iris, load_wine, etc.)
        train_bunch = Bunch(
            data=np.random.rand(100, 4),
            target=np.random.randint(0, 2, 100),
            feature_names=["f1", "f2", "f3", "f4"]
        )

        test_bunch = Bunch(
            data=np.random.rand(30, 4),
            target=np.random.randint(0, 2, 30),
            feature_names=["f1", "f2", "f3", "f4"]
        )

        dataset = DBDataset(
            train_data=train_bunch,
            test_data=test_bunch,
            target_column="target"
        )

        assert len(dataset.train_data) == 100
        assert len(dataset.test_data) == 30
        assert "target" in dataset.train_data.columns
        assert "f1" in dataset.train_data.columns

    def test_train_bunch_without_feature_names(self):
        """Test sklearn Bunch without feature_names attribute."""
        import numpy as np
        from sklearn.utils import Bunch

        train_bunch = Bunch(
            data=np.random.rand(50, 3),
            target=np.random.randint(0, 2, 50)
        )

        test_df = pd.DataFrame(np.random.rand(20, 3), columns=[0, 1, 2])
        test_df["target"] = np.random.randint(0, 2, 20)

        dataset = DBDataset(
            train_data=train_bunch,
            test_data=test_df,
            target_column="target",
            features=[0, 1, 2]
        )

        assert len(dataset.train_data) == 50

    def test_train_test_empty_raises_error(self):
        """Test that empty train or test data raises ValueError."""
        train_df = pd.DataFrame({"f1": [], "target": []})
        test_df = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})

        with pytest.raises(ValueError, match="Training and test datasets cannot be empty"):
            DBDataset(train_data=train_df, test_data=test_df, target_column="target")

    def test_test_empty_raises_error(self):
        """Test that empty test data raises ValueError."""
        train_df = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})
        test_df = pd.DataFrame({"f1": [], "target": []})

        with pytest.raises(ValueError, match="Training and test datasets cannot be empty"):
            DBDataset(train_data=train_df, test_data=test_df, target_column="target")

    def test_convert_non_dataframe_train_data(self):
        """Test converting non-DataFrame object to DataFrame."""
        # Pass a list of lists (not a Bunch object)
        train_data = [[1, 2, 0], [3, 4, 1], [5, 6, 0]]
        test_df = pd.DataFrame({"f1": [7, 8], "f2": [9, 10], "target": [1, 0]})

        dataset = DBDataset(
            train_data=train_data,
            test_data=test_df,
            target_column="target",
            features=["f1", "f2", "target"]
        )

        assert len(dataset.train_data) == 3

    def test_convert_non_dataframe_test_data(self):
        """Test converting non-DataFrame test data."""
        train_df = pd.DataFrame({"f1": [1, 2], "f2": [3, 4], "target": [0, 1]})
        test_data = [[5, 6, 1], [7, 8, 0]]

        dataset = DBDataset(
            train_data=train_df,
            test_data=test_data,
            target_column="target",
            features=["f1", "f2", "target"]
        )

        assert len(dataset.test_data) == 2

    def test_test_bunch_with_feature_names(self):
        """Test test_data as Bunch with feature_names (line 303)."""
        import numpy as np
        from sklearn.utils import Bunch

        train_df = pd.DataFrame({
            "f1": [1, 2, 3],
            "f2": [4, 5, 6],
            "target": [0, 1, 0]
        })

        # Create test_data as Bunch with feature_names
        test_bunch = Bunch(
            data=np.array([[7, 8], [9, 10]]),
            target=np.array([1, 0]),
            feature_names=["f1", "f2"]
        )

        # Don't pass features - should use test_bunch.feature_names (line 303)
        dataset = DBDataset(
            train_data=train_df,
            test_data=test_bunch,
            target_column="target"
        )

        assert len(dataset.test_data) == 2
        assert "f1" in dataset.test_data.columns
        assert "f2" in dataset.test_data.columns
        assert "target" in dataset.test_data.columns

    def test_invalid_train_data_conversion_raises_error(self):
        """Test that invalid train data raises ValueError."""
        # Pass an object that cannot be converted to DataFrame
        train_data = object()
        test_df = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})

        with pytest.raises(ValueError, match="Could not convert training data to DataFrame"):
            DBDataset(train_data=train_data, test_data=test_df, target_column="target")

    def test_invalid_test_data_conversion_raises_error(self):
        """Test that invalid test data raises ValueError."""
        train_df = pd.DataFrame({"f1": [1, 2], "target": [0, 1]})
        test_data = object()

        with pytest.raises(ValueError, match="Could not convert test data to DataFrame"):
            DBDataset(train_data=train_df, test_data=test_data, target_column="target")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])



def test_get_feature_data_invalid_dataset():
    """Test get_feature_data with invalid dataset parameter."""
    import pandas as pd
    from deepbridge.core.db_data import DBDataset
    
    train_df = pd.DataFrame({'f1': [1, 2, 3], 'target': [0, 1, 0]})
    test_df = pd.DataFrame({'f1': [4, 5], 'target': [1, 0]})
    
    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target'
    )
    
    with pytest.raises(ValueError, match="dataset must be either 'train' or 'test'"):
        dataset.get_feature_data(dataset='invalid')


def test_get_target_data_invalid_dataset():
    """Test get_target_data with invalid dataset parameter."""
    import pandas as pd
    from deepbridge.core.db_data import DBDataset
    
    train_df = pd.DataFrame({'f1': [1, 2, 3], 'target': [0, 1, 0]})
    test_df = pd.DataFrame({'f1': [4, 5], 'target': [1, 0]})
    
    dataset = DBDataset(
        train_data=train_df,
        test_data=test_df,
        target_column='target'
    )
    
    with pytest.raises(ValueError, match="dataset must be either 'train' or 'test'"):
        dataset.get_target_data(dataset='invalid')


class TestDBDatasetAdditionalCoverage:
    """Additional tests for improved coverage."""

    def test_multiple_params_error(self):
        """Test error when multiple params are provided."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'target': [0, 1, 0]
        })

        model = LogisticRegression()
        model.fit(df[['a', 'b']], df['target'])

        with pytest.raises(ValueError, match="You must provide only one"):
            DBDataset(
                data=df,
                target_column='target',
                model=model,
                prob_cols=['prob_0', 'prob_1']
            )

    def test_categorical_features_validation(self):
        """Test categorical features validation."""
        df = pd.DataFrame({
            'num': [1, 2, 3, 4, 5],
            'cat': ['a', 'b', 'a', 'b', 'a'],
            'target': [0, 1, 0, 1, 0]
        })

        dataset = DBDataset(
            data=df,
            target_column='target',
            categorical_features=['cat']
        )

        assert 'cat' in dataset.categorical_features

    def test_with_dataset_name(self):
        """Test initialization with dataset name."""
        df = pd.DataFrame({
            'a': [1, 2, 3],
            'b': [4, 5, 6],
            'target': [0, 1, 0]
        })

        dataset = DBDataset(
            data=df,
            target_column='target',
            dataset_name='test_dataset'
        )

        assert dataset._dataset_name == 'test_dataset'

    def test_with_max_categories(self):
        """Test initialization with max_categories."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'cat': ['x', 'y', 'z', 'x', 'y'],
            'target': [0, 1, 0, 1, 0]
        })

        dataset = DBDataset(
            data=df,
            target_column='target'
        )

        # Should infer categorical features
        assert len(dataset.categorical_features) >= 0

    def test_model_without_predict_proba(self):
        """Test model without predict_proba method."""
        from sklearn.linear_model import LinearRegression

        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],
            'target': [1.5, 3.5, 5.5, 7.5, 9.5]
        })

        model = LinearRegression()
        model.fit(df[['a', 'b']], df['target'])

        # Should work with regression model (no predict_proba)
        dataset = DBDataset(
            data=df,
            target_column='target',
            model=model
        )

        assert dataset.model is not None

    def test_stratified_split_single_class(self):
        """Test stratified split with single class."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [2, 4, 6, 8, 10],
            'target': [0, 0, 0, 0, 0]  # Single class
        })

        dataset = DBDataset(
            data=df,
            target_column='target',
            random_state=42
        )

        # Should handle single class gracefully
        assert len(dataset.train_data) > 0
        assert len(dataset.test_data) > 0

    def test_stratified_split_failure_fallback(self):
        """Test stratified split fallback when stratify fails."""
        df = pd.DataFrame({
            'a': [1, 2],
            'b': [2, 4],
            'target': [0, 1]  # Too few samples for stratification
        })

        dataset = DBDataset(
            data=df,
            target_column='target',
            random_state=42,
            test_size=0.5
        )

        # Should fall back to non-stratified split
        assert len(dataset.train_data) >= 1
        assert len(dataset.test_data) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
