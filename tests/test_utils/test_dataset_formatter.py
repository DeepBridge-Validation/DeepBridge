"""
Testes para deepbridge.utils.dataset_formatter.DatasetFormatter

Objetivo: Elevar coverage de 0% para 100%
Foco: format_dataset_info com diferentes combinações de data
"""

import pytest
import pandas as pd
from unittest.mock import Mock

from deepbridge.utils.dataset_formatter import DatasetFormatter


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_feature_manager():
    """Cria mock de FeatureManager."""
    manager = Mock()
    manager.features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
    manager.categorical_features = ['feature1', 'feature2']
    manager.numerical_features = ['feature3', 'feature4', 'feature5']
    return manager


@pytest.fixture
def mock_model_handler_with_model():
    """Cria mock de ModelHandler com modelo."""
    handler = Mock()
    handler.model = Mock()  # Model loaded
    handler.predictions = None  # No predictions yet
    return handler


@pytest.fixture
def mock_model_handler_without_model():
    """Cria mock de ModelHandler sem modelo."""
    handler = Mock()
    handler.model = None
    handler.predictions = None
    return handler


@pytest.fixture
def mock_model_handler_with_predictions():
    """Cria mock de ModelHandler com modelo e predictions."""
    handler = Mock()
    handler.model = Mock()
    handler.predictions = pd.Series([0, 1, 0, 1, 1])
    return handler


@pytest.fixture
def sample_data():
    """Cria dataset de exemplo."""
    return pd.DataFrame({
        'feature1': ['A', 'B', 'C'],
        'feature2': ['X', 'Y', 'Z'],
        'feature3': [1, 2, 3],
        'target': [0, 1, 0]
    })


@pytest.fixture
def sample_train_test():
    """Cria datasets train/test."""
    train = pd.DataFrame({
        'feature1': ['A', 'B', 'C', 'D', 'E'],
        'target': [0, 1, 0, 1, 1]
    })
    test = pd.DataFrame({
        'feature1': ['X', 'Y'],
        'target': [0, 1]
    })
    return train, test


# ============================================================================
# Testes de Inicialização
# ============================================================================

class TestDatasetFormatterInitialization:
    """Testes de inicialização do DatasetFormatter."""

    def test_init_with_dataset_name(self, mock_feature_manager, mock_model_handler_with_model):
        """Testa inicialização com nome do dataset."""
        formatter = DatasetFormatter(
            dataset_name='MyDataset',
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target'
        )

        assert formatter._dataset_name == 'MyDataset'
        assert formatter._feature_manager == mock_feature_manager
        assert formatter._model_handler == mock_model_handler_with_model
        assert formatter._target_column == 'target'

    def test_init_without_dataset_name(self, mock_feature_manager, mock_model_handler_with_model):
        """Testa inicialização sem nome do dataset."""
        formatter = DatasetFormatter(
            dataset_name=None,
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='y'
        )

        assert formatter._dataset_name is None
        assert formatter._target_column == 'y'


# ============================================================================
# Testes de format_dataset_info
# ============================================================================

class TestDatasetFormatterFormatInfo:
    """Testes do método format_dataset_info."""

    def test_format_with_unified_data(
        self, mock_feature_manager, mock_model_handler_with_model, sample_data
    ):
        """Testa formatação com data unificado (não split)."""
        formatter = DatasetFormatter(
            dataset_name='TestDataset',
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target'
        )

        result = formatter.format_dataset_info(data=sample_data)

        # Verificar componentes
        assert "'TestDataset'" in result
        assert '3 samples (not split)' in result
        assert 'Features: 5 total' in result
        assert '2 categorical' in result
        assert '3 numerical' in result
        assert "Target: 'target'" in result
        assert 'Model: loaded' in result
        assert 'Predictions: not available' in result

    def test_format_with_train_test_split(
        self, mock_feature_manager, mock_model_handler_with_model, sample_train_test
    ):
        """Testa formatação com train/test split."""
        train, test = sample_train_test

        formatter = DatasetFormatter(
            dataset_name='SplitDataset',
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target'
        )

        result = formatter.format_dataset_info(train_data=train, test_data=test)

        # Verificar componentes
        assert "'SplitDataset'" in result
        assert '5 training samples' in result
        assert '2 test samples' in result
        assert 'not split' not in result
        assert 'Features: 5 total' in result
        assert 'Model: loaded' in result

    def test_format_without_dataset_name(
        self, mock_feature_manager, mock_model_handler_with_model, sample_data
    ):
        """Testa formatação sem nome do dataset."""
        formatter = DatasetFormatter(
            dataset_name=None,
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target'
        )

        result = formatter.format_dataset_info(data=sample_data)

        # Não deve ter nome do dataset
        assert "'" not in result.split('\n')[0]  # Primeira linha não tem aspas
        assert 'DBDataset(with 3 samples' in result

    def test_format_without_model(
        self, mock_feature_manager, mock_model_handler_without_model, sample_data
    ):
        """Testa formatação sem modelo carregado."""
        formatter = DatasetFormatter(
            dataset_name='NoModel',
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_without_model,
            target_column='target'
        )

        result = formatter.format_dataset_info(data=sample_data)

        assert 'Model: not loaded' in result
        assert 'Predictions: not available' in result

    def test_format_with_predictions(
        self, mock_feature_manager, mock_model_handler_with_predictions, sample_data
    ):
        """Testa formatação com predictions disponíveis."""
        formatter = DatasetFormatter(
            dataset_name='WithPredictions',
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_predictions,
            target_column='target'
        )

        result = formatter.format_dataset_info(data=sample_data)

        assert 'Model: loaded' in result
        assert 'Predictions: available' in result

    def test_format_structure(
        self, mock_feature_manager, mock_model_handler_with_model, sample_data
    ):
        """Testa estrutura do output formatado."""
        formatter = DatasetFormatter(
            dataset_name='StructureTest',
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target'
        )

        result = formatter.format_dataset_info(data=sample_data)

        # Deve ter 5 linhas
        lines = result.split('\n')
        assert len(lines) == 5

        # Verificar conteúdo de cada linha
        assert lines[0].startswith("DBDataset('StructureTest'")
        assert lines[1].startswith('Features:')
        assert lines[2].startswith('Target:')
        assert lines[3].startswith('Model:')
        assert lines[4].startswith('Predictions:')


# ============================================================================
# Testes de Edge Cases
# ============================================================================

class TestDatasetFormatterEdgeCases:
    """Testes de casos extremos."""

    def test_format_with_zero_categorical_features(
        self, mock_model_handler_with_model, sample_data
    ):
        """Testa com zero features categóricas."""
        feature_manager = Mock()
        feature_manager.features = ['f1', 'f2', 'f3']
        feature_manager.categorical_features = []  # Zero categorical
        feature_manager.numerical_features = ['f1', 'f2', 'f3']

        formatter = DatasetFormatter(
            dataset_name='NoCategorical',
            feature_manager=feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target'
        )

        result = formatter.format_dataset_info(data=sample_data)

        assert '3 total' in result
        assert '0 categorical' in result
        assert '3 numerical' in result

    def test_format_with_zero_numerical_features(
        self, mock_model_handler_with_model, sample_data
    ):
        """Testa com zero features numéricas."""
        feature_manager = Mock()
        feature_manager.features = ['f1', 'f2']
        feature_manager.categorical_features = ['f1', 'f2']
        feature_manager.numerical_features = []  # Zero numerical

        formatter = DatasetFormatter(
            dataset_name='NoNumerical',
            feature_manager=feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target'
        )

        result = formatter.format_dataset_info(data=sample_data)

        assert '2 total' in result
        assert '2 categorical' in result
        assert '0 numerical' in result

    def test_format_with_single_sample(
        self, mock_feature_manager, mock_model_handler_with_model
    ):
        """Testa com dataset de uma única amostra."""
        single_sample = pd.DataFrame({'feature1': ['A'], 'target': [1]})

        formatter = DatasetFormatter(
            dataset_name='SingleSample',
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target'
        )

        result = formatter.format_dataset_info(data=single_sample)

        assert '1 samples (not split)' in result

    def test_format_with_large_dataset(
        self, mock_feature_manager, mock_model_handler_with_model
    ):
        """Testa com dataset grande."""
        large_data = pd.DataFrame({
            'feature1': range(10000),
            'target': [0] * 10000
        })

        formatter = DatasetFormatter(
            dataset_name='LargeDataset',
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target'
        )

        result = formatter.format_dataset_info(data=large_data)

        assert '10000 samples (not split)' in result

    def test_format_with_special_characters_in_name(
        self, mock_feature_manager, mock_model_handler_with_model, sample_data
    ):
        """Testa com caracteres especiais no nome do dataset."""
        formatter = DatasetFormatter(
            dataset_name="Dataset-With_Special.Chars!",
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target'
        )

        result = formatter.format_dataset_info(data=sample_data)

        assert "'Dataset-With_Special.Chars!'" in result

    def test_format_with_special_characters_in_target(
        self, mock_feature_manager, mock_model_handler_with_model, sample_data
    ):
        """Testa com caracteres especiais no nome do target."""
        formatter = DatasetFormatter(
            dataset_name='Test',
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target_var-2.0'
        )

        result = formatter.format_dataset_info(data=sample_data)

        assert "'target_var-2.0'" in result


# ============================================================================
# Testes de Integração
# ============================================================================

class TestDatasetFormatterIntegration:
    """Testes de integração (múltiplos cenários)."""

    def test_full_pipeline_unified_to_split(
        self, mock_feature_manager, mock_model_handler_with_model
    ):
        """Testa pipeline de unified para split."""
        formatter = DatasetFormatter(
            dataset_name='Pipeline',
            feature_manager=mock_feature_manager,
            model_handler=mock_model_handler_with_model,
            target_column='target'
        )

        # Primeiro com unified
        unified = pd.DataFrame({'f1': [1, 2, 3]})
        result1 = formatter.format_dataset_info(data=unified)
        assert '3 samples (not split)' in result1

        # Depois com split
        train = pd.DataFrame({'f1': [1, 2]})
        test = pd.DataFrame({'f1': [3]})
        result2 = formatter.format_dataset_info(train_data=train, test_data=test)
        assert '2 training samples' in result2
        assert '1 test samples' in result2

    def test_different_feature_combinations(self, mock_model_handler_with_model, sample_data):
        """Testa diferentes combinações de features."""
        # 10 categorical, 5 numerical
        fm1 = Mock()
        fm1.features = ['f' + str(i) for i in range(15)]
        fm1.categorical_features = ['f' + str(i) for i in range(10)]
        fm1.numerical_features = ['f' + str(i) for i in range(10, 15)]

        formatter1 = DatasetFormatter('Test1', fm1, mock_model_handler_with_model, 'target')
        result1 = formatter1.format_dataset_info(data=sample_data)
        assert '15 total' in result1
        assert '10 categorical' in result1
        assert '5 numerical' in result1

        # 1 categorical, 1 numerical
        fm2 = Mock()
        fm2.features = ['f1', 'f2']
        fm2.categorical_features = ['f1']
        fm2.numerical_features = ['f2']

        formatter2 = DatasetFormatter('Test2', fm2, mock_model_handler_with_model, 'target')
        result2 = formatter2.format_dataset_info(data=sample_data)
        assert '2 total' in result2
        assert '1 categorical' in result2
        assert '1 numerical' in result2

    def test_model_lifecycle(self, mock_feature_manager, sample_data):
        """Testa ciclo de vida do modelo (no model -> model -> predictions)."""
        # Stage 1: No model
        mh1 = Mock()
        mh1.model = None
        mh1.predictions = None

        formatter = DatasetFormatter('Lifecycle', mock_feature_manager, mh1, 'target')
        result1 = formatter.format_dataset_info(data=sample_data)
        assert 'Model: not loaded' in result1
        assert 'Predictions: not available' in result1

        # Stage 2: Model loaded
        mh2 = Mock()
        mh2.model = Mock()
        mh2.predictions = None

        formatter._model_handler = mh2
        result2 = formatter.format_dataset_info(data=sample_data)
        assert 'Model: loaded' in result2
        assert 'Predictions: not available' in result2

        # Stage 3: Predictions available
        mh3 = Mock()
        mh3.model = Mock()
        mh3.predictions = pd.Series([0, 1, 0])

        formatter._model_handler = mh3
        result3 = formatter.format_dataset_info(data=sample_data)
        assert 'Model: loaded' in result3
        assert 'Predictions: available' in result3
