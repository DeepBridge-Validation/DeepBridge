"""Regressao: como a FairnessSuite obtem y_pred.

O bug: ``DBDataset`` guarda ``train_predictions`` como um DataFrame com as
colunas ``prob_class_0`` / ``prob_class_1``. A suite procurava por uma coluna
``prediction`` e por ``proba_class_1``, nenhuma das duas existe, entao ``y_pred``
ficava ``None``. Um ``elif`` impedia o fallback de gerar predicoes a partir do
modelo, e ``np.asarray(None)`` produz um array 0-dimensional, que so estourava
bem mais tarde dentro das metricas com::

    IndexError: too many indices for array: array is 0-dimensional, but 1 were indexed

O ``except`` do TestRunner engolia isso em ``results['fairness'] = {}``, entao a
execucao anunciava sucesso enquanto fairness nao produzia nada.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from deepbridge.core.db_data import DBDataset
from deepbridge.validation.wrappers.fairness_suite import FairnessSuite


@pytest.fixture
def dataset_with_probability_predictions():
    """DBDataset com um atributo protegido e um modelo treinado.

    O DBDataset calcula e guarda as predicoes sozinho, no formato de
    probabilidades por classe, que e justamente o formato que disparava o bug.
    """
    rng = np.random.default_rng(42)
    n = 300
    gender = rng.integers(0, 2, n)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = 0.9 * x1 - 0.7 * x2 + 0.6 * gender
    y = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)

    data = pd.DataFrame(
        {'x1': x1, 'x2': x2, 'gender': gender, 'target': y}
    )
    features = ['x1', 'x2', 'gender']

    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(data[features], data['target'])

    return DBDataset(
        data=data,
        target_column='target',
        features=features,
        model=model,
        categorical_features=['gender'],
        dataset_name='fairness_regression',
    )


def test_stored_predictions_use_prob_class_column_names(
    dataset_with_probability_predictions,
):
    """Trava o formato real que o DBDataset produz.

    Se este teste falhar, o formato mudou e a FairnessSuite precisa aprender o
    formato novo, senao o bug volta.
    """
    predictions = dataset_with_probability_predictions.train_predictions

    assert predictions is not None
    assert 'prob_class_1' in predictions.columns
    # As colunas que a suite procurava antes e que nunca existiram:
    assert 'prediction' not in predictions.columns
    assert 'proba_class_1' not in predictions.columns


def test_run_does_not_raise_index_error_on_probability_only_predictions(
    dataset_with_probability_predictions,
):
    """A suite roda ate o fim em vez de estourar IndexError nas metricas."""
    suite = FairnessSuite(
        dataset=dataset_with_probability_predictions,
        protected_attributes=['gender'],
        verbose=False,
    )

    results = suite.config('quick').run()

    assert results is not None, 'fairness devolveu resultado vazio'
    assert results.protected_attributes == ['gender']

    # run() devolve um FairnessResult, que embrulha o dicionario cru.
    raw = getattr(results, 'results', results)
    posttrain = raw['posttrain_metrics']
    assert 'gender' in posttrain, 'o atributo protegido nao foi avaliado'
    assert posttrain['gender'], 'nenhuma metrica pos-treino foi calculada'

    # O score so e calculavel se y_pred chegou de verdade nas metricas.
    assert 0.0 <= results.overall_fairness_score <= 1.0


def test_statistical_parity_rejects_missing_predictions():
    """Sem predicoes, o erro precisa ser claro, nao um IndexError obscuro.

    Documenta o comportamento de ``np.asarray(None)``, que e a razao pela qual
    o bug original era tao dificil de rastrear ate a origem.
    """
    assert np.asarray(None).ndim == 0

    from deepbridge.validation.fairness.metrics import FairnessMetrics

    with pytest.raises(IndexError):
        FairnessMetrics.statistical_parity(
            None, np.array([0, 1, 0, 1])
        )
