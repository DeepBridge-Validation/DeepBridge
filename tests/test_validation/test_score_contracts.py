"""Trava o contrato de dominio dos scores agregados de relatorio.

Contexto (item 3.3 do plano de reorganizacao): o ReportGenerator novo recusava
resultado legitimo das suites de validacao com

    robustness_score must be between 0 and 1, got 1.0105318355379287
    uncertainty_score must be between 0 and 1, got 1.0857142857142859

A decisao tomada foi que o teto de 1.0 estava errado, nao o calculo. Os quatro
scores sao razoes ("fracao do desempenho retida", "fracao da cobertura nominal
atingida"), nao notas normalizadas, e passar de 1.0 e um resultado verdadeiro:
o modelo foi melhor sob perturbacao. O contrato passou a ser
[SCORE_MIN, SCORE_MAX] = [0, 1.25], com o teto servindo para pegar erro de
codificacao (valor em escala percentual, score nao normalizado, NaN) e nao
desempenho do modelo.

Estes testes travam esse contrato para os quatro tipos. Se alguem reapertar o
teto para 1.0, o bug volta e estes testes quebram.

REVISAO DO CONTRATO (auditoria seguinte). Duas coisas estavam erradas na
primeira versao:

1. O piso. O argumento "o numero e verdadeiro, nao clampe" valia igual para
   baixo e so tinha sido aplicado para cima. robustness_suite.py calcula
   impact = (perturbado - base)/base para MSE/MAE/RMSE, logo uma perturbacao
   que mais que dobra o erro da impact > 1 e score NEGATIVO - medicao
   verdadeira, nao erro de codificacao. Recusar isso impedia a geracao do
   relatorio de um resultado real, exatamente o bug que o teto consertou.
   Agora o piso avisa (ScoreOutOfRangeWarning) e devolve o valor medido; so o
   teto levanta ValueError, porque o excedente acima de 1.0 e limitado por
   mecanismo e o abaixo de 0.0 nao e.

2. O gap de resilience. Em duas das cinco rotas o gap era
   'referencia - degradado' sem corrigir o sinal para metrica de erro, entao
   com metric='mse'/'mae' o gap saia negativo justamente quando o modelo era
   MENOS resiliente e o score passava de 1.0 ("mais que perfeitamente
   resiliente") no caso em que o modelo degradou. E o gap era uma diferenca
   ABSOLUTA de MSE, sem limite de magnitude, entao nenhum intervalo fechado
   era defensavel. As duas coisas estao consertadas em resilience_suite.py e
   travadas aqui.
"""

import math

import pytest

from deepbridge.core.experiment.report.data.base import (
    SCORE_MAX,
    SCORE_MIN,
    ScoreOutOfRangeWarning,
    validate_score,
)
from deepbridge.core.experiment.report.data.fairness import (
    FairnessMetrics,
    FairnessReportData,
)
from deepbridge.core.experiment.report.data.resilience import (
    ResilienceMetrics,
    ResilienceReportData,
)
from deepbridge.core.experiment.report.data.robustness import (
    RobustnessMetrics,
    RobustnessReportData,
)
from deepbridge.core.experiment.report.data.uncertainty import (
    UncertaintyMetrics,
    UncertaintyReportData,
)

# Valores medidos de verdade pelo spike desenvolvimento/spikes/
# compare_report_paths.py num dataset de 1200 linhas. Eram exatamente os
# valores que o ReportGenerator recusava antes da correcao.
MEASURED_ROBUSTNESS_OVERSHOOT = 1.0105318355379287
MEASURED_UNCERTAINTY_OVERSHOOT = 1.0857142857142859


def _robustness(score):
    return RobustnessReportData(
        generated_at=None,
        report_type='robustness',
        model_name='M',
        model_type='RandomForestClassifier',
        metrics=RobustnessMetrics(robustness_score=score, base_score=0.8),
    )


def _uncertainty(score):
    return UncertaintyReportData(
        generated_at=None,
        report_type='uncertainty',
        model_name='M',
        model_type='RandomForestClassifier',
        metrics=UncertaintyMetrics(
            uncertainty_score=score, avg_coverage=0.9, avg_width=0.3
        ),
    )


def _resilience(score):
    return ResilienceReportData(
        generated_at=None,
        report_type='resilience',
        model_name='M',
        model_type='RandomForestClassifier',
        metrics=ResilienceMetrics(resilience_score=score, base_score=0.8),
    )


def _fairness(score):
    return FairnessReportData(
        generated_at=None,
        report_type='fairness',
        model_name='M',
        model_type='RandomForestClassifier',
        metrics=FairnessMetrics(fairness_score=score),
    )


BUILDERS = [
    pytest.param(_robustness, 'robustness_score', id='robustness'),
    pytest.param(_uncertainty, 'uncertainty_score', id='uncertainty'),
    pytest.param(_resilience, 'resilience_score', id='resilience'),
    pytest.param(_fairness, 'fairness_score', id='fairness'),
]


# ---------------------------------------------------------------------------
# O contrato em si
# ---------------------------------------------------------------------------


def test_contract_ceiling_is_above_one():
    """O teto tem de ficar acima de 1.0, senao o bug do item 3.3 volta."""
    assert SCORE_MIN == 0.0
    assert SCORE_MAX > 1.0


@pytest.mark.parametrize('build,field_name', BUILDERS)
@pytest.mark.parametrize('score', [0.0, 0.5, 1.0])
def test_scores_inside_zero_one_are_accepted(build, field_name, score):
    assert build(score).validate() is True


@pytest.mark.parametrize('build,field_name', BUILDERS)
def test_score_slightly_above_one_is_accepted(build, field_name):
    """1.05 significa "foi melhor sob perturbacao" e e um valor valido."""
    assert build(1.05).validate() is True


@pytest.mark.parametrize('build,field_name', BUILDERS)
def test_score_exactly_at_ceiling_is_accepted(build, field_name):
    """A borda superior e inclusiva."""
    assert build(SCORE_MAX).validate() is True


@pytest.mark.parametrize('build,field_name', BUILDERS)
def test_score_just_above_ceiling_is_rejected(build, field_name):
    with pytest.raises(ValueError, match=field_name):
        build(SCORE_MAX + 0.01).validate()


@pytest.mark.parametrize('build,field_name', BUILDERS)
@pytest.mark.parametrize('absurd', [10.0, 85.0])
def test_absurd_scores_are_still_rejected(build, field_name, absurd):
    """Valor absurdo continua recusado: o teto pega erro de codificacao.

    85.0 e o caso concreto de valor em escala percentual (deveria ser 0.85).
    """
    with pytest.raises(ValueError, match=field_name):
        build(absurd).validate()


@pytest.mark.parametrize('build,field_name', BUILDERS)
def test_score_below_floor_is_reported_with_a_warning(build, field_name):
    """Score negativo e medicao possivel, entao avisa em vez de recusar.

    Com metrica de erro (MSE/MAE/RMSE) a perda relativa pode passar de 100%:
    impact = (perturbado - base)/base > 1 da score < 0. Recusar bloquearia o
    relatorio de um resultado real. O valor nao e clampado.
    """
    with pytest.warns(ScoreOutOfRangeWarning, match=field_name):
        assert build(SCORE_MIN - 0.01).validate() is True


def test_negative_score_is_not_clamped():
    """O numero publicado e o medido, nao 0.0."""
    with pytest.warns(ScoreOutOfRangeWarning):
        assert validate_score('robustness_score', -1.4) == -1.4


def test_floor_and_ceiling_messages_are_different():
    """A mensagem do teto nao servia para o piso: falava de "melhor sob
    perturbacao" para um score negativo, texto sem sentido para quem le o
    traceback."""
    with pytest.raises(ValueError) as excinfo:
        validate_score('robustness_score', SCORE_MAX + 1)
    ceiling_message = str(excinfo.value)
    assert 'must not exceed' in ceiling_message

    with pytest.warns(ScoreOutOfRangeWarning) as record:
        validate_score('robustness_score', -0.5)
    floor_message = str(record[0].message)
    assert 'below' in floor_message
    assert 'performed better under perturbation' not in floor_message


@pytest.mark.parametrize('truthy', [True, False])
def test_validate_score_rejects_bool(truthy):
    """bool e subclasse de int: float(True) daria um score "perfeito" 1.0."""
    with pytest.raises(ValueError, match='must be a number'):
        validate_score('robustness_score', truthy)


def test_validate_score_rejects_numpy_bool():
    import numpy as np

    with pytest.raises(ValueError, match='must be a number'):
        validate_score('robustness_score', np.bool_(True))


def test_the_two_layers_agree_on_the_contract():
    """O layer pydantic (report/domain/) usava le=1.0 enquanto o layer de
    dataclasses (report/data/) usava 1.25: duas regras contraditorias para o
    mesmo campo, com teste ativo de cada lado."""
    from pydantic import ValidationError

    from deepbridge.core.experiment.report.domain.resilience import (
        ResilienceMetrics as PydanticResilience,
    )
    from deepbridge.core.experiment.report.domain.robustness import (
        RobustnessMetrics as PydanticRobustness,
    )
    from deepbridge.core.experiment.report.domain.uncertainty import (
        UncertaintyMetrics as PydanticUncertainty,
    )

    cases = [
        (PydanticRobustness, 'robustness_score'),
        (PydanticUncertainty, 'uncertainty_score'),
        (PydanticResilience, 'resilience_score'),
    ]
    for model, field in cases:
        assert getattr(model(**{field: 1.05}), field) == 1.05
        assert getattr(model(**{field: SCORE_MAX}), field) == SCORE_MAX
        with pytest.raises(ValidationError):
            model(**{field: SCORE_MAX + 0.01})


@pytest.mark.parametrize('build,field_name', BUILDERS)
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
def test_non_finite_scores_are_rejected(build, field_name, bad):
    """NaN/inf nunca passam: sao sinal de divisao por zero mascarada."""
    with pytest.raises(ValueError, match=field_name):
        build(bad).validate()


# ---------------------------------------------------------------------------
# Os valores que o spike mediu de verdade
# ---------------------------------------------------------------------------


def test_measured_robustness_overshoot_is_accepted():
    """O valor exato que o ReportGenerator recusava antes da correcao."""
    assert _robustness(MEASURED_ROBUSTNESS_OVERSHOOT).validate() is True


def test_measured_uncertainty_overshoot_is_accepted():
    assert _uncertainty(MEASURED_UNCERTAINTY_OVERSHOOT).validate() is True


# ---------------------------------------------------------------------------
# validate_score, o helper compartilhado
# ---------------------------------------------------------------------------


def test_validate_score_returns_the_float():
    assert validate_score('robustness_score', 1.05) == 1.05


def test_validate_score_rejects_non_numeric():
    with pytest.raises(ValueError, match='must be a number'):
        validate_score('robustness_score', 'alto')


def test_validate_score_rejects_none():
    with pytest.raises(ValueError, match='must be a number'):
        validate_score('robustness_score', None)


def test_validate_score_message_names_the_field():
    with pytest.raises(ValueError) as excinfo:
        validate_score('uncertainty_score', 42.0)
    assert 'uncertainty_score' in str(excinfo.value)
    assert '1.25' in str(excinfo.value)


# ---------------------------------------------------------------------------
# Os calculos das suites nao devem reintroduzir o clamp em 1.0
# ---------------------------------------------------------------------------


def test_robustness_formula_keeps_the_overshoot():
    """robustness_score = 1 - impact, sem clamp, impact pode ser negativo."""
    avg_overall_impact = -0.0105318355379287
    score = 1.0 - avg_overall_impact
    assert score > 1.0
    assert _robustness(score).validate() is True


def test_robustness_transformer_does_not_clamp():
    from deepbridge.core.experiment.report.data.robustness import (
        RobustnessDataTransformer,
    )

    data = {
        'model_name': 'M',
        'model_type': 'RandomForestClassifier',
        'base_score': 0.8,
        'avg_overall_impact': -0.01,
    }
    metrics = RobustnessDataTransformer()._extract_metrics(data)
    assert metrics.robustness_score == pytest.approx(1.01)


def test_resilience_transformer_does_not_clamp_at_one():
    """Gap negativo (foi melhor sob shift) tem de virar score acima de 1."""
    from deepbridge.core.experiment.report.data.resilience import (
        ResilienceDataTransformer,
    )

    data = {
        'model_name': 'M',
        'model_type': 'RandomForestClassifier',
        'distribution_shift': {
            'all_results': [
                {'performance_gap': -0.04, 'feature_name': 'x1'},
                {'performance_gap': -0.02, 'feature_name': 'x2'},
            ]
        },
    }
    metrics = ResilienceDataTransformer()._extract_metrics(data)
    assert metrics.resilience_score == pytest.approx(1.03)
    # a magnitude ainda ranqueia a feature mais afetada
    assert metrics.max_performance_gap == pytest.approx(0.04)
    assert metrics.most_affected_feature == 'x1'


def test_uncertainty_transformer_reads_the_suite_score():
    """A suite publica 'uncertainty_quality_score'; o transformer tem de ler."""
    from deepbridge.core.experiment.report.data.uncertainty import (
        UncertaintyDataTransformer,
    )

    data = {'uncertainty_quality_score': 0.73}
    metrics = UncertaintyDataTransformer()._extract_metrics(data, [])
    assert metrics.uncertainty_score == pytest.approx(0.73)


def test_fairness_transformer_reads_the_suite_score():
    """A suite publica 'overall_fairness_score'; sem ler, caia em 1.0 falso."""
    from deepbridge.core.experiment.report.data.fairness import (
        FairnessDataTransformer,
    )

    data = {'overall_fairness_score': 0.62}
    metrics = FairnessDataTransformer()._extract_metrics(data)
    assert metrics.fairness_score == pytest.approx(0.62)


def test_uncertainty_coverage_ratio_fallback_can_exceed_one():
    """Sem score da suite, a razao de cobertura ainda pode passar de 1."""
    from deepbridge.core.experiment.report.data.uncertainty import (
        CRQRResult,
        UncertaintyDataTransformer,
    )

    results = [
        CRQRResult(
            alpha=0.2,
            coverage=0.95,
            expected_coverage=0.8,
            mean_width=0.3,
            median_width=0.3,
        )
    ]
    metrics = UncertaintyDataTransformer()._extract_metrics({}, results)
    assert metrics.uncertainty_score > 1.0
    assert not math.isnan(metrics.uncertainty_score)
    assert _uncertainty(metrics.uncertainty_score).validate() is True


# ---------------------------------------------------------------------------
# O gap de resilience: sinal e escala
# ---------------------------------------------------------------------------
#
# Sem estas duas propriedades o contrato de score acima nao tem como valer
# para resilience: o score e 1 - mean(gap).


def test_signed_gap_points_the_same_way_for_every_metric():
    """gap > 0 tem de significar "o subconjunto degradado esta pior".

    Para metrica de erro (MSE/MAE/RMSE/SMAPE) menor e melhor, logo o sinal se
    inverte. Duas rotas da suite nao faziam essa correcao.
    """
    from deepbridge.validation.wrappers.resilience_suite import (
        signed_performance_gap,
    )

    # metrica de erro: o pior subconjunto tem erro MAIOR
    for metric in ['mse', 'mae', 'rmse', 'smape', 'MSE']:
        assert signed_performance_gap(metric, 300.0, 50.0) > 0, metric
        assert signed_performance_gap(metric, 50.0, 300.0) < 0, metric

    # metrica de score: o pior subconjunto tem score MENOR
    for metric in ['auc', 'f1', 'accuracy', 'r2', 'precision', 'recall']:
        assert signed_performance_gap(metric, 0.5, 0.9) > 0, metric
        assert signed_performance_gap(metric, 0.9, 0.5) < 0, metric


def test_relative_gap_is_dimensionless():
    """1 - gap so faz sentido se o gap for uma fracao.

    A diferenca absoluta de MSE esta em unidade de erro ao quadrado e nao tem
    limite: um gap medio de 30 produzia score -29, e um gap de -30 produzia
    31 (acima de SCORE_MAX, derrubando a geracao do relatorio).
    """
    from deepbridge.validation.wrappers.resilience_suite import (
        relative_performance_gap,
    )

    # erro dobrou em relacao a referencia -> perdeu 100% do desempenho
    assert relative_performance_gap('mse', 100.0, 50.0) == pytest.approx(1.0)
    # metade do erro da referencia -> foi MELHOR, fracao negativa
    assert relative_performance_gap('mse', 25.0, 50.0) == pytest.approx(-0.5)
    # score metric: perdeu 1/4 do f1 de referencia
    assert relative_performance_gap('f1', 0.6, 0.8) == pytest.approx(0.25)


def test_relative_gap_is_undefined_when_the_reference_is_degenerate():
    """Referencia ~0 nao serve de denominador: nan em vez de 1e10.

    O nan fica de fora da media do score (e nao vira 0.0, que seria
    indistinguivel de "gap medido zero").
    """
    from deepbridge.validation.wrappers.resilience_suite import (
        relative_performance_gap,
    )

    assert math.isnan(relative_performance_gap('mse', 5.0, 0.0))
    assert math.isnan(relative_performance_gap('mse', 5.0, float('nan')))
    assert math.isnan(relative_performance_gap('mse', None, 1.0))


@pytest.fixture(scope='module')
def regression_dataset_with_model():
    """Dataset de regressao com modelo treinado (MSE e metrica de erro)."""
    import numpy as np
    import pandas as pd
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression

    from deepbridge.core.db_data import DBDataset

    X, y = make_regression(
        n_samples=300,
        n_features=5,
        n_informative=3,
        noise=10.0,
        random_state=123,
    )
    frame = pd.DataFrame({f'f{i}': X[:, i] for i in range(5)})
    frame['target'] = y
    dataset = DBDataset(data=frame, target_column='target')
    features = [f'f{i}' for i in range(5)]
    model = LinearRegression().fit(
        dataset.train_data[features], dataset.train_data['target']
    )
    dataset.set_model(model)
    return dataset


@pytest.mark.parametrize(
    'method,params',
    [
        (
            'distribution_shift',
            {'alpha': 0.2, 'metric': 'mse', 'distance_metric': 'PSI'},
        ),
        (
            'worst_sample',
            {'alpha': 0.2, 'metric': 'mse', 'ranking_method': 'residual'},
        ),
    ],
)
def test_error_metric_degradation_never_reads_as_resilient(
    regression_dataset_with_model, method, params
):
    """As duas rotas que nao corrigiam o sinal, no caminho real.

    O subconjunto degradado tem MSE maior, logo o gap tem de ser POSITIVO e o
    score (1 - gap relativo) tem de ficar ABAIXO de 1.0. Antes o gap saia
    negativo e o score passava de 1.0 - "mais que perfeitamente resiliente"
    exatamente no caso em que o modelo degradou - e a magnitude sem limite
    ainda estourava SCORE_MAX, derrubando a geracao do relatorio.
    """
    from deepbridge.validation.wrappers.resilience_suite import (
        ResilienceSuite,
    )

    suite = ResilienceSuite(
        dataset=regression_dataset_with_model, verbose=False, random_state=42
    )
    result = getattr(suite, f'evaluate_{method}')(
        method=method, params=params
    )

    assert result['worst_metric'] > result['remaining_metric']
    assert result['performance_gap'] > 0
    assert result['performance_gap_relative'] > 0
    assert 1.0 - result['performance_gap_relative'] < 1.0


# ---------------------------------------------------------------------------
# O transformer de resilience: mesma regra do resto do contrato
# ---------------------------------------------------------------------------


def test_resilience_transformer_prefers_the_relative_gap():
    """Com MSE, so o gap relativo e comparavel a 1.0.

    O gap absoluto esta em unidade de erro ao quadrado: 1 - 271 nao e um
    score. O transformer tem de usar o gap relativo que a suite publica.
    """
    from deepbridge.core.experiment.report.data.resilience import (
        ResilienceDataTransformer,
    )

    data = {
        'model_name': 'M',
        'model_type': 'LinearRegression',
        'distribution_shift': {
            'all_results': [
                {
                    'performance_gap': 271.7,
                    'performance_gap_relative': 0.6,
                    'feature_name': 'x1',
                },
                {
                    'performance_gap': 180.0,
                    'performance_gap_relative': 0.4,
                    'feature_name': 'x2',
                },
            ]
        },
    }
    metrics = ResilienceDataTransformer()._extract_metrics(data)
    assert metrics.resilience_score == pytest.approx(0.5)
    # a magnitude absoluta continua disponivel para ranquear features
    assert metrics.max_performance_gap == pytest.approx(271.7)
    assert metrics.most_affected_feature == 'x1'


def test_resilience_transformer_does_not_clamp_at_the_floor():
    """Perda acima de 100% do desempenho de referencia nao vira 0.0.

    O max(0.0, ...) antigo publicava "nenhuma resiliencia" para qualquer
    magnitude de perda, apagando a diferenca entre perder tudo e perder tres
    vezes tudo. O valor sai como medido; quem valida avisa.
    """
    from deepbridge.core.experiment.report.data.resilience import (
        ResilienceDataTransformer,
    )

    data = {
        'model_name': 'M',
        'model_type': 'LinearRegression',
        'distribution_shift': {
            'all_results': [
                {
                    'performance_gap': 300.0,
                    'performance_gap_relative': 3.0,
                    'feature_name': 'x1',
                }
            ]
        },
    }
    metrics = ResilienceDataTransformer()._extract_metrics(data)
    assert metrics.resilience_score == pytest.approx(-2.0)


def test_crqr_result_publishes_the_name_the_templates_read():
    """Os templates de uncertainty leem result.actual_coverage.

    to_dict() emite 'coverage' (nome interno, usado no round-trip do
    transformer) e 'actual_coverage' (nome dos templates e do caminho
    legado). Sem o alias o Jinja2 resolvia Undefined e o render morria.
    """
    from deepbridge.core.experiment.report.data.uncertainty import CRQRResult

    payload = CRQRResult(
        alpha=0.1,
        coverage=0.93,
        expected_coverage=0.9,
        mean_width=0.3,
        median_width=0.3,
    ).to_dict()

    assert payload['actual_coverage'] == payload['coverage'] == 0.93
    assert payload['expected_coverage'] == 0.9
