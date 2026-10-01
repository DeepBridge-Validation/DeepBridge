"""
Base classes for report data structures and transformers.

This module defines the base protocols and abstract classes used throughout
the report generation data layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from datetime import datetime
import math
import warnings


# ---------------------------------------------------------------------------
# Score domain contract
# ---------------------------------------------------------------------------
#
# Todos os quatro tipos de relatorio (robustness, uncertainty, resilience,
# fairness) publicam um score agregado onde 1.0 significa "o modelo manteve
# integralmente a qualidade medida". Os scores NAO sao notas normalizadas:
# cada suite os calcula como uma razao ou como 1 - perda RELATIVA, por exemplo
#
#   robustness_score  = 1 - mean(impact)   , impact = (base - perturbado)/base
#   resilience_score  = 1 - mean(gap relativo ao subconjunto de referencia)
#   uncertainty_score = mean(coverage / expected_coverage)
#   fairness_score    = media ponderada de razoes entre grupos
#
# O denominador em todas elas e o desempenho de referencia, logo o score e
# adimensional e comparavel a 1.0. Isso e condicao para o contrato abaixo
# fazer sentido: enquanto a suite de resilience somava diferencas ABSOLUTAS de
# MSE (unidade de erro ao quadrado, sem limite), nenhum intervalo fechado era
# defensavel - um gap medio de 30 produzia score -29 e um gap de -30 produzia
# 31. Essa normalizacao esta em validation/wrappers/resilience_suite.py
# (relative_performance_gap).
#
# ---------------------------------------------------------------------------
# Por que o TETO e 1.25 e nao 1.0 (rejeicao dura)
# ---------------------------------------------------------------------------
#
# Passar de 1.0 e um resultado LEGITIMO: significa "o modelo foi melhor sob
# perturbacao / cobriu mais que o nominal", o que acontece por variacao
# amostral em datasets pequenos (medido: 1.0105 em robustness e 1.0857 em
# uncertainty num dataset de 1200 linhas). Clampar em 1.0 tornaria
# indistinguivel "nao foi afetado" de "melhorou", e recusar o valor impede a
# geracao do relatorio de um resultado perfeitamente valido.
#
# O excedente ACIMA de 1.0 e mecanicamente limitado, e e isso que justifica
# um teto fixo:
#   - para metrica em [0, 1] (AUC, accuracy, f1, R2), o melhor caso possivel
#     sob perturbacao e o valor maximo da metrica, logo
#     score <= 1 + (1 - base)/base: 1.25 para base 0.8, que e a ordem de
#     grandeza dos modelos que o toolkit valida;
#   - o fallback de uncertainty limita a sobre-cobertura por item em 1.1
#     (data/uncertainty.py), logo a media nunca passa de 1.1.
# A folga de 25% cobre com sobra a variacao amostral observada (1.01 a 1.09) e
# nao cobre nenhum ERRO DE CODIFICACAO, que e o que o teto existe para pegar:
# valor em escala percentual (85.0 em vez de 0.85), score nao normalizado,
# divisao pelo denominador errado, NaN/inf de divisao por zero mascarada.
#
# ---------------------------------------------------------------------------
# Por que o PISO e 0.0 mas NAO e rejeicao (assimetria deliberada)
# ---------------------------------------------------------------------------
#
# 0.0 tem leitura semantica propria ("nada da qualidade medida foi retido"),
# mas abaixo de 0.0 NAO ha limite mecanico: a perda relativa pode passar de
# 100%. robustness_suite.py calcula impact = (perturbado - base)/base para
# MSE/MAE/RMSE, entao uma perturbacao que mais que dobra o erro da impact > 1
# e score negativo; o mesmo vale para um gap relativo de resilience maior que
# 1. Esses valores sao MEDICOES VERDADEIRAS, nao erros de codificacao, e
# recusa-los impediria a geracao do relatorio de um resultado real - o oposto
# do que o teto consertou.
#
# Por isso a assimetria: acima do teto e erro (excedente limitado por
# mecanismo, qualquer coisa maior e bug) e levanta ValueError; abaixo do piso
# e medicao possivel (nao limitada por mecanismo) e passa com um aviso
# ScoreOutOfRangeWarning. O valor NAO e clampado em nenhum dos dois casos: o
# relatorio publica o numero medido ou falha dizendo por que.
SCORE_MIN = 0.0
SCORE_MAX = 1.25


class ScoreOutOfRangeWarning(UserWarning):
    """Score fora de [SCORE_MIN, SCORE_MAX] que ainda e uma medicao possivel.

    Emitido quando um score agregado fica abaixo de SCORE_MIN. Nao e um erro:
    perda relativa acima de 100% e um resultado real para metrica de erro.
    Serve para que o numero fora do intervalo esperado apareca em vez de
    passar calado.
    """


def validate_score(name: str, value: Any) -> float:
    """Valida um score agregado de relatorio contra o contrato de dominio.

    O contrato e o mesmo para os quatro tipos de relatorio e e assimetrico de
    proposito (veja o comentario em cima de SCORE_MIN):

    - nao numerico, bool, NaN ou inf  -> ValueError (nao e medicao);
    - acima de SCORE_MAX (1.25)       -> ValueError (excedente acima de 1.0 e
      limitado por mecanismo, logo um valor assim e erro de codificacao);
    - abaixo de SCORE_MIN (0.0)       -> aceito, com ScoreOutOfRangeWarning
      (perda relativa acima de 100% e medicao verdadeira para metrica de
      erro);
    - dentro do intervalo             -> aceito.

    O valor nunca e clampado: o score devolvido e sempre o medido.

    Args:
        name: Nome do campo, usado na mensagem
            (ex.: 'robustness_score').
        value: Valor a validar.

    Returns:
        O score como float, sem alteracao de valor.

    Raises:
        ValueError: Se o valor nao for numerico/bool, nao for finito, ou for
            maior que SCORE_MAX.

    Example:
        validate_score('robustness_score', 1.05)  -> 1.05 (aceito)
        validate_score('robustness_score', -0.40) -> -0.40 (aceito, com aviso)
        validate_score('robustness_score', 10.0)  -> ValueError
    """
    # bool e subclasse de int: float(True) daria 1.0 e um score "perfeito"
    # saido de um return acidental. np.bool_ nao e subclasse de bool, por isso
    # o dtype tambem e checado (sem importar numpy neste modulo).
    value_dtype = getattr(value, 'dtype', None)
    if isinstance(value, bool) or getattr(value_dtype, 'kind', None) == 'b':
        raise ValueError(
            f'{name} must be a number, got {value!r} '
            f'({type(value).__name__} is a boolean, not a score)'
        )

    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f'{name} must be a number, got {value!r} '
            f'({type(value).__name__})'
        )

    if math.isnan(score) or math.isinf(score):
        raise ValueError(f'{name} must be a finite number, got {score}')

    if score > SCORE_MAX:
        # A mensagem usa os limites formatados sem zeros a direita de
        # proposito: ela e o contrato visivel para quem le o traceback.
        raise ValueError(
            f'{name} must not exceed {SCORE_MAX:g}, got {score} '
            f'(a value slightly above 1.0 is legitimate - the model '
            f'performed better under perturbation - but a value this far '
            f'out indicates a percentage-scale or unnormalized value)'
        )

    if score < SCORE_MIN:
        warnings.warn(
            f'{name} is below {SCORE_MIN:g}: {score}. The model lost more '
            f'than all of its reference performance (relative loss above '
            f'100%, which happens with error metrics such as MSE/MAE). The '
            f'measured value is reported as is, not clamped.',
            ScoreOutOfRangeWarning,
            stacklevel=2,
        )

    return score


@dataclass
class ReportData(ABC):
    """Base class for all report data structures.

    All specific report data classes (RobustnessReportData, ResilienceReportData,
    etc.) should inherit from this class and implement the to_dict method.

    This provides type safety and structure to what were previously nested
    dictionaries with 8+ levels of nesting.

    Attributes:
        generated_at: Timestamp when the report data was generated
        report_type: Type of report (e.g., 'robustness', 'resilience')
        version: Schema version for backward compatibility
        metadata: Additional metadata about the report
    """
    generated_at: datetime
    report_type: str
    version: str = "1.0.0"
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if self.metadata is None:
            self.metadata = {}
        if not self.generated_at:
            self.generated_at = datetime.now()

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert report data to dictionary for serialization.

        This method should be implemented by all subclasses to provide
        a clean dictionary representation suitable for JSON serialization
        and template rendering.

        Returns:
            Dictionary representation of the report data

        Example:
            >>> data = SomeReportData(...)
            >>> dict_data = data.to_dict()
            >>> json.dumps(dict_data)  # Should work without errors
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """Validate the report data structure.

        This method should check that all required fields are present
        and contain valid values.

        Returns:
            True if data is valid

        Raises:
            ValueError: If data is invalid with descriptive error message
        """
        pass

    def to_json_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary.

        This is a convenience method that ensures datetime objects
        and other non-JSON-serializable types are properly converted.

        Returns:
            JSON-serializable dictionary
        """
        data = self.to_dict()

        # Convert datetime objects to ISO format strings
        def convert_value(v):
            if isinstance(v, datetime):
                return v.isoformat()
            elif isinstance(v, dict):
                return {k: convert_value(val) for k, val in v.items()}
            elif isinstance(v, (list, tuple)):
                return [convert_value(item) for item in v]
            return v

        return {k: convert_value(v) for k, v in data.items()}


@dataclass
class ModelResult:
    """Base class for model test results.

    Attributes:
        model_id: Unique identifier for the model
        model_name: Human-readable model name
        metrics: Dictionary of metric names to values
        test_results: List of test results
        metadata: Additional model-specific metadata
    """
    model_id: str
    model_name: str
    metrics: Dict[str, float]
    test_results: List[Dict[str, Any]] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.test_results is None:
            self.test_results = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class MetricValue:
    """Represents a single metric value with metadata.

    Attributes:
        name: Metric name
        value: Metric value
        unit: Unit of measurement (optional)
        threshold: Threshold for pass/fail (optional)
        passed: Whether the metric passed the threshold
        metadata: Additional metric metadata
    """
    name: str
    value: float
    unit: Optional[str] = None
    threshold: Optional[float] = None
    passed: Optional[bool] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize and validate."""
        if self.metadata is None:
            self.metadata = {}

        # Auto-calculate passed if threshold is provided
        if self.threshold is not None and self.passed is None:
            self.passed = self.value >= self.threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DataTransformer(ABC):
    """Base class for data transformers.

    Data transformers convert raw experiment results (typically nested
    dictionaries) into typed ReportData structures.

    This replaces the multiple transformer variants:
    - robustness.py, robustness_simple.py, static_robustness.py -> RobustnessDataTransformer
    - resilience.py, resilience_simple.py, static_resilience.py -> ResilienceDataTransformer
    etc.

    Each report type now has ONE transformer that handles all variants,
    with behavior controlled by RenderConfig.
    """

    @abstractmethod
    def transform(self, raw_data: Dict[str, Any]) -> ReportData:
        """Transform raw data into typed ReportData.

        Args:
            raw_data: Raw experiment results (typically nested dictionaries)

        Returns:
            Typed ReportData instance

        Raises:
            ValueError: If raw_data is invalid or cannot be transformed

        Example:
            >>> transformer = RobustnessDataTransformer()
            >>> raw_results = experiment.get_results()
            >>> typed_data = transformer.transform(raw_results)
            >>> assert isinstance(typed_data, RobustnessReportData)
        """
        pass

    def validate_raw_data(self, raw_data: Dict[str, Any]) -> None:
        """Validate raw input data before transformation.

        Args:
            raw_data: Raw data to validate

        Raises:
            ValueError: If data is invalid with descriptive error

        Example:
            >>> transformer.validate_raw_data(raw_results)
            # Raises ValueError if invalid
        """
        if not isinstance(raw_data, dict):
            raise ValueError(
                f"Expected dict, got {type(raw_data).__name__}"
            )

        if not raw_data:
            raise ValueError("Raw data cannot be empty")

    def _extract_metadata(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from raw data.

        Args:
            raw_data: Raw experiment results

        Returns:
            Extracted metadata dictionary
        """
        metadata = {}

        # Common metadata fields
        for key in ['experiment_id', 'timestamp', 'version', 'config']:
            if key in raw_data:
                metadata[key] = raw_data[key]

        return metadata

    def _safe_get(
        self,
        data: Dict[str, Any],
        key: str,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """Safely get value from dictionary with error handling.

        Args:
            data: Dictionary to get value from
            key: Key to retrieve
            default: Default value if key not found
            required: Whether the key is required

        Returns:
            Value from dictionary or default

        Raises:
            ValueError: If required key is not found
        """
        if required and key not in data:
            raise ValueError(f"Required key '{key}' not found in data")

        return data.get(key, default)
