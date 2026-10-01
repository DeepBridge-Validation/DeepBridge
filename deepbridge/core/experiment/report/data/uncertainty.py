"""
Uncertainty report data structures and transformer.

This module provides typed data structures and transformation logic for
uncertainty quantification reports, replacing the old dictionary-based approach.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .base import ReportData, DataTransformer, validate_score


logger = logging.getLogger(__name__)


@dataclass
class CRQRResult:
    """Result of CRQR (Conformalized Quantile Regression) test.

    Attributes:
        alpha: Significance level (e.g., 0.05, 0.10, 0.15)
        coverage: Actual coverage achieved
        expected_coverage: Expected coverage (1 - alpha)
        mean_width: Mean width of prediction intervals
        median_width: Median width of prediction intervals
        coverage_ratio: Ratio of actual to expected coverage
    """
    alpha: float
    coverage: float
    expected_coverage: float
    mean_width: float
    median_width: float = 0.0
    coverage_ratio: float = 1.0

    def __post_init__(self):
        """Calculate coverage ratio after initialization."""
        if self.expected_coverage > 0:
            self.coverage_ratio = self.coverage / self.expected_coverage

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        A cobertura observada sai sob DOIS nomes, de proposito:

        - 'coverage' e o nome interno. O proprio
          UncertaintyDataTransformer._extract_crqr_results le essa chave ao
          reconstruir CRQRResult a partir de crqr.by_alpha, logo remove-la
          quebraria o round-trip transform -> to_dict -> transform.
        - 'actual_coverage' e o nome que os templates HTML de
          templates/html/uncertainty/ usam (14 referencias em full.html,
          simple.html e static.html) e tambem o nome que o caminho legado
          publica (core/experiment/results.py). Sem ele o Jinja2 resolvia
          result.actual_coverage como Undefined e o render morria com
          "'dict object' has no attribute 'actual_coverage'".

        Sao o mesmo numero: a cobertura efetivamente observada. O alias fica
        aqui, num dict unico, em vez de espalhado pelos tres templates.
        """
        return {
            'alpha': self.alpha,
            'coverage': self.coverage,
            'actual_coverage': self.coverage,
            'expected_coverage': self.expected_coverage,
            'mean_width': self.mean_width,
            'median_width': self.median_width,
            'coverage_ratio': self.coverage_ratio,
        }


@dataclass
class UncertaintyMetrics:
    """Aggregated uncertainty metrics.

    Attributes:
        uncertainty_score: Qualidade da quantificacao de incerteza. Vem do
            uncertainty_quality_score da suite quando presente (0-1); caso
            contrario e a razao media coverage/expected_coverage, que passa
            de 1.0 em caso de sobre-cobertura. Contrato: ate SCORE_MAX
            (ver data/base.py).
        avg_coverage: Average coverage across all alpha levels
        avg_width: Average interval width
        method: Uncertainty quantification method used
        best_alpha: Best performing alpha level
    """
    uncertainty_score: float
    avg_coverage: float
    avg_width: float
    method: str = "crqr"
    best_alpha: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'uncertainty_score': self.uncertainty_score,
            'avg_coverage': self.avg_coverage,
            'avg_width': self.avg_width,
            'method': self.method,
            'best_alpha': self.best_alpha,
        }


@dataclass
class UncertaintyReportData(ReportData):
    """Typed data structure for uncertainty reports.

    This replaces the deeply nested dictionaries with a clean, typed structure.

    Attributes:
        model_name: Name of the model being tested
        model_type: Type of model (e.g., 'RandomForest', 'XGBoost')
        metrics: Aggregated uncertainty metrics
        crqr_results: List of CRQR test results by alpha
        alpha_levels: List of alpha levels tested
        metric_name: Name of the evaluation metric
        test_config: Configuration used for testing
    """
    # Required fields
    model_name: str = ""
    model_type: str = ""
    metrics: Optional[UncertaintyMetrics] = None
    # Optional fields with defaults
    crqr_results: List[CRQRResult] = field(default_factory=list)
    alpha_levels: List[float] = field(default_factory=list)
    metric_name: str = "score"
    test_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and set defaults."""
        super().__post_init__()
        if not self.report_type:
            self.report_type = "uncertainty"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization and templating.

        Returns:
            Dictionary representation suitable for JSON and Jinja2
        """
        result = {
            'report_type': self.report_type,
            'version': self.version,
            'generated_at': self.generated_at.isoformat() if isinstance(self.generated_at, datetime) else self.generated_at,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'metrics': self.metrics.to_dict() if self.metrics else {},
            'crqr_results': [r.to_dict() for r in self.crqr_results],
            'alpha_levels': self.alpha_levels,
            'metric_name': self.metric_name,
            'metric': self.metric_name,  # Backward compatibility
            'method': self.metrics.method if self.metrics else 'crqr',
            'test_config': self.test_config,
            'metadata': self.metadata,
        }

        # Backward compatibility aliases (only if metrics exist)
        if self.metrics:
            result['uncertainty_score'] = self.metrics.uncertainty_score
            result['avg_coverage'] = self.metrics.avg_coverage
            result['avg_width'] = self.metrics.avg_width

        # Backward compatibility: crqr structure
        if self.crqr_results:
            result['crqr'] = {
                'by_alpha': {}
            }
            for crqr_result in self.crqr_results:
                alpha_key = str(crqr_result.alpha)
                result['crqr']['by_alpha'][alpha_key] = {
                    'overall_result': crqr_result.to_dict()
                }

        return result

    def validate(self) -> bool:
        """Validate report data.

        Returns:
            True if valid

        Raises:
            ValueError: If data is invalid
        """
        if not self.model_name:
            raise ValueError("model_name is required")

        if not self.model_type:
            raise ValueError("model_type is required")

        if self.metrics is None:
            raise ValueError("metrics is required")

        if not isinstance(self.metrics, UncertaintyMetrics):
            raise ValueError("metrics must be UncertaintyMetrics instance")

        # Validate uncertainty_score against the shared score domain contract.
        # The accepted range is [0.0, SCORE_MAX] with SCORE_MAX > 1.0 on
        # purpose: uncertainty_score is a ratio, not a normalized grade, so a
        # value slightly above 1.0 is a legitimate measurement. See the
        # rationale next to SCORE_MIN in data/base.py.
        validate_score('uncertainty_score', self.metrics.uncertainty_score)

        return True


class UncertaintyDataTransformer(DataTransformer):
    """Transformer for uncertainty report data.

    This single transformer replaces the multiple variants:
    - transformers/uncertainty.py
    - transformers/uncertainty_simple.py
    - transformers/static_uncertainty.py

    The transformation behavior is controlled by RenderConfig, not separate classes.
    """

    def transform(self, raw_data: Dict[str, Any]) -> UncertaintyReportData:
        """Transform raw uncertainty results into typed data structure.

        Args:
            raw_data: Raw experiment results from uncertainty tests

        Returns:
            UncertaintyReportData instance

        Raises:
            ValueError: If raw_data is invalid

        Example:
            >>> transformer = UncertaintyDataTransformer()
            >>> typed_data = transformer.transform(experiment.results)
            >>> assert isinstance(typed_data, UncertaintyReportData)
        """
        self.validate_raw_data(raw_data)

        logger.info("Transforming uncertainty data to typed structure...")

        # Handle to_dict() method if available (backward compatibility)
        if hasattr(raw_data, 'to_dict'):
            raw_data = raw_data.to_dict()

        # Extract primary model data if nested
        data = self._extract_primary_model_data(raw_data)

        # Extract model information
        model_name = self._extract_model_name(data)
        model_type = self._extract_model_type(data)

        # Extract CRQR results and alpha levels
        crqr_results, alpha_levels = self._extract_crqr_results(data)

        # Extract metrics
        metrics = self._extract_metrics(data, crqr_results)

        # Extract metric name
        metric_name = data.get('metric', 'score')

        # Extract test configuration
        test_config = self._extract_test_config(data)

        # Extract metadata
        metadata = self._extract_metadata(data)

        # Add timestamp if provided
        timestamp = data.get('timestamp')
        if timestamp:
            if isinstance(timestamp, str):
                try:
                    generated_at = datetime.fromisoformat(timestamp.replace(' ', 'T'))
                except ValueError:
                    generated_at = datetime.now()
            else:
                generated_at = datetime.now()
        else:
            generated_at = datetime.now()

        # Create typed data structure
        report_data = UncertaintyReportData(
            generated_at=generated_at,
            report_type="uncertainty",
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            crqr_results=crqr_results,
            alpha_levels=alpha_levels,
            metric_name=metric_name,
            test_config=test_config,
            metadata=metadata,
        )

        logger.info(f"Transformed data for model: {model_name}")
        logger.info(f"Uncertainty score: {metrics.uncertainty_score:.3f}")
        logger.info(f"Average coverage: {metrics.avg_coverage:.3f}")
        logger.info(f"Alpha levels: {alpha_levels}")

        return report_data

    def _extract_primary_model_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from primary_model if nested.

        Args:
            data: Raw data

        Returns:
            Flattened data dictionary
        """
        if 'primary_model' in data:
            logger.info("Extracting data from primary_model...")
            primary = data['primary_model']
            result = data.copy()
            for key, value in primary.items():
                if key not in result or key == 'crqr':
                    result[key] = value
            return result

        return data

    def _extract_model_name(self, data: Dict[str, Any]) -> str:
        """Extract model name from data.

        Args:
            data: Raw data

        Returns:
            Model name
        """
        return self._safe_get(data, 'model_name', default='Model', required=False)

    def _extract_model_type(self, data: Dict[str, Any]) -> str:
        """Extract model type from various possible locations.

        Args:
            data: Raw data

        Returns:
            Model type
        """
        # Try direct key
        if 'model_type' in data:
            return data['model_type']

        # Try primary_model
        if 'primary_model' in data and 'model_type' in data['primary_model']:
            return data['primary_model']['model_type']

        logger.warning("model_type not found, using 'Unknown Model'")
        return 'Unknown Model'

    def _extract_crqr_results(self, data: Dict[str, Any]) -> tuple[List[CRQRResult], List[float]]:
        """Extract CRQR results and alpha levels.

        Args:
            data: Raw data

        Returns:
            Tuple of (crqr_results, alpha_levels)
        """
        results = []
        alpha_levels = []

        # Try to find CRQR data
        crqr_data = data.get('crqr', {})
        if not crqr_data and 'results' in data and 'uncertainty' in data['results']:
            crqr_data = data['results']['uncertainty'].get('crqr', {})

        # Extract from by_alpha structure
        by_alpha = crqr_data.get('by_alpha', {})

        for alpha_str, alpha_data in by_alpha.items():
            alpha = float(alpha_str)
            alpha_levels.append(alpha)

            # Extract overall result
            overall = alpha_data.get('overall_result', {})

            coverage = float(overall.get('coverage', 0.0))
            expected_coverage = float(overall.get('expected_coverage', 1.0 - alpha))
            mean_width = float(overall.get('mean_width', 0.0))
            median_width = float(overall.get('median_width', 0.0))

            results.append(CRQRResult(
                alpha=alpha,
                coverage=coverage,
                expected_coverage=expected_coverage,
                mean_width=mean_width,
                median_width=median_width,
            ))

        # Sort by alpha
        alpha_levels.sort()
        results.sort(key=lambda r: r.alpha)

        logger.debug(f"Extracted {len(results)} CRQR results for alpha levels: {alpha_levels}")
        return results, alpha_levels

    def _extract_metrics(self, data: Dict[str, Any], crqr_results: List[CRQRResult]) -> UncertaintyMetrics:
        """Extract and compute uncertainty metrics.

        Args:
            data: Raw data
            crqr_results: Extracted CRQR results

        Returns:
            UncertaintyMetrics instance
        """
        # Extract or compute uncertainty score.
        # A suite (UncertaintySuite.run) publica o seu score agregado com a
        # chave 'uncertainty_quality_score', nao 'uncertainty_score'; sem ler
        # essa chave este transformer ignorava o numero oficial da suite e
        # caia sempre no proxy de razao de cobertura. O caminho legado
        # (transformers/static/static_uncertainty.py) ja fazia esse mapeamento.
        if 'uncertainty_score' in data:
            uncertainty_score = float(data['uncertainty_score'])
        elif 'uncertainty_quality_score' in data:
            uncertainty_score = float(data['uncertainty_quality_score'])
        elif crqr_results:
            # Fallback: razao media entre cobertura obtida e cobertura nominal.
            # Sobre-cobertura (razao > 1) e real e fica registrada, mas o
            # credito e limitado a 1.1 porque um intervalo largo demais tambem
            # e um defeito: nao se ganha nota por ser conservador sem limite.
            coverage_ratios = []
            for result in crqr_results:
                if result.coverage > result.expected_coverage:
                    ratio = min(result.coverage / result.expected_coverage, 1.1)
                else:
                    ratio = result.coverage / result.expected_coverage
                coverage_ratios.append(ratio)

            uncertainty_score = sum(coverage_ratios) / len(coverage_ratios) if coverage_ratios else 0.5
        else:
            uncertainty_score = 0.5

        # Calculate average coverage and width
        if crqr_results:
            avg_coverage = sum(r.coverage for r in crqr_results) / len(crqr_results)
            avg_width = sum(r.mean_width for r in crqr_results) / len(crqr_results)

            # Find best alpha (closest coverage ratio to 1.0)
            best_result = min(crqr_results, key=lambda r: abs(r.coverage_ratio - 1.0))
            best_alpha = best_result.alpha
        else:
            avg_coverage = data.get('avg_coverage', 0.0)
            avg_width = data.get('avg_width', 0.0)
            best_alpha = None

        method = data.get('method', 'crqr')

        return UncertaintyMetrics(
            uncertainty_score=uncertainty_score,
            avg_coverage=avg_coverage,
            avg_width=avg_width,
            method=method,
            best_alpha=best_alpha,
        )

    def _extract_test_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract test configuration.

        Args:
            data: Raw data

        Returns:
            Test configuration dictionary
        """
        config = {}

        # Extract common configuration keys
        for key in ['alpha_levels', 'method', 'n_runs', 'random_state', 'test_config']:
            if key in data:
                config[key] = data[key]

        return config
