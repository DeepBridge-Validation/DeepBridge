"""
Resilience report data structures and transformer.

This module provides typed data structures and transformation logic for
resilience test reports, replacing the old dictionary-based approach.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import math
from datetime import datetime
import logging

from .base import ReportData, DataTransformer, validate_score


logger = logging.getLogger(__name__)


@dataclass
class DistributionShiftResult:
    """Result of a single distribution shift test.

    Attributes:
        distance_metric: Distance metric used (e.g., 'PSI', 'KS', 'WD1')
        alpha: Shift level/alpha value
        feature_name: Feature being tested
        baseline_score: Baseline score before shift
        shifted_score: Score after distribution shift
        performance_gap: Difference in performance
        distance_value: Measured distance/shift value
    """
    distance_metric: str
    alpha: float
    feature_name: str
    baseline_score: float
    shifted_score: float
    performance_gap: float = 0.0
    distance_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'distance_metric': self.distance_metric,
            'alpha': self.alpha,
            'feature_name': self.feature_name,
            'baseline_score': self.baseline_score,
            'shifted_score': self.shifted_score,
            'performance_gap': self.performance_gap,
            'distance_value': self.distance_value,
        }


@dataclass
class ResilienceMetrics:
    """Aggregated resilience metrics.

    Attributes:
        resilience_score: 1 - avg_performance_gap. 1.0 = nenhuma perda sob
            shift de distribuicao. Pode passar de 1.0 quando o subconjunto
            deslocado vai melhor que o resto; contrato ate SCORE_MAX
            (ver data/base.py).
        base_score: Baseline model score without shifts
        avg_distribution_shift: Average distribution shift across tests
        max_performance_gap: Maximum performance gap observed
        most_affected_feature: Feature most affected by shifts
        avg_performance_gap: Average performance gap
    """
    resilience_score: float
    base_score: float
    avg_distribution_shift: float = 0.0
    max_performance_gap: float = 0.0
    most_affected_feature: str = ""
    avg_performance_gap: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'resilience_score': self.resilience_score,
            'base_score': self.base_score,
            'avg_distribution_shift': self.avg_distribution_shift,
            'max_performance_gap': self.max_performance_gap,
            'most_affected_feature': self.most_affected_feature,
            'avg_performance_gap': self.avg_performance_gap,
        }


@dataclass
class FeatureImportance:
    """Feature importance information.

    Attributes:
        feature_name: Name of the feature
        importance: Importance score
        rank: Importance rank
    """
    feature_name: str
    importance: float
    rank: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'importance': self.importance,
            'rank': self.rank,
        }


@dataclass
class ResilienceReportData(ReportData):
    """Typed data structure for resilience reports.

    This replaces the deeply nested dictionaries with a clean, typed structure.

    Attributes:
        model_name: Name of the model being tested
        model_type: Type of model (e.g., 'RandomForest', 'XGBoost')
        metrics: Aggregated resilience metrics
        shift_results: List of distribution shift test results
        feature_importance: List of feature importance information
        distance_metrics: List of distance metrics used
        alphas: List of alpha values tested
        metric_name: Name of the evaluation metric
        test_config: Configuration used for testing
    """
    # Required fields
    model_name: str = ""
    model_type: str = ""
    metrics: Optional[ResilienceMetrics] = None
    # Optional fields with defaults
    shift_results: List[DistributionShiftResult] = field(default_factory=list)
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    distance_metrics: List[str] = field(default_factory=list)
    alphas: List[float] = field(default_factory=list)
    metric_name: str = "score"
    test_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and set defaults."""
        super().__post_init__()
        if not self.report_type:
            self.report_type = "resilience"

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
            'shift_results': [r.to_dict() for r in self.shift_results],
            'distribution_shift_results': [r.to_dict() for r in self.shift_results],  # Backward compatibility
            'feature_importance': [f.to_dict() for f in self.feature_importance],
            'distance_metrics': self.distance_metrics,
            'alphas': self.alphas,
            'metric_name': self.metric_name,
            'metric': self.metric_name,  # Backward compatibility
            'test_config': self.test_config,
            'metadata': self.metadata,
        }

        # Backward compatibility aliases (only if metrics exist)
        if self.metrics:
            result['resilience_score'] = self.metrics.resilience_score
            result['base_score'] = self.metrics.base_score
            result['avg_distribution_shift'] = self.metrics.avg_distribution_shift
            result['max_performance_gap'] = self.metrics.max_performance_gap

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

        if not isinstance(self.metrics, ResilienceMetrics):
            raise ValueError("metrics must be ResilienceMetrics instance")

        # Validate resilience_score against the shared score domain contract.
        # The accepted range is [0.0, SCORE_MAX] with SCORE_MAX > 1.0 on
        # purpose: resilience_score is a ratio, not a normalized grade, so a
        # value slightly above 1.0 is a legitimate measurement. See the
        # rationale next to SCORE_MIN in data/base.py.
        validate_score('resilience_score', self.metrics.resilience_score)

        return True


class ResilienceDataTransformer(DataTransformer):
    """Transformer for resilience report data.

    This single transformer replaces the multiple variants:
    - transformers/resilience.py
    - transformers/resilience_simple.py
    - transformers/static_resilience.py

    The transformation behavior is controlled by RenderConfig, not separate classes.
    """

    def transform(self, raw_data: Dict[str, Any]) -> ResilienceReportData:
        """Transform raw resilience results into typed data structure.

        Args:
            raw_data: Raw experiment results from resilience tests

        Returns:
            ResilienceReportData instance

        Raises:
            ValueError: If raw_data is invalid

        Example:
            >>> transformer = ResilienceDataTransformer()
            >>> typed_data = transformer.transform(experiment.results)
            >>> assert isinstance(typed_data, ResilienceReportData)
        """
        self.validate_raw_data(raw_data)

        logger.info("Transforming resilience data to typed structure...")

        # Handle to_dict() method if available (backward compatibility)
        if hasattr(raw_data, 'to_dict'):
            raw_data = raw_data.to_dict()

        # Extract primary model data if nested
        data = self._extract_primary_model_data(raw_data)

        # Extract model information
        model_name = self._extract_model_name(data)
        model_type = self._extract_model_type(data)

        # Extract metrics
        metrics = self._extract_metrics(data)

        # Extract distribution shift results
        shift_results = self._extract_shift_results(data)

        # Extract feature importance
        feature_importance = self._extract_feature_importance(data)

        # Extract distance metrics and alphas
        distance_metrics = self._extract_distance_metrics(data, shift_results)
        alphas = self._extract_alphas(data, shift_results)

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
        report_data = ResilienceReportData(
            generated_at=generated_at,
            report_type="resilience",
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            shift_results=shift_results,
            feature_importance=feature_importance,
            distance_metrics=distance_metrics,
            alphas=alphas,
            metric_name=metric_name,
            test_config=test_config,
            metadata=metadata,
        )

        logger.info(f"Transformed data for model: {model_name}")
        logger.info(f"Resilience score: {metrics.resilience_score:.3f}")
        logger.info(f"Distribution shift results: {len(shift_results)}")
        logger.info(f"Distance metrics: {distance_metrics}")
        logger.info(f"Alphas tested: {alphas}")

        return report_data

    def _extract_primary_model_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from primary_model if nested.

        Args:
            data: Raw data

        Returns:
            Flattened data dictionary
        """
        # Extract from test_results.primary_model
        if 'test_results' in data and isinstance(data['test_results'], dict):
            if 'primary_model' in data['test_results']:
                logger.info("Extracting data from test_results.primary_model...")
                primary = data['test_results']['primary_model']
                # Merge primary model data to top level
                result = data.copy()
                for key, value in primary.items():
                    if key not in result:
                        result[key] = value
                return result

        # Extract from primary_model at root level
        if 'primary_model' in data:
            logger.info("Extracting data from root primary_model...")
            primary = data['primary_model']
            result = data.copy()
            for key, value in primary.items():
                if key not in result:
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

        # Try test_results.primary_model
        if 'test_results' in data and isinstance(data['test_results'], dict):
            if 'primary_model' in data['test_results']:
                pm = data['test_results']['primary_model']
                if 'model_type' in pm:
                    return pm['model_type']

        logger.warning("model_type not found, using 'Unknown Model'")
        return 'Unknown Model'

    def _extract_metrics(self, data: Dict[str, Any]) -> ResilienceMetrics:
        """Extract and compute resilience metrics.

        Args:
            data: Raw data

        Returns:
            ResilienceMetrics instance
        """
        # Extract base score
        base_score = 0.0
        if 'base_score' in data:
            base_score = float(data['base_score'])
        elif 'metrics' in data:
            metrics = data['metrics']
            # Try common metric names
            for metric_name in ['accuracy', 'roc_auc', 'f1', 'precision']:
                if metric_name in metrics:
                    base_score = float(metrics[metric_name])
                    break

        # Extract or compute distribution shift metrics
        avg_distribution_shift = 0.0
        max_performance_gap = 0.0
        most_affected_feature = ""
        avg_performance_gap = 0.0
        avg_relative_gap = None

        # Try from distribution_shift summary
        if 'distribution_shift' in data:
            dist_shift = data['distribution_shift']

            # Calculate average distribution shift
            if 'by_distance_metric' in dist_shift:
                dist_values = []
                for dm, dm_data in dist_shift['by_distance_metric'].items():
                    for feature, val in dm_data.get('avg_feature_distances', {}).items():
                        dist_values.append(val)
                if dist_values:
                    avg_distribution_shift = sum(dist_values) / len(dist_values)

            # Find max performance gap and most affected feature
            if 'all_results' in dist_shift:
                gaps = []
                relative_gaps = []
                for result in dist_shift['all_results']:
                    gap = float(result.get('performance_gap', 0.0))
                    # O gap com sinal entra na media: um gap negativo significa
                    # que o subconjunto deslocado foi MELHOR, e tomar abs()
                    # transformaria essa melhora em perda. Para ranquear a
                    # feature mais afetada interessa a magnitude, entao ali o
                    # abs() continua.
                    gaps.append(gap)
                    relative = result.get('performance_gap_relative')
                    if relative is not None and not math.isnan(
                        float(relative)
                    ):
                        relative_gaps.append(float(relative))
                    if abs(gap) > max_performance_gap:
                        max_performance_gap = abs(gap)
                        most_affected_feature = result.get('feature_name', '')

                if gaps:
                    avg_performance_gap = sum(gaps) / len(gaps)
                if relative_gaps:
                    avg_relative_gap = sum(relative_gaps) / len(
                        relative_gaps
                    )

        # Compute resilience score (1 - avg_performance_gap).
        #
        # Sem clamp em nenhum dos dois lados. O teto em 1.0 tornava "nao foi
        # afetado" indistinguivel de "foi melhor sob shift"; o piso em 0.0
        # escondia "perdeu mais que todo o desempenho de referencia", que e
        # medicao real para metrica de erro. O contrato em data/base.py aceita
        # os dois excedentes (acima de SCORE_MAX falha, abaixo de SCORE_MIN
        # avisa).
        #
        # O gap RELATIVO tem preferencia quando a suite o publica: so ele e
        # adimensional, logo so com ele 1 - gap tem significado quando a
        # metrica e MSE/MAE (diferenca absoluta de erro ao quadrado nao e
        # comparavel a 1.0). Ver validation/wrappers/resilience_suite.py.
        if 'resilience_score' in data:
            resilience_score = float(data['resilience_score'])
        elif avg_relative_gap is not None:
            resilience_score = 1.0 - avg_relative_gap
        else:
            resilience_score = 1.0 - avg_performance_gap

        return ResilienceMetrics(
            resilience_score=resilience_score,
            base_score=base_score,
            avg_distribution_shift=avg_distribution_shift,
            max_performance_gap=max_performance_gap,
            most_affected_feature=most_affected_feature,
            avg_performance_gap=avg_performance_gap,
        )

    def _extract_shift_results(self, data: Dict[str, Any]) -> List[DistributionShiftResult]:
        """Extract distribution shift test results.

        Args:
            data: Raw data

        Returns:
            List of DistributionShiftResult instances
        """
        results = []

        # Try from distribution_shift_results
        if 'distribution_shift_results' in data:
            raw_results = data['distribution_shift_results']
        elif 'distribution_shift' in data and 'all_results' in data['distribution_shift']:
            raw_results = data['distribution_shift']['all_results']
        elif 'test_results' in data and isinstance(data['test_results'], list):
            raw_results = data['test_results']
        else:
            raw_results = []

        for result in raw_results:
            if not isinstance(result, dict):
                continue

            distance_metric = result.get('distance_metric', 'unknown')
            alpha = float(result.get('alpha', 0.0))
            feature_name = result.get('feature_name', 'unknown')
            baseline_score = float(result.get('baseline_score', 0.0))
            shifted_score = float(result.get('shifted_score', 0.0))
            performance_gap = float(result.get('performance_gap', 0.0))
            distance_value = result.get('distance_value')

            results.append(DistributionShiftResult(
                distance_metric=distance_metric,
                alpha=alpha,
                feature_name=feature_name,
                baseline_score=baseline_score,
                shifted_score=shifted_score,
                performance_gap=performance_gap,
                distance_value=float(distance_value) if distance_value is not None else None,
            ))

        logger.debug(f"Extracted {len(results)} distribution shift results")
        return results

    def _extract_feature_importance(self, data: Dict[str, Any]) -> List[FeatureImportance]:
        """Extract feature importance information.

        Args:
            data: Raw data

        Returns:
            List of FeatureImportance instances
        """
        # Try multiple possible locations
        feature_data = None

        # Direct keys
        if 'feature_importance' in data:
            feature_data = data['feature_importance']
        elif 'model_feature_importance' in data:
            feature_data = data['model_feature_importance']

        # From initial_model_evaluation
        if not feature_data and 'initial_model_evaluation' in data:
            initial_eval = data['initial_model_evaluation']
            if 'models' in initial_eval and 'primary_model' in initial_eval['models']:
                initial_primary = initial_eval['models']['primary_model']
                feature_data = initial_primary.get('feature_importance')

        # From nested results
        if not feature_data and 'results' in data and 'resilience' in data['results']:
            resilience_results = data['results']['resilience']

            if 'feature_importance' in resilience_results:
                feature_data = resilience_results['feature_importance']
            elif 'results' in resilience_results and 'primary_model' in resilience_results['results']:
                feature_data = resilience_results['results']['primary_model'].get('feature_importance')

        if not feature_data:
            logger.debug("No feature importance data found")
            return []

        # Convert to list of FeatureImportance objects
        features = []
        for rank, (feature_name, importance) in enumerate(feature_data.items(), start=1):
            features.append(FeatureImportance(
                feature_name=feature_name,
                importance=float(importance),
                rank=rank
            ))

        # Sort by importance (descending)
        features.sort(key=lambda f: f.importance, reverse=True)

        # Update ranks after sorting
        for rank, feature in enumerate(features, start=1):
            feature.rank = rank

        logger.debug(f"Extracted {len(features)} feature importance entries")
        return features

    def _extract_distance_metrics(
        self,
        data: Dict[str, Any],
        shift_results: List[DistributionShiftResult]
    ) -> List[str]:
        """Extract distance metrics used in tests.

        Args:
            data: Raw data
            shift_results: Extracted shift results

        Returns:
            List of distance metric names
        """
        # Try from data first
        if 'distance_metrics' in data:
            return data['distance_metrics']

        # Extract from shift results
        metrics = set()
        for result in shift_results:
            metrics.add(result.distance_metric)

        # Return sorted list, or defaults if empty
        return sorted(list(metrics)) if metrics else ['PSI', 'KS', 'WD1']

    def _extract_alphas(
        self,
        data: Dict[str, Any],
        shift_results: List[DistributionShiftResult]
    ) -> List[float]:
        """Extract alpha values tested.

        Args:
            data: Raw data
            shift_results: Extracted shift results

        Returns:
            List of alpha values
        """
        # Try from data first
        if 'alphas' in data:
            return sorted([float(a) for a in data['alphas']])

        # Extract from shift results
        alphas = set()
        for result in shift_results:
            alphas.add(result.alpha)

        # Return sorted list, or defaults if empty
        return sorted(list(alphas)) if alphas else [0.1, 0.2, 0.3]

    def _extract_test_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract test configuration.

        Args:
            data: Raw data

        Returns:
            Test configuration dictionary
        """
        config = {}

        # Extract common configuration keys
        for key in ['distance_metrics', 'alphas', 'n_runs', 'random_state', 'test_config']:
            if key in data:
                config[key] = data[key]

        return config
