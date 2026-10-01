"""
Robustness report data structures and transformer.

This module provides typed data structures and transformation logic for
robustness test reports, replacing the old dictionary-based approach.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .base import (
    ReportData,
    ModelResult,
    MetricValue,
    DataTransformer,
    validate_score,
)


logger = logging.getLogger(__name__)


@dataclass
class PerturbationResult:
    """Result of a single perturbation test.

    Attributes:
        level: Perturbation level (e.g., 0.1, 0.2)
        mean_score: Average score after perturbation
        worst_score: Worst score after perturbation
        impact: Impact on model performance
        num_samples: Number of samples tested
    """
    level: float
    mean_score: float
    worst_score: Optional[float] = None
    impact: float = 0.0
    num_samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'level': self.level,
            'mean_score': self.mean_score,
            'worst_score': self.worst_score,
            'impact': self.impact,
            'num_samples': self.num_samples,
        }


@dataclass
class FeatureImportance:
    """Feature importance information.

    Attributes:
        feature_name: Name of the feature
        importance: Importance score
        rank: Importance rank
        impact: Impact on robustness
    """
    feature_name: str
    importance: float
    rank: int = 0
    impact: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'feature_name': self.feature_name,
            'importance': self.importance,
            'rank': self.rank,
            'impact': self.impact,
        }


@dataclass
class RobustnessMetrics:
    """Aggregated robustness metrics.

    Attributes:
        robustness_score: Fracao do desempenho base retida sob perturbacao
            (1 - avg_overall_impact). 1.0 = nenhum impacto. Pode passar
            de 1.0 quando o modelo vai melhor perturbado; o contrato
            aceita ate SCORE_MAX (ver data/base.py).
        base_score: Baseline model score without perturbations
        avg_raw_impact: Average impact from raw perturbations
        avg_quantile_impact: Average impact from quantile perturbations
        avg_overall_impact: Overall average impact
    """
    robustness_score: float
    base_score: float
    avg_raw_impact: float = 0.0
    avg_quantile_impact: float = 0.0
    avg_overall_impact: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'robustness_score': self.robustness_score,
            'base_score': self.base_score,
            'avg_raw_impact': self.avg_raw_impact,
            'avg_quantile_impact': self.avg_quantile_impact,
            'avg_overall_impact': self.avg_overall_impact,
        }


@dataclass
class RobustnessReportData(ReportData):
    """Typed data structure for robustness reports.

    This replaces the deeply nested dictionaries with a clean, typed structure.

    Attributes:
        model_name: Name of the model being tested
        model_type: Type of model (e.g., 'RandomForest', 'XGBoost')
        metrics: Aggregated robustness metrics
        perturbation_results_raw: Results from raw perturbations
        perturbation_results_quantile: Results from quantile perturbations
        feature_importance: List of feature importance information
        metric_name: Name of the evaluation metric
        test_config: Configuration used for testing
    """
    # Required fields (no defaults) - must come before parent's defaults
    model_name: str = ""
    model_type: str = ""
    metrics: Optional[RobustnessMetrics] = None
    # Optional fields with defaults
    perturbation_results_raw: List[PerturbationResult] = field(default_factory=list)
    perturbation_results_quantile: List[PerturbationResult] = field(default_factory=list)
    feature_importance: List[FeatureImportance] = field(default_factory=list)
    metric_name: str = "score"
    test_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and set defaults."""
        super().__post_init__()
        if not self.report_type:
            self.report_type = "robustness"

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
            'perturbation_results_raw': [r.to_dict() for r in self.perturbation_results_raw],
            'perturbation_results_quantile': [r.to_dict() for r in self.perturbation_results_quantile],
            'feature_importance': [f.to_dict() for f in self.feature_importance],
            'metric_name': self.metric_name,
            'test_config': self.test_config,
            'metadata': self.metadata,
        }

        # Backward compatibility aliases (only if metrics exist)
        if self.metrics:
            result['robustness_score'] = self.metrics.robustness_score
            result['base_score'] = self.metrics.base_score
            result['avg_raw_impact'] = self.metrics.avg_raw_impact
            result['avg_quantile_impact'] = self.metrics.avg_quantile_impact
            result['raw_impact'] = self.metrics.avg_raw_impact
            result['quantile_impact'] = self.metrics.avg_quantile_impact

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

        if not isinstance(self.metrics, RobustnessMetrics):
            raise ValueError("metrics must be RobustnessMetrics instance")

        # Validate robustness_score against the shared score domain contract.
        # The accepted range is [0.0, SCORE_MAX] with SCORE_MAX > 1.0 on
        # purpose: robustness_score is a ratio, not a normalized grade, so a
        # value slightly above 1.0 is a legitimate measurement. See the
        # rationale next to SCORE_MIN in data/base.py.
        validate_score('robustness_score', self.metrics.robustness_score)

        return True


class RobustnessDataTransformer(DataTransformer):
    """Transformer for robustness report data.

    This single transformer replaces the multiple variants:
    - transformers/robustness.py
    - transformers/robustness_simple.py
    - transformers/static_robustness.py

    The transformation behavior is controlled by RenderConfig, not separate classes.
    """

    def transform(self, raw_data: Dict[str, Any]) -> RobustnessReportData:
        """Transform raw robustness results into typed data structure.

        Args:
            raw_data: Raw experiment results from robustness tests

        Returns:
            RobustnessReportData instance

        Raises:
            ValueError: If raw_data is invalid

        Example:
            >>> transformer = RobustnessDataTransformer()
            >>> typed_data = transformer.transform(experiment.results)
            >>> assert isinstance(typed_data, RobustnessReportData)
        """
        self.validate_raw_data(raw_data)

        logger.info("Transforming robustness data to typed structure...")

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

        # Extract perturbation results
        perturbation_raw = self._extract_perturbation_results(data, 'raw')
        perturbation_quantile = self._extract_perturbation_results(data, 'quantile')

        # Extract feature importance
        feature_importance = self._extract_feature_importance(data)

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
                    generated_at = datetime.fromisoformat(timestamp)
                except ValueError:
                    generated_at = datetime.now()
            else:
                generated_at = datetime.now()
        else:
            generated_at = datetime.now()

        # Create typed data structure
        report_data = RobustnessReportData(
            generated_at=generated_at,
            report_type="robustness",
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            perturbation_results_raw=perturbation_raw,
            perturbation_results_quantile=perturbation_quantile,
            feature_importance=feature_importance,
            metric_name=metric_name,
            test_config=test_config,
            metadata=metadata,
        )

        logger.info(f"Transformed data for model: {model_name}")
        logger.info(f"Robustness score: {metrics.robustness_score:.3f}")
        logger.info(f"Raw perturbations: {len(perturbation_raw)}")
        logger.info(f"Quantile perturbations: {len(perturbation_quantile)}")
        logger.info(f"Feature importance entries: {len(feature_importance)}")

        return report_data

    def _extract_primary_model_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from primary_model if nested.

        Args:
            data: Raw data

        Returns:
            Flattened data dictionary
        """
        if 'primary_model' not in data:
            return data

        logger.info("Extracting data from primary_model...")
        primary = data['primary_model']

        # Merge primary model data to top level
        result = data.copy()
        for key, value in primary.items():
            if key not in result or key in ('raw', 'quantile', 'feature_importance'):
                result[key] = value

        return result

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

        # Try initial_results
        if ('initial_results' in data
            and 'models' in data['initial_results']
            and 'primary_model' in data['initial_results']['models']
            and 'type' in data['initial_results']['models']['primary_model']):
            return data['initial_results']['models']['primary_model']['type']

        logger.warning("model_type not found, using 'Unknown Model'")
        return 'Unknown Model'

    def _extract_metrics(self, data: Dict[str, Any]) -> RobustnessMetrics:
        """Extract and compute robustness metrics.

        Args:
            data: Raw data

        Returns:
            RobustnessMetrics instance
        """
        base_score = float(data.get('base_score', 0.0))

        # Extract raw impact
        avg_raw_impact = 0.0
        if 'avg_raw_impact' in data:
            avg_raw_impact = float(data['avg_raw_impact'])
        elif 'raw' in data and 'overall' in data['raw']:
            avg_raw_impact = float(data['raw']['overall'].get('avg_impact', 0.0))

        # Extract quantile impact
        avg_quantile_impact = 0.0
        if 'avg_quantile_impact' in data:
            avg_quantile_impact = float(data['avg_quantile_impact'])
        elif 'quantile' in data and 'overall' in data['quantile']:
            avg_quantile_impact = float(data['quantile']['overall'].get('avg_impact', 0.0))

        # Compute overall impact (average of raw and quantile)
        avg_overall_impact = (avg_raw_impact + avg_quantile_impact) / 2.0

        # Extract or compute robustness score
        if 'robustness_score' in data:
            robustness_score = float(data['robustness_score'])
        elif 'avg_overall_impact' in data:
            robustness_score = 1.0 - float(data['avg_overall_impact'])
        else:
            robustness_score = 1.0 - avg_overall_impact

        return RobustnessMetrics(
            robustness_score=robustness_score,
            base_score=base_score,
            avg_raw_impact=avg_raw_impact,
            avg_quantile_impact=avg_quantile_impact,
            avg_overall_impact=avg_overall_impact,
        )

    def _extract_perturbation_results(
        self,
        data: Dict[str, Any],
        result_type: str
    ) -> List[PerturbationResult]:
        """Extract perturbation results for a specific type (raw or quantile).

        Args:
            data: Raw data
            result_type: Type of results ('raw' or 'quantile')

        Returns:
            List of PerturbationResult instances
        """
        if result_type not in data:
            return []

        type_data = data[result_type]
        if 'by_level' not in type_data:
            return []

        results = []
        by_level = type_data['by_level']

        # Sort levels numerically
        levels = sorted([float(level) for level in by_level.keys()])

        for level in levels:
            level_str = str(level)
            level_data = by_level.get(level_str, {})

            # Extract scores
            mean_score = None
            worst_score = None
            impact = 0.0

            # Try overall_result first
            if 'overall_result' in level_data:
                overall = level_data['overall_result']
                if 'all_features' in overall:
                    all_features = overall['all_features']
                    mean_score = all_features.get('mean_score')
                    worst_score = all_features.get('worst_score')
                    impact = all_features.get('impact', 0.0)
                else:
                    mean_score = overall.get('mean_score')
                    worst_score = overall.get('worst_score')
                    impact = overall.get('impact', 0.0)

            # Fallback to runs if needed
            if mean_score is None and 'runs' in level_data:
                if 'all_features' in level_data['runs']:
                    all_features_runs = level_data['runs']['all_features']
                    if all_features_runs:
                        first_run = all_features_runs[0]
                        mean_score = first_run.get('perturbed_score')
                        worst_score = first_run.get('worst_score')

            if mean_score is not None:
                results.append(PerturbationResult(
                    level=level,
                    mean_score=float(mean_score),
                    worst_score=float(worst_score) if worst_score is not None else None,
                    impact=float(impact),
                    num_samples=len(level_data.get('runs', {}).get('all_features', []))
                ))

        logger.debug(f"Extracted {len(results)} {result_type} perturbation results")
        return results

    def _extract_feature_importance(self, data: Dict[str, Any]) -> List[FeatureImportance]:
        """Extract feature importance information.

        Args:
            data: Raw data

        Returns:
            List of FeatureImportance instances
        """
        # Try direct key first
        feature_data = data.get('feature_importance') or data.get('model_feature_importance')

        # Try nested structures
        if not feature_data:
            if 'results' in data and 'robustness' in data['results']:
                rob_results = data['results']['robustness']
                feature_data = rob_results.get('feature_importance') or rob_results.get('model_feature_importance')

                # Try nested results
                if not feature_data and 'results' in rob_results:
                    nested = rob_results['results']
                    if 'primary_model' in nested:
                        primary = nested['primary_model']
                        feature_data = primary.get('feature_importance') or primary.get('model_feature_importance')

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

    def _extract_test_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract test configuration.

        Args:
            data: Raw data

        Returns:
            Test configuration dictionary
        """
        config = {}

        # Extract common configuration keys
        for key in ['perturbation_levels', 'n_runs', 'random_state', 'test_config']:
            if key in data:
                config[key] = data[key]

        return config
