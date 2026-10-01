"""
Fairness report data structures and transformer.

This module provides typed data structures and transformation logic for
fairness test reports, replacing the old dictionary-based approach.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from .base import ReportData, DataTransformer, validate_score


logger = logging.getLogger(__name__)


@dataclass
class FairnessMetric:
    """Individual fairness metric with its value.

    Attributes:
        name: Name of the metric (e.g., 'demographic_parity', 'equalized_odds')
        value: Metric value
        threshold: Threshold for metric compliance (optional)
        compliant: Whether metric meets threshold
    """
    name: str
    value: float
    threshold: Optional[float] = None
    compliant: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'threshold': self.threshold,
            'compliant': self.compliant,
        }


@dataclass
class GroupMetrics:
    """Metrics for a specific demographic group.

    Attributes:
        group_name: Name/identifier of the group
        group_value: The value of the protected attribute for this group
        size: Number of samples in group
        accuracy: Accuracy score for group
        precision: Precision score for group
        recall: Recall score for group
        f1: F1 score for group
    """
    group_name: str
    group_value: Any
    size: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'group_name': self.group_name,
            'group_value': str(self.group_value),
            'size': self.size,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
        }


@dataclass
class FairnessMetrics:
    """Aggregated fairness metrics.

    Attributes:
        fairness_score: Media ponderada das razoes/paridades entre grupos.
            1.0 = paridade perfeita. Normalmente fica em 0-1, mas o
            contrato aceita ate SCORE_MAX como os outros tres
            (ver data/base.py).
        worst_group: Group with worst performance
        best_group: Group with best performance
        metric_disparities: Dict of metric names to disparity values
        demographic_parity: Demographic parity metric value
        equalized_odds: Equalized odds metric value
    """
    fairness_score: float
    worst_group: str = ""
    best_group: str = ""
    metric_disparities: Dict[str, float] = field(default_factory=dict)
    demographic_parity: float = 0.0
    equalized_odds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'fairness_score': self.fairness_score,
            'worst_group': self.worst_group,
            'best_group': self.best_group,
            'metric_disparities': self.metric_disparities,
            'demographic_parity': self.demographic_parity,
            'equalized_odds': self.equalized_odds,
        }


@dataclass
class FairnessReportData(ReportData):
    """Typed data structure for fairness reports.

    This replaces the deeply nested dictionaries with a clean, typed structure.

    Attributes:
        model_name: Name of the model being tested
        model_type: Type of model (e.g., 'RandomForest', 'XGBoost')
        metrics: Aggregated fairness metrics
        group_results: Results for each demographic group
        fairness_metrics: List of individual fairness metrics
        protected_attributes: Attributes used for fairness analysis
        metric_name: Name of the primary metric
        test_config: Configuration used for testing
    """
    # Required fields (no defaults) - must come before parent's defaults
    model_name: str = ""
    model_type: str = ""
    metrics: Optional[FairnessMetrics] = None
    # Optional fields with defaults
    group_results: List[GroupMetrics] = field(default_factory=list)
    fairness_metrics: List[FairnessMetric] = field(default_factory=list)
    protected_attributes: List[str] = field(default_factory=list)
    metric_name: str = "accuracy"
    test_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Initialize and set defaults."""
        super().__post_init__()
        if not self.report_type:
            self.report_type = "fairness"

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
            'group_results': [g.to_dict() for g in self.group_results],
            'fairness_metrics': [m.to_dict() for m in self.fairness_metrics],
            'protected_attributes': self.protected_attributes,
            'metric_name': self.metric_name,
            'test_config': self.test_config,
            'metadata': self.metadata,
        }

        # Backward compatibility aliases (only if metrics exist)
        if self.metrics:
            result['fairness_score'] = self.metrics.fairness_score
            result['worst_group'] = self.metrics.worst_group
            result['best_group'] = self.metrics.best_group
            result['demographic_parity'] = self.metrics.demographic_parity
            result['equalized_odds'] = self.metrics.equalized_odds

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

        if not isinstance(self.metrics, FairnessMetrics):
            raise ValueError("metrics must be FairnessMetrics instance")

        # Validate fairness_score against the shared score domain contract.
        # The accepted range is [0.0, SCORE_MAX] with SCORE_MAX > 1.0 on
        # purpose: fairness_score is a ratio, not a normalized grade, so a
        # value slightly above 1.0 is a legitimate measurement. See the
        # rationale next to SCORE_MIN in data/base.py.
        validate_score('fairness_score', self.metrics.fairness_score)

        return True


class FairnessDataTransformer(DataTransformer):
    """Transformer for fairness report data.

    This single transformer replaces the multiple variants:
    - transformers/fairness.py
    - transformers/fairness_simple.py
    - transformers/static_fairness.py

    The transformation behavior is controlled by RenderConfig, not separate classes.
    """

    def transform(self, raw_data: Dict[str, Any]) -> FairnessReportData:
        """Transform raw fairness results into typed data structure.

        Args:
            raw_data: Raw experiment results from fairness tests

        Returns:
            FairnessReportData instance

        Raises:
            ValueError: If raw_data is invalid

        Example:
            >>> transformer = FairnessDataTransformer()
            >>> typed_data = transformer.transform(experiment.results)
            >>> assert isinstance(typed_data, FairnessReportData)
        """
        self.validate_raw_data(raw_data)

        logger.info("Transforming fairness data to typed structure...")

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

        # Extract group results
        group_results = self._extract_group_results(data)

        # Extract fairness metrics
        fairness_metrics = self._extract_fairness_metrics(data)

        # Extract protected attributes
        protected_attributes = self._extract_protected_attributes(data)

        # Extract metric name
        metric_name = data.get('metric', 'accuracy')

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
        report_data = FairnessReportData(
            generated_at=generated_at,
            report_type="fairness",
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            group_results=group_results,
            fairness_metrics=fairness_metrics,
            protected_attributes=protected_attributes,
            metric_name=metric_name,
            test_config=test_config,
            metadata=metadata,
        )

        logger.info(f"Transformed data for model: {model_name}")
        logger.info(f"Fairness score: {metrics.fairness_score:.3f}")
        logger.info(f"Groups analyzed: {len(group_results)}")
        logger.info(f"Fairness metrics: {len(fairness_metrics)}")
        logger.info(f"Protected attributes: {protected_attributes}")

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
            if key not in result or key in ('fairness_by_group', 'demographic_parity', 'equalized_odds'):
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

    def _extract_metrics(self, data: Dict[str, Any]) -> FairnessMetrics:
        """Extract and compute fairness metrics.

        Args:
            data: Raw data

        Returns:
            FairnessMetrics instance
        """
        # Extract fairness score.
        # A FairnessSuite publica o seu score agregado como
        # 'overall_fairness_score' (ver validation/wrappers/fairness_suite.py
        # e o caminho legado transformers/fairness/data_transformer.py). Sem
        # ler essa chave, o resultado real da suite era ignorado e o relatorio
        # caia no fallback 1.0, ou seja, mostrava paridade perfeita quando na
        # verdade nada tinha sido lido.
        if 'fairness_score' in data:
            fairness_score = float(data['fairness_score'])
        elif 'overall_fairness_score' in data:
            fairness_score = float(data['overall_fairness_score'])
        else:
            # Compute from disparities if available
            disparities = data.get('metric_disparities', {})
            if disparities:
                disparity_values = [abs(float(v)) for v in disparities.values()]
                avg_disparity = sum(disparity_values) / len(disparity_values) if disparity_values else 0.0
                # Piso em 0.0 (paridade nula); sem teto em 1.0, pelo
                # mesmo motivo dos outros tres tipos - ver data/base.py.
                fairness_score = max(0.0, 1.0 - avg_disparity)
            else:
                fairness_score = 1.0

        # Extract worst and best groups
        worst_group = data.get('worst_group', '')
        best_group = data.get('best_group', '')

        # Extract metric disparities
        metric_disparities = data.get('metric_disparities', {})
        if metric_disparities and not isinstance(metric_disparities, dict):
            metric_disparities = {}

        # Extract demographic parity
        demographic_parity = float(data.get('demographic_parity', 0.0))

        # Extract equalized odds
        equalized_odds = float(data.get('equalized_odds', 0.0))

        return FairnessMetrics(
            fairness_score=fairness_score,
            worst_group=worst_group,
            best_group=best_group,
            metric_disparities=metric_disparities,
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
        )

    def _extract_group_results(self, data: Dict[str, Any]) -> List[GroupMetrics]:
        """Extract results for each demographic group.

        Args:
            data: Raw data

        Returns:
            List of GroupMetrics instances
        """
        results = []

        # Try from fairness_by_group
        group_data = data.get('fairness_by_group') or data.get('group_metrics') or {}

        if not group_data:
            # Try nested structures
            if 'results' in data and 'fairness' in data['results']:
                fairness_results = data['results']['fairness']
                group_data = fairness_results.get('fairness_by_group') or fairness_results.get('group_metrics') or {}

        if not group_data:
            logger.debug("No group metrics data found")
            return []

        for group_name, group_info in group_data.items():
            if not isinstance(group_info, dict):
                continue

            # Extract metrics for this group
            accuracy = float(group_info.get('accuracy', 0.0))
            precision = float(group_info.get('precision', 0.0))
            recall = float(group_info.get('recall', 0.0))
            f1 = float(group_info.get('f1', 0.0))
            size = int(group_info.get('size', 0))

            # Extract the group value if available
            group_value = group_info.get('value') or group_name

            results.append(GroupMetrics(
                group_name=str(group_name),
                group_value=group_value,
                size=size,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
            ))

        logger.debug(f"Extracted {len(results)} group metrics")
        return results

    def _extract_fairness_metrics(self, data: Dict[str, Any]) -> List[FairnessMetric]:
        """Extract individual fairness metrics.

        Args:
            data: Raw data

        Returns:
            List of FairnessMetric instances
        """
        metrics = []

        # Extract demographic parity
        if 'demographic_parity' in data:
            dp_value = float(data['demographic_parity'])
            metrics.append(FairnessMetric(
                name='demographic_parity',
                value=dp_value,
                compliant=dp_value <= 0.1 if dp_value is not None else True,
            ))

        # Extract equalized odds
        if 'equalized_odds' in data:
            eo_value = float(data['equalized_odds'])
            metrics.append(FairnessMetric(
                name='equalized_odds',
                value=eo_value,
                compliant=eo_value <= 0.1 if eo_value is not None else True,
            ))

        # Extract disparities from metric_disparities
        if 'metric_disparities' in data and isinstance(data['metric_disparities'], dict):
            for metric_name, disparity_value in data['metric_disparities'].items():
                try:
                    value = float(disparity_value)
                    metrics.append(FairnessMetric(
                        name=f'disparity_{metric_name}',
                        value=value,
                        compliant=abs(value) <= 0.1,
                    ))
                except (ValueError, TypeError):
                    continue

        # Extract from fairness_metrics if already available
        if 'fairness_metrics' in data and isinstance(data['fairness_metrics'], list):
            for metric in data['fairness_metrics']:
                if isinstance(metric, dict):
                    name = metric.get('name', 'unknown')
                    value = float(metric.get('value', 0.0))
                    threshold = metric.get('threshold')
                    compliant = metric.get('compliant', True)

                    # Skip if we already added this metric
                    if not any(m.name == name for m in metrics):
                        metrics.append(FairnessMetric(
                            name=name,
                            value=value,
                            threshold=float(threshold) if threshold is not None else None,
                            compliant=compliant,
                        ))

        logger.debug(f"Extracted {len(metrics)} fairness metrics")
        return metrics

    def _extract_protected_attributes(self, data: Dict[str, Any]) -> List[str]:
        """Extract protected attributes used in analysis.

        Args:
            data: Raw data

        Returns:
            List of protected attribute names
        """
        # Try direct key
        if 'protected_attributes' in data:
            attrs = data['protected_attributes']
            if isinstance(attrs, list):
                return [str(a) for a in attrs]
            elif isinstance(attrs, dict):
                return list(attrs.keys())
            else:
                return [str(attrs)]

        # Try from fairness_by_group keys (might indicate protected attribute values)
        if 'fairness_by_group' in data and isinstance(data['fairness_by_group'], dict):
            return list(data['fairness_by_group'].keys())

        # Try from test_config
        if 'test_config' in data and isinstance(data['test_config'], dict):
            config = data['test_config']
            if 'protected_attributes' in config:
                attrs = config['protected_attributes']
                if isinstance(attrs, list):
                    return [str(a) for a in attrs]

        logger.debug("No protected attributes found")
        return []

    def _extract_test_config(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract test configuration.

        Args:
            data: Raw data

        Returns:
            Test configuration dictionary
        """
        config = {}

        # Extract common configuration keys
        for key in ['protected_attributes', 'fairness_metrics', 'thresholds', 'test_config']:
            if key in data:
                config[key] = data[key]

        return config
