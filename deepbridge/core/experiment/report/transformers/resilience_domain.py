"""
Domain-model based resilience transformer (Phase 3 Sprint 10.4).

Demonstrates migration from Dict[str, Any] to type-safe Pydantic models for
complex multi-test-type resilience experiments.

Benefits over resilience_simple.py:
- Type safety with Pydantic validation
- Eliminates 150+ .get() calls
- Eliminates 50+ isinstance checks
- IDE autocomplete support
- Automatic data validation
- Clear contracts for complex test structures

Migration Strategy:
-----------------
This transformer can be used in two ways:

1. **Type-safe mode** (recommended):
   ```python
   transformer = ResilienceDomainTransformer()
   report: ResilienceReportData = transformer.transform_to_model(results, "MyModel")
   # Type-safe access:
   score = report.metrics.resilience_score
   has_critical = report.metrics.has_critical_gaps
   ```

2. **Backward-compatible mode**:
   ```python
   transformer = ResilienceDomainTransformer()
   report_dict = transformer.transform(results, "MyModel")  # Returns Dict
   # Old code still works!
   ```
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

from ..domain import (
    ResilienceReportData,
    ResilienceMetrics,
    ScenarioData,
    WorstSampleTestData,
    WorstClusterTestData,
    OuterSampleTestData,
    HardSampleTestData,
)

logger = logging.getLogger("deepbridge.reports")


class ResilienceDomainTransformer:
    """
    Type-safe resilience data transformer using Pydantic domain models.

    **Phase 3 Sprint 10.4**: Replaces Dict[str, Any] with ResilienceReportData.

    Advantages:
    - Automatic validation
    - Type hints everywhere
    - IDE autocomplete
    - No .get() calls needed
    - Clear data contracts for complex multi-test structures
    """

    def transform_to_model(
        self,
        results: Dict[str, Any],
        model_name: str = "Model"
    ) -> ResilienceReportData:
        """
        Transform raw resilience results to type-safe domain model.

        **Recommended method** - provides full type safety and validation.

        Args:
            results: Raw experiment results
            model_name: Name of the model

        Returns:
            ResilienceReportData: Validated, type-safe model

        Example:
            >>> transformer = ResilienceDomainTransformer()
            >>> report = transformer.transform_to_model(results, "MyModel")
            >>> print(report.metrics.resilience_score)  # Type-safe!
            >>> print(report.available_test_types)  # Property access!
        """
        logger.info("Transforming resilience data to domain model")

        # Extract main components
        if 'test_results' in results:
            test_results = results.get('test_results', {})
            primary_model = test_results.get('primary_model', {})
        else:
            primary_model = results.get('primary_model', {})

        initial_eval = results.get('initial_model_evaluation', {})

        # Create metrics from summary
        summary = self._create_summary(primary_model)
        metrics = ResilienceMetrics(
            resilience_score=summary['resilience_score'],
            total_scenarios=summary['total_scenarios'],
            valid_scenarios=summary['valid_scenarios'],
            avg_performance_gap=summary['avg_performance_gap'],
            max_performance_gap=summary['max_performance_gap'],
            min_performance_gap=summary['min_performance_gap'],
            base_performance=summary['base_performance']
        )

        # Transform all test types
        distribution_shift = self._transform_scenarios(primary_model)
        worst_sample = self._transform_worst_sample(primary_model)
        worst_cluster = self._transform_worst_cluster(primary_model)
        outer_sample = self._transform_outer_sample(primary_model)
        hard_sample = self._transform_hard_sample(primary_model)

        # Get test scores
        test_scores = primary_model.get('test_scores', {})

        # Get feature importance
        feature_data = self._transform_features(initial_eval)

        # Get metadata
        distance_metrics = primary_model.get('distance_metrics', [])
        alphas = primary_model.get('alphas', [])

        # Create the domain model
        report = ResilienceReportData(
            model_name=model_name,
            model_type=primary_model.get('model_type', 'Unknown'),
            metrics=metrics,
            distribution_shift_scenarios=distribution_shift,
            worst_sample_tests=worst_sample,
            worst_cluster_tests=worst_cluster,
            outer_sample_tests=outer_sample,
            hard_sample_tests=hard_sample,
            test_scores=test_scores,
            feature_importance=feature_data['importance'],
            features=feature_data['feature_list'],
            distance_metrics=distance_metrics,
            alphas=alphas,
        )

        logger.info(
            f"Transformation complete. Model: {report.model_name}, "
            f"Score: {report.metrics.resilience_score:.3f}, "
            f"Test Types: {report.num_test_types}"
        )

        return report

    def transform(
        self,
        results: Dict[str, Any],
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """
        Transform to dictionary (backward-compatible mode).

        This method maintains compatibility with existing code that expects Dict.
        Internally uses domain models for validation, then converts to dict.

        Args:
            results: Raw experiment results
            model_name: Name of the model

        Returns:
            Dictionary with transformed data (backward compatible)
        """
        # Transform to domain model (gets validation benefits)
        report = self.transform_to_model(results, model_name)

        # Convert back to dict for backward compatibility
        return self._model_to_dict(report)

    def _model_to_dict(self, report: ResilienceReportData) -> Dict[str, Any]:
        """
        Convert domain model to dictionary for backward compatibility.

        Args:
            report: ResilienceReportData model

        Returns:
            Dictionary in format expected by existing renderers
        """
        # Convert scenarios
        scenarios = [
            {
                'id': s.id,
                'name': s.name,
                'alpha': s.alpha,
                'distance_metric': s.distance_metric,
                'metric': s.metric,
                'performance_gap': s.performance_gap,
                'baseline_performance': s.baseline_performance,
                'target_performance': s.target_performance,
                'is_valid': s.is_valid
            }
            for s in report.distribution_shift_scenarios
        ]

        # Convert worst sample
        worst_sample_list = [
            {
                'id': t.id,
                'alpha': t.alpha,
                'ranking_method': t.ranking_method,
                'metric': t.metric,
                'performance_gap': t.performance_gap,
                'worst_metric': t.worst_metric,
                'remaining_metric': t.remaining_metric,
                'n_worst_samples': t.n_worst_samples,
                'n_remaining_samples': t.n_remaining_samples,
                'is_valid': t.is_valid
            }
            for t in report.worst_sample_tests
        ]

        # Convert worst cluster
        worst_cluster_list = [
            {
                'id': t.id,
                'n_clusters': t.n_clusters,
                'worst_cluster_id': t.worst_cluster_id,
                'metric': t.metric,
                'performance_gap': t.performance_gap,
                'worst_cluster_metric': t.worst_cluster_metric,
                'remaining_metric': t.remaining_metric,
                'worst_cluster_size': t.worst_cluster_size,
                'remaining_size': t.remaining_size,
                'top_features': t.top_features,
                'is_valid': t.is_valid
            }
            for t in report.worst_cluster_tests
        ]

        # Convert outer sample
        outer_sample_list = [
            {
                'id': t.id,
                'alpha': t.alpha,
                'outlier_method': t.outlier_method,
                'metric': t.metric,
                'performance_gap': t.performance_gap,
                'outer_metric': t.outer_metric,
                'inner_metric': t.inner_metric,
                'n_outer_samples': t.n_outer_samples,
                'n_inner_samples': t.n_inner_samples,
                'is_valid': t.is_valid
            }
            for t in report.outer_sample_tests
        ]

        # Convert hard sample
        hard_sample_list = [
            {
                'id': t.id,
                'skipped': t.skipped,
                'disagreement_threshold': t.disagreement_threshold,
                'metric': t.metric,
                'performance_gap': t.performance_gap,
                'hard_metric': t.hard_metric,
                'easy_metric': t.easy_metric,
                'n_hard_samples': t.n_hard_samples,
                'n_easy_samples': t.n_easy_samples,
                'model_disagreements': t.model_disagreements,
                'is_valid': t.is_valid,
                'reason': t.reason
            }
            for t in report.hard_sample_tests
        ]

        # Convert features
        features_data = {
            'total': len(report.features),
            'importance': report.feature_importance,
            'top_10': [
                {
                    'name': name,
                    'importance': float(importance),
                    'importance_abs': abs(float(importance))
                }
                for name, importance in report.top_features
            ],
            'feature_list': [
                {
                    'name': name,
                    'importance': float(importance),
                    'importance_abs': abs(float(importance))
                }
                for name, importance in sorted(
                    report.feature_importance.items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
            ]
        }

        result = {
            'model_name': report.model_name,
            'model_type': report.model_type,

            # Summary metrics
            'resilience_score': report.metrics.resilience_score,
            'summary': {
                'resilience_score': report.metrics.resilience_score,
                'total_scenarios': report.metrics.total_scenarios,
                'valid_scenarios': report.metrics.valid_scenarios,
                'avg_performance_gap': report.metrics.avg_performance_gap,
                'max_performance_gap': report.metrics.max_performance_gap,
                'min_performance_gap': report.metrics.min_performance_gap,
                'base_performance': report.metrics.base_performance,
                'test_counts': {
                    'distribution_shift': len(report.distribution_shift_scenarios),
                    'worst_sample': len(report.worst_sample_tests),
                    'worst_cluster': len(report.worst_cluster_tests),
                    'outer_sample': len(report.outer_sample_tests),
                    'hard_sample': len(report.hard_sample_tests)
                }
            },

            # Test scores
            'test_scores': report.test_scores,

            # Test results
            'distribution_shift': scenarios,
            'worst_sample': {
                'all_results': worst_sample_list,
                'total_tests': len(worst_sample_list)
            },
            'worst_cluster': {
                'all_results': worst_cluster_list,
                'total_tests': len(worst_cluster_list)
            },
            'outer_sample': {
                'all_results': outer_sample_list,
                'total_tests': len(outer_sample_list)
            },
            'hard_sample': {
                'all_results': hard_sample_list,
                'total_tests': len(hard_sample_list)
            },

            # Features
            'features': features_data,

            # Metadata
            'metadata': {
                'total_scenarios': report.metrics.total_scenarios,
                'total_features': len(report.features),
                'distance_metrics': report.distance_metrics,
                'alphas': report.alphas,
                'test_types': report.available_test_types
            }
        }

        return result

    # Helper methods (same as resilience_simple.py but with type-safe returns)

    def _create_summary(self, primary_model: Dict) -> Dict[str, Any]:
        """Create summary statistics across all test types."""
        # Collect all performance gaps from all test types
        all_gaps = []

        # Distribution shift
        dist_shift = primary_model.get('distribution_shift', {}).get('all_results', [])
        for s in dist_shift:
            gap = s.get('performance_gap')
            if gap is not None and not (isinstance(gap, float) and np.isnan(gap)):
                all_gaps.append(gap)

        # Worst sample
        worst_sample = primary_model.get('worst_sample', {}).get('all_results', [])
        for s in worst_sample:
            gap = s.get('performance_gap')
            if gap is not None and not (isinstance(gap, float) and np.isnan(gap)):
                all_gaps.append(gap)

        # Worst cluster
        worst_cluster = primary_model.get('worst_cluster', {}).get('all_results', [])
        for s in worst_cluster:
            gap = s.get('performance_gap')
            if gap is not None and not (isinstance(gap, float) and np.isnan(gap)):
                all_gaps.append(gap)

        # Outer sample
        outer_sample = primary_model.get('outer_sample', {}).get('all_results', [])
        for s in outer_sample:
            gap = s.get('performance_gap')
            if gap is not None and not (isinstance(gap, float) and np.isnan(gap)):
                all_gaps.append(gap)

        # Hard sample
        hard_sample = primary_model.get('hard_sample', {}).get('all_results', [])
        for s in hard_sample:
            gap = s.get('performance_gap')
            if gap is not None and not (isinstance(gap, float) and np.isnan(gap)):
                all_gaps.append(gap)

        # Calculate statistics
        if all_gaps:
            avg_gap = np.mean(all_gaps)
            max_gap = np.max(all_gaps)
            min_gap = np.min(all_gaps)
        else:
            avg_gap = max_gap = min_gap = 0.0

        # Count scenarios
        total_scenarios = (len(dist_shift) + len(worst_sample) +
                          len(worst_cluster) + len(outer_sample) + len(hard_sample))

        # Get base performance
        base_performance = 0.0
        metrics_data = primary_model.get('metrics', {})
        if isinstance(metrics_data, dict):
            base_performance = float(metrics_data.get('accuracy', 0))

        return {
            'resilience_score': float(primary_model.get('resilience_score', 1.0)),
            'total_scenarios': total_scenarios,
            'valid_scenarios': len(all_gaps),
            'avg_performance_gap': float(avg_gap),
            'max_performance_gap': float(max_gap),
            'min_performance_gap': float(min_gap),
            'base_performance': base_performance,
        }

    def _transform_scenarios(self, primary_model: Dict) -> List[ScenarioData]:
        """Transform distribution shift scenarios."""
        scenarios = primary_model.get('distribution_shift', {}).get('all_results', [])

        scenario_models = []
        for i, scenario in enumerate(scenarios):
            # Handle NaN values
            perf_gap = scenario.get('performance_gap')
            worst_metric = scenario.get('worst_metric')
            remaining_metric = scenario.get('remaining_metric')

            if isinstance(perf_gap, float) and np.isnan(perf_gap):
                perf_gap = None
            if isinstance(worst_metric, float) and np.isnan(worst_metric):
                worst_metric = None
            if isinstance(remaining_metric, float) and np.isnan(remaining_metric):
                remaining_metric = None

            scenario_models.append(
                ScenarioData(
                    id=i + 1,
                    name=scenario.get('name', f"Scenario {i + 1}"),
                    alpha=float(scenario.get('alpha', 0)),
                    distance_metric=scenario.get('distance_metric', 'unknown'),
                    metric=scenario.get('metric', 'unknown'),
                    performance_gap=float(perf_gap) if perf_gap is not None else None,
                    baseline_performance=float(worst_metric) if worst_metric is not None else None,
                    target_performance=float(remaining_metric) if remaining_metric is not None else None,
                    is_valid=perf_gap is not None
                )
            )

        return scenario_models

    def _transform_worst_sample(self, primary_model: Dict) -> List[WorstSampleTestData]:
        """Transform worst-sample test results."""
        worst_sample_data = primary_model.get('worst_sample', {})
        all_results = worst_sample_data.get('all_results', [])

        test_models = []
        for i, result in enumerate(all_results):
            perf_gap = result.get('performance_gap')
            worst_metric = result.get('worst_metric')
            remaining_metric = result.get('remaining_metric')

            if isinstance(perf_gap, float) and np.isnan(perf_gap):
                perf_gap = None
            if isinstance(worst_metric, float) and np.isnan(worst_metric):
                worst_metric = None
            if isinstance(remaining_metric, float) and np.isnan(remaining_metric):
                remaining_metric = None

            test_models.append(
                WorstSampleTestData(
                    id=i + 1,
                    alpha=float(result.get('alpha', 0)),
                    ranking_method=result.get('ranking_method', 'unknown'),
                    metric=result.get('metric', 'unknown'),
                    performance_gap=float(perf_gap) if perf_gap is not None else None,
                    worst_metric=float(worst_metric) if worst_metric is not None else None,
                    remaining_metric=float(remaining_metric) if remaining_metric is not None else None,
                    n_worst_samples=int(result.get('n_worst_samples', 0)),
                    n_remaining_samples=int(result.get('n_remaining_samples', 0)),
                    is_valid=perf_gap is not None
                )
            )

        return test_models

    def _transform_worst_cluster(self, primary_model: Dict) -> List[WorstClusterTestData]:
        """Transform worst-cluster test results."""
        worst_cluster_data = primary_model.get('worst_cluster', {})
        all_results = worst_cluster_data.get('all_results', [])

        test_models = []
        for i, result in enumerate(all_results):
            perf_gap = result.get('performance_gap')
            worst_metric = result.get('worst_cluster_metric')
            remaining_metric = result.get('remaining_metric')

            if isinstance(perf_gap, float) and np.isnan(perf_gap):
                perf_gap = None
            if isinstance(worst_metric, float) and np.isnan(worst_metric):
                worst_metric = None
            if isinstance(remaining_metric, float) and np.isnan(remaining_metric):
                remaining_metric = None

            # Extract top features
            feature_contributions = result.get('feature_contributions', {})
            top_features = sorted(
                [{'name': k, 'contribution': float(v)} for k, v in feature_contributions.items()],
                key=lambda x: abs(x['contribution']),
                reverse=True
            )[:10]

            # Handle None values
            worst_cluster_id_raw = result.get('worst_cluster_id', -1)
            worst_cluster_id = int(worst_cluster_id_raw) if worst_cluster_id_raw is not None else -1

            worst_cluster_size_raw = result.get('worst_cluster_size', 0)
            worst_cluster_size = int(worst_cluster_size_raw) if worst_cluster_size_raw is not None else 0

            remaining_size_raw = result.get('remaining_size', 0)
            remaining_size = int(remaining_size_raw) if remaining_size_raw is not None else 0

            test_models.append(
                WorstClusterTestData(
                    id=i + 1,
                    n_clusters=int(result.get('n_clusters', 0)),
                    worst_cluster_id=worst_cluster_id,
                    metric=result.get('metric', 'unknown'),
                    performance_gap=float(perf_gap) if perf_gap is not None else None,
                    worst_cluster_metric=float(worst_metric) if worst_metric is not None else None,
                    remaining_metric=float(remaining_metric) if remaining_metric is not None else None,
                    worst_cluster_size=worst_cluster_size,
                    remaining_size=remaining_size,
                    top_features=top_features,
                    is_valid=perf_gap is not None
                )
            )

        return test_models

    def _transform_outer_sample(self, primary_model: Dict) -> List[OuterSampleTestData]:
        """Transform outer-sample test results."""
        outer_sample_data = primary_model.get('outer_sample', {})
        all_results = outer_sample_data.get('all_results', [])

        test_models = []
        for i, result in enumerate(all_results):
            perf_gap = result.get('performance_gap')
            outer_metric = result.get('outer_metric')
            inner_metric = result.get('inner_metric')

            if isinstance(perf_gap, float) and np.isnan(perf_gap):
                perf_gap = None
            if isinstance(outer_metric, float) and np.isnan(outer_metric):
                outer_metric = None
            if isinstance(inner_metric, float) and np.isnan(inner_metric):
                inner_metric = None

            test_models.append(
                OuterSampleTestData(
                    id=i + 1,
                    alpha=float(result.get('alpha', 0)),
                    outlier_method=result.get('outlier_method', 'unknown'),
                    metric=result.get('metric', 'unknown'),
                    performance_gap=float(perf_gap) if perf_gap is not None else None,
                    outer_metric=float(outer_metric) if outer_metric is not None else None,
                    inner_metric=float(inner_metric) if inner_metric is not None else None,
                    n_outer_samples=int(result.get('n_outer_samples', 0)),
                    n_inner_samples=int(result.get('n_inner_samples', 0)),
                    is_valid=perf_gap is not None
                )
            )

        return test_models

    def _transform_hard_sample(self, primary_model: Dict) -> List[HardSampleTestData]:
        """Transform hard-sample test results."""
        hard_sample_data = primary_model.get('hard_sample', {})
        all_results = hard_sample_data.get('all_results', [])

        test_models = []
        for i, result in enumerate(all_results):
            perf_gap = result.get('performance_gap')
            hard_metric = result.get('hard_metric')
            easy_metric = result.get('easy_metric')

            # Check if test was skipped
            if hard_metric is None and easy_metric is None:
                test_models.append(
                    HardSampleTestData(
                        id=i + 1,
                        skipped=True,
                        reason='No alternative models available',
                        is_valid=False
                    )
                )
                continue

            if isinstance(perf_gap, float) and np.isnan(perf_gap):
                perf_gap = None
            if isinstance(hard_metric, float) and np.isnan(hard_metric):
                hard_metric = None
            if isinstance(easy_metric, float) and np.isnan(easy_metric):
                easy_metric = None

            # Extract model disagreements
            model_disagreements = result.get('model_disagreements', {})
            disagreement_list = [
                {'model_pair': k, 'disagreement': float(v)}
                for k, v in model_disagreements.items()
            ]

            test_models.append(
                HardSampleTestData(
                    id=i + 1,
                    skipped=False,
                    disagreement_threshold=float(result.get('disagreement_threshold', 0)),
                    metric=result.get('metric', 'unknown'),
                    performance_gap=float(perf_gap) if perf_gap is not None else None,
                    hard_metric=float(hard_metric) if hard_metric is not None else None,
                    easy_metric=float(easy_metric) if easy_metric is not None else None,
                    n_hard_samples=int(result.get('n_hard_samples', 0)),
                    n_easy_samples=int(result.get('n_easy_samples', 0)),
                    model_disagreements=disagreement_list,
                    is_valid=perf_gap is not None
                )
            )

        return test_models

    def _transform_features(self, initial_eval: Dict) -> Dict[str, Any]:
        """Transform feature importance data."""
        models = initial_eval.get('models', {})
        primary_model = models.get('primary_model', {})
        feature_importance = primary_model.get('feature_importance', {})

        if not feature_importance:
            logger.warning("No feature importance data found")
            return {
                'total': 0,
                'importance': {},
                'feature_list': []
            }

        # Get feature names sorted by importance
        feature_list = sorted(
            feature_importance.keys(),
            key=lambda x: abs(feature_importance[x]),
            reverse=True
        )

        return {
            'total': len(feature_importance),
            'importance': feature_importance,
            'feature_list': feature_list
        }
