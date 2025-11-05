"""
Domain-model based robustness transformer (Phase 3 Sprint 10.3).

Demonstrates migration from Dict[str, Any] to type-safe Pydantic models.

Benefits over robustness_simple.py:
- Type safety with Pydantic validation
- Eliminates 30+ .get() calls
- IDE autocomplete support
- Automatic data validation
- Clear data contracts

Migration Strategy:
-----------------
This transformer can be used in two ways:

1. **Type-safe mode** (recommended):
   ```python
   transformer = RobustnessDomainTransformer()
   report_data: RobustnessReportData = transformer.transform_to_model(results, "MyModel")
   # Access with type safety:
   score = report_data.metrics.robustness_score
   ```

2. **Backward-compatible mode**:
   ```python
   transformer = RobustnessDomainTransformer()
   report_dict = transformer.transform(results, "MyModel")  # Returns Dict
   # Old code still works!
   ```
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
import logging

from ..domain import (
    RobustnessReportData,
    RobustnessMetrics,
    PerturbationLevelData,
    FeatureRobustnessData,
)

logger = logging.getLogger("deepbridge.reports")


class RobustnessDomainTransformer:
    """
    Type-safe robustness data transformer using Pydantic domain models.

    **Phase 3 Sprint 10.3**: Replaces Dict[str, Any] with RobustnessReportData.

    Advantages:
    - Automatic validation
    - Type hints everywhere
    - IDE autocomplete
    - No .get() calls needed
    - Clear data contracts
    """

    def transform_to_model(
        self,
        results: Dict[str, Any],
        model_name: str = "Model"
    ) -> RobustnessReportData:
        """
        Transform raw robustness results to type-safe domain model.

        **Recommended method** - provides full type safety and validation.

        Args:
            results: Raw experiment results
            model_name: Name of the model

        Returns:
            RobustnessReportData: Validated, type-safe model

        Example:
            >>> transformer = RobustnessDomainTransformer()
            >>> report = transformer.transform_to_model(results, "MyModel")
            >>> print(report.metrics.robustness_score)  # Type-safe!
            >>> print(report.metrics.is_robust)  # Property access!
        """
        logger.info("Transforming robustness data to domain model")

        # Extract main components
        if 'test_results' in results:
            test_results = results.get('test_results', {})
            primary_model = test_results.get('primary_model', {})
        else:
            primary_model = results.get('primary_model', {})

        initial_eval = results.get('initial_model_evaluation', {})

        # Create metrics object
        base_score = primary_model.get('base_score', 0.0)
        robustness_score = primary_model.get('robustness_score', 0.0)
        avg_raw_impact = primary_model.get('avg_raw_impact', 0.0)
        avg_quantile_impact = primary_model.get('avg_quantile_impact', 0.0)
        avg_overall_impact = (avg_raw_impact + avg_quantile_impact) / 2 if avg_raw_impact or avg_quantile_impact else 0.0

        metrics = RobustnessMetrics(
            base_score=base_score,
            robustness_score=robustness_score,
            avg_raw_impact=avg_raw_impact,
            avg_quantile_impact=avg_quantile_impact,
            avg_overall_impact=avg_overall_impact,
            metric=primary_model.get('metric', 'AUC')
        )

        # Transform perturbation levels
        perturbation_levels = self._transform_levels(primary_model)

        # Transform features
        features = self._transform_features(initial_eval, primary_model)

        # Create the domain model
        report = RobustnessReportData(
            model_name=model_name,
            model_type=primary_model.get('model_type', 'Unknown'),
            metrics=metrics,
            perturbation_levels=perturbation_levels,
            features=features,
            n_iterations=primary_model.get('n_iterations', 10),
        )

        logger.info(
            f"Transformation complete. Model: {report.model_name}, "
            f"Score: {report.metrics.robustness_score:.3f}, "
            f"Robust: {report.metrics.is_robust}"
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

        Example:
            >>> transformer = RobustnessDomainTransformer()
            >>> report_dict = transformer.transform(results, "MyModel")
            >>> # Old code still works:
            >>> score = report_dict.get('summary', {}).get('robustness_score', 0.0)
        """
        # Transform to domain model (gets validation benefits)
        report = self.transform_to_model(results, model_name)

        # Convert back to dict for backward compatibility
        return self._model_to_dict(report)

    def _model_to_dict(self, report: RobustnessReportData) -> Dict[str, Any]:
        """
        Convert domain model to dictionary for backward compatibility.

        Args:
            report: RobustnessReportData model

        Returns:
            Dictionary in format expected by existing renderers
        """
        # Convert perturbation levels
        levels_list = [
            {
                'level': level.level,
                'level_display': level.level_display or f"{level.level:.1f}",
                'mean_score': level.mean_score,
                'std_score': level.std_score,
                'impact': level.impact,
                'worst_score': level.worst_score
            }
            for level in report.perturbation_levels
        ]

        # Convert features
        features_list = [
            {
                'name': feature.name,
                'importance': feature.importance,
                'robustness_impact': feature.robustness_impact
            }
            for feature in report.features
        ]

        result = {
            'model_name': report.model_name,
            'model_type': report.model_type,

            # Summary metrics (flattened)
            'base_score': report.metrics.base_score,
            'robustness_score': report.metrics.robustness_score,
            'avg_raw_impact': report.metrics.avg_raw_impact,
            'avg_quantile_impact': report.metrics.avg_quantile_impact,
            'avg_overall_impact': report.metrics.avg_overall_impact,

            # Summary dict (nested format)
            'summary': {
                'base_score': report.metrics.base_score,
                'robustness_score': report.metrics.robustness_score,
                'avg_raw_impact': report.metrics.avg_raw_impact,
                'avg_quantile_impact': report.metrics.avg_quantile_impact,
                'avg_overall_impact': report.metrics.avg_overall_impact,
                'metric': report.metrics.metric
            },

            # Levels
            'levels': levels_list,

            # Features
            'features': features_list,

            # Metadata
            'metadata': {
                'total_levels': report.num_perturbation_levels,
                'total_features': report.num_features,
                'n_iterations': report.n_iterations,
                'metric': report.metrics.metric
            }
        }

        return result

    def _transform_levels(self, primary_model: Dict) -> List[PerturbationLevelData]:
        """Transform perturbation levels data to domain models."""
        levels_data = []

        # Get raw perturbation data
        raw_data = primary_model.get('raw', {}).get('by_level', {})

        for level_str, level_data in sorted(raw_data.items(), key=lambda x: float(x[0])):
            level_float = float(level_str)
            overall_result = level_data.get('overall_result', {}).get('all_features', {})

            levels_data.append(
                PerturbationLevelData(
                    level=level_float,
                    level_display=f"{level_float:.1f}",
                    mean_score=float(overall_result.get('mean_score', 0.0)),
                    std_score=float(overall_result.get('std_score', 0.0)),
                    impact=float(overall_result.get('impact', 0.0)),
                    worst_score=float(overall_result.get('worst_score', 0.0))
                )
            )

        return levels_data

    def _transform_features(
        self,
        initial_eval: Dict,
        primary_model: Dict
    ) -> List[FeatureRobustnessData]:
        """Transform feature importance data to domain models."""
        features_data = []

        # Get feature importance from initial evaluation
        models_data = initial_eval.get('models', {})
        primary_model_data = models_data.get('primary_model', {})
        feature_importance = primary_model_data.get('feature_importance', {})

        # Get robustness-specific feature importance (from perturbation impact)
        robustness_importance = primary_model.get('feature_importance', {})

        for feature_name, importance in sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            features_data.append(
                FeatureRobustnessData(
                    name=feature_name,
                    importance=float(importance),
                    robustness_impact=float(robustness_importance.get(feature_name, 0.0))
                )
            )

        return features_data
