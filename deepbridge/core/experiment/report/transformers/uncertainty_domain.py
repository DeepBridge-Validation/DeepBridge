"""
Domain-model based uncertainty transformer (Phase 3 Sprint 10.2).

Demonstrates migration from Dict[str, Any] to type-safe Pydantic models.

Benefits over uncertainty_simple.py:
- Type safety with Pydantic validation
- Eliminates 56+ .get() calls
- IDE autocomplete support
- Automatic data validation
- Clear data contracts

Migration Strategy:
-----------------
This transformer can be used in two ways:

1. **Type-safe mode** (recommended):
   ```python
   transformer = UncertaintyDomainTransformer()
   report_data: UncertaintyReportData = transformer.transform_to_model(results, "MyModel")
   # Access with type safety:
   score = report_data.metrics.uncertainty_score
   ```

2. **Backward-compatible mode**:
   ```python
   transformer = UncertaintyDomainTransformer()
   report_dict = transformer.transform(results, "MyModel")  # Returns Dict
   # Old code still works!
   ```
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
import logging

from ..domain import (
    UncertaintyReportData,
    UncertaintyMetrics,
    CalibrationResults,
    AlternativeModelData,
)

logger = logging.getLogger("deepbridge.reports")


class UncertaintyDomainTransformer:
    """
    Type-safe uncertainty data transformer using Pydantic domain models.

    **Phase 3 Sprint 10.2**: Replaces Dict[str, Any] with UncertaintyReportData.

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
    ) -> UncertaintyReportData:
        """
        Transform raw uncertainty results to type-safe domain model.

        **Recommended method** - provides full type safety and validation.

        Args:
            results: Raw experiment results
            model_name: Name of the model

        Returns:
            UncertaintyReportData: Validated, type-safe model

        Example:
            >>> transformer = UncertaintyDomainTransformer()
            >>> report = transformer.transform_to_model(results, "MyModel")
            >>> print(report.metrics.uncertainty_score)  # Type-safe!
            >>> print(report.is_well_calibrated)  # Property access!
        """
        logger.info("Transforming uncertainty data to domain model")

        # Extract main components
        if 'test_results' in results:
            test_results = results.get('test_results', {})
            primary_model = test_results.get('primary_model', {})
        else:
            primary_model = results.get('primary_model', {})

        initial_eval = results.get('initial_model_evaluation', {})

        # Extract core metrics
        summary = self._create_summary(primary_model)

        # Create metrics object
        metrics = UncertaintyMetrics(
            uncertainty_score=summary.get('uncertainty_score', 0.0),
            coverage=summary.get('avg_coverage', 0.0),
            mean_width=summary.get('avg_width', 0.0),
            expected_coverage=summary.get('expected_coverage', 0.9),  # Default target
            # calibration_error will be auto-computed
        )

        # Create calibration results
        calibration_data = self._transform_alphas(primary_model)
        calibration_results = None

        if calibration_data and calibration_data.get('data'):
            alpha_data = calibration_data['data']
            calibration_results = CalibrationResults(
                alpha_values=alpha_data.get('alphas', []),
                coverage_values=alpha_data.get('coverages', []),
                expected_coverages=alpha_data.get('expected_coverages', []),
                width_values=alpha_data.get('widths', [])
            )

        # Extract feature importance
        feature_data = self._transform_features(initial_eval, primary_model)
        feature_importance = feature_data.get('importance_map', {})
        features = feature_data.get('feature_names', [])

        # Create the domain model
        report = UncertaintyReportData(
            model_name=model_name,
            model_type=primary_model.get('model_type', 'Unknown'),
            timestamp=primary_model.get('timestamp', datetime.now().isoformat()),
            metrics=metrics,
            calibration_results=calibration_results,
            feature_importance=feature_importance,
            features=features,
            dataset_size=primary_model.get('dataset_size', 0),
        )

        logger.info(
            f"Transformation complete. Model: {report.model_name}, "
            f"Score: {report.metrics.uncertainty_score:.3f}, "
            f"Calibrated: {report.is_well_calibrated}"
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
            >>> transformer = UncertaintyDomainTransformer()
            >>> report_dict = transformer.transform(results, "MyModel")
            >>> # Old code still works:
            >>> score = report_dict.get('uncertainty_score', 0.0)
        """
        # Transform to domain model (gets validation benefits)
        report = self.transform_to_model(results, model_name)

        # Convert back to dict for backward compatibility
        return self._model_to_dict(report)

    def _model_to_dict(self, report: UncertaintyReportData) -> Dict[str, Any]:
        """
        Convert domain model to dictionary for backward compatibility.

        Args:
            report: UncertaintyReportData model

        Returns:
            Dictionary in format expected by existing renderers
        """
        result = {
            'model_name': report.model_name,
            'model_type': report.model_type,

            # Summary metrics (flattened for compatibility)
            'uncertainty_score': report.metrics.uncertainty_score,
            'avg_coverage': report.metrics.coverage,
            'avg_width': report.metrics.mean_width,
            'calibration_error': report.metrics.calibration_error,

            # Summary dict (nested format)
            'summary': {
                'uncertainty_score': report.metrics.uncertainty_score,
                'avg_coverage': report.metrics.coverage,
                'avg_coverage_error': report.metrics.calibration_error,
                'avg_width': report.metrics.mean_width,
                'expected_coverage': report.metrics.expected_coverage,
            },

            # Alpha results
            'alphas': {
                'total': report.calibration_results.num_alpha_levels if report.calibration_results else 0,
                'data': {
                    'alphas': report.calibration_results.alpha_values if report.calibration_results else [],
                    'coverages': report.calibration_results.coverage_values if report.calibration_results else [],
                    'expected_coverages': report.calibration_results.expected_coverages if report.calibration_results else [],
                    'widths': report.calibration_results.width_values if report.calibration_results else [],
                }
            } if report.has_calibration_results else {},

            # Features
            'features': {
                'total': len(report.features),
                'feature_names': report.features,
                'importance_map': report.feature_importance,
                'top_features': dict(report.top_features),
            },

            # Metadata
            'metadata': {
                'timestamp': report.timestamp,
                'method': 'CRQR',
                'total_alphas': report.calibration_results.num_alpha_levels if report.calibration_results else 0,
            }
        }

        return result

    # Helper methods (same as uncertainty_simple.py)

    def _create_summary(self, primary_model: Dict) -> Dict[str, Any]:
        """Create summary statistics."""
        crqr = primary_model.get('crqr', {})
        by_alpha = crqr.get('by_alpha', {})

        if by_alpha:
            coverages = []
            coverage_errors = []
            widths = []

            for alpha_key, alpha_data in by_alpha.items():
                overall = alpha_data.get('overall_result', {})
                coverage = overall.get('coverage', 0)
                expected_coverage = overall.get('expected_coverage', 0)
                mean_width = overall.get('mean_width', 0)

                coverages.append(coverage)
                coverage_errors.append(abs(coverage - expected_coverage))
                widths.append(mean_width)

            avg_coverage = float(np.mean(coverages)) if coverages else 0.0
            avg_coverage_error = float(np.mean(coverage_errors)) if coverage_errors else 0.0
            avg_width = float(np.mean(widths)) if widths else 0.0
        else:
            avg_coverage = 0.0
            avg_coverage_error = 0.0
            avg_width = 0.0

        # Compute uncertainty score (1 - normalized error)
        max_error = 1.0
        uncertainty_score = max(0.0, 1.0 - (avg_coverage_error / max_error))

        return {
            'uncertainty_score': uncertainty_score,
            'avg_coverage': avg_coverage,
            'avg_coverage_error': avg_coverage_error,
            'avg_width': avg_width,
            'expected_coverage': 0.9,  # Common target
        }

    def _transform_alphas(self, primary_model: Dict) -> Dict[str, Any]:
        """Transform alpha results for calibration plots."""
        crqr = primary_model.get('crqr', {})
        by_alpha = crqr.get('by_alpha', {})

        if not by_alpha:
            return {}

        alphas = []
        coverages = []
        expected_coverages = []
        widths = []

        for alpha_key, alpha_data in sorted(by_alpha.items()):
            try:
                alpha_value = float(alpha_key.replace('alpha_', ''))
                overall = alpha_data.get('overall_result', {})

                alphas.append(alpha_value)
                coverages.append(overall.get('coverage', 0))
                expected_coverages.append(overall.get('expected_coverage', 1.0 - alpha_value))
                widths.append(overall.get('mean_width', 0))
            except (ValueError, AttributeError) as e:
                logger.warning(f"Error processing alpha {alpha_key}: {e}")
                continue

        return {
            'total': len(alphas),
            'data': {
                'alphas': alphas,
                'coverages': coverages,
                'expected_coverages': expected_coverages,
                'widths': widths,
            }
        }

    def _transform_features(self, initial_eval: Dict, primary_model: Dict) -> Dict[str, Any]:
        """Transform feature importance data."""
        # Try to get feature importance from initial evaluation
        feature_importance = initial_eval.get('feature_importance', {})

        if not feature_importance:
            # Fallback: try to get from primary model
            feature_importance = primary_model.get('feature_importance', {})

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'total': len(sorted_features),
            'feature_names': [f[0] for f in sorted_features],
            'importance_map': dict(sorted_features),
            'top_features': dict(sorted_features[:5]) if sorted_features else {},
        }
