"""
Domain models for Uncertainty reports (Phase 3 Sprint 10).

Type-safe data structures for uncertainty quantification reports.

Replaces Dict[str, Any] with validated Pydantic models:
- Eliminates .get() calls with defaults
- Provides type hints for IDE autocomplete
- Automatic validation
- Clear data contracts
"""

from typing import Any, Dict, List, Optional
from pydantic import Field, field_validator, model_validator

from .base import ReportBaseModel


class UncertaintyMetrics(ReportBaseModel):
    """
    Core uncertainty quantification metrics.

    These metrics represent the fundamental quality measures
    of a conformal prediction model.
    """

    uncertainty_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall uncertainty quality score (0-1)"
    )

    coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Empirical coverage rate (0-1)"
    )

    mean_width: float = Field(
        default=0.0,
        ge=0.0,
        description="Mean prediction interval width"
    )

    expected_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Target coverage rate (e.g., 0.9 for 90% confidence)"
    )

    calibration_error: float = Field(
        default=0.0,
        ge=0.0,
        description="Absolute difference between coverage and expected coverage"
    )

    @model_validator(mode='after')
    def compute_calibration_error_from_coverage(self) -> 'UncertaintyMetrics':
        """Compute calibration error from coverage values if not explicitly set."""
        # Only compute if calibration_error is still default (0.0)
        # and we have coverage values
        if self.calibration_error == 0.0 and (self.coverage != 0.0 or self.expected_coverage != 0.0):
            # Use object.__setattr__ to avoid triggering validation recursion
            object.__setattr__(self, 'calibration_error', abs(self.coverage - self.expected_coverage))
        return self


class CalibrationResults(ReportBaseModel):
    """
    Calibration analysis results across alpha levels.

    Tracks how well the prediction intervals are calibrated
    at different confidence levels.
    """

    alpha_values: List[float] = Field(
        default_factory=list,
        description="Significance levels (1 - confidence)"
    )

    coverage_values: List[float] = Field(
        default_factory=list,
        description="Empirical coverage at each alpha"
    )

    expected_coverages: List[float] = Field(
        default_factory=list,
        description="Expected coverage (1 - alpha) at each alpha"
    )

    width_values: List[float] = Field(
        default_factory=list,
        description="Mean interval width at each alpha"
    )

    @field_validator('coverage_values', 'expected_coverages', 'width_values')
    @classmethod
    def ensure_same_length_as_alpha(cls, v, info) -> List[float]:
        """Ensure all arrays have same length as alpha_values."""
        alpha_values = info.data.get('alpha_values', [])
        if alpha_values and v and len(v) != len(alpha_values):
            raise ValueError(
                f"Length mismatch: {info.field_name} has {len(v)} elements "
                f"but alpha_values has {len(alpha_values)} elements"
            )
        return v

    @property
    def has_calibration_data(self) -> bool:
        """Check if calibration data is available."""
        return bool(self.alpha_values)

    @property
    def num_alpha_levels(self) -> int:
        """Get number of alpha levels tested."""
        return len(self.alpha_values)


class AlternativeModelData(ReportBaseModel):
    """
    Performance data for an alternative uncertainty quantification method.

    Used for comparing different UQ approaches (e.g., Monte Carlo Dropout,
    Deep Ensembles, Conformal Prediction variants).
    """

    name: str = Field(
        description="Name of the alternative method"
    )

    uncertainty_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Overall uncertainty score for this method"
    )

    coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Coverage rate for this method"
    )

    mean_width: float = Field(
        default=0.0,
        ge=0.0,
        description="Mean interval width for this method"
    )

    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional method-specific metrics"
    )

    @property
    def is_better_than(self) -> bool:
        """
        Check if this method is better than baseline.

        Better = higher coverage with narrower intervals.
        This is a simplified heuristic.
        """
        return self.uncertainty_score > 0.5


class FeatureImportance(ReportBaseModel):
    """
    Feature importance scores for uncertainty prediction.

    Indicates which features contribute most to prediction uncertainty.
    """

    feature_name: str = Field(
        description="Name of the feature"
    )

    importance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Importance score (0-1)"
    )

    rank: int = Field(
        default=0,
        ge=0,
        description="Rank by importance (1 = most important)"
    )


class UncertaintyReportData(ReportBaseModel):
    """
    Complete uncertainty quantification report data.

    This is the top-level model that replaces Dict[str, Any] for
    uncertainty reports, eliminating hundreds of .get() calls.

    Before (Dict):
        score = report_data.get('uncertainty_score', 0.0)
        coverage = report_data.get('coverage', 0.0)
        has_alt = 'alternative_models' in report_data and \
                  isinstance(report_data['alternative_models'], dict)

    After (Domain Model):
        score = report_data.metrics.uncertainty_score  # Type-safe!
        coverage = report_data.metrics.coverage        # No .get()!
        has_alt = report_data.has_alternative_models   # Property!
    """

    # Metadata
    model_name: str = Field(
        description="Name of the model being analyzed"
    )

    model_type: str = Field(
        default="Unknown",
        description="Type/architecture of the model"
    )

    timestamp: str = Field(
        description="Report generation timestamp (ISO format)"
    )

    # Core metrics
    metrics: UncertaintyMetrics = Field(
        default_factory=UncertaintyMetrics,
        description="Core uncertainty metrics"
    )

    # Calibration analysis
    calibration_results: Optional[CalibrationResults] = Field(
        default=None,
        description="Calibration analysis results (optional)"
    )

    # Feature analysis
    feature_importance: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature importance scores (feature_name -> importance)"
    )

    features: List[str] = Field(
        default_factory=list,
        description="List of feature names"
    )

    # Alternative methods comparison
    alternative_models: Dict[str, AlternativeModelData] = Field(
        default_factory=dict,
        description="Alternative UQ methods for comparison"
    )

    # Chart data (kept as dict for flexibility)
    charts: Dict[str, str] = Field(
        default_factory=dict,
        description="Generated chart images (chart_name -> base64_image)"
    )

    # Additional metadata
    dataset_size: int = Field(
        default=0,
        ge=0,
        description="Size of test dataset"
    )

    notes: str = Field(
        default="",
        description="Additional notes or comments"
    )

    # Properties for convenience

    @property
    def has_alternative_models(self) -> bool:
        """Check if alternative models were evaluated."""
        return bool(self.alternative_models)

    @property
    def has_calibration_results(self) -> bool:
        """Check if calibration analysis was performed."""
        return self.calibration_results is not None and \
               self.calibration_results.has_calibration_data

    @property
    def has_feature_importance(self) -> bool:
        """Check if feature importance was calculated."""
        return bool(self.feature_importance)

    @property
    def num_alternative_models(self) -> int:
        """Get number of alternative models evaluated."""
        return len(self.alternative_models)

    @property
    def top_features(self) -> List[tuple[str, float]]:
        """
        Get top 5 most important features.

        Returns:
            List of (feature_name, importance_score) tuples, sorted descending
        """
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_features[:5]

    @property
    def is_well_calibrated(self) -> bool:
        """
        Check if model is well-calibrated (calibration error < 0.05).

        Returns:
            True if calibration error is less than 5%
        """
        return self.metrics.calibration_error < 0.05

    def get_summary_stats(self) -> Dict[str, float]:
        """
        Get summary statistics for the report.

        Returns:
            Dictionary with key metrics for quick overview
        """
        return {
            'uncertainty_score': self.metrics.uncertainty_score,
            'coverage': self.metrics.coverage,
            'mean_width': self.metrics.mean_width,
            'calibration_error': self.metrics.calibration_error,
            'num_features': len(self.features),
            'num_alternative_models': self.num_alternative_models,
        }

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"UncertaintyReport({self.model_name}): "
            f"score={self.metrics.uncertainty_score:.3f}, "
            f"coverage={self.metrics.coverage:.3f}, "
            f"calibration_error={self.metrics.calibration_error:.3f}"
        )
