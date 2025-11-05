"""
Domain models for report data (Phase 3 Sprint 10).

Provides type-safe, validated data structures using Pydantic to replace
Dict[str, Any] throughout the report system.

Benefits:
- Type safety with IDE autocomplete
- Automatic validation
- Clear data contracts
- Eliminates 371+ .get() calls with defaults
- Eliminates 201+ isinstance checks

Usage:
    from deepbridge.core.experiment.report.domain import UncertaintyReportData

    report = UncertaintyReportData(
        model_name="MyModel",
        timestamp="2025-11-05",
        metrics=UncertaintyMetrics(
            uncertainty_score=0.85,
            coverage=0.90,
            mean_width=0.15
        )
    )

    # Type-safe access (no .get() needed!)
    print(report.metrics.uncertainty_score)  # IDE autocomplete!
    print(report.has_alternative_models)     # Property access!
"""

from .base import ReportBaseModel
from .uncertainty import (
    UncertaintyMetrics,
    CalibrationResults,
    AlternativeModelData,
    UncertaintyReportData
)

__all__ = [
    'ReportBaseModel',
    'UncertaintyMetrics',
    'CalibrationResults',
    'AlternativeModelData',
    'UncertaintyReportData',
]
