"""
Domain models for report data (Phase 3 Sprint 10-13).

Provides type-safe, validated data structures using Pydantic to replace
Dict[str, Any] throughout the report system.

Sprint 10: Test-specific models (Uncertainty, Robustness, Resilience)
Sprint 13: General presentation-agnostic models (Report, Section, Metric)

Benefits:
- Type safety with IDE autocomplete
- Automatic validation
- Clear data contracts
- Eliminates 371+ .get() calls with defaults
- Eliminates 201+ isinstance checks
- Presentation-agnostic (HTML, JSON, PDF ready)

Usage (Test-Specific):
    from deepbridge.core.experiment.report.domain import UncertaintyReportData

    report = UncertaintyReportData(
        model_name="MyModel",
        metrics=UncertaintyMetrics(uncertainty_score=0.85)
    )

Usage (General):
    from deepbridge.core.experiment.report.domain import Report, ReportSection, Metric

    report = Report(
        metadata=ReportMetadata(model_name="MyModel", test_type=ReportType.UNCERTAINTY)
    )
    section = ReportSection(id="results", title="Results")
    section.add_metric(Metric(name="accuracy", value=0.95))
    report.add_section(section)
"""

from .base import ReportBaseModel

# Sprint 10: Test-specific models
from .uncertainty import (
    UncertaintyMetrics,
    CalibrationResults,
    AlternativeModelData,
    UncertaintyReportData
)
from .robustness import (
    RobustnessMetrics,
    PerturbationLevelData,
    FeatureRobustnessData,
    RobustnessReportData
)
from .resilience import (
    ResilienceMetrics,
    ScenarioData,
    WorstSampleTestData,
    WorstClusterTestData,
    OuterSampleTestData,
    HardSampleTestData,
    TestTypeSummary,
    ResilienceReportData
)

# Sprint 13: General presentation-agnostic models
from .general import (
    ReportType,
    MetricType,
    ChartType,
    ReportMetadata,
    Metric,
    ChartSpec,
    ReportSection,
    Report
)

__all__ = [
    # Base
    'ReportBaseModel',

    # Sprint 10: Test-specific models
    'UncertaintyMetrics',
    'CalibrationResults',
    'AlternativeModelData',
    'UncertaintyReportData',
    'RobustnessMetrics',
    'PerturbationLevelData',
    'FeatureRobustnessData',
    'RobustnessReportData',
    'ResilienceMetrics',
    'ScenarioData',
    'WorstSampleTestData',
    'WorstClusterTestData',
    'OuterSampleTestData',
    'HardSampleTestData',
    'TestTypeSummary',
    'ResilienceReportData',

    # Sprint 13: General models
    'ReportType',
    'MetricType',
    'ChartType',
    'ReportMetadata',
    'Metric',
    'ChartSpec',
    'ReportSection',
    'Report',
]
