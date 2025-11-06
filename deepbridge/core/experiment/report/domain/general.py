"""
General presentation-agnostic domain models for reports (Phase 3 Sprint 13).

Provides core classes for building reports independent of output format:
- Report: Main report container
- ReportSection: Logical section within a report
- Metric: Individual metric/measurement
- ChartSpec: Chart specification (data + config)
- ReportMetadata: Report metadata (model name, timestamp, etc.)

These classes focus on WHAT to display, not HOW to display it.
Adapters (Sprint 14) will handle conversion to specific formats (HTML, JSON, PDF).

Example:
    >>> from deepbridge.core.experiment.report.domain.general import Report, ReportSection, Metric
    >>>
    >>> # Build a report
    >>> report = Report(
    ...     metadata=ReportMetadata(
    ...         model_name="MyModel",
    ...         test_type="uncertainty"
    ...     )
    ... )
    >>>
    >>> # Add a section
    >>> section = ReportSection(
    ...     id="results",
    ...     title="Test Results",
    ...     description="Main results"
    ... )
    >>>
    >>> # Add metrics
    >>> section.add_metric(Metric(name="accuracy", value=0.95, description="Test accuracy"))
    >>> section.add_metric(Metric(name="coverage", value=0.92, description="Coverage at 90%"))
    >>>
    >>> # Add to report
    >>> report.add_section(section)
    >>>
    >>> # Export to different formats via adapters (Sprint 14)
    >>> # html = HTMLAdapter().render(report)
    >>> # json_data = JSONAdapter().render(report)
"""

from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from enum import Enum
from pydantic import Field, field_validator
from .base import ReportBaseModel


# ==================================================================================
# Enums
# ==================================================================================

class ReportType(str, Enum):
    """Report type enumeration."""
    UNCERTAINTY = "uncertainty"
    ROBUSTNESS = "robustness"
    RESILIENCE = "resilience"
    FAIRNESS = "fairness"
    DISTILLATION = "distillation"
    HYPERPARAMETER = "hyperparameter"
    CUSTOM = "custom"


class MetricType(str, Enum):
    """Metric type enumeration."""
    SCALAR = "scalar"  # Single numeric value
    PERCENTAGE = "percentage"  # 0-1 or 0-100
    DURATION = "duration"  # Time measurement
    COUNT = "count"  # Integer count
    BOOLEAN = "boolean"  # True/False
    TEXT = "text"  # Text value
    ARRAY = "array"  # List of values


class ChartType(str, Enum):
    """Chart type enumeration."""
    # From ChartRegistry
    LINE = "line_chart"
    BAR = "bar_chart"
    COVERAGE = "coverage_chart"
    WIDTH_VS_COVERAGE = "width_vs_coverage"
    CALIBRATION_ERROR = "calibration_error"
    ALTERNATIVE_METHODS = "alternative_methods_comparison"
    PERTURBATION_IMPACT = "perturbation_impact"
    FEATURE_ROBUSTNESS = "feature_robustness"
    TEST_TYPE_COMPARISON = "test_type_comparison"
    SCENARIO_DEGRADATION = "scenario_degradation"
    MODEL_COMPARISON = "model_comparison"
    INTERVAL_BOXPLOT = "interval_boxplot"
    # Static versions
    WIDTH_VS_COVERAGE_STATIC = "width_vs_coverage_static"
    PERTURBATION_IMPACT_STATIC = "perturbation_impact_static"
    BAR_IMAGE = "bar_image"


# ==================================================================================
# Metadata
# ==================================================================================

class ReportMetadata(ReportBaseModel):
    """
    Metadata for a report.

    Contains high-level information about the report:
    - What model was tested
    - When the test was run
    - What type of test
    - Who created it
    - Version info

    Example:
        >>> metadata = ReportMetadata(
        ...     model_name="ResNet50",
        ...     test_type=ReportType.UNCERTAINTY,
        ...     created_at=datetime.now()
        ... )
    """
    model_name: str = Field(description="Name of the model being tested")
    test_type: ReportType = Field(description="Type of test/report")
    created_at: datetime = Field(default_factory=datetime.now, description="Report creation timestamp")
    version: str = Field(default="1.0", description="Report format version")
    created_by: Optional[str] = Field(default=None, description="User/system that created the report")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    description: Optional[str] = Field(default=None, description="Report description")

    # Test configuration
    dataset_name: Optional[str] = Field(default=None, description="Dataset used for testing")
    dataset_size: Optional[int] = Field(default=None, description="Number of samples")
    test_duration: Optional[float] = Field(default=None, description="Test duration in seconds")

    # Model info
    model_type: Optional[str] = Field(default=None, description="Type of model (e.g., 'classification', 'regression')")
    model_architecture: Optional[str] = Field(default=None, description="Model architecture")
    model_version: Optional[str] = Field(default=None, description="Model version")

    # Additional metadata
    extra: Dict[str, Any] = Field(default_factory=dict, description="Additional custom metadata")


# ==================================================================================
# Metric
# ==================================================================================

class Metric(ReportBaseModel):
    """
    A single metric/measurement.

    Represents a named value with optional metadata:
    - name: Metric identifier (e.g., "accuracy", "coverage")
    - value: The measured value
    - type: Type of metric (scalar, percentage, etc.)
    - description: Human-readable description
    - unit: Unit of measurement
    - bounds: Expected min/max values
    - threshold: Pass/fail threshold
    - status: Pass/fail/warning status

    Example:
        >>> metric = Metric(
        ...     name="test_accuracy",
        ...     value=0.95,
        ...     type=MetricType.PERCENTAGE,
        ...     description="Accuracy on test set",
        ...     threshold=0.90,
        ...     unit="%"
        ... )
        >>> metric.is_passing  # True (0.95 >= 0.90)
    """
    name: str = Field(description="Metric name/identifier")
    value: Union[float, int, str, bool, List, None] = Field(description="Metric value")
    type: MetricType = Field(default=MetricType.SCALAR, description="Type of metric")

    # Optional metadata
    description: Optional[str] = Field(default=None, description="Human-readable description")
    unit: Optional[str] = Field(default=None, description="Unit of measurement")
    format_string: Optional[str] = Field(default=None, description="Format string for display (e.g., '.2f', '.1%')")

    # Validation/thresholds
    min_value: Optional[float] = Field(default=None, description="Minimum expected value")
    max_value: Optional[float] = Field(default=None, description="Maximum expected value")
    threshold: Optional[float] = Field(default=None, description="Pass/fail threshold")
    higher_is_better: bool = Field(default=True, description="Whether higher values are better")

    # Status
    is_primary: bool = Field(default=False, description="Whether this is a primary metric")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    @property
    def is_passing(self) -> Optional[bool]:
        """
        Check if metric passes threshold.

        Returns:
            True if passing, False if failing, None if no threshold
        """
        if self.threshold is None or not isinstance(self.value, (int, float)):
            return None

        if self.higher_is_better:
            return self.value >= self.threshold
        else:
            return self.value <= self.threshold

    @property
    def formatted_value(self) -> str:
        """
        Get formatted value string.

        Returns:
            Formatted string representation of value
        """
        if self.value is None:
            return "N/A"

        if self.format_string:
            try:
                return f"{self.value:{self.format_string}}"
            except:
                pass

        # Default formatting by type
        if self.type == MetricType.PERCENTAGE:
            if isinstance(self.value, (int, float)):
                return f"{self.value * 100:.1f}%"
        elif self.type == MetricType.COUNT:
            return str(int(self.value))
        elif isinstance(self.value, float):
            return f"{self.value:.4f}"

        return str(self.value)


# ==================================================================================
# ChartSpec
# ==================================================================================

class ChartSpec(ReportBaseModel):
    """
    Specification for a chart.

    Contains everything needed to generate a chart:
    - Chart type (from ChartRegistry)
    - Data to visualize
    - Configuration/options
    - Metadata

    This is a SPECIFICATION, not the rendered chart itself.
    Adapters will use this to generate actual charts via ChartRegistry.

    Example:
        >>> chart = ChartSpec(
        ...     id="coverage_plot",
        ...     type=ChartType.COVERAGE,
        ...     title="Coverage Analysis",
        ...     data={
        ...         'alphas': [0.1, 0.2, 0.3],
        ...         'coverage': [0.91, 0.81, 0.72],
        ...         'expected': [0.90, 0.80, 0.70]
        ...     }
        ... )
    """
    id: str = Field(description="Unique identifier for the chart")
    type: ChartType = Field(description="Type of chart (from ChartRegistry)")
    title: str = Field(description="Chart title")

    # Data
    data: Dict[str, Any] = Field(description="Chart data (format depends on chart type)")

    # Optional configuration
    description: Optional[str] = Field(default=None, description="Chart description")
    width: Optional[int] = Field(default=None, description="Chart width in pixels")
    height: Optional[int] = Field(default=None, description="Chart height in pixels")
    options: Dict[str, Any] = Field(default_factory=dict, description="Additional chart options")

    # Metadata
    is_primary: bool = Field(default=False, description="Whether this is a primary chart")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")


# ==================================================================================
# ReportSection
# ==================================================================================

class ReportSection(ReportBaseModel):
    """
    A logical section within a report.

    Sections organize content hierarchically:
    - Metrics: List of measurements
    - Charts: List of visualizations
    - Subsections: Nested sections
    - Content: Free-form text/HTML

    Example:
        >>> section = ReportSection(
        ...     id="calibration",
        ...     title="Calibration Analysis",
        ...     description="Analysis of model calibration"
        ... )
        >>>
        >>> section.add_metric(Metric(name="calibration_error", value=0.02))
        >>> section.add_chart(ChartSpec(id="calib_plot", type=ChartType.COVERAGE, ...))
    """
    id: str = Field(description="Unique section identifier")
    title: str = Field(description="Section title")
    description: Optional[str] = Field(default=None, description="Section description")

    # Content
    metrics: List[Metric] = Field(default_factory=list, description="Metrics in this section")
    charts: List[ChartSpec] = Field(default_factory=list, description="Charts in this section")
    subsections: List['ReportSection'] = Field(default_factory=list, description="Nested subsections")
    content: Optional[str] = Field(default=None, description="Free-form content (markdown/HTML)")

    # Organization
    order: int = Field(default=0, description="Display order")
    is_collapsed: bool = Field(default=False, description="Whether section is initially collapsed")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    def add_metric(self, metric: Metric) -> 'ReportSection':
        """Add a metric to this section."""
        self.metrics.append(metric)
        return self

    def add_chart(self, chart: ChartSpec) -> 'ReportSection':
        """Add a chart to this section."""
        self.charts.append(chart)
        return self

    def add_subsection(self, section: 'ReportSection') -> 'ReportSection':
        """Add a subsection."""
        self.subsections.append(section)
        return self

    @property
    def primary_metrics(self) -> List[Metric]:
        """Get primary metrics in this section."""
        return [m for m in self.metrics if m.is_primary]

    @property
    def primary_charts(self) -> List[ChartSpec]:
        """Get primary charts in this section."""
        return [c for c in self.charts if c.is_primary]


# ==================================================================================
# Report
# ==================================================================================

class Report(ReportBaseModel):
    """
    Main report container.

    The top-level class representing a complete report:
    - Metadata: Who, what, when
    - Sections: Organized content
    - Summary: Executive summary metrics

    This is presentation-agnostic - it describes WHAT to show,
    not HOW to show it. Adapters handle conversion to specific formats.

    Example:
        >>> # Create report
        >>> report = Report(
        ...     metadata=ReportMetadata(
        ...         model_name="MyModel",
        ...         test_type=ReportType.UNCERTAINTY
        ...     )
        ... )
        >>>
        >>> # Add summary metrics
        >>> report.add_summary_metric(Metric(name="overall_score", value=0.92, is_primary=True))
        >>>
        >>> # Add section
        >>> section = ReportSection(id="results", title="Results")
        >>> section.add_metric(Metric(name="accuracy", value=0.95))
        >>> report.add_section(section)
        >>>
        >>> # Export via adapter (Sprint 14)
        >>> # html = HTMLAdapter().render(report)
    """
    metadata: ReportMetadata = Field(description="Report metadata")
    sections: List[ReportSection] = Field(default_factory=list, description="Report sections")
    summary_metrics: List[Metric] = Field(default_factory=list, description="Summary/overview metrics")

    # Optional content
    title: Optional[str] = Field(default=None, description="Report title (defaults to test type)")
    subtitle: Optional[str] = Field(default=None, description="Report subtitle")
    introduction: Optional[str] = Field(default=None, description="Introduction text")
    conclusion: Optional[str] = Field(default=None, description="Conclusion text")

    # Configuration
    theme: str = Field(default="default", description="Visual theme")
    language: str = Field(default="en", description="Report language")

    def add_section(self, section: ReportSection) -> 'Report':
        """Add a section to the report."""
        self.sections.append(section)
        return self

    def add_summary_metric(self, metric: Metric) -> 'Report':
        """Add a summary metric."""
        self.summary_metrics.append(metric)
        return self

    def get_section(self, section_id: str) -> Optional[ReportSection]:
        """Get section by ID."""
        for section in self.sections:
            if section.id == section_id:
                return section
        return None

    def get_all_metrics(self) -> List[Metric]:
        """Get all metrics from all sections."""
        metrics = list(self.summary_metrics)
        for section in self.sections:
            metrics.extend(section.metrics)
            for subsection in section.subsections:
                metrics.extend(subsection.metrics)
        return metrics

    def get_all_charts(self) -> List[ChartSpec]:
        """Get all charts from all sections."""
        charts = []
        for section in self.sections:
            charts.extend(section.charts)
            for subsection in section.subsections:
                charts.extend(subsection.charts)
        return charts

    @property
    def display_title(self) -> str:
        """Get display title (uses metadata if no explicit title)."""
        if self.title:
            return self.title
        return f"{self.metadata.test_type.value.title()} Report - {self.metadata.model_name}"


# Update forward refs for recursive types
ReportSection.model_rebuild()
