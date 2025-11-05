"""
Domain models for Resilience reports (Phase 3 Sprint 10.4).

Type-safe Pydantic models for resilience experiment results.

Benefits:
- Eliminates .get() calls
- Type safety with validation
- IDE autocomplete support
- Clear data contracts for complex multi-test resilience data
"""

from typing import Dict, List, Optional, Any
from pydantic import Field
from .base import ReportBaseModel


class ResilienceMetrics(ReportBaseModel):
    """Core resilience metrics across all test types."""

    resilience_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Overall resilience quality score (0-1)"
    )
    total_scenarios: int = Field(
        default=0,
        ge=0,
        description="Total number of scenarios tested across all types"
    )
    valid_scenarios: int = Field(
        default=0,
        ge=0,
        description="Number of valid scenarios (non-NaN results)"
    )
    avg_performance_gap: float = Field(
        default=0.0,
        ge=0.0,
        description="Average performance gap across all scenarios"
    )
    max_performance_gap: float = Field(
        default=0.0,
        ge=0.0,
        description="Maximum performance gap observed"
    )
    min_performance_gap: float = Field(
        default=0.0,
        ge=0.0,
        description="Minimum performance gap observed"
    )
    base_performance: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Baseline model performance"
    )

    @property
    def is_resilient(self) -> bool:
        """Check if model is considered resilient (score > 0.7)."""
        return self.resilience_score > 0.7

    @property
    def has_critical_gaps(self) -> bool:
        """Check if there are critical performance gaps (> 0.3)."""
        return self.max_performance_gap > 0.3


class ScenarioData(ReportBaseModel):
    """Data for a single distribution shift scenario."""

    id: int = Field(description="Scenario ID")
    name: str = Field(description="Scenario name")
    alpha: float = Field(default=0.0, ge=0.0, le=1.0, description="Alpha value")
    distance_metric: str = Field(default="unknown", description="Distance metric used")
    metric: str = Field(default="unknown", description="Performance metric")
    performance_gap: Optional[float] = Field(
        default=None,
        description="Performance gap (None if invalid)"
    )
    baseline_performance: Optional[float] = Field(
        default=None,
        description="Baseline performance on worst samples"
    )
    target_performance: Optional[float] = Field(
        default=None,
        description="Performance on remaining samples"
    )
    is_valid: bool = Field(
        default=False,
        description="Whether this scenario has valid results"
    )

    @property
    def has_significant_gap(self) -> bool:
        """Check if this scenario has significant gap (> 0.2)."""
        return self.performance_gap is not None and self.performance_gap > 0.2


class WorstSampleTestData(ReportBaseModel):
    """Data for a single worst-sample test."""

    id: int = Field(description="Test ID")
    alpha: float = Field(default=0.0, ge=0.0, le=1.0, description="Alpha value")
    ranking_method: str = Field(default="unknown", description="Ranking method used")
    metric: str = Field(default="unknown", description="Performance metric")
    performance_gap: Optional[float] = Field(default=None)
    worst_metric: Optional[float] = Field(default=None)
    remaining_metric: Optional[float] = Field(default=None)
    n_worst_samples: int = Field(default=0, ge=0)
    n_remaining_samples: int = Field(default=0, ge=0)
    is_valid: bool = Field(default=False)


class WorstClusterTestData(ReportBaseModel):
    """Data for a single worst-cluster test."""

    id: int = Field(description="Test ID")
    n_clusters: int = Field(default=0, ge=0)
    worst_cluster_id: int = Field(default=-1)
    metric: str = Field(default="unknown", description="Performance metric")
    performance_gap: Optional[float] = Field(default=None)
    worst_cluster_metric: Optional[float] = Field(default=None)
    remaining_metric: Optional[float] = Field(default=None)
    worst_cluster_size: int = Field(default=0, ge=0)
    remaining_size: int = Field(default=0, ge=0)
    top_features: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Top contributing features"
    )
    is_valid: bool = Field(default=False)


class OuterSampleTestData(ReportBaseModel):
    """Data for a single outer-sample test."""

    id: int = Field(description="Test ID")
    alpha: float = Field(default=0.0, ge=0.0, le=1.0)
    outlier_method: str = Field(default="unknown", description="Outlier detection method")
    metric: str = Field(default="unknown", description="Performance metric")
    performance_gap: Optional[float] = Field(default=None)
    outer_metric: Optional[float] = Field(default=None)
    inner_metric: Optional[float] = Field(default=None)
    n_outer_samples: int = Field(default=0, ge=0)
    n_inner_samples: int = Field(default=0, ge=0)
    is_valid: bool = Field(default=False)


class HardSampleTestData(ReportBaseModel):
    """Data for a single hard-sample test."""

    id: int = Field(description="Test ID")
    skipped: bool = Field(
        default=False,
        description="Whether test was skipped"
    )
    disagreement_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    metric: str = Field(default="unknown")
    performance_gap: Optional[float] = Field(default=None)
    hard_metric: Optional[float] = Field(default=None)
    easy_metric: Optional[float] = Field(default=None)
    n_hard_samples: int = Field(default=0, ge=0)
    n_easy_samples: int = Field(default=0, ge=0)
    model_disagreements: List[Dict[str, Any]] = Field(default_factory=list)
    is_valid: bool = Field(default=False)
    reason: str = Field(
        default="",
        description="Reason if test was skipped"
    )


class TestTypeSummary(ReportBaseModel):
    """Results summary for a specific test type."""

    test_type: str = Field(description="Name of test type")
    total_tests: int = Field(default=0, ge=0)
    valid_tests: int = Field(default=0, ge=0)
    avg_performance_gap: float = Field(default=0.0, ge=0.0)
    has_results: bool = Field(default=False)


class ResilienceReportData(ReportBaseModel):
    """Complete resilience experiment report data."""

    model_name: str = Field(description="Name of the model being tested")
    model_type: str = Field(
        default="Unknown",
        description="Type/architecture of the model"
    )
    metrics: ResilienceMetrics = Field(
        default_factory=ResilienceMetrics,
        description="Core resilience metrics"
    )

    # Test type results
    distribution_shift_scenarios: List[ScenarioData] = Field(
        default_factory=list,
        description="Distribution shift test scenarios"
    )
    worst_sample_tests: List[WorstSampleTestData] = Field(
        default_factory=list,
        description="Worst-sample test results"
    )
    worst_cluster_tests: List[WorstClusterTestData] = Field(
        default_factory=list,
        description="Worst-cluster test results"
    )
    outer_sample_tests: List[OuterSampleTestData] = Field(
        default_factory=list,
        description="Outer-sample test results"
    )
    hard_sample_tests: List[HardSampleTestData] = Field(
        default_factory=list,
        description="Hard-sample test results"
    )

    # Test type scores
    test_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Resilience scores by test type"
    )

    # Feature importance
    feature_importance: Dict[str, float] = Field(
        default_factory=dict,
        description="Feature importance scores"
    )
    features: List[str] = Field(
        default_factory=list,
        description="List of feature names"
    )

    # Metadata
    distance_metrics: List[str] = Field(
        default_factory=list,
        description="Distance metrics used"
    )
    alphas: List[float] = Field(
        default_factory=list,
        description="Alpha values tested"
    )

    # Charts
    charts: Dict[str, Any] = Field(
        default_factory=dict,
        description="Generated chart data (Plotly data structures)"
    )
    notes: str = Field(default="", description="Additional notes")

    @property
    def has_distribution_shift(self) -> bool:
        """Check if distribution shift tests were run."""
        return len(self.distribution_shift_scenarios) > 0

    @property
    def has_worst_sample(self) -> bool:
        """Check if worst-sample tests were run."""
        return len(self.worst_sample_tests) > 0

    @property
    def has_worst_cluster(self) -> bool:
        """Check if worst-cluster tests were run."""
        return len(self.worst_cluster_tests) > 0

    @property
    def has_outer_sample(self) -> bool:
        """Check if outer-sample tests were run."""
        return len(self.outer_sample_tests) > 0

    @property
    def has_hard_sample(self) -> bool:
        """Check if hard-sample tests were run."""
        return len(self.hard_sample_tests) > 0

    @property
    def available_test_types(self) -> List[str]:
        """Get list of available test types."""
        available = []
        if self.has_distribution_shift:
            available.append('distribution_shift')
        if self.has_worst_sample:
            available.append('worst_sample')
        if self.has_worst_cluster:
            available.append('worst_cluster')
        if self.has_outer_sample:
            available.append('outer_sample')
        if self.has_hard_sample:
            available.append('hard_sample')
        return available

    @property
    def num_test_types(self) -> int:
        """Get number of test types run."""
        return len(self.available_test_types)

    @property
    def has_feature_importance(self) -> bool:
        """Check if feature importance data is available."""
        return len(self.feature_importance) > 0

    @property
    def top_features(self) -> List[tuple[str, float]]:
        """Get top 10 most important features."""
        sorted_features = sorted(
            self.feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_features[:10]

    @property
    def worst_test_type(self) -> Optional[str]:
        """Get test type with lowest resilience score."""
        if not self.test_scores:
            return None
        return min(self.test_scores.items(), key=lambda x: x[1])[0]

    @property
    def best_test_type(self) -> Optional[str]:
        """Get test type with highest resilience score."""
        if not self.test_scores:
            return None
        return max(self.test_scores.items(), key=lambda x: x[1])[0]

    def get_test_type_summary(self, test_type: str) -> TestTypeSummary:
        """Get summary for a specific test type."""
        test_data_map = {
            'distribution_shift': self.distribution_shift_scenarios,
            'worst_sample': self.worst_sample_tests,
            'worst_cluster': self.worst_cluster_tests,
            'outer_sample': self.outer_sample_tests,
            'hard_sample': self.hard_sample_tests,
        }

        tests = test_data_map.get(test_type, [])
        valid_tests = [t for t in tests if t.is_valid]

        avg_gap = 0.0
        if valid_tests:
            gaps = [t.performance_gap for t in valid_tests if t.performance_gap is not None]
            avg_gap = sum(gaps) / len(gaps) if gaps else 0.0

        return TestTypeSummary(
            test_type=test_type,
            total_tests=len(tests),
            valid_tests=len(valid_tests),
            avg_performance_gap=avg_gap,
            has_results=len(tests) > 0
        )

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for quick overview."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'resilience_score': self.metrics.resilience_score,
            'is_resilient': self.metrics.is_resilient,
            'total_scenarios': self.metrics.total_scenarios,
            'valid_scenarios': self.metrics.valid_scenarios,
            'avg_performance_gap': self.metrics.avg_performance_gap,
            'max_performance_gap': self.metrics.max_performance_gap,
            'num_test_types': self.num_test_types,
            'available_test_types': self.available_test_types,
            'worst_test_type': self.worst_test_type,
            'best_test_type': self.best_test_type,
        }

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"ResilienceReport({self.model_name}, "
            f"score={self.metrics.resilience_score:.3f}, "
            f"gap={self.metrics.avg_performance_gap:.3f}, "
            f"tests={self.num_test_types})"
        )
