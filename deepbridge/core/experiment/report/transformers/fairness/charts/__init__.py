"""
Fairness chart modules.

Contains all chart implementations for fairness visualizations.
"""

from .base_chart import BaseChart
from .posttrain_charts import (
    DisparateImpactGaugeChart,
    DisparityComparisonChart,
    ComplianceStatusMatrixChart
)
from .pretrain_charts import (
    PretrainMetricsOverviewChart,
    GroupSizesChart,
    ConceptBalanceChart
)
from .complementary_charts import (
    PrecisionAccuracyComparisonChart,
    TreatmentEqualityScatterChart,
    ComplementaryMetricsRadarChart
)
from .distribution_charts import (
    ProtectedAttributesDistributionChart,
    TargetDistributionChart
)
from .legacy_charts import (
    MetricsComparisonChart,
    FairnessRadarChart,
    ConfusionMatricesChart,
    ThresholdAnalysisChart
)

__all__ = [
    'BaseChart',
    # Post-training
    'DisparateImpactGaugeChart',
    'DisparityComparisonChart',
    'ComplianceStatusMatrixChart',
    # Pre-training
    'PretrainMetricsOverviewChart',
    'GroupSizesChart',
    'ConceptBalanceChart',
    # Complementary
    'PrecisionAccuracyComparisonChart',
    'TreatmentEqualityScatterChart',
    'ComplementaryMetricsRadarChart',
    # Distribution
    'ProtectedAttributesDistributionChart',
    'TargetDistributionChart',
    # Legacy
    'MetricsComparisonChart',
    'FairnessRadarChart',
    'ConfusionMatricesChart',
    'ThresholdAnalysisChart'
]
