"""
Fairness chart modules.

Contains all chart implementations for fairness visualizations.
"""

from .base_chart import BaseChart
from .complementary_charts import (
    ComplementaryMetricsRadarChart,
    PrecisionAccuracyComparisonChart,
    TreatmentEqualityScatterChart,
)
from .distribution_charts import (
    ProtectedAttributesDistributionChart,
    TargetDistributionChart,
)
from .legacy_charts import (
    ConfusionMatricesChart,
    FairnessRadarChart,
    MetricsComparisonChart,
    ThresholdAnalysisChart,
)
from .posttrain_charts import (
    ComplianceStatusMatrixChart,
    DisparateImpactGaugeChart,
    DisparityComparisonChart,
)
from .pretrain_charts import (
    ConceptBalanceChart,
    GroupSizesChart,
    PretrainMetricsOverviewChart,
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
    'ThresholdAnalysisChart',
]
