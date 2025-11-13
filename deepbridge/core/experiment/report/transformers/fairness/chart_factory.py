"""
Chart factory for fairness visualizations.

Orchestrates the creation of all fairness charts.
"""

from typing import Dict, Any
import logging

from .charts.posttrain_charts import (
    DisparateImpactGaugeChart,
    DisparityComparisonChart,
    ComplianceStatusMatrixChart
)
from .charts.pretrain_charts import (
    PretrainMetricsOverviewChart,
    GroupSizesChart,
    ConceptBalanceChart
)
from .charts.complementary_charts import (
    PrecisionAccuracyComparisonChart,
    TreatmentEqualityScatterChart,
    ComplementaryMetricsRadarChart
)
from .charts.distribution_charts import (
    ProtectedAttributesDistributionChart,
    TargetDistributionChart
)
from .charts.legacy_charts import (
    MetricsComparisonChart,
    FairnessRadarChart,
    ConfusionMatricesChart,
    ThresholdAnalysisChart
)

logger = logging.getLogger("deepbridge.reports")


class ChartFactory:
    """
    Factory class for creating fairness charts.

    Orchestrates the creation of all chart types and manages their dependencies.
    """

    def __init__(self):
        """Initialize chart factory with all chart instances."""
        # Post-training charts
        self.disparate_impact_gauge = DisparateImpactGaugeChart()
        self.disparity_comparison = DisparityComparisonChart()
        self.status_matrix = ComplianceStatusMatrixChart()

        # Pre-training charts
        self.pretrain_overview = PretrainMetricsOverviewChart()
        self.group_sizes = GroupSizesChart()
        self.concept_balance = ConceptBalanceChart()

        # Complementary charts
        self.precision_accuracy = PrecisionAccuracyComparisonChart()
        self.treatment_equality = TreatmentEqualityScatterChart()
        self.complementary_radar = ComplementaryMetricsRadarChart()

        # Distribution charts
        self.protected_attrs_distribution = ProtectedAttributesDistributionChart()
        self.target_distribution = TargetDistributionChart()

        # Legacy charts
        self.metrics_comparison = MetricsComparisonChart()
        self.fairness_radar = FairnessRadarChart()
        self.confusion_matrices = ConfusionMatricesChart()
        self.threshold_analysis = ThresholdAnalysisChart()

    def create_all_charts(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Create all applicable charts based on available data.

        Args:
            results: Dictionary containing fairness analysis results

        Returns:
            Dictionary with chart names as keys and Plotly JSON strings as values
        """
        logger.info("Creating fairness charts")
        charts = {}

        # Extract common data
        protected_attrs = results.get('protected_attributes', [])
        posttrain_metrics = results.get('posttrain_metrics', {})
        pretrain_metrics = results.get('pretrain_metrics', {})
        confusion_matrix = results.get('confusion_matrix', {})
        threshold_analysis_data = results.get('threshold_analysis', None)
        dataset_info = results.get('dataset_info', {})

        # Legacy charts (for backward compatibility)
        try:
            charts['metrics_comparison'] = self.metrics_comparison.create({
                'posttrain_metrics': posttrain_metrics,
                'protected_attrs': protected_attrs
            })
        except Exception as e:
            logger.warning(f"Failed to create metrics_comparison chart: {e}")

        try:
            charts['fairness_radar'] = self.fairness_radar.create({
                'posttrain_metrics': posttrain_metrics
            })
        except Exception as e:
            logger.warning(f"Failed to create fairness_radar chart: {e}")

        # Confusion matrices
        if confusion_matrix:
            try:
                charts['confusion_matrices'] = self.confusion_matrices.create({
                    'confusion_matrix': confusion_matrix,
                    'protected_attrs': protected_attrs
                })
            except Exception as e:
                logger.warning(f"Failed to create confusion_matrices chart: {e}")

        # Threshold analysis
        if threshold_analysis_data:
            try:
                charts['threshold_analysis'] = self.threshold_analysis.create({
                    'threshold_analysis': threshold_analysis_data
                })
            except Exception as e:
                logger.warning(f"Failed to create threshold_analysis chart: {e}")

        # Distribution charts
        if dataset_info and dataset_info.get('protected_attributes_distribution'):
            try:
                charts['protected_attributes_distribution'] = self.protected_attrs_distribution.create({
                    'protected_attrs_distribution': dataset_info.get('protected_attributes_distribution', {}),
                    'protected_attrs': protected_attrs
                })
            except Exception as e:
                logger.warning(f"Failed to create protected_attributes_distribution chart: {e}")

        if dataset_info and dataset_info.get('target_distribution'):
            try:
                charts['target_distribution'] = self.target_distribution.create({
                    'target_distribution': dataset_info.get('target_distribution', {})
                })
            except Exception as e:
                logger.warning(f"Failed to create target_distribution chart: {e}")

        # Post-training charts
        if posttrain_metrics:
            try:
                charts['posttrain_disparate_impact_gauge'] = self.disparate_impact_gauge.create({
                    'posttrain_metrics': posttrain_metrics,
                    'protected_attrs': protected_attrs
                })
            except Exception as e:
                logger.warning(f"Failed to create disparate_impact_gauge chart: {e}")

            try:
                charts['posttrain_disparity_comparison'] = self.disparity_comparison.create({
                    'posttrain_metrics': posttrain_metrics,
                    'protected_attrs': protected_attrs
                })
            except Exception as e:
                logger.warning(f"Failed to create disparity_comparison chart: {e}")

            try:
                charts['posttrain_status_matrix'] = self.status_matrix.create({
                    'posttrain_metrics': posttrain_metrics,
                    'protected_attrs': protected_attrs
                })
            except Exception as e:
                logger.warning(f"Failed to create status_matrix chart: {e}")

        # Pre-training charts
        if pretrain_metrics:
            try:
                charts['pretrain_metrics_overview'] = self.pretrain_overview.create({
                    'pretrain_metrics': pretrain_metrics,
                    'protected_attrs': protected_attrs
                })
            except Exception as e:
                logger.warning(f"Failed to create pretrain_overview chart: {e}")

            try:
                charts['pretrain_concept_balance'] = self.concept_balance.create({
                    'pretrain_metrics': pretrain_metrics,
                    'protected_attrs': protected_attrs
                })
            except Exception as e:
                logger.warning(f"Failed to create concept_balance chart: {e}")

        # Group sizes from dataset_info
        if dataset_info and dataset_info.get('protected_attributes_distribution'):
            try:
                charts['pretrain_group_sizes'] = self.group_sizes.create({
                    'protected_attrs_distribution': dataset_info.get('protected_attributes_distribution', {}),
                    'protected_attrs': protected_attrs
                })
            except Exception as e:
                logger.warning(f"Failed to create group_sizes chart: {e}")

        # Complementary charts
        if confusion_matrix and posttrain_metrics:
            try:
                charts['complementary_precision_accuracy'] = self.precision_accuracy.create({
                    'posttrain_metrics': posttrain_metrics,
                    'confusion_matrix': confusion_matrix,
                    'protected_attrs': protected_attrs
                })
            except Exception as e:
                logger.warning(f"Failed to create precision_accuracy chart: {e}")

            try:
                charts['complementary_treatment_equality'] = self.treatment_equality.create({
                    'confusion_matrix': confusion_matrix,
                    'protected_attrs': protected_attrs
                })
            except Exception as e:
                logger.warning(f"Failed to create treatment_equality chart: {e}")

        if posttrain_metrics:
            try:
                charts['complementary_radar'] = self.complementary_radar.create({
                    'posttrain_metrics': posttrain_metrics,
                    'protected_attrs': protected_attrs
                })
            except Exception as e:
                logger.warning(f"Failed to create complementary_radar chart: {e}")

        logger.info(f"Created {len(charts)} charts")
        return charts
