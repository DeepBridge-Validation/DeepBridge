"""
Uncertainty charts package - provides chart generation for uncertainty reports.
"""

from .base_chart import BaseChartGenerator
from .coverage_vs_expected import CoverageVsExpectedChart
from .width_vs_coverage import WidthVsCoverageChart
from .uncertainty_metrics import UncertaintyMetricsChart
from .feature_importance import FeatureImportanceChart
from .model_comparison import ModelComparisonChart
from .performance_gap_by_alpha import PerformanceGapByAlphaChart


class UncertaintyChartGenerator:
    """
    Main class that provides access to all uncertainty chart generators.
    """
    
    def __init__(self, seaborn_chart_generator=None):
        """
        Initialize the uncertainty chart generator.
        
        Parameters:
        ----------
        seaborn_chart_generator : SeabornChartGenerator, optional
            Existing chart generator to use for rendering
        """
        self.seaborn_chart_generator = seaborn_chart_generator
        
        # Initialize individual chart generators
        self.coverage_vs_expected = CoverageVsExpectedChart(seaborn_chart_generator)
        self.width_vs_coverage = WidthVsCoverageChart(seaborn_chart_generator)
        self.uncertainty_metrics = UncertaintyMetricsChart(seaborn_chart_generator)
        self.feature_importance = FeatureImportanceChart(seaborn_chart_generator)
        self.model_comparison = ModelComparisonChart(seaborn_chart_generator)
        self.performance_gap_by_alpha = PerformanceGapByAlphaChart(seaborn_chart_generator)
    
    # Wrapper methods to maintain backward compatibility
    
    def generate_coverage_vs_expected(self, models_data, title="Coverage vs Expected Coverage", add_annotations=True):
        """Generate a chart comparing real coverage with expected coverage for different alpha values."""
        return self.coverage_vs_expected.generate(models_data, title, add_annotations)
    
    def generate_width_vs_coverage(self, models_data, title="Interval Width vs Coverage"):
        """Generate a chart showing the relationship between interval width and coverage."""
        return self.width_vs_coverage.generate(models_data, title)
    
    def generate_uncertainty_metrics(self, models_data, title="Uncertainty Metrics Comparison"):
        """Generate a chart comparing different uncertainty metrics across models."""
        return self.uncertainty_metrics.generate(models_data, title)
    
    def generate_feature_importance(self, feature_importance_data, title="Feature Importance for Uncertainty"):
        """Generate a chart showing feature importance for uncertainty."""
        return self.feature_importance.generate(feature_importance_data, title)
    
    def generate_model_comparison(self, models_data, title="Model Comparison"):
        """Generate a chart comparing models based on uncertainty metrics."""
        return self.model_comparison.generate(models_data, title)
        
    def generate_performance_gap_by_alpha(self, models_data, title="Performance Gap by Alpha Level", add_annotations=True):
        """Generate a chart showing performance gaps across alpha levels for different models."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        logger.info(f"generate_performance_gap_by_alpha called with: models_data={type(models_data)}, title={title}, add_annotations={add_annotations}")

        # Log the models_data keys
        if isinstance(models_data, dict):
            logger.info(f"models_data keys: {list(models_data.keys())}")

            # Check for calibration_results
            if 'calibration_results' in models_data:
                logger.info(f"calibration_results keys: {list(models_data['calibration_results'].keys())}")

            # Check for alpha_levels
            if 'alpha_levels' in models_data:
                logger.info(f"alpha_levels: {models_data['alpha_levels']}")

        return self.performance_gap_by_alpha.generate(models_data, title, add_annotations)