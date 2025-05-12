"""
Resilience charts package - provides chart generation for resilience reports.
"""

from .base_chart import BaseChartGenerator
from .feature_distribution_shift import FeatureDistributionShiftChart
from .performance_gap import PerformanceGapChart
from .critical_feature_distributions import CriticalFeatureDistributionsChart
from .feature_residual_correlation import FeatureResidualCorrelationChart
from .residual_distribution import ResidualDistributionChart
from .model_comparison import ModelComparisonChart
from .model_comparison_scatter import ModelComparisonScatterChart
from .distance_metrics_comparison import DistanceMetricsComparisonChart
from .feature_distance_heatmap import FeatureDistanceHeatmapChart
from .model_resilience_scores import ModelResilienceScoresChart
from .performance_gap_by_alpha import PerformanceGapByAlphaChart


class ResilienceChartGenerator:
    """
    Main class that provides access to all resilience chart generators.
    """
    
    def __init__(self, seaborn_chart_generator=None):
        """
        Initialize the resilience chart generator.
        
        Parameters:
        ----------
        seaborn_chart_generator : SeabornChartGenerator, optional
            Existing chart generator to use for rendering
        """
        self.seaborn_chart_generator = seaborn_chart_generator
        
        # Initialize individual chart generators
        self.feature_distribution_shift = FeatureDistributionShiftChart(seaborn_chart_generator)
        self.performance_gap = PerformanceGapChart(seaborn_chart_generator)
        self.critical_feature_distributions = CriticalFeatureDistributionsChart(seaborn_chart_generator)
        self.feature_residual_correlation = FeatureResidualCorrelationChart(seaborn_chart_generator)
        self.residual_distribution = ResidualDistributionChart(seaborn_chart_generator)
        self.model_comparison = ModelComparisonChart(seaborn_chart_generator)
        self.model_comparison_scatter = ModelComparisonScatterChart(seaborn_chart_generator)
        self.distance_metrics_comparison = DistanceMetricsComparisonChart(seaborn_chart_generator)
        self.feature_distance_heatmap = FeatureDistanceHeatmapChart(seaborn_chart_generator)
        self.model_resilience_scores = ModelResilienceScoresChart(seaborn_chart_generator)
        self.performance_gap_by_alpha = PerformanceGapByAlphaChart(seaborn_chart_generator)
    
    # Wrapper methods to maintain backward compatibility
    
    def generate_feature_distribution_shift(self, feature_distances, title="Feature Distribution Shift", top_n=10):
        """Generate a chart showing the distribution shift for different features."""
        return self.feature_distribution_shift.generate(feature_distances, title, top_n)
    
    def generate_performance_gap(self, performance_metrics, title="Performance Comparison: Worst vs Remaining Samples", task_type="classification"):
        """Generate a chart comparing performance between worst and remaining samples."""
        return self.performance_gap.generate(performance_metrics, title, task_type)
    
    def generate_critical_feature_distributions(self, worst_samples, remaining_samples, top_features, title="Critical Feature Distributions"):
        """Generate a chart showing distributions of critical features."""
        return self.critical_feature_distributions.generate(worst_samples, remaining_samples, top_features, title)
    
    def generate_feature_residual_correlation(self, feature_correlations, title="Feature-Residual Correlation", top_n=8):
        """Generate a chart showing correlation between features and residuals."""
        return self.feature_residual_correlation.generate(feature_correlations, title, top_n)
    
    def generate_residual_distribution(self, worst_residuals=None, remaining_residuals=None, all_residuals=None, title="Model Residual Distribution"):
        """Generate a chart showing the distribution of residuals."""
        return self.residual_distribution.generate(worst_residuals, remaining_residuals, all_residuals, title)
    
    def generate_model_comparison(self, perturbation_levels, models_data, title="Model Resilience Comparison", metric_name="Score"):
        """Generate a chart comparing multiple models across perturbation levels."""
        return self.model_comparison.generate(perturbation_levels, models_data, title, metric_name)
    
    def generate_model_comparison_scatter(self, models_data, title="Accuracy vs Resilience Score", x_label="Accuracy", y_label="Resilience Score"):
        """Generate a scatter plot comparing accuracy and resilience scores across models."""
        return self.model_comparison_scatter.generate(models_data, title, x_label, y_label)
    
    def generate_distance_metrics_comparison(self, alpha_levels, metrics_data, title="Distance Metrics Comparison by Alpha", y_label="Distance Value"):
        """Generate a line chart comparing different distance metrics across alpha levels."""
        return self.distance_metrics_comparison.generate(alpha_levels, metrics_data, title, y_label)
    
    def generate_feature_distance_heatmap(self, feature_distances, title="Feature Distance Heatmap", top_n=15, cmap="viridis"):
        """Generate a heatmap showing feature distances across multiple models or metrics."""
        return self.feature_distance_heatmap.generate(feature_distances, title, top_n, cmap)
    
    def generate_model_resilience_scores(self, models_data, title="Resilience Scores by Model", sort_by="score", ascending=False):
        """Generate a horizontal bar chart showing resilience scores for different models."""
        return self.model_resilience_scores.generate(models_data, title, sort_by, ascending)
    
    def generate_performance_gap_by_alpha(self, alpha_levels, models_data, title="Performance Gap by Alpha Level", y_label="Performance Gap"):
        """Generate a line chart showing performance gaps across alpha levels for different models."""
        return self.performance_gap_by_alpha.generate(alpha_levels, models_data, title, y_label)