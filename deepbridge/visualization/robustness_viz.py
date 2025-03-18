"""
Facade class for robustness visualizations.
Maintains backward compatibility while delegating to specialized classes.
"""

from typing import Dict, Optional
from plotly.subplots import make_subplots

from deepbridge.visualization.robustness.performance_viz import PerformanceViz
from deepbridge.visualization.robustness.feature_viz import FeatureViz
from deepbridge.visualization.robustness.distribution_viz import DistributionViz
from deepbridge.visualization.robustness.comparison_viz import ComparisonViz
from deepbridge.visualization.robustness.dashboard_viz import DashboardViz

class RobustnessViz:
    """
    Visualization tools for model robustness analysis using Plotly.
    
    This class provides various interactive plotting functions to visualize
    the results of robustness tests, showing how model performance
    changes under different perturbation conditions.
    """
    
    @staticmethod
    def create_performance_chart(results, metric_name=None, height=None, width=None):
        """Delegate to PerformanceViz."""
        return PerformanceViz.create_performance_chart(results, metric_name, height, width)
    
    @staticmethod
    def plot_robustness_heatmap(results, model_name, title=None, height=600, width=800):
        """Delegate to FeatureViz."""
        return FeatureViz.plot_robustness_heatmap(results, model_name, title, height, width)
    
    @staticmethod
    def plot_3d_robustness_surface(results, title=None, height=800, width=900):
        """Delegate to DistributionViz."""
        return DistributionViz.plot_3d_robustness_surface(results, title, height, width)
    
    @staticmethod
    def plot_robustness_radar(robustness_indices, feature_importance, title=None, height=700, width=700):
        """Delegate to DistributionViz."""
        return DistributionViz.plot_robustness_radar(robustness_indices, feature_importance, title, height, width)
    
    @staticmethod
    def plot_boxplot_performance(results, model_name, title=None, metric_name=None, height=300, width=900):
        """Delegate to PerformanceViz."""
        return PerformanceViz.plot_boxplot_performance(results, model_name, title, metric_name, height, width)
    
    @staticmethod
    def plot_models_comparison(results, use_worst=False, alpha=0.3, title=None, metric_name=None, height=None, width=None):
        """Delegate to ComparisonViz."""
        return ComparisonViz.plot_models_comparison(results, use_worst, alpha, title, metric_name, height, width)
    
    @staticmethod
    def plot_feature_importance(feature_importance_results, title=None, top_n=8, height=None, width=None):
        """Delegate to FeatureViz."""
        return FeatureViz.plot_feature_importance(feature_importance_results, title, top_n, height, width)
    

    @staticmethod
    def plot_feature_importance_multiple(feature_importance_results_dict, title_prefix="Feature Impact on Model Robustness - ", top_n=7, height=None, width=None, subplot_layout=False):
        """Delegate to FeatureViz."""
        return FeatureViz.plot_feature_importance_multiple(feature_importance_results_dict, title_prefix, top_n, height, width, subplot_layout)
    
    @staticmethod
    def plot_robustness_index(results, robustness_indices, title=None, height=600, width=800):
        """Delegate to ComparisonViz."""
        return ComparisonViz.plot_robustness_index(results, robustness_indices, title, height, width)
    
    @staticmethod
    def plot_perturbation_methods_comparison(methods_comparison_results, title=None, metric_name=None, height=None, width=None):
        """Delegate to ComparisonViz."""
        return ComparisonViz.plot_perturbation_methods_comparison(methods_comparison_results, title, metric_name, height, width)
    
    @staticmethod
    def create_robustness_dashboard(results, feature_importance_results, robustness_indices, metric_name=None, height=1200, width=1000):
        """Delegate to DashboardViz."""
        return DashboardViz.create_robustness_dashboard(results, feature_importance_results, robustness_indices, metric_name, height, width)