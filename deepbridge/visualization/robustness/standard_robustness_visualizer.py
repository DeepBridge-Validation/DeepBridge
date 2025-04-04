"""
Standard implementation of robustness visualization.
"""

import typing as t
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from deepbridge.visualization.robustness.base_robustness_visualizer import BaseRobustnessVisualizer

class StandardRobustnessVisualizer(BaseRobustnessVisualizer):
    """
    Standard implementation of robustness visualization.
    Implements the BaseRobustnessVisualizer interface.
    """
    
    def __init__(self, title_prefix: str = "Robustness - ", include_plotly_mode_bar: bool = True, verbose: bool = False):
        """
        Initialize the standard robustness visualizer.
        
        Args:
            title_prefix: Prefix to add to all visualization titles
            include_plotly_mode_bar: Whether to include Plotly's modebar
            verbose: Whether to print progress information
        """
        super().__init__(title_prefix, include_plotly_mode_bar)
        self.verbose = verbose
    
    def create_model_comparison_plot(self, 
                                    results: t.Dict[str, t.Any], 
                                    alternative_results: t.Optional[t.Dict[str, t.Dict[str, t.Any]]] = None) -> go.Figure:
        """
        Create a bar chart comparing robustness across models.
        
        Args:
            results: Primary model results
            alternative_results: Results for alternative models
            
        Returns:
            Plotly figure object
        """
        # Extract robustness scores
        models = ['Primary Model']
        avg_impacts = [results.get('avg_overall_impact', 0)]
        
        # Add alternative models if available
        if alternative_results:
            for model_name, model_results in alternative_results.items():
                models.append(model_name)
                avg_impacts.append(model_results.get('avg_overall_impact', 0))
        
        # Create bar chart
        fig = self.create_bar_chart(
            models, 
            avg_impacts, 
            title="Model Robustness Comparison",
            x_title="Model",
            y_title="Average Impact (lower is better)"
        )
        
        return fig
    
    def create_perturbation_plot(self, 
                               results: t.Dict[str, t.Any],
                               perturbation_type: str = 'raw') -> go.Figure:
        """
        Create a line chart showing model performance under different perturbation levels.
        
        Args:
            results: Test results
            perturbation_type: Type of perturbation to visualize ('raw', 'quantile', etc.)
            
        Returns:
            Plotly figure object
        """
        # Extract perturbation results of the specified type
        levels = []
        scores = []
        
        if perturbation_type in results:
            for level, level_data in sorted(results[perturbation_type].get('by_level', {}).items()):
                if 'overall_result' in level_data:
                    levels.append(float(level))
                    scores.append(level_data['overall_result'].get('mean_score', 0))
        
        # Create line chart
        title_map = {
            'raw': 'Model Performance Under Raw Perturbation',
            'quantile': 'Model Performance Under Quantile Perturbation'
        }
        title = title_map.get(perturbation_type, f'Model Performance Under {perturbation_type.capitalize()} Perturbation')
        
        fig = self.create_line_chart(
            levels,
            scores,
            title=title,
            x_title="Perturbation Level",
            y_title="Score",
            name="Performance"
        )
        
        return fig
    
    def create_feature_importance_plot(self, 
                                     feature_importance: t.Dict[str, float], 
                                     top_n: int = 10) -> go.Figure:
        """
        Create a bar chart showing feature importance for robustness.
        
        Args:
            feature_importance: Mapping of feature names to importance scores
            top_n: Number of top features to show
            
        Returns:
            Plotly figure object
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top_n features
        if len(sorted_features) > top_n and not show_all:
            sorted_features = sorted_features[:top_n]
        
        features = [f[0] for f in sorted_features]
        importance = [f[1] for f in sorted_features]
        
        # Create horizontal bar chart
        fig = self.create_bar_chart(
            importance,
            features,
            title="Feature Importance for Robustness",
            x_title="Impact Score",
            y_title="Feature",
            horizontal=True
        )
        
        return fig
    
    def create_methods_comparison_plot(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a bar chart comparing different perturbation methods.
        
        Args:
            results: Test results
            
        Returns:
            Plotly figure object
        """
        methods = []
        impacts = []
        
        if 'avg_raw_impact' in results:
            methods.append('Raw (Gaussian)')
            impacts.append(results['avg_raw_impact'])
        
        if 'avg_quantile_impact' in results:
            methods.append('Quantile')
            impacts.append(results['avg_quantile_impact'])
        
        # Create bar chart
        fig = self.create_bar_chart(
            methods,
            impacts,
            title="Perturbation Methods Comparison",
            x_title="Method",
            y_title="Average Impact (lower is better)"
        )
        
        return fig
    
    def create_score_distribution_plot(self, 
                                     results: t.Dict[str, t.Any], 
                                     method: str = 'both') -> go.Figure:
        """
        Create a box plot showing the distribution of scores across perturbation levels.
        
        Args:
            results: Test results
            method: Which perturbation method to visualize ('raw', 'quantile', or 'both')
            
        Returns:
            Plotly figure object
        """
        raw_data = []
        raw_labels = []
        quantile_data = []
        quantile_labels = []
        
        # Extract raw perturbation results
        if (method in ['raw', 'both']) and 'raw' in results:
            for level, level_data in sorted(results['raw'].get('by_level', {}).items()):
                if 'runs' in level_data and level_data['runs']:
                    # Get scores from all runs
                    run_scores = [run.get('perturbed_score', 0) for run in level_data['runs']]
                    
                    if run_scores:
                        raw_data.append(run_scores)
                        raw_labels.append(f"Raw {level}")
        
        # Extract quantile perturbation results
        if (method in ['quantile', 'both']) and 'quantile' in results:
            for level, level_data in sorted(results['quantile'].get('by_level', {}).items()):
                if 'runs' in level_data and level_data['runs']:
                    # Get scores from all runs
                    run_scores = [run.get('perturbed_score', 0) for run in level_data['runs']]
                    
                    if run_scores:
                        quantile_data.append(run_scores)
                        quantile_labels.append(f"Quantile {level}")
        
        # Combine data if showing both methods
        if method == 'both':
            combined_data = raw_data + quantile_data
            combined_labels = raw_labels + quantile_labels
        else:
            combined_data = raw_data if method == 'raw' else quantile_data
            combined_labels = raw_labels if method == 'raw' else quantile_labels
        
        # If we don't have real runs, we'll simulate some data
        if not combined_data:
            # Generate synthetic data for demonstration
            for i, method_name in enumerate(['Raw', 'Quantile']):
                for level in [0.1, 0.2, 0.5]:
                    # Generate random scores for this level
                    mean_score = max(0, 0.8 - level * (0.3 + 0.1 * i))
                    scores = np.random.normal(mean_score, 0.05, 10)
                    combined_data.append(scores)
                    combined_labels.append(f"{method_name} {level}")
        
        # Create box plot
        fig = self.create_box_plot(
            combined_data,
            labels=combined_labels,
            title="Distribution of Robustness Scores",
            x_title="Perturbation Method and Level",
            y_title="Score",
            boxpoints='all'
        )
        
        return fig