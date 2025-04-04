"""
Module for creating visualizations of robustness test results.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Union, Any

class RobustnessVisualizer:
    """
    Creates visualizations for robustness test results.
    Extracted from RobustnessSuite to separate visualization responsibilities.
    """
    
    def __init__(self, title_prefix: str = ""):
        """
        Initialize the visualizer.
        
        Parameters:
        -----------
        title_prefix : str
            Prefix to add to all visualization titles
        """
        self.title_prefix = title_prefix
    
    def create_model_comparison_plot(self, 
                                    results: Dict[str, Any], 
                                    alternative_results: Optional[Dict[str, Dict[str, Any]]] = None) -> go.Figure:
        """
        Create a bar chart comparing robustness across models.
        
        Parameters:
        -----------
        results : Dict
            Primary model results
        alternative_results : Dict or None
            Results for alternative models
            
        Returns:
        --------
        Figure : Plotly figure object
        """
        models = ['Primary Model']
        avg_impacts = [results.get('avg_overall_impact', 0)]
        
        # Add alternative models if available
        if alternative_results:
            for model_name, model_results in alternative_results.items():
                models.append(model_name)
                avg_impacts.append(model_results.get('avg_overall_impact', 0))
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=avg_impacts,
                text=[f"{impact:.3f}" for impact in avg_impacts],
                textposition='auto'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Model Robustness Comparison",
            xaxis_title="Model",
            yaxis_title="Average Impact (lower is better)",
            template="plotly_white"
        )
        
        return fig
    
    def create_score_distribution_plot(self, 
                                      results: Dict[str, Any], 
                                      method: str = 'both') -> go.Figure:
        """
        Create a box plot showing the distribution of scores across perturbation levels.
        
        Parameters:
        -----------
        results : Dict
            Robustness test results
        method : str
            Which perturbation method to visualize ('raw', 'quantile', or 'both')
            
        Returns:
        --------
        Figure : Plotly figure object
        """
        data = []
        
        if method in ['raw', 'both'] and 'raw' in results:
            # Extract raw perturbation results by level
            raw_levels = []
            raw_scores = []
            raw_methods = []
            
            for level, level_data in results['raw'].get('by_level', {}).items():
                if 'overall_result' in level_data:
                    level_float = float(level)
                    score = level_data['overall_result'].get('mean_score', 0)
                    raw_levels.append(level_float)
                    raw_scores.append(score)
                    raw_methods.append('Raw')
            
            data.extend([
                go.Box(
                    y=raw_scores,
                    x=raw_levels,
                    name='Raw Perturbation',
                    boxpoints='all'
                )
            ])
        
        if method in ['quantile', 'both'] and 'quantile' in results:
            # Extract quantile perturbation results by level
            quantile_levels = []
            quantile_scores = []
            quantile_methods = []
            
            for level, level_data in results['quantile'].get('by_level', {}).items():
                if 'overall_result' in level_data:
                    level_float = float(level)
                    score = level_data['overall_result'].get('mean_score', 0)
                    quantile_levels.append(level_float)
                    quantile_scores.append(score)
                    quantile_methods.append('Quantile')
            
            data.extend([
                go.Box(
                    y=quantile_scores,
                    x=quantile_levels,
                    name='Quantile Perturbation',
                    boxpoints='all'
                )
            ])
        
        # Create combined figure
        fig = go.Figure(data=data)
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Score Distribution by Perturbation Level",
            xaxis_title="Perturbation Level",
            yaxis_title="Score",
            template="plotly_white",
            boxmode='group'
        )
        
        return fig
    
    def create_feature_importance_plot(self, 
                                      feature_importance: Dict[str, float], 
                                      top_n: int = 10,
                                      show_all: bool = False) -> go.Figure:
        """
        Create a bar chart showing feature importance for robustness.
        
        Parameters:
        -----------
        feature_importance : Dict
            Mapping of feature names to importance scores
        top_n : int
            Number of top features to show
        show_all : bool
            Whether to show all features regardless of top_n
            
        Returns:
        --------
        Figure : Plotly figure object
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top_n if needed
        if not show_all and len(sorted_features) > top_n:
            sorted_features = sorted_features[:top_n]
        
        features = [feat[0] for feat in sorted_features]
        importance = [feat[1] for feat in sorted_features]
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=importance,
                y=features,
                orientation='h',
                text=[f"{imp:.3f}" for imp in importance],
                textposition='auto'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Feature Importance for Robustness",
            xaxis_title="Impact Score (higher means more impactful)",
            yaxis_title="Feature",
            template="plotly_white"
        )
        
        return fig
    
    def create_methods_comparison_plot(self, results: Dict[str, Any]) -> go.Figure:
        """
        Create a bar chart comparing different perturbation methods.
        
        Parameters:
        -----------
        results : Dict
            Robustness test results
            
        Returns:
        --------
        Figure : Plotly figure object
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
        fig = go.Figure(data=[
            go.Bar(
                x=methods,
                y=impacts,
                text=[f"{impact:.3f}" for impact in impacts],
                textposition='auto'
            )
        ])
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Perturbation Methods Comparison",
            xaxis_title="Method",
            yaxis_title="Average Impact (lower is better)",
            template="plotly_white"
        )
        
        return fig
    
    def create_heatmap(self, 
                      feature_importance: Dict[str, float], 
                      X: pd.DataFrame,
                      top_n: int = 10) -> go.Figure:
        """
        Create a heatmap showing feature correlations with importance.
        
        Parameters:
        -----------
        feature_importance : Dict
            Mapping of feature names to importance scores
        X : DataFrame
            Feature data to calculate correlations
        top_n : int
            Number of top features to include
            
        Returns:
        --------
        Figure : Plotly figure object
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top_n
        if len(sorted_features) > top_n:
            sorted_features = sorted_features[:top_n]
        
        top_features = [feat[0] for feat in sorted_features]
        
        # Calculate correlation matrix for top features
        corr_matrix = X[top_features].corr()
        
        # Create heatmap
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Feature Correlation Matrix for Top {len(top_features)} Features",
            template="plotly_white"
        )
        
        return fig