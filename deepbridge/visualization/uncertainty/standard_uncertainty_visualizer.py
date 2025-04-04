"""
Standard implementation of uncertainty visualization.
"""

import typing as t
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from deepbridge.visualization.uncertainty.base_uncertainty_visualizer import BaseUncertaintyVisualizer

class StandardUncertaintyVisualizer(BaseUncertaintyVisualizer):
    """
    Standard implementation of uncertainty visualization.
    Implements the BaseUncertaintyVisualizer interface.
    """
    
    def __init__(self, title_prefix: str = "Uncertainty - ", include_plotly_mode_bar: bool = True, verbose: bool = False):
        """
        Initialize the standard uncertainty visualizer.
        
        Args:
            title_prefix: Prefix to add to all visualization titles
            include_plotly_mode_bar: Whether to include Plotly's modebar
            verbose: Whether to print progress information
        """
        super().__init__(title_prefix, include_plotly_mode_bar)
        self.verbose = verbose
    
    def create_calibration_plot(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a calibration plot comparing expected vs actual probabilities.
        
        Args:
            results: Uncertainty test results
            
        Returns:
            Plotly figure object
        """
        # Extract calibration data
        expected_probs = []
        actual_probs = []
        
        if 'calibration' in results:
            calibration_data = results['calibration']
            if isinstance(calibration_data, dict):
                if 'expected_probs' in calibration_data and 'actual_probs' in calibration_data:
                    expected_probs = calibration_data['expected_probs']
                    actual_probs = calibration_data['actual_probs']
        
        # If no data, generate synthetic data for demonstration
        if not expected_probs or not actual_probs:
            # Generate bins from 0.05 to 0.95
            expected_probs = np.arange(0.05, 1.0, 0.1)
            # Generate actual probs with slight miscalibration
            actual_probs = np.clip(expected_probs * 0.9 + 0.05, 0, 1)
        
        # Create calibration plot
        fig = go.Figure()
        
        # Add perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(dash='dash', color='gray')
        ))
        
        # Add actual calibration line
        fig.add_trace(go.Scatter(
            x=expected_probs,
            y=actual_probs,
            mode='lines+markers',
            name='Model Calibration'
        ))
        
        # Update layout
        fig = self.update_figure_layout(
            fig,
            title="Probability Calibration Curve",
            x_title="Expected Probability",
            y_title="Observed Probability"
        )
        
        return fig
    
    def create_confidence_histogram(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a histogram of confidence/probability values.
        
        Args:
            results: Uncertainty test results
            
        Returns:
            Plotly figure object
        """
        # Extract confidence values
        confidence_values = []
        
        if 'confidence_distribution' in results:
            confidence_data = results['confidence_distribution']
            if isinstance(confidence_data, dict) and 'values' in confidence_data:
                confidence_values = confidence_data['values']
            elif isinstance(confidence_data, (list, np.ndarray)):
                confidence_values = confidence_data
        
        # If no data, generate synthetic data for demonstration
        if not confidence_values:
            # Generate random confidence values, skewed towards higher values
            confidence_values = np.random.beta(5, 2, 1000)
        
        # Create histogram
        fig = go.Figure(data=[
            go.Histogram(
                x=confidence_values,
                nbinsx=20,
                marker_color='rgba(0, 0, 255, 0.7)'
            )
        ])
        
        # Update layout
        fig = self.update_figure_layout(
            fig,
            title="Confidence Distribution",
            x_title="Confidence Value",
            y_title="Frequency"
        )
        
        return fig
    
    def create_feature_importance_plot(self, 
                                     feature_importance: t.Dict[str, float], 
                                     top_n: int = 10) -> go.Figure:
        """
        Create a bar chart showing feature importance for uncertainty.
        
        Args:
            feature_importance: Mapping of feature names to importance scores
            top_n: Number of top features to show
            
        Returns:
            Plotly figure object
        """
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top_n features
        if len(sorted_features) > top_n:
            sorted_features = sorted_features[:top_n]
        
        features = [f[0] for f in sorted_features]
        importance = [f[1] for f in sorted_features]
        
        # Create horizontal bar chart
        fig = self.create_bar_chart(
            importance,
            features,
            title="Feature Importance for Uncertainty",
            x_title="Importance Score",
            y_title="Feature",
            horizontal=True
        )
        
        return fig
    
    def create_alpha_comparison_plot(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a plot comparing different alpha levels for confidence intervals.
        
        Args:
            results: Uncertainty test results
            
        Returns:
            Plotly figure object
        """
        # Extract alpha comparison data
        alphas = []
        coverage = []
        width = []
        
        if 'alpha_comparison' in results:
            alpha_data = results['alpha_comparison']
            if isinstance(alpha_data, dict):
                for alpha, alpha_results in sorted(alpha_data.items()):
                    if isinstance(alpha_results, dict):
                        alphas.append(float(alpha))
                        coverage.append(alpha_results.get('coverage', 0))
                        width.append(alpha_results.get('avg_width', 0))
        
        # If no data, generate synthetic data for demonstration
        if not alphas:
            alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            # Generate coverage that generally increases with alpha
            coverage = [0.1 + 0.85 * a + np.random.uniform(-0.05, 0.05) for a in alphas]
            # Generate width that increases with alpha
            width = [0.2 + 1.8 * a + np.random.uniform(-0.1, 0.1) for a in alphas]
        
        # Create dual-axis plot
        fig = go.Figure()
        
        # Add coverage trace
        fig.add_trace(go.Scatter(
            x=alphas,
            y=coverage,
            mode='lines+markers',
            name='Coverage',
            marker=dict(size=8, color='blue')
        ))
        
        # Add width trace on secondary axis
        fig.add_trace(go.Scatter(
            x=alphas,
            y=width,
            mode='lines+markers',
            name='Interval Width',
            marker=dict(size=8, color='red'),
            yaxis='y2'
        ))
        
        # Update layout for dual axis
        fig.update_layout(
            title=f"{self.title_prefix}Effect of Alpha on Coverage and Width",
            xaxis=dict(title="Alpha Level"),
            yaxis=dict(title="Coverage Rate", side="left", range=[0, 1.1]),
            yaxis2=dict(title="Interval Width", side="right", overlaying="y", range=[0, max(width) * 1.1]),
            legend=dict(x=0.02, y=0.98),
            template="plotly_white"
        )
        
        return fig
    
    def create_width_distribution_plot(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a box plot showing the distribution of interval widths.
        
        Args:
            results: Uncertainty test results
            
        Returns:
            Plotly figure object
        """
        # Extract width distribution data
        alpha_levels = []
        width_distributions = []
        
        if 'interval_widths' in results:
            widths_data = results['interval_widths']
            if isinstance(widths_data, dict):
                for alpha, widths in sorted(widths_data.items()):
                    if isinstance(widths, (list, np.ndarray)) and len(widths) > 0:
                        alpha_levels.append(str(alpha))
                        width_distributions.append(widths)
        
        # If no data, generate synthetic data for demonstration
        if not width_distributions:
            alpha_levels = ['0.1', '0.3', '0.5', '0.7', '0.9']
            for i, a in enumerate(alpha_levels):
                alpha_float = float(a)
                # Generate widths with increasing mean and variance as alpha increases
                mean_width = 0.2 + 1.8 * alpha_float
                width_distribution = np.random.normal(mean_width, 0.1 + 0.4 * alpha_float, 100)
                width_distributions.append(width_distribution)
        
        # Create box plot
        fig = self.create_box_plot(
            width_distributions,
            labels=alpha_levels,
            title="Distribution of Interval Widths by Alpha Level",
            x_title="Alpha Level",
            y_title="Interval Width"
        )
        
        return fig
    
    def create_coverage_vs_width_plot(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a plot showing the trade-off between coverage and interval width.
        
        Args:
            results: Uncertainty test results
            
        Returns:
            Plotly figure object
        """
        # Extract coverage and width data
        coverage = []
        width = []
        labels = []
        
        if 'coverage_vs_width' in results:
            cv_data = results['coverage_vs_width']
            if isinstance(cv_data, dict) and 'methods' in cv_data:
                for method_name, method_data in cv_data['methods'].items():
                    if isinstance(method_data, dict):
                        if 'coverage' in method_data and 'width' in method_data:
                            coverage.append(method_data['coverage'])
                            width.append(method_data['width'])
                            labels.append(method_name)
        
        # If no data, generate synthetic data for demonstration
        if not coverage or not width:
            method_names = ['Method A', 'Method B', 'Method C', 'Method D', 'Method E']
            # Generate random coverage and width values
            coverage = [0.7 + 0.25 * np.random.random() for _ in method_names]
            width = [0.5 + 2.0 * np.random.random() for _ in method_names]
            labels = method_names
        
        # Create scatter plot
        fig = go.Figure(data=[
            go.Scatter(
                x=width,
                y=coverage,
                mode='markers+text',
                marker=dict(
                    size=12,
                    color=np.arange(len(labels)),
                    colorscale='Viridis',
                    showscale=False
                ),
                text=labels,
                textposition="top center",
                textfont=dict(size=10)
            )
        ])
        
        # Update layout
        fig = self.update_figure_layout(
            fig,
            title="Coverage vs Width Trade-off",
            x_title="Interval Width",
            y_title="Coverage Rate"
        )
        
        # Add shaded area for better than random region
        # Typically, methods with higher coverage and lower width are better
        fig.add_shape(
            type="rect",
            x0=0,
            y0=0.95,
            x1=min(width) * 0.9,
            y1=1.05,
            fillcolor="rgba(0,255,0,0.1)",
            line=dict(width=0),
            layer="below"
        )
        
        # Add note for the ideal region
        fig.add_annotation(
            x=min(width) * 0.45,
            y=1.0,
            text="Ideal Region",
            showarrow=False,
            font=dict(size=10, color="green")
        )
        
        return fig