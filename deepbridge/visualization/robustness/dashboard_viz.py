import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional

from deepbridge.visualization.robustness.base_viz import RobustnessBaseViz
from deepbridge.visualization.robustness.performance_viz import PerformanceViz
from deepbridge.visualization.robustness.feature_viz import FeatureViz
from deepbridge.visualization.robustness.distribution_viz import DistributionViz
from deepbridge.visualization.robustness.comparison_viz import ComparisonViz

class DashboardViz(RobustnessBaseViz):
    """Creates comprehensive dashboards combining multiple visualization types."""
    
    @staticmethod
    def create_robustness_dashboard(
        results: Dict,
        feature_importance_results: Dict,
        robustness_indices: Dict[str, float],
        metric_name: Optional[str] = None,
        height: int = 1200,
        width: int = 1000
    ):
        """
        Create a comprehensive interactive dashboard combining multiple visualizations.
        
        Args:
            results: Results from RobustnessTest.evaluate_robustness
            feature_importance_results: Results from feature importance analysis
            robustness_indices: Robustness indices from RobustnessScore.calculate_robustness_index
            metric_name: Name of the performance metric
            height: Height of the overall dashboard
            width: Width of the overall dashboard
            
        Returns:
            Plotly figure with the comprehensive dashboard
        """
        # Create subplots layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Model Performance Under Perturbation",
                "Robustness Index Comparison",
                "Top Features Impact on Robustness",
                "Performance Distribution for Best Model",
                "Robustness Radar",
                "Feature Importance Summary"
            ),
            specs=[
                [{"colspan": 2}, None],
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "polar"}, {"type": "table"}]
            ],
            column_widths=[0.6, 0.4],
            row_heights=[0.3, 0.4, 0.3],
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        # 1. Model Comparison (top row, spans both columns)
        models_fig = RobustnessViz.plot_models_comparison(
            results=results,
            metric_name=metric_name
        )
        
        for trace in models_fig.data:
            fig.add_trace(trace, row=1, col=1)
        
        # 2. Robustness Index (middle row, left)
        indices_fig = RobustnessViz.plot_robustness_index(
            results=results,
            robustness_indices=robustness_indices
        )
        
        for trace in indices_fig.data:
            fig.add_trace(trace, row=2, col=1)
        
        # 3. Feature Importance (middle row, right)
        # Find best model
        best_model = max(robustness_indices, key=robustness_indices.get)
        
        boxplot_fig = RobustnessViz.plot_boxplot_performance(
            results=results,
            model_name=best_model,
            metric_name=metric_name
        )
        
        for trace in boxplot_fig.data:
            fig.add_trace(trace, row=2, col=2)
        
        # 4. Robustness Radar (bottom row, left)
        # Get top 5 features
        top_features = feature_importance_results['sorted_features'][:5]
        feature_impacts = feature_importance_results['sorted_impacts'][:5]
        feature_importance_dict = {f: i for f, i in zip(top_features, feature_impacts)}
        
        radar_fig = RobustnessViz.plot_robustness_radar(
            robustness_indices=robustness_indices,
            feature_importance=feature_importance_dict
        )
        
        for trace in radar_fig.data:
            fig.add_trace(trace, row=3, col=1)
        
        # 5. Feature Importance Table (bottom row, right)
        top_features = feature_importance_results['sorted_features'][:8]
        feature_impacts = feature_importance_results['sorted_impacts'][:8]
        
        # Add table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Feature', 'Impact'],
                    fill_color='rgba(52, 152, 219, 0.8)',
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[
                        top_features,
                        [f"{impact:.3f}" for impact in feature_impacts]
                    ],
                    fill_color=[
                        'rgba(255, 255, 255, 0.8)',
                        [
                            'rgba(231, 76, 60, 0.8)' if impact > 0 else 'rgba(46, 204, 113, 0.8)' 
                            for impact in feature_impacts
                        ]
                    ],
                    align='left',
                    font=dict(color='black', size=11)
                )
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"Comprehensive Robustness Analysis Dashboard{' - ' + metric_name if metric_name else ''}",
                y=0.99,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=20)
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=height,
            width=width
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Perturbation Size", row=1, col=1)
        fig.update_yaxes(title_text=metric_name if metric_name else "Performance Metric", row=1, col=1)
        
        fig.update_xaxes(title_text="Robustness Index", row=2, col=1)
        fig.update_yaxes(title_text="Models", row=2, col=1)
        
        fig.update_xaxes(title_text="Perturbation Size", row=2, col=2)
        fig.update_yaxes(title_text=metric_name if metric_name else "Performance Metric", row=2, col=2)
        
        # Apply consistent styling across all subplots
        for i in range(1, 4):
            for j in range(1, 3):
                if not (i == 3 and j == 1):  # Skip polar chart
                    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(230,230,230,0.6)', row=i, col=j)
                    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(230,230,230,0.6)', row=i, col=j)
        
        return fig