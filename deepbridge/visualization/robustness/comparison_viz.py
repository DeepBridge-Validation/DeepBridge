import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional

from deepbridge.visualization.robustness.base_viz import RobustnessBaseViz

class ComparisonViz(RobustnessBaseViz):
    """Visualizations related to comparing models and methods."""
    
    @staticmethod
    def plot_models_comparison(
        results: Dict, 
        use_worst: bool = False, 
        alpha: float = 0.3,
        title: Optional[str] = None, 
        metric_name: Optional[str] = None,
        height: Optional[int] = None, 
        width: Optional[int] = None
    ):
        """
        Create an interactive line plot comparing different models.
        
        Args:
            results: Results from RobustnessTest.evaluate_robustness
            use_worst: If True, use worst cases (according to alpha)
            alpha: Proportion of worst cases to consider
            title: Custom title for the plot
            metric_name: Name of the performance metric
            height: Height of the plot (now optional)
            width: Width of the plot (now optional)
            
        Returns:
            Plotly figure with the comparison
        """
        # Create figure
        fig = go.Figure()
        
        # Colors for different models
        colors = px.colors.qualitative.Plotly
        
        # For each model
        for i, (model_name, model_results) in enumerate(results.items()):
            # Choose scores to use (mean or worst)
            if use_worst:
                scores = model_results['worst_scores']
                default_title = f"{int(alpha*100)}%-Worst Sample Performance"
            else:
                scores = model_results['mean_scores']
                default_title = "Model Performance: Perturb on All Features"
            
            # Plot line with points
            perturb_sizes = model_results['perturb_sizes']
            
            # Add main line
            fig.add_trace(go.Scatter(
                x=perturb_sizes,
                y=scores,
                mode='lines+markers',
                name=model_name,
                line=dict(
                    color=colors[i % len(colors)],
                    width=2  # Thinner line
                ),
                marker=dict(
                    color=colors[i % len(colors)],
                    size=6,  # Smaller markers
                    line=dict(
                        color='white',
                        width=1
                    )
                ),
                hovertemplate='Perturbation: %{x:.2f}<br>' +
                            f'{metric_name if metric_name else "Score"}: %{{y:.4f}}<br>' +
                            f'Model: {model_name}<extra></extra>'
            ))
            
            # Add error bands if available and not using worst cases
            if not use_worst and len(model_results['all_scores']) > 0:
                # Calculate standard deviations for error bands
                stds = [np.std(scores_array) for scores_array in model_results['all_scores']]
                upper = [s + std for s, std in zip(scores, stds)]
                lower = [s - std for s, std in zip(scores, stds)]
                
                # Add confidence interval as a filled area
                fig.add_trace(go.Scatter(
                    x=perturb_sizes + perturb_sizes[::-1],
                    y=upper + lower[::-1],
                    fill='toself',
                    fillcolor=f'rgba({px.colors.hex_to_rgb(colors[i % len(colors)])[0]}, '
                               f'{px.colors.hex_to_rgb(colors[i % len(colors)])[1]}, '
                               f'{px.colors.hex_to_rgb(colors[i % len(colors)])[2]}, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    showlegend=False,
                    hoverinfo='none'
                ))
        
        # Set title
        if title:
            plot_title = title
        else:
            plot_title = default_title
        
        # Set y-axis title based on metric
        y_axis_title = metric_name if metric_name else "Performance Metric"
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                y=0.95,
                font=dict(size=16)  # Smaller title font
            ),
            xaxis=dict(
                title=dict(
                    text='Perturbation Size',
                    font=dict(size=12)  # Smaller axis title font
                ),
                gridcolor='rgba(230,230,230,0.5)'
            ),
            yaxis=dict(
                title=dict(
                    text=y_axis_title,
                    font=dict(size=12)  # Smaller axis title font
                ),
                gridcolor='rgba(230,230,230,0.5)'
            ),
            plot_bgcolor='white',
            hovermode='closest',
            margin=dict(l=40, r=20, t=60, b=60),  # Reduced margins
            autosize=True,  # Enable autosize to fit container
            legend=dict(
                title=dict(
                    text='Model',
                    font=dict(size=10)  # Smaller legend title font
                ),
                font=dict(size=9),  # Smaller legend font
                bordercolor='rgba(0, 0, 0, 0.1)',
                borderwidth=1
            )
        )
        
        # Add height and width only if specified
        if height:
            fig.update_layout(height=height)
        if width:
            fig.update_layout(width=width)
        
        # Add a horizontal reference line at baseline performance
        if not use_worst and all(len(model_results['mean_scores']) > 0 for model_results in results.values()):
            baseline_values = [model_results['mean_scores'][0] for model_results in results.values()]
            avg_baseline = sum(baseline_values) / len(baseline_values)
            
            fig.add_shape(
                type="line",
                x0=min(perturb_sizes),
                y0=avg_baseline,
                x1=max(perturb_sizes),
                y1=avg_baseline,
                line=dict(
                    color="gray",
                    width=1,
                    dash="dot",
                ),
            )
            
            fig.add_annotation(
                x=min(perturb_sizes),
                y=avg_baseline,
                text="Average Baseline",
                showarrow=False,
                xshift=-45,  # Reduced offset
                yshift=8,  # Reduced offset
                font=dict(size=9)  # Smaller annotation font
            )
        
        return fig
    
    @staticmethod
    def plot_robustness_index(
        results: Dict,
        robustness_indices: Dict[str, float],
        title: Optional[str] = None,
        height: int = 600,
        width: int = 800
    ):
        """
        Create an interactive bar chart of robustness indices for different models.
        
        Args:
            results: Results from RobustnessTest.evaluate_robustness
            robustness_indices: Robustness indices from RobustnessScore.calculate_robustness_index
            title: Custom title for the plot
            height: Height of the plot
            width: Width of the plot
            
        Returns:
            Plotly figure with robustness indices
        """
        # Extract model names and indices
        models = list(robustness_indices.keys())
        indices = [robustness_indices[model] for model in models]
        
        # Sort by robustness index
        sorted_idx = np.argsort(indices)[::-1]  # Descending
        sorted_models = [models[i] for i in sorted_idx]
        sorted_indices = [indices[i] for i in sorted_idx]
        
        # Create figure with color gradient
        colors = []
        for idx in sorted_indices:
            if idx >= 0.8:
                colors.append('rgba(46, 204, 113, 0.8)')  # Green for excellent robustness
            elif idx >= 0.6:
                colors.append('rgba(52, 152, 219, 0.8)')  # Blue for good robustness  
            elif idx >= 0.4:
                colors.append('rgba(241, 196, 15, 0.8)')  # Yellow for moderate robustness
            else:
                colors.append('rgba(231, 76, 60, 0.8)')   # Red for poor robustness
        
        # Create horizontal bar chart (easier to read model names)
        fig = go.Figure(go.Bar(
            y=sorted_models,
            x=sorted_indices,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(
                    color='rgba(50, 50, 50, 0.2)',
                    width=1
                )
            ),
            text=[f"{idx:.3f}" for idx in sorted_indices],
            textposition='auto',
            hovertemplate='Model: %{y}<br>Robustness Index: %{x:.4f}<extra></extra>'
        ))
        
        # Add reference lines for robustness thresholds
        thresholds = [
            (0.4, 'Poor', 'rgba(231, 76, 60, 0.4)'),
            (0.6, 'Moderate', 'rgba(241, 196, 15, 0.4)'),
            (0.8, 'Good', 'rgba(52, 152, 219, 0.4)')
        ]
        
        for threshold, label, color in thresholds:
            fig.add_shape(
                type="line",
                x0=threshold,
                y0=-0.5,
                x1=threshold,
                y1=len(sorted_models) - 0.5,
                line=dict(
                    color=color,
                    width=2,
                    dash="dash",
                ),
            )
            
            fig.add_annotation(
                x=threshold,
                y=len(sorted_models) - 0.5,
                text=label,
                showarrow=False,
                yshift=10,
                font=dict(
                    size=10,
                    color='rgba(50, 50, 50, 0.7)'
                )
            )
        
        # Set title
        if title:
            plot_title = title
        else:
            plot_title = 'Model Robustness Index Comparison'
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                y=0.97,
                font=dict(size=16)
            ),
            xaxis=dict(
                title=dict(
                    text='Robustness Index (higher is better)',
                    font=dict(size=14)
                ),
                range=[0, 1.05],
                gridcolor='rgba(230,230,230,0.5)'
            ),
            yaxis=dict(
                title=dict(
                    text='Models',
                    font=dict(size=14)
                ),
                automargin=True
            ),
            plot_bgcolor='white',
            height=height,
            width=width,
            annotations=[
                dict(
                    text="Higher values indicate better performance stability under perturbation",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.15,
                    showarrow=False,
                    font=dict(size=12),
                    align="center"
                )
            ]
        )
        
        return fig

    @staticmethod
    def plot_perturbation_methods_comparison(methods_comparison_results: Dict, 
                                        title: Optional[str] = None,
                                        metric_name: Optional[str] = None, 
                                        height: Optional[int] = None,
                                        width: Optional[int] = None):
        """
        Create an interactive plot comparing different perturbation methods.
        
        Args:
            methods_comparison_results: Results from RobustnessTest.compare_perturbation_methods
            title: Custom title for the plot
            metric_name: Name of the performance metric
            height: Height of the plot
            width: Width of the plot
            
        Returns:
            Plotly figure comparing perturbation methods
        """
        # Use ComparisonViz's own plot_models_comparison method instead of RobustnessViz
        fig = ComparisonViz.plot_models_comparison(
            results=methods_comparison_results,
            title=title or "Comparison of Perturbation Methods",
            metric_name=metric_name,
            height=height,
            width=width
        )
        
        # Update legend title to "Perturbation Method"
        fig.update_layout(
            legend=dict(
                title=dict(
                    text="Perturbation Method",
                    font=dict(size=10)
                )
            )
        )
        
        # Add annotations explaining different methods
        method_explanations = {
            "Method: raw": "Raw: Adds Gaussian noise proportional to feature variance",
            "Method: quantile": "Quantile: Transforms to quantile space, adds noise, transforms back",
            "Method: categorical": "Categorical: Randomly samples values based on frequency distribution"
        }
        
        # Find which methods are present in the results
        present_methods = []
        for method_name in method_explanations.keys():
            if method_name in methods_comparison_results:
                present_methods.append(method_name)
        
        # Add explanation text at the bottom
        explanation_text = "<br>".join([method_explanations[method] for method in present_methods])
        
        if explanation_text:
            fig.add_annotation(
                text=explanation_text,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.15,
                showarrow=False,
                font=dict(size=9),
                align="center",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.1)",
                borderwidth=1,
                borderpad=4
            )
            
            # Adjust margins to accommodate the annotation
            fig.update_layout(
                margin=dict(l=40, r=20, t=60, b=130)
            )
        
        return fig