import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional

from deepbridge.validation.robustness_metrics import is_metric_higher_better
from deepbridge.visualization.robustness.base_viz import RobustnessBaseViz

class FeatureViz(RobustnessBaseViz):
    """Visualizations related to feature importance and impact."""
    
    @staticmethod
    def plot_feature_importance(
        feature_importance_results: Dict, 
        title: Optional[str] = None,
        top_n: int = 7, 
        height: Optional[int] = None,
        width: Optional[int] = None
    ):
        """
        Create an interactive horizontal bar chart of feature importance.
        
        Args:
            feature_importance_results: Results from RobustnessTest.analyze_feature_importance
            title: Custom title for the plot
            top_n: Number of top features to show
            height: Height of the plot (now optional)
            width: Width of the plot (now optional)
            
        Returns:
            Plotly figure with the feature importance visualization
        """
        # Extract data
        sorted_features = feature_importance_results['sorted_features'][:top_n]
        sorted_impacts = feature_importance_results['sorted_impacts'][:top_n]
        metric = feature_importance_results.get('metric', 'Unknown Metric')
        
        # Reverse order for better visual hierarchy (most important at top)
        sorted_features = sorted_features[::-1]
        sorted_impacts = sorted_impacts[::-1]
        
        # Create a gradient color scale based on impact values
        colors = []
        for impact in sorted_impacts:
            if impact > 0:
                # Red for positive impact (reduces robustness)
                intensity = min(abs(impact), 1.0)
                colors.append(f'rgba(255, {int(255 * (1 - intensity))}, {int(255 * (1 - intensity))}, 0.8)')
            else:
                # Green for negative impact (improves robustness)
                intensity = min(abs(impact), 1.0)
                colors.append(f'rgba({int(255 * (1 - intensity))}, 255, {int(255 * (1 - intensity))}, 0.8)')
        
        # Create figure
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            y=sorted_features,
            x=sorted_impacts,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(
                    color='rgba(50, 50, 50, 0.2)',
                    width=1
                )
            ),
            hovertemplate='Feature: %{y}<br>Impact on Robustness: %{x:.4f}<extra></extra>'
        ))
        
        # Add a vertical line at x=0
        fig.add_shape(
            type='line',
            x0=0, x1=0,
            y0=-0.5, y1=len(sorted_features) - 0.5,
            line=dict(
                color='black',
                width=1.5,  # Thinner line
                dash='solid'
            )
        )
        
        # Add annotations with impact values
        for i, (feature, impact) in enumerate(zip(sorted_features, sorted_impacts)):
            text_color = 'white' if abs(impact) > 0.5 else 'black'
            text_pos = 'inside' if abs(impact) > 0.1 else 'outside'
            
            fig.add_annotation(
                x=impact + (0.02 * np.sign(impact)) if text_pos == 'outside' else impact,
                y=i,
                text=f"{impact:.3f}",
                showarrow=False,
                font=dict(
                    color=text_color,
                    size=9  # Smaller font for impact values
                ),
                xanchor='left' if impact >= 0 else 'right',
                yanchor='middle'
            )
        
        # Set title
        if title:
            plot_title = title
        else:
            plot_title = 'Feature Impact on Model Robustness'
        
        # Check if it's classification or regression based on metric
        higher_is_better = is_metric_higher_better(metric)
        if higher_is_better:
            explanation = "Negative values (green) indicate features that improve robustness when perturbed.<br>Positive values (red) indicate features that reduce robustness when perturbed."
        else:
            explanation = "Negative values (green) indicate features that reduce error when perturbed.<br>Positive values (red) indicate features that increase error when perturbed."
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=plot_title,
                x=0.5,
                y=0.97,
                font=dict(size=16)  # Smaller title font
            ),
            xaxis=dict(
                title=dict(
                    text='Normalized Impact on Robustness',
                    font=dict(size=12)  # Smaller axis title font
                ),
                zeroline=False,
                gridcolor='rgba(230,230,230,0.5)'
            ),
            yaxis=dict(
                title=dict(
                    text='Features',
                    font=dict(size=12)  # Smaller axis title font
                ),
                automargin=True
            ),
            plot_bgcolor='white',
            margin=dict(l=20, r=20, t=60, b=80),  # Reduced margins
            autosize=True,  # Enable autosize to fit container
            annotations=[
                dict(
                    text=explanation,
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=-0.2,  # Adjusted position for explanation
                    showarrow=False,
                    font=dict(size=10),  # Smaller font for explanation
                    align="center",
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.1)",
                    borderwidth=1,
                    borderpad=4  # Reduced padding
                )
            ]
        )
        
        # Add height and width only if specified
        if height:
            fig.update_layout(height=height)
        if width:
            fig.update_layout(width=width)
        
        return fig

    @staticmethod
    def plot_feature_importance_multiple(
        feature_importance_results_dict,
        title_prefix="Feature Impact on Model Robustness - ",
        top_n=7,
        height=None,
        width=None,
        subplot_layout=False
    ):
        """
        Create multiple feature importance visualizations, one for each model.
        """
        
        # Check if we have multiple models
        if not feature_importance_results_dict or not isinstance(feature_importance_results_dict, dict):
            raise ValueError("Expected a dictionary with feature importance results for multiple models")
        
        if subplot_layout:
            # Create a figure with subplots
            n_models = len(feature_importance_results_dict)
            fig = make_subplots(
                rows=n_models, 
                cols=1,
                subplot_titles=[f"{title_prefix}{model_name}" for model_name in feature_importance_results_dict.keys()],
                vertical_spacing=0.15
            )
            
            # Para cada modelo, criar um gráfico separado e depois extrair os dados
            for i, (model_name, feature_results) in enumerate(feature_importance_results_dict.items(), start=1):
                # Extrair dados diretamente
                sorted_features = feature_results['sorted_features'][:top_n]
                sorted_impacts = feature_results['sorted_impacts'][:top_n]
                metric = feature_results.get('metric', 'Unknown Metric')
                
                # Reverse order for better visual hierarchy
                sorted_features = sorted_features[::-1]
                sorted_impacts = sorted_impacts[::-1]
                
                # Create a gradient color scale
                colors = []
                for impact in sorted_impacts:
                    if impact > 0:
                        # Red for positive impact (reduces robustness)
                        intensity = min(abs(impact), 1.0)
                        colors.append(f'rgba(255, {int(255 * (1 - intensity))}, {int(255 * (1 - intensity))}, 0.8)')
                    else:
                        # Green for negative impact (improves robustness)
                        intensity = min(abs(impact), 1.0)
                        colors.append(f'rgba({int(255 * (1 - intensity))}, 255, {int(255 * (1 - intensity))}, 0.8)')
                
                # Add bars to the subplot
                fig.add_trace(
                    go.Bar(
                        y=sorted_features,
                        x=sorted_impacts,
                        orientation='h',
                        marker=dict(
                            color=colors,
                            line=dict(
                                color='rgba(50, 50, 50, 0.2)',
                                width=1
                            )
                        ),
                        hovertemplate='Feature: %{y}<br>Impact on Robustness: %{x:.4f}<extra></extra>'
                    ),
                    row=i, col=1
                )
                
                # Add a vertical line at x=0
                fig.add_shape(
                    type='line',
                    x0=0, x1=0,
                    y0=-0.5, y1=len(sorted_features) - 0.5,
                    line=dict(
                        color='black',
                        width=1.5,
                        dash='solid'
                    ),
                    xref=f'x{i}' if i > 1 else 'x',
                    yref=f'y{i}' if i > 1 else 'y'
                )
                
                # Add annotations with impact values
                for j, (feature, impact) in enumerate(zip(sorted_features, sorted_impacts)):
                    text_color = 'white' if abs(impact) > 0.5 else 'black'
                    text_pos = 'inside' if abs(impact) > 0.1 else 'outside'
                    
                    fig.add_annotation(
                        x=impact + (0.02 * np.sign(impact)) if text_pos == 'outside' else impact,
                        y=j,
                        text=f"{impact:.3f}",
                        showarrow=False,
                        font=dict(
                            color=text_color,
                            size=9
                        ),
                        xanchor='left' if impact >= 0 else 'right',
                        yanchor='middle',
                        xref=f'x{i}' if i > 1 else 'x',
                        yref=f'y{i}' if i > 1 else 'y'
                    )
                
                # Set axis titles for each subplot
                fig.update_xaxes(
                    title=dict(
                        text='Normalized Impact on Robustness' if i == n_models else "",
                        font=dict(size=12)
                    ),
                    row=i, col=1
                )
                
                fig.update_yaxes(
                    title=dict(
                        text='Features',
                        font=dict(size=12)
                    ),
                    row=i, col=1
                )
            
            # Set overall title and layout
            fig.update_layout(
                title=dict(
                    text="Feature Impact on Model Robustness - Comparison",
                    x=0.5,
                    y=0.99,
                    font=dict(size=18)
                ),
                plot_bgcolor='white',
                margin=dict(l=20, r=20, t=100, b=100),
                autosize=True,
            )
            
            # Add height and width only if specified
            if height:
                fig.update_layout(height=height * n_models)  # Adjust height based on number of models
            if width:
                fig.update_layout(width=width)
                
            # Add explanation at the bottom
            first_model_results = next(iter(feature_importance_results_dict.values()))
            metric = first_model_results.get('metric', 'Unknown Metric')
            higher_is_better = is_metric_higher_better(metric)
            
            if higher_is_better:
                explanation = "Negative values (green) indicate features that improve robustness when perturbed.<br>Positive values (red) indicate features that reduce robustness when perturbed."
            else:
                explanation = "Negative values (green) indicate features that reduce error when perturbed.<br>Positive values (red) indicate features that increase error when perturbed."
                
            fig.add_annotation(
                text=explanation,
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.05,
                showarrow=False,
                font=dict(size=10),
                align="center",
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="rgba(0, 0, 0, 0.1)",
                borderwidth=1,
                borderpad=4
            )
            
            return fig
            
        else:
            # Create individual figures for each model separately
            figures = {}
            
            for model_name, feature_results in feature_importance_results_dict.items():
                # Extrair dados diretamente
                sorted_features = feature_results['sorted_features'][:top_n]
                sorted_impacts = feature_results['sorted_impacts'][:top_n]
                metric = feature_results.get('metric', 'Unknown Metric')
                
                # Reverse order for better visual hierarchy
                sorted_features = sorted_features[::-1]
                sorted_impacts = sorted_impacts[::-1]
                
                # Create a gradient color scale
                colors = []
                for impact in sorted_impacts:
                    if impact > 0:
                        intensity = min(abs(impact), 1.0)
                        colors.append(f'rgba(255, {int(255 * (1 - intensity))}, {int(255 * (1 - intensity))}, 0.8)')
                    else:
                        intensity = min(abs(impact), 1.0)
                        colors.append(f'rgba({int(255 * (1 - intensity))}, 255, {int(255 * (1 - intensity))}, 0.8)')
                
                # Create a new figure
                fig = go.Figure()
                
                # Add bars
                fig.add_trace(go.Bar(
                    y=sorted_features,
                    x=sorted_impacts,
                    orientation='h',
                    marker=dict(
                        color=colors,
                        line=dict(
                            color='rgba(50, 50, 50, 0.2)',
                            width=1
                        )
                    ),
                    hovertemplate='Feature: %{y}<br>Impact on Robustness: %{x:.4f}<extra></extra>'
                ))
                
                # Add a vertical line at x=0
                fig.add_shape(
                    type='line',
                    x0=0, x1=0,
                    y0=-0.5, y1=len(sorted_features) - 0.5,
                    line=dict(
                        color='black',
                        width=1.5,
                        dash='solid'
                    )
                )
                
                # Add annotations with impact values
                for i, (feature, impact) in enumerate(zip(sorted_features, sorted_impacts)):
                    text_color = 'white' if abs(impact) > 0.5 else 'black'
                    text_pos = 'inside' if abs(impact) > 0.1 else 'outside'
                    
                    fig.add_annotation(
                        x=impact + (0.02 * np.sign(impact)) if text_pos == 'outside' else impact,
                        y=i,
                        text=f"{impact:.3f}",
                        showarrow=False,
                        font=dict(
                            color=text_color,
                            size=9
                        ),
                        xanchor='left' if impact >= 0 else 'right',
                        yanchor='middle'
                    )
                
                # Set title
                plot_title = f"{title_prefix}{model_name}"
                
                # Check if it's classification or regression based on metric
                higher_is_better = is_metric_higher_better(metric)
                if higher_is_better:
                    explanation = "Negative values (green) indicate features that improve robustness when perturbed.<br>Positive values (red) indicate features that reduce robustness when perturbed."
                else:
                    explanation = "Negative values (green) indicate features that reduce error when perturbed.<br>Positive values (red) indicate features that increase error when perturbed."
                
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
                            text='Normalized Impact on Robustness',
                            font=dict(size=12)
                        ),
                        zeroline=False,
                        gridcolor='rgba(230,230,230,0.5)'
                    ),
                    yaxis=dict(
                        title=dict(
                            text='Features',
                            font=dict(size=12)
                        ),
                        automargin=True
                    ),
                    plot_bgcolor='white',
                    margin=dict(l=20, r=20, t=60, b=80),
                    autosize=True,
                    annotations=[
                        dict(
                            text=explanation,
                            xref="paper",
                            yref="paper",
                            x=0.5,
                            y=-0.2,
                            showarrow=False,
                            font=dict(size=10),
                            align="center",
                            bgcolor="rgba(255, 255, 255, 0.8)",
                            bordercolor="rgba(0, 0, 0, 0.1)",
                            borderwidth=1,
                            borderpad=4
                        )
                    ]
                )
                
                # Add height and width only if specified
                if height:
                    fig.update_layout(height=height)
                if width:
                    fig.update_layout(width=width)
                
                figures[model_name] = fig
                
            return figures


    @staticmethod
    def plot_robustness_heatmap(
        results: Dict,
        model_name: str,
        title: Optional[str] = None,
        height: int = 600,
        width: int = 800
    ):
        """
        Create a heatmap showing performance across different features and perturbation sizes.
        
        Args:
            results: Feature-level perturbation results
            model_name: Name of the model to visualize
            title: Custom title for the plot
            height: Height of the plot
            width: Width of the plot
            
        Returns:
            Plotly figure with the heatmap
        """
        # Extract data for the heatmap
        features = []
        perturb_sizes = []
        performance_matrix = []
        
        # Process the results into a matrix format
        for feature, feature_results in results[model_name].items():
            features.append(feature)
            
            if not perturb_sizes:
                perturb_sizes = feature_results['perturb_sizes']
                
            performance_matrix.append(feature_results['mean_scores'])
        
        # Convert to numpy array for the heatmap
        matrix = np.array(performance_matrix)
        
        # Create heatmap using Plotly
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f"{size:.1f}" for size in perturb_sizes],
            y=features,
            colorscale='Viridis',
            hoverongaps=False,
            hovertemplate='Feature: %{y}<br>Perturbation: %{x}<br>Score: %{z:.4f}<extra></extra>',
            text=[[f"{val:.2f}" for val in row] for row in matrix]
        ))
        
        # Add text annotations to each cell
        for i in range(len(features)):
            for j in range(len(perturb_sizes)):
                fig.add_annotation(
                    x=j,
                    y=i,
                    text=f"{matrix[i, j]:.2f}",
                    font=dict(
                        color='white' if matrix[i, j] < 0.7 else 'black',
                        size=9
                    ),
                    showarrow=False,
                    xref='x',
                    yref='y'
                )
        
        # Update layout
        if title:
            plot_title = title
        else:
            plot_title = f"{model_name}: Feature-level Robustness Analysis"
            
        fig.update_layout(
            title=dict(
                text=plot_title,
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=16)
            ),
            xaxis=dict(
                title=dict(
                    text='Perturbation Size',
                    font=dict(size=14)
                ),
                side='bottom'
            ),
            yaxis=dict(
                title=dict(
                    text='Features',
                    font=dict(size=14)
                )
            ),
            height=height,
            width=width,
            coloraxis_colorbar=dict(
                title='Performance Score',
                titleside='right',
                ticks='outside'
            )
        )
        
        return fig        