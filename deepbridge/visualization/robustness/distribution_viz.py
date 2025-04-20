import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Optional

from deepbridge.visualization.robustness.base_viz import RobustnessBaseViz

class DistributionViz(RobustnessBaseViz):
    """Visualizations related to distribution analysis and 3D plots."""
    
    @staticmethod
    def plot_3d_robustness_surface(
        results: Dict, 
        title: Optional[str] = None,
        height: int = 800, 
        width: int = 900
    ):
        """
        Create a 3D surface plot showing performance across models, 
        features, and perturbation sizes.
        
        Args:
            results: Comprehensive perturbation results
            title: Custom title for the plot
            height: Height of the plot
            width: Width of the plot
            
        Returns:
            Plotly figure with the 3D surface
        """
        # Create figure
        fig = go.Figure()
        
        # Colors for different models
        colors = px.colors.qualitative.Plotly
        
        # For each model
        for i, (model_name, model_results) in enumerate(results.items()):
            X = model_results['perturb_sizes']
            Y = range(len(model_results['features']))
            X_grid, Y_grid = np.meshgrid(X, Y)
            
            # Create Z values (performance scores)
            Z = np.array(model_results['feature_scores'])
            
            # Plot surface
            fig.add_trace(go.Surface(
                x=X_grid,
                y=Y_grid,
                z=Z,
                name=model_name,
                colorscale='Viridis',
                opacity=0.8,
                showscale=(i == 0),  # Only show colorbar for first model
                contours = {
                    "z": {"show": True, "start": 0.5, "end": 1, "size": 0.05, "width": 2}
                },
                hovertemplate='Perturbation: %{x:.2f}<br>Feature: %{y}<br>Score: %{z:.4f}<extra>' + model_name + '</extra>'
            ))
        
        # Set axis labels
        fig.update_layout(
            scene = dict(
                xaxis_title=dict(
                    text='Perturbation Size',
                    font=dict(size=12)
                ),
                yaxis_title=dict(
                    text='Feature Index',
                    font=dict(size=12)
                ),
                zaxis_title=dict(
                    text='Performance Score',
                    font=dict(size=12)
                ),
                xaxis = dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis = dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)',
                    tickmode='array',
                    tickvals=list(range(len(next(iter(results.values()))['features']))),
                    ticktext=next(iter(results.values()))['features']
                ),
                zaxis = dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                )
            ),
            title = dict(
                text=title if title else '3D Robustness Surface Analysis',
                x=0.5,
                y=0.95,
                font=dict(size=16)
            ),
            height=height,
            width=width,
            margin=dict(l=65, r=50, b=65, t=90),
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(0, 0, 0, 0.1)'
            )
        )
        
        # Add a discrete legend for models
        for i, model_name in enumerate(results.keys()):
            fig.add_trace(go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                name=model_name,
                mode='markers',
                marker=dict(
                    color=colors[i % len(colors)],
                    size=10
                )
            ))
        
        # Set camera position
        fig.update_layout(scene_camera=dict(
            eye=dict(x=1.5, y=1.5, z=1.2)
        ))
        
        return fig  

    @staticmethod
    def plot_robustness_radar(
        robustness_indices: Dict[str, float],
        feature_importance: Dict[str, float],
        title: Optional[str] = None,
        height: int = 700,
        width: int = 700
    ):
        """
        Create a radar chart combining model robustness and feature importance.
        
        Args:
            robustness_indices: Robustness indices from RobustnessScore.calculate_robustness_index
            feature_importance: Feature importance scores
            title: Custom title for the plot
            height: Height of the plot
            width: Width of the plot
            
        Returns:
            Plotly figure with the radar chart
        """
        # Get data for the radar chart
        models = list(robustness_indices.keys())
        
        # Combine robustness and top features
        categories = ['Robustness']
        
        # Add top feature names
        top_features = list(feature_importance.keys())[:5]  # Top 5 features
        categories.extend(top_features)
        
        # Number of variables
        N = len(categories)
        
        # Create figure
        fig = go.Figure()
        
        # Colors for different models
        colors = px.colors.qualitative.Plotly
        
        # Add a trace for each model
        for i, model in enumerate(models):
            values = [robustness_indices[model]]
            
            # Add feature importance values
            for feature in top_features:
                values.append(abs(feature_importance.get(feature, 0)))
            
            # Close the loop for radar chart
            values.append(values[0])
            categories_closed = categories + [categories[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories_closed,
                name=model,
                fill='toself',
                fillcolor=f'rgba({px.colors.hex_to_rgb(colors[i % len(colors)])[0]}, '
                           f'{px.colors.hex_to_rgb(colors[i % len(colors)])[1]}, '
                           f'{px.colors.hex_to_rgb(colors[i % len(colors)])[2]}, 0.2)',
                line=dict(
                    color=colors[i % len(colors)],
                    width=2
                )
            ))
        
        # Add concentric circles as references
        for level in [0.2, 0.4, 0.6, 0.8]:
            circle_values = [level] * (N + 1)
            fig.add_trace(go.Scatterpolar(
                r=circle_values,
                theta=categories + [categories[0]],
                mode='lines',
                line=dict(color='gray', width=0.5),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    showticklabels=False
                ),
                angularaxis=dict(
                    direction='clockwise'
                )
            ),
            title=dict(
                text=title if title else 'Model Robustness Radar',
                x=0.5,
                y=0.95,
                font=dict(size=16)
            ),
            showlegend=True,
            legend=dict(
                x=1.1,
                y=1,
                bordercolor='rgba(0, 0, 0, 0.1)',
                borderwidth=1,
                font=dict(size=12)
            ),
            height=height,
            width=width
        )
        
        return fig      