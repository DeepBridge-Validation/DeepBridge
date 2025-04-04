"""
Performance visualization for synthetic data.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import typing as t
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from deepbridge.visualization.synthetic.base_synthetic_visualizer import BaseSyntheticVisualizer

class PerformanceViz(BaseSyntheticVisualizer):
    """
    Visualizations for comparing model performances between real and synthetic data.
    """
    
    def __init__(self, title_prefix: str = "Synthetic Data - ", include_plotly_mode_bar: bool = True):
        """
        Initialize the performance visualizer.
        
        Args:
            title_prefix: Prefix to add to all visualization titles
            include_plotly_mode_bar: Whether to include Plotly's modebar
        """
        super().__init__(title_prefix, include_plotly_mode_bar)
    
    def create_distribution_comparison(self, 
                                     real_data: pd.DataFrame, 
                                     synthetic_data: pd.DataFrame,
                                     feature: str) -> go.Figure:
        """
        Create a distribution comparison plot for model predictions.
        
        Args:
            real_data: Original data with predictions
            synthetic_data: Synthetic data with predictions
            feature: Prediction column to compare
            
        Returns:
            Plotly figure object
        """
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add histograms
        real_color = 'rgba(0, 123, 255, 0.5)'
        synth_color = 'rgba(220, 53, 69, 0.5)'
        
        # Add histogram for real data
        fig.add_trace(
            go.Histogram(
                x=real_data[feature],
                name="Real Data",
                opacity=0.5,
                marker_color=real_color,
                nbinsx=30,
                histnorm='probability'
            )
        )
        
        # Add histogram for synthetic data
        fig.add_trace(
            go.Histogram(
                x=synthetic_data[feature],
                name="Synthetic Data",
                opacity=0.5,
                marker_color=synth_color,
                nbinsx=30,
                histnorm='probability'
            )
        )
        
        # Add KDE plots
        from scipy.stats import gaussian_kde
        
        # Create x range for KDE
        min_val = min(real_data[feature].min(), synthetic_data[feature].min())
        max_val = max(real_data[feature].max(), synthetic_data[feature].max())
        x_range = np.linspace(min_val, max_val, 1000)
        
        # KDE for real data
        try:
            real_kde = gaussian_kde(real_data[feature].dropna())
            real_kde_y = real_kde(x_range)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=real_kde_y,
                    mode='lines',
                    name="Real KDE",
                    line=dict(color='rgb(0, 123, 255)'),
                ),
                secondary_y=True
            )
        except:
            # Skip KDE if it fails (e.g., not enough data)
            pass
        
        # KDE for synthetic data
        try:
            synth_kde = gaussian_kde(synthetic_data[feature].dropna())
            synth_kde_y = synth_kde(x_range)
            
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=synth_kde_y,
                    mode='lines',
                    name="Synthetic KDE",
                    line=dict(color='rgb(220, 53, 69)'),
                ),
                secondary_y=True
            )
        except:
            # Skip KDE if it fails
            pass
        
        # Add statistics to the plot
        real_mean = real_data[feature].mean()
        synth_mean = synthetic_data[feature].mean()
        real_std = real_data[feature].std()
        synth_std = synthetic_data[feature].std()
        
        stats_text = (
            f"Real mean: {real_mean:.4f}, std: {real_std:.4f}<br>"
            f"Synthetic mean: {synth_mean:.4f}, std: {synth_std:.4f}<br>"
            f"Mean diff: {abs(real_mean - synth_mean):.4f} ({abs(real_mean - synth_mean) / abs(real_mean) * 100:.2f}%)<br>"
            f"Std diff: {abs(real_std - synth_std):.4f} ({abs(real_std - synth_std) / abs(real_std) * 100:.2f}%)"
        )
        
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            font=dict(size=10),
            align="right",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Distribution of {feature}",
            xaxis_title=feature,
            yaxis_title="Probability",
            yaxis2_title="Density",
            barmode='overlay',
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(0, 0, 0, 0.1)'
            ),
            margin=dict(l=40, r=40, t=40, b=40),
        )
        
        return fig
    
    def create_correlation_comparison(self, 
                                    real_data: pd.DataFrame, 
                                    synthetic_data: pd.DataFrame,
                                    features: t.Optional[t.List[str]] = None,
                                    top_n: int = 10) -> go.Figure:
        """
        Create comparison of feature importance across models.
        
        Args:
            real_data: Original data with feature importances
            synthetic_data: Synthetic data with feature importances
            features: Features to include (None for all)
            top_n: Number of top features to show
            
        Returns:
            Plotly figure object
        """
        # If features not specified, assume columns are feature names with importances as values
        if features is None:
            features = list(set(real_data.columns) & set(synthetic_data.columns))
            
        # Filter to top features
        if len(features) > top_n:
            # Get top features by average importance
            avg_importance = (real_data[features].mean() + synthetic_data[features].mean()) / 2
            top_features = avg_importance.nlargest(top_n).index.tolist()
        else:
            top_features = features
            
        # Create a DataFrame for plotting
        plot_data = pd.DataFrame({
            'Feature': top_features,
            'Real': [real_data[f].mean() for f in top_features],
            'Synthetic': [synthetic_data[f].mean() for f in top_features]
        })
        
        # Sort by real importance
        plot_data = plot_data.sort_values('Real', ascending=True)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces
        fig.add_trace(go.Bar(
            y=plot_data['Feature'],
            x=plot_data['Real'],
            name='Real Data',
            orientation='h',
            marker_color='rgba(0, 123, 255, 0.7)'
        ))
        
        fig.add_trace(go.Bar(
            y=plot_data['Feature'],
            x=plot_data['Synthetic'],
            name='Synthetic Data',
            orientation='h',
            marker_color='rgba(220, 53, 69, 0.7)'
        ))
        
        # Calculate correlation between feature importances
        import scipy.stats
        corr, p_value = scipy.stats.pearsonr(plot_data['Real'], plot_data['Synthetic'])
        
        # Add annotation
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Correlation: {corr:.4f}<br>p-value: {p_value:.4f}",
            showarrow=False,
            font=dict(size=10),
            align="right",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Feature Importance Comparison",
            xaxis_title="Importance",
            barmode='group',
            legend=dict(
                x=0.7,
                y=1.1,
                orientation='h',
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(0, 0, 0, 0.1)'
            ),
            margin=dict(l=150, r=40, t=80, b=40),
            height=500,
            width=900
        )
        
        return fig
    
    def create_quality_metrics_plot(self, metrics: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a bar chart for model performance metrics.
        
        Args:
            metrics: Model performance metrics dictionary
            
        Returns:
            Plotly figure object
        """
        # Extract performance metrics
        metric_names = []
        real_values = []
        synth_values = []
        
        # Organize metrics
        for metric_name, values in metrics.items():
            if isinstance(values, dict) and 'real' in values and 'synthetic' in values:
                metric_names.append(metric_name)
                real_values.append(values['real'])
                synth_values.append(values['synthetic'])
        
        # Check if we have data to plot
        if not metric_names:
            raise ValueError("No valid metrics found for performance plot")
            
        # Create bar chart
        fig = go.Figure()
        
        # Add bars for real data
        fig.add_trace(go.Bar(
            x=metric_names,
            y=real_values,
            name='Real Data',
            marker_color='rgba(0, 123, 255, 0.7)'
        ))
        
        # Add bars for synthetic data
        fig.add_trace(go.Bar(
            x=metric_names,
            y=synth_values,
            name='Synthetic Data',
            marker_color='rgba(220, 53, 69, 0.7)'
        ))
        
        # Add delta markers
        for i, (name, real, synth) in enumerate(zip(metric_names, real_values, synth_values)):
            # Calculate percentage difference
            if real != 0:
                pct_diff = (synth - real) / abs(real) * 100
                
                # Determine color based on difference
                color = 'green' if pct_diff >= 0 else 'red'
                
                # Add annotation
                fig.add_annotation(
                    x=name,
                    y=max(real, synth) + 0.05,
                    text=f"{pct_diff:+.1f}%",
                    showarrow=False,
                    font=dict(
                        size=10,
                        color=color
                    )
                )
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Model Performance Metrics",
            yaxis_title="Score",
            barmode='group',
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(0, 0, 0, 0.1)'
            ),
            margin=dict(l=40, r=40, t=80, b=40),
        )
        
        return fig
    
    def create_pca_comparison(self, 
                            real_data: pd.DataFrame, 
                            synthetic_data: pd.DataFrame,
                            n_components: int = 2) -> go.Figure:
        """
        Create comparison of model prediction distributions using dimensionality reduction.
        
        Args:
            real_data: Original data with model outputs
            synthetic_data: Synthetic data with model outputs
            n_components: Number of components to use
            
        Returns:
            Plotly figure object
        """
        # This is a simplified version that treats the data as general prediction data
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Assume all columns are model predictions
        # Combine data for scaling
        combined = pd.concat([real_data, synthetic_data])
        
        # Standardize
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        projections = pca.fit_transform(combined_scaled)
        
        # Split back
        real_proj = projections[:len(real_data)]
        synth_proj = projections[len(real_data):]
        
        # Create plotting data
        if n_components == 2:
            # Create 2D scatter plot
            fig = go.Figure()
            
            # Add real data
            fig.add_trace(go.Scatter(
                x=real_proj[:, 0],
                y=real_proj[:, 1],
                mode='markers',
                name='Real Data',
                marker=dict(
                    color='blue',
                    opacity=0.6,
                    size=8
                )
            ))
            
            # Add synthetic data
            fig.add_trace(go.Scatter(
                x=synth_proj[:, 0],
                y=synth_proj[:, 1],
                mode='markers',
                name='Synthetic Data',
                marker=dict(
                    color='red',
                    opacity=0.6,
                    size=8
                )
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{self.title_prefix}PCA of Model Predictions",
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bgcolor='rgba(255, 255, 255, 0.5)',
                    bordercolor='rgba(0, 0, 0, 0.1)'
                ),
                margin=dict(l=40, r=40, t=80, b=40),
            )
            
        else:
            # Create 3D scatter plot
            fig = go.Figure()
            
            # Add real data
            fig.add_trace(go.Scatter3d(
                x=real_proj[:, 0],
                y=real_proj[:, 1],
                z=real_proj[:, 2],
                mode='markers',
                name='Real Data',
                marker=dict(
                    color='blue',
                    opacity=0.6,
                    size=4
                )
            ))
            
            # Add synthetic data
            fig.add_trace(go.Scatter3d(
                x=synth_proj[:, 0],
                y=synth_proj[:, 1],
                z=synth_proj[:, 2],
                mode='markers',
                name='Synthetic Data',
                marker=dict(
                    color='red',
                    opacity=0.6,
                    size=4
                )
            ))
            
            # Update layout
            fig.update_layout(
                title=f"{self.title_prefix}3D PCA of Model Predictions",
                scene=dict(
                    xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%})",
                    yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%})",
                    zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%})"
                ),
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bgcolor='rgba(255, 255, 255, 0.5)',
                    bordercolor='rgba(0, 0, 0, 0.1)'
                )
            )
            
        return fig
    
    def create_privacy_risk_plot(self, privacy_metrics: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create ROC and PR curve comparison plots.
        
        Args:
            privacy_metrics: Dictionary with prediction metrics
                Must contain 'real' and 'synthetic' keys, each with
                'y_true' and 'y_pred' keys for actual and predicted values
            
        Returns:
            Plotly figure object
        """
        # Extract prediction data
        real_y_true = privacy_metrics.get('real', {}).get('y_true')
        real_y_pred = privacy_metrics.get('real', {}).get('y_pred')
        synthetic_y_true = privacy_metrics.get('synthetic', {}).get('y_true')
        synthetic_y_pred = privacy_metrics.get('synthetic', {}).get('y_pred')
        
        # Check if we have all required data
        if not all([
            isinstance(real_y_true, (list, np.ndarray, pd.Series)),
            isinstance(real_y_pred, (list, np.ndarray, pd.Series)),
            isinstance(synthetic_y_true, (list, np.ndarray, pd.Series)),
            isinstance(synthetic_y_pred, (list, np.ndarray, pd.Series))
        ]):
            raise ValueError("Missing required prediction data in metrics dictionary")
            
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('ROC Curve', 'Precision-Recall Curve'),
            horizontal_spacing=0.1
        )
        
        # Calculate ROC curves
        real_fpr, real_tpr, _ = roc_curve(real_y_true, real_y_pred)
        real_roc_auc = auc(real_fpr, real_tpr)
        
        synth_fpr, synth_tpr, _ = roc_curve(synthetic_y_true, synthetic_y_pred)
        synth_roc_auc = auc(synth_fpr, synth_tpr)
        
        # Calculate PR curves
        real_precision, real_recall, _ = precision_recall_curve(real_y_true, real_y_pred)
        real_pr_auc = auc(real_recall, real_precision)
        
        synth_precision, synth_recall, _ = precision_recall_curve(synthetic_y_true, synthetic_y_pred)
        synth_pr_auc = auc(synth_recall, synth_precision)
        
        # Add ROC curves
        fig.add_trace(
            go.Scatter(
                x=real_fpr,
                y=real_tpr,
                name=f"Real (AUC={real_roc_auc:.3f})",
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=synth_fpr,
                y=synth_tpr,
                name=f"Synthetic (AUC={synth_roc_auc:.3f})",
                line=dict(color='red', width=2)
            ),
            row=1, col=1
        )
        
        # Add diagonal reference line for ROC
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(color='black', dash='dash', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Add PR curves
        fig.add_trace(
            go.Scatter(
                x=real_recall,
                y=real_precision,
                name=f"Real (AUC={real_pr_auc:.3f})",
                line=dict(color='blue', width=2)
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=synth_recall,
                y=synth_precision,
                name=f"Synthetic (AUC={synth_pr_auc:.3f})",
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        
        # Add AUC comparison
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"ROC AUC Difference: {abs(real_roc_auc - synth_roc_auc):.3f} | PR AUC Difference: {abs(real_pr_auc - synth_pr_auc):.3f}",
            showarrow=False,
            font=dict(size=12)
        )
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Model Performance Curves",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=500,
            width=1000,
            margin=dict(l=40, r=40, t=80, b=40),
        )
        
        return fig
