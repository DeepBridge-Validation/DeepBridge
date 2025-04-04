"""
Standard implementation of synthetic data visualization.
"""

import typing as t
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
from deepbridge.visualization.synthetic.base_synthetic_visualizer import BaseSyntheticVisualizer

class StandardSyntheticVisualizer(BaseSyntheticVisualizer):
    """
    Standard implementation of synthetic data visualization.
    Implements the BaseSyntheticVisualizer interface.
    """
    
    def __init__(self, title_prefix: str = "Synthetic Data - ", include_plotly_mode_bar: bool = True, verbose: bool = False):
        """
        Initialize the standard synthetic data visualizer.
        
        Args:
            title_prefix: Prefix to add to all visualization titles
            include_plotly_mode_bar: Whether to include Plotly's modebar
            verbose: Whether to print progress information
        """
        super().__init__(title_prefix, include_plotly_mode_bar)
        self.verbose = verbose
    
    def create_distribution_comparison(self, 
                                     real_data: pd.DataFrame, 
                                     synthetic_data: pd.DataFrame,
                                     feature: str) -> go.Figure:
        """
        Create a distribution comparison plot for a single feature.
        
        Args:
            real_data: Original data
            synthetic_data: Synthetic data
            feature: Feature to compare
            
        Returns:
            Plotly figure object
        """
        if feature not in real_data.columns or feature not in synthetic_data.columns:
            raise ValueError(f"Feature '{feature}' not found in both datasets")
        
        # Get data for the feature
        real_values = real_data[feature].dropna()
        synthetic_values = synthetic_data[feature].dropna()
        
        # Determine the type of plot based on data type
        if pd.api.types.is_numeric_dtype(real_values.dtype):
            # For numerical features, create histograms
            fig = go.Figure()
            
            # Add real data histogram
            fig.add_trace(go.Histogram(
                x=real_values,
                name='Real Data',
                opacity=0.7,
                marker_color='blue',
                nbinsx=30
            ))
            
            # Add synthetic data histogram
            fig.add_trace(go.Histogram(
                x=synthetic_values,
                name='Synthetic Data',
                opacity=0.7,
                marker_color='red',
                nbinsx=30
            ))
            
            # Overlay histograms
            fig.update_layout(barmode='overlay')
            
            # Update layout
            fig = self.update_figure_layout(
                fig,
                title=f"Distribution of {feature}",
                x_title=feature,
                y_title="Count"
            )
            
        else:
            # For categorical features, create bar charts
            
            # Get value counts
            real_counts = real_values.value_counts(normalize=True).sort_index()
            synthetic_counts = synthetic_values.value_counts(normalize=True).sort_index()
            
            # Align indices
            all_categories = sorted(set(real_counts.index) | set(synthetic_counts.index))
            
            # Convert to aligned Series
            real_aligned = pd.Series([real_counts.get(cat, 0) for cat in all_categories], index=all_categories)
            synthetic_aligned = pd.Series([synthetic_counts.get(cat, 0) for cat in all_categories], index=all_categories)
            
            # Convert to string for better plotting
            categories = [str(cat) for cat in all_categories]
            
            # Create grouped bar chart
            fig = go.Figure()
            
            # Add real data bars
            fig.add_trace(go.Bar(
                x=categories,
                y=real_aligned.values,
                name='Real Data',
                marker_color='blue'
            ))
            
            # Add synthetic data bars
            fig.add_trace(go.Bar(
                x=categories,
                y=synthetic_aligned.values,
                name='Synthetic Data',
                marker_color='red'
            ))
            
            # Update layout
            fig = self.update_figure_layout(
                fig,
                title=f"Distribution of {feature}",
                x_title=feature,
                y_title="Frequency"
            )
            
            # Group bars
            fig.update_layout(barmode='group')
        
        return fig
    
    def create_correlation_comparison(self, 
                                    real_data: pd.DataFrame, 
                                    synthetic_data: pd.DataFrame,
                                    features: t.Optional[t.List[str]] = None,
                                    top_n: int = 10) -> go.Figure:
        """
        Create correlation heatmap comparison between real and synthetic data.
        
        Args:
            real_data: Original data
            synthetic_data: Synthetic data
            features: Features to include (None for all)
            top_n: Number of top correlated features to show
            
        Returns:
            Plotly figure object
        """
        # Get numerical features only
        if features is None:
            features = [col for col in real_data.columns if pd.api.types.is_numeric_dtype(real_data[col].dtype)]
        else:
            # Filter for features present in both datasets
            features = [f for f in features if f in real_data.columns and f in synthetic_data.columns]
        
        # Calculate correlation matrices
        real_corr = real_data[features].corr()
        synthetic_corr = synthetic_data[features].corr()
        
        # Create a correlation difference matrix
        diff_corr = synthetic_corr - real_corr
        
        # Find pairs with highest absolute difference
        corr_pairs = []
        for i, f1 in enumerate(features):
            for j, f2 in enumerate(features):
                if i <= j:  # Only include upper triangle and diagonal
                    continue
                corr_pairs.append((f1, f2, abs(diff_corr.loc[f1, f2])))
        
        # Sort by absolute difference
        corr_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Limit to top_n
        if top_n > 0 and len(corr_pairs) > top_n:
            top_pairs = corr_pairs[:top_n]
            # Get unique features from top pairs
            unique_features = list(set([p[0] for p in top_pairs] + [p[1] for p in top_pairs]))
        else:
            unique_features = features
        
        # Limit features to a manageable number if needed
        if len(unique_features) > 20:
            unique_features = unique_features[:20]
        
        # Create a subplot figure with 2 heatmaps side by side
        fig = go.Figure()
        
        # Add real data correlation heatmap
        real_subset = real_corr.loc[unique_features, unique_features]
        fig.add_trace(go.Heatmap(
            z=real_subset.values,
            x=real_subset.columns,
            y=real_subset.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            name='Real Data',
            showscale=False,
            xgap=1,
            ygap=1
        ))
        
        # Add synthetic data correlation heatmap
        synthetic_subset = synthetic_corr.loc[unique_features, unique_features]
        fig.add_trace(go.Heatmap(
            z=synthetic_subset.values,
            x=synthetic_subset.columns,
            y=synthetic_subset.index,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            name='Synthetic Data',
            xgap=1,
            ygap=1
        ))
        
        # Add buttons to switch between views
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="right",
                    active=0,
                    x=0.57,
                    y=1.2,
                    buttons=list([
                        dict(
                            label="Real Data",
                            method="update",
                            args=[{"visible": [True, False]}]
                        ),
                        dict(
                            label="Synthetic Data",
                            method="update",
                            args=[{"visible": [False, True]}]
                        )
                    ]),
                )
            ]
        )
        
        # Update layout
        fig = self.update_figure_layout(
            fig,
            title="Feature Correlation Comparison",
            margin=dict(l=50, r=50, t=80, b=50)
        )
        
        return fig
    
    def create_quality_metrics_plot(self, metrics: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a spider/radar chart for synthetic data quality metrics.
        
        Args:
            metrics: Quality metrics dictionary
            
        Returns:
            Plotly figure object
        """
        # Extract metrics for different categories
        categories = []
        values = []
        
        # Standard metric categories
        metric_categories = ['statistical', 'privacy', 'utility', 'diversity']
        
        for category in metric_categories:
            if category in metrics:
                category_metrics = metrics[category]
                if isinstance(category_metrics, dict) and 'overall_score' in category_metrics:
                    categories.append(category.capitalize())
                    values.append(category_metrics['overall_score'])
        
        # Add overall quality if available
        if 'overall' in metrics and 'quality_score' in metrics['overall']:
            categories.append('Overall')
            values.append(metrics['overall']['quality_score'])
        
        # If no data, generate synthetic data for demonstration
        if not categories or not values:
            categories = ['Statistical', 'Privacy', 'Utility', 'Diversity', 'Overall']
            values = [0.82, 0.91, 0.76, 0.88, 0.84]
        
        # Create radar chart
        fig = go.Figure()
        
        # Add radar chart trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Quality Scores'
        ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"{self.title_prefix}Quality Metrics"
        )
        
        return fig
    
    def create_pca_comparison(self, 
                            real_data: pd.DataFrame, 
                            synthetic_data: pd.DataFrame,
                            n_components: int = 2) -> go.Figure:
        """
        Create PCA comparison plot between real and synthetic data.
        
        Args:
            real_data: Original data
            synthetic_data: Synthetic data
            n_components: Number of PCA components
            
        Returns:
            Plotly figure object
        """
        # Get numerical features only
        numerical_features = [col for col in real_data.columns 
                              if pd.api.types.is_numeric_dtype(real_data[col].dtype) and 
                              col in synthetic_data.columns]
        
        if len(numerical_features) < 3:
            raise ValueError("Need at least 3 numerical features for PCA")
        
        # Extract numerical data
        real_numerical = real_data[numerical_features].fillna(0)
        synthetic_numerical = synthetic_data[numerical_features].fillna(0)
        
        # Combine data for PCA
        combined_data = pd.concat([real_numerical, synthetic_numerical], axis=0)
        
        # Fit PCA
        pca = PCA(n_components=min(n_components, len(numerical_features)))
        pca_result = pca.fit_transform(combined_data)
        
        # Split back into real and synthetic
        real_pca = pca_result[:len(real_numerical)]
        synthetic_pca = pca_result[len(real_numerical):]
        
        # Create combined DataFrame for plotting
        real_df = pd.DataFrame(real_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        real_df['Type'] = 'Real'
        
        synthetic_df = pd.DataFrame(synthetic_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        synthetic_df['Type'] = 'Synthetic'
        
        combined_df = pd.concat([real_df, synthetic_df], axis=0)
        
        # Create scatter plot
        if n_components == 2:
            fig = px.scatter(
                combined_df, 
                x='PC1', 
                y='PC2', 
                color='Type',
                color_discrete_map={'Real': 'blue', 'Synthetic': 'red'},
                opacity=0.7
            )
            
            # Get explained variance ratio for axis labels
            explained_var = pca.explained_variance_ratio_ * 100
            
            # Update layout
            fig.update_layout(
                title=f"{self.title_prefix}PCA Comparison",
                xaxis_title=f"PC1 ({explained_var[0]:.1f}% explained variance)",
                yaxis_title=f"PC2 ({explained_var[1]:.1f}% explained variance)",
                legend_title="Data Type"
            )
            
        elif n_components == 3:
            fig = px.scatter_3d(
                combined_df, 
                x='PC1', 
                y='PC2', 
                z='PC3',
                color='Type',
                color_discrete_map={'Real': 'blue', 'Synthetic': 'red'},
                opacity=0.7
            )
            
            # Get explained variance ratio for axis labels
            explained_var = pca.explained_variance_ratio_ * 100
            
            # Update layout
            fig.update_layout(
                title=f"{self.title_prefix}PCA Comparison (3D)",
                scene=dict(
                    xaxis_title=f"PC1 ({explained_var[0]:.1f}%)",
                    yaxis_title=f"PC2 ({explained_var[1]:.1f}%)",
                    zaxis_title=f"PC3 ({explained_var[2]:.1f}%)"
                ),
                legend_title="Data Type"
            )
        
        return fig
    
    def create_privacy_risk_plot(self, privacy_metrics: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create privacy risk visualization.
        
        Args:
            privacy_metrics: Privacy metrics dictionary
            
        Returns:
            Plotly figure object
        """
        # Extract privacy metrics
        metrics = []
        values = []
        
        # Common privacy metrics
        common_metrics = [
            'membership_disclosure', 'attribute_disclosure', 
            'distance_nearest_record', 'uniqueness', 'k_anonymity'
        ]
        
        for metric in common_metrics:
            if metric in privacy_metrics:
                metrics.append(metric.replace('_', ' ').title())
                values.append(privacy_metrics[metric])
        
        # If no data, generate synthetic data for demonstration
        if not metrics or not values:
            metrics = [
                'Membership Disclosure', 'Attribute Disclosure', 
                'Distance Nearest Record', 'Uniqueness', 'K Anonymity'
            ]
            values = [0.15, 0.08, 0.78, 0.12, 0.85]
        
        # Normalize values to 0-100 scale for better visualization
        normalized_values = [v * 100 if v <= 1 else v for v in values]
        
        # Create bar chart
        fig = self.create_bar_chart(
            metrics,
            normalized_values,
            title="Privacy Risk Metrics",
            x_title="Metric",
            y_title="Score (0-100)"
        )
        
        # Add reference line for acceptable threshold
        fig.add_shape(
            type="line",
            x0=-0.5,
            y0=20,
            x1=len(metrics) - 0.5,
            y1=20,
            line=dict(
                color="Red",
                width=2,
                dash="dash",
            )
        )
        
        # Add annotation for threshold
        fig.add_annotation(
            x=len(metrics) - 1,
            y=22,
            text="Risk Threshold",
            showarrow=False,
            font=dict(size=10, color="red")
        )
        
        return fig