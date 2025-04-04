"""
Distribution visualization for synthetic data.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import typing as t
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from deepbridge.visualization.synthetic.base_synthetic_visualizer import BaseSyntheticVisualizer

class DistributionViz(BaseSyntheticVisualizer):
    """
    Visualizations for comparing distributions between real and synthetic data.
    """
    
    def __init__(self, title_prefix: str = "Synthetic Data - ", include_plotly_mode_bar: bool = True):
        """
        Initialize the distribution visualizer.
        
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
        Create a distribution comparison plot for a single feature.
        
        Args:
            real_data: Original data
            synthetic_data: Synthetic data
            feature: Feature to compare
            
        Returns:
            Plotly figure object
        """
        # Check if feature exists in both datasets
        if feature not in real_data.columns or feature not in synthetic_data.columns:
            raise ValueError(f"Feature '{feature}' not found in both datasets")
        
        # Determine plot type based on data type
        if pd.api.types.is_numeric_dtype(real_data[feature]):
            return self._create_numerical_distribution(real_data, synthetic_data, feature)
        else:
            return self._create_categorical_distribution(real_data, synthetic_data, feature)
    
    def _create_numerical_distribution(self, 
                                     real_data: pd.DataFrame, 
                                     synthetic_data: pd.DataFrame,
                                     feature: str) -> go.Figure:
        """Create distribution comparison for numerical feature."""
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
        
        # Add Q-Q plot as an inset
        try:
            from scipy.stats import probplot
            
            # Calculate Q-Q data
            real_quantiles = np.sort(real_data[feature].dropna())
            synth_quantiles = np.sort(synthetic_data[feature].dropna())
            
            # Interpolate to match lengths
            if len(real_quantiles) != len(synth_quantiles):
                if len(real_quantiles) > len(synth_quantiles):
                    indices = np.linspace(0, len(real_quantiles)-1, len(synth_quantiles))
                    real_quantiles = np.interp(indices, np.arange(len(real_quantiles)), real_quantiles)
                else:
                    indices = np.linspace(0, len(synth_quantiles)-1, len(real_quantiles))
                    synth_quantiles = np.interp(indices, np.arange(len(synth_quantiles)), synth_quantiles)
            
            # Add Q-Q plot as inset
            fig.add_trace(
                go.Scatter(
                    x=real_quantiles,
                    y=synth_quantiles,
                    mode='markers',
                    name="Q-Q Plot",
                    marker=dict(
                        color='rgba(128, 128, 128, 0.5)',
                        size=4
                    ),
                    xaxis="x2",
                    yaxis="y2"
                )
            )
            
            # Add reference line
            min_val = min(real_quantiles.min(), synth_quantiles.min())
            max_val = max(real_quantiles.max(), synth_quantiles.max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name="Reference Line",
                    line=dict(
                        color='black',
                        dash='dash'
                    ),
                    xaxis="x2",
                    yaxis="y2",
                    showlegend=False
                )
            )
            
            # Add layout for inset
            fig.update_layout(
                xaxis2=dict(
                    domain=[0.65, 0.95],
                    anchor="y2",
                    title="Real Quantiles"
                ),
                yaxis2=dict(
                    domain=[0.15, 0.45],
                    anchor="x2",
                    title="Synthetic Quantiles"
                ),
                annotations=[
                    dict(
                        x=0.8,
                        y=0.45,
                        xref="paper",
                        yref="paper",
                        text="Q-Q Plot",
                        showarrow=False,
                        font=dict(size=12)
                    )
                ]
            )
        except:
            # Skip Q-Q plot if it fails
            pass
        
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
    
    def _create_categorical_distribution(self, 
                                       real_data: pd.DataFrame, 
                                       synthetic_data: pd.DataFrame,
                                       feature: str) -> go.Figure:
        """Create distribution comparison for categorical feature."""
        # Calculate value counts
        real_counts = real_data[feature].value_counts(normalize=True).reset_index()
        real_counts.columns = [feature, 'real_count']
        
        synth_counts = synthetic_data[feature].value_counts(normalize=True).reset_index()
        synth_counts.columns = [feature, 'synth_count']
        
        # Merge counts
        merged = pd.merge(real_counts, synth_counts, on=feature, how='outer').fillna(0)
        
        # Sort by real counts
        merged = merged.sort_values('real_count', ascending=False)
        
        # Limit to top 20 categories if there are too many
        if len(merged) > 20:
            merged = merged.head(20)
            
        # Calculate difference
        merged['difference'] = merged['synth_count'] - merged['real_count']
        
        # Create a grouped bar chart
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=merged[feature],
                y=merged['real_count'],
                name='Real Data',
                marker_color='rgba(0, 123, 255, 0.7)'
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=merged[feature],
                y=merged['synth_count'],
                name='Synthetic Data',
                marker_color='rgba(220, 53, 69, 0.7)'
            )
        )
        
        # Calculate JS divergence
        js_divergence = self._js_divergence(merged['real_count'], merged['synth_count'])
        
        # Add statistics
        stats_text = (
            f"Categories: {len(merged)} of {max(len(real_counts), len(synth_counts))}<br>"
            f"JS Divergence: {js_divergence:.4f}<br>"
            f"Biggest diff: {merged['difference'].abs().max():.4f}"
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
            yaxis_title="Frequency",
            barmode='group',
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
        Create correlation heatmap comparison between real and synthetic data.
        
        Args:
            real_data: Original data
            synthetic_data: Synthetic data
            features: Features to include (None for all)
            top_n: Number of top correlated features to show
            
        Returns:
            Plotly figure object
        """
        # Select numerical features if not specified
        if features is None:
            features = []
            for col in set(real_data.columns) & set(synthetic_data.columns):
                if pd.api.types.is_numeric_dtype(real_data[col]) and pd.api.types.is_numeric_dtype(synthetic_data[col]):
                    features.append(col)
        
        # Ensure we have at least 2 features
        if len(features) < 2:
            raise ValueError("Need at least 2 numerical features for correlation analysis")
        
        # Calculate correlation matrices
        real_corr = real_data[features].corr()
        synth_corr = synthetic_data[features].corr()
        
        # Calculate correlation difference
        diff_corr = synth_corr - real_corr
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Real Data', 'Synthetic Data', 'Difference'),
            horizontal_spacing=0.05
        )
        
        # Add heatmaps
        fig.add_trace(
            go.Heatmap(
                z=real_corr.values,
                x=real_corr.columns,
                y=real_corr.index,
                colorscale='Blues',
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title="Correlation",
                    titleside="right",
                    x=0.28
                )
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Heatmap(
                z=synth_corr.values,
                x=synth_corr.columns,
                y=synth_corr.index,
                colorscale='Reds',
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title="Correlation",
                    titleside="right",
                    x=0.61
                )
            ),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Heatmap(
                z=diff_corr.values,
                x=diff_corr.columns,
                y=diff_corr.index,
                colorscale='RdBu',
                zmid=0,
                zmin=-1,
                zmax=1,
                colorbar=dict(
                    title="Difference",
                    titleside="right",
                    x=0.94
                )
            ),
            row=1, col=3
        )
        
        # Calculate Frobenius norm of difference matrix
        import numpy as np
        frob_norm = np.linalg.norm(diff_corr.values)
        mean_abs_diff = np.abs(diff_corr.values).mean()
        
        # Add annotation with statistics
        fig.add_annotation(
            x=0.5,
            y=1.05,
            xref="paper",
            yref="paper",
            text=f"Frobenius Norm: {frob_norm:.4f} | Mean Absolute Difference: {mean_abs_diff:.4f}",
            showarrow=False,
            font=dict(size=12)
        )
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Correlation Matrix Comparison",
            height=600,
            width=1200,
            margin=dict(l=40, r=40, t=80, b=40),
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
        # Extract metrics from the input dictionary
        radar_metrics = {}
        
        # Process statistical similarity metrics
        if 'numerical' in metrics:
            numerical_metrics = metrics['numerical']
            
            # Calculate average metric per type
            ks_metrics = [v.get('ks_statistic', 1.0) for k, v in numerical_metrics.items() 
                         if isinstance(v, dict) and 'ks_statistic' in v]
            js_metrics = [v.get('jensen_shannon_dist', 1.0) for k, v in numerical_metrics.items() 
                         if isinstance(v, dict) and 'jensen_shannon_dist' in v]
            
            if ks_metrics:
                radar_metrics['KS Distance'] = 1 - sum(ks_metrics) / len(ks_metrics)
            
            if js_metrics:
                radar_metrics['JS Divergence'] = 1 - sum(js_metrics) / len(js_metrics)
        
        # Process categorical metrics
        if 'categorical' in metrics:
            categorical_metrics = metrics['categorical']
            
            # Calculate average metric per type
            chi2_metrics = [v.get('chi2_pvalue', 0.0) for k, v in categorical_metrics.items() 
                           if isinstance(v, dict) and 'chi2_pvalue' in v]
            dist_diff_metrics = [v.get('distribution_difference', 1.0) for k, v in categorical_metrics.items() 
                              if isinstance(v, dict) and 'distribution_difference' in v]
            
            if chi2_metrics:
                radar_metrics['Chi2 p-value'] = sum(chi2_metrics) / len(chi2_metrics)
            
            if dist_diff_metrics:
                radar_metrics['Category Dist'] = 1 - sum(dist_diff_metrics) / len(dist_diff_metrics)
        
        # Process privacy metrics
        if 'privacy' in metrics:
            privacy_metrics = metrics['privacy']
            
            # Extract common privacy metrics
            identifiability = privacy_metrics.get('identifiability_score', 0.5)
            membership = privacy_metrics.get('membership_disclosure_score', 0.5)
            attribute = privacy_metrics.get('attribute_disclosure_score', 0.5)
            
            radar_metrics['Identifiability'] = 1 - identifiability
            radar_metrics['Membership Privacy'] = 1 - membership
            radar_metrics['Attribute Privacy'] = 1 - attribute
        
        # Process overall metrics
        if 'overall' in metrics:
            overall_metrics = metrics['overall']
            
            # Extract common overall metrics
            similarity = overall_metrics.get('statistical_similarity', 0.5)
            utility = overall_metrics.get('utility_score', 0.5)
            quality = overall_metrics.get('overall_quality', 0.5)
            
            radar_metrics['Stat Similarity'] = similarity
            radar_metrics['Utility'] = utility
            radar_metrics['Overall Quality'] = quality
        
        # Ensure we have metrics to plot
        if not radar_metrics:
            raise ValueError("No valid metrics found for quality radar chart")
        
        # Create radar chart data
        categories = list(radar_metrics.keys())
        values = list(radar_metrics.values())
        
        # Close the loop for radar chart
        categories.append(categories[0])
        values.append(values[0])
        
        # Create figure
        fig = go.Figure()
        
        # Add trace
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            line=dict(color='rgb(31, 119, 180)', width=2),
            fillcolor='rgba(31, 119, 180, 0.3)',
            name='Quality Metrics'
        ))
        
        # Add reference circles
        for level in [0.2, 0.4, 0.6, 0.8]:
            fig.add_trace(go.Scatterpolar(
                r=[level] * len(categories),
                theta=categories,
                mode='lines',
                line=dict(color='gray', width=0.5),
                showlegend=False,
                hoverinfo='none'
            ))
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Quality Metrics",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=["0.2", "0.4", "0.6", "0.8", "1.0"],
                )
            ),
            showlegend=False,
            margin=dict(l=80, r=80, t=80, b=80),
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
        # Get common numerical columns
        numerical_columns = []
        for col in set(real_data.columns) & set(synthetic_data.columns):
            if pd.api.types.is_numeric_dtype(real_data[col]) and pd.api.types.is_numeric_dtype(synthetic_data[col]):
                numerical_columns.append(col)
        
        if len(numerical_columns) < 2:
            raise ValueError("Need at least 2 numerical features for PCA analysis")
        
        # Adjust n_components if necessary
        n_components = min(n_components, len(numerical_columns))
        
        # Extract numerical data
        real_numerical = real_data[numerical_columns].select_dtypes(include=['number'])
        synth_numerical = synthetic_data[numerical_columns].select_dtypes(include=['number'])
        
        # Standardize the data
        scaler = StandardScaler()
        
        # Combine data for consistent scaling
        combined = pd.concat([real_numerical, synth_numerical])
        combined_scaled = scaler.fit_transform(combined)
        
        # Split back
        real_scaled = combined_scaled[:len(real_numerical)]
        synth_scaled = combined_scaled[len(real_numerical):]
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        pca.fit(combined_scaled)
        
        # Transform data
        real_pca = pca.transform(real_scaled)
        synth_pca = pca.transform(synth_scaled)
        
        # Create DataFrame for plotting
        real_pca_df = pd.DataFrame(real_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        real_pca_df['source'] = 'Real'
        
        synth_pca_df = pd.DataFrame(synth_pca, columns=[f'PC{i+1}' for i in range(n_components)])
        synth_pca_df['source'] = 'Synthetic'
        
        # Combine for plotting
        pca_df = pd.concat([real_pca_df, synth_pca_df])
        
        # Create plot
        if n_components == 2:
            # 2D scatter plot
            fig = px.scatter(
                pca_df, x='PC1', y='PC2', color='source',
                color_discrete_map={'Real': 'blue', 'Synthetic': 'red'},
                opacity=0.7
            )
            
            # Add density contours
            fig.add_trace(
                go.Histogram2dContour(
                    x=real_pca_df['PC1'],
                    y=real_pca_df['PC2'],
                    colorscale='Blues',
                    showscale=False,
                    opacity=0.3,
                    name='Real Density',
                    ncontours=10
                )
            )
            
            fig.add_trace(
                go.Histogram2dContour(
                    x=synth_pca_df['PC1'],
                    y=synth_pca_df['PC2'],
                    colorscale='Reds',
                    showscale=False,
                    opacity=0.3,
                    name='Synthetic Density',
                    ncontours=10
                )
            )
            
        else:
            # 3D scatter plot for 3 components
            fig = px.scatter_3d(
                pca_df, x='PC1', y='PC2', z='PC3', color='source',
                color_discrete_map={'Real': 'blue', 'Synthetic': 'red'},
                opacity=0.7
            )
        
        # Add variance explained
        explained_variance = pca.explained_variance_ratio_
        total_explained = sum(explained_variance) * 100
        
        var_text = '<br>'.join([
            f"PC{i+1}: {v:.1f}%" for i, v in enumerate(explained_variance * 100)
        ])
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Explained Variance:<br>{var_text}<br>Total: {total_explained:.1f}%",
            showarrow=False,
            font=dict(size=10),
            align="left",
            bordercolor="black",
            borderwidth=1,
            borderpad=4,
            bgcolor="white",
            opacity=0.8
        )
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}PCA Comparison",
            legend=dict(
                x=0.01,
                y=0.99,
                bgcolor='rgba(255, 255, 255, 0.5)',
                bordercolor='rgba(0, 0, 0, 0.1)'
            ),
            margin=dict(l=40, r=40, t=40, b=40),
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
        # Extract key privacy metrics
        metrics = {
            'Identifiability Risk': privacy_metrics.get('identifiability_score', 0.5),
            'Membership Disclosure': privacy_metrics.get('membership_disclosure_score', 0.5),
            'Attribute Disclosure': privacy_metrics.get('attribute_disclosure_score', 0.5),
            'Distance from Original': 1 - privacy_metrics.get('distance_score', 0.5),
            'Uniqueness Risk': privacy_metrics.get('uniqueness_score', 0.5),
            'k-Anonymity Level': 1 - min(privacy_metrics.get('k_anonymity_level', 5) / 10, 1.0)
        }
        
        # Create gauge charts
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=list(metrics.keys())
        )
        
        # Add gauges
        row, col = 1, 1
        for metric_name, value in metrics.items():
            # Determine color based on risk
            if value < 0.3:
                color = "green"
            elif value < 0.7:
                color = "orange"
            else:
                color = "red"
                
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=value,
                    title={'text': metric_name},
                    gauge={
                        'axis': {'range': [0, 1]},
                        'bar': {'color': color},
                        'steps': [
                            {'range': [0, 0.3], 'color': 'rgba(0, 250, 0, 0.3)'},
                            {'range': [0.3, 0.7], 'color': 'rgba(255, 165, 0, 0.3)'},
                            {'range': [0.7, 1], 'color': 'rgba(255, 0, 0, 0.3)'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': value
                        }
                    },
                    domain={'row': row-1, 'column': col-1}
                )
            )
            
            # Move to next position
            col += 1
            if col > 3:
                col = 1
                row += 1
        
        # Calculate overall privacy risk
        overall_risk = sum(metrics.values()) / len(metrics)
        
        # Update layout
        fig.update_layout(
            title=f"{self.title_prefix}Privacy Risk Assessment (Overall: {overall_risk:.2f})",
            height=600,
            width=1000,
            margin=dict(l=40, r=40, t=80, b=40),
        )
        
        return fig
    
    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence between two discrete distributions.
        
        Both inputs must be valid probability distributions (sum to 1).
        """
        # Ensure inputs are numpy arrays
        p = np.asarray(p)
        q = np.asarray(q)
        
        # Handle zero values to avoid log(0)
        p = np.maximum(p, 1e-15)
        q = np.maximum(q, 1e-15)
        
        # Normalize if not already probability distributions
        if abs(np.sum(p) - 1.0) > 1e-10:
            p = p / np.sum(p)
        
        if abs(np.sum(q) - 1.0) > 1e-10:
            q = q / np.sum(q)
        
        # Calculate the mixture distribution
        m = (p + q) / 2
        
        # Calculate KL divergences
        # KL(P || M)
        kl_pm = np.sum(p * np.log2(p / m), where=(p != 0))
        
        # KL(Q || M)
        kl_qm = np.sum(q * np.log2(q / m), where=(q != 0))
        
        # JS divergence
        js = (kl_pm + kl_qm) / 2
        
        return js
