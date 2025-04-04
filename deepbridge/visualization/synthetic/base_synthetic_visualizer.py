"""
Base interface for synthetic data visualization components.
"""

import typing as t
from abc import abstractmethod
import pandas as pd
import plotly.graph_objects as go
from deepbridge.visualization.base_visualizer import BaseVisualizer

class BaseSyntheticVisualizer(BaseVisualizer):
    """
    Abstract base class for synthetic data visualization components.
    Extends the BaseVisualizer with synthetic data-specific visualization methods.
    """
    
    def __init__(self, title_prefix: str = "Synthetic Data - ", include_plotly_mode_bar: bool = True):
        """
        Initialize the synthetic data visualizer.
        
        Args:
            title_prefix: Prefix to add to all visualization titles
            include_plotly_mode_bar: Whether to include Plotly's modebar
        """
        super().__init__(title_prefix, include_plotly_mode_bar)
        self.verbose = False
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def create_quality_metrics_plot(self, metrics: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a spider/radar chart for synthetic data quality metrics.
        
        Args:
            metrics: Quality metrics dictionary
            
        Returns:
            Plotly figure object
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def create_privacy_risk_plot(self, privacy_metrics: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create privacy risk visualization.
        
        Args:
            privacy_metrics: Privacy metrics dictionary
            
        Returns:
            Plotly figure object
        """
        pass
    
    def create_visualization(self, 
                           real_data: pd.DataFrame,
                           synthetic_data: pd.DataFrame,
                           metrics: t.Dict[str, t.Any] = None,
                           **kwargs) -> t.Dict[str, go.Figure]:
        """
        Create standard synthetic data visualizations.
        
        Args:
            real_data: Original data
            synthetic_data: Synthetic data
            metrics: Optional quality metrics
            **kwargs: Additional visualization parameters
            
        Returns:
            Dict of visualization name to plotly figure
        """
        visualizations = {}
        
        # Get numerical features
        numerical_features = real_data.select_dtypes(include=['number']).columns.tolist()
        categorical_features = real_data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Create distribution comparison for each numerical feature (limit to first 5)
        for i, feature in enumerate(numerical_features[:5]):
            try:
                visualizations[f'dist_{feature}'] = self.create_distribution_comparison(
                    real_data, synthetic_data, feature
                )
            except Exception as e:
                if self.verbose:
                    print(f"Error creating distribution comparison for {feature}: {str(e)}")
        
        # Create correlation comparison
        try:
            visualizations['correlation'] = self.create_correlation_comparison(
                real_data, synthetic_data
            )
        except Exception as e:
            if self.verbose:
                print(f"Error creating correlation comparison: {str(e)}")
        
        # Create quality metrics plot if metrics are provided
        if metrics:
            try:
                visualizations['quality_metrics'] = self.create_quality_metrics_plot(metrics)
            except Exception as e:
                if self.verbose:
                    print(f"Error creating quality metrics plot: {str(e)}")
            
            # Create privacy risk plot if privacy metrics are available
            if 'privacy' in metrics:
                try:
                    visualizations['privacy_risk'] = self.create_privacy_risk_plot(
                        metrics['privacy']
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"Error creating privacy risk plot: {str(e)}")
        
        # Create PCA comparison
        try:
            visualizations['pca'] = self.create_pca_comparison(
                real_data, synthetic_data
            )
        except Exception as e:
            if self.verbose:
                print(f"Error creating PCA comparison: {str(e)}")
        
        return visualizations