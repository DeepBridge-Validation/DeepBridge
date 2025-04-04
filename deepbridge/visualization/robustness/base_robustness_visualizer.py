"""
Base interface for robustness visualization components.
"""

import typing as t
from abc import abstractmethod
import plotly.graph_objects as go
from deepbridge.visualization.base_visualizer import BaseVisualizer

class BaseRobustnessVisualizer(BaseVisualizer):
    """
    Abstract base class for robustness visualization components.
    Extends the BaseVisualizer with robustness-specific visualization methods.
    """
    
    def __init__(self, title_prefix: str = "Robustness - ", include_plotly_mode_bar: bool = True):
        """
        Initialize the robustness visualizer.
        
        Args:
            title_prefix: Prefix to add to all visualization titles
            include_plotly_mode_bar: Whether to include Plotly's modebar
        """
        super().__init__(title_prefix, include_plotly_mode_bar)
    
    @abstractmethod
    def create_model_comparison_plot(self, 
                                    results: t.Dict[str, t.Any],
                                    alternative_results: t.Optional[t.Dict[str, t.Dict[str, t.Any]]] = None) -> go.Figure:
        """
        Create a bar chart comparing robustness across models.
        
        Args:
            results: Primary model results
            alternative_results: Results for alternative models
            
        Returns:
            Plotly figure object
        """
        pass
    
    @abstractmethod
    def create_perturbation_plot(self, 
                                results: t.Dict[str, t.Any],
                                perturbation_type: str = 'raw') -> go.Figure:
        """
        Create a line chart showing model performance under different perturbation levels.
        
        Args:
            results: Test results
            perturbation_type: Type of perturbation to visualize ('raw', 'quantile', etc.)
            
        Returns:
            Plotly figure object
        """
        pass
    
    @abstractmethod
    def create_feature_importance_plot(self, 
                                      feature_importance: t.Dict[str, float], 
                                      top_n: int = 10) -> go.Figure:
        """
        Create a bar chart showing feature importance for robustness.
        
        Args:
            feature_importance: Mapping of feature names to importance scores
            top_n: Number of top features to show
            
        Returns:
            Plotly figure object
        """
        pass
    
    @abstractmethod
    def create_methods_comparison_plot(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a bar chart comparing different perturbation methods.
        
        Args:
            results: Test results
            
        Returns:
            Plotly figure object
        """
        pass
    
    @abstractmethod
    def create_score_distribution_plot(self, 
                                      results: t.Dict[str, t.Any], 
                                      method: str = 'both') -> go.Figure:
        """
        Create a box plot showing the distribution of scores across perturbation levels.
        
        Args:
            results: Test results
            method: Which perturbation method to visualize ('raw', 'quantile', or 'both')
            
        Returns:
            Plotly figure object
        """
        pass
    
    def create_visualization(self, data: t.Dict[str, t.Any], **kwargs) -> t.Dict[str, go.Figure]:
        """
        Create standard robustness visualizations from test results.
        
        Args:
            data: Robustness test results
            **kwargs: Additional visualization parameters
            
        Returns:
            Dict of visualization name to plotly figure
        """
        visualizations = {}
        
        # Extract results and alternative results
        results = data.get('main_model', data)
        alternative_results = data.get('alternative_models', {})
        
        # Create model comparison plot
        try:
            visualizations['models_comparison'] = self.create_model_comparison_plot(
                results, alternative_results
            )
        except Exception as e:
            if self.verbose:
                print(f"Error creating model comparison plot: {str(e)}")
        
        # Create perturbation plots
        try:
            visualizations['raw_perturbation'] = self.create_perturbation_plot(
                results, 'raw'
            )
        except Exception as e:
            if self.verbose:
                print(f"Error creating raw perturbation plot: {str(e)}")
                
        try:
            visualizations['quantile_perturbation'] = self.create_perturbation_plot(
                results, 'quantile'
            )
        except Exception as e:
            if self.verbose:
                print(f"Error creating quantile perturbation plot: {str(e)}")
        
        # Create feature importance plot if available
        if 'feature_importance' in results:
            try:
                visualizations['feature_importance'] = self.create_feature_importance_plot(
                    results['feature_importance']
                )
            except Exception as e:
                if self.verbose:
                    print(f"Error creating feature importance plot: {str(e)}")
        
        # Create methods comparison plot
        try:
            visualizations['perturbation_methods'] = self.create_methods_comparison_plot(results)
        except Exception as e:
            if self.verbose:
                print(f"Error creating methods comparison plot: {str(e)}")
        
        # Create score distribution plot
        try:
            visualizations['score_distribution'] = self.create_score_distribution_plot(
                results, 'both'
            )
        except Exception as e:
            if self.verbose:
                print(f"Error creating score distribution plot: {str(e)}")
        
        return visualizations