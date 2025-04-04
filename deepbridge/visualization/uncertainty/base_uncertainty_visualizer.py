"""
Base interface for uncertainty visualization components.
"""

import typing as t
from abc import abstractmethod
import plotly.graph_objects as go
from deepbridge.visualization.base_visualizer import BaseVisualizer

class BaseUncertaintyVisualizer(BaseVisualizer):
    """
    Abstract base class for uncertainty visualization components.
    Extends the BaseVisualizer with uncertainty-specific visualization methods.
    """
    
    def __init__(self, title_prefix: str = "Uncertainty - ", include_plotly_mode_bar: bool = True):
        """
        Initialize the uncertainty visualizer.
        
        Args:
            title_prefix: Prefix to add to all visualization titles
            include_plotly_mode_bar: Whether to include Plotly's modebar
        """
        super().__init__(title_prefix, include_plotly_mode_bar)
        self.verbose = False
    
    @abstractmethod
    def create_calibration_plot(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a calibration plot comparing expected vs actual probabilities.
        
        Args:
            results: Uncertainty test results
            
        Returns:
            Plotly figure object
        """
        pass
    
    @abstractmethod
    def create_confidence_histogram(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a histogram of confidence/probability values.
        
        Args:
            results: Uncertainty test results
            
        Returns:
            Plotly figure object
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def create_alpha_comparison_plot(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a plot comparing different alpha levels for confidence intervals.
        
        Args:
            results: Uncertainty test results
            
        Returns:
            Plotly figure object
        """
        pass
    
    @abstractmethod
    def create_width_distribution_plot(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a box plot showing the distribution of interval widths.
        
        Args:
            results: Uncertainty test results
            
        Returns:
            Plotly figure object
        """
        pass
    
    @abstractmethod
    def create_coverage_vs_width_plot(self, results: t.Dict[str, t.Any]) -> go.Figure:
        """
        Create a plot showing the trade-off between coverage and interval width.
        
        Args:
            results: Uncertainty test results
            
        Returns:
            Plotly figure object
        """
        pass
    
    def create_visualization(self, data: t.Dict[str, t.Any], **kwargs) -> t.Dict[str, go.Figure]:
        """
        Create standard uncertainty visualizations from test results.
        
        Args:
            data: Uncertainty test results
            **kwargs: Additional visualization parameters
            
        Returns:
            Dict of visualization name to plotly figure
        """
        visualizations = {}
        
        # Extract results
        results = data.get('main_model', data)
        
        # Create calibration plot
        try:
            visualizations['calibration'] = self.create_calibration_plot(results)
        except Exception as e:
            if self.verbose:
                print(f"Error creating calibration plot: {str(e)}")
        
        # Create confidence histogram
        try:
            visualizations['confidence_histogram'] = self.create_confidence_histogram(results)
        except Exception as e:
            if self.verbose:
                print(f"Error creating confidence histogram: {str(e)}")
        
        # Create feature importance plot if available
        if 'feature_importance' in results:
            try:
                visualizations['feature_importance'] = self.create_feature_importance_plot(
                    results['feature_importance']
                )
            except Exception as e:
                if self.verbose:
                    print(f"Error creating feature importance plot: {str(e)}")
        
        # Create alpha comparison plot
        try:
            visualizations['alpha_comparison'] = self.create_alpha_comparison_plot(results)
        except Exception as e:
            if self.verbose:
                print(f"Error creating alpha comparison plot: {str(e)}")
        
        # Create width distribution plot
        try:
            visualizations['width_distribution'] = self.create_width_distribution_plot(results)
        except Exception as e:
            if self.verbose:
                print(f"Error creating width distribution plot: {str(e)}")
        
        # Create coverage vs width plot
        try:
            visualizations['coverage_vs_width'] = self.create_coverage_vs_width_plot(results)
        except Exception as e:
            if self.verbose:
                print(f"Error creating coverage vs width plot: {str(e)}")
        
        return visualizations