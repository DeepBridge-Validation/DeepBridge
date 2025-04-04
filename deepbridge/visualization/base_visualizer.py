"""
Base interface for visualization components.
"""

import typing as t
from abc import ABC, abstractmethod
import pandas as pd
import plotly.graph_objects as go

class BaseVisualizer(ABC):
    """
    Abstract base class for all visualization components.
    Defines the common interface that all visualizers should implement.
    """
    
    def __init__(self, title_prefix: str = "", include_plotly_mode_bar: bool = True):
        """
        Initialize the base visualizer.
        
        Args:
            title_prefix: Prefix to add to all visualization titles
            include_plotly_mode_bar: Whether to include Plotly's modebar
        """
        self.title_prefix = title_prefix
        self.include_plotly_mode_bar = include_plotly_mode_bar
        
    @abstractmethod
    def create_visualization(self, data: t.Any, **kwargs) -> t.Dict[str, go.Figure]:
        """
        Create visualizations from data.
        
        Args:
            data: Data to visualize
            **kwargs: Additional visualization parameters
            
        Returns:
            Dict of visualization name to plotly figure
        """
        pass
    
    def update_figure_layout(self, fig: go.Figure, title: str = None, 
                            x_title: str = None, y_title: str = None,
                            template: str = "plotly_white") -> go.Figure:
        """
        Apply consistent styling to a plotly figure.
        
        Args:
            fig: Plotly figure to style
            title: Figure title
            x_title: X-axis title
            y_title: Y-axis title
            template: Plotly template to use
            
        Returns:
            Styled plotly figure
        """
        layout_updates = {
            "template": template,
            "margin": {"l": 50, "r": 50, "t": 50, "b": 50}
        }
        
        if title:
            layout_updates["title"] = f"{self.title_prefix}{title}"
        
        if x_title:
            layout_updates["xaxis_title"] = x_title
            
        if y_title:
            layout_updates["yaxis_title"] = y_title
            
        if not self.include_plotly_mode_bar:
            layout_updates["modebar"] = {"display": False}
            
        fig.update_layout(**layout_updates)
        
        return fig
    
    def create_bar_chart(self, x, y, title: str = None, x_title: str = None, 
                         y_title: str = None, horizontal: bool = False, 
                         text_auto: bool = True) -> go.Figure:
        """
        Create a standard bar chart.
        
        Args:
            x: X-axis values
            y: Y-axis values
            title: Chart title
            x_title: X-axis title
            y_title: Y-axis title
            horizontal: Whether to create a horizontal bar chart
            text_auto: Whether to automatically add text labels
            
        Returns:
            Bar chart figure
        """
        if horizontal:
            # For horizontal bar chart, x and y are swapped
            fig = go.Figure(data=[
                go.Bar(
                    x=y, y=x,
                    orientation='h',
                    text=y if text_auto else None,
                    textposition='auto' if text_auto else 'none'
                )
            ])
        else:
            fig = go.Figure(data=[
                go.Bar(
                    x=x, y=y,
                    text=y if text_auto else None,
                    textposition='auto' if text_auto else 'none'
                )
            ])
            
        return self.update_figure_layout(fig, title, x_title, y_title)
    
    def create_line_chart(self, x, y, title: str = None, x_title: str = None, 
                          y_title: str = None, mode: str = 'lines+markers',
                          name: str = None) -> go.Figure:
        """
        Create a standard line chart.
        
        Args:
            x: X-axis values
            y: Y-axis values
            title: Chart title
            x_title: X-axis title
            y_title: Y-axis title
            mode: Line mode ('lines', 'markers', 'lines+markers')
            name: Line name for legend
            
        Returns:
            Line chart figure
        """
        fig = go.Figure(data=[
            go.Scatter(
                x=x, y=y,
                mode=mode,
                name=name
            )
        ])
            
        return self.update_figure_layout(fig, title, x_title, y_title)
    
    def create_multi_line_chart(self, data_dict: t.Dict[str, t.Dict[str, t.List]], 
                               x_key: str, y_key: str, title: str = None, 
                               x_title: str = None, y_title: str = None, 
                               mode: str = 'lines+markers') -> go.Figure:
        """
        Create a multi-line chart from a dictionary of series.
        
        Args:
            data_dict: Dictionary mapping series name to data dict with x and y values
            x_key: Key in data dict for x values
            y_key: Key in data dict for y values
            title: Chart title
            x_title: X-axis title
            y_title: Y-axis title
            mode: Line mode ('lines', 'markers', 'lines+markers')
            
        Returns:
            Multi-line chart figure
        """
        fig = go.Figure()
        
        for name, data in data_dict.items():
            if x_key in data and y_key in data:
                fig.add_trace(go.Scatter(
                    x=data[x_key],
                    y=data[y_key],
                    mode=mode,
                    name=name
                ))
            
        return self.update_figure_layout(fig, title, x_title, y_title)
    
    def create_box_plot(self, data, labels=None, title: str = None, 
                       x_title: str = None, y_title: str = None, 
                       boxpoints: str = 'outliers') -> go.Figure:
        """
        Create a box plot from data.
        
        Args:
            data: Data for box plot (list of lists or DataFrame)
            labels: Optional labels for each box
            title: Chart title
            x_title: X-axis title
            y_title: Y-axis title
            boxpoints: Box points mode ('all', 'outliers', 'suspectedoutliers', False)
            
        Returns:
            Box plot figure
        """
        if isinstance(data, pd.DataFrame):
            # If data is a DataFrame, create box plot for each column
            fig = go.Figure()
            for col in data.columns:
                fig.add_trace(go.Box(
                    y=data[col],
                    name=col,
                    boxpoints=boxpoints
                ))
        else:
            # If data is a list of lists
            fig = go.Figure()
            for i, y in enumerate(data):
                label = labels[i] if labels and i < len(labels) else f"Series {i+1}"
                fig.add_trace(go.Box(
                    y=y,
                    name=label,
                    boxpoints=boxpoints
                ))
            
        return self.update_figure_layout(fig, title, x_title, y_title)
    
    def create_heatmap(self, data, title: str = None, x_labels=None, y_labels=None,
                      colorscale="RdBu_r") -> go.Figure:
        """
        Create a heatmap.
        
        Args:
            data: 2D array for heatmap
            title: Chart title
            x_labels: Optional labels for x-axis
            y_labels: Optional labels for y-axis
            colorscale: Colorscale to use
            
        Returns:
            Heatmap figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale
        ))
            
        return self.update_figure_layout(fig, title)