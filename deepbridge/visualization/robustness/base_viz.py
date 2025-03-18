import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Dict, Optional, Any

class RobustnessBaseViz:
    """
    Base class for robustness visualization components.
    Provides common utilities and settings.
    """
    
    @staticmethod
    def get_color_palette():
        """Return standard color palette for consistency."""
        return ['#3A6EA5', '#FF6B6B', '#52D273', '#FFD166', '#6A67CE']
    
    @staticmethod
    def apply_common_layout(fig, title=None, x_title=None, y_title=None, height=None, width=None):
        """Apply common layout settings to figures."""
        layout_settings = dict(
            plot_bgcolor='white',
            hovermode="closest",
            autosize=True,
            margin=dict(l=40, r=40, t=80, b=60)
        )
        
        if title:
            layout_settings['title'] = dict(
                text=title,
                y=0.95,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(size=16)
            )
            
        if x_title:
            layout_settings['xaxis'] = dict(
                title=dict(
                    text=x_title,
                    font=dict(size=12)
                ),
                gridcolor='rgba(230,230,230,0.5)'
            )
            
        if y_title:
            layout_settings['yaxis'] = dict(
                title=dict(
                    text=y_title,
                    font=dict(size=12)
                ),
                gridcolor='rgba(230,230,230,0.5)'
            )
        
        # Add dimensions if specified
        if height:
            layout_settings['height'] = height
        if width:
            layout_settings['width'] = width
            
        fig.update_layout(**layout_settings)
        return fig