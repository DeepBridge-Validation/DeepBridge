"""
Base chart class for fairness visualizations.

Provides common functionality and configuration for all fairness charts.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import plotly.graph_objects as go
import plotly.io as pio


class BaseChart(ABC):
    """
    Abstract base class for fairness charts.

    Provides common configuration and utilities for all chart types.
    """

    # Common color mappings
    COLOR_SUCCESS = '#2ecc71'
    COLOR_WARNING = '#f39c12'
    COLOR_CRITICAL = '#e74c3c'
    COLOR_INFO = '#3498db'
    COLOR_NEUTRAL = '#95a5a6'

    COLOR_MAP_STATUS = {
        'success': COLOR_SUCCESS,
        'ok': COLOR_SUCCESS,
        'warning': COLOR_WARNING,
        'critical': COLOR_CRITICAL
    }

    # Common layout settings
    COMMON_LAYOUT = {
        'template': 'plotly_white',
        'paper_bgcolor': '#FFFFFF',
        'plot_bgcolor': '#FFFFFF',
        'font': {'color': '#2c3e50'},
        'autosize': True
    }

    @abstractmethod
    def create(self, data: Dict[str, Any]) -> str:
        """
        Create the chart and return as Plotly JSON string.

        Args:
            data: Dictionary containing the data needed for the chart

        Returns:
            Plotly figure as JSON string
        """
        pass

    def _apply_common_layout(self, fig: go.Figure, **kwargs) -> go.Figure:
        """
        Apply common layout settings to a figure.

        Args:
            fig: Plotly figure
            **kwargs: Additional layout settings to override defaults

        Returns:
            Modified figure
        """
        layout_settings = {**self.COMMON_LAYOUT, **kwargs}
        fig.update_layout(**layout_settings)
        return fig

    def _to_json(self, fig: go.Figure) -> str:
        """
        Convert figure to JSON string.

        Args:
            fig: Plotly figure

        Returns:
            JSON string representation
        """
        import json
        import numpy as np

        # Convert to dict first to avoid binary data encoding issues
        fig_dict = fig.to_dict()

        # Recursively convert any binary-encoded arrays to regular lists
        def decode_arrays(obj):
            # Handle numpy arrays
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Handle numpy scalars
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, dict):
                # Check if this is a binary-encoded array
                if 'dtype' in obj and 'bdata' in obj:
                    # This is a binary array - convert to regular list
                    import base64
                    import struct
                    bdata = obj['bdata']
                    decoded = base64.b64decode(bdata)
                    dtype = obj['dtype']
                    if dtype == 'f8':  # float64
                        values = struct.unpack('d' * (len(decoded) // 8), decoded)
                        return list(values)
                    elif dtype == 'i4':  # int32
                        values = struct.unpack('i' * (len(decoded) // 4), decoded)
                        return list(values)
                    else:
                        return obj
                else:
                    # Recursively process dict values
                    return {k: decode_arrays(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                # Recursively process list items
                return [decode_arrays(item) for item in obj]
            elif isinstance(obj, tuple):
                # Convert tuples to lists
                return [decode_arrays(item) for item in obj]
            else:
                return obj

        # Decode all binary arrays in the figure
        fig_dict = decode_arrays(fig_dict)

        # Convert to JSON string
        return json.dumps(fig_dict)

    def _format_percentage(self, value: float, decimals: int = 1) -> str:
        """Format value as percentage string."""
        return f"{value:.{decimals}f}%"

    def _format_decimal(self, value: float, decimals: int = 4) -> str:
        """Format value as decimal string."""
        return f"{value:.{decimals}f}"

    def _get_color_for_status(self, status: str) -> str:
        """Get color for a given status."""
        return self.COLOR_MAP_STATUS.get(status, self.COLOR_NEUTRAL)
