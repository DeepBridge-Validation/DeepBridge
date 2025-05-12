"""
Base chart module for uncertainty visualizations.
"""

import logging
import numpy as np
import base64
import io
from typing import Dict, List, Any, Optional, Tuple

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class BaseChartGenerator:
    """
    Base class for uncertainty chart generators.
    """
    
    def __init__(self, seaborn_chart_generator=None):
        """
        Initialize the base chart generator.

        Parameters:
        ----------
        seaborn_chart_generator : SeabornChartGenerator, optional
            Existing chart generator to use for rendering
        """
        self.chart_generator = seaborn_chart_generator

        # Make visualization libraries available globally
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            from scipy import stats

            self.sns = sns
            self.plt = plt
            self.pd = pd
            self.np = np
            self.stats = stats

            # Set default style
            sns.set_theme(style="whitegrid")
            self.has_visualization_libs = True
        except ImportError as e:
            logger.error(f"Required libraries for visualization not available: {str(e)}")
            self.has_visualization_libs = False

        # If using existing chart generator, ensure we have access to its plt object
        if self.chart_generator and hasattr(self.chart_generator, 'plt'):
            self.plt = self.chart_generator.plt
    
    def _validate_chart_generator(self):
        """Check if we have a valid chart generator to work with."""
        if not hasattr(self, 'plt') or not hasattr(self, 'sns') or not hasattr(self, 'pd'):
            # Try to import visualization libraries if not already loaded
            try:
                import seaborn as sns
                import matplotlib.pyplot as plt
                import pandas as pd
                import numpy as np
                from scipy import stats

                self.sns = sns
                self.plt = plt
                self.pd = pd
                self.np = np
                self.stats = stats

                # Set default style
                sns.set_theme(style="whitegrid")
                self.has_visualization_libs = True
            except ImportError as e:
                logger.error(f"Required libraries for visualization not available: {str(e)}")
                self.has_visualization_libs = False

        if not self.chart_generator and not getattr(self, 'has_visualization_libs', False):
            raise ValueError("No chart generator or visualization libraries available")

    def _validate_data(self, data):
        """
        Check if the data is valid for visualization with relaxed constraints.
        Attempts to extract meaningful data even if partially incomplete.
        """
        # If the data is None, it's not valid
        if data is None:
            return False
            
        # If the data is a dictionary, it should have at least one entry
        if isinstance(data, dict):
            return len(data) > 0
            
        # If the data is a list, it should have at least one element
        if isinstance(data, list) or hasattr(data, '__len__'):
            return len(data) > 0
            
        # If the data is a numeric value, it's valid
        if isinstance(data, (int, float)):
            return True
            
        # If we've gotten this far, we can't determine if the data is valid
        return False
    
    def _save_figure_to_base64(self, fig):
        """
        Save matplotlib figure to base64 encoded string.

        Parameters:
        -----------
        fig : matplotlib.figure.Figure
            The figure to save

        Returns:
        --------
        str : Base64 encoded image
        """
        # Check if plt is available
        if not hasattr(self, 'plt'):
            logger.error("matplotlib.pyplot not available for saving figure")
            return ""

        try:
            # Save to buffer
            buffer = io.BytesIO()
            fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
            buffer.seek(0)

            # Encode to base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Close figure
            self.plt.close(fig)

            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            logger.error(f"Error saving figure to base64: {str(e)}")
            return ""