"""
Base renderer for generating static HTML reports using Seaborn.

**Phase 3 Sprint 9:** Enhanced with flexible template method pattern for custom charts.
"""

import os
import base64
import tempfile
import logging
import datetime
import io
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

# Configure logger
logger = logging.getLogger("deepbridge.reports")

# Import JSON formatter
from ...utils.json_formatter import JsonFormatter

# Import CSS Manager
from ...css_manager import CSSManager

class BaseStaticRenderer:
    """
    Base class for static report renderers that use Seaborn for visualizations.
    """
    
    def __init__(self, template_manager, asset_manager):
        """
        Initialize the renderer.

        Parameters:
        -----------
        template_manager : TemplateManager
            Manager for templates
        asset_manager : AssetManager
            Manager for assets (CSS, JS, images)
        """
        self.template_manager = template_manager
        self.asset_manager = asset_manager

        # Initialize CSS Manager
        self.css_manager = CSSManager()

        # Import data transformer base
        from ...base import DataTransformer
        self.data_transformer = DataTransformer()

        # Try to import required libraries
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            self.sns = sns
            self.plt = plt
            self.pd = pd
            self.np = np
            self.has_visualization_libs = True
            # Set default style
            sns.set_theme(style="whitegrid")
        except ImportError as e:
            logger.error(f"Required libraries for static visualization not available: {str(e)}")
            self.has_visualization_libs = False
    
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model", report_type: str = "static", save_chart: bool = False) -> str:
        """
        Render static report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Experiment results data
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
        report_type : str, optional
            Type of report to generate ('interactive' or 'static')
        save_chart : bool, optional
            Whether to save charts as separate PNG files (default: False)

        Returns:
        --------
        str : Path to the generated report

        Raises:
        -------
        NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement render method")

    def save_charts_as_png(self, charts: Dict[str, str], file_path: str) -> None:
        """
        Save charts as PNG files in the same directory as the HTML report.

        Parameters:
        -----------
        charts : Dict[str, str]
            Dictionary of chart names and their base64 encoded images
        file_path : str
            Path to the HTML report file
        """
        import os
        import base64

        # Get the directory of the HTML report
        output_dir = os.path.dirname(os.path.abspath(file_path))

        # Get the filename (without extension) to use as a prefix
        file_basename = os.path.splitext(os.path.basename(file_path))[0]

        # Create the charts directory if it doesn't exist
        charts_dir = os.path.join(output_dir, f"{file_basename}_charts")
        os.makedirs(charts_dir, exist_ok=True)

        logger.info(f"Saving charts to directory: {charts_dir}")

        # Save each chart as a PNG file
        for chart_name, chart_data in charts.items():
            try:
                # Extract the base64 encoded image data
                if chart_data and isinstance(chart_data, str) and chart_data.startswith('data:image/png;base64,'):
                    # Remove the data URL prefix
                    base64_data = chart_data.replace('data:image/png;base64,', '')

                    # Decode the base64 data
                    image_data = base64.b64decode(base64_data)

                    # Generate a filename for the chart
                    chart_filename = f"{chart_name}.png"
                    chart_path = os.path.join(charts_dir, chart_filename)

                    # Save the image data to a file
                    with open(chart_path, 'wb') as f:
                        f.write(image_data)

                    logger.info(f"Saved chart to: {chart_path}")
                else:
                    logger.warning(f"Chart '{chart_name}' does not contain valid PNG data, skipping")
            except Exception as e:
                logger.error(f"Error saving chart '{chart_name}' to PNG: {str(e)}")
                logger.error(f"Traceback: {str(e)}")
    
    def generate_chart(self, chart_type: str, data: Dict[str, Any], title: str = None, figsize: tuple = (10, 6)) -> str:
        """
        Generate a chart using Seaborn and return base64 encoded image.
        
        Parameters:
        -----------
        chart_type : str
            Type of chart to generate (e.g., 'bar', 'line', 'boxplot')
        data : Dict[str, Any]
            Data for the chart
        title : str, optional
            Chart title
        figsize : tuple, optional
            Figure size in inches (width, height)
            
        Returns:
        --------
        str : Base64 encoded image data
        """
        if not self.has_visualization_libs:
            logger.error("Required libraries for visualization not available")
            return ""
        
        try:
            # Create a figure and axis
            fig, ax = self.plt.subplots(figsize=figsize)
            
            # Generate the chart based on chart_type
            if chart_type == 'bar':
                self._generate_bar_chart(ax, data)
            elif chart_type == 'line':
                self._generate_line_chart(ax, data)
            elif chart_type == 'boxplot':
                self._generate_boxplot_chart(ax, data)
            elif chart_type == 'heatmap':
                self._generate_heatmap_chart(ax, data)
            else:
                logger.error(f"Unsupported chart type: {chart_type}")
                return ""
            
            # Set the title if provided
            if title:
                ax.set_title(title)
            
            # Save the figure to a bytes buffer
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            
            # Encode the image to base64
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            
            # Close the figure to avoid memory leaks
            self.plt.close(fig)
            
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            logger.error(f"Error generating {chart_type} chart: {str(e)}")
            return ""
    
    def _generate_bar_chart(self, ax, data: Dict[str, Any]) -> None:
        """
        Generate a bar chart.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to draw the chart on
        data : Dict[str, Any]
            Data for the chart
        """
        x = data.get('x', [])
        y = data.get('y', [])
        
        if len(x) == 0 or len(y) == 0:
            logger.error("Empty data for bar chart")
            return
        
        # Create dataframe for seaborn
        df = self.pd.DataFrame({'x': x, 'y': y})
        
        # Generate bar chart
        self.sns.barplot(x='x', y='y', data=df, ax=ax)
        
        # Set labels
        ax.set_xlabel(data.get('x_label', 'X'))
        ax.set_ylabel(data.get('y_label', 'Y'))
        
        # Rotate x-axis labels if there are many categories
        if len(x) > 5:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _generate_line_chart(self, ax, data: Dict[str, Any]) -> None:
        """
        Generate a line chart.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to draw the chart on
        data : Dict[str, Any]
            Data for the chart
        """
        x = data.get('x', [])
        
        # Handle multiple y-series
        if 'y_series' in data and isinstance(data['y_series'], dict):
            for name, values in data['y_series'].items():
                if len(values) != len(x):
                    logger.warning(f"Series '{name}' length ({len(values)}) doesn't match x-axis length ({len(x)})")
                    continue
                ax.plot(x, values, label=name)
            
            # Add legend
            ax.legend()
        else:
            # Single y-series
            y = data.get('y', [])
            if len(x) == 0 or len(y) == 0:
                logger.error("Empty data for line chart")
                return
                
            # Create dataframe for seaborn
            df = self.pd.DataFrame({'x': x, 'y': y})
            
            # Generate line chart
            self.sns.lineplot(x='x', y='y', data=df, ax=ax)
        
        # Set labels
        ax.set_xlabel(data.get('x_label', 'X'))
        ax.set_ylabel(data.get('y_label', 'Y'))
    
    def _generate_boxplot_chart(self, ax, data: Dict[str, Any]) -> None:
        """
        Generate a boxplot chart.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to draw the chart on
        data : Dict[str, Any]
            Data for the chart
        """
        # Handle data differently depending on the format
        if 'models' in data:
            # Multiple models with scores
            models_data = []
            labels = []
            
            for model in data['models']:
                if 'scores' not in model or not model['scores']:
                    continue
                    
                models_data.append(model['scores'])
                labels.append(model.get('name', 'Unknown'))
            
            if not models_data:
                logger.error("No valid data for boxplot chart")
                return
                
            # Create boxplot
            self.sns.boxplot(data=models_data, ax=ax)
            ax.set_xticklabels(labels)
        else:
            # Simple key-value structure
            categories = list(data.keys())
            values = [data[c] for c in categories]
            
            # Create dataframe for seaborn
            df = self.pd.DataFrame()
            for i, category in enumerate(categories):
                if not values[i]:
                    continue
                category_data = values[i]
                df_cat = self.pd.DataFrame({
                    'category': [category] * len(category_data),
                    'value': category_data
                })
                df = df.append(df_cat)
                
            if df.empty:
                logger.error("No valid data for boxplot chart")
                return
                
            # Generate boxplot
            self.sns.boxplot(x='category', y='value', data=df, ax=ax)
        
        # Set labels
        ax.set_xlabel(data.get('x_label', ''))
        ax.set_ylabel(data.get('y_label', 'Value'))
    
    def _generate_heatmap_chart(self, ax, data: Dict[str, Any]) -> None:
        """
        Generate a heatmap chart.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes
            Axes to draw the chart on
        data : Dict[str, Any]
            Data for the chart
        """
        if 'matrix' not in data:
            logger.error("No matrix data for heatmap")
            return
            
        matrix = data['matrix']
        
        # Generate heatmap
        self.sns.heatmap(
            matrix, 
            ax=ax,
            annot=data.get('show_values', True),
            fmt=data.get('format', '.2f'),
            cmap=data.get('colormap', 'viridis'),
            xticklabels=data.get('x_labels', True),
            yticklabels=data.get('y_labels', True),
            cbar=data.get('show_colorbar', True)
        )
        
        # Set labels
        ax.set_xlabel(data.get('x_label', ''))
        ax.set_ylabel(data.get('y_label', ''))

    def generate_custom_chart(
        self,
        draw_function: Callable,
        data: Dict[str, Any],
        title: Optional[str] = None,
        figsize: tuple = (10, 6),
        **kwargs
    ) -> str:
        """
        Generate a custom chart using a provided drawing function (Template Method Pattern).

        **Phase 3 Sprint 9:** Flexible chart generation without modifying base class.

        This method encapsulates all the boilerplate for chart generation:
        - Figure/axes creation
        - Drawing (delegates to provided function)
        - Title configuration
        - Base64 encoding
        - Memory cleanup

        Parameters:
        -----------
        draw_function : Callable
            Function that draws the chart. Signature: draw_function(ax, data, **kwargs)
            The function receives:
            - ax: matplotlib.axes.Axes to draw on
            - data: Dict[str, Any] with chart data
            - **kwargs: Additional parameters passed through
        data : Dict[str, Any]
            Data for the chart (passed to draw_function)
        title : str, optional
            Chart title
        figsize : tuple, optional
            Figure size in inches (width, height), default (10, 6)
        **kwargs
            Additional keyword arguments passed to draw_function

        Returns:
        --------
        str : Base64 encoded image data URL (data:image/png;base64,...)

        Example:
        --------
        >>> def draw_my_chart(ax, data, color='blue'):
        ...     ax.plot(data['x'], data['y'], color=color)
        ...     ax.set_xlabel('X Axis')
        ...     ax.set_ylabel('Y Axis')
        >>>
        >>> chart = renderer.generate_custom_chart(
        ...     draw_my_chart,
        ...     data={'x': [1,2,3], 'y': [4,5,6]},
        ...     title='My Custom Chart',
        ...     color='red'
        ... )

        Benefits:
        ---------
        - Eliminates ~50 lines of boilerplate per custom chart
        - Consistent error handling and memory management
        - Subclasses can create charts without modifying base class
        - Easy to test drawing logic in isolation
        """
        if not self.has_visualization_libs:
            logger.error("Required libraries for visualization not available")
            return ""

        try:
            # Create figure and axes
            fig, ax = self.plt.subplots(figsize=figsize)

            # Call the custom drawing function
            draw_function(ax, data, **kwargs)

            # Set title if provided
            if title:
                ax.set_title(title, fontsize=14, fontweight='bold')

            # Save to buffer
            buf = io.BytesIO()
            fig.tight_layout()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)

            # Encode to base64
            img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Clean up
            self.plt.close(fig)

            logger.debug(f"Successfully generated custom chart: {title or 'untitled'}")
            return f"data:image/png;base64,{img_base64}"

        except Exception as e:
            logger.error(f"Error generating custom chart: {str(e)}", exc_info=True)
            return ""

    def _create_static_context(self, report_data: Dict[str, Any], test_type: str, css_content: str) -> Dict[str, Any]:
        """
        Create template context with common data for static reports.
        
        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
        css_content : str
            Combined CSS content
            
        Returns:
        --------
        Dict[str, Any] : Template context
        """
        try:
            # Get base64 encoded favicon and logo
            favicon_base64 = self.asset_manager.get_favicon_base64()
            logo_base64 = self.asset_manager.get_logo_base64()
        except Exception as e:
            logger.warning(f"Error loading images: {str(e)}")
            favicon_base64 = ""
            logo_base64 = ""
        
        # Get current timestamp if not provided
        timestamp = report_data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Base context that all static reports will have
        context = {
            # Complete report data for template access
            'report_data': report_data,
            
            # CSS content
            'css_content': css_content,
            
            # Basic metadata
            'model_name': report_data.get('model_name', 'Model'),
            'timestamp': timestamp,
            'current_year': datetime.datetime.now().year,
            'favicon_base64': favicon_base64,
            'logo': logo_base64,
            'block_title': f"{test_type.capitalize()} Analysis: {report_data.get('model_name', 'Model')}",
            
            # Main metrics for direct access in templates
            'model_type': report_data.get('model_type', 'Unknown Model'),
            'metric': report_data.get('metric', 'score'),
            'base_score': report_data.get('base_score', 0.0),
            
            # Feature details
            'feature_subset': report_data.get('feature_subset', []),
            'feature_subset_display': report_data.get('feature_subset_display', 'All Features'),
            
            # For component display logic
            'has_alternative_models': 'alternative_models' in report_data and bool(report_data['alternative_models']),
            
            # Test type information
            'test_type': test_type,
            'test_report_type': test_type,  # The type of test
            'report_type': 'static',  # Always static for this renderer
            
            # Error message (None by default)
            'error_message': None,
            
            # Static charts container
            'charts': {}
        }
        
        return context
    
    def _load_static_css_content(self, report_type: str = "static") -> str:
        """
        Load and combine CSS files for static reports using CSSManager.

        Parameters:
        -----------
        report_type : str, optional
            Type of report (e.g., 'uncertainty', 'robustness', 'resilience')
            Defaults to 'static' for generic static styles

        Returns:
        --------
        str : Combined CSS content compiled by CSSManager with static additions
        """
        try:
            # Use CSSManager to get compiled CSS for the report type
            css_content = self.css_manager.get_compiled_css(report_type)

            # Add static-specific styles and overrides
            css_content += """

            /* ========================================================================== */
            /* STATIC REPORT ADDITIONS */
            /* ========================================================================== */

            /* Additional styles for static reports */
            .chart-container {
                margin: 2rem 0;
                text-align: center;
                display: block !important; /* Override any display:none from interactive CSS */
            }

            .chart-container img {
                max-width: 100%;
                height: auto;
                border: 1px solid var(--border-color, #ddd);
                border-radius: 4px;
            }

            /* Ensure all chart containers are visible in static reports */
            .chart-container.active,
            .chart-container {
                display: block !important;
            }

            /* Print styles for static reports */
            @media print {
                .chart-container {
                    page-break-inside: avoid;
                }

                .section {
                    page-break-inside: avoid;
                }
            }
            """

            logger.info(f"CSS compiled successfully using CSSManager for static {report_type} report")
            return css_content

        except Exception as e:
            logger.warning(f"Error loading CSS with CSSManager: {str(e)}, falling back to basic CSS")

            # Fallback to basic CSS if CSSManager fails
            return self._get_fallback_static_css()

    def _get_fallback_static_css(self) -> str:
        """
        Fallback basic CSS for static reports if CSSManager fails.

        Returns:
        --------
        str : Basic CSS content for static reports
        """
        return """
        /* Fallback base styles for static reports */
        :root {
            --primary-color: #1b78de;
            --secondary-color: #2c3e50;
            --success-color: #28a745;
            --danger-color: #dc3545;
            --warning-color: #f39c12;
            --info-color: #17a2b8;
            --light-color: #f8f9fa;
            --dark-color: #343a40;
            --text-color: #333;
            --text-muted: #6c757d;
            --border-color: #ddd;
            --background-color: #f8f9fa;
            --card-bg: #fff;
            --header-bg: #ffffff;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        html, body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            font-size: 16px;
            line-height: 1.5;
            color: var(--text-color);
            background-color: var(--background-color);
        }

        h1, h2, h3, h4, h5, h6 {
            margin-bottom: 1rem;
            font-weight: 500;
            line-height: 1.2;
        }

        p {
            margin-bottom: 1rem;
        }

        .header {
            background-color: white;
            padding: 1rem;
            border-bottom: 1px solid var(--border-color);
            text-align: center;
            margin-bottom: 2rem;
        }

        .header h1 {
            font-size: 1.75rem;
            color: var(--primary-color);
        }

        .report-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 1rem;
        }

        .section {
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            padding: 1.5rem;
            margin-bottom: 2rem;
        }

        .section-title {
            border-left: 4px solid var(--primary-color);
            padding-left: 0.75rem;
            margin-bottom: 1.5rem;
            font-size: 1.5rem;
        }

        .chart-container {
            margin: 2rem 0;
            text-align: center;
        }

        .chart-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-color);
            border-radius: 4px;
        }

        .metrics-card {
            background-color: white;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }

        .metrics-card h3 {
            margin-bottom: 0.5rem;
            font-size: 1.1rem;
            color: var(--secondary-color);
        }

        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary-color);
        }

        .metric-label {
            font-size: 0.875rem;
            color: #666;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }

        th, td {
            padding: 0.75rem;
            text-align: left;
            border: 1px solid var(--border-color);
        }

        th {
            background-color: #f5f5f5;
            font-weight: 600;
        }

        tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        .footer {
            text-align: center;
            padding: 2rem 0;
            margin-top: 3rem;
            color: #666;
            font-size: 0.875rem;
            border-top: 1px solid var(--border-color);
        }
        """
    
    def _ensure_output_dir(self, file_path: str) -> None:
        """
        Ensure output directory exists.
        
        Parameters:
        -----------
        file_path : str
            Path where the HTML report will be saved
        """
        output_dir = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Output directory ensured: {output_dir}")
    
    def _write_report(self, rendered_html: str, file_path: str) -> str:
        """
        Write rendered HTML to file.
        
        Parameters:
        -----------
        rendered_html : str
            Rendered HTML content
        file_path : str
            Path where the HTML report will be saved
            
        Returns:
        --------
        str : Path to the written file
        """
        # Ensure output directory exists
        self._ensure_output_dir(file_path)
        
        # Write to file with explicit UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(rendered_html)
            
        logger.info(f"Static report saved to: {file_path}")
        return file_path