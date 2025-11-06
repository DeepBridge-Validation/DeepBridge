"""
Base renderer for generating HTML reports.
"""

import os
import json
import logging
import datetime
import math
import warnings
from typing import Dict, Any, Optional

# Configure logger
logger = logging.getLogger("deepbridge.reports")

# Import JSON formatter
from ..utils.json_formatter import JsonFormatter

class BaseRenderer:
    """
    Base class for report renderers.
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

        # Import data transformers
        from ..base import DataTransformer
        self.data_transformer = DataTransformer()

        # Import CSS Manager
        from ..css_manager import CSSManager
        self.css_manager = CSSManager()
    
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model", report_type: str = "interactive", save_chart: bool = False) -> str:
        """
        Render report from results data.

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
    
    def _json_serializer(self, obj: Any) -> Any:
        """
        JSON serializer for objects not serializable by default json code.
        
        Parameters:
        -----------
        obj : Any
            Object to serialize
            
        Returns:
        --------
        Any : Serialized object
            
        Raises:
        -------
        TypeError: If object cannot be serialized
        """
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        # Return None for any other unserializable types to prevent exceptions
        try:
            json.dumps(obj)
            return obj
        except:
            logger.warning(f"Unserializable type {type(obj)} detected, defaulting to None")
            return None
    
    def _create_serializable_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a serializable copy of the data with defaults for undefined values.

        .. deprecated:: 2.0
            This method is deprecated and will be removed in a future version.
            Use JsonFormatter.format_for_javascript() directly instead.

        Parameters:
        -----------
        data : Dict[str, Any]
            Original data dictionary

        Returns:
        --------
        Dict[str, Any] : Serializable data
        """
        warnings.warn(
            "_create_serializable_data is deprecated and will be removed in a future version. "
            "Use JsonFormatter.format_for_javascript() directly instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if data is None:
            return {}
        
        serializable = {}
        
        # Process common report attributes with appropriate defaults
        serializable.update({
            # Basic metadata
            'model_name': data.get('model_name', 'Model'),
            'model_type': data.get('model_type', 'Unknown'),
            'timestamp': data.get('timestamp', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            'metric': data.get('metric', 'accuracy'),
            'base_score': data.get('base_score', 0.0),
            
            # Common metrics
            'robustness_score': data.get('robustness_score', data.get('resilience_score', data.get('uncertainty_score', 0.0))),
            'resilience_score': data.get('resilience_score', data.get('robustness_score', 0.0)),
            'uncertainty_score': data.get('uncertainty_score', data.get('robustness_score', 0.0)),
            
            # Feature data
            'feature_importance': data.get('feature_importance', {}),
            'model_feature_importance': data.get('model_feature_importance', {}),
            'feature_subset': data.get('feature_subset', []),
            'feature_subset_display': data.get('feature_subset_display', 'All Features'),
            'features': data.get('features', []),
            
            # Impact metrics
            'raw_impact': data.get('raw_impact', 0.0),
            'quantile_impact': data.get('quantile_impact', 0.0),
            'avg_performance_gap': data.get('avg_performance_gap', 0.0),
            
            # Results and metrics
            'metrics': data.get('metrics', {}),
            'metrics_details': data.get('metrics_details', {}),
            
            # Resilience-specific fields
            'distance_metrics': data.get('distance_metrics', []),
            'alphas': data.get('alphas', []),
            'shift_scenarios': data.get('shift_scenarios', []),
            'sensitive_features': data.get('sensitive_features', []),
            'baseline_dataset': data.get('baseline_dataset', 'Baseline'),
            'target_dataset': data.get('target_dataset', 'Target'),
            
            # Clean copy of alternative models data if exists
            'alternative_models': self._process_alternative_models(data.get('alternative_models', {}))
        })
        
        # Copy any other keys that may be needed by templates
        for key, value in data.items():
            if key not in serializable:
                # Apply sensible defaults based on value type
                if value is None:
                    if key.endswith('_score') or key.endswith('_impact') or key.endswith('_gap'):
                        serializable[key] = 0.0
                    elif key.endswith('metrics') or key.startswith('feature'):
                        serializable[key] = []
                    else:
                        serializable[key] = None
                else:
                    serializable[key] = value
        
        return serializable
    
    def _process_alternative_models(self, alt_models: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process alternative models data to ensure it's serializable.

        .. deprecated:: 2.0
            This method is deprecated and will be removed in a future version.
            Use JsonFormatter.format_for_javascript() directly instead.

        Parameters:
        -----------
        alt_models : Dict[str, Any]
            Alternative models data

        Returns:
        --------
        Dict[str, Any] : Serializable alternative models data
        """
        warnings.warn(
            "_process_alternative_models is deprecated and will be removed in a future version. "
            "Use JsonFormatter.format_for_javascript() directly instead.",
            DeprecationWarning,
            stacklevel=2
        )

        if not alt_models:
            return {}
            
        result = {}
        for model_name, model_data in alt_models.items():
            if not model_data:
                continue
                
            # Create serializable copy of model data with defaults
            serializable_model = {
                'model_name': model_data.get('model_name', model_name),
                'model_type': model_data.get('model_type', 'Unknown'),
                'base_score': model_data.get('base_score', 0.0),
                'robustness_score': model_data.get('robustness_score', 0.0),
                'resilience_score': model_data.get('resilience_score', 0.0),
                'raw_impact': model_data.get('raw_impact', 0.0),
                'metrics': model_data.get('metrics', {})
            }
            
            result[model_name] = serializable_model
            
        return result
    
    def _create_context(self, report_data: Dict[str, Any], test_type: str,
                       css_content: str, js_content: str, report_type: str = "interactive") -> Dict[str, Any]:
        """
        Create template context with common data.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
        css_content : str
            Combined CSS content
        js_content : str
            Combined JavaScript content
        report_type : str, optional
            Type of report ('interactive' or 'static')

        Returns:
        --------
        Dict[str, Any] : Template context
        """
        warnings.warn(
            "_create_context is deprecated. Use _create_base_context() with _get_assets() instead.",
            DeprecationWarning,
            stacklevel=2
        )

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
        
        # Base context that all reports will have
        context = {
            # Complete report data for template access
            'report_data': report_data,
            # JSON string of report data for JavaScript processing - create a safe copy with defaults
            'report_data_json': JsonFormatter.format_for_javascript(self._create_serializable_data(report_data)),
            
            # CSS and JS content
            'css_content': css_content,
            'js_content': js_content,  # Fixed variable name to match usage
            
            # Basic metadata
            'model_name': report_data.get('model_name', 'Model'),
            'timestamp': timestamp,
            'current_year': datetime.datetime.now().year,
            'favicon_base64': favicon_base64,  # Fixed variable name to match usage in template
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
            'report_type': report_type,  # The type of report (interactive or static)
            
            # Error message (None by default)
            'error_message': None
        }
        
        return context
    
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
        warnings.warn(
            "_write_report is deprecated. Use _write_html() instead.",
            DeprecationWarning,
            stacklevel=2
        )

        # Ensure output directory exists
        self._ensure_output_dir(file_path)
        
        # Unescape any HTML entities that might affect JavaScript
        html_fixed = self._fix_html_entities(rendered_html)
        
        # Write to file with explicit UTF-8 encoding
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_fixed)
            
        logger.info(f"Report saved to: {file_path}")
        return file_path
        
    def _fix_html_entities(self, html_content: str) -> str:
        """
        Fix HTML entities in the content, particularly for JavaScript and CSS.
        
        Parameters:
        -----------
        html_content : str
            HTML content with potentially escaped entities
            
        Returns:
        --------
        str : Fixed HTML content
        """
        # Replace common HTML entities
        replacements = {
            '&#34;': '"',
            '&#39;': "'",
            '&quot;': '"',
            '&lt;': '<',
            '&gt;': '>',
            '&amp;': '&'
        }
        
        # Process the HTML to find and fix JavaScript and CSS sections
        result = []
        in_script = False
        in_style = False
        
        for line in html_content.split('\n'):
            # Check if entering or exiting script section
            if '<script>' in line:
                in_script = True
                result.append(line)
                continue
            elif '</script>' in line:
                in_script = False
                result.append(line)
                continue
                
            # Check if entering or exiting style section
            if '<style>' in line:
                in_style = True
                result.append(line)
                continue
            elif '</style>' in line:
                in_style = False
                result.append(line)
                continue
            
            # Apply replacements in script or style sections
            if in_script or in_style:
                # Replace entities in script and style sections
                for entity, char in replacements.items():
                    line = line.replace(entity, char)
                result.append(line)
            else:
                result.append(line)
        
        # Also fix font-family declarations that often get mangled
        fixed_content = '\n'.join(result)
        
        # Fix font-family declarations with single quotes
        fixed_content = fixed_content.replace("font-family: &#39;", "font-family: '")
        fixed_content = fixed_content.replace("&#39;, ", "', ")
        fixed_content = fixed_content.replace("&#39;;", "';")
        
        # Fix font-family declarations with double quotes
        fixed_content = fixed_content.replace("font-family: &quot;", 'font-family: "')
        fixed_content = fixed_content.replace("&quot;, ", '", ')
        fixed_content = fixed_content.replace("&quot;;", '";')

        return fixed_content

    def _get_css_content(self, report_type: str) -> str:
        """
        Get compiled CSS content for a specific report type using CSSManager.

        Parameters:
        -----------
        report_type : str
            Type of report ('uncertainty', 'robustness', 'resilience', etc.)

        Returns:
        --------
        str : Compiled CSS (base + components + custom for report type)
        """
        try:
            # Use CSSManager to compile CSS layers
            compiled_css = self.css_manager.get_compiled_css(report_type)
            logger.info(f"CSS compiled successfully using CSSManager for {report_type}: {len(compiled_css)} chars")
            return compiled_css
        except Exception as e:
            logger.error(f"Error loading CSS with CSSManager for {report_type}: {str(e)}")

            # Fallback: return minimal CSS if CSSManager fails
            logger.warning(f"Using fallback minimal CSS for {report_type}")
            return """
            :root {
                --primary-color: #1b78de;
                --secondary-color: #2c3e50;
                --success-color: #28a745;
                --danger-color: #dc3545;
                --background-color: #f8f9fa;
            }
            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background-color: var(--background-color);
                margin: 0;
                padding: 20px;
            }
            """

    def _safe_json_dumps(self, data: Dict[str, Any]) -> str:
        """
        Safely serialize data to JSON, handling NaN, infinity, and other special values.

        This method is now a wrapper around JsonFormatter for consistency,
        but kept for backwards compatibility.

        Parameters:
        -----------
        data : Dict[str, Any]
            Data to serialize

        Returns:
        --------
        str : JSON string
        """
        # Use JsonFormatter for proper handling
        return JsonFormatter.format_for_javascript(data)

    def _write_html(self, html: str, file_path: str) -> str:
        """
        Write HTML content to file.

        Parameters:
        -----------
        html : str
            HTML content to write
        file_path : str
            Path where the HTML report will be saved

        Returns:
        --------
        str : Path to the written file
        """
        self._ensure_output_dir(file_path)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Report saved to: {file_path}")
        return file_path

    # ==================================================================================
    # Template Method Pattern - Phase 2 Consolidation
    # ==================================================================================

    def _load_template(self, test_type: str, report_type: str = "interactive"):
        """
        Load template for specific test type and report type.

        Parameters:
        -----------
        test_type : str
            Type of test ('uncertainty', 'robustness', 'resilience', etc.')
        report_type : str, optional
            Type of report ('interactive' or 'static')

        Returns:
        --------
        Template : Loaded Jinja2 template

        Raises:
        -------
        FileNotFoundError: If template not found
        """
        template_paths = self.template_manager.get_template_paths(
            test_type, report_type
        )
        template_path = self.template_manager.find_template(template_paths)

        if not template_path:
            raise FileNotFoundError(
                f"Template not found for {test_type}/{report_type}"
            )

        logger.debug(f"Loading template: {template_path}")
        return self.template_manager.load_template(template_path)

    def _get_assets(self, test_type: str) -> Dict[str, str]:
        """
        Get all assets for report (CSS, JS, images).

        Parameters:
        -----------
        test_type : str
            Type of test ('uncertainty', 'robustness', etc.')

        Returns:
        --------
        Dict[str, str] : Dictionary with asset contents:
            - css_content: Compiled CSS
            - js_content: JavaScript code
            - logo: Base64 encoded logo
            - favicon_base64: Base64 encoded favicon
        """
        return {
            'css_content': self._get_css_content(test_type),
            'js_content': self._get_js_content(test_type),
            'logo': self.asset_manager.get_logo_base64(),
            'favicon_base64': self.asset_manager.get_favicon_base64()
        }

    def _get_js_content(self, test_type: str) -> str:
        """
        Get JavaScript content for report.

        Subclasses can override this to provide custom JS.
        Default implementation returns basic tab navigation.

        Parameters:
        -----------
        test_type : str
            Type of test (for future use)

        Returns:
        --------
        str : JavaScript code
        """
        return """
        // Basic tab navigation
        function initTabs() {
            const tabButtons = document.querySelectorAll('.tab-button');
            const tabContents = document.querySelectorAll('.tab-content');

            tabButtons.forEach(button => {
                button.addEventListener('click', () => {
                    const targetTab = button.dataset.tab;

                    // Deactivate all
                    tabButtons.forEach(btn => btn.classList.remove('active'));
                    tabContents.forEach(content => content.classList.remove('active'));

                    // Activate target
                    button.classList.add('active');
                    document.getElementById(targetTab).classList.add('active');
                });
            });
        }

        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing report...');

            // Initialize tabs
            initTabs();

            // Render charts if data available
            if (window.reportData && window.reportData.charts) {
                renderCharts(window.reportData.charts);
            }
        });

        function renderCharts(charts) {
            // Render all Plotly charts
            for (const [chartName, chartData] of Object.entries(charts)) {
                const elementId = 'chart-' + chartName.replace(/_/g, '-');
                const element = document.getElementById(elementId);

                if (element && chartData.data && chartData.data.length > 0) {
                    // Ensure layout is responsive
                    const layout = {...chartData.layout};
                    layout.autosize = true;
                    delete layout.width; // Remove fixed width

                    // Render with responsive config
                    Plotly.newPlot(element, chartData.data, layout, {
                        responsive: true,
                        displayModeBar: false
                    }).then(() => {
                        // Force resize on window resize
                        window.addEventListener('resize', () => {
                            Plotly.Plots.resize(element);
                        });
                    });
                }
            }
        }
        """

    def _render_template(self, template, context: Dict[str, Any]) -> str:
        """
        Render Jinja2 template with context.

        Parameters:
        -----------
        template : Template
            Loaded Jinja2 template
        context : Dict[str, Any]
            Template context variables

        Returns:
        --------
        str : Rendered HTML
        """
        logger.debug(f"Rendering template with context keys: {list(context.keys())}")
        return self.template_manager.render_template(template, context)

    def _create_base_context(self, report_data: Dict[str, Any],
                             test_type: str, assets: Dict[str, str]) -> Dict[str, Any]:
        """
        Create base context common to ALL reports.

        This method creates the foundational context that every report needs.
        Subclasses should call this and then add their specific context.

        Parameters:
        -----------
        report_data : Dict[str, Any]
            Transformed report data
        test_type : str
            Type of test ('uncertainty', 'robustness', etc.')
        assets : Dict[str, str]
            Pre-loaded assets (from _get_assets)

        Returns:
        --------
        Dict[str, Any] : Base context with common fields
        """
        import datetime

        return {
            # Data
            'report_data': report_data,
            'report_data_json': self._safe_json_dumps(report_data),

            # Assets
            'css_content': assets['css_content'],
            'js_content': assets['js_content'],
            'logo': assets['logo'],
            'favicon_base64': assets['favicon_base64'],

            # Metadata
            'model_name': report_data.get('model_name', 'Model'),
            'model_type': report_data.get('model_type', 'Unknown'),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'test_type': test_type,
            'current_year': datetime.datetime.now().year
        }