"""
Base renderer for generating HTML reports.
"""

import os
import json
import logging
import datetime
import math
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
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Original data dictionary
            
        Returns:
        --------
        Dict[str, Any] : Serializable data
        """
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
        
        Parameters:
        -----------
        alt_models : Dict[str, Any]
            Alternative models data
            
        Returns:
        --------
        Dict[str, Any] : Serializable alternative models data
        """
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