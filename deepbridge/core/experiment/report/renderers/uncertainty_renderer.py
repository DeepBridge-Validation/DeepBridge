"""
Uncertainty report renderer.
"""

import os
import logging
from typing import Dict, Any, Optional, List

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class UncertaintyRenderer:
    """
    Renderer for uncertainty test reports.
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
        from .base_renderer import BaseRenderer
        self.base_renderer = BaseRenderer(template_manager, asset_manager)
        self.template_manager = template_manager
        self.asset_manager = asset_manager
        
        # Import specific data transformer
        from ..transformers.uncertainty import UncertaintyDataTransformer
        self.data_transformer = UncertaintyDataTransformer()
    
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model") -> str:
        """
        Render uncertainty report from results data.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Uncertainty test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
            
        Returns:
        --------
        str : Path to the generated report
        
        Raises:
        -------
        FileNotFoundError: If template or assets not found
        ValueError: If required data missing
        """
        logger.info(f"Generating uncertainty report to: {file_path}")
        
        try:
            # Find template
            template_paths = self.template_manager.get_template_paths("uncertainty")
            template_path = self.template_manager.find_template(template_paths)
            
            if not template_path:
                raise FileNotFoundError(f"No template found for uncertainty report in: {template_paths}")
            
            logger.info(f"Using template: {template_path}")
            
            # Find CSS and JS paths
            css_dir = self.asset_manager.find_css_path("uncertainty")
            js_dir = self.asset_manager.find_js_path("uncertainty")
            
            if not css_dir:
                raise FileNotFoundError("CSS directory not found for uncertainty report")
            
            if not js_dir:
                raise FileNotFoundError("JavaScript directory not found for uncertainty report")
            
            # Get CSS and JS content
            css_content = self.asset_manager.get_css_content(css_dir)
            js_content = self.asset_manager.get_js_content(js_dir)
            
            # Load the template
            template = self.template_manager.load_template(template_path)
            
            # Transform the data
            report_data = self.data_transformer.transform(results, model_name)
            
            # Create template context
            context = self.base_renderer._create_context(report_data, "uncertainty", css_content, js_content)
            
            # Add uncertainty-specific context with default values for all variables
            uncertainty_score = report_data.get('uncertainty_score', 0)
            avg_coverage = report_data.get('avg_coverage', 0)
            
            context.update({
                # Core metrics with defaults
                'uncertainty_score': uncertainty_score,
                'robustness_score': uncertainty_score,  # Backward compatibility
                'coverage_score': avg_coverage,
                'calibration_error': report_data.get('calibration_error', 0),
                'sharpness': report_data.get('avg_width', 0),
                'consistency': report_data.get('consistency', 0),
                'avg_coverage': avg_coverage,
                'avg_width': report_data.get('avg_width', 0),
                
                # Metadata
                'method': report_data.get('method', 'crqr'),
                'alpha_levels': report_data.get('alpha_levels', []),
                'test_type': 'uncertainty',  # Explicit test type
                
                # Additional context to ensure backward compatibility
                'features': report_data.get('features', []),
                'metrics': report_data.get('metrics', {}),
                'metrics_details': report_data.get('metrics_details', {})
            })
            
            # Render the template
            rendered_html = self.template_manager.render_template(template, context)
            
            # Write the report to file
            return self.base_renderer._write_report(rendered_html, file_path)
            
        except Exception as e:
            logger.error(f"Error generating uncertainty report: {str(e)}")
            raise ValueError(f"Failed to generate uncertainty report: {str(e)}")