"""
Hyperparameter report renderer.
"""

import os
import logging
from typing import Dict, Any, Optional, List

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class HyperparameterRenderer:
    """
    Renderer for hyperparameter test reports.
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
        from ..transformers.hyperparameter import HyperparameterDataTransformer
        self.data_transformer = HyperparameterDataTransformer()
    
    def render(self, results: Dict[str, Any], file_path: str, model_name: str = "Model", report_type: str = "interactive") -> str:
        """
        Render hyperparameter report from results data.

        Parameters:
        -----------
        results : Dict[str, Any]
            Hyperparameter test results
        file_path : str
            Path where the HTML report will be saved
        model_name : str, optional
            Name of the model for display in the report
        report_type : str, optional
            Type of report to generate ('interactive' or 'static')
            
        Returns:
        --------
        str : Path to the generated report
        
        Raises:
        -------
        FileNotFoundError: If template or assets not found
        ValueError: If required data missing
        """
        logger.info(f"Generating hyperparameter report to: {file_path}")
        
        try:
            # Find template
            template_paths = self.template_manager.get_template_paths("hyperparameter")
            template_path = self.template_manager.find_template(template_paths)
            
            if not template_path:
                raise FileNotFoundError(f"No template found for hyperparameter report in: {template_paths}")
            
            logger.info(f"Using template: {template_path}")
            
            # Find CSS and JS paths
            css_dir = self.asset_manager.find_css_path("hyperparameter")
            js_dir = self.asset_manager.find_js_path("hyperparameter")
            
            if not css_dir:
                raise FileNotFoundError("CSS directory not found for hyperparameter report")
            
            if not js_dir:
                raise FileNotFoundError("JavaScript directory not found for hyperparameter report")
            
            # Get CSS and JS content
            css_content = self.asset_manager.get_css_content(css_dir)
            js_content = self.asset_manager.get_js_content(js_dir)
            
            # Load the template
            template = self.template_manager.load_template(template_path)
            
            # Transform the data
            report_data = self.data_transformer.transform(results, model_name)
            
            # Create template context
            context = self.base_renderer._create_context(report_data, "hyperparameter", css_content, js_content, report_type)
            
            # Add hyperparameter-specific context
            context.update({
                'importance_scores': report_data.get('importance_scores', {}),
                'tuning_order': report_data.get('tuning_order', []),
                'importance_results': report_data.get('importance_results', []),
                'optimization_results': report_data.get('optimization_results', [])
            })
            
            # Render the template
            rendered_html = self.template_manager.render_template(template, context)
            
            # Write the report to file
            return self.base_renderer._write_report(rendered_html, file_path)
            
        except Exception as e:
            logger.error(f"Error generating hyperparameter report: {str(e)}")
            raise ValueError(f"Failed to generate hyperparameter report: {str(e)}")