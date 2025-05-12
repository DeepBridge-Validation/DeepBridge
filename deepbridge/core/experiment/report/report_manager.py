"""
Report generation module for experiment results - main manager.
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class ReportManager:
    """
    Handles the generation of HTML reports from experiment results.
    Coordinates the process without implementing specifics.
    """
    
    def __init__(self, templates_dir: Optional[str] = None):
        """
        Initialize the report manager.
        
        Parameters:
        -----------
        templates_dir : str, optional
            Directory containing report templates. If None, use the default
            templates directory in deepbridge/templates.
            
        Raises:
        -------
        FileNotFoundError: If templates directory doesn't exist
        """
        # Import required modules
        from .template_manager import TemplateManager
        from .asset_manager import AssetManager
        
        # Set up templates directory
        if templates_dir is None:
            # Use default templates directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.templates_dir = os.path.join(base_dir, 'templates')
        else:
            self.templates_dir = templates_dir
            
        if not os.path.exists(self.templates_dir):
            raise FileNotFoundError(f"Templates directory not found: {self.templates_dir}")
        
        logger.info(f"Using templates directory: {self.templates_dir}")
        
        # Initialize managers
        self.template_manager = TemplateManager(self.templates_dir)
        self.asset_manager = AssetManager(self.templates_dir)
        
        # Import renderers
        from .renderers import (
            RobustnessRenderer,
            UncertaintyRenderer,
            ResilienceRenderer,
            HyperparameterRenderer
        )

        # Import static renderers
        try:
            from .renderers.static import StaticRobustnessRenderer, StaticUncertaintyRenderer, StaticResilienceRenderer
            self.has_static_renderers = True
        except ImportError:
            logger.warning("Static renderers not available, will use interactive renderers for all reports")
            self.has_static_renderers = False

        # Set up renderers for different report types
        self.renderers = {
            'robustness': RobustnessRenderer(self.template_manager, self.asset_manager),
            'uncertainty': UncertaintyRenderer(self.template_manager, self.asset_manager),
            'resilience': ResilienceRenderer(self.template_manager, self.asset_manager),
            'hyperparameter': HyperparameterRenderer(self.template_manager, self.asset_manager),
            'hyperparameters': HyperparameterRenderer(self.template_manager, self.asset_manager)
        }

        # Set up static renderers if available
        self.static_renderers = {}
        if self.has_static_renderers:
            self.static_renderers = {
                'robustness': StaticRobustnessRenderer(self.template_manager, self.asset_manager),
                'uncertainty': StaticUncertaintyRenderer(self.template_manager, self.asset_manager),
                'resilience': StaticResilienceRenderer(self.template_manager, self.asset_manager)
                # Add other static renderers as they are implemented
            }

    def generate_report(self, test_type: str, results: Dict[str, Any], file_path: str, model_name: str = "Model", report_type: str = "interactive", save_chart: bool = False) -> str:
        """
        Generate report for the specified test type.

        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', 'resilience', 'hyperparameter')
        results : Dict[str, Any]
            Test results
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
        NotImplementedError: If the test type is not supported
        ValueError: If report generation fails
        """
        test_type_lower = test_type.lower()
        report_type_lower = report_type.lower()

        # Validate report_type
        if report_type_lower not in ["interactive", "static"]:
            logger.warning(f"Invalid report_type '{report_type}', defaulting to 'interactive'")
            report_type_lower = "interactive"

        # Handle static report request
        if report_type_lower == "static" and self.has_static_renderers:
            if test_type_lower in self.static_renderers:
                renderer = self.static_renderers[test_type_lower]
                logger.info(f"Using static renderer for {test_type} report")
            else:
                logger.warning(f"Static renderer for {test_type} is not implemented, falling back to interactive renderer")
                if test_type_lower not in self.renderers:
                    raise NotImplementedError(f"Report generation for test type '{test_type}' is not implemented")
                renderer = self.renderers[test_type_lower]
        else:
            # Use interactive renderer (default)
            if test_type_lower not in self.renderers:
                raise NotImplementedError(f"Report generation for test type '{test_type}' is not implemented")
            renderer = self.renderers[test_type_lower]

        try:
            # Generate the report using the appropriate renderer
            report_path = renderer.render(results, file_path, model_name, report_type_lower, save_chart)
            logger.info(f"Report generated and saved to: {report_path} (type: {report_type_lower})")
            return report_path
        except Exception as e:
            logger.error(f"Error generating {test_type} report: {str(e)}")
            raise ValueError(f"Failed to generate {test_type} report: {str(e)}")