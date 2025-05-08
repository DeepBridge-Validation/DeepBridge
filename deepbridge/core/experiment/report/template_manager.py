"""
Template management module for report generation.
"""

import os
import logging
from typing import Optional, List

# Try to import markupsafe for safe rendering
try:
    from markupsafe import Markup
except ImportError:
    # Fallback implementation if markupsafe not available
    class Markup(str):
        def __new__(cls, base=""):
            return str.__new__(cls, base)

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class TemplateManager:
    """
    Manages loading and processing of report templates.
    """
    
    def __init__(self, templates_dir: str):
        """
        Initialize the template manager.
        
        Parameters:
        -----------
        templates_dir : str
            Directory containing report templates
            
        Raises:
        -------
        ImportError: If Jinja2 is not installed
        """
        self.templates_dir = templates_dir
        
        # Import Jinja2
        try:
            import jinja2
            self.jinja2 = jinja2
        except ImportError:
            logger.error("Jinja2 is required for HTML report generation")
            raise ImportError(
                "Jinja2 is required for HTML report generation. "
                "Please install it with: pip install jinja2"
            )
        
        # Set up Jinja2 environment with explicit UTF-8 encoding
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir, encoding='utf-8'),
            autoescape=jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def find_template(self, template_paths: List[str]) -> str:
        """
        Find the template from the primary path.
        
        Parameters:
        -----------
        template_paths : List[str]
            List of possible template paths to check
            
        Returns:
        --------
        str : Path to the found template
        
        Raises:
        -------
        FileNotFoundError: If the template is not found
        """
        path = template_paths[0]  # Apenas usa o primeiro caminho (que será o principal)
        if os.path.exists(path):
            logger.info(f"Found template at: {path}")
            return path
        
        raise FileNotFoundError(f"Template not found at the specified path: {path}")
    
    def get_template_paths(self, test_type: str) -> List[str]:
        """
        Get potential template paths for the specified test type.
        
        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
            
        Returns:
        --------
        List[str] : List of potential template paths
        """
        return [
            # Primary path - report_types directory
            os.path.join(self.templates_dir, f'report_types/{test_type}/index.html'),
        ]
    
    def load_template(self, template_path: str):
        """
        Load a template from the specified path.
        
        Parameters:
        -----------
        template_path : str
            Path to the template file
            
        Returns:
        --------
        Template : Jinja2 Template object
            
        Raises:
        -------
        FileNotFoundError: If template file doesn't exist
        """
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        # Get the directory containing the template
        template_dir = os.path.dirname(template_path)
        
        # Create a file system loader for this directory and the templates root
        # Add both the template's directory and the templates root directory to the search path
        # This allows templates to include files from the common directory
        loader = self.jinja2.FileSystemLoader([template_dir, self.templates_dir], encoding='utf-8')
        
        # Create a new environment with this loader
        env = self.jinja2.Environment(
            loader=loader,
            autoescape=self.jinja2.select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filter to prevent autoescape in JavaScript blocks
        env.filters['safe_js'] = lambda s: Markup(s)
        
        # Load the template
        return env.get_template(os.path.basename(template_path))
    
    def render_template(self, template, context: dict) -> str:
        """
        Render a template with the provided context.
        
        Parameters:
        -----------
        template : Template
            Jinja2 Template object
        context : dict
            Context data for template rendering
            
        Returns:
        --------
        str : Rendered template content
        """
        return template.render(**context)