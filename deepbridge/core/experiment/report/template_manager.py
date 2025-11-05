"""
Template management module for report generation.

**Phase 3 Sprint 9:** Added performance caching for template path resolution.
"""

import os
import logging
from typing import Optional, List
from functools import lru_cache

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

        # Add safe numeric conversion filters to the global environment
        self._add_safe_filters(self.jinja_env)

    def _add_safe_filters(self, env):
        """Add safe filters to a Jinja2 environment."""
        # Safe numeric conversion filters
        def safe_float(value, default=0.0):
            """Safely convert a value to float."""
            if value is None:
                return default
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                # Skip error messages
                if 'error' in value.lower() or 'classification' in value.lower():
                    return default
                try:
                    # Remove common formatting characters
                    cleaned = value.strip().replace('%', '').replace(',', '')
                    return float(cleaned)
                except (ValueError, TypeError):
                    return default
            return default

        def safe_round(value, precision=2):
            """Safely round a numeric value."""
            numeric_value = safe_float(value, 0.0)
            try:
                return round(numeric_value, precision)
            except (ValueError, TypeError):
                return 0.0

        def safe_multiply(x, y=100):
            """Safely multiply two values."""
            x_val = safe_float(x, 0.0)
            y_val = safe_float(y, 100 if y == 100 else 0.0)
            return x_val * y_val

        # Register the safe filters
        env.filters['safe_float'] = safe_float
        env.filters['safe_round'] = safe_round
        env.filters['safe_multiply'] = safe_multiply

        # Also add the safe_js and abs_value filters
        env.filters['safe_js'] = lambda s: Markup(s)
        env.filters['abs_value'] = lambda x: abs(safe_float(x)) if x is not None else 0.0

        # Add format_number filter for fairness reports
        def format_number(value):
            """Format number with thousands separator."""
            try:
                return f"{int(value):,}"
            except (ValueError, TypeError):
                return value

        env.filters['format_number'] = format_number

    @lru_cache(maxsize=64)
    def _find_template_cached(self, template_paths: tuple) -> str:
        """
        Internal cached template finder (uses tuple for hashability).

        **Phase 3 Sprint 9:** Cached to avoid repeated os.path.exists() calls.
        """
        # Try each path in order until we find an existing template
        for path in template_paths:
            if os.path.exists(path):
                logger.info(f"Found template at: {path}")
                return path

        # If no template is found, raise an error with all paths that were checked
        paths_str = '\n  - '.join(template_paths)
        raise FileNotFoundError(f"Template not found at any of the specified paths:\n  - {paths_str}")

    def find_template(self, template_paths: List[str]) -> str:
        """
        Find the template from the list of possible paths.

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
        # Convert list to tuple for caching, then call cached version
        return self._find_template_cached(tuple(template_paths))
    
    @lru_cache(maxsize=64)
    def get_template_paths(self, test_type: str, report_type: str = "interactive") -> List[str]:
        """
        Get potential template paths for the specified test type and report type.

        **Phase 3 Sprint 9:** Cached to avoid rebuilding paths for same test_type/report_type.

        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.')
        report_type : str, optional
            Type of report ('interactive' or 'static')

        Returns:
        --------
        List[str] : List of potential template paths
        """
        if report_type == "static":
            return [
                # Static report template path
                os.path.join(self.templates_dir, f'report_types/{test_type}/static/index.html'),
                # Fallback to default template if static not found
                os.path.join(self.templates_dir, f'report_types/{test_type}/index.html'),
            ]
        else:
            return [
                # Interactive template path
                os.path.join(self.templates_dir, f'report_types/{test_type}/interactive/index.html'),
                # Fallback to default template if interactive directory not found
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

        # Add all safe filters to this environment
        self._add_safe_filters(env)
        
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