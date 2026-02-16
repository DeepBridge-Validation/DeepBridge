"""
Template engine for report generation.

This module provides a Jinja2-based template engine for rendering HTML reports.
It replaces the embedded JavaScript/HTML strings scattered throughout the old
renderer files with external, maintainable templates.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable
import logging

from jinja2 import (
    Environment,
    FileSystemLoader,
    Template,
    TemplateNotFound,
    select_autoescape
)

from .filters import register_custom_filters


logger = logging.getLogger(__name__)


class TemplateEngine:
    """Jinja2-based template engine for report rendering.

    This class manages HTML templates and provides rendering capabilities
    with custom filters and functions.

    Attributes:
        env: Jinja2 Environment instance
        template_dir: Directory containing templates

    Example:
        >>> engine = TemplateEngine(Path("templates/html"))
        >>> template = engine.get_template("robustness/full.html")
        >>> html = template.render(data=report_data)
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        autoescape: bool = True
    ):
        """Initialize template engine.

        Args:
            template_dir: Directory containing templates. If None, uses default.
            autoescape: Whether to enable autoescape for HTML/XML templates

        Raises:
            ValueError: If template_dir doesn't exist
        """
        if template_dir is None:
            # Default to templates/html in this package
            template_dir = Path(__file__).parent / "html"

        self.template_dir = Path(template_dir)

        if not self.template_dir.exists():
            raise ValueError(
                f"Template directory does not exist: {self.template_dir}"
            )

        # Create Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=select_autoescape(['html', 'xml']) if autoescape else False,
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Register custom filters
        register_custom_filters(self.env)

        # Add custom globals
        self._register_globals()

        logger.info(f"TemplateEngine initialized with directory: {self.template_dir}")

    def _register_globals(self) -> None:
        """Register global functions available in all templates."""
        self.env.globals.update({
            'now': self._get_current_datetime,
            'version': self._get_version,
        })

    def get_template(self, name: str) -> Template:
        """Get template by name.

        Args:
            name: Template name (relative to template_dir)
                  e.g., "robustness/full.html"

        Returns:
            Jinja2 Template instance

        Raises:
            TemplateNotFound: If template doesn't exist

        Example:
            >>> template = engine.get_template("robustness/full.html")
            >>> html = template.render(data={...})
        """
        try:
            template = self.env.get_template(name)
            logger.debug(f"Template loaded: {name}")
            return template
        except TemplateNotFound as e:
            logger.error(f"Template not found: {name}")
            raise TemplateNotFound(
                f"Template '{name}' not found in {self.template_dir}"
            ) from e

    def render(
        self,
        template_name: str,
        context: Dict[str, Any]
    ) -> str:
        """Render template with context.

        Convenience method that combines get_template and render.

        Args:
            template_name: Name of template to render
            context: Context dictionary for template

        Returns:
            Rendered HTML string

        Example:
            >>> html = engine.render("robustness/full.html", {"data": data})
        """
        template = self.get_template(template_name)
        return template.render(**context)

    def render_string(self, template_string: str, context: Dict[str, Any]) -> str:
        """Render template from string.

        Useful for rendering small template snippets.

        Args:
            template_string: Template content as string
            context: Context dictionary

        Returns:
            Rendered string

        Example:
            >>> html = engine.render_string("<h1>{{ title }}</h1>", {"title": "Report"})
        """
        template = self.env.from_string(template_string)
        return template.render(**context)

    def register_filter(self, name: str, func: Callable) -> None:
        """Register custom Jinja2 filter.

        Args:
            name: Filter name (used in templates)
            func: Filter function

        Example:
            >>> def uppercase(text):
            ...     return text.upper()
            >>> engine.register_filter('upper', uppercase)
            >>> # In template: {{ name | upper }}
        """
        self.env.filters[name] = func
        logger.debug(f"Custom filter registered: {name}")

    def register_global(self, name: str, value: Any) -> None:
        """Register global variable/function.

        Args:
            name: Global name (available in all templates)
            value: Value or function

        Example:
            >>> engine.register_global('company_name', 'DeepBridge')
            >>> # In template: {{ company_name }}
        """
        self.env.globals[name] = value
        logger.debug(f"Global registered: {name}")

    def list_templates(self, pattern: Optional[str] = None) -> list:
        """List available templates.

        Args:
            pattern: Optional glob pattern to filter templates

        Returns:
            List of template names

        Example:
            >>> engine.list_templates("robustness/*.html")
            ['robustness/full.html', 'robustness/simple.html']
        """
        templates = self.env.list_templates()

        if pattern:
            import fnmatch
            templates = [t for t in templates if fnmatch.fnmatch(t, pattern)]

        return sorted(templates)

    @staticmethod
    def _get_current_datetime() -> str:
        """Get current datetime as ISO string.

        Returns:
            Current datetime in ISO format
        """
        from datetime import datetime
        return datetime.now().isoformat()

    @staticmethod
    def _get_version() -> str:
        """Get DeepBridge version.

        Returns:
            Version string
        """
        try:
            from deepbridge import __version__
            return __version__
        except ImportError:
            return "unknown"


# Default template directory
DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "html"


def get_default_engine() -> TemplateEngine:
    """Get default template engine instance.

    Returns:
        TemplateEngine with default configuration

    Example:
        >>> engine = get_default_engine()
        >>> template = engine.get_template("robustness/full.html")
    """
    return TemplateEngine(DEFAULT_TEMPLATE_DIR)
