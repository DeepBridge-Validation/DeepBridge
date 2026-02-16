"""
HTML renderer for report generation.

This module provides HTML rendering capabilities using the template engine,
replacing the old approach of embedding HTML/JS strings in Python code.
"""

import logging
from pathlib import Path
from typing import Optional

from ..data.base import ReportData
from ..config import RenderConfig, ReportStyle
from ..templates import TemplateEngine
from .base import BaseRenderer


logger = logging.getLogger(__name__)


class HTMLRenderer(BaseRenderer):
    """Renderer for HTML reports.

    This single renderer replaces all the old *_renderer.py variants:
    - robustness_renderer.py (FULL style)
    - robustness_renderer_simple.py (SIMPLE style)
    - static_robustness_renderer.py (STATIC style)
    - And similar for resilience, uncertainty, etc.

    The rendering style is now controlled by RenderConfig, not separate classes.

    Example:
        >>> from deepbridge.core.experiment.report import (
        ...     HTMLRenderer, TemplateEngine, RenderConfig, ReportStyle
        ... )
        >>> engine = TemplateEngine()
        >>> renderer = HTMLRenderer(engine)
        >>> html = renderer.render(report_data, RenderConfig(style=ReportStyle.FULL))
    """

    def __init__(self, template_engine: TemplateEngine):
        """Initialize HTML renderer.

        Args:
            template_engine: Template engine for rendering HTML
        """
        self.template_engine = template_engine
        logger.info("HTMLRenderer initialized")

    def render(self, data: ReportData, config: RenderConfig) -> str:
        """Render report data as HTML.

        Args:
            data: Typed report data to render
            config: Rendering configuration

        Returns:
            Rendered HTML string

        Raises:
            ValueError: If data or config is invalid
            RuntimeError: If rendering fails

        Example:
            >>> html = renderer.render(robustness_data, RenderConfig())
            >>> assert '<html>' in html
        """
        # Validate inputs
        self.validate_data(data)
        self.validate_config(config)

        # Only HTML format is supported by this renderer
        from ..config import OutputFormat
        if config.format != OutputFormat.HTML:
            raise ValueError(
                f"HTMLRenderer only supports HTML format, got {config.format}"
            )

        logger.info(f"Rendering {data.report_type} report in {config.style.value} style")

        try:
            # Get appropriate template based on report type and style
            template_name = self._get_template_name(data.report_type, config.style)

            # Prepare context for template
            context = self._prepare_context(data, config)

            # Render template
            html = self.template_engine.render(template_name, context)

            logger.info(f"Successfully rendered HTML ({len(html)} characters)")
            return html

        except Exception as e:
            logger.error(f"Failed to render HTML: {e}", exc_info=True)
            raise RuntimeError(f"HTML rendering failed: {e}") from e

    def render_to_file(
        self,
        data: ReportData,
        config: RenderConfig,
        output_path: Path
    ) -> Path:
        """Render report and save to file.

        Convenience method that combines render() and file writing.

        Args:
            data: Report data to render
            config: Rendering configuration
            output_path: Path where to save HTML file

        Returns:
            Path to created file

        Example:
            >>> path = renderer.render_to_file(
            ...     data, config, Path("report.html")
            ... )
            >>> assert path.exists()
        """
        html = self.render(data, config)
        output_path = Path(output_path)
        return self._write_output(html, output_path)

    def _get_template_name(self, report_type: str, style: ReportStyle) -> str:
        """Get template name based on report type and style.

        Args:
            report_type: Type of report (e.g., 'robustness', 'resilience')
            style: Rendering style

        Returns:
            Template name (relative to template directory)

        Example:
            >>> renderer._get_template_name('robustness', ReportStyle.FULL)
            'robustness/full.html'
        """
        # Map style to template filename
        style_map = {
            ReportStyle.FULL: 'full.html',
            ReportStyle.SIMPLE: 'simple.html',
            ReportStyle.STATIC: 'static.html',
            ReportStyle.INTERACTIVE: 'interactive.html',
        }

        style_filename = style_map.get(style, 'full.html')
        template_name = f"{report_type}/{style_filename}"

        logger.debug(f"Using template: {template_name}")
        return template_name

    def _prepare_context(self, data: ReportData, config: RenderConfig) -> dict:
        """Prepare template context from data and config.

        Args:
            data: Report data
            config: Rendering configuration

        Returns:
            Context dictionary for template rendering
        """
        # Convert data to dictionary
        data_dict = data.to_dict()

        # Build context
        context = {
            'data': data_dict,
            'config': config.to_dict(),
            'report_type': data.report_type,
            'style': config.style.value,
            'include_charts': config.include_charts,
            'interactive_charts': config.interactive_charts,
            'theme': config.theme,
            'generated_at': data.generated_at,
        }

        # Add custom metadata
        if config.metadata:
            context['metadata'] = config.metadata

        logger.debug(f"Prepared context with {len(context)} keys")
        return context


class HTMLRendererWithAssets(HTMLRenderer):
    """HTML renderer with asset management capabilities.

    This extended version handles CSS/JS asset embedding or linking,
    preserving compatibility with the existing AssetManager.

    Example:
        >>> from deepbridge.core.experiment.report import AssetManager
        >>> asset_manager = AssetManager()
        >>> renderer = HTMLRendererWithAssets(engine, asset_manager)
    """

    def __init__(
        self,
        template_engine: TemplateEngine,
        asset_manager=None
    ):
        """Initialize renderer with asset manager.

        Args:
            template_engine: Template engine
            asset_manager: Asset manager instance (optional)
        """
        super().__init__(template_engine)

        # Import AssetManager if provided as None
        if asset_manager is None:
            try:
                from ..asset_manager import AssetManager
                # AssetManager needs the parent of the html directory
                template_dir = template_engine.template_dir
                if template_dir.name == 'html':
                    asset_manager_dir = template_dir.parent
                else:
                    asset_manager_dir = template_dir
                asset_manager = AssetManager(str(asset_manager_dir))
            except (ImportError, FileNotFoundError) as e:
                logger.warning(f"AssetManager not available: {e}, using basic renderer")
                asset_manager = None

        self.asset_manager = asset_manager
        logger.info("HTMLRendererWithAssets initialized")

    def _prepare_context(self, data: ReportData, config: RenderConfig) -> dict:
        """Prepare context with asset information.

        Args:
            data: Report data
            config: Rendering configuration

        Returns:
            Context with asset URLs/content
        """
        context = super()._prepare_context(data, config)

        # Add asset information if asset manager is available
        if self.asset_manager:
            try:
                if config.embed_assets:
                    # Embed assets directly in HTML
                    context['css_content'] = self.asset_manager.get_css_content()
                    context['js_content'] = self.asset_manager.get_js_content()
                    context['embed_assets'] = True
                else:
                    # Use external asset URLs
                    context['css_url'] = self.asset_manager.get_css_url()
                    context['js_url'] = self.asset_manager.get_js_url()
                    context['embed_assets'] = False

                logger.debug("Added asset information to context")
            except Exception as e:
                logger.warning(f"Failed to add asset information: {e}")

        return context
