"""
Configuration system for report generation.

This module defines the configuration classes and enums used to control
report rendering style, format, and behavior.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional


class ReportStyle(Enum):
    """Report rendering styles.

    Attributes:
        FULL: Complete interactive report with all charts and features
              (replaces robustness_renderer.py, resilience_renderer.py, etc.)
        SIMPLE: Simplified report with basic information
                (replaces robustness_renderer_simple.py, etc.)
        STATIC: Static report without interactive features
                (replaces static_robustness_renderer.py, etc.)
        INTERACTIVE: Fully interactive report with advanced visualizations
    """
    FULL = "full"
    SIMPLE = "simple"
    STATIC = "static"
    INTERACTIVE = "interactive"


class OutputFormat(Enum):
    """Output format for reports.

    Attributes:
        HTML: HTML format with embedded assets
        JSON: JSON format for API consumption
        PDF: PDF format (future implementation)
        MARKDOWN: Markdown format for documentation
    """
    HTML = "html"
    JSON = "json"
    PDF = "pdf"
    MARKDOWN = "markdown"


@dataclass
class RenderConfig:
    """Configuration for report rendering.

    This class controls how reports are generated, including style,
    format, and various feature flags.

    Attributes:
        style: Report style (FULL, SIMPLE, STATIC, or INTERACTIVE)
        format: Output format (HTML, JSON, PDF, or MARKDOWN)
        include_charts: Whether to include charts in the report
        interactive_charts: Whether charts should be interactive
        embed_assets: Whether to embed CSS/JS or use external files
        include_raw_data: Whether to include raw data in JSON format
        theme: Color theme for the report ('light' or 'dark')
        custom_css: Path to custom CSS file (optional)
        custom_js: Path to custom JavaScript file (optional)
        metadata: Additional metadata to include in the report

    Example:
        >>> config = RenderConfig(
        ...     style=ReportStyle.FULL,
        ...     format=OutputFormat.HTML,
        ...     include_charts=True,
        ...     interactive_charts=True
        ... )
    """
    style: ReportStyle = ReportStyle.FULL
    format: OutputFormat = OutputFormat.HTML
    include_charts: bool = True
    interactive_charts: bool = False
    embed_assets: bool = True
    include_raw_data: bool = False
    theme: str = "light"
    custom_css: Optional[str] = None
    custom_js: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate theme
        if self.theme not in ('light', 'dark'):
            raise ValueError(f"Invalid theme: {self.theme}. Must be 'light' or 'dark'")

        # STATIC style cannot have interactive charts
        if self.style == ReportStyle.STATIC and self.interactive_charts:
            raise ValueError("STATIC style cannot have interactive_charts=True")

        # SIMPLE style typically doesn't include charts
        if self.style == ReportStyle.SIMPLE and self.include_charts:
            # Just a warning, not an error
            import warnings
            warnings.warn(
                "SIMPLE style typically doesn't include charts. "
                "Consider setting include_charts=False.",
                UserWarning
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration
        """
        return {
            'style': self.style.value,
            'format': self.format.value,
            'include_charts': self.include_charts,
            'interactive_charts': self.interactive_charts,
            'embed_assets': self.embed_assets,
            'include_raw_data': self.include_raw_data,
            'theme': self.theme,
            'custom_css': self.custom_css,
            'custom_js': self.custom_js,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RenderConfig':
        """Create configuration from dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            RenderConfig instance
        """
        # Convert string enums back to enum values
        if 'style' in data and isinstance(data['style'], str):
            data['style'] = ReportStyle(data['style'])
        if 'format' in data and isinstance(data['format'], str):
            data['format'] = OutputFormat(data['format'])

        return cls(**data)


# Predefined configurations for common use cases
PRESET_CONFIGS = {
    'full_interactive': RenderConfig(
        style=ReportStyle.FULL,
        format=OutputFormat.HTML,
        include_charts=True,
        interactive_charts=True,
        embed_assets=True
    ),
    'simple_static': RenderConfig(
        style=ReportStyle.SIMPLE,
        format=OutputFormat.HTML,
        include_charts=False,
        interactive_charts=False,
        embed_assets=True
    ),
    'static_embedded': RenderConfig(
        style=ReportStyle.STATIC,
        format=OutputFormat.HTML,
        include_charts=True,
        interactive_charts=False,
        embed_assets=True
    ),
    'json_api': RenderConfig(
        style=ReportStyle.FULL,
        format=OutputFormat.JSON,
        include_charts=False,
        include_raw_data=True
    ),
}


def get_preset_config(preset_name: str) -> RenderConfig:
    """Get a predefined configuration by name.

    Args:
        preset_name: Name of the preset configuration

    Returns:
        RenderConfig instance

    Raises:
        ValueError: If preset name is not found

    Example:
        >>> config = get_preset_config('full_interactive')
    """
    if preset_name not in PRESET_CONFIGS:
        available = ', '.join(PRESET_CONFIGS.keys())
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available presets: {available}"
        )

    return PRESET_CONFIGS[preset_name]
