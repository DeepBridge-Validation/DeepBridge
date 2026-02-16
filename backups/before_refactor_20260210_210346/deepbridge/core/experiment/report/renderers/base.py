"""
Base protocols and classes for report renderers.

This module defines the modern Protocol-based interfaces for renderers,
replacing the old abstract base classes with more flexible typing.
"""

from typing import Protocol, runtime_checkable
from pathlib import Path

from ..data.base import ReportData
from ..config import RenderConfig


@runtime_checkable
class ReportRenderer(Protocol):
    """Protocol for report renderers.

    All renderers (HTML, JSON, PDF, etc.) should implement this protocol.
    This provides a consistent interface while allowing flexibility in
    implementation.

    Using Protocol instead of ABC allows for structural subtyping and
    better compatibility with type checkers.

    Example:
        >>> class MyRenderer:
        ...     def render(self, data: ReportData, config: RenderConfig) -> str:
        ...         return "<html>...</html>"
        ...
        >>> renderer: ReportRenderer = MyRenderer()  # Type checks!
    """

    def render(self, data: ReportData, config: RenderConfig) -> str:
        """Render report data to string format.

        Args:
            data: Typed report data to render
            config: Rendering configuration

        Returns:
            Rendered report as string (HTML, JSON, etc.)

        Raises:
            ValueError: If data or config is invalid
            RuntimeError: If rendering fails

        Example:
            >>> from deepbridge.core.experiment.report import (
            ...     HTMLRenderer, RobustnessReportData, RenderConfig
            ... )
            >>> renderer = HTMLRenderer(template_engine)
            >>> html = renderer.render(robustness_data, RenderConfig())
        """
        ...


class BaseRenderer:
    """Base class for renderers with common functionality.

    This class provides shared utilities for all renderers, such as
    validation, error handling, and logging.

    Renderers can inherit from this class to get common functionality
    while still implementing the ReportRenderer protocol.
    """

    def validate_data(self, data: ReportData) -> None:
        """Validate report data before rendering.

        Args:
            data: Report data to validate

        Raises:
            ValueError: If data is invalid
        """
        if not isinstance(data, ReportData):
            raise ValueError(
                f"Expected ReportData instance, got {type(data).__name__}"
            )

        # Call the data's own validation method
        try:
            data.validate()
        except Exception as e:
            raise ValueError(f"Data validation failed: {e}")

    def validate_config(self, config: RenderConfig) -> None:
        """Validate render configuration.

        Args:
            config: Render configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        if not isinstance(config, RenderConfig):
            raise ValueError(
                f"Expected RenderConfig instance, got {type(config).__name__}"
            )

    def _ensure_output_dir(self, file_path: Path) -> None:
        """Ensure output directory exists.

        Args:
            file_path: Path where report will be saved
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_output(self, content: str, file_path: Path) -> Path:
        """Write rendered content to file.

        Args:
            content: Rendered report content
            file_path: Path where to save the file

        Returns:
            Path to the created file

        Example:
            >>> renderer._write_output(html_content, Path("report.html"))
            Path('report.html')
        """
        self._ensure_output_dir(file_path)
        file_path.write_text(content, encoding='utf-8')
        return file_path
