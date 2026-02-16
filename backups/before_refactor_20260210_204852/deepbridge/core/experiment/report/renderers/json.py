"""
JSON renderer for report generation.

This module provides JSON rendering capabilities for API consumption
and data interchange.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from ..data.base import ReportData
from ..config import RenderConfig, OutputFormat
from .base import BaseRenderer


logger = logging.getLogger(__name__)


class JSONRenderer(BaseRenderer):
    """Renderer for JSON reports.

    This renderer converts typed ReportData into JSON format suitable
    for API consumption, data storage, and interchange.

    Example:
        >>> from deepbridge.core.experiment.report import (
        ...     JSONRenderer, RenderConfig, OutputFormat
        ... )
        >>> renderer = JSONRenderer()
        >>> json_str = renderer.render(
        ...     report_data,
        ...     RenderConfig(format=OutputFormat.JSON)
        ... )
        >>> data = json.loads(json_str)
    """

    def __init__(self, indent: Optional[int] = 2, sort_keys: bool = False):
        """Initialize JSON renderer.

        Args:
            indent: Indentation level for pretty printing (None for compact)
            sort_keys: Whether to sort dictionary keys
        """
        self.indent = indent
        self.sort_keys = sort_keys
        logger.info(f"JSONRenderer initialized (indent={indent}, sort_keys={sort_keys})")

    def render(self, data: ReportData, config: RenderConfig) -> str:
        """Render report data as JSON.

        Args:
            data: Typed report data to render
            config: Rendering configuration

        Returns:
            JSON string

        Raises:
            ValueError: If data or config is invalid
            RuntimeError: If JSON serialization fails

        Example:
            >>> json_str = renderer.render(data, config)
            >>> parsed = json.loads(json_str)
            >>> assert 'report_type' in parsed
        """
        # Validate inputs
        self.validate_data(data)
        self.validate_config(config)

        # Only JSON format is supported
        if config.format != OutputFormat.JSON:
            raise ValueError(
                f"JSONRenderer only supports JSON format, got {config.format}"
            )

        logger.info(f"Rendering {data.report_type} report as JSON")

        try:
            # Get JSON-serializable dictionary
            data_dict = data.to_json_dict()

            # Add configuration info if requested
            if config.include_raw_data:
                output = {
                    'report': data_dict,
                    'config': config.to_dict(),
                }
            else:
                output = data_dict

            # Serialize to JSON
            json_str = json.dumps(
                output,
                indent=self.indent,
                sort_keys=self.sort_keys,
                default=str  # Handle any remaining non-serializable types
            )

            logger.info(f"Successfully rendered JSON ({len(json_str)} characters)")
            return json_str

        except (TypeError, ValueError) as e:
            logger.error(f"Failed to serialize to JSON: {e}", exc_info=True)
            raise RuntimeError(f"JSON serialization failed: {e}") from e

    def render_to_file(
        self,
        data: ReportData,
        config: RenderConfig,
        output_path: Path
    ) -> Path:
        """Render report and save to JSON file.

        Args:
            data: Report data to render
            config: Rendering configuration
            output_path: Path where to save JSON file

        Returns:
            Path to created file

        Example:
            >>> path = renderer.render_to_file(
            ...     data, config, Path("report.json")
            ... )
            >>> assert path.exists()
        """
        json_str = self.render(data, config)
        output_path = Path(output_path)
        return self._write_output(json_str, output_path)

    def render_compact(self, data: ReportData, config: RenderConfig) -> str:
        """Render report as compact JSON (no indentation).

        Args:
            data: Report data
            config: Rendering configuration

        Returns:
            Compact JSON string

        Example:
            >>> json_str = renderer.render_compact(data, config)
            >>> # No newlines or extra whitespace
        """
        original_indent = self.indent
        self.indent = None
        try:
            return self.render(data, config)
        finally:
            self.indent = original_indent

    def render_pretty(self, data: ReportData, config: RenderConfig) -> str:
        """Render report as pretty-printed JSON.

        Args:
            data: Report data
            config: Rendering configuration

        Returns:
            Pretty-printed JSON string

        Example:
            >>> json_str = renderer.render_pretty(data, config)
            >>> # Nicely formatted with indentation
        """
        original_indent = self.indent
        self.indent = 2
        try:
            return self.render(data, config)
        finally:
            self.indent = original_indent


class JSONLinesRenderer(JSONRenderer):
    """Renderer for JSON Lines format (newline-delimited JSON).

    Useful for streaming reports or appending to log files.

    Example:
        >>> renderer = JSONLinesRenderer()
        >>> for report_data in reports:
        ...     json_line = renderer.render_line(report_data, config)
        ...     file.write(json_line)
    """

    def __init__(self):
        """Initialize JSON Lines renderer with compact formatting."""
        super().__init__(indent=None, sort_keys=False)

    def render_line(self, data: ReportData, config: RenderConfig) -> str:
        """Render single line of JSON (compact, with newline).

        Args:
            data: Report data
            config: Rendering configuration

        Returns:
            Single line of JSON with trailing newline

        Example:
            >>> line = renderer.render_line(data, config)
            >>> assert line.endswith('\\n')
        """
        json_str = self.render(data, config)
        return json_str + '\n'

    def render_to_file(
        self,
        data: ReportData,
        config: RenderConfig,
        output_path: Path,
        append: bool = False
    ) -> Path:
        """Render to JSON Lines file.

        Args:
            data: Report data
            config: Rendering configuration
            output_path: Path to output file
            append: Whether to append (True) or overwrite (False)

        Returns:
            Path to created/updated file

        Example:
            >>> # Create new file
            >>> path = renderer.render_to_file(data1, config, Path("log.jsonl"))
            >>> # Append to existing file
            >>> path = renderer.render_to_file(data2, config, Path("log.jsonl"), append=True)
        """
        line = self.render_line(data, config)
        output_path = Path(output_path)

        self._ensure_output_dir(output_path)

        mode = 'a' if append else 'w'
        with open(output_path, mode, encoding='utf-8') as f:
            f.write(line)

        return output_path
