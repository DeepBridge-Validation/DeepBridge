"""
JSON adapter for reports (Phase 3 Sprint 14).

Converts domain models to JSON format for API responses, storage, etc.
"""

import json
from typing import Dict, Any
from datetime import datetime
from .base import ReportAdapter
from ..domain.general import Report


class JSONAdapter(ReportAdapter):
    """
    Adapter to convert Report domain model to JSON.

    Features:
    - Serializes complete report structure
    - Handles datetime objects
    - Handles NaN/Inf values
    - Pretty printing option
    - JSON-safe output

    Example:
        >>> adapter = JSONAdapter(indent=2)
        >>> json_str = adapter.render(report)
        >>> data = json.loads(json_str)
    """

    def __init__(self, indent: int = None, ensure_ascii: bool = False):
        """
        Initialize JSON adapter.

        Args:
            indent: Number of spaces for indentation (None for compact)
            ensure_ascii: Whether to escape non-ASCII characters
        """
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def render(self, report: Report) -> str:
        """
        Render report to JSON string.

        Args:
            report: Report domain model

        Returns:
            JSON string representation

        Example:
            >>> json_str = adapter.render(report)
            >>> print(json_str)
            >>> # {
            >>> #   "metadata": {...},
            >>> #   "sections": [...]
            >>> # }
        """
        self._validate_report(report)

        # Convert to dict using Pydantic's JSON-safe serialization
        data = report.model_dump(mode='json')

        # Clean up None values (optional)
        data = self._clean_none_values(data)

        # Serialize to JSON
        return json.dumps(
            data,
            indent=self.indent,
            ensure_ascii=self.ensure_ascii,
            default=self._json_serializer
        )

    def render_dict(self, report: Report) -> Dict[str, Any]:
        """
        Render report to dictionary (without JSON serialization).

        Useful when you need the dict for further processing.

        Args:
            report: Report domain model

        Returns:
            Dictionary representation

        Example:
            >>> data = adapter.render_dict(report)
            >>> data['metadata']['model_name']  # Access data directly
        """
        self._validate_report(report)

        data = report.model_dump(mode='json')
        return self._clean_none_values(data)

    def _clean_none_values(self, data: Any) -> Any:
        """
        Recursively remove None values from nested structures.

        Args:
            data: Data to clean

        Returns:
            Cleaned data
        """
        if isinstance(data, dict):
            return {
                k: self._clean_none_values(v)
                for k, v in data.items()
                if v is not None
            }
        elif isinstance(data, list):
            return [self._clean_none_values(item) for item in data]
        else:
            return data

    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer for special types.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation

        Raises:
            TypeError: If object is not serializable
        """
        if isinstance(obj, datetime):
            return obj.isoformat()

        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
