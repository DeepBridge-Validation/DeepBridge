"""
Base adapter interface (Phase 3 Sprint 14).

Defines the contract for converting domain models to specific output formats.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict
from ..domain.general import Report


class ReportAdapter(ABC):
    """
    Base class for all report adapters.

    Adapters convert domain models (Report) into specific output formats
    (HTML, JSON, PDF, Markdown, etc.).

    The adapter pattern separates:
    - WHAT to display (domain model)
    - HOW to display (adapter)

    Example:
        class MyAdapter(ReportAdapter):
            def render(self, report: Report) -> str:
                # Convert report to desired format
                return formatted_output
    """

    @abstractmethod
    def render(self, report: Report) -> Any:
        """
        Render a report to the target format.

        Args:
            report: Report domain model to render

        Returns:
            Rendered output (type depends on adapter)

        Example:
            >>> adapter = HTMLAdapter()
            >>> html = adapter.render(report)  # Returns HTML string
        """
        pass

    def _validate_report(self, report: Report) -> None:
        """
        Validate that report has required data.

        Args:
            report: Report to validate

        Raises:
            ValueError: If report is missing required data
        """
        if not report.metadata:
            raise ValueError("Report must have metadata")

        if not report.metadata.model_name:
            raise ValueError("Report metadata must have model_name")

        if not report.metadata.test_type:
            raise ValueError("Report metadata must have test_type")
