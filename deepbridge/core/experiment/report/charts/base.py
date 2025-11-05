"""
Base classes for chart generation.

Provides abstract base for chart generators and result container
(Phase 2 Sprint 7-8, preparing for Phase 3).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger("deepbridge.reports")


# ==================================================================================
# Chart Result Container
# ==================================================================================

@dataclass
class ChartResult:
    """
    Result of chart generation.

    Attributes:
        content: Chart content (base64 string, JSON, or HTML)
        format: Output format ('png', 'plotly', 'svg', 'html')
        metadata: Optional metadata about the chart
        error: Optional error message if generation failed

    Example:
        >>> result = ChartResult(
        ...     content='<div id="chart">...</div>',
        ...     format='plotly',
        ...     metadata={'title': 'Accuracy Over Time'}
        ... )
    """
    content: str
    format: str
    metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_success(self) -> bool:
        """Check if chart generation succeeded."""
        return self.error is None and bool(self.content)

    @property
    def is_base64(self) -> bool:
        """Check if content is base64 encoded."""
        return self.format in ['png', 'jpg', 'jpeg', 'svg']

    @property
    def is_interactive(self) -> bool:
        """Check if chart is interactive."""
        return self.format in ['plotly', 'html']

    def __repr__(self) -> str:
        status = "success" if self.is_success else f"error: {self.error}"
        return f"ChartResult(format={self.format}, {status})"


# ==================================================================================
# Chart Generator Base Class
# ==================================================================================

class ChartGenerator(ABC):
    """
    Base class for all chart generators.

    Subclasses must implement generate() to create specific chart types.

    Example:
        >>> class LineChartGenerator(ChartGenerator):
        ...     def generate(self, data, **kwargs):
        ...         # Create line chart
        ...         return ChartResult(content=chart_json, format='plotly')
    """

    @abstractmethod
    def generate(self, data: Dict[str, Any], **kwargs) -> ChartResult:
        """
        Generate chart from data.

        Args:
            data: Input data for chart
            **kwargs: Additional chart-specific options

        Returns:
            ChartResult with generated chart

        Raises:
            ValueError: If data is invalid
            Exception: If generation fails

        Example:
            >>> generator = LineChartGenerator()
            >>> result = generator.generate(
            ...     data={'x': [1,2,3], 'y': [4,5,6]},
            ...     title='My Chart'
            ... )
        """
        pass

    def _create_error_result(self, error_msg: str, format: str = 'html') -> ChartResult:
        """
        Create an error result.

        Args:
            error_msg: Error message
            format: Output format

        Returns:
            ChartResult with error
        """
        logger.error(f"{self.__class__.__name__}: {error_msg}")
        return ChartResult(
            content="",
            format=format,
            error=error_msg
        )

    def _validate_data(self, data: Dict[str, Any], required_keys: list) -> bool:
        """
        Validate that data contains required keys.

        Args:
            data: Data to validate
            required_keys: List of required key names

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        missing = [key for key in required_keys if key not in data]
        if missing:
            raise ValueError(f"Missing required keys: {missing}")
        return True

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


# ==================================================================================
# Example Chart Generators
# ==================================================================================

class PlotlyChartGenerator(ChartGenerator):
    """
    Base for Plotly-based chart generators.

    Subclasses should override _create_plotly_figure().
    """

    def generate(self, data: Dict[str, Any], **kwargs) -> ChartResult:
        """Generate Plotly chart."""
        try:
            figure = self._create_plotly_figure(data, **kwargs)

            # Convert to JSON
            import json
            chart_json = json.dumps(figure, ensure_ascii=False)

            return ChartResult(
                content=chart_json,
                format='plotly',
                metadata={
                    'title': kwargs.get('title', ''),
                    'type': self.__class__.__name__
                }
            )

        except Exception as e:
            return self._create_error_result(str(e), 'plotly')

    @abstractmethod
    def _create_plotly_figure(self, data: Dict[str, Any], **kwargs) -> Dict:
        """
        Create Plotly figure dictionary.

        Args:
            data: Input data
            **kwargs: Chart options

        Returns:
            Plotly figure dict with 'data' and 'layout'
        """
        pass


class StaticImageGenerator(ChartGenerator):
    """
    Base for static image chart generators (PNG, SVG).

    Subclasses should override _create_image().
    """

    def generate(self, data: Dict[str, Any], **kwargs) -> ChartResult:
        """Generate static image chart."""
        try:
            image_data, format_type = self._create_image(data, **kwargs)

            return ChartResult(
                content=image_data,
                format=format_type,
                metadata={
                    'title': kwargs.get('title', ''),
                    'type': self.__class__.__name__
                }
            )

        except Exception as e:
            return self._create_error_result(str(e), 'png')

    @abstractmethod
    def _create_image(self, data: Dict[str, Any], **kwargs) -> tuple[str, str]:
        """
        Create image as base64 string.

        Args:
            data: Input data
            **kwargs: Chart options

        Returns:
            Tuple of (base64_string, format_type)
        """
        pass


# ==================================================================================
# Main - Example Usage
# ==================================================================================

if __name__ == "__main__":
    """
    Demonstrate base chart classes.
    """
    print("=" * 80)
    print("Chart Base Classes Example")
    print("=" * 80)

    # Example: Create a chart result
    result = ChartResult(
        content='{"data": [], "layout": {}}',
        format='plotly',
        metadata={'title': 'Example Chart'}
    )

    print(f"\nChart Result: {result}")
    print(f"Success: {result.is_success}")
    print(f"Interactive: {result.is_interactive}")

    # Example: Error result
    error_result = ChartResult(
        content="",
        format='plotly',
        error="Failed to generate chart"
    )

    print(f"\nError Result: {error_result}")
    print(f"Success: {error_result.is_success}")

    print("\n" + "=" * 80)
    print("Chart Base Classes Example Complete")
    print("=" * 80)
