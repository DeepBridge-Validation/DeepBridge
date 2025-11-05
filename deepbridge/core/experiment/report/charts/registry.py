"""
Chart registry for managing chart generators.

Provides centralized registry for registering and accessing chart generators
(Phase 2 Sprint 7-8, preparing for Phase 3).
"""

from typing import Dict, List, Type, Optional
import logging
from .base import ChartGenerator, ChartResult

logger = logging.getLogger("deepbridge.reports")


# ==================================================================================
# Chart Registry
# ==================================================================================

class ChartRegistry:
    """
    Registry for chart generators.

    Provides centralized management of chart generators with registration,
    lookup, and generation capabilities.

    Example:
        >>> # Register a generator
        >>> ChartRegistry.register('line_chart', LineChartGenerator())
        >>>
        >>> # Generate a chart
        >>> result = ChartRegistry.generate('line_chart', data={'x': [1,2], 'y': [3,4]})
        >>>
        >>> # List available charts
        >>> charts = ChartRegistry.list_charts()
    """

    # Class-level storage for chart generators
    _generators: Dict[str, ChartGenerator] = {}

    @classmethod
    def register(cls, name: str, generator: ChartGenerator) -> None:
        """
        Register a chart generator.

        Args:
            name: Unique name for the chart type
            generator: ChartGenerator instance

        Raises:
            TypeError: If generator is not a ChartGenerator
            ValueError: If name is already registered

        Example:
            >>> generator = LineChartGenerator()
            >>> ChartRegistry.register('line_chart', generator)
        """
        if not isinstance(generator, ChartGenerator):
            raise TypeError(f"Generator must be ChartGenerator, got {type(generator)}")

        if name in cls._generators:
            logger.warning(f"Chart '{name}' already registered, overwriting")

        cls._generators[name] = generator
        logger.info(f"Registered chart generator: {name} ({generator.__class__.__name__})")

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a chart generator.

        Args:
            name: Name of chart to unregister

        Returns:
            True if removed, False if not found

        Example:
            >>> ChartRegistry.unregister('line_chart')
        """
        if name in cls._generators:
            del cls._generators[name]
            logger.info(f"Unregistered chart generator: {name}")
            return True
        return False

    @classmethod
    def generate(cls, name: str, data: Dict, **kwargs) -> ChartResult:
        """
        Generate chart by name.

        Args:
            name: Name of registered chart
            data: Input data for chart
            **kwargs: Additional chart-specific options

        Returns:
            ChartResult with generated chart

        Raises:
            ValueError: If chart name not registered

        Example:
            >>> result = ChartRegistry.generate(
            ...     'line_chart',
            ...     data={'x': [1,2,3], 'y': [4,5,6]},
            ...     title='My Chart'
            ... )
        """
        if name not in cls._generators:
            error_msg = f"Chart '{name}' not registered. Available: {cls.list_charts()}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        generator = cls._generators[name]
        logger.debug(f"Generating chart: {name}")

        try:
            return generator.generate(data, **kwargs)
        except Exception as e:
            logger.error(f"Error generating chart '{name}': {e}")
            raise

    @classmethod
    def list_charts(cls) -> List[str]:
        """
        List all registered chart names.

        Returns:
            Sorted list of chart names

        Example:
            >>> charts = ChartRegistry.list_charts()
            >>> print(charts)
            >>> # ['bar_chart', 'line_chart', 'scatter_plot']
        """
        return sorted(cls._generators.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        Check if chart is registered.

        Args:
            name: Chart name to check

        Returns:
            True if registered

        Example:
            >>> if ChartRegistry.is_registered('line_chart'):
            ...     result = ChartRegistry.generate('line_chart', data)
        """
        return name in cls._generators

    @classmethod
    def get_generator(cls, name: str) -> Optional[ChartGenerator]:
        """
        Get generator instance by name.

        Args:
            name: Chart name

        Returns:
            ChartGenerator instance or None if not found

        Example:
            >>> generator = ChartRegistry.get_generator('line_chart')
        """
        return cls._generators.get(name)

    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered generators.

        Warning: This removes all registrations. Use with caution.

        Example:
            >>> ChartRegistry.clear()
        """
        cls._generators.clear()
        logger.info("Cleared all chart generators")

    @classmethod
    def count(cls) -> int:
        """
        Get number of registered charts.

        Returns:
            Number of registered charts

        Example:
            >>> count = ChartRegistry.count()
        """
        return len(cls._generators)

    @classmethod
    def get_info(cls) -> Dict[str, str]:
        """
        Get information about all registered charts.

        Returns:
            Dictionary mapping chart names to generator class names

        Example:
            >>> info = ChartRegistry.get_info()
            >>> # {'line_chart': 'LineChartGenerator', ...}
        """
        return {
            name: generator.__class__.__name__
            for name, generator in cls._generators.items()
        }


# ==================================================================================
# Decorator for Registration
# ==================================================================================

def register_chart(name: str):
    """
    Decorator to automatically register chart generator classes.

    Args:
        name: Name to register chart as

    Example:
        >>> @register_chart('my_chart')
        >>> class MyChartGenerator(ChartGenerator):
        ...     def generate(self, data, **kwargs):
        ...         return ChartResult(...)
    """
    def decorator(generator_class: Type[ChartGenerator]):
        # Instantiate and register
        instance = generator_class()
        ChartRegistry.register(name, instance)
        return generator_class
    return decorator


# ==================================================================================
# Main - Example Usage
# ==================================================================================

if __name__ == "__main__":
    """
    Demonstrate registry usage.
    """
    print("=" * 80)
    print("Chart Registry Example")
    print("=" * 80)

    # Example: Create mock generator
    from .base import ChartGenerator, ChartResult

    class ExampleLineChart(ChartGenerator):
        """Example line chart generator."""
        def generate(self, data, **kwargs):
            return ChartResult(
                content='{"type": "line", "data": []}',
                format='plotly',
                metadata={'title': kwargs.get('title', 'Line Chart')}
            )

    class ExampleBarChart(ChartGenerator):
        """Example bar chart generator."""
        def generate(self, data, **kwargs):
            return ChartResult(
                content='{"type": "bar", "data": []}',
                format='plotly',
                metadata={'title': kwargs.get('title', 'Bar Chart')}
            )

    # Register generators
    print("\nRegistering generators...")
    ChartRegistry.register('line_chart', ExampleLineChart())
    ChartRegistry.register('bar_chart', ExampleBarChart())

    # List charts
    print(f"\nAvailable charts: {ChartRegistry.list_charts()}")
    print(f"Total charts: {ChartRegistry.count()}")

    # Get info
    print(f"\nChart info: {ChartRegistry.get_info()}")

    # Generate a chart
    print("\nGenerating line chart...")
    result = ChartRegistry.generate(
        'line_chart',
        data={'x': [1, 2, 3], 'y': [4, 5, 6]},
        title='Example Line Chart'
    )
    print(f"Result: {result}")
    print(f"Success: {result.is_success}")

    # Try to generate non-existent chart
    print("\nTrying to generate non-existent chart...")
    try:
        ChartRegistry.generate('non_existent', {})
    except ValueError as e:
        print(f"Expected error: {e}")

    print("\n" + "=" * 80)
    print("Chart Registry Example Complete")
    print("=" * 80)
