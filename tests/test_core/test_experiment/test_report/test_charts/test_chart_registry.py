"""
Tests for Chart Registry.

Tests the chart generation infrastructure created in Phase 2 Sprint 7-8.
"""

import pytest

from deepbridge.core.experiment.report.charts.base import (
    ChartGenerator,
    ChartResult,
    PlotlyChartGenerator,
)
from deepbridge.core.experiment.report.charts.registry import (
    ChartRegistry,
    register_chart,
)

# ==================================================================================
# Mock Chart Generators
# ==================================================================================


class MockChartGenerator(ChartGenerator):
    """Mock chart generator for testing."""

    def generate(self, data, **kwargs):
        return ChartResult(
            content='{"mock": "data"}',
            format='plotly',
            metadata={'type': 'mock'},
        )


class FailingChartGenerator(ChartGenerator):
    """Chart generator that always fails."""

    def generate(self, data, **kwargs):
        raise ValueError('Intentional failure')


class MockPlotlyGenerator(PlotlyChartGenerator):
    """Mock Plotly generator."""

    def _create_plotly_figure(self, data, **kwargs):
        return {
            'data': [{'x': data.get('x', []), 'y': data.get('y', [])}],
            'layout': {'title': kwargs.get('title', 'Test')},
        }


# ==================================================================================
# Fixtures
# ==================================================================================


@pytest.fixture(autouse=True)
def clear_registry():
    """Clear registry before each test."""
    ChartRegistry.clear()
    yield
    ChartRegistry.clear()


# ==================================================================================
# Tests for ChartResult
# ==================================================================================


class TestChartResult:
    """Tests for ChartResult dataclass."""

    def test_successful_result(self):
        """Test successful chart result."""
        result = ChartResult(
            content='chart data', format='plotly', metadata={'title': 'Test'}
        )

        assert result.is_success
        assert result.content == 'chart data'
        assert result.format == 'plotly'
        assert result.error is None

    def test_error_result(self):
        """Test error result."""
        result = ChartResult(
            content='', format='plotly', error='Generation failed'
        )

        assert not result.is_success
        assert result.error == 'Generation failed'

    def test_is_base64(self):
        """Test base64 format detection."""
        png_result = ChartResult(content='data', format='png')
        svg_result = ChartResult(content='data', format='svg')
        plotly_result = ChartResult(content='data', format='plotly')

        assert png_result.is_base64
        assert svg_result.is_base64
        assert not plotly_result.is_base64

    def test_is_interactive(self):
        """Test interactive format detection."""
        plotly_result = ChartResult(content='data', format='plotly')
        html_result = ChartResult(content='data', format='html')
        png_result = ChartResult(content='data', format='png')

        assert plotly_result.is_interactive
        assert html_result.is_interactive
        assert not png_result.is_interactive

    def test_repr(self):
        """Test string representation."""
        result = ChartResult(content='data', format='plotly')
        repr_str = repr(result)

        assert 'ChartResult' in repr_str
        assert 'plotly' in repr_str
        assert 'success' in repr_str


# ==================================================================================
# Tests for ChartRegistry
# ==================================================================================


class TestChartRegistry:
    """Tests for ChartRegistry."""

    def test_register_chart(self):
        """Test registering a chart generator."""
        generator = MockChartGenerator()
        ChartRegistry.register('test_chart', generator)

        assert ChartRegistry.is_registered('test_chart')
        assert ChartRegistry.count() == 1

    def test_register_invalid_type(self):
        """Test registering non-generator raises error."""
        with pytest.raises(TypeError):
            ChartRegistry.register('invalid', 'not a generator')

    def test_register_duplicate_warns(self):
        """Test registering duplicate name."""
        generator1 = MockChartGenerator()
        generator2 = MockChartGenerator()

        ChartRegistry.register('test', generator1)
        ChartRegistry.register('test', generator2)  # Should warn but succeed

        assert ChartRegistry.count() == 1

    def test_unregister_chart(self):
        """Test unregistering a chart."""
        generator = MockChartGenerator()
        ChartRegistry.register('test', generator)

        result = ChartRegistry.unregister('test')

        assert result is True
        assert not ChartRegistry.is_registered('test')
        assert ChartRegistry.count() == 0

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent chart."""
        result = ChartRegistry.unregister('nonexistent')
        assert result is False

    def test_generate_chart(self):
        """Test generating a chart."""
        generator = MockChartGenerator()
        ChartRegistry.register('mock', generator)

        result = ChartRegistry.generate('mock', data={})

        assert isinstance(result, ChartResult)
        assert result.is_success

    def test_generate_nonexistent_raises(self):
        """Test generating non-existent chart raises error."""
        with pytest.raises(ValueError) as exc_info:
            ChartRegistry.generate('nonexistent', data={})

        assert 'not registered' in str(exc_info.value)

    def test_list_charts(self):
        """Test listing all registered charts."""
        ChartRegistry.register('chart_a', MockChartGenerator())
        ChartRegistry.register('chart_b', MockChartGenerator())
        ChartRegistry.register('chart_c', MockChartGenerator())

        charts = ChartRegistry.list_charts()

        assert charts == ['chart_a', 'chart_b', 'chart_c']  # Sorted

    def test_get_generator(self):
        """Test getting generator instance."""
        generator = MockChartGenerator()
        ChartRegistry.register('test', generator)

        retrieved = ChartRegistry.get_generator('test')

        assert retrieved is generator

    def test_get_nonexistent_generator(self):
        """Test getting non-existent generator returns None."""
        result = ChartRegistry.get_generator('nonexistent')
        assert result is None

    def test_clear_registry(self):
        """Test clearing all registrations."""
        ChartRegistry.register('chart1', MockChartGenerator())
        ChartRegistry.register('chart2', MockChartGenerator())

        assert ChartRegistry.count() == 2

        ChartRegistry.clear()

        assert ChartRegistry.count() == 0

    def test_count(self):
        """Test counting registered charts."""
        assert ChartRegistry.count() == 0

        ChartRegistry.register('chart1', MockChartGenerator())
        assert ChartRegistry.count() == 1

        ChartRegistry.register('chart2', MockChartGenerator())
        assert ChartRegistry.count() == 2

    def test_get_info(self):
        """Test getting registry info."""
        ChartRegistry.register('mock', MockChartGenerator())

        info = ChartRegistry.get_info()

        assert 'mock' in info
        assert info['mock'] == 'MockChartGenerator'


# ==================================================================================
# Tests for Decorator
# ==================================================================================


class TestRegisterDecorator:
    """Tests for @register_chart decorator."""

    def test_decorator_registers_chart(self, clear_registry):
        """Test decorator registers chart automatically."""

        @register_chart('decorated_chart')
        class DecoratedGenerator(ChartGenerator):
            def generate(self, data, **kwargs):
                return ChartResult(content='test', format='plotly')

        assert ChartRegistry.is_registered('decorated_chart')

    def test_decorated_chart_works(self, clear_registry):
        """Test decorated chart can be used."""

        @register_chart('working_chart')
        class WorkingGenerator(ChartGenerator):
            def generate(self, data, **kwargs):
                return ChartResult(content='success', format='plotly')

        result = ChartRegistry.generate('working_chart', data={})

        assert result.is_success
        assert result.content == 'success'


# ==================================================================================
# Integration Tests
# ==================================================================================


class TestChartRegistryIntegration:
    """Integration tests for complete workflows."""

    def test_multiple_chart_types(self):
        """Test registering and using multiple chart types."""
        ChartRegistry.register('chart1', MockChartGenerator())
        ChartRegistry.register('chart2', MockChartGenerator())
        ChartRegistry.register('chart3', MockChartGenerator())

        # Generate each chart
        result1 = ChartRegistry.generate('chart1', data={})
        result2 = ChartRegistry.generate('chart2', data={})
        result3 = ChartRegistry.generate('chart3', data={})

        assert all([r.is_success for r in [result1, result2, result3]])

    def test_error_handling(self):
        """Test error handling in chart generation."""
        ChartRegistry.register('failing', FailingChartGenerator())

        with pytest.raises(ValueError) as exc_info:
            ChartRegistry.generate('failing', data={})

        assert 'Intentional failure' in str(exc_info.value)

    def test_plotly_generator_integration(self):
        """Test Plotly generator integration."""
        ChartRegistry.register('plotly_test', MockPlotlyGenerator())

        result = ChartRegistry.generate(
            'plotly_test',
            data={'x': [1, 2, 3], 'y': [4, 5, 6]},
            title='Test Chart',
        )

        assert result.is_success
        assert result.format == 'plotly'
        assert 'data' in result.content
        assert 'layout' in result.content


# ==================================================================================
# Tests for ChartGenerator Base Class
# ==================================================================================


class TestChartGeneratorBase:
    """Tests for ChartGenerator base class."""

    def test_create_error_result(self):
        """Test creating error result."""
        generator = MockChartGenerator()
        error_result = generator._create_error_result('Test error')

        assert not error_result.is_success
        assert error_result.error == 'Test error'

    def test_validate_data_success(self):
        """Test data validation success."""
        generator = MockChartGenerator()
        data = {'x': [1, 2], 'y': [3, 4]}

        # Should not raise
        generator._validate_data(data, ['x', 'y'])

    def test_validate_data_failure(self):
        """Test data validation failure."""
        generator = MockChartGenerator()
        data = {'x': [1, 2]}  # Missing 'y'

        with pytest.raises(ValueError) as exc_info:
            generator._validate_data(data, ['x', 'y'])

        assert 'Missing required keys' in str(exc_info.value)
        assert 'y' in str(exc_info.value)

    def test_generator_repr(self):
        """Test generator string representation."""
        generator = MockChartGenerator()
        repr_str = repr(generator)

        assert 'MockChartGenerator' in repr_str


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
