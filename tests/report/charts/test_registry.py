"""
Tests for Chart Registry.

Tests the chart registration and generation system created in Phase 2 Sprint 7-8.
"""

import pytest
from deepbridge.core.experiment.report.charts import (
    ChartRegistry,
    ChartGenerator,
    ChartResult,
    register_chart
)


# Mock Generators for Testing
class MockLineChart(ChartGenerator):
    """Mock line chart generator."""
    def generate(self, data, **kwargs):
        return ChartResult(
            content='{"type": "line", "data": []}',
            format='plotly',
            metadata={'title': kwargs.get('title', 'Line Chart')}
        )


class MockBarChart(ChartGenerator):
    """Mock bar chart generator."""
    def generate(self, data, **kwargs):
        return ChartResult(
            content='{"type": "bar", "data": []}',
            format='plotly',
            metadata={'title': kwargs.get('title', 'Bar Chart')}
        )


class FailingChart(ChartGenerator):
    """Chart generator that always fails."""
    def generate(self, data, **kwargs):
        raise ValueError("Chart generation failed")


# ==================================================================================
# Tests
# ==================================================================================

class TestChartResult:
    """Tests for ChartResult dataclass."""

    def test_successful_result(self):
        """Test creating successful result."""
        result = ChartResult(
            content='<div>chart</div>',
            format='html',
            metadata={'title': 'Test Chart'}
        )

        assert result.is_success
        assert not result.is_base64
        assert result.is_interactive

    def test_error_result(self):
        """Test creating error result."""
        result = ChartResult(
            content='',
            format='plotly',
            error='Generation failed'
        )

        assert not result.is_success
        assert result.error == 'Generation failed'

    def test_base64_format(self):
        """Test base64 format detection."""
        result = ChartResult(content='abc123', format='png')

        assert result.is_base64
        assert not result.is_interactive

    def test_interactive_format(self):
        """Test interactive format detection."""
        result = ChartResult(content='{}', format='plotly')

        assert result.is_interactive
        assert not result.is_base64

    def test_result_repr(self):
        """Test string representation."""
        result = ChartResult(content='data', format='plotly')
        repr_str = repr(result)

        assert 'ChartResult' in repr_str
        assert 'plotly' in repr_str


class TestChartRegistry:
    """Tests for ChartRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        ChartRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        ChartRegistry.clear()

    def test_register_chart(self):
        """Test registering a chart generator."""
        generator = MockLineChart()
        ChartRegistry.register('line_chart', generator)

        assert ChartRegistry.is_registered('line_chart')
        assert ChartRegistry.count() == 1

    def test_register_multiple_charts(self):
        """Test registering multiple charts."""
        ChartRegistry.register('line_chart', MockLineChart())
        ChartRegistry.register('bar_chart', MockBarChart())

        assert ChartRegistry.count() == 2
        assert 'line_chart' in ChartRegistry.list_charts()
        assert 'bar_chart' in ChartRegistry.list_charts()

    def test_register_invalid_type(self):
        """Test registering non-ChartGenerator raises error."""
        with pytest.raises(TypeError):
            ChartRegistry.register('invalid', "not a generator")

    def test_register_duplicate_warns(self):
        """Test registering duplicate name overwrites."""
        ChartRegistry.register('chart', MockLineChart())
        ChartRegistry.register('chart', MockBarChart())  # Overwrite

        assert ChartRegistry.count() == 1

    def test_unregister_chart(self):
        """Test unregistering a chart."""
        ChartRegistry.register('line_chart', MockLineChart())
        assert ChartRegistry.is_registered('line_chart')

        result = ChartRegistry.unregister('line_chart')

        assert result is True
        assert not ChartRegistry.is_registered('line_chart')
        assert ChartRegistry.count() == 0

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent chart."""
        result = ChartRegistry.unregister('nonexistent')
        assert result is False

    def test_generate_chart(self):
        """Test generating a chart."""
        ChartRegistry.register('line_chart', MockLineChart())

        result = ChartRegistry.generate(
            'line_chart',
            data={'x': [1, 2], 'y': [3, 4]},
            title='Test Chart'
        )

        assert result.is_success
        assert result.format == 'plotly'
        assert result.metadata['title'] == 'Test Chart'

    def test_generate_nonexistent_chart(self):
        """Test generating non-existent chart raises error."""
        with pytest.raises(ValueError) as exc_info:
            ChartRegistry.generate('nonexistent', data={})

        assert 'not registered' in str(exc_info.value)

    def test_list_charts(self):
        """Test listing registered charts."""
        ChartRegistry.register('bar_chart', MockBarChart())
        ChartRegistry.register('line_chart', MockLineChart())

        charts = ChartRegistry.list_charts()

        assert len(charts) == 2
        assert charts == ['bar_chart', 'line_chart']  # Sorted

    def test_is_registered(self):
        """Test checking if chart is registered."""
        assert not ChartRegistry.is_registered('line_chart')

        ChartRegistry.register('line_chart', MockLineChart())

        assert ChartRegistry.is_registered('line_chart')

    def test_get_generator(self):
        """Test getting generator instance."""
        generator = MockLineChart()
        ChartRegistry.register('line_chart', generator)

        retrieved = ChartRegistry.get_generator('line_chart')

        assert retrieved is generator

    def test_get_nonexistent_generator(self):
        """Test getting non-existent generator returns None."""
        result = ChartRegistry.get_generator('nonexistent')
        assert result is None

    def test_clear(self):
        """Test clearing all registrations."""
        ChartRegistry.register('chart1', MockLineChart())
        ChartRegistry.register('chart2', MockBarChart())
        assert ChartRegistry.count() == 2

        ChartRegistry.clear()

        assert ChartRegistry.count() == 0

    def test_count(self):
        """Test counting registered charts."""
        assert ChartRegistry.count() == 0

        ChartRegistry.register('chart1', MockLineChart())
        assert ChartRegistry.count() == 1

        ChartRegistry.register('chart2', MockBarChart())
        assert ChartRegistry.count() == 2

    def test_get_info(self):
        """Test getting registry information."""
        ChartRegistry.register('line', MockLineChart())
        ChartRegistry.register('bar', MockBarChart())

        info = ChartRegistry.get_info()

        assert info['line'] == 'MockLineChart'
        assert info['bar'] == 'MockBarChart'


class TestRegisterChartDecorator:
    """Tests for @register_chart decorator."""

    def setup_method(self):
        """Clear registry before each test."""
        ChartRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        ChartRegistry.clear()

    def test_decorator_registers_class(self):
        """Test decorator automatically registers chart."""
        @register_chart('decorated_chart')
        class DecoratedChart(ChartGenerator):
            def generate(self, data, **kwargs):
                return ChartResult(content='{}', format='plotly')

        assert ChartRegistry.is_registered('decorated_chart')

    def test_decorated_chart_generates(self):
        """Test decorated chart can generate."""
        @register_chart('test_chart')
        class TestChart(ChartGenerator):
            def generate(self, data, **kwargs):
                return ChartResult(
                    content='test_content',
                    format='html'
                )

        result = ChartRegistry.generate('test_chart', data={})

        assert result.is_success
        assert result.content == 'test_content'


class TestChartGenerationErrors:
    """Tests for error handling during chart generation."""

    def setup_method(self):
        """Clear registry before each test."""
        ChartRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        ChartRegistry.clear()

    def test_generator_exception_propagates(self):
        """Test exceptions from generators propagate."""
        ChartRegistry.register('failing', FailingChart())

        with pytest.raises(ValueError) as exc_info:
            ChartRegistry.generate('failing', data={})

        assert 'Chart generation failed' in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
