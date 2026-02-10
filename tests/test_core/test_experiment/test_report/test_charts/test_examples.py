"""
Tests for chart examples module.

Coverage Target: 100%
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from deepbridge.core.experiment.report.charts.examples import (
    LineChartGenerator,
    BarChartGenerator,
    CoverageChartGenerator,
    SimpleBarImageGenerator,
    register_example_charts,
)


class TestLineChartGenerator:
    """Tests for LineChartGenerator class"""

    def test_create_plotly_figure_basic(self):
        """Test creating a basic line chart"""
        generator = LineChartGenerator()
        data = {'x': [1, 2, 3], 'y': [4, 5, 6]}

        figure = generator._create_plotly_figure(data)

        assert 'data' in figure
        assert 'layout' in figure
        assert len(figure['data']) == 1
        assert figure['data'][0]['type'] == 'scatter'
        assert figure['data'][0]['mode'] == 'lines+markers'

    def test_create_plotly_figure_with_kwargs(self):
        """Test line chart with custom parameters"""
        generator = LineChartGenerator()
        data = {'x': [1, 2, 3], 'y': [4, 5, 6]}

        figure = generator._create_plotly_figure(
            data,
            title='Custom Title',
            series_name='My Series',
            color='#ff0000',
            xaxis_title='Time',
            yaxis_title='Value'
        )

        assert figure['layout']['title'] == 'Custom Title'
        assert figure['data'][0]['name'] == 'My Series'
        assert figure['data'][0]['line']['color'] == '#ff0000'
        assert figure['layout']['xaxis']['title'] == 'Time'
        assert figure['layout']['yaxis']['title'] == 'Value'

    def test_create_plotly_figure_validates_data(self):
        """Test that line chart validates required fields"""
        generator = LineChartGenerator()
        data = {'x': [1, 2, 3]}  # Missing 'y'

        with pytest.raises(ValueError):
            generator._create_plotly_figure(data)


class TestBarChartGenerator:
    """Tests for BarChartGenerator class"""

    def test_create_plotly_figure_basic(self):
        """Test creating a basic bar chart"""
        generator = BarChartGenerator()
        data = {'labels': ['A', 'B', 'C'], 'values': [10, 20, 15]}

        figure = generator._create_plotly_figure(data)

        assert 'data' in figure
        assert 'layout' in figure
        assert len(figure['data']) == 1
        assert figure['data'][0]['type'] == 'bar'
        assert figure['data'][0]['x'] == ['A', 'B', 'C']
        assert figure['data'][0]['y'] == [10, 20, 15]

    def test_create_plotly_figure_with_kwargs(self):
        """Test bar chart with custom parameters"""
        generator = BarChartGenerator()
        data = {'labels': ['A', 'B'], 'values': [10, 20]}

        figure = generator._create_plotly_figure(
            data,
            title='My Bar Chart',
            series_name='Values',
            color='#00ff00',
            xaxis_title='Categories',
            yaxis_title='Count'
        )

        assert figure['layout']['title'] == 'My Bar Chart'
        assert figure['data'][0]['name'] == 'Values'
        assert figure['data'][0]['marker']['color'] == '#00ff00'
        assert figure['layout']['xaxis']['title'] == 'Categories'
        assert figure['layout']['yaxis']['title'] == 'Count'

    def test_create_plotly_figure_validates_data(self):
        """Test that bar chart validates required fields"""
        generator = BarChartGenerator()
        data = {'labels': ['A', 'B']}  # Missing 'values'

        with pytest.raises(ValueError):
            generator._create_plotly_figure(data)


class TestCoverageChartGenerator:
    """Tests for CoverageChartGenerator class"""

    def test_create_plotly_figure_basic(self):
        """Test creating a coverage chart"""
        generator = CoverageChartGenerator()
        data = {
            'alphas': [0.1, 0.2, 0.3],
            'coverage': [0.91, 0.81, 0.71],
            'expected': [0.90, 0.80, 0.70]
        }

        figure = generator._create_plotly_figure(data)

        assert 'data' in figure
        assert 'layout' in figure
        assert len(figure['data']) == 2  # Actual and expected
        assert figure['data'][0]['name'] == 'Actual Coverage'
        assert figure['data'][1]['name'] == 'Expected Coverage'

    def test_create_plotly_figure_with_custom_title(self):
        """Test coverage chart with custom title"""
        generator = CoverageChartGenerator()
        data = {
            'alphas': [0.1, 0.2],
            'coverage': [0.91, 0.81],
            'expected': [0.90, 0.80]
        }

        figure = generator._create_plotly_figure(data, title='Calibration Test')

        assert figure['layout']['title'] == 'Calibration Test'

    def test_create_plotly_figure_validates_data(self):
        """Test that coverage chart validates required fields"""
        generator = CoverageChartGenerator()
        data = {'alphas': [0.1, 0.2], 'coverage': [0.91, 0.81]}  # Missing 'expected'

        with pytest.raises(ValueError):
            generator._create_plotly_figure(data)


class TestSimpleBarImageGenerator:
    """Tests for SimpleBarImageGenerator class"""

    def test_create_image_basic(self):
        """Test creating a static bar image"""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('base64.b64encode') as mock_base64, \
             patch('io.BytesIO') as mock_bytesio:

            generator = SimpleBarImageGenerator()
            data = {'labels': ['A', 'B', 'C'], 'values': [10, 20, 15]}

            # Mock the figure and buffer
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            mock_buffer = MagicMock()
            mock_bytesio.return_value = mock_buffer
            mock_buffer.getvalue.return_value = b'fake_image_data'

            mock_base64.return_value = b'encoded_image'

            img_base64, img_format = generator._create_image(data)

            assert img_format == 'png'
            assert img_base64 == 'encoded_image'
            mock_subplots.assert_called_once()
            mock_ax.bar.assert_called_once()

    def test_create_image_with_kwargs(self):
        """Test creating image with custom parameters"""
        with patch('matplotlib.pyplot.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.close') as mock_close, \
             patch('base64.b64encode') as mock_base64, \
             patch('io.BytesIO') as mock_bytesio:

            generator = SimpleBarImageGenerator()
            data = {'labels': ['A', 'B'], 'values': [10, 20]}

            # Mock the figure and buffer
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)

            mock_buffer = MagicMock()
            mock_bytesio.return_value = mock_buffer
            mock_buffer.getvalue.return_value = b'fake_image_data'

            mock_base64.return_value = b'encoded_image'

            img_base64, img_format = generator._create_image(
                data,
                title='Test Chart',
                color='#ff0000',
                xlabel='X Label',
                ylabel='Y Label',
                figsize=(12, 8)
            )

            mock_subplots.assert_called_once_with(figsize=(12, 8))
            mock_ax.set_title.assert_called_once_with('Test Chart')
            mock_ax.set_xlabel.assert_called_once_with('X Label')
            mock_ax.set_ylabel.assert_called_once_with('Y Label')
            mock_ax.bar.assert_called_once_with(['A', 'B'], [10, 20], color='#ff0000')

    def test_create_image_missing_matplotlib(self):
        """Test error when matplotlib is not available"""
        # Mock to make matplotlib import fail inside _create_image
        import sys
        matplotlib_backup = sys.modules.get('matplotlib.pyplot')

        try:
            # Remove matplotlib from modules to simulate ImportError
            if 'matplotlib.pyplot' in sys.modules:
                del sys.modules['matplotlib.pyplot']
            if 'matplotlib' in sys.modules:
                del sys.modules['matplotlib']

            # Patch the import to raise ImportError
            with patch.dict('sys.modules', {'matplotlib.pyplot': None, 'matplotlib': None}):
                generator = SimpleBarImageGenerator()
                data = {'labels': ['A', 'B'], 'values': [10, 20]}

                with pytest.raises(ValueError, match='Matplotlib required'):
                    generator._create_image(data)
        finally:
            # Restore matplotlib
            if matplotlib_backup is not None:
                sys.modules['matplotlib.pyplot'] = matplotlib_backup

    def test_create_image_validates_data(self):
        """Test that image generator validates required fields"""
        generator = SimpleBarImageGenerator()
        data = {'labels': ['A', 'B']}  # Missing 'values'

        with pytest.raises(ValueError):
            generator._create_image(data)


class TestRegisterExampleCharts:
    """Tests for chart registration"""

    def test_register_example_charts(self):
        """Test registering example charts"""
        from deepbridge.core.experiment.report.charts.registry import ChartRegistry

        # Register charts
        register_example_charts()

        # Check that charts are registered
        charts = ChartRegistry.list_charts()
        assert 'line_chart' in charts
        assert 'bar_chart' in charts
        assert 'coverage_chart' in charts
        # bar_image may or may not be registered depending on execution order

    def test_register_example_charts_idempotent(self):
        """Test that registering multiple times doesn't cause issues"""
        from deepbridge.core.experiment.report.charts.registry import ChartRegistry

        initial_count = ChartRegistry.count()

        # Register twice
        register_example_charts()
        count_after_first = ChartRegistry.count()

        register_example_charts()
        count_after_second = ChartRegistry.count()

        # Count should not increase on second registration
        assert count_after_first == count_after_second


class TestIntegration:
    """Integration tests for chart generators"""

    def test_line_chart_full_generation(self):
        """Test complete line chart generation"""
        from deepbridge.core.experiment.report.charts.registry import ChartRegistry

        data = {'x': [1, 2, 3, 4], 'y': [2.1, 3.5, 2.8, 4.2]}

        result = ChartRegistry.generate(
            'line_chart',
            data=data,
            title='Test Line Chart',
            xaxis_title='X Axis',
            yaxis_title='Y Axis'
        )

        assert result.is_success
        assert result.format == 'plotly'

    def test_bar_chart_full_generation(self):
        """Test complete bar chart generation"""
        from deepbridge.core.experiment.report.charts.registry import ChartRegistry

        data = {'labels': ['A', 'B', 'C'], 'values': [10, 20, 15]}

        result = ChartRegistry.generate(
            'bar_chart',
            data=data,
            title='Test Bar Chart'
        )

        assert result.is_success
        assert result.format == 'plotly'

    def test_coverage_chart_full_generation(self):
        """Test complete coverage chart generation"""
        from deepbridge.core.experiment.report.charts.registry import ChartRegistry

        data = {
            'alphas': [0.1, 0.2, 0.3],
            'coverage': [0.91, 0.81, 0.71],
            'expected': [0.90, 0.80, 0.70]
        }

        result = ChartRegistry.generate(
            'coverage_chart',
            data=data,
            title='Coverage Analysis'
        )

        assert result.is_success
        assert result.format == 'plotly'
