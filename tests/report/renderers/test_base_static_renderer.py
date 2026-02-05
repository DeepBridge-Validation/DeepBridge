"""
Tests for BaseStaticRenderer (Phase 3 Sprint 9).

Tests the template method pattern for custom chart generation.
"""

from pathlib import Path

import pytest

from deepbridge.core.experiment.report.asset_manager import AssetManager
from deepbridge.core.experiment.report.renderers.static.base_static_renderer import (
    BaseStaticRenderer,
)
from deepbridge.core.experiment.report.template_manager import TemplateManager


class TestBaseStaticRenderer:
    """Tests for BaseStaticRenderer template method pattern."""

    @pytest.fixture
    def minimal_setup(self, tmp_path):
        """Create minimal setup for BaseStaticRenderer."""
        # Create template structure
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        assets_dir = templates_dir / "assets"
        assets_dir.mkdir()

        images_dir = assets_dir / "images"
        images_dir.mkdir()

        # Create minimal logo/favicon
        logo_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        (images_dir / "logo.png").write_bytes(logo_bytes)
        (images_dir / "favicon.ico").write_bytes(logo_bytes)

        common_dir = templates_dir / "common"
        common_dir.mkdir()

        template_manager = TemplateManager(str(templates_dir))
        asset_manager = AssetManager(str(templates_dir))

        return BaseStaticRenderer(template_manager, asset_manager)

    def test_generate_custom_chart_simple(self, minimal_setup):
        """Test generating a simple custom chart."""
        renderer = minimal_setup

        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        # Define custom drawing function
        def draw_simple_line(ax, data):
            ax.plot(data['x'], data['y'])
            ax.set_xlabel('X Values')
            ax.set_ylabel('Y Values')

        # Generate chart
        result = renderer.generate_custom_chart(
            draw_simple_line,
            data={'x': [1, 2, 3, 4], 'y': [2, 4, 6, 8]},
            title='Simple Line Chart'
        )

        # Verify result
        assert result.startswith('data:image/png;base64,')
        assert len(result) > 100  # Has actual image data

    def test_generate_custom_chart_with_kwargs(self, minimal_setup):
        """Test generating chart with additional kwargs."""
        renderer = minimal_setup

        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        # Define function that accepts kwargs
        def draw_colored_scatter(ax, data, color='blue', marker='o', size=50):
            ax.scatter(data['x'], data['y'], c=color, marker=marker, s=size)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        # Generate with custom parameters
        result = renderer.generate_custom_chart(
            draw_colored_scatter,
            data={'x': [1, 2, 3], 'y': [3, 1, 4]},
            title='Scatter Plot',
            color='red',
            marker='s',
            size=100
        )

        assert result.startswith('data:image/png;base64,')

    def test_generate_custom_chart_with_seaborn(self, minimal_setup):
        """Test generating chart using seaborn."""
        renderer = minimal_setup

        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        # Use seaborn for more complex visualization
        def draw_seaborn_bar(ax, data):
            import pandas as pd
            df = pd.DataFrame({
                'category': data['categories'],
                'value': data['values']
            })
            renderer.sns.barplot(x='category', y='value', data=df, ax=ax)
            ax.set_xlabel('Categories')
            ax.set_ylabel('Values')

        result = renderer.generate_custom_chart(
            draw_seaborn_bar,
            data={
                'categories': ['A', 'B', 'C', 'D'],
                'values': [10, 25, 15, 30]
            },
            title='Bar Chart Example'
        )

        assert result.startswith('data:image/png;base64,')

    def test_generate_custom_chart_custom_figsize(self, minimal_setup):
        """Test generating chart with custom figure size."""
        renderer = minimal_setup

        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        def draw_wide_chart(ax, data):
            ax.plot(data['x'], data['y'])

        # Generate with custom figsize
        result = renderer.generate_custom_chart(
            draw_wide_chart,
            data={'x': [1, 2, 3, 4, 5], 'y': [1, 4, 9, 16, 25]},
            title='Wide Chart',
            figsize=(16, 4)  # Wide chart
        )

        assert result.startswith('data:image/png;base64,')

    def test_generate_custom_chart_no_title(self, minimal_setup):
        """Test generating chart without title."""
        renderer = minimal_setup

        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        def draw_simple(ax, data):
            ax.plot([1, 2, 3], [1, 2, 3])

        result = renderer.generate_custom_chart(
            draw_simple,
            data={}
            # No title provided
        )

        assert result.startswith('data:image/png;base64,')

    def test_generate_custom_chart_error_handling(self, minimal_setup):
        """Test error handling in custom chart generation."""
        renderer = minimal_setup

        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        # Define function that raises error
        def draw_broken(ax, data):
            raise ValueError("Intentional error")

        result = renderer.generate_custom_chart(
            draw_broken,
            data={}
        )

        # Should return empty string on error
        assert result == ""

    def test_generate_custom_chart_complex_visualization(self, minimal_setup):
        """Test complex multi-element visualization."""
        renderer = minimal_setup

        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        # Complex drawing function with multiple elements
        def draw_complex(ax, data):
            # Plot multiple series
            ax.plot(data['x'], data['y1'], label='Series 1', marker='o')
            ax.plot(data['x'], data['y2'], label='Series 2', marker='s')

            # Add grid
            ax.grid(True, alpha=0.3)

            # Add legend
            ax.legend()

            # Set labels
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')

            # Add horizontal line
            ax.axhline(y=5, color='r', linestyle='--', alpha=0.5)

        result = renderer.generate_custom_chart(
            draw_complex,
            data={
                'x': [0, 1, 2, 3, 4],
                'y1': [1, 3, 2, 5, 4],
                'y2': [2, 4, 3, 6, 5]
            },
            title='Complex Visualization'
        )

        assert result.startswith('data:image/png;base64,')
        assert len(result) > 200  # Should have substantial data


class TestTemplateMethodPatternBenefits:
    """Tests demonstrating benefits of template method pattern."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create renderer."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        assets_dir = templates_dir / "assets"
        assets_dir.mkdir()

        images_dir = assets_dir / "images"
        images_dir.mkdir()

        logo_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        (images_dir / "logo.png").write_bytes(logo_bytes)
        (images_dir / "favicon.ico").write_bytes(logo_bytes)

        common_dir = templates_dir / "common"
        common_dir.mkdir()

        template_manager = TemplateManager(str(templates_dir))
        asset_manager = AssetManager(str(templates_dir))

        return BaseStaticRenderer(template_manager, asset_manager)

    def test_before_and_after_comparison(self, renderer):
        """
        Demonstrate code reduction with template method pattern.

        Before: 50-80 lines of boilerplate per chart
        After: 5-10 lines per chart
        """
        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        # AFTER: Using template method (5 lines)
        def draw_coverage_chart(ax, data):
            ax.plot(data['alpha'], data['coverage'], marker='o')
            ax.set_xlabel('Alpha Level')
            ax.set_ylabel('Coverage')
            ax.grid(True, alpha=0.3)

        chart = renderer.generate_custom_chart(
            draw_coverage_chart,
            data={
                'alpha': [0.1, 0.2, 0.3, 0.4, 0.5],
                'coverage': [0.9, 0.85, 0.8, 0.75, 0.7]
            },
            title='Coverage vs Alpha'
        )

        assert chart.startswith('data:image/png;base64,')

        # Benefits demonstrated:
        # - No manual figure/axes creation
        # - No buffer management
        # - No base64 encoding
        # - No memory cleanup (plt.close)
        # - No error handling boilerplate
        # - Consistent behavior across all charts

    def test_easy_to_test_drawing_logic(self, renderer):
        """Demonstrate that drawing logic can be tested in isolation."""
        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        # Drawing function can be tested independently
        def draw_test_chart(ax, data, test_param=False):
            """Drawing function that can be unit tested."""
            if test_param:
                ax.plot(data['x'], data['y'], 'r--')
            else:
                ax.plot(data['x'], data['y'], 'b-')

        # Test different parameters
        chart1 = renderer.generate_custom_chart(
            draw_test_chart,
            data={'x': [1, 2, 3], 'y': [1, 4, 9]},
            test_param=False
        )

        chart2 = renderer.generate_custom_chart(
            draw_test_chart,
            data={'x': [1, 2, 3], 'y': [1, 4, 9]},
            test_param=True
        )

        # Both should succeed (content will differ due to test_param)
        assert chart1.startswith('data:image/png;base64,')
        assert chart2.startswith('data:image/png;base64,')


class TestExistingChartMethods:
    """Tests for existing chart generation methods."""

    @pytest.fixture
    def renderer(self, tmp_path):
        """Create renderer."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        assets_dir = templates_dir / "assets"
        assets_dir.mkdir()

        images_dir = assets_dir / "images"
        images_dir.mkdir()

        logo_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        (images_dir / "logo.png").write_bytes(logo_bytes)
        (images_dir / "favicon.ico").write_bytes(logo_bytes)

        common_dir = templates_dir / "common"
        common_dir.mkdir()

        template_manager = TemplateManager(str(templates_dir))
        asset_manager = AssetManager(str(templates_dir))

        return BaseStaticRenderer(template_manager, asset_manager)

    def test_generate_bar_chart(self, renderer):
        """Test existing bar chart generation."""
        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        chart = renderer.generate_chart(
            'bar',
            data={
                'x': ['A', 'B', 'C'],
                'y': [10, 20, 15],
                'x_label': 'Category',
                'y_label': 'Value'
            },
            title='Bar Chart Test'
        )

        assert chart.startswith('data:image/png;base64,')

    def test_generate_line_chart(self, renderer):
        """Test existing line chart generation."""
        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        chart = renderer.generate_chart(
            'line',
            data={
                'x': [1, 2, 3, 4],
                'y': [2, 4, 3, 5],
                'x_label': 'X',
                'y_label': 'Y'
            },
            title='Line Chart Test'
        )

        assert chart.startswith('data:image/png;base64,')

    @pytest.mark.xfail(reason="Existing bug: DataFrame.append() deprecated in pandas 2.0+")
    def test_generate_boxplot_chart(self, renderer):
        """Test existing boxplot generation."""
        if not renderer.has_visualization_libs:
            pytest.skip("Visualization libraries not available")

        chart = renderer.generate_chart(
            'boxplot',
            data={
                'values': [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]],
                'labels': ['Group A', 'Group B', 'Group C'],
                'y_label': 'Value'
            },
            title='Boxplot Test'
        )

        assert chart.startswith('data:image/png;base64,')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
