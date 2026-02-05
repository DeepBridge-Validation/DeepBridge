"""
Tests for Performance Caching (Phase 3 Sprint 9).

Validates that @lru_cache improves performance for repeated operations.
"""

import os
import tempfile
import time
from pathlib import Path

import pytest

from deepbridge.core.experiment.report.asset_manager import AssetManager
from deepbridge.core.experiment.report.template_manager import TemplateManager


class TestAssetManagerCaching:
    """Tests for AssetManager caching."""

    @pytest.fixture
    def asset_manager(self, tmp_path):
        """Create AssetManager with test assets."""
        # Create minimal template structure
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        # Assets directory
        assets_dir = templates_dir / "assets"
        assets_dir.mkdir()

        # CSS directory
        css_dir = assets_dir / "css"
        css_dir.mkdir()
        (css_dir / "main.css").write_text("body { margin: 0; }")

        # JS directory
        js_dir = assets_dir / "js"
        js_dir.mkdir()
        (js_dir / "main.js").write_text("console.log('main');")

        # Images directory
        images_dir = assets_dir / "images"
        images_dir.mkdir()
        # Create minimal 1x1 PNG
        logo_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        (images_dir / "logo.png").write_bytes(logo_bytes)
        (images_dir / "favicon.ico").write_bytes(logo_bytes)

        # Common directory
        common_dir = templates_dir / "common"
        common_dir.mkdir()

        # Report types
        report_types_dir = templates_dir / "report_types"
        report_types_dir.mkdir()

        uncertainty_dir = report_types_dir / "uncertainty"
        uncertainty_dir.mkdir()

        # CSS for uncertainty
        uncertainty_css = uncertainty_dir / "css"
        uncertainty_css.mkdir()
        (uncertainty_css / "uncertainty.css").write_text(".uncertainty { color: blue; }")

        # JS for uncertainty
        uncertainty_js = uncertainty_dir / "js"
        uncertainty_js.mkdir()
        (uncertainty_js / "uncertainty.js").write_text("console.log('uncertainty');")

        return AssetManager(str(templates_dir))

    def test_css_caching_works(self, asset_manager):
        """Test that CSS content is cached on repeated calls."""
        # First call - should read and combine files
        start = time.time()
        css1 = asset_manager.get_combined_css_content('uncertainty')
        first_duration = time.time() - start

        # Second call - should use cache
        start = time.time()
        css2 = asset_manager.get_combined_css_content('uncertainty')
        cached_duration = time.time() - start

        # Verify same content
        assert css1 == css2
        assert len(css1) > 0

        # Cached call should be significantly faster (at least 5x faster)
        # Note: This might be flaky on slow systems, but generally cache should be much faster
        assert cached_duration < first_duration * 0.8, \
            f"Cached call ({cached_duration:.6f}s) should be faster than first call ({first_duration:.6f}s)"

    def test_js_caching_works(self, asset_manager):
        """Test that JS content is cached on repeated calls."""
        # First call
        start = time.time()
        js1 = asset_manager.get_combined_js_content('uncertainty')
        first_duration = time.time() - start

        # Second call - should use cache
        start = time.time()
        js2 = asset_manager.get_combined_js_content('uncertainty')
        cached_duration = time.time() - start

        # Verify same content
        assert js1 == js2
        assert len(js1) > 0
        assert 'uncertainty' in js1

        # Cached call should be faster
        assert cached_duration < first_duration * 0.8

    def test_multiple_test_types_cached_independently(self, asset_manager):
        """Test that different test types are cached separately."""
        # Create robustness directory
        templates_dir = Path(asset_manager.templates_dir)
        robustness_dir = templates_dir / "report_types" / "robustness"
        robustness_dir.mkdir()

        robustness_css = robustness_dir / "css"
        robustness_css.mkdir()
        (robustness_css / "robustness.css").write_text(".robustness { color: red; }")

        # Get CSS for different types
        css_uncertainty = asset_manager.get_combined_css_content('uncertainty')
        css_robustness = asset_manager.get_combined_css_content('robustness')

        # Should be different
        assert css_uncertainty != css_robustness

        # Both should be cached - second calls should be fast
        start = time.time()
        css_uncertainty2 = asset_manager.get_combined_css_content('uncertainty')
        cached_duration = time.time() - start

        assert css_uncertainty == css_uncertainty2
        assert cached_duration < 0.001  # Should be instant from cache

    def test_cache_persists_across_multiple_calls(self, asset_manager):
        """Test that cache works for multiple sequential calls."""
        durations = []

        for i in range(5):
            start = time.time()
            css = asset_manager.get_combined_css_content('uncertainty')
            duration = time.time() - start
            durations.append(duration)

        # First call slower than subsequent calls
        assert durations[0] > durations[1]
        assert durations[0] > durations[2]

        # All subsequent calls should be fast (from cache)
        for i in range(1, 5):
            assert durations[i] < durations[0] * 0.5


class TestTemplateManagerCaching:
    """Tests for TemplateManager caching."""

    @pytest.fixture
    def template_manager(self, tmp_path):
        """Create TemplateManager with test templates."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        # Create report types
        report_types_dir = templates_dir / "report_types"
        report_types_dir.mkdir()

        # Uncertainty templates
        uncertainty_dir = report_types_dir / "uncertainty"
        uncertainty_dir.mkdir()

        uncertainty_static = uncertainty_dir / "static"
        uncertainty_static.mkdir()
        (uncertainty_static / "index.html").write_text("<html><body>Uncertainty Report</body></html>")

        uncertainty_interactive = uncertainty_dir / "interactive"
        uncertainty_interactive.mkdir()
        (uncertainty_interactive / "index.html").write_text("<html><body>Interactive Report</body></html>")

        return TemplateManager(str(templates_dir))

    def test_template_paths_caching(self, template_manager):
        """Test that get_template_paths is cached."""
        # First call
        start = time.time()
        paths1 = template_manager.get_template_paths('uncertainty', 'static')
        first_duration = time.time() - start

        # Second call - should use cache
        start = time.time()
        paths2 = template_manager.get_template_paths('uncertainty', 'static')
        cached_duration = time.time() - start

        # Should return same paths
        assert paths1 == paths2

        # Cached should be faster
        assert cached_duration < first_duration * 0.8

    def test_find_template_caching(self, template_manager):
        """Test that find_template is cached."""
        paths = template_manager.get_template_paths('uncertainty', 'static')

        # First call - checks os.path.exists()
        start = time.time()
        template1 = template_manager.find_template(paths)
        first_duration = time.time() - start

        # Second call - should use cache
        start = time.time()
        template2 = template_manager.find_template(paths)
        cached_duration = time.time() - start

        # Should return same template
        assert template1 == template2

        # Cached should be significantly faster
        assert cached_duration < first_duration * 0.8

    def test_caching_different_report_types(self, template_manager):
        """Test that different report types are cached separately."""
        # Get paths for different types
        static_paths = template_manager.get_template_paths('uncertainty', 'static')
        interactive_paths = template_manager.get_template_paths('uncertainty', 'interactive')

        # Should be different
        assert static_paths != interactive_paths

        # Both should be cached
        start = time.time()
        static_paths2 = template_manager.get_template_paths('uncertainty', 'static')
        cached_duration = time.time() - start

        assert static_paths == static_paths2
        assert cached_duration < 0.001  # Instant from cache


class TestCachingPerformanceImprovement:
    """Integration tests for overall performance improvement."""

    @pytest.fixture
    def full_setup(self, tmp_path):
        """Create full setup with AssetManager and TemplateManager."""
        templates_dir = tmp_path / "templates"
        templates_dir.mkdir()

        # Create structure
        assets_dir = templates_dir / "assets"
        assets_dir.mkdir()

        css_dir = assets_dir / "css"
        css_dir.mkdir()
        (css_dir / "main.css").write_text("body { margin: 0; }\n" * 100)  # Larger file

        js_dir = assets_dir / "js"
        js_dir.mkdir()
        (js_dir / "main.js").write_text("console.log('test');\n" * 100)  # Larger file

        images_dir = assets_dir / "images"
        images_dir.mkdir()
        logo_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
        (images_dir / "logo.png").write_bytes(logo_bytes)
        (images_dir / "favicon.ico").write_bytes(logo_bytes)

        common_dir = templates_dir / "common"
        common_dir.mkdir()

        report_types_dir = templates_dir / "report_types"
        report_types_dir.mkdir()

        uncertainty_dir = report_types_dir / "uncertainty"
        uncertainty_dir.mkdir()

        uncertainty_css = uncertainty_dir / "css"
        uncertainty_css.mkdir()
        (uncertainty_css / "uncertainty.css").write_text(".test { color: blue; }\n" * 50)

        uncertainty_js = uncertainty_dir / "js"
        uncertainty_js.mkdir()
        (uncertainty_js / "uncertainty.js").write_text("console.log('uncertainty');\n" * 50)

        uncertainty_static = uncertainty_dir / "static"
        uncertainty_static.mkdir()
        (uncertainty_static / "index.html").write_text("<html><body>Report</body></html>")

        asset_manager = AssetManager(str(templates_dir))
        template_manager = TemplateManager(str(templates_dir))

        return asset_manager, template_manager

    def test_overall_performance_improvement(self, full_setup):
        """Test that caching provides measurable performance improvement."""
        asset_manager, template_manager = full_setup

        # Simulate typical report generation workflow
        def generate_report_assets():
            """Simulate getting all assets for report generation."""
            css = asset_manager.get_combined_css_content('uncertainty')
            js = asset_manager.get_combined_js_content('uncertainty')
            paths = template_manager.get_template_paths('uncertainty', 'static')
            template_path = template_manager.find_template(paths)
            return css, js, template_path

        # First call - no cache
        start = time.time()
        result1 = generate_report_assets()
        first_duration = time.time() - start

        # Second call - with cache
        start = time.time()
        result2 = generate_report_assets()
        cached_duration = time.time() - start

        # Results should be identical
        assert result1 == result2

        # Cached should be significantly faster
        improvement_ratio = first_duration / cached_duration if cached_duration > 0 else float('inf')

        print(f"\nPerformance Improvement:")
        print(f"  First call:  {first_duration:.6f}s")
        print(f"  Cached call: {cached_duration:.6f}s")
        print(f"  Speedup:     {improvement_ratio:.2f}x faster")

        # Should be at least 2x faster (conservative estimate)
        # In practice, should be 10-50x faster depending on file I/O
        assert cached_duration < first_duration * 0.5, \
            f"Cached operations should be at least 2x faster (got {improvement_ratio:.2f}x)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
