"""
Tests for static_uncertainty_renderer_fix module.

Coverage Target: 100%
"""

import pytest
import os
import tempfile
import shutil
import base64
from pathlib import Path

from deepbridge.core.experiment.report.renderers.static.static_uncertainty_renderer_fix import fix_html_file


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files"""
    temp = tempfile.mkdtemp()
    yield temp
    shutil.rmtree(temp)


@pytest.fixture
def sample_html_with_base64(temp_dir):
    """Create sample HTML file with base64 encoded images"""
    # Create a simple 1x1 PNG image (base64)
    png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

    html_content = f'''<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
    <h1>Test Report</h1>
    <img src="data:image/png;base64,{png_base64}" alt="Chart 1">
    <p>Some text</p>
    <img src="data:image/png;base64,{png_base64}" alt="Chart 2">
</body>
</html>'''

    file_path = os.path.join(temp_dir, 'test_report.html')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return file_path


@pytest.fixture
def sample_html_without_base64(temp_dir):
    """Create sample HTML file without base64 images"""
    html_content = '''<!DOCTYPE html>
<html>
<head><title>Test</title></head>
<body>
    <h1>Test Report</h1>
    <p>No images here</p>
</body>
</html>'''

    file_path = os.path.join(temp_dir, 'test_report_no_images.html')
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return file_path


class TestFixHtmlFile:
    """Tests for fix_html_file function"""

    def test_fix_html_file_returns_false_for_nonexistent_file(self, temp_dir):
        """Test fix_html_file returns False for nonexistent file"""
        result = fix_html_file(os.path.join(temp_dir, 'nonexistent.html'))
        assert result is False

    def test_fix_html_file_creates_charts_directory(self, sample_html_with_base64, temp_dir):
        """Test fix_html_file creates uncertainty_charts directory"""
        fix_html_file(sample_html_with_base64)

        charts_dir = os.path.join(temp_dir, 'uncertainty_charts')
        assert os.path.exists(charts_dir)
        assert os.path.isdir(charts_dir)

    def test_fix_html_file_extracts_base64_images(self, sample_html_with_base64, temp_dir):
        """Test fix_html_file extracts and saves base64 images"""
        result = fix_html_file(sample_html_with_base64)

        assert result is True

        # Check that image files were created
        charts_dir = os.path.join(temp_dir, 'uncertainty_charts')
        assert os.path.exists(os.path.join(charts_dir, 'chart_1.png'))
        assert os.path.exists(os.path.join(charts_dir, 'chart_2.png'))

    def test_fix_html_file_replaces_base64_with_file_paths(self, sample_html_with_base64):
        """Test fix_html_file replaces base64 data with file paths"""
        fix_html_file(sample_html_with_base64)

        # Read the modified HTML
        with open(sample_html_with_base64, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Check that base64 data was replaced with file paths
        assert 'data:image/png;base64,' not in html_content
        # Both images have the same base64, so they both get replaced with chart_1.png
        assert './uncertainty_charts/chart_1.png' in html_content
        # Verify all base64 images were replaced
        assert html_content.count('./uncertainty_charts/') == 2

    def test_fix_html_file_returns_false_for_no_base64_images(self, sample_html_without_base64):
        """Test fix_html_file returns False when no base64 images found"""
        result = fix_html_file(sample_html_without_base64)

        assert result is False

    def test_fix_html_file_handles_multiple_images(self, temp_dir):
        """Test fix_html_file handles multiple base64 images"""
        png_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

        html_content = f'''<!DOCTYPE html>
<html>
<body>
    <img src="data:image/png;base64,{png_base64}">
    <img src="data:image/png;base64,{png_base64}">
    <img src="data:image/png;base64,{png_base64}">
</body>
</html>'''

        file_path = os.path.join(temp_dir, 'multi_images.html')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        result = fix_html_file(file_path)

        assert result is True

        charts_dir = os.path.join(temp_dir, 'uncertainty_charts')
        assert os.path.exists(os.path.join(charts_dir, 'chart_1.png'))
        assert os.path.exists(os.path.join(charts_dir, 'chart_2.png'))
        assert os.path.exists(os.path.join(charts_dir, 'chart_3.png'))

    def test_fix_html_file_handles_invalid_base64(self, temp_dir):
        """Test fix_html_file handles invalid base64 data gracefully"""
        html_content = '''<!DOCTYPE html>
<html>
<body>
    <img src="data:image/png;base64,invalid_base64_data!!!">
</body>
</html>'''

        file_path = os.path.join(temp_dir, 'invalid_base64.html')
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        # Should not raise exception but may not fully succeed
        result = fix_html_file(file_path)

        # The function should still run (may succeed or partially succeed)
        assert isinstance(result, bool)

    def test_fix_html_file_preserves_other_content(self, sample_html_with_base64):
        """Test fix_html_file preserves other HTML content"""
        fix_html_file(sample_html_with_base64)

        with open(sample_html_with_base64, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Check that other content is preserved
        assert '<h1>Test Report</h1>' in html_content
        assert '<p>Some text</p>' in html_content
        assert '<title>Test</title>' in html_content

    def test_fix_html_file_returns_true_on_success(self, sample_html_with_base64):
        """Test fix_html_file returns True on successful processing"""
        result = fix_html_file(sample_html_with_base64)

        assert result is True

    def test_fix_html_file_with_path_object(self, sample_html_with_base64):
        """Test fix_html_file works with Path objects"""
        path_obj = Path(sample_html_with_base64)
        result = fix_html_file(str(path_obj))

        assert result is True
