"""
Tests for file_utils module.

Tests the file discovery utilities created in Phase 2 Sprint 5-6.
"""

import os
import tempfile
from pathlib import Path

import pytest

from deepbridge.core.experiment.report.utils import file_utils


class TestFindFilesByPattern:
    """Tests for find_files_by_pattern()."""

    def test_find_css_files(self, tmp_path):
        """Test finding CSS files with pattern."""
        # Create test CSS files
        (tmp_path / 'main.css').write_text('body {}')
        (tmp_path / 'components.css').write_text('.btn {}')

        files = file_utils.find_files_by_pattern(str(tmp_path), '*.css')

        assert len(files) == 2
        assert any('main.css' in f for f in files)
        assert any('components.css' in f for f in files)

    def test_find_js_files_with_subdirs(self, tmp_path):
        """Test finding JS files in subdirectories."""
        # Create structure
        (tmp_path / 'main.js').write_text('console.log()')
        charts_dir = tmp_path / 'charts'
        charts_dir.mkdir()
        (charts_dir / 'line.js').write_text('// line chart')

        files = file_utils.find_files_by_pattern(str(tmp_path), '**/*.js')

        assert len(files) >= 2

    def test_nonexistent_directory(self):
        """Test handling of non-existent directory."""
        files = file_utils.find_files_by_pattern('/nonexistent/path', '*.css')
        assert files == []


class TestFindCssFiles:
    """Tests for find_css_files()."""

    def test_find_main_css(self, tmp_path):
        """Test finding main CSS file."""
        (tmp_path / 'main.css').write_text('body {}')

        files = file_utils.find_css_files(str(tmp_path))

        assert 'main' in files
        assert files['main'] == 'main.css'

    def test_find_styles_css_as_main(self, tmp_path):
        """Test finding styles.css as main."""
        (tmp_path / 'styles.css').write_text('body {}')

        files = file_utils.find_css_files(str(tmp_path))

        assert 'main' in files
        assert files['main'] == 'styles.css'

    def test_find_components_css(self, tmp_path):
        """Test finding CSS files in components directory."""
        # Create main CSS
        (tmp_path / 'main.css').write_text('body {}')

        # Create components
        components_dir = tmp_path / 'components'
        components_dir.mkdir()
        (components_dir / 'buttons.css').write_text('.btn {}')
        (components_dir / 'cards.css').write_text('.card {}')

        files = file_utils.find_css_files(str(tmp_path))

        assert 'main' in files
        assert 'buttons' in files
        assert files['buttons'] == 'components/buttons.css'
        assert 'cards' in files

    def test_empty_directory(self, tmp_path):
        """Test handling of empty directory."""
        files = file_utils.find_css_files(str(tmp_path))
        assert files == {}


class TestFindJsFiles:
    """Tests for find_js_files()."""

    def test_find_main_js(self, tmp_path):
        """Test finding main.js."""
        (tmp_path / 'main.js').write_text("console.log('main')")

        files = file_utils.find_js_files(str(tmp_path))

        assert 'main' in files
        assert files['main'] == 'main.js'

    def test_find_utils_js(self, tmp_path):
        """Test finding utils.js."""
        (tmp_path / 'utils.js').write_text('// utils')

        files = file_utils.find_js_files(str(tmp_path))

        assert 'utils' in files
        assert files['utils'] == 'utils.js'

    def test_find_js_in_subdirs(self, tmp_path):
        """Test finding JS in common subdirectories."""
        # Create main
        (tmp_path / 'main.js').write_text("console.log('main')")

        # Create charts subdir
        charts_dir = tmp_path / 'charts'
        charts_dir.mkdir()
        (charts_dir / 'line.js').write_text('// line chart')
        (charts_dir / 'bar.js').write_text('// bar chart')

        files = file_utils.find_js_files(str(tmp_path))

        assert 'main' in files
        assert 'charts_line' in files
        assert files['charts_line'] == 'charts/line.js'
        assert 'charts_bar' in files


class TestFindAssetPath:
    """Tests for find_asset_path()."""

    def test_find_interactive_css(self, tmp_path):
        """Test finding interactive CSS path."""
        # Create structure
        css_path = (
            tmp_path / 'report_types' / 'uncertainty' / 'interactive' / 'css'
        )
        css_path.mkdir(parents=True)

        result = file_utils.find_asset_path(
            str(tmp_path), 'uncertainty', 'css', 'interactive'
        )

        assert result is not None
        assert 'uncertainty' in result
        assert 'interactive' in result
        assert 'css' in result

    def test_find_static_js(self, tmp_path):
        """Test finding static JS path."""
        # Create structure
        js_path = tmp_path / 'report_types' / 'robustness' / 'static' / 'js'
        js_path.mkdir(parents=True)

        result = file_utils.find_asset_path(
            str(tmp_path), 'robustness', 'js', 'static'
        )

        assert result is not None
        assert 'robustness' in result
        assert 'static' in result

    def test_find_nonexistent_path(self, tmp_path):
        """Test handling of non-existent path."""
        result = file_utils.find_asset_path(
            str(tmp_path), 'nonexistent', 'css'
        )

        assert result is None


class TestReadHtmlFiles:
    """Tests for read_html_files()."""

    def test_read_html_files(self, tmp_path):
        """Test reading HTML files from directory."""
        (tmp_path / 'header.html').write_text('<header>Header</header>')
        (tmp_path / 'footer.html').write_text('<footer>Footer</footer>')

        files = file_utils.read_html_files(str(tmp_path))

        assert len(files) == 2
        assert 'header.html' in files
        assert '<header>' in files['header.html']
        assert 'footer.html' in files
        assert '<footer>' in files['footer.html']

    def test_empty_directory(self, tmp_path):
        """Test handling of empty directory."""
        files = file_utils.read_html_files(str(tmp_path))
        assert files == {}


class TestCombineTextFiles:
    """Tests for combine_text_files()."""

    def test_combine_css_files(self, tmp_path):
        """Test combining CSS files."""
        file1 = tmp_path / 'base.css'
        file2 = tmp_path / 'components.css'

        file1.write_text('body { margin: 0; }')
        file2.write_text('.btn { padding: 10px; }')

        combined = file_utils.combine_text_files([str(file1), str(file2)])

        assert 'body { margin: 0; }' in combined
        assert '.btn { padding: 10px; }' in combined

    def test_custom_separator(self, tmp_path):
        """Test combining with custom separator."""
        file1 = tmp_path / 'file1.txt'
        file2 = tmp_path / 'file2.txt'

        file1.write_text('Content 1')
        file2.write_text('Content 2')

        combined = file_utils.combine_text_files(
            [str(file1), str(file2)], separator='\n---\n'
        )

        assert 'Content 1' in combined
        assert 'Content 2' in combined
        assert '\n---\n' in combined

    def test_empty_list(self):
        """Test combining empty file list."""
        combined = file_utils.combine_text_files([])
        assert combined == ''


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
