"""
Tests for CSSManager class.

Coverage Target: 100%
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from deepbridge.core.experiment.report.css_manager import CSSManager, compile_report_css


@pytest.fixture
def temp_templates_dir():
    """Create temporary templates directory with CSS files"""
    temp_dir = tempfile.mkdtemp()

    # Create base CSS file
    base_css = """/* Base Styles */
body { font-family: Arial; }
.container { max-width: 1200px; }
"""
    with open(Path(temp_dir) / 'base_styles.css', 'w') as f:
        f.write(base_css)

    # Create components CSS file
    components_css = """/* Components */
.button { padding: 10px; }
.card { border: 1px solid #ccc; }
"""
    with open(Path(temp_dir) / 'report_components.css', 'w') as f:
        f.write(components_css)

    # Create custom CSS for robustness report
    custom_dir = Path(temp_dir) / 'report_types' / 'robustness' / 'interactive'
    custom_dir.mkdir(parents=True, exist_ok=True)

    custom_css = """/* Robustness Custom */
.robustness-specific { color: blue; }
"""
    with open(custom_dir / 'robustness_custom.css', 'w') as f:
        f.write(custom_css)

    # Create custom CSS in css subfolder for resilience
    css_subdir = Path(temp_dir) / 'report_types' / 'resilience' / 'interactive' / 'css'
    css_subdir.mkdir(parents=True, exist_ok=True)

    resilience_css = """/* Resilience Custom */
.resilience-specific { color: green; }
"""
    with open(css_subdir / 'resilience_custom.css', 'w') as f:
        f.write(resilience_css)

    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_incomplete_dir():
    """Create temporary directory without all required CSS files"""
    temp_dir = tempfile.mkdtemp()

    # Only create base CSS, missing components
    base_css = "/* Base only */"
    with open(Path(temp_dir) / 'base_styles.css', 'w') as f:
        f.write(base_css)

    yield temp_dir
    shutil.rmtree(temp_dir)


class TestInitialization:
    """Tests for CSSManager initialization"""

    def test_init_with_custom_templates_dir(self, temp_templates_dir):
        """Test initialization with custom templates directory"""
        manager = CSSManager(temp_templates_dir)

        assert manager.templates_dir == Path(temp_templates_dir)
        assert manager.base_css_path == Path(temp_templates_dir) / 'base_styles.css'
        assert manager.components_css_path == Path(temp_templates_dir) / 'report_components.css'

    def test_init_with_none_uses_default(self):
        """Test initialization with None uses default templates dir"""
        manager = CSSManager(None)

        assert manager.templates_dir is not None
        assert 'templates' in str(manager.templates_dir)


class TestGetCompiledCSS:
    """Tests for get_compiled_css method"""

    def test_get_compiled_css_includes_all_layers(self, temp_templates_dir):
        """Test get_compiled_css includes base, components, and custom CSS"""
        manager = CSSManager(temp_templates_dir)

        result = manager.get_compiled_css('robustness')

        # Should include all three layers
        assert '/* Base Styles */' in result
        assert '/* Components */' in result
        assert '/* Robustness Custom */' in result

    def test_get_compiled_css_without_custom(self, temp_templates_dir):
        """Test get_compiled_css works without custom CSS"""
        manager = CSSManager(temp_templates_dir)

        # Request report type with no custom CSS
        result = manager.get_compiled_css('uncertainty')

        # Should include base and components only
        assert '/* Base Styles */' in result
        assert '/* Components */' in result

    def test_get_compiled_css_raises_error_for_missing_base(self, temp_incomplete_dir):
        """Test get_compiled_css raises error when base CSS missing"""
        # Remove base CSS
        (Path(temp_incomplete_dir) / 'base_styles.css').unlink()

        manager = CSSManager(temp_incomplete_dir)

        with pytest.raises(FileNotFoundError, match='Required CSS file not found'):
            manager.get_compiled_css('robustness')

    def test_get_compiled_css_raises_error_for_missing_components(self, temp_incomplete_dir):
        """Test get_compiled_css raises error when components CSS missing"""
        manager = CSSManager(temp_incomplete_dir)

        with pytest.raises(FileNotFoundError, match='Required CSS file not found'):
            manager.get_compiled_css('robustness')

    def test_get_compiled_css_includes_timestamp(self, temp_templates_dir):
        """Test get_compiled_css includes timestamp in header"""
        manager = CSSManager(temp_templates_dir)

        result = manager.get_compiled_css('robustness')

        assert 'Generated:' in result

    def test_get_compiled_css_includes_report_type(self, temp_templates_dir):
        """Test get_compiled_css includes report type in header"""
        manager = CSSManager(temp_templates_dir)

        result = manager.get_compiled_css('robustness')

        assert 'Robustness' in result


class TestGetCustomCSSPath:
    """Tests for _get_custom_css_path method"""

    def test_get_custom_css_path_finds_direct_path(self, temp_templates_dir):
        """Test _get_custom_css_path finds CSS in interactive folder"""
        manager = CSSManager(temp_templates_dir)

        path = manager._get_custom_css_path('robustness')

        assert path.exists()
        assert path.name == 'robustness_custom.css'

    def test_get_custom_css_path_finds_css_subfolder(self, temp_templates_dir):
        """Test _get_custom_css_path finds CSS in css subfolder"""
        manager = CSSManager(temp_templates_dir)

        path = manager._get_custom_css_path('resilience')

        assert path.exists()
        assert path.name == 'resilience_custom.css'

    def test_get_custom_css_path_returns_default_for_missing(self, temp_templates_dir):
        """Test _get_custom_css_path returns default path when not found"""
        manager = CSSManager(temp_templates_dir)

        path = manager._get_custom_css_path('nonexistent')

        # Should return path1 as default
        assert 'nonexistent_custom.css' in str(path)
        assert not path.exists()


class TestReadCSSFile:
    """Tests for _read_css_file method"""

    def test_read_css_file_reads_existing_file(self, temp_templates_dir):
        """Test _read_css_file reads existing CSS file"""
        manager = CSSManager(temp_templates_dir)

        content = manager._read_css_file(manager.base_css_path, required=False)

        assert '/* Base Styles */' in content

    def test_read_css_file_raises_error_for_missing_required(self, temp_templates_dir):
        """Test _read_css_file raises error for missing required file"""
        manager = CSSManager(temp_templates_dir)
        missing_path = Path(temp_templates_dir) / 'nonexistent.css'

        with pytest.raises(FileNotFoundError):
            manager._read_css_file(missing_path, required=True)

    def test_read_css_file_returns_empty_for_missing_optional(self, temp_templates_dir):
        """Test _read_css_file returns empty string for missing optional file"""
        manager = CSSManager(temp_templates_dir)
        missing_path = Path(temp_templates_dir) / 'nonexistent.css'

        content = manager._read_css_file(missing_path, required=False)

        assert content == ''

    def test_read_css_file_handles_read_error(self, temp_templates_dir):
        """Test _read_css_file handles read errors gracefully"""
        manager = CSSManager(temp_templates_dir)

        # Create a file with invalid permissions (if possible)
        bad_file = Path(temp_templates_dir) / 'bad.css'
        bad_file.write_text('test')
        bad_file.chmod(0o000)  # No read permissions

        try:
            content = manager._read_css_file(bad_file, required=False)
            assert content == ''
        finally:
            bad_file.chmod(0o644)  # Restore permissions for cleanup


class TestCompileLayers:
    """Tests for _compile_layers method"""

    def test_compile_layers_combines_all_layers(self, temp_templates_dir):
        """Test _compile_layers combines all CSS layers"""
        manager = CSSManager(temp_templates_dir)

        result = manager._compile_layers(
            'base css',
            'components css',
            'custom css',
            'robustness'
        )

        assert 'base css' in result
        assert 'components css' in result
        assert 'custom css' in result

    def test_compile_layers_includes_separators(self, temp_templates_dir):
        """Test _compile_layers includes layer separators"""
        manager = CSSManager(temp_templates_dir)

        result = manager._compile_layers(
            'base',
            'components',
            'custom',
            'test'
        )

        assert 'LAYER 2: SHARED COMPONENTS' in result
        assert 'LAYER 3: TEST CUSTOM STYLES' in result

    def test_compile_layers_skips_empty_custom(self, temp_templates_dir):
        """Test _compile_layers skips custom layer if empty"""
        manager = CSSManager(temp_templates_dir)

        result = manager._compile_layers(
            'base',
            'components',
            '',
            'test'
        )

        assert 'LAYER 3' not in result


class TestValidateCSSFiles:
    """Tests for validate_css_files method"""

    def test_validate_css_files_returns_true_when_all_exist(self, temp_templates_dir):
        """Test validate_css_files returns true when all files exist"""
        manager = CSSManager(temp_templates_dir)

        validation = manager.validate_css_files()

        assert validation['base_styles'] is True
        assert validation['components'] is True
        assert len(validation['errors']) == 0

    def test_validate_css_files_returns_false_for_missing(self, temp_incomplete_dir):
        """Test validate_css_files returns false for missing files"""
        manager = CSSManager(temp_incomplete_dir)

        validation = manager.validate_css_files()

        assert validation['base_styles'] is True
        assert validation['components'] is False
        assert len(validation['errors']) > 0

    def test_validate_css_files_includes_error_messages(self, temp_incomplete_dir):
        """Test validate_css_files includes error messages"""
        manager = CSSManager(temp_incomplete_dir)

        validation = manager.validate_css_files()

        assert 'Components not found' in validation['errors'][0]


class TestGetCustomCSSInfo:
    """Tests for get_custom_css_info method"""

    def test_get_custom_css_info_returns_info_when_exists(self, temp_templates_dir):
        """Test get_custom_css_info returns info when CSS exists"""
        manager = CSSManager(temp_templates_dir)

        info = manager.get_custom_css_info('robustness')

        assert info['exists'] is True
        assert 'robustness_custom.css' in info['path']
        assert info['size'] > 0

    def test_get_custom_css_info_returns_info_when_not_exists(self, temp_templates_dir):
        """Test get_custom_css_info returns info when CSS doesn't exist"""
        manager = CSSManager(temp_templates_dir)

        info = manager.get_custom_css_info('nonexistent')

        assert info['exists'] is False
        assert info['size'] == 0


class TestConvenienceFunction:
    """Tests for compile_report_css convenience function"""

    def test_compile_report_css_works(self, temp_templates_dir):
        """Test compile_report_css convenience function"""
        result = compile_report_css('robustness', temp_templates_dir)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_compile_report_css_with_none_templates_dir(self):
        """Test compile_report_css with None templates_dir"""
        # Should use default templates dir (may fail if not set up)
        try:
            result = compile_report_css('robustness', None)
            assert isinstance(result, str)
        except FileNotFoundError:
            # Expected if default templates don't exist
            pass
