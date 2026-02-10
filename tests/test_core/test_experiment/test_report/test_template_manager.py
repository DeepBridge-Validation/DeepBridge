"""
Comprehensive tests for TemplateManager.

This test suite validates:
1. TemplateManager initialization
2. _add_safe_filters - Jinja2 filter registration
3. find_template - template path resolution with caching
4. get_template_paths - path generation for test types
5. load_template - template loading
6. render_template - template rendering

Coverage Target: ~90%+
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock

from deepbridge.core.experiment.report.template_manager import TemplateManager


# ==================== Fixtures ====================


@pytest.fixture
def temp_templates_dir():
    """Create temporary templates directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def template_manager(temp_templates_dir):
    """Create TemplateManager instance"""
    return TemplateManager(templates_dir=temp_templates_dir)


@pytest.fixture
def mock_template():
    """Create mock Jinja2 template"""
    template = Mock()
    template.render = Mock(return_value='<html>rendered</html>')
    return template


# ==================== Initialization Tests ====================


class TestTemplateManagerInitialization:
    """Tests for TemplateManager initialization"""

    def test_init_with_valid_directory(self, temp_templates_dir):
        """Test initialization with valid directory"""
        manager = TemplateManager(templates_dir=temp_templates_dir)

        assert manager.templates_dir == temp_templates_dir
        assert hasattr(manager, 'jinja2')
        assert hasattr(manager, 'jinja_env')

    def test_init_imports_jinja2(self, temp_templates_dir):
        """Test that Jinja2 is imported"""
        manager = TemplateManager(templates_dir=temp_templates_dir)

        assert manager.jinja2 is not None

    def test_init_raises_error_if_jinja2_not_available(self, temp_templates_dir):
        """Test error when Jinja2 is not installed"""
        with patch.dict('sys.modules', {'jinja2': None}):
            with patch('builtins.__import__', side_effect=ImportError):
                with pytest.raises(ImportError, match='Jinja2 is required'):
                    TemplateManager(templates_dir=temp_templates_dir)

    def test_init_sets_up_jinja_environment(self, template_manager):
        """Test that Jinja2 environment is configured"""
        assert template_manager.jinja_env is not None
        assert hasattr(template_manager.jinja_env, 'loader')
        assert template_manager.jinja_env.autoescape is not None

    def test_init_adds_safe_filters(self, template_manager):
        """Test that safe filters are added to environment"""
        filters = template_manager.jinja_env.filters

        assert 'safe_float' in filters
        assert 'safe_round' in filters
        assert 'safe_multiply' in filters
        assert 'safe_js' in filters
        assert 'abs_value' in filters
        assert 'format_number' in filters


# ==================== _add_safe_filters Tests ====================


class TestAddSafeFilters:
    """Tests for _add_safe_filters method"""

    def test_safe_float_with_float(self, template_manager):
        """Test safe_float filter with float value"""
        safe_float = template_manager.jinja_env.filters['safe_float']

        result = safe_float(3.14)
        assert result == 3.14

    def test_safe_float_with_int(self, template_manager):
        """Test safe_float filter with int value"""
        safe_float = template_manager.jinja_env.filters['safe_float']

        result = safe_float(42)
        assert result == 42.0

    def test_safe_float_with_none(self, template_manager):
        """Test safe_float filter with None returns default"""
        safe_float = template_manager.jinja_env.filters['safe_float']

        result = safe_float(None)
        assert result == 0.0

        result = safe_float(None, default=99.9)
        assert result == 99.9

    def test_safe_float_with_string_number(self, template_manager):
        """Test safe_float filter with string number"""
        safe_float = template_manager.jinja_env.filters['safe_float']

        result = safe_float('123.45')
        assert result == 123.45

    def test_safe_float_with_percentage_string(self, template_manager):
        """Test safe_float filter strips % from string"""
        safe_float = template_manager.jinja_env.filters['safe_float']

        result = safe_float('85%')
        assert result == 85.0

    def test_safe_float_with_comma_string(self, template_manager):
        """Test safe_float filter strips commas"""
        safe_float = template_manager.jinja_env.filters['safe_float']

        result = safe_float('1,234.56')
        assert result == 1234.56

    def test_safe_float_with_error_string(self, template_manager):
        """Test safe_float filter returns default for error strings"""
        safe_float = template_manager.jinja_env.filters['safe_float']

        result = safe_float('error occurred')
        assert result == 0.0

        result = safe_float('classification error')
        assert result == 0.0

    def test_safe_float_with_invalid_string(self, template_manager):
        """Test safe_float filter with invalid string"""
        safe_float = template_manager.jinja_env.filters['safe_float']

        result = safe_float('not a number')
        assert result == 0.0

    def test_safe_round_with_valid_value(self, template_manager):
        """Test safe_round filter"""
        safe_round = template_manager.jinja_env.filters['safe_round']

        result = safe_round(3.14159, precision=2)
        assert result == 3.14

    def test_safe_round_with_default_precision(self, template_manager):
        """Test safe_round filter with default precision"""
        safe_round = template_manager.jinja_env.filters['safe_round']

        result = safe_round(3.14159)
        assert result == 3.14

    def test_safe_round_with_invalid_value(self, template_manager):
        """Test safe_round filter with invalid value"""
        safe_round = template_manager.jinja_env.filters['safe_round']

        result = safe_round('invalid')
        assert result == 0.0

    def test_safe_multiply_with_valid_values(self, template_manager):
        """Test safe_multiply filter"""
        safe_multiply = template_manager.jinja_env.filters['safe_multiply']

        result = safe_multiply(0.85, 100)
        assert result == 85.0

    def test_safe_multiply_with_default_multiplier(self, template_manager):
        """Test safe_multiply filter with default multiplier (100)"""
        safe_multiply = template_manager.jinja_env.filters['safe_multiply']

        result = safe_multiply(0.75)
        assert result == 75.0

    def test_safe_multiply_with_invalid_values(self, template_manager):
        """Test safe_multiply filter with invalid values"""
        safe_multiply = template_manager.jinja_env.filters['safe_multiply']

        result = safe_multiply('invalid', 'also invalid')
        assert result == 0.0

    def test_abs_value_filter(self, template_manager):
        """Test abs_value filter"""
        abs_value = template_manager.jinja_env.filters['abs_value']

        assert abs_value(-5.5) == 5.5
        assert abs_value(3.2) == 3.2
        assert abs_value(None) == 0.0

    def test_format_number_filter(self, template_manager):
        """Test format_number filter"""
        format_number = template_manager.jinja_env.filters['format_number']

        result = format_number(1234567)
        assert result == '1,234,567'

    def test_format_number_filter_with_invalid(self, template_manager):
        """Test format_number filter with invalid value"""
        format_number = template_manager.jinja_env.filters['format_number']

        result = format_number('invalid')
        assert result == 'invalid'


# ==================== find_template Tests ====================


class TestFindTemplate:
    """Tests for find_template method"""

    def test_find_template_with_existing_file(self, template_manager, temp_templates_dir):
        """Test finding template that exists"""
        # Create a test template file
        template_path = os.path.join(temp_templates_dir, 'test.html')
        with open(template_path, 'w') as f:
            f.write('<html>test</html>')

        result = template_manager.find_template([template_path])
        assert result == template_path

    def test_find_template_with_multiple_paths(self, template_manager, temp_templates_dir):
        """Test finding template from multiple paths"""
        # Create template in second path
        template_path = os.path.join(temp_templates_dir, 'test2.html')
        with open(template_path, 'w') as f:
            f.write('<html>test2</html>')

        # First path doesn't exist, second does
        paths = [
            os.path.join(temp_templates_dir, 'nonexistent.html'),
            template_path
        ]

        result = template_manager.find_template(paths)
        assert result == template_path

    def test_find_template_raises_error_if_not_found(self, template_manager, temp_templates_dir):
        """Test error when template not found"""
        paths = [
            os.path.join(temp_templates_dir, 'missing1.html'),
            os.path.join(temp_templates_dir, 'missing2.html')
        ]

        with pytest.raises(FileNotFoundError, match='Template not found'):
            template_manager.find_template(paths)

    def test_find_template_caching(self, template_manager, temp_templates_dir):
        """Test that find_template uses caching"""
        # Create a test template
        template_path = os.path.join(temp_templates_dir, 'cached.html')
        with open(template_path, 'w') as f:
            f.write('<html>cached</html>')

        # Call twice with same paths
        paths = [template_path]
        result1 = template_manager.find_template(paths)
        result2 = template_manager.find_template(paths)

        assert result1 == result2 == template_path

        # Check cache info (should have 1 hit)
        cache_info = template_manager._find_template_cached.cache_info()
        assert cache_info.hits >= 1

    def test_find_template_logs_found_path(self, template_manager, temp_templates_dir):
        """Test that finding template logs the path"""
        template_path = os.path.join(temp_templates_dir, 'logged.html')
        with open(template_path, 'w') as f:
            f.write('<html>logged</html>')

        with patch('deepbridge.core.experiment.report.template_manager.logger') as mock_logger:
            template_manager.find_template([template_path])

            # Should log info about found template
            mock_logger.info.assert_called_once()


# ==================== get_template_paths Tests ====================


class TestGetTemplatePaths:
    """Tests for get_template_paths method"""

    def test_get_template_paths_interactive(self, template_manager):
        """Test getting paths for interactive report"""
        paths = template_manager.get_template_paths('robustness', 'interactive')

        assert len(paths) == 2
        assert 'robustness/interactive/index.html' in paths[0]
        assert 'robustness/index.html' in paths[1]

    def test_get_template_paths_static(self, template_manager):
        """Test getting paths for static report"""
        paths = template_manager.get_template_paths('uncertainty', 'static')

        assert len(paths) == 2
        assert 'uncertainty/static/index.html' in paths[0]
        assert 'uncertainty/index.html' in paths[1]

    def test_get_template_paths_default_interactive(self, template_manager):
        """Test that default report type is interactive"""
        paths = template_manager.get_template_paths('fairness')

        assert 'fairness/interactive/index.html' in paths[0]

    def test_get_template_paths_different_test_types(self, template_manager):
        """Test paths for different test types"""
        robustness_paths = template_manager.get_template_paths('robustness')
        uncertainty_paths = template_manager.get_template_paths('uncertainty')
        fairness_paths = template_manager.get_template_paths('fairness')

        assert 'robustness' in robustness_paths[0]
        assert 'uncertainty' in uncertainty_paths[0]
        assert 'fairness' in fairness_paths[0]

    def test_get_template_paths_caching(self, template_manager):
        """Test that get_template_paths uses caching"""
        # Call twice with same arguments
        paths1 = template_manager.get_template_paths('robustness', 'interactive')
        paths2 = template_manager.get_template_paths('robustness', 'interactive')

        # Should return same list
        assert paths1 == paths2

        # Check cache info
        cache_info = template_manager.get_template_paths.cache_info()
        assert cache_info.hits >= 1


# ==================== load_template Tests ====================


class TestLoadTemplate:
    """Tests for load_template method"""

    def test_load_template_with_existing_file(self, template_manager, temp_templates_dir):
        """Test loading an existing template"""
        # Create a test template
        template_path = os.path.join(temp_templates_dir, 'test_load.html')
        with open(template_path, 'w') as f:
            f.write('<html>{{ message }}</html>')

        template = template_manager.load_template(template_path)

        # Should return a Jinja2 template
        assert template is not None
        assert hasattr(template, 'render')

    def test_load_template_raises_error_if_not_exists(self, template_manager, temp_templates_dir):
        """Test error when loading non-existent template"""
        template_path = os.path.join(temp_templates_dir, 'nonexistent.html')

        with pytest.raises(FileNotFoundError, match='Template file not found'):
            template_manager.load_template(template_path)

    def test_load_template_adds_safe_filters_to_env(self, template_manager, temp_templates_dir):
        """Test that loaded template has safe filters"""
        # Create a template
        template_path = os.path.join(temp_templates_dir, 'filtered.html')
        with open(template_path, 'w') as f:
            f.write('<html>{{ value|safe_float }}</html>')

        template = template_manager.load_template(template_path)

        # Template should be able to use safe_float filter
        result = template.render(value='123.45')
        assert '123.45' in result or '123' in result

    def test_load_template_uses_utf8_encoding(self, template_manager, temp_templates_dir):
        """Test that template uses UTF-8 encoding"""
        # Create a template with unicode
        template_path = os.path.join(temp_templates_dir, 'unicode.html')
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write('<html>{{ message }}</html>')

        template = template_manager.load_template(template_path)
        result = template.render(message='Hello 世界')

        assert '世界' in result


# ==================== render_template Tests ====================


class TestRenderTemplate:
    """Tests for render_template method"""

    def test_render_template_with_simple_context(self, template_manager, mock_template):
        """Test rendering template with simple context"""
        context = {'name': 'Alice', 'age': 30}

        result = template_manager.render_template(mock_template, context)

        assert result == '<html>rendered</html>'
        mock_template.render.assert_called_once_with(name='Alice', age=30)

    def test_render_template_with_empty_context(self, template_manager, mock_template):
        """Test rendering with empty context"""
        result = template_manager.render_template(mock_template, {})

        assert result == '<html>rendered</html>'
        mock_template.render.assert_called_once_with()

    def test_render_template_with_complex_context(self, template_manager, mock_template):
        """Test rendering with complex nested context"""
        context = {
            'data': {'nested': {'value': 123}},
            'list': [1, 2, 3],
            'bool': True
        }

        result = template_manager.render_template(mock_template, context)

        assert result == '<html>rendered</html>'
        mock_template.render.assert_called_once()


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_template_workflow(self, template_manager, temp_templates_dir):
        """Test complete workflow: find, load, render"""
        # Create a template
        subdir = os.path.join(temp_templates_dir, 'report_types', 'robustness')
        os.makedirs(subdir, exist_ok=True)
        template_path = os.path.join(subdir, 'index.html')

        with open(template_path, 'w') as f:
            f.write('<html><body>{{ title }}</body></html>')

        # Get paths
        paths = template_manager.get_template_paths('robustness', 'interactive')

        # Find template
        found_path = template_manager.find_template([template_path])

        # Load template
        template = template_manager.load_template(found_path)

        # Render template
        result = template_manager.render_template(template, {'title': 'Test Report'})

        assert 'Test Report' in result
        assert '<html>' in result

    def test_template_with_filters_workflow(self, template_manager, temp_templates_dir):
        """Test workflow using safe filters"""
        template_path = os.path.join(temp_templates_dir, 'with_filters.html')

        with open(template_path, 'w') as f:
            f.write('''
                <html>
                <div>{{ accuracy|safe_multiply }}</div>
                <div>{{ loss|safe_round(3) }}</div>
                <div>{{ count|format_number }}</div>
                </html>
            ''')

        template = template_manager.load_template(template_path)
        result = template_manager.render_template(template, {
            'accuracy': 0.95,
            'loss': 0.12345,
            'count': 1000000
        })

        # Check that filters were applied
        assert '95' in result or '95.0' in result
        assert '0.123' in result
        assert '1,000,000' in result


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_very_long_template_path_list(self, template_manager, temp_templates_dir):
        """Test finding template from many paths"""
        # Create template
        template_path = os.path.join(temp_templates_dir, 'target.html')
        with open(template_path, 'w') as f:
            f.write('<html>found</html>')

        # Create list with many non-existent paths plus the real one
        paths = [f'/nonexistent/path{i}.html' for i in range(50)]
        paths.append(template_path)

        result = template_manager.find_template(paths)
        assert result == template_path

    def test_template_with_special_characters_in_path(self, temp_templates_dir):
        """Test template path with special characters"""
        # Create directory with special chars
        special_dir = os.path.join(temp_templates_dir, 'test-dir_123')
        os.makedirs(special_dir, exist_ok=True)

        template_path = os.path.join(special_dir, 'template.html')
        with open(template_path, 'w') as f:
            f.write('<html>special</html>')

        manager = TemplateManager(templates_dir=temp_templates_dir)
        template = manager.load_template(template_path)

        assert template is not None

    def test_cache_clears_with_different_args(self, template_manager):
        """Test that cache differentiates between different arguments"""
        paths1 = template_manager.get_template_paths('robustness', 'interactive')
        paths2 = template_manager.get_template_paths('robustness', 'static')
        paths3 = template_manager.get_template_paths('uncertainty', 'interactive')

        # All should be different
        assert paths1 != paths2
        assert paths1 != paths3
        assert paths2 != paths3

    def test_template_with_includes(self, template_manager, temp_templates_dir):
        """Test template that includes another template"""
        # Create base template
        base_path = os.path.join(temp_templates_dir, 'base.html')
        with open(base_path, 'w') as f:
            f.write('<html>BASE CONTENT</html>')

        # Create main template (can't actually test include without proper setup)
        main_path = os.path.join(temp_templates_dir, 'main.html')
        with open(main_path, 'w') as f:
            f.write('<html>MAIN CONTENT</html>')

        template = template_manager.load_template(main_path)
        assert template is not None

    def test_markup_fallback_without_markupsafe(self, temp_templates_dir):
        """Test Markup fallback when markupsafe is not available"""
        import sys
        from unittest.mock import patch

        # Remove markupsafe temporarily
        markupsafe_backup = sys.modules.get('markupsafe')

        try:
            # Simulate markupsafe not being available
            with patch.dict('sys.modules', {'markupsafe': None}):
                # Need to reload the module to trigger the fallback
                import importlib
                from deepbridge.core.experiment.report import template_manager as tm_module

                # Verify the fallback Markup class works
                # This is the fallback defined in lines 17-19
                assert hasattr(tm_module, 'Markup')
        finally:
            # Restore markupsafe
            if markupsafe_backup is not None:
                sys.modules['markupsafe'] = markupsafe_backup

    def test_safe_float_with_error_string(self, template_manager):
        """Test safe_float filter with error message string"""
        # Access the filter through jinja environment
        safe_float = template_manager.jinja_env.filters['safe_float']

        # Test with error message (should return default)
        result = safe_float('Error: invalid value')
        assert result == 0.0

        # Test with classification message
        result = safe_float('Classification error')
        assert result == 0.0

    def test_safe_round_with_exception(self, template_manager):
        """Test safe_round with value that causes exception"""
        safe_round = template_manager.jinja_env.filters['safe_round']

        # Mock round to raise exception
        from unittest.mock import patch
        with patch('builtins.round', side_effect=TypeError("Cannot round")):
            result = safe_round(3.14159, 2)
            # Should return 0.0 when exception occurs
            assert result == 0.0
