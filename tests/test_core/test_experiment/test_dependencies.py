"""
Comprehensive tests for the dependencies module.
"""
import importlib
import sys
import warnings
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import dependencies module directly to avoid initialization issues
import deepbridge.core.experiment.dependencies as deps_module

DEPENDENCIES = deps_module.DEPENDENCIES
MIN_VERSIONS = deps_module.MIN_VERSIONS
DependencyError = deps_module.DependencyError
IncompatibleVersionError = deps_module.IncompatibleVersionError
MissingDependencyError = deps_module.MissingDependencyError
check_component_dependencies = deps_module.check_component_dependencies
check_dependencies = deps_module.check_dependencies
check_version_compatibility = deps_module.check_version_compatibility
fallback_dependencies = deps_module.fallback_dependencies
get_install_command = deps_module.get_install_command
get_package_version = deps_module.get_package_version
is_package_installed = deps_module.is_package_installed
optional_import = deps_module.optional_import
print_dependency_status = deps_module.print_dependency_status
require_package = deps_module.require_package
try_import = deps_module.try_import


class TestGetPackageVersion:
    """Tests for get_package_version function"""

    def test_get_version_installed_package(self):
        """Test getting version of an installed package"""
        version = get_package_version('numpy')
        assert version is not None
        assert isinstance(version, str)
        assert len(version) > 0

    def test_get_version_not_installed(self):
        """Test getting version of a package that is not installed"""
        version = get_package_version('nonexistent_package_xyz123')
        assert version is None

    @patch('deepbridge.core.experiment.dependencies.importlib.metadata.version')
    def test_get_version_exception_handling(self, mock_version):
        """Test exception handling in get_package_version"""
        mock_version.side_effect = Exception('Test error')
        # Clear cache first
        deps_module.get_package_version.cache_clear()
        version = get_package_version('test_package')
        assert version is None

    def test_get_version_caching(self):
        """Test that results are cached"""
        # Clear cache
        deps_module.get_package_version.cache_clear()

        # First call
        v1 = get_package_version('numpy')
        # Second call should use cache
        v2 = get_package_version('numpy')
        assert v1 == v2


class TestIsPackageInstalled:
    """Tests for is_package_installed function"""

    def test_package_installed(self):
        """Test checking an installed package"""
        assert is_package_installed('numpy') is True
        assert is_package_installed('pandas') is True

    def test_package_not_installed(self):
        """Test checking a non-installed package"""
        assert is_package_installed('nonexistent_xyz123') is False

    def test_package_installed_caching(self):
        """Test that results are cached"""
        deps_module.is_package_installed.cache_clear()

        result1 = is_package_installed('numpy')
        result2 = is_package_installed('numpy')
        assert result1 == result2


class TestCheckVersionCompatibility:
    """Tests for check_version_compatibility function"""

    def test_compatible_version(self):
        """Test checking a compatible version"""
        is_compatible, installed, required = check_version_compatibility('numpy')
        assert installed is not None
        if required:
            assert is_compatible in (True, False)

    def test_package_not_installed(self):
        """Test version check for non-installed package"""
        is_compatible, installed, required = check_version_compatibility(
            'nonexistent_xyz123'
        )
        assert is_compatible is False
        assert installed is None

    def test_no_min_version_requirement(self):
        """Test package with no minimum version requirement"""
        is_compatible, installed, required = check_version_compatibility('scipy')
        assert required is None
        if installed:
            assert is_compatible is True

    @patch('deepbridge.core.experiment.dependencies.get_package_version')
    def test_incompatible_version(self, mock_get_version):
        """Test detection of incompatible version"""
        mock_get_version.return_value = '0.1.0'
        is_compatible, installed, required = check_version_compatibility('numpy')
        assert is_compatible is False
        assert installed == '0.1.0'
        assert required == MIN_VERSIONS['numpy']


class TestCheckComponentDependencies:
    """Tests for check_component_dependencies function"""

    def test_check_core_component(self):
        """Test checking core component dependencies"""
        result = check_component_dependencies('core')
        assert 'component' in result
        assert result['component'] == 'core'
        assert 'all_required_installed' in result
        assert 'missing_required' in result
        assert 'missing_optional' in result
        assert 'version_issues' in result

    def test_check_reporting_component(self):
        """Test checking reporting component dependencies"""
        result = check_component_dependencies('reporting')
        assert result['component'] == 'reporting'
        assert isinstance(result['all_required_installed'], bool)

    def test_check_visualization_component(self):
        """Test checking visualization component dependencies"""
        result = check_component_dependencies('visualization')
        assert result['component'] == 'visualization'

    def test_invalid_component(self):
        """Test checking an invalid component raises ValueError"""
        with pytest.raises(ValueError, match='Unknown component'):
            check_component_dependencies('invalid_component')

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    def test_missing_required_package(self, mock_is_installed):
        """Test detection of missing required package"""
        mock_is_installed.return_value = False
        result = check_component_dependencies('core')
        assert result['all_required_installed'] is False
        assert len(result['missing_required']) > 0

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    @patch('deepbridge.core.experiment.dependencies.check_version_compatibility')
    def test_version_issue_detection(self, mock_version_check, mock_is_installed):
        """Test detection of version compatibility issues"""
        mock_is_installed.return_value = True
        mock_version_check.return_value = (False, '0.1.0', '1.0.0')
        result = check_component_dependencies('core')
        # Should have version issues
        assert result['all_required_installed'] is False


class TestCheckDependencies:
    """Tests for check_dependencies function"""

    def test_check_all_dependencies(self):
        """Test checking all dependencies"""
        all_ok, missing_req, missing_opt, issues = check_dependencies()
        assert isinstance(all_ok, bool)
        assert isinstance(missing_req, list)
        assert isinstance(missing_opt, list)
        assert isinstance(issues, list)

    def test_check_specific_components(self):
        """Test checking specific components"""
        all_ok, missing_req, missing_opt, issues = check_dependencies(['core'])
        assert isinstance(all_ok, bool)

    def test_check_multiple_components(self):
        """Test checking multiple components"""
        all_ok, missing_req, missing_opt, issues = check_dependencies(
            ['core', 'reporting']
        )
        assert isinstance(all_ok, bool)

    def test_invalid_component_in_list(self):
        """Test checking with invalid component raises ValueError"""
        with pytest.raises(ValueError, match='Unknown component'):
            check_dependencies(['core', 'invalid_component'])

    @patch('deepbridge.core.experiment.dependencies.check_component_dependencies')
    def test_unique_missing_packages(self, mock_check_component):
        """Test that missing packages are deduplicated"""
        mock_check_component.return_value = {
            'all_required_installed': False,
            'missing_required': ['pkg1', 'pkg2'],
            'missing_optional': ['pkg3'],
            'version_issues': [],
        }
        all_ok, missing_req, missing_opt, issues = check_dependencies(
            ['core', 'reporting']
        )
        # Should be unique
        assert len(missing_req) == len(set(missing_req))
        assert len(missing_opt) == len(set(missing_opt))


class TestGetInstallCommand:
    """Tests for get_install_command function"""

    def test_no_packages(self):
        """Test with no packages"""
        cmd = get_install_command([])
        assert cmd == 'No packages to install'

    def test_single_package_no_version(self):
        """Test with a single package without version requirement"""
        cmd = get_install_command(['scipy'])
        assert 'pip install' in cmd
        assert 'scipy' in cmd

    def test_single_package_with_version(self):
        """Test with a single package with version requirement"""
        cmd = get_install_command(['numpy'])
        assert 'pip install' in cmd
        assert 'numpy>=' in cmd
        assert MIN_VERSIONS['numpy'] in cmd

    def test_multiple_packages(self):
        """Test with multiple packages"""
        cmd = get_install_command(['numpy', 'pandas'])
        assert 'pip install' in cmd
        assert 'numpy>=' in cmd
        assert 'pandas>=' in cmd

    def test_upgrade_flag(self):
        """Test with upgrade flag"""
        cmd = get_install_command(['numpy'], upgrade=True)
        assert '--upgrade' in cmd

    def test_no_upgrade_flag(self):
        """Test without upgrade flag"""
        cmd = get_install_command(['numpy'], upgrade=False)
        assert '--upgrade' not in cmd


class TestTryImport:
    """Tests for try_import function"""

    def test_import_existing_package(self):
        """Test importing an existing package"""
        module = try_import('numpy')
        assert module is not None
        assert hasattr(module, '__version__')

    def test_import_nonexistent_package(self):
        """Test importing a non-existent package"""
        module = try_import('nonexistent_xyz123')
        assert module is None

    def test_import_with_different_name(self):
        """Test importing with different import name"""
        module = try_import('scikit-learn', 'sklearn')
        assert module is not None


class TestRequirePackage:
    """Tests for require_package function"""

    def test_require_installed_package(self):
        """Test requiring an installed package"""
        module = require_package('numpy')
        assert module is not None

    def test_require_missing_package(self):
        """Test requiring a missing package raises error"""
        with pytest.raises(MissingDependencyError):
            require_package('nonexistent_xyz123')

    @patch('deepbridge.core.experiment.dependencies.try_import')
    @patch('deepbridge.core.experiment.dependencies.get_package_version')
    def test_require_incompatible_version(self, mock_get_version, mock_try_import):
        """Test requiring package with incompatible version"""
        mock_try_import.return_value = MagicMock()
        mock_get_version.return_value = '0.1.0'

        with pytest.raises(IncompatibleVersionError):
            require_package('numpy', min_version='10.0.0')

    def test_require_with_different_import_name(self):
        """Test requiring package with different import name"""
        module = require_package('scikit-learn', 'sklearn')
        assert module is not None

    def test_require_uses_min_versions(self):
        """Test that require_package uses MIN_VERSIONS"""
        # This should not raise if numpy meets minimum version
        module = require_package('numpy')
        assert module is not None


class TestOptionalImport:
    """Tests for optional_import function"""

    def test_optional_import_installed(self):
        """Test optional import of installed package"""
        module = optional_import('numpy', warning=False)
        assert module is not None

    def test_optional_import_missing(self):
        """Test optional import of missing package"""
        module = optional_import('nonexistent_xyz123', warning=False)
        assert module is None

    def test_optional_import_with_warning(self):
        """Test optional import with warning enabled"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            module = optional_import('nonexistent_xyz123', warning=True)
            assert module is None
            assert len(w) >= 1
            assert 'Optional package' in str(w[-1].message)

    def test_optional_import_no_warning(self):
        """Test optional import with warning disabled"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            module = optional_import('nonexistent_xyz123', warning=False)
            assert module is None


class TestFallbackDependencies:
    """Tests for fallback_dependencies function"""

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    def test_jinja2_fallback(self, mock_is_installed):
        """Test jinja2 fallback creation"""
        def side_effect(pkg):
            return pkg != 'jinja2'

        mock_is_installed.side_effect = side_effect

        # Remove jinja2 from sys.modules if it exists
        if 'jinja2' in sys.modules:
            old_jinja2 = sys.modules.pop('jinja2')
        else:
            old_jinja2 = None

        try:
            fallbacks = fallback_dependencies()
            assert 'jinja2' in fallbacks
            assert 'jinja2' in sys.modules
        finally:
            # Restore original state
            if old_jinja2:
                sys.modules['jinja2'] = old_jinja2

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    def test_plotly_fallback(self, mock_is_installed):
        """Test plotly fallback creation"""
        def side_effect(pkg):
            return pkg != 'plotly'

        mock_is_installed.side_effect = side_effect

        # Remove plotly from sys.modules if it exists
        plotly_modules = [k for k in sys.modules.keys() if k.startswith('plotly')]
        old_plotly = {}
        for mod in plotly_modules:
            old_plotly[mod] = sys.modules.pop(mod)

        try:
            fallbacks = fallback_dependencies()
            assert 'plotly' in fallbacks
            assert 'plotly' in sys.modules
        finally:
            # Restore original state
            for mod, val in old_plotly.items():
                sys.modules[mod] = val

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    def test_no_fallbacks_needed(self, mock_is_installed):
        """Test when no fallbacks are needed"""
        mock_is_installed.return_value = True
        fallbacks = fallback_dependencies()
        # Should be empty since all packages are "installed"
        assert isinstance(fallbacks, dict)


class TestJinja2Fallback:
    """Tests for Jinja2 fallback implementation"""

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    def test_simple_template_render(self, mock_is_installed):
        """Test SimpleTemplate render method"""
        from deepbridge.core.experiment.dependencies import _create_jinja2_fallback

        fallback = _create_jinja2_fallback()
        template = fallback.Template('Hello {{name}}')
        result = template.render(name='World')
        assert 'World' in result

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    def test_template_with_spaces(self, mock_is_installed):
        """Test template with spaces in placeholders"""
        from deepbridge.core.experiment.dependencies import _create_jinja2_fallback

        fallback = _create_jinja2_fallback()
        template = fallback.Template('Hello {{ name }}')
        result = template.render(name='World')
        assert 'World' in result

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    def test_environment_creation(self, mock_is_installed):
        """Test Environment class creation"""
        from deepbridge.core.experiment.dependencies import _create_jinja2_fallback

        fallback = _create_jinja2_fallback()
        env = fallback.Environment()
        assert env is not None
        assert hasattr(env, 'from_string')
        assert hasattr(env, 'get_template')


class TestPlotlyFallback:
    """Tests for Plotly fallback implementation"""

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    def test_figure_creation(self, mock_is_installed):
        """Test Figure creation"""
        from deepbridge.core.experiment.dependencies import _create_plotly_fallback

        fallback = _create_plotly_fallback()
        fig = fallback.graph_objects.Figure()
        assert fig is not None
        assert hasattr(fig, 'update_layout')
        assert hasattr(fig, 'add_trace')

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    def test_scatter_trace(self, mock_is_installed):
        """Test Scatter trace creation"""
        from deepbridge.core.experiment.dependencies import _create_plotly_fallback

        fallback = _create_plotly_fallback()
        scatter = fallback.graph_objects.Scatter(x=[1, 2], y=[3, 4])
        assert scatter is not None
        assert scatter.x == [1, 2]
        assert scatter.y == [3, 4]

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    def test_express_functions(self, mock_is_installed):
        """Test express module functions"""
        from deepbridge.core.experiment.dependencies import _create_plotly_fallback

        fallback = _create_plotly_fallback()

        fig = fallback.express.line()
        assert fig is not None

        fig = fallback.express.bar()
        assert fig is not None

        fig = fallback.express.scatter()
        assert fig is not None

    @patch('deepbridge.core.experiment.dependencies.is_package_installed')
    def test_figure_to_json(self, mock_is_installed):
        """Test Figure to_json method"""
        from deepbridge.core.experiment.dependencies import _create_plotly_fallback

        fallback = _create_plotly_fallback()
        fig = fallback.graph_objects.Figure()
        json_str = fig.to_json()
        assert isinstance(json_str, str)
        assert 'data' in json_str
        assert 'layout' in json_str


class TestPrintDependencyStatus:
    """Tests for print_dependency_status function"""

    @patch('deepbridge.core.experiment.dependencies.check_dependencies')
    @patch('builtins.print')
    def test_all_installed(self, mock_print, mock_check):
        """Test printing when all dependencies are installed"""
        mock_check.return_value = (True, [], [], [])
        print_dependency_status()
        # Should print success message
        calls = [str(call) for call in mock_print.call_args_list]
        assert any('✅' in str(call) or 'All dependencies' in str(call) for call in calls)

    @patch('deepbridge.core.experiment.dependencies.check_dependencies')
    @patch('builtins.print')
    def test_missing_required(self, mock_print, mock_check):
        """Test printing when required dependencies are missing"""
        mock_check.return_value = (False, ['numpy', 'pandas'], [], [])
        print_dependency_status()
        # Should print error message
        calls = [str(call) for call in mock_print.call_args_list]
        assert any('❌' in str(call) or 'Missing required' in str(call) for call in calls)

    @patch('deepbridge.core.experiment.dependencies.check_dependencies')
    @patch('builtins.print')
    def test_missing_optional(self, mock_print, mock_check):
        """Test printing when optional dependencies are missing"""
        mock_check.return_value = (True, [], ['matplotlib'], [])
        print_dependency_status()
        # Should print warning message
        calls = [str(call) for call in mock_print.call_args_list]
        assert any('⚠️' in str(call) or 'optional' in str(call) for call in calls)

    @patch('deepbridge.core.experiment.dependencies.check_dependencies')
    @patch('builtins.print')
    def test_version_issues(self, mock_print, mock_check):
        """Test printing when there are version issues"""
        mock_check.return_value = (
            True,
            [],
            [],
            [{'package': 'numpy', 'installed': '1.0.0', 'required': '2.0.0'}],
        )
        print_dependency_status()
        # Should print version issue message
        calls = [str(call) for call in mock_print.call_args_list]
        assert any('version' in str(call).lower() for call in calls)


class TestExceptions:
    """Tests for custom exception classes"""

    def test_dependency_error(self):
        """Test DependencyError exception"""
        with pytest.raises(DependencyError):
            raise DependencyError('Test error')

    def test_incompatible_version_error(self):
        """Test IncompatibleVersionError exception"""
        with pytest.raises(IncompatibleVersionError):
            raise IncompatibleVersionError('Test error')

    def test_missing_dependency_error(self):
        """Test MissingDependencyError exception"""
        with pytest.raises(MissingDependencyError):
            raise MissingDependencyError('Test error')

    def test_error_inheritance(self):
        """Test that custom errors inherit from DependencyError"""
        assert issubclass(IncompatibleVersionError, DependencyError)
        assert issubclass(MissingDependencyError, DependencyError)


class TestConstants:
    """Tests for module constants"""

    def test_dependencies_structure(self):
        """Test DEPENDENCIES constant structure"""
        assert isinstance(DEPENDENCIES, dict)
        for component, deps in DEPENDENCIES.items():
            assert 'required' in deps
            assert 'optional' in deps
            assert isinstance(deps['required'], dict)
            assert isinstance(deps['optional'], dict)

    def test_min_versions_structure(self):
        """Test MIN_VERSIONS constant structure"""
        assert isinstance(MIN_VERSIONS, dict)
        for package, version in MIN_VERSIONS.items():
            assert isinstance(package, str)
            assert isinstance(version, str)
            assert '.' in version  # Version should have dots

    def test_core_dependencies_present(self):
        """Test that core dependencies are defined"""
        assert 'core' in DEPENDENCIES
        assert 'pandas' in DEPENDENCIES['core']['required']
        assert 'numpy' in DEPENDENCIES['core']['required']
        assert 'scikit-learn' in DEPENDENCIES['core']['required']
