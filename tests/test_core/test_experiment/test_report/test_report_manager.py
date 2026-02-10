"""
Comprehensive tests for ReportManager.

This test suite validates:
1. Initialization with default and custom templates directory
2. generate_report - report generation for different test types
3. Renderer selection (interactive vs static)
4. Error handling and validation
5. Edge cases

Coverage Target: ~90%+
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
import os
import tempfile

from deepbridge.core.experiment.report.report_manager import ReportManager


# ==================== Fixtures ====================


@pytest.fixture
def temp_templates_dir():
    """Create temporary templates directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_template_manager():
    """Create mock TemplateManager"""
    return Mock()


@pytest.fixture
def mock_asset_manager():
    """Create mock AssetManager"""
    return Mock()


@pytest.fixture
def mock_renderer():
    """Create mock renderer"""
    renderer = Mock()
    renderer.render = Mock(return_value='/path/to/report.html')
    return renderer


# ==================== Initialization Tests ====================


class TestInitialization:
    """Tests for ReportManager initialization"""

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_init_with_default_templates_dir(self, mock_exists, mock_template_mgr, mock_asset_mgr):
        """Test initialization with default templates directory"""
        manager = ReportManager()

        assert manager.templates_dir is not None
        assert 'templates' in manager.templates_dir
        mock_template_mgr.assert_called_once()
        mock_asset_mgr.assert_called_once()

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_init_with_custom_templates_dir(self, mock_exists, mock_template_mgr, mock_asset_mgr, temp_templates_dir):
        """Test initialization with custom templates directory"""
        manager = ReportManager(templates_dir=temp_templates_dir)

        assert manager.templates_dir == temp_templates_dir
        mock_template_mgr.assert_called_once_with(temp_templates_dir)
        mock_asset_mgr.assert_called_once_with(temp_templates_dir)

    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=False)
    def test_init_raises_error_if_templates_dir_not_found(self, mock_exists):
        """Test that initialization raises error if templates directory doesn't exist"""
        with pytest.raises(FileNotFoundError, match='Templates directory not found'):
            ReportManager(templates_dir='/nonexistent/path')

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_init_creates_renderers_dict(self, mock_exists, mock_template_mgr, mock_asset_mgr):
        """Test that renderers dictionary is created"""
        manager = ReportManager()

        assert 'robustness' in manager.renderers
        assert 'uncertainty' in manager.renderers
        assert 'resilience' in manager.renderers
        assert 'hyperparameter' in manager.renderers
        assert 'fairness' in manager.renderers

    @patch('deepbridge.core.experiment.report.renderers.static.StaticResilienceRenderer', side_effect=ImportError)
    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    @patch('deepbridge.core.experiment.report.report_manager.logger')
    def test_init_handles_missing_static_renderers(self, mock_logger, mock_exists, mock_template_mgr, mock_asset_mgr, mock_static):
        """Test initialization when static renderers are not available"""
        # Simulate missing static renderers module
        import sys
        # Remove the module if it exists
        sys.modules['deepbridge.core.experiment.report.renderers.static'] = None

        manager = ReportManager()

        assert manager.has_static_renderers is False
        assert manager.static_renderers == {}
        mock_logger.warning.assert_called()


# ==================== generate_report Tests - Interactive ====================


class TestGenerateReportInteractive:
    """Tests for generate_report with interactive mode"""

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_robustness_report(self, mock_exists, mock_template_mgr, mock_asset_mgr, mock_renderer):
        """Test generating robustness report"""
        manager = ReportManager()
        manager.renderers['robustness'] = mock_renderer

        results = {'score': 0.85}
        report_path = manager.generate_report('robustness', results, '/path/output.html')

        mock_renderer.render.assert_called_once_with(
            results, '/path/output.html', 'Model', 'interactive', False
        )
        assert report_path == '/path/to/report.html'

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_uncertainty_report(self, mock_exists, mock_template_mgr, mock_asset_mgr, mock_renderer):
        """Test generating uncertainty report"""
        manager = ReportManager()
        manager.renderers['uncertainty'] = mock_renderer

        results = {'coverage': 0.90}
        report_path = manager.generate_report('uncertainty', results, '/path/output.html')

        mock_renderer.render.assert_called_once()
        assert report_path == '/path/to/report.html'

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_with_custom_model_name(self, mock_exists, mock_template_mgr, mock_asset_mgr, mock_renderer):
        """Test generating report with custom model name"""
        manager = ReportManager()
        manager.renderers['robustness'] = mock_renderer

        results = {'score': 0.85}
        manager.generate_report('robustness', results, '/path/output.html', model_name='CustomModel')

        call_args = mock_renderer.render.call_args
        assert call_args[0][2] == 'CustomModel'

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_with_save_chart_enabled(self, mock_exists, mock_template_mgr, mock_asset_mgr, mock_renderer):
        """Test generating report with save_chart enabled"""
        manager = ReportManager()
        manager.renderers['uncertainty'] = mock_renderer

        results = {'coverage': 0.90}
        manager.generate_report('uncertainty', results, '/path/output.html', save_chart=True)

        call_args = mock_renderer.render.call_args
        assert call_args[0][4] is True  # save_chart parameter

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_case_insensitive_test_type(self, mock_exists, mock_template_mgr, mock_asset_mgr, mock_renderer):
        """Test that test_type is case-insensitive"""
        manager = ReportManager()
        manager.renderers['robustness'] = mock_renderer

        results = {'score': 0.85}
        manager.generate_report('ROBUSTNESS', results, '/path/output.html')

        mock_renderer.render.assert_called_once()

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_hyperparameters_plural(self, mock_exists, mock_template_mgr, mock_asset_mgr, mock_renderer):
        """Test generating report with 'hyperparameters' (plural)"""
        manager = ReportManager()
        manager.renderers['hyperparameters'] = mock_renderer

        results = {'importance': {}}
        report_path = manager.generate_report('hyperparameters', results, '/path/output.html')

        mock_renderer.render.assert_called_once()
        assert report_path == '/path/to/report.html'


# ==================== generate_report Tests - Static ====================


class TestGenerateReportStatic:
    """Tests for generate_report with static mode"""

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_static_report_when_available(self, mock_exists, mock_template_mgr, mock_asset_mgr, mock_renderer):
        """Test generating static report when static renderers are available"""
        manager = ReportManager()
        manager.has_static_renderers = True
        manager.static_renderers['robustness'] = mock_renderer

        results = {'score': 0.85}
        report_path = manager.generate_report(
            'robustness', results, '/path/output.html', report_type='static'
        )

        mock_renderer.render.assert_called_once()
        call_args = mock_renderer.render.call_args
        assert call_args[0][3] == 'static'  # report_type parameter
        assert report_path == '/path/to/report.html'

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    @patch('deepbridge.core.experiment.report.report_manager.logger')
    def test_generate_falls_back_to_interactive_if_static_not_available(
        self, mock_logger, mock_exists, mock_template_mgr, mock_asset_mgr, mock_renderer
    ):
        """Test fallback to interactive when static renderer not available for test type"""
        manager = ReportManager()
        manager.has_static_renderers = True
        manager.static_renderers = {}  # No static renderers for this test type
        manager.renderers['robustness'] = mock_renderer

        results = {'score': 0.85}
        report_path = manager.generate_report(
            'robustness', results, '/path/output.html', report_type='static'
        )

        mock_renderer.render.assert_called_once()
        mock_logger.warning.assert_called()
        assert report_path == '/path/to/report.html'

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    @patch('deepbridge.core.experiment.report.report_manager.logger')
    def test_generate_with_invalid_report_type(
        self, mock_logger, mock_exists, mock_template_mgr, mock_asset_mgr, mock_renderer
    ):
        """Test that invalid report_type defaults to interactive"""
        manager = ReportManager()
        manager.renderers['robustness'] = mock_renderer

        results = {'score': 0.85}
        manager.generate_report('robustness', results, '/path/output.html', report_type='invalid')

        mock_logger.warning.assert_called()
        call_args = mock_renderer.render.call_args
        assert call_args[0][3] == 'interactive'  # Defaulted to interactive


# ==================== Error Handling Tests ====================


class TestErrorHandling:
    """Tests for error handling"""

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_raises_error_for_unsupported_test_type(
        self, mock_exists, mock_template_mgr, mock_asset_mgr
    ):
        """Test that unsupported test type raises NotImplementedError"""
        manager = ReportManager()

        with pytest.raises(NotImplementedError, match='not implemented'):
            manager.generate_report('unsupported_type', {}, '/path/output.html')

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_raises_error_when_renderer_fails(
        self, mock_exists, mock_template_mgr, mock_asset_mgr
    ):
        """Test that renderer errors are caught and re-raised as ValueError"""
        manager = ReportManager()

        failing_renderer = Mock()
        failing_renderer.render = Mock(side_effect=Exception("Render error"))
        manager.renderers['robustness'] = failing_renderer

        with pytest.raises(ValueError, match='Failed to generate'):
            manager.generate_report('robustness', {}, '/path/output.html')

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    @patch('deepbridge.core.experiment.report.report_manager.logger')
    def test_generate_logs_error_when_renderer_fails(
        self, mock_logger, mock_exists, mock_template_mgr, mock_asset_mgr
    ):
        """Test that errors are logged"""
        manager = ReportManager()

        failing_renderer = Mock()
        failing_renderer.render = Mock(side_effect=Exception("Render error"))
        manager.renderers['uncertainty'] = failing_renderer

        with pytest.raises(ValueError):
            manager.generate_report('uncertainty', {}, '/path/output.html')

        mock_logger.error.assert_called()


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_multiple_report_types(self, mock_exists, mock_template_mgr, mock_asset_mgr):
        """Test generating different report types in sequence"""
        manager = ReportManager()

        # Setup mock renderers
        for test_type in ['robustness', 'uncertainty', 'resilience']:
            mock_renderer = Mock()
            mock_renderer.render = Mock(return_value=f'/path/{test_type}_report.html')
            manager.renderers[test_type] = mock_renderer

        # Generate reports
        robustness_path = manager.generate_report('robustness', {'score': 0.85}, '/path/rob.html')
        uncertainty_path = manager.generate_report('uncertainty', {'coverage': 0.90}, '/path/unc.html')
        resilience_path = manager.generate_report('resilience', {'gap': 0.10}, '/path/res.html')

        assert 'robustness' in robustness_path
        assert 'uncertainty' in uncertainty_path
        assert 'resilience' in resilience_path

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_full_workflow_with_static_and_interactive(self, mock_exists, mock_template_mgr, mock_asset_mgr):
        """Test switching between static and interactive reports"""
        manager = ReportManager()
        manager.has_static_renderers = True

        interactive_renderer = Mock()
        interactive_renderer.render = Mock(return_value='/path/interactive.html')
        manager.renderers['robustness'] = interactive_renderer

        static_renderer = Mock()
        static_renderer.render = Mock(return_value='/path/static.html')
        manager.static_renderers['robustness'] = static_renderer

        results = {'score': 0.85}

        # Generate interactive
        interactive_path = manager.generate_report(
            'robustness', results, '/path/output.html', report_type='interactive'
        )

        # Generate static
        static_path = manager.generate_report(
            'robustness', results, '/path/output.html', report_type='static'
        )

        assert interactive_path == '/path/interactive.html'
        assert static_path == '/path/static.html'
        interactive_renderer.render.assert_called_once()
        static_renderer.render.assert_called_once()


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases"""

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_with_empty_results(self, mock_exists, mock_template_mgr, mock_asset_mgr, mock_renderer):
        """Test generating report with empty results dict"""
        manager = ReportManager()
        manager.renderers['robustness'] = mock_renderer

        report_path = manager.generate_report('robustness', {}, '/path/output.html')

        mock_renderer.render.assert_called_once()
        assert report_path == '/path/to/report.html'

    @patch('deepbridge.core.experiment.report.asset_manager.AssetManager')
    @patch('deepbridge.core.experiment.report.template_manager.TemplateManager')
    @patch('deepbridge.core.experiment.report.report_manager.os.path.exists', return_value=True)
    def test_generate_with_mixed_case_report_type(self, mock_exists, mock_template_mgr, mock_asset_mgr, mock_renderer):
        """Test that report_type is case-insensitive"""
        manager = ReportManager()
        manager.renderers['robustness'] = mock_renderer

        manager.generate_report('robustness', {}, '/path/output.html', report_type='INTERACTIVE')

        call_args = mock_renderer.render.call_args
        assert call_args[0][3] == 'interactive'
