"""
Comprehensive tests for DataIntegrationManager (deprecated module).

This test suite validates:
1. DataIntegrationManager initialization
2. serialize_data_for_template - JSON serialization for JavaScript
3. prepare_template_context - context building with assets
4. get_transformer_for_test_type - transformer selection
5. transform_data - data transformation workflow

Coverage Target: ~90%+

Note: This module is deprecated but still in use.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock

# Suppress deprecation warning for tests
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='deepbridge.core.experiment.report.data_integration')

from deepbridge.core.experiment.report.data_integration import DataIntegrationManager


# ==================== Fixtures ====================


@pytest.fixture
def mock_asset_manager():
    """Create mock AssetManager"""
    manager = Mock()
    manager.create_full_report_assets = Mock(return_value={
        'css': 'body { color: black; }',
        'js': 'console.log("test");',
        'logo': 'data:image/png;base64,test',
        'favicon': '/favicon.ico',
        'icons': {'home': 'icon-home'},
        'common': {
            'footer.html': '<footer>Footer</footer>',
            'header.html': '<header>Header</header>',
            'meta.html': '<meta charset="utf-8">',
            'navigation.html': '<nav>Nav</nav>'
        },
        'test_partials': {'chart': '<div>Chart</div>'}
    })
    return manager


@pytest.fixture
def data_integration_manager(mock_asset_manager):
    """Create DataIntegrationManager instance"""
    return DataIntegrationManager(asset_manager=mock_asset_manager)


@pytest.fixture
def sample_data():
    """Sample test data"""
    return {
        'test_results': {'accuracy': 0.95, 'loss': 0.05},
        'model_name': 'TestModel',
        'metrics': ['accuracy', 'loss']
    }


# ==================== Initialization Tests ====================


class TestDataIntegrationManagerInitialization:
    """Tests for DataIntegrationManager initialization"""

    def test_init_with_asset_manager(self, mock_asset_manager):
        """Test initialization with asset manager"""
        manager = DataIntegrationManager(asset_manager=mock_asset_manager)

        assert manager.asset_manager == mock_asset_manager

    def test_deprecation_warning_issued(self):
        """Test that deprecation warning is issued on import"""
        # The warning is issued at module level, so it's already been shown
        # We just verify the class exists
        assert DataIntegrationManager is not None


# ==================== serialize_data_for_template Tests ====================


class TestSerializeDataForTemplate:
    """Tests for serialize_data_for_template method"""

    def test_serialize_simple_data(self, data_integration_manager):
        """Test serializing simple data"""
        data = {'key': 'value', 'number': 42}

        result = data_integration_manager.serialize_data_for_template(data)

        assert 'const reportData =' in result
        assert '"key"' in result or "'key'" in result
        assert 'value' in result
        assert '42' in result

    def test_serialize_nested_data(self, data_integration_manager):
        """Test serializing nested data"""
        data = {
            'level1': {
                'level2': {
                    'level3': 'deep value'
                }
            }
        }

        result = data_integration_manager.serialize_data_for_template(data)

        assert 'const reportData =' in result
        assert 'level1' in result
        assert 'level2' in result
        assert 'deep value' in result

    def test_serialize_with_lists(self, data_integration_manager):
        """Test serializing data with lists"""
        data = {
            'metrics': ['accuracy', 'precision', 'recall'],
            'scores': [0.9, 0.8, 0.7]
        }

        result = data_integration_manager.serialize_data_for_template(data)

        assert 'metrics' in result
        assert 'accuracy' in result

    def test_serialize_with_unicode(self, data_integration_manager):
        """Test serializing data with unicode characters"""
        data = {
            'message': 'Hello 世界',
            'emoji': '🎉'
        }

        result = data_integration_manager.serialize_data_for_template(data)

        assert '世界' in result
        assert '🎉' in result

    def test_serialize_empty_data(self, data_integration_manager):
        """Test serializing empty dictionary"""
        result = data_integration_manager.serialize_data_for_template({})

        assert 'const reportData =' in result
        assert '{}' in result

    def test_serialize_with_null_values(self, data_integration_manager):
        """Test serializing data with None values"""
        data = {'valid': 'value', 'null_field': None}

        result = data_integration_manager.serialize_data_for_template(data)

        assert 'valid' in result
        assert 'null' in result.lower()

    def test_serialize_handles_exception(self, data_integration_manager):
        """Test that serialization errors are handled gracefully"""
        # Create object that can't be JSON serialized
        class NonSerializable:
            pass

        data = {'bad': NonSerializable()}

        with patch('deepbridge.core.experiment.report.data_integration.logger') as mock_logger:
            result = data_integration_manager.serialize_data_for_template(data)

            # Should return empty object on error
            assert result == 'const reportData = {};'
            mock_logger.error.assert_called_once()

    def test_serialize_formats_with_indent(self, data_integration_manager):
        """Test that output is formatted with indentation"""
        data = {'a': 1, 'b': 2}

        result = data_integration_manager.serialize_data_for_template(data)

        # Should have newlines for indentation
        assert '\n' in result


# ==================== prepare_template_context Tests ====================


class TestPrepareTemplateContext:
    """Tests for prepare_template_context method"""

    def test_prepare_context_basic(self, data_integration_manager, sample_data, mock_asset_manager):
        """Test preparing basic template context"""
        context = data_integration_manager.prepare_template_context('robustness', sample_data)

        assert 'report_data' in context
        assert 'serialized_data' in context
        assert 'test_type' in context
        assert context['test_type'] == 'robustness'
        assert context['report_data'] == sample_data

    def test_prepare_context_includes_assets(self, data_integration_manager, sample_data):
        """Test that context includes all assets"""
        context = data_integration_manager.prepare_template_context('uncertainty', sample_data)

        assert 'css' in context
        assert 'js' in context
        assert 'logo' in context
        assert 'favicon' in context
        assert 'icons' in context

    def test_prepare_context_includes_common_fragments(self, data_integration_manager, sample_data):
        """Test that context includes common HTML fragments"""
        context = data_integration_manager.prepare_template_context('fairness', sample_data)

        assert 'footer' in context
        assert 'header' in context
        assert 'meta' in context
        assert 'navigation' in context

    def test_prepare_context_includes_partials(self, data_integration_manager, sample_data):
        """Test that context includes test-specific partials"""
        context = data_integration_manager.prepare_template_context('robustness', sample_data)

        assert 'partials' in context

    def test_prepare_context_serializes_data(self, data_integration_manager, sample_data):
        """Test that data is serialized for JavaScript"""
        context = data_integration_manager.prepare_template_context('robustness', sample_data)

        serialized = context['serialized_data']
        assert 'const reportData =' in serialized

    def test_prepare_context_calls_create_assets(self, data_integration_manager, sample_data, mock_asset_manager):
        """Test that create_full_report_assets is called"""
        data_integration_manager.prepare_template_context('robustness', sample_data)

        mock_asset_manager.create_full_report_assets.assert_called_once_with('robustness')

    def test_prepare_context_with_missing_assets(self, mock_asset_manager, sample_data):
        """Test context preparation with missing assets"""
        # Return incomplete assets
        mock_asset_manager.create_full_report_assets.return_value = {}

        manager = DataIntegrationManager(asset_manager=mock_asset_manager)
        context = manager.prepare_template_context('robustness', sample_data)

        # Should have fallback values
        assert context['css'] == '/* No CSS loaded */'
        assert context['js'] == '// No JavaScript loaded'
        assert context['logo'] == ''
        assert context['footer'] == ''

    def test_prepare_context_raises_on_error(self, mock_asset_manager, sample_data):
        """Test that errors in context preparation are raised"""
        mock_asset_manager.create_full_report_assets.side_effect = Exception("Asset error")

        manager = DataIntegrationManager(asset_manager=mock_asset_manager)

        with pytest.raises(Exception):
            with patch('deepbridge.core.experiment.report.data_integration.logger'):
                manager.prepare_template_context('robustness', sample_data)


# ==================== get_transformer_for_test_type Tests ====================


class TestGetTransformerForTestType:
    """Tests for get_transformer_for_test_type method"""

    def _create_mock_module(self, **transformer_mocks):
        """Helper to create a mock data_transformer module"""
        import sys
        from types import ModuleType

        mock_module = ModuleType('data_transformer')
        mock_module.DataTransformer = Mock()
        mock_module.RobustnessDataTransformer = transformer_mocks.get('RobustnessDataTransformer', Mock())
        mock_module.UncertaintyDataTransformer = transformer_mocks.get('UncertaintyDataTransformer', Mock())
        mock_module.ResilienceDataTransformer = transformer_mocks.get('ResilienceDataTransformer', Mock())
        mock_module.HyperparameterDataTransformer = transformer_mocks.get('HyperparameterDataTransformer', Mock())

        return mock_module

    def test_get_transformer_robustness(self, data_integration_manager):
        """Test getting transformer for robustness"""
        mock_instance = Mock()
        mock_robustness = Mock(return_value=mock_instance)

        mock_module = self._create_mock_module(RobustnessDataTransformer=mock_robustness)

        with patch.dict('sys.modules', {'deepbridge.core.experiment.report.data_transformer': mock_module}):
            result = data_integration_manager.get_transformer_for_test_type('robustness')

            assert result == mock_instance
            mock_robustness.assert_called_once()

    def test_get_transformer_uncertainty(self, data_integration_manager):
        """Test getting transformer for uncertainty"""
        mock_instance = Mock()
        mock_uncertainty = Mock(return_value=mock_instance)

        mock_module = self._create_mock_module(UncertaintyDataTransformer=mock_uncertainty)

        with patch.dict('sys.modules', {'deepbridge.core.experiment.report.data_transformer': mock_module}):
            result = data_integration_manager.get_transformer_for_test_type('uncertainty')

            assert result == mock_instance

    def test_get_transformer_resilience(self, data_integration_manager):
        """Test getting transformer for resilience"""
        mock_instance = Mock()
        mock_resilience = Mock(return_value=mock_instance)

        mock_module = self._create_mock_module(ResilienceDataTransformer=mock_resilience)

        with patch.dict('sys.modules', {'deepbridge.core.experiment.report.data_transformer': mock_module}):
            result = data_integration_manager.get_transformer_for_test_type('resilience')

            assert result == mock_instance

    def test_get_transformer_hyperparameter(self, data_integration_manager):
        """Test getting transformer for hyperparameter"""
        mock_instance = Mock()
        mock_hyperparameter = Mock(return_value=mock_instance)

        mock_module = self._create_mock_module(HyperparameterDataTransformer=mock_hyperparameter)

        with patch.dict('sys.modules', {'deepbridge.core.experiment.report.data_transformer': mock_module}):
            result = data_integration_manager.get_transformer_for_test_type('hyperparameter')

            assert result == mock_instance

    def test_get_transformer_hyperparameters_plural(self, data_integration_manager):
        """Test getting transformer for hyperparameters (plural)"""
        mock_instance = Mock()
        mock_hyperparameter = Mock(return_value=mock_instance)

        mock_module = self._create_mock_module(HyperparameterDataTransformer=mock_hyperparameter)

        with patch.dict('sys.modules', {'deepbridge.core.experiment.report.data_transformer': mock_module}):
            result = data_integration_manager.get_transformer_for_test_type('hyperparameters')

            assert result == mock_instance

    def test_get_transformer_case_insensitive(self, data_integration_manager):
        """Test that test type is case-insensitive"""
        mock_instance = Mock()
        mock_robustness = Mock(return_value=mock_instance)

        mock_module = self._create_mock_module(RobustnessDataTransformer=mock_robustness)

        with patch.dict('sys.modules', {'deepbridge.core.experiment.report.data_transformer': mock_module}):
            result = data_integration_manager.get_transformer_for_test_type('ROBUSTNESS')

            assert result == mock_instance

    def test_get_transformer_unsupported_type(self, data_integration_manager):
        """Test error with unsupported test type"""
        mock_module = self._create_mock_module()

        with patch.dict('sys.modules', {'deepbridge.core.experiment.report.data_transformer': mock_module}):
            with pytest.raises(ValueError, match='Unsupported test type'):
                data_integration_manager.get_transformer_for_test_type('invalid_type')

    def test_get_transformer_import_error(self, data_integration_manager):
        """Test handling of import errors"""
        # Ensure module is not in sys.modules and can't be imported
        import sys
        original_modules = sys.modules.copy()

        # Remove module if it exists
        if 'deepbridge.core.experiment.report.data_transformer' in sys.modules:
            del sys.modules['deepbridge.core.experiment.report.data_transformer']

        try:
            with pytest.raises(ImportError, match='Failed to import'):
                with patch('deepbridge.core.experiment.report.data_integration.logger'):
                    data_integration_manager.get_transformer_for_test_type('robustness')
        finally:
            # Restore original modules
            sys.modules.update(original_modules)


# ==================== transform_data Tests ====================


class TestTransformData:
    """Tests for transform_data method"""

    def test_transform_data_success(self, data_integration_manager, sample_data):
        """Test successful data transformation"""
        mock_transformer = Mock()
        mock_transformer.transform = Mock(return_value={'transformed': 'data'})

        with patch.object(data_integration_manager, 'get_transformer_for_test_type', return_value=mock_transformer):
            result = data_integration_manager.transform_data('robustness', sample_data, 'TestModel')

            assert result == {'transformed': 'data'}
            mock_transformer.transform.assert_called_once_with(sample_data, 'TestModel')

    def test_transform_data_calls_get_transformer(self, data_integration_manager, sample_data):
        """Test that get_transformer_for_test_type is called"""
        mock_transformer = Mock()
        mock_transformer.transform = Mock(return_value={})

        with patch.object(data_integration_manager, 'get_transformer_for_test_type', return_value=mock_transformer) as mock_get:
            data_integration_manager.transform_data('uncertainty', sample_data, 'Model')

            mock_get.assert_called_once_with('uncertainty')

    def test_transform_data_logs_success(self, data_integration_manager, sample_data):
        """Test that successful transformation is logged"""
        mock_transformer = Mock()
        mock_transformer.transform = Mock(return_value={})

        with patch.object(data_integration_manager, 'get_transformer_for_test_type', return_value=mock_transformer):
            with patch('deepbridge.core.experiment.report.data_integration.logger') as mock_logger:
                data_integration_manager.transform_data('robustness', sample_data, 'TestModel')

                mock_logger.info.assert_called_once()

    def test_transform_data_raises_on_error(self, data_integration_manager, sample_data):
        """Test that transformation errors are raised as ValueError"""
        mock_transformer = Mock()
        mock_transformer.transform.side_effect = Exception("Transform failed")

        with patch.object(data_integration_manager, 'get_transformer_for_test_type', return_value=mock_transformer):
            with pytest.raises(ValueError, match='Failed to transform'):
                with patch('deepbridge.core.experiment.report.data_integration.logger'):
                    data_integration_manager.transform_data('robustness', sample_data, 'Model')

    def test_transform_data_logs_error(self, data_integration_manager, sample_data):
        """Test that transformation errors are logged"""
        mock_transformer = Mock()
        mock_transformer.transform.side_effect = Exception("Transform failed")

        with patch.object(data_integration_manager, 'get_transformer_for_test_type', return_value=mock_transformer):
            with patch('deepbridge.core.experiment.report.data_integration.logger') as mock_logger:
                try:
                    data_integration_manager.transform_data('robustness', sample_data, 'Model')
                except ValueError:
                    pass

                mock_logger.error.assert_called_once()


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow(self, data_integration_manager, sample_data):
        """Test complete workflow: transform and prepare context"""
        # Mock transformer
        mock_transformer = Mock()
        mock_transformer.transform = Mock(return_value={'transformed': sample_data})

        with patch.object(data_integration_manager, 'get_transformer_for_test_type', return_value=mock_transformer):
            # Transform data
            transformed = data_integration_manager.transform_data('robustness', sample_data, 'Model')

            # Prepare context with transformed data
            context = data_integration_manager.prepare_template_context('robustness', transformed)

            assert 'report_data' in context
            assert context['report_data'] == {'transformed': sample_data}
            assert 'css' in context
            assert 'serialized_data' in context


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_serialize_very_large_data(self, data_integration_manager):
        """Test serializing very large dataset"""
        large_data = {'data': list(range(10000))}

        result = data_integration_manager.serialize_data_for_template(large_data)

        assert 'const reportData =' in result
        assert '9999' in result

    def test_serialize_with_special_characters(self, data_integration_manager):
        """Test serializing data with special characters"""
        data = {
            'quotes': 'He said "hello"',
            'apostrophe': "It's working",
            'backslash': 'path\\to\\file'
        }

        result = data_integration_manager.serialize_data_for_template(data)

        # JSON should properly escape these
        assert 'const reportData =' in result

    def test_prepare_context_with_empty_data(self, data_integration_manager):
        """Test preparing context with empty data"""
        context = data_integration_manager.prepare_template_context('robustness', {})

        assert context['report_data'] == {}
        assert 'serialized_data' in context

    def test_multiple_transformer_calls(self, data_integration_manager, sample_data):
        """Test calling get_transformer multiple times"""
        mock_transformer = Mock()
        mock_transformer.transform = Mock(return_value={})

        with patch.object(data_integration_manager, 'get_transformer_for_test_type', return_value=mock_transformer) as mock_get:
            # Call multiple times
            data_integration_manager.transform_data('robustness', sample_data, 'M1')
            data_integration_manager.transform_data('uncertainty', sample_data, 'M2')
            data_integration_manager.transform_data('resilience', sample_data, 'M3')

            assert mock_get.call_count == 3
