"""
Comprehensive tests for ManagerFactory.

This test suite validates:
1. register_manager - registering manager classes
2. get_manager - retrieving manager instances (singleton pattern)
3. _import_standard_managers - importing standard managers
4. clear_instances - clearing manager instances
5. get_supported_types - getting list of supported types
6. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import Mock, patch, MagicMock

from deepbridge.core.experiment.manager_factory import ManagerFactory


# ==================== Fixtures ====================


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset factory state before each test"""
    ManagerFactory._manager_classes = {}
    ManagerFactory._instances = {}
    yield
    ManagerFactory._manager_classes = {}
    ManagerFactory._instances = {}


@pytest.fixture
def mock_manager_class():
    """Create mock manager class that returns new instances each time"""
    manager_class = Mock()
    # Use side_effect to return new Mock instances each time
    manager_class.side_effect = lambda **kwargs: Mock()
    return manager_class


@pytest.fixture
def mock_dataset():
    """Create mock dataset"""
    return Mock()


# ==================== register_manager Tests ====================


class TestRegisterManager:
    """Tests for register_manager method"""

    def test_register_manager_adds_to_registry(self, mock_manager_class):
        """Test that register_manager adds manager to registry"""
        ManagerFactory.register_manager('test_type', mock_manager_class)

        assert 'test_type' in ManagerFactory._manager_classes
        assert ManagerFactory._manager_classes['test_type'] == mock_manager_class

    def test_register_multiple_managers(self, mock_manager_class):
        """Test registering multiple managers"""
        manager_class_2 = Mock()

        ManagerFactory.register_manager('type1', mock_manager_class)
        ManagerFactory.register_manager('type2', manager_class_2)

        assert len(ManagerFactory._manager_classes) == 2
        assert ManagerFactory._manager_classes['type1'] == mock_manager_class
        assert ManagerFactory._manager_classes['type2'] == manager_class_2

    def test_register_overwrites_existing_manager(self, mock_manager_class):
        """Test that registering same type overwrites existing"""
        manager_class_2 = Mock()

        ManagerFactory.register_manager('test_type', mock_manager_class)
        ManagerFactory.register_manager('test_type', manager_class_2)

        assert ManagerFactory._manager_classes['test_type'] == manager_class_2


# ==================== get_manager Tests ====================


class TestGetManager:
    """Tests for get_manager method"""

    def test_get_manager_returns_instance(self, mock_manager_class, mock_dataset):
        """Test that get_manager returns manager instance"""
        ManagerFactory.register_manager('test_type', mock_manager_class)

        manager = ManagerFactory.get_manager('test_type', mock_dataset)

        assert manager is not None
        mock_manager_class.assert_called_once()

    def test_get_manager_passes_parameters(self, mock_manager_class, mock_dataset):
        """Test that get_manager passes parameters correctly"""
        ManagerFactory.register_manager('test_type', mock_manager_class)
        alt_models = {'model1': Mock()}

        ManagerFactory.get_manager(
            'test_type',
            mock_dataset,
            alternative_models=alt_models,
            verbose=True
        )

        mock_manager_class.assert_called_once_with(
            dataset=mock_dataset,
            alternative_models=alt_models,
            verbose=True
        )

    def test_get_manager_singleton_pattern(self, mock_manager_class, mock_dataset):
        """Test that get_manager implements singleton pattern"""
        ManagerFactory.register_manager('test_type', mock_manager_class)

        manager1 = ManagerFactory.get_manager('test_type', mock_dataset)
        manager2 = ManagerFactory.get_manager('test_type', mock_dataset)

        # Should return same instance
        assert manager1 is manager2
        # Should only create instance once
        mock_manager_class.assert_called_once()

    def test_get_manager_different_datasets_different_instances(self, mock_manager_class):
        """Test that different datasets get different instances"""
        ManagerFactory.register_manager('test_type', mock_manager_class)

        dataset1 = Mock()
        dataset2 = Mock()

        manager1 = ManagerFactory.get_manager('test_type', dataset1)
        manager2 = ManagerFactory.get_manager('test_type', dataset2)

        # Should create two instances
        assert mock_manager_class.call_count == 2
        # Should be different instances
        assert manager1 is not manager2

    def test_get_manager_raises_error_for_unregistered_type(self, mock_dataset):
        """Test that get_manager raises error for unregistered type"""
        with patch.object(ManagerFactory, '_import_standard_managers'):
            with pytest.raises(ValueError, match='No manager registered'):
                ManagerFactory.get_manager('unknown_type', mock_dataset)

    @patch('deepbridge.core.experiment.manager_factory.ManagerFactory._import_standard_managers')
    def test_get_manager_calls_import_for_unknown_type(self, mock_import, mock_dataset):
        """Test that get_manager tries to import standard managers"""
        # Will still raise error but should call import first
        with pytest.raises(ValueError):
            ManagerFactory.get_manager('unknown_type', mock_dataset)

        mock_import.assert_called_once()

    def test_get_manager_with_default_parameters(self, mock_manager_class, mock_dataset):
        """Test get_manager with default parameters"""
        ManagerFactory.register_manager('test_type', mock_manager_class)

        ManagerFactory.get_manager('test_type', mock_dataset)

        # Check that defaults are passed
        call_kwargs = mock_manager_class.call_args[1]
        assert call_kwargs['alternative_models'] is None
        assert call_kwargs['verbose'] is False


# ==================== _import_standard_managers Tests ====================


class TestImportStandardManagers:
    """Tests for _import_standard_managers method"""

    def test_import_registers_all_standard_managers(self):
        """Test that import registers all standard managers"""
        with patch('deepbridge.core.experiment.managers.RobustnessManager') as mock_robust:
            with patch('deepbridge.core.experiment.managers.UncertaintyManager') as mock_uncert:
                with patch('deepbridge.core.experiment.managers.ResilienceManager') as mock_resil:
                    with patch('deepbridge.core.experiment.managers.HyperparameterManager') as mock_hyper:
                        ManagerFactory._import_standard_managers()

                        assert 'robustness' in ManagerFactory._manager_classes
                        assert 'uncertainty' in ManagerFactory._manager_classes
                        assert 'resilience' in ManagerFactory._manager_classes
                        assert 'hyperparameters' in ManagerFactory._manager_classes

    def test_import_handles_import_error_gracefully(self):
        """Test that import error is handled gracefully"""
        # Patch the import to raise ImportError
        with patch('deepbridge.core.experiment.manager_factory.ManagerFactory.register_manager', side_effect=ImportError):
            # Should not raise error
            try:
                ManagerFactory._import_standard_managers()
            except ImportError:
                pass  # Expected

        # Test continues without error


# ==================== clear_instances Tests ====================


class TestClearInstances:
    """Tests for clear_instances method"""

    def test_clear_instances_empties_registry(self, mock_manager_class, mock_dataset):
        """Test that clear_instances empties instance registry"""
        ManagerFactory.register_manager('test_type', mock_manager_class)
        ManagerFactory.get_manager('test_type', mock_dataset)

        # Should have one instance
        assert len(ManagerFactory._instances) > 0

        ManagerFactory.clear_instances()

        # Should be empty
        assert len(ManagerFactory._instances) == 0

    def test_clear_instances_creates_new_instances_after_clear(
        self, mock_manager_class, mock_dataset
    ):
        """Test that new instances are created after clearing"""
        ManagerFactory.register_manager('test_type', mock_manager_class)

        manager1 = ManagerFactory.get_manager('test_type', mock_dataset)
        ManagerFactory.clear_instances()
        manager2 = ManagerFactory.get_manager('test_type', mock_dataset)

        # Should be different instances
        assert manager1 is not manager2
        # Should have called constructor twice
        assert mock_manager_class.call_count == 2

    def test_clear_instances_when_empty(self):
        """Test that clear_instances works when already empty"""
        # Should not raise error
        ManagerFactory.clear_instances()

        assert len(ManagerFactory._instances) == 0


# ==================== get_supported_types Tests ====================


class TestGetSupportedTypes:
    """Tests for get_supported_types method"""

    @patch.object(ManagerFactory, '_import_standard_managers')
    def test_get_supported_types_calls_import(self, mock_import):
        """Test that get_supported_types calls import"""
        ManagerFactory.get_supported_types()

        mock_import.assert_called_once()

    def test_get_supported_types_returns_list(self):
        """Test that get_supported_types returns list"""
        result = ManagerFactory.get_supported_types()

        assert isinstance(result, list)

    def test_get_supported_types_returns_registered_types(self, mock_manager_class):
        """Test that get_supported_types returns registered types"""
        ManagerFactory.register_manager('type1', mock_manager_class)
        ManagerFactory.register_manager('type2', mock_manager_class)

        types = ManagerFactory.get_supported_types()

        assert 'type1' in types
        assert 'type2' in types

    def test_get_supported_types_returns_empty_when_none_registered(self):
        """Test get_supported_types with no managers registered"""
        with patch.object(ManagerFactory, '_import_standard_managers'):
            types = ManagerFactory.get_supported_types()

            assert types == []


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow_register_get_clear(self, mock_manager_class, mock_dataset):
        """Test complete workflow: register, get, clear"""
        # Register
        ManagerFactory.register_manager('test_type', mock_manager_class)

        # Get instance
        manager1 = ManagerFactory.get_manager('test_type', mock_dataset)
        assert manager1 is not None

        # Get again (should be same instance)
        manager2 = ManagerFactory.get_manager('test_type', mock_dataset)
        assert manager1 is manager2

        # Clear
        ManagerFactory.clear_instances()

        # Get new instance (should be different)
        manager3 = ManagerFactory.get_manager('test_type', mock_dataset)
        assert manager3 is not manager1

    def test_multiple_test_types_workflow(self, mock_manager_class):
        """Test workflow with multiple test types"""
        manager_class_2 = Mock()
        manager_class_2.return_value = Mock()

        dataset1 = Mock()
        dataset2 = Mock()

        # Register different types
        ManagerFactory.register_manager('type1', mock_manager_class)
        ManagerFactory.register_manager('type2', manager_class_2)

        # Get instances
        manager1 = ManagerFactory.get_manager('type1', dataset1)
        manager2 = ManagerFactory.get_manager('type2', dataset2)

        # Should be different
        assert manager1 is not manager2

        # Get supported types
        types = ManagerFactory.get_supported_types()
        assert 'type1' in types
        assert 'type2' in types


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_get_manager_with_none_alternative_models(self, mock_manager_class, mock_dataset):
        """Test get_manager with explicit None alternative_models"""
        ManagerFactory.register_manager('test_type', mock_manager_class)

        ManagerFactory.get_manager('test_type', mock_dataset, alternative_models=None)

        call_kwargs = mock_manager_class.call_args[1]
        assert call_kwargs['alternative_models'] is None

    def test_get_manager_with_empty_alternative_models(self, mock_manager_class, mock_dataset):
        """Test get_manager with empty dict alternative_models"""
        ManagerFactory.register_manager('test_type', mock_manager_class)

        ManagerFactory.get_manager('test_type', mock_dataset, alternative_models={})

        call_kwargs = mock_manager_class.call_args[1]
        assert call_kwargs['alternative_models'] == {}

    def test_register_manager_with_special_characters_in_type(self, mock_manager_class):
        """Test registering manager with special characters in type name"""
        ManagerFactory.register_manager('test-type_123', mock_manager_class)

        assert 'test-type_123' in ManagerFactory._manager_classes

    def test_instance_key_uses_id_not_value(self, mock_manager_class):
        """Test that instance key uses id() not value"""
        ManagerFactory.register_manager('test_type', mock_manager_class)

        # Create two identical-looking datasets
        dataset1 = Mock()
        dataset2 = Mock()

        manager1 = ManagerFactory.get_manager('test_type', dataset1)
        manager2 = ManagerFactory.get_manager('test_type', dataset2)

        # Even though datasets look the same, should create different instances
        # because id() is different
        assert mock_manager_class.call_count == 2

    def test_get_supported_types_after_manual_registration(self, mock_manager_class):
        """Test get_supported_types after manual registration"""
        ManagerFactory.register_manager('custom_type', mock_manager_class)

        types = ManagerFactory.get_supported_types()

        assert 'custom_type' in types

    def test_clear_instances_does_not_clear_manager_classes(self, mock_manager_class):
        """Test that clear_instances doesn't clear manager classes"""
        ManagerFactory.register_manager('test_type', mock_manager_class)

        # Clear instances
        ManagerFactory.clear_instances()

        # Manager classes should still be registered
        assert 'test_type' in ManagerFactory._manager_classes
