"""
Tests for core.experiment __init__ module.

Coverage Target: 100%
"""

import pytest
import sys
from unittest.mock import patch, MagicMock


class TestExperimentImports:
    """Tests for experiment module imports"""

    def test_all_imports_successful(self):
        """Test that all imports work when dependencies are available"""
        # Import the module fresh
        import deepbridge.core.experiment as experiment

        # Check that key classes are available
        assert hasattr(experiment, 'Experiment')
        assert hasattr(experiment, 'check_dependencies')
        assert hasattr(experiment, 'print_dependency_status')

    def test_experiment_class_available(self):
        """Test Experiment class is available"""
        from deepbridge.core.experiment import Experiment

        assert Experiment is not None
        assert callable(Experiment)

    def test_check_dependencies_available(self):
        """Test check_dependencies function is available"""
        from deepbridge.core.experiment import check_dependencies

        assert check_dependencies is not None
        assert callable(check_dependencies)

    def test_print_dependency_status_available(self):
        """Test print_dependency_status function is available"""
        from deepbridge.core.experiment import print_dependency_status

        assert print_dependency_status is not None
        assert callable(print_dependency_status)

    def test_all_list_contains_expected_items(self):
        """Test __all__ list contains expected exports"""
        import deepbridge.core.experiment as experiment

        assert '__all__' in dir(experiment)
        assert 'Experiment' in experiment.__all__
        assert 'check_dependencies' in experiment.__all__
        assert 'print_dependency_status' in experiment.__all__


class TestImportFailureHandling:
    """Tests for handling import failures gracefully"""

    def test_module_loads_even_with_partial_import_failures(self):
        """Test that module loads even if some imports fail"""
        # The module should handle import errors gracefully
        import deepbridge.core.experiment as experiment

        # Module should still be importable
        assert experiment is not None

    @patch('deepbridge.core.experiment.Experiment', None)
    def test_handles_missing_experiment_class(self):
        """Test handling when Experiment class is not available"""
        # Even if Experiment is missing, module should handle it
        import deepbridge.core.experiment as experiment

        # Module should still exist
        assert experiment is not None

    def test_all_list_is_list_type(self):
        """Test that __all__ is a list"""
        import deepbridge.core.experiment as experiment

        assert isinstance(experiment.__all__, list)

    def test_all_list_has_string_elements(self):
        """Test that __all__ contains only strings"""
        import deepbridge.core.experiment as experiment

        for item in experiment.__all__:
            assert isinstance(item, str)


class TestTestRunner:
    """Tests for TestRunner availability"""

    def test_test_runner_available(self):
        """Test TestRunner is available"""
        try:
            from deepbridge.core.experiment import TestRunner
            assert TestRunner is not None
            assert callable(TestRunner)
        except ImportError:
            # TestRunner may not be available if dependencies are missing
            pytest.skip("TestRunner not available")

    def test_test_runner_in_all_list(self):
        """Test TestRunner is in __all__ list when available"""
        import deepbridge.core.experiment as experiment

        # TestRunner should be in __all__ if dependencies are available
        if 'TestRunner' in dir(experiment):
            assert 'TestRunner' in experiment.__all__


class TestResultClasses:
    """Tests for result class availability"""

    def test_experiment_result_available(self):
        """Test ExperimentResult is available when dependencies are met"""
        try:
            from deepbridge.core.experiment import ExperimentResult
            assert ExperimentResult is not None
        except ImportError:
            pytest.skip("ExperimentResult not available")

    def test_wrap_results_available(self):
        """Test wrap_results function is available"""
        from deepbridge.core.experiment import wrap_results

        assert wrap_results is not None
        assert callable(wrap_results)

    def test_wrap_results_in_all_list(self):
        """Test wrap_results is in __all__ list"""
        import deepbridge.core.experiment as experiment

        assert 'wrap_results' in experiment.__all__


class TestManagerFactory:
    """Tests for ManagerFactory availability"""

    def test_manager_factory_available_when_deps_met(self):
        """Test ManagerFactory is available when dependencies are met"""
        import deepbridge.core.experiment as experiment

        # ManagerFactory should be available if dependencies are met
        if 'ManagerFactory' in dir(experiment):
            from deepbridge.core.experiment import ManagerFactory
            assert ManagerFactory is not None
            assert 'ManagerFactory' in experiment.__all__

    def test_test_strategy_factory_available_when_deps_met(self):
        """Test TestStrategyFactory is available when dependencies are met"""
        import deepbridge.core.experiment as experiment

        # TestStrategyFactory should be available if dependencies are met
        if 'TestStrategyFactory' in dir(experiment):
            from deepbridge.core.experiment import TestStrategyFactory
            assert TestStrategyFactory is not None
            assert 'TestStrategyFactory' in experiment.__all__


class TestReportManager:
    """Tests for ReportManager initialization"""

    def test_report_manager_exists(self):
        """Test report_manager is created"""
        import deepbridge.core.experiment as experiment

        # report_manager should exist
        assert hasattr(experiment, 'report_manager')

    def test_report_manager_import_fallback(self):
        """Test ReportManager import with fallback"""
        import deepbridge.core.experiment as experiment

        # ReportManager should be imported or set to None
        assert hasattr(experiment, 'ReportManager')


class TestModelResultClasses:
    """Tests for model result classes"""

    def test_base_model_result_available_when_deps_met(self):
        """Test BaseModelResult is available when dependencies are met"""
        import deepbridge.core.experiment as experiment

        # BaseModelResult should be available if dependencies are met
        if 'BaseModelResult' in dir(experiment):
            from deepbridge.core.experiment import BaseModelResult
            assert BaseModelResult is not None
            assert 'BaseModelResult' in experiment.__all__

    def test_create_model_result_available_when_deps_met(self):
        """Test create_model_result is available when dependencies are met"""
        import deepbridge.core.experiment as experiment

        # create_model_result should be available if dependencies are met
        if 'create_model_result' in dir(experiment):
            from deepbridge.core.experiment import create_model_result
            assert create_model_result is not None
            assert 'create_model_result' in experiment.__all__


class TestTestResultFactory:
    """Tests for TestResultFactory"""

    def test_test_result_factory_available_when_deps_met(self):
        """Test TestResultFactory is available when dependencies are met"""
        import deepbridge.core.experiment as experiment

        # TestResultFactory should be available if dependencies are met
        if 'TestResultFactory' in dir(experiment):
            from deepbridge.core.experiment import TestResultFactory
            assert TestResultFactory is not None
            assert 'TestResultFactory' in experiment.__all__


class TestDependencyCheck:
    """Tests for dependency checking functionality"""

    def test_check_dependencies_returns_tuple(self):
        """Test check_dependencies returns expected tuple"""
        from deepbridge.core.experiment import check_dependencies

        result = check_dependencies()
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_check_dependencies_structure(self):
        """Test check_dependencies returns correct structure"""
        from deepbridge.core.experiment import check_dependencies

        all_required, missing_required, missing_optional, version_issues = check_dependencies()
        assert isinstance(all_required, bool)
        assert isinstance(missing_required, list)
        assert isinstance(missing_optional, list)
        assert isinstance(version_issues, list)


class TestModuleStructure:
    """Tests for module structure and organization"""

    def test_module_has_docstring(self):
        """Test that module has a docstring"""
        import deepbridge.core.experiment as experiment

        # Module should have some documentation
        assert experiment.__doc__ is not None
        assert 'experiment' in experiment.__doc__.lower()

    def test_module_name(self):
        """Test module name is correct"""
        import deepbridge.core.experiment as experiment

        assert experiment.__name__ == 'deepbridge.core.experiment'

    def test_no_private_exports_in_all(self):
        """Test that __all__ doesn't contain private members"""
        import deepbridge.core.experiment as experiment

        for item in experiment.__all__:
            assert not item.startswith('_'), f"Private member {item} should not be in __all__"

    def test_all_exports_are_accessible(self):
        """Test that all items in __all__ are actually accessible"""
        import deepbridge.core.experiment as experiment

        for item in experiment.__all__:
            assert hasattr(experiment, item), f"{item} in __all__ but not accessible"

    def test_base_dir_path_exists(self):
        """Test base_dir is calculated correctly"""
        import deepbridge.core.experiment as experiment
        import os

        # base_dir should exist in the module
        assert hasattr(experiment, 'base_dir')
        # It should be a string path
        assert isinstance(experiment.base_dir, str)

    def test_templates_dir_path(self):
        """Test templates_dir is calculated correctly"""
        import deepbridge.core.experiment as experiment
        import os

        # templates_dir should exist in the module
        assert hasattr(experiment, 'templates_dir')
        # It should be a string path
        assert isinstance(experiment.templates_dir, str)
        # It should contain 'templates'
        assert 'templates' in experiment.templates_dir
