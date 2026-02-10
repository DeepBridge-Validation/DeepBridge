"""
Tests for BaseManager abstract class.

Coverage Target: 100%
"""

import pytest
from unittest.mock import Mock

from deepbridge.core.experiment.managers.base_manager import BaseManager


# Create concrete implementation for testing
class ConcreteManager(BaseManager):
    """Concrete implementation for testing"""
    
    def run_tests(self, config_name='quick', **kwargs):
        self.log(f"Running tests with config: {config_name}")
        return {'test': 'results'}
    
    def compare_models(self, config_name='quick', **kwargs):
        self.log(f"Comparing models with config: {config_name}")
        return {'comparison': 'results'}


@pytest.fixture
def mock_dataset():
    """Create mock dataset"""
    return Mock()


@pytest.fixture
def manager(mock_dataset):
    """Create ConcreteManager instance"""
    return ConcreteManager(dataset=mock_dataset)


@pytest.fixture
def verbose_manager(mock_dataset):
    """Create ConcreteManager with verbose=True"""
    return ConcreteManager(dataset=mock_dataset, verbose=True)


class TestGetResults:
    """Tests for get_results method"""
    
    def test_get_results_without_results_attribute(self, manager):
        """Test get_results when no results attribute exists"""
        result = manager.get_results()
        
        assert result == {}
    
    def test_get_results_with_results_attribute(self, manager):
        """Test get_results when results attribute exists"""
        manager.results = {'metric1': 0.95, 'metric2': 0.85}
        
        result = manager.get_results()
        
        assert result == {'metric1': 0.95, 'metric2': 0.85}
    
    def test_get_results_with_specific_type(self, manager):
        """Test get_results with specific result type"""
        manager.results = {
            'robustness': {'score': 0.9},
            'uncertainty': {'score': 0.8}
        }
        
        result = manager.get_results('robustness')
        
        assert result == {'score': 0.9}
    
    def test_get_results_with_nonexistent_type(self, manager):
        """Test get_results with type that doesn't exist"""
        manager.results = {'robustness': {'score': 0.9}}
        
        result = manager.get_results('nonexistent')
        
        # Returns all results when specific type not found
        assert result == {'robustness': {'score': 0.9}}
    
    def test_get_results_with_none_type(self, manager):
        """Test get_results with None returns all results"""
        manager.results = {'metric': 0.95}
        
        result = manager.get_results(None)
        
        assert result == {'metric': 0.95}
