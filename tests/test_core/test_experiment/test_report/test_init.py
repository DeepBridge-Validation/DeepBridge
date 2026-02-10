"""
Tests for report __init__ module.

Coverage Target: 100%
"""

import pytest

from deepbridge.core.experiment.report import get_transformer


class TestGetTransformer:
    """Tests for get_transformer function"""
    
    def test_get_transformer_robustness(self):
        """Test getting robustness transformer"""
        transformer = get_transformer('robustness')
        
        assert transformer is not None
        assert transformer.__class__.__name__ == 'RobustnessDataTransformer'
    
    def test_get_transformer_uncertainty(self):
        """Test getting uncertainty transformer"""
        transformer = get_transformer('uncertainty')
        
        assert transformer is not None
        assert transformer.__class__.__name__ == 'UncertaintyDataTransformer'
    
    def test_get_transformer_resilience(self):
        """Test getting resilience transformer"""
        transformer = get_transformer('resilience')
        
        assert transformer is not None
        assert transformer.__class__.__name__ == 'ResilienceDataTransformer'
    
    def test_get_transformer_hyperparameter(self):
        """Test getting hyperparameter transformer"""
        transformer = get_transformer('hyperparameter')
        
        assert transformer is not None
        assert transformer.__class__.__name__ == 'HyperparameterDataTransformer'
    
    def test_get_transformer_case_insensitive(self):
        """Test that function is case insensitive"""
        transformer1 = get_transformer('ROBUSTNESS')
        transformer2 = get_transformer('Robustness')
        transformer3 = get_transformer('robustness')
        
        assert transformer1.__class__.__name__ == 'RobustnessDataTransformer'
        assert transformer2.__class__.__name__ == 'RobustnessDataTransformer'
        assert transformer3.__class__.__name__ == 'RobustnessDataTransformer'
    
    def test_get_transformer_invalid_type_raises_error(self):
        """Test that invalid type raises ValueError"""
        with pytest.raises(ValueError, match='Unsupported report type'):
            get_transformer('invalid')
    
    def test_get_transformer_error_message_lists_supported_types(self):
        """Test that error message lists supported types"""
        with pytest.raises(ValueError) as exc_info:
            get_transformer('invalid')
        
        error_msg = str(exc_info.value)
        assert 'robustness' in error_msg
        assert 'uncertainty' in error_msg
        assert 'resilience' in error_msg
        assert 'hyperparameter' in error_msg
