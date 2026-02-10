"""
Tests for ReportAdapter base class.

Coverage Target: 100%
"""

import pytest
from unittest.mock import Mock

from deepbridge.core.experiment.report.adapters.base import ReportAdapter


# Concrete implementation for testing
class ConcreteAdapter(ReportAdapter):
    """Concrete implementation for testing"""
    
    def render(self, report):
        """Implement render method"""
        self._validate_report(report)
        return f"Rendered {report.metadata.model_name}"


@pytest.fixture
def adapter():
    """Create ConcreteAdapter instance"""
    return ConcreteAdapter()


@pytest.fixture
def valid_report():
    """Create valid report mock"""
    report = Mock()
    report.metadata = Mock()
    report.metadata.model_name = 'TestModel'
    report.metadata.test_type = 'robustness'
    return report


class TestValidateReport:
    """Tests for _validate_report method"""
    
    def test_validate_with_valid_report(self, adapter, valid_report):
        """Test validation with valid report"""
        # Should not raise
        adapter._validate_report(valid_report)
    
    def test_validate_raises_error_for_missing_metadata(self, adapter):
        """Test validation raises error when metadata is missing"""
        report = Mock()
        report.metadata = None
        
        with pytest.raises(ValueError, match='Report must have metadata'):
            adapter._validate_report(report)
    
    def test_validate_raises_error_for_missing_model_name(self, adapter):
        """Test validation raises error when model_name is missing"""
        report = Mock()
        report.metadata = Mock()
        report.metadata.model_name = None
        report.metadata.test_type = 'robustness'
        
        with pytest.raises(ValueError, match='metadata must have model_name'):
            adapter._validate_report(report)
    
    def test_validate_raises_error_for_missing_test_type(self, adapter):
        """Test validation raises error when test_type is missing"""
        report = Mock()
        report.metadata = Mock()
        report.metadata.model_name = 'TestModel'
        report.metadata.test_type = None
        
        with pytest.raises(ValueError, match='metadata must have test_type'):
            adapter._validate_report(report)
