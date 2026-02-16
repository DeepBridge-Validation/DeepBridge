"""Tests for base data structures and transformers."""

import pytest
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Any
from deepbridge.core.experiment.report.data.base import (
    ReportData,
    ModelResult,
    MetricValue,
    DataTransformer
)


# Mock implementations for testing abstract classes
@dataclass
class MockReportData(ReportData):
    """Mock report data for testing."""
    test_field: str = "test"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'generated_at': self.generated_at.isoformat(),
            'report_type': self.report_type,
            'version': self.version,
            'test_field': self.test_field,
            'metadata': self.metadata
        }

    def validate(self) -> bool:
        """Validate data."""
        if not self.test_field:
            raise ValueError("test_field is required")
        return True


class MockDataTransformer(DataTransformer):
    """Mock transformer for testing."""

    def transform(self, raw_data: Dict[str, Any]) -> ReportData:
        """Transform raw data."""
        self.validate_raw_data(raw_data)

        return MockReportData(
            generated_at=datetime.now(),
            report_type='mock',
            test_field=raw_data.get('test_field', 'default'),
            metadata=self._extract_metadata(raw_data)
        )


class TestReportData:
    """Test ReportData base class."""

    def test_creation_with_defaults(self):
        """Test creating report data with default values."""
        data = MockReportData(
            generated_at=datetime.now(),
            report_type='test'
        )

        assert data.report_type == 'test'
        assert data.version == '1.0.0'
        assert isinstance(data.metadata, dict)
        assert len(data.metadata) == 0

    def test_creation_with_custom_values(self):
        """Test creating report data with custom values."""
        now = datetime.now()
        metadata = {'key': 'value'}

        data = MockReportData(
            generated_at=now,
            report_type='custom',
            version='2.0.0',
            metadata=metadata,
            test_field='custom_test'
        )

        assert data.report_type == 'custom'
        assert data.version == '2.0.0'
        assert data.metadata == metadata
        assert data.test_field == 'custom_test'
        assert data.generated_at == now

    def test_to_dict(self):
        """Test to_dict method."""
        data = MockReportData(
            generated_at=datetime.now(),
            report_type='test',
            test_field='test_value'
        )

        result = data.to_dict()

        assert isinstance(result, dict)
        assert result['report_type'] == 'test'
        assert result['test_field'] == 'test_value'
        assert 'generated_at' in result

    def test_to_json_dict(self):
        """Test to_json_dict converts datetime to ISO format."""
        now = datetime.now()
        data = MockReportData(
            generated_at=now,
            report_type='test'
        )

        result = data.to_json_dict()

        assert isinstance(result, dict)
        assert isinstance(result['generated_at'], str)
        assert result['generated_at'] == now.isoformat()

    def test_to_json_dict_nested_datetime(self):
        """Test to_json_dict handles nested datetime objects."""
        now = datetime.now()
        data = MockReportData(
            generated_at=now,
            report_type='test',
            metadata={'timestamp': now}
        )

        result = data.to_json_dict()

        assert isinstance(result['metadata']['timestamp'], str)
        assert result['metadata']['timestamp'] == now.isoformat()

    def test_validate_valid_data(self):
        """Test validation with valid data."""
        data = MockReportData(
            generated_at=datetime.now(),
            report_type='test',
            test_field='valid'
        )

        assert data.validate() is True

    def test_validate_invalid_data(self):
        """Test validation with invalid data."""
        data = MockReportData(
            generated_at=datetime.now(),
            report_type='test',
            test_field=''  # Invalid
        )

        with pytest.raises(ValueError, match="test_field is required"):
            data.validate()


class TestModelResult:
    """Test ModelResult dataclass."""

    def test_creation_with_defaults(self):
        """Test creating model result with default values."""
        result = ModelResult(
            model_id='model_1',
            model_name='Test Model',
            metrics={'accuracy': 0.95}
        )

        assert result.model_id == 'model_1'
        assert result.model_name == 'Test Model'
        assert result.metrics == {'accuracy': 0.95}
        assert isinstance(result.test_results, list)
        assert len(result.test_results) == 0
        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0

    def test_creation_with_custom_values(self):
        """Test creating model result with custom values."""
        test_results = [{'test': 'result1'}, {'test': 'result2'}]
        metadata = {'version': '1.0'}

        result = ModelResult(
            model_id='model_2',
            model_name='Custom Model',
            metrics={'precision': 0.90, 'recall': 0.85},
            test_results=test_results,
            metadata=metadata
        )

        assert result.model_id == 'model_2'
        assert len(result.test_results) == 2
        assert result.metadata == metadata

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = ModelResult(
            model_id='model_1',
            model_name='Test Model',
            metrics={'accuracy': 0.95},
            test_results=[{'test': 1}],
            metadata={'key': 'value'}
        )

        data = result.to_dict()

        assert isinstance(data, dict)
        assert data['model_id'] == 'model_1'
        assert data['model_name'] == 'Test Model'
        assert data['metrics'] == {'accuracy': 0.95}
        assert len(data['test_results']) == 1
        assert data['metadata'] == {'key': 'value'}


class TestMetricValue:
    """Test MetricValue dataclass."""

    def test_creation_basic(self):
        """Test creating basic metric value."""
        metric = MetricValue(
            name='accuracy',
            value=0.95
        )

        assert metric.name == 'accuracy'
        assert metric.value == 0.95
        assert metric.unit is None
        assert metric.threshold is None
        assert metric.passed is None
        assert isinstance(metric.metadata, dict)

    def test_creation_with_threshold_auto_pass(self):
        """Test auto-calculation of passed when threshold is provided."""
        # Value above threshold -> passed
        metric = MetricValue(
            name='accuracy',
            value=0.95,
            threshold=0.90
        )

        assert metric.passed is True

    def test_creation_with_threshold_auto_fail(self):
        """Test auto-calculation of passed when below threshold."""
        # Value below threshold -> failed
        metric = MetricValue(
            name='accuracy',
            value=0.85,
            threshold=0.90
        )

        assert metric.passed is False

    def test_creation_with_explicit_passed(self):
        """Test that explicit passed value is preserved."""
        # Explicit passed value should not be overridden
        metric = MetricValue(
            name='accuracy',
            value=0.95,
            threshold=0.90,
            passed=False  # Explicitly set to False despite value > threshold
        )

        assert metric.passed is False

    def test_creation_with_all_fields(self):
        """Test creating metric with all fields."""
        metadata = {'source': 'test'}

        metric = MetricValue(
            name='precision',
            value=0.88,
            unit='%',
            threshold=0.85,
            passed=True,
            metadata=metadata
        )

        assert metric.name == 'precision'
        assert metric.value == 0.88
        assert metric.unit == '%'
        assert metric.threshold == 0.85
        assert metric.passed is True
        assert metric.metadata == metadata

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metric = MetricValue(
            name='recall',
            value=0.92,
            unit='ratio',
            threshold=0.90,
            metadata={'note': 'test'}
        )

        data = metric.to_dict()

        assert isinstance(data, dict)
        assert data['name'] == 'recall'
        assert data['value'] == 0.92
        assert data['unit'] == 'ratio'
        assert data['threshold'] == 0.90
        assert data['passed'] is True  # Auto-calculated
        assert data['metadata'] == {'note': 'test'}


class TestDataTransformer:
    """Test DataTransformer base class."""

    def test_validate_raw_data_valid(self):
        """Test validation with valid raw data."""
        transformer = MockDataTransformer()
        raw_data = {'test_field': 'test'}

        # Should not raise
        transformer.validate_raw_data(raw_data)

    def test_validate_raw_data_not_dict(self):
        """Test validation rejects non-dict data."""
        transformer = MockDataTransformer()

        with pytest.raises(ValueError, match="Expected dict"):
            transformer.validate_raw_data("not a dict")

        with pytest.raises(ValueError, match="Expected dict"):
            transformer.validate_raw_data([1, 2, 3])

    def test_validate_raw_data_empty(self):
        """Test validation rejects empty dict."""
        transformer = MockDataTransformer()

        with pytest.raises(ValueError, match="cannot be empty"):
            transformer.validate_raw_data({})

    def test_transform_basic(self):
        """Test basic transformation."""
        transformer = MockDataTransformer()
        raw_data = {'test_field': 'custom_value'}

        result = transformer.transform(raw_data)

        assert isinstance(result, MockReportData)
        assert result.test_field == 'custom_value'
        assert result.report_type == 'mock'

    def test_transform_invalid_data(self):
        """Test transformation with invalid data."""
        transformer = MockDataTransformer()

        with pytest.raises(ValueError):
            transformer.transform("invalid")

    def test_extract_metadata(self):
        """Test metadata extraction."""
        transformer = MockDataTransformer()
        raw_data = {
            'experiment_id': 'exp_123',
            'timestamp': '2024-01-01',
            'version': '1.0',
            'config': {'param': 'value'},
            'other_field': 'ignored'
        }

        metadata = transformer._extract_metadata(raw_data)

        assert 'experiment_id' in metadata
        assert 'timestamp' in metadata
        assert 'version' in metadata
        assert 'config' in metadata
        assert 'other_field' not in metadata

    def test_safe_get_default(self):
        """Test _safe_get with default value."""
        transformer = MockDataTransformer()
        data = {'key1': 'value1'}

        # Existing key
        assert transformer._safe_get(data, 'key1', 'default') == 'value1'

        # Missing key with default
        assert transformer._safe_get(data, 'key2', 'default') == 'default'

        # Missing key without default
        assert transformer._safe_get(data, 'key2') is None

    def test_safe_get_required(self):
        """Test _safe_get with required parameter."""
        transformer = MockDataTransformer()
        data = {'key1': 'value1'}

        # Required key exists
        assert transformer._safe_get(data, 'key1', required=True) == 'value1'

        # Required key missing
        with pytest.raises(ValueError, match="Required key 'key2' not found"):
            transformer._safe_get(data, 'key2', required=True)
