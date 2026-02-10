"""
Tests for JSON utilities module.

Coverage Target: 100%
"""

import pytest
import json
import datetime
import math
from unittest.mock import patch, Mock

from deepbridge.core.experiment.report.utils.json_utils import (
    SafeJSONEncoder,
    safe_json_dumps,
    safe_json_loads,
    clean_for_json,
    format_for_javascript,
    json_serializer,
    prepare_data_for_template,
    HAS_NUMPY
)

# Import numpy if available
if HAS_NUMPY:
    import numpy as np


class TestSafeJSONEncoder:
    """Tests for SafeJSONEncoder class"""

    def test_encoder_handles_nan(self):
        """Test encoder handles NaN values"""
        encoder = SafeJSONEncoder()
        result = encoder.default(float('nan'))
        assert result is None

    def test_encoder_handles_infinity(self):
        """Test encoder handles Infinity values"""
        encoder = SafeJSONEncoder()
        result = encoder.default(float('inf'))
        assert result is None

    def test_encoder_handles_negative_infinity(self):
        """Test encoder handles negative Infinity"""
        encoder = SafeJSONEncoder()
        result = encoder.default(float('-inf'))
        assert result is None

    def test_encoder_handles_datetime(self):
        """Test encoder handles datetime objects"""
        encoder = SafeJSONEncoder()
        dt = datetime.datetime(2025, 11, 5, 12, 0, 0)
        result = encoder.default(dt)
        assert result == '2025-11-05T12:00:00'

    def test_encoder_handles_date(self):
        """Test encoder handles date objects"""
        encoder = SafeJSONEncoder()
        d = datetime.date(2025, 11, 5)
        result = encoder.default(d)
        assert result == '2025-11-05'

    def test_encoder_handles_time(self):
        """Test encoder handles time objects"""
        encoder = SafeJSONEncoder()
        t = datetime.time(12, 30, 45)
        result = encoder.default(t)
        assert result == '12:30:45'

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_encoder_handles_numpy_integer(self):
        """Test encoder handles numpy integer types"""
        encoder = SafeJSONEncoder()
        result = encoder.default(np.int64(42))
        assert result == 42
        assert isinstance(result, int)

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_encoder_handles_numpy_float(self):
        """Test encoder handles numpy float types"""
        encoder = SafeJSONEncoder()
        result = encoder.default(np.float64(3.14))
        assert result == 3.14
        assert isinstance(result, float)

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_encoder_handles_numpy_nan(self):
        """Test encoder handles numpy NaN"""
        encoder = SafeJSONEncoder()
        result = encoder.default(np.float64(np.nan))
        assert result is None

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_encoder_handles_numpy_inf(self):
        """Test encoder handles numpy Inf"""
        encoder = SafeJSONEncoder()
        result = encoder.default(np.float64(np.inf))
        assert result is None

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_encoder_handles_numpy_array(self):
        """Test encoder handles numpy arrays"""
        encoder = SafeJSONEncoder()
        result = encoder.default(np.array([1, 2, 3]))
        assert result == [1, 2, 3]

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_encoder_handles_numpy_bool(self):
        """Test encoder handles numpy boolean"""
        encoder = SafeJSONEncoder()
        result = encoder.default(np.bool_(True))
        assert result is True
        assert isinstance(result, bool)

    def test_encoder_handles_custom_object_as_string(self):
        """Test encoder converts custom objects to strings"""
        encoder = SafeJSONEncoder()

        class CustomObj:
            def __str__(self):
                return "custom_string"

        result = encoder.default(CustomObj())
        assert result == "custom_string"

    def test_encoder_handles_unserializable_object(self):
        """Test encoder handles objects that can't be converted to string"""
        encoder = SafeJSONEncoder()

        class BadObj:
            def __str__(self):
                raise Exception("Cannot convert")

        result = encoder.default(BadObj())
        assert result is None


class TestSafeJSONDumps:
    """Tests for safe_json_dumps function"""

    def test_safe_json_dumps_basic_dict(self):
        """Test safe_json_dumps with basic dictionary"""
        data = {'key': 'value', 'number': 42}
        result = safe_json_dumps(data)

        assert isinstance(result, str)
        loaded = json.loads(result)
        assert loaded == data

    def test_safe_json_dumps_with_nan(self):
        """Test safe_json_dumps handles NaN"""
        data = {'value': float('nan')}
        result = safe_json_dumps(data)

        loaded = json.loads(result)
        assert loaded['value'] is None

    def test_safe_json_dumps_with_indent(self):
        """Test safe_json_dumps with indentation"""
        data = {'key': 'value'}
        result = safe_json_dumps(data, indent=2)

        assert '\n' in result
        assert '  ' in result

    def test_safe_json_dumps_handles_error(self):
        """Test safe_json_dumps handles serialization errors"""
        # Create a data structure that will fail during cleaning
        with patch('deepbridge.core.experiment.report.utils.json_utils.clean_for_json', side_effect=Exception("Test error")):
            result = safe_json_dumps({'key': 'value'})
            assert result == '{}'


class TestSafeJSONLoads:
    """Tests for safe_json_loads function"""

    def test_safe_json_loads_valid_json(self):
        """Test safe_json_loads with valid JSON"""
        json_str = '{"key": "value", "number": 42}'
        result = safe_json_loads(json_str)

        assert result == {'key': 'value', 'number': 42}

    def test_safe_json_loads_invalid_json_raises_error(self):
        """Test safe_json_loads raises error for invalid JSON"""
        with pytest.raises(json.JSONDecodeError):
            safe_json_loads('not valid json')


class TestCleanForJSON:
    """Tests for clean_for_json function"""

    def test_clean_for_json_dict(self):
        """Test clean_for_json with dictionary"""
        data = {'value': float('nan'), 'number': 42}
        result = clean_for_json(data)

        assert result['value'] is None
        assert result['number'] == 42

    def test_clean_for_json_list(self):
        """Test clean_for_json with list"""
        data = [1.0, float('nan'), 3.0]
        result = clean_for_json(data)

        assert result == [1.0, None, 3.0]

    def test_clean_for_json_nested_structure(self):
        """Test clean_for_json with nested structures"""
        data = {
            'metrics': {'score': float('inf'), 'count': 10},
            'values': [1.0, float('nan'), 3.0]
        }
        result = clean_for_json(data)

        assert result['metrics']['score'] is None
        assert result['metrics']['count'] == 10
        assert result['values'] == [1.0, None, 3.0]

    def test_clean_for_json_float_nan(self):
        """Test clean_for_json with NaN float"""
        result = clean_for_json(float('nan'))
        assert result is None

    def test_clean_for_json_float_inf(self):
        """Test clean_for_json with Infinity"""
        result = clean_for_json(float('inf'))
        assert result is None

    def test_clean_for_json_normal_float(self):
        """Test clean_for_json with normal float"""
        result = clean_for_json(3.14)
        assert result == 3.14

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_clean_for_json_numpy_float(self):
        """Test clean_for_json with numpy float"""
        result = clean_for_json(np.float64(3.14))
        assert result == 3.14

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_clean_for_json_numpy_nan(self):
        """Test clean_for_json with numpy NaN"""
        result = clean_for_json(np.float64(np.nan))
        assert result is None

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_clean_for_json_numpy_int(self):
        """Test clean_for_json with numpy integer"""
        result = clean_for_json(np.int64(42))
        assert result == 42

    def test_clean_for_json_other_types(self):
        """Test clean_for_json with other types returns as-is"""
        assert clean_for_json('string') == 'string'
        assert clean_for_json(True) is True
        assert clean_for_json(None) is None


class TestFormatForJavaScript:
    """Tests for format_for_javascript function"""

    def test_format_for_javascript_basic(self):
        """Test format_for_javascript with basic data"""
        data = {'key': 'value', 'number': 42}
        result = format_for_javascript(data)

        assert isinstance(result, str)
        loaded = json.loads(result)
        assert loaded == data

    def test_format_for_javascript_with_nan(self):
        """Test format_for_javascript handles NaN"""
        data = {'value': float('nan')}
        result = format_for_javascript(data)

        loaded = json.loads(result)
        assert loaded['value'] is None


class TestJSONSerializer:
    """Tests for json_serializer function"""

    def test_json_serializer_with_datetime(self):
        """Test json_serializer with datetime"""
        dt = datetime.datetime(2025, 11, 5, 12, 0, 0)
        result = json_serializer(dt)

        assert result == '2025-11-05T12:00:00'

    def test_json_serializer_with_nan(self):
        """Test json_serializer with NaN"""
        result = json_serializer(float('nan'))
        assert result is None


class TestPrepareDataForTemplate:
    """Tests for prepare_data_for_template function"""

    def test_prepare_data_for_template_structure(self):
        """Test prepare_data_for_template returns correct structure"""
        data = {'model_name': 'MyModel', 'metrics': {'accuracy': 0.95}}
        result = prepare_data_for_template(data, 'uncertainty')

        assert 'data' in result
        assert 'data_json' in result
        assert 'test_type' in result

    def test_prepare_data_for_template_preserves_data(self):
        """Test prepare_data_for_template preserves original data"""
        data = {'model_name': 'MyModel'}
        result = prepare_data_for_template(data, 'robustness')

        assert result['data'] == data
        assert result['test_type'] == 'robustness'

    def test_prepare_data_for_template_creates_json(self):
        """Test prepare_data_for_template creates JSON string"""
        data = {'value': 42}
        result = prepare_data_for_template(data, 'test')

        assert isinstance(result['data_json'], str)
        loaded = json.loads(result['data_json'])
        assert loaded == data


class TestIntegration:
    """Integration tests for JSON utilities"""

    def test_full_serialization_workflow(self):
        """Test complete serialization workflow"""
        data = {
            'model_name': 'TestModel',
            'metrics': {
                'accuracy': 0.95,
                'loss': float('nan'),
                'score': float('inf')
            },
            'timestamp': datetime.datetime(2025, 11, 5),
            'values': [1.0, 2.0, float('nan'), 4.0]
        }

        # Serialize
        json_str = safe_json_dumps(data, indent=2)

        # Deserialize
        loaded = safe_json_loads(json_str)

        # Check values
        assert loaded['model_name'] == 'TestModel'
        assert loaded['metrics']['accuracy'] == 0.95
        assert loaded['metrics']['loss'] is None
        assert loaded['metrics']['score'] is None
        assert loaded['values'] == [1.0, 2.0, None, 4.0]

    @pytest.mark.skipif(not HAS_NUMPY, reason="NumPy not available")
    def test_numpy_integration(self):
        """Test integration with numpy types"""
        data = {
            'int_val': np.int64(42),
            'float_val': np.float64(3.14),
            'nan_val': np.float64(np.nan),
            'array': np.array([1, 2, 3]),
            'bool_val': np.bool_(True)
        }

        json_str = safe_json_dumps(data)
        loaded = safe_json_loads(json_str)

        assert loaded['int_val'] == 42
        assert loaded['float_val'] == 3.14
        assert loaded['nan_val'] is None
        assert loaded['array'] == [1, 2, 3]
        assert loaded['bool_val'] is True

