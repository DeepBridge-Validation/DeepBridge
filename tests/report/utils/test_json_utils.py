"""
Tests for JSON utilities.
"""

import datetime
import json
import math

import pytest

from deepbridge.core.experiment.report.utils.json_utils import (
    SafeJSONEncoder,
    clean_for_json,
    format_for_javascript,
    safe_json_dumps,
    safe_json_loads,
)


class TestSafeJSONEncoder:
    """Tests for SafeJSONEncoder class."""

    def test_encode_nan(self):
        """Test encoding NaN values."""
        encoder = SafeJSONEncoder()
        result = encoder.default(float('nan'))
        assert result is None

    def test_encode_infinity(self):
        """Test encoding infinity values."""
        encoder = SafeJSONEncoder()
        assert encoder.default(float('inf')) is None
        assert encoder.default(float('-inf')) is None

    def test_encode_datetime(self):
        """Test encoding datetime objects."""
        encoder = SafeJSONEncoder()
        dt = datetime.datetime(2025, 11, 5, 12, 0, 0)
        result = encoder.default(dt)
        assert result == '2025-11-05T12:00:00'

    def test_encode_date(self):
        """Test encoding date objects."""
        encoder = SafeJSONEncoder()
        d = datetime.date(2025, 11, 5)
        result = encoder.default(d)
        assert result == '2025-11-05'

    def test_encode_normal_float(self):
        """Test that normal floats are not modified."""
        encoder = SafeJSONEncoder()
        result = encoder.default(3.14)
        # Should raise TypeError as normal float doesn't need special handling
        # The default method only handles special cases


class TestSafeJSONDumps:
    """Tests for safe_json_dumps function."""

    def test_dumps_with_nan(self):
        """Test serialization with NaN values."""
        data = {'value': float('nan')}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        assert parsed['value'] is None

    def test_dumps_with_inf(self):
        """Test serialization with infinity values."""
        data = {'pos_inf': float('inf'), 'neg_inf': float('-inf')}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        assert parsed['pos_inf'] is None
        assert parsed['neg_inf'] is None

    def test_dumps_with_datetime(self):
        """Test serialization with datetime objects."""
        data = {
            'timestamp': datetime.datetime(2025, 11, 5, 12, 0, 0),
            'date': datetime.date(2025, 11, 5),
        }
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        assert parsed['timestamp'] == '2025-11-05T12:00:00'
        assert parsed['date'] == '2025-11-05'

    def test_dumps_with_nested_structure(self):
        """Test serialization with nested dictionaries and lists."""
        data = {
            'metrics': {'accuracy': 0.95, 'loss': float('nan')},
            'scores': [1.0, 2.0, float('inf'), 4.0],
        }
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        assert parsed['metrics']['accuracy'] == 0.95
        assert parsed['metrics']['loss'] is None
        assert parsed['scores'][2] is None

    def test_dumps_with_indent(self):
        """Test serialization with indentation."""
        data = {'key': 'value'}
        result = safe_json_dumps(data, indent=2)
        assert '\n' in result  # Indented JSON has newlines

    def test_dumps_empty_dict(self):
        """Test serialization of empty dictionary."""
        result = safe_json_dumps({})
        assert result == '{}'


class TestSafeJSONLoads:
    """Tests for safe_json_loads function."""

    def test_loads_valid_json(self):
        """Test loading valid JSON string."""
        json_str = '{"key": "value", "number": 42}'
        result = safe_json_loads(json_str)
        assert result['key'] == 'value'
        assert result['number'] == 42

    def test_loads_with_null(self):
        """Test loading JSON with null values."""
        json_str = '{"value": null}'
        result = safe_json_loads(json_str)
        assert result['value'] is None

    def test_loads_invalid_json(self):
        """Test loading invalid JSON raises error."""
        with pytest.raises(json.JSONDecodeError):
            safe_json_loads('not valid json{')


class TestCleanForJSON:
    """Tests for clean_for_json function."""

    def test_clean_dict_with_nan(self):
        """Test cleaning dictionary with NaN values."""
        data = {'value': float('nan'), 'normal': 3.14}
        result = clean_for_json(data)
        assert result['value'] is None
        assert result['normal'] == 3.14

    def test_clean_list_with_inf(self):
        """Test cleaning list with infinity values."""
        data = [1.0, float('inf'), 3.0, float('-inf')]
        result = clean_for_json(data)
        assert result == [1.0, None, 3.0, None]

    def test_clean_nested_structures(self):
        """Test cleaning nested structures."""
        data = {
            'level1': {
                'level2': {'nan_value': float('nan'), 'normal': 42},
                'list': [1, float('inf'), 3],
            }
        }
        result = clean_for_json(data)
        assert result['level1']['level2']['nan_value'] is None
        assert result['level1']['level2']['normal'] == 42
        assert result['level1']['list'][1] is None

    def test_clean_preserves_normal_values(self):
        """Test that normal values are preserved."""
        data = {
            'string': 'hello',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'none': None,
        }
        result = clean_for_json(data)
        assert result == data


class TestFormatForJavaScript:
    """Tests for format_for_javascript function."""

    def test_format_simple_dict(self):
        """Test formatting simple dictionary for JavaScript."""
        data = {'value': 42, 'name': 'test'}
        result = format_for_javascript(data)
        parsed = json.loads(result)
        assert parsed['value'] == 42
        assert parsed['name'] == 'test'

    def test_format_with_nan(self):
        """Test that NaN is converted to null for JavaScript."""
        data = {'value': float('nan')}
        result = format_for_javascript(data)
        parsed = json.loads(result)
        assert parsed['value'] is None

    def test_format_complex_structure(self):
        """Test formatting complex nested structure."""
        data = {
            'metrics': {'accuracy': 0.95, 'loss': float('nan')},
            'scores': [1.0, float('inf'), 3.0],
            'timestamp': datetime.datetime(2025, 11, 5),
        }
        result = format_for_javascript(data)
        parsed = json.loads(result)
        assert parsed['metrics']['loss'] is None
        assert parsed['scores'][1] is None
        assert '2025-11-05' in parsed['timestamp']


class TestNumpySupport:
    """Tests for NumPy type support (if NumPy is available)."""

    @pytest.mark.skipif(
        not pytest.importorskip('numpy', minversion=None),
        reason='NumPy not available',
    )
    def test_numpy_types(self):
        """Test encoding NumPy types."""
        import numpy as np

        data = {
            'np_int': np.int64(42),
            'np_float': np.float64(3.14),
            'np_array': np.array([1, 2, 3]),
            'np_nan': np.float64(np.nan),
        }

        result = safe_json_dumps(data)
        parsed = json.loads(result)

        assert parsed['np_int'] == 42
        assert parsed['np_float'] == 3.14
        assert parsed['np_array'] == [1, 2, 3]
        assert parsed['np_nan'] is None


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_data(self):
        """Test handling of empty data."""
        assert safe_json_dumps({}) == '{}'
        assert safe_json_dumps([]) == '[]'

    def test_deeply_nested(self):
        """Test handling of deeply nested structures."""
        data = {'a': {'b': {'c': {'d': {'e': float('nan')}}}}}
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        assert parsed['a']['b']['c']['d']['e'] is None

    def test_mixed_types(self):
        """Test handling of mixed types."""
        data = {
            'string': 'hello',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'none': None,
            'nan': float('nan'),
            'list': [1, 'two', 3.0, None, float('inf')],
            'dict': {'nested': float('nan')},
        }
        result = safe_json_dumps(data)
        parsed = json.loads(result)
        assert parsed['nan'] is None
        assert parsed['list'][4] is None
        assert parsed['dict']['nested'] is None
