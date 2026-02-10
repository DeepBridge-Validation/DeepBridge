"""
Comprehensive tests for DataTypeConverter.

This test suite validates:
1. convert_numpy_types - numpy to Python native type conversion
2. json_serializer - JSON serialization for special types
3. Handling of NaN, Inf, and special float values
4. Nested data structure conversion
5. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import patch, Mock
import datetime
import math

from deepbridge.core.experiment.report.utils.converters import DataTypeConverter


# ==================== Fixtures ====================


@pytest.fixture
def converter():
    """Create DataTypeConverter instance"""
    return DataTypeConverter()


@pytest.fixture
def converter_without_numpy():
    """Create converter without numpy"""
    converter = DataTypeConverter()
    converter.np = None  # Manually set np to None
    return converter


# ==================== Initialization Tests ====================


class TestInitialization:
    """Tests for converter initialization"""

    def test_initialization_with_numpy(self, converter):
        """Test initialization when numpy is available"""
        # Should have np attribute
        assert hasattr(converter, 'np')
        # May or may not be None depending on environment
        # Just verify it doesn't crash

    @patch('deepbridge.core.experiment.report.utils.converters.logger')
    def test_initialization_without_numpy(self, mock_logger):
        """Test initialization when numpy is not available"""
        with patch.dict('sys.modules', {'numpy': None}):
            with patch('builtins.__import__', side_effect=ImportError("No numpy")):
                converter = DataTypeConverter()

                assert converter.np is None
                mock_logger.warning.assert_called_once()


# ==================== convert_numpy_types - Basic Types Tests ====================


class TestConvertBasicTypes:
    """Tests for converting basic Python types"""

    def test_convert_int(self, converter):
        """Test converting int (should remain unchanged)"""
        result = converter.convert_numpy_types(42)
        assert result == 42
        assert isinstance(result, int)

    def test_convert_float(self, converter):
        """Test converting float (should remain unchanged)"""
        result = converter.convert_numpy_types(3.14)
        assert result == 3.14
        assert isinstance(result, float)

    def test_convert_string(self, converter):
        """Test converting string (should remain unchanged)"""
        result = converter.convert_numpy_types("test")
        assert result == "test"
        assert isinstance(result, str)

    def test_convert_none(self, converter):
        """Test converting None (should remain unchanged)"""
        result = converter.convert_numpy_types(None)
        assert result is None

    def test_convert_boolean(self, converter):
        """Test converting boolean (should remain unchanged)"""
        result = converter.convert_numpy_types(True)
        assert result is True
        assert isinstance(result, bool)


# ==================== convert_numpy_types - Container Types Tests ====================


class TestConvertContainers:
    """Tests for converting container types"""

    def test_convert_dict(self, converter):
        """Test converting dict"""
        data = {'a': 1, 'b': 2, 'c': 3}
        result = converter.convert_numpy_types(data)

        assert result == {'a': 1, 'b': 2, 'c': 3}
        assert isinstance(result, dict)

    def test_convert_nested_dict(self, converter):
        """Test converting nested dict"""
        data = {'outer': {'inner': {'deep': 42}}}
        result = converter.convert_numpy_types(data)

        assert result == {'outer': {'inner': {'deep': 42}}}

    def test_convert_list(self, converter):
        """Test converting list"""
        data = [1, 2, 3, 4, 5]
        result = converter.convert_numpy_types(data)

        assert result == [1, 2, 3, 4, 5]
        assert isinstance(result, list)

    def test_convert_nested_list(self, converter):
        """Test converting nested list"""
        data = [[1, 2], [3, 4], [5, 6]]
        result = converter.convert_numpy_types(data)

        assert result == [[1, 2], [3, 4], [5, 6]]

    def test_convert_tuple(self, converter):
        """Test converting tuple"""
        data = (1, 2, 3)
        result = converter.convert_numpy_types(data)

        assert result == (1, 2, 3)
        assert isinstance(result, tuple)

    def test_convert_nested_tuple(self, converter):
        """Test converting nested tuple"""
        data = ((1, 2), (3, 4))
        result = converter.convert_numpy_types(data)

        assert result == ((1, 2), (3, 4))

    def test_convert_mixed_containers(self, converter):
        """Test converting mixed nested containers"""
        data = {
            'list': [1, 2, 3],
            'tuple': (4, 5, 6),
            'dict': {'nested': [7, 8, 9]}
        }
        result = converter.convert_numpy_types(data)

        assert result == data


# ==================== convert_numpy_types - DateTime Tests ====================


class TestConvertDateTime:
    """Tests for converting datetime types"""

    def test_convert_datetime(self, converter):
        """Test converting datetime to ISO format"""
        dt = datetime.datetime(2025, 6, 15, 14, 30, 0)
        result = converter.convert_numpy_types(dt)

        assert result == '2025-06-15T14:30:00'
        assert isinstance(result, str)

    def test_convert_date(self, converter):
        """Test converting date to ISO format"""
        dt = datetime.date(2025, 6, 15)
        result = converter.convert_numpy_types(dt)

        assert result == '2025-06-15'
        assert isinstance(result, str)

    def test_convert_datetime_in_dict(self, converter):
        """Test converting datetime within dict"""
        dt = datetime.datetime(2025, 1, 1, 0, 0, 0)
        data = {'timestamp': dt, 'value': 42}
        result = converter.convert_numpy_types(data)

        assert result['timestamp'] == '2025-01-01T00:00:00'
        assert result['value'] == 42


# ==================== convert_numpy_types - Special Float Values Tests ====================


class TestConvertSpecialFloats:
    """Tests for converting NaN and Inf"""

    def test_convert_nan(self, converter):
        """Test converting NaN to None"""
        result = converter.convert_numpy_types(float('nan'))
        assert result is None

    def test_convert_inf(self, converter):
        """Test converting Inf to None"""
        result = converter.convert_numpy_types(float('inf'))
        assert result is None

    def test_convert_negative_inf(self, converter):
        """Test converting -Inf to None"""
        result = converter.convert_numpy_types(float('-inf'))
        assert result is None

    def test_convert_nan_in_list(self, converter):
        """Test converting NaN within list"""
        data = [1.0, float('nan'), 3.0]
        result = converter.convert_numpy_types(data)

        assert result == [1.0, None, 3.0]

    def test_convert_inf_in_dict(self, converter):
        """Test converting Inf within dict"""
        data = {'a': 1.0, 'b': float('inf'), 'c': float('-inf')}
        result = converter.convert_numpy_types(data)

        assert result == {'a': 1.0, 'b': None, 'c': None}


# ==================== convert_numpy_types - NumPy Types Tests ====================


class TestConvertNumPyTypes:
    """Tests for converting numpy types (requires numpy)"""

    @pytest.mark.skipif(True, reason="NumPy types tested separately if available")
    def test_convert_numpy_int(self, converter):
        """Test converting numpy integer types"""
        # This would test np.int64, np.int32, etc.
        pass

    @pytest.mark.skipif(True, reason="NumPy types tested separately if available")
    def test_convert_numpy_float(self, converter):
        """Test converting numpy float types"""
        # This would test np.float64, np.float32, etc.
        pass

    @pytest.mark.skipif(True, reason="NumPy types tested separately if available")
    def test_convert_numpy_array(self, converter):
        """Test converting numpy arrays"""
        # This would test np.ndarray conversion
        pass


# ==================== convert_numpy_types - Without NumPy Tests ====================


class TestConvertWithoutNumPy:
    """Tests for converter behavior when numpy is not available"""

    def test_convert_without_numpy_returns_unchanged(self, converter_without_numpy):
        """Test that data is returned unchanged when numpy is None"""
        # When numpy is None, the function returns data unchanged
        data = {'a': 1, 'b': [2, 3], 'c': (4, 5)}
        result = converter_without_numpy.convert_numpy_types(data)
        assert result == data

        # DateTime is also returned unchanged (not converted)
        dt = datetime.datetime(2025, 1, 1, 0, 0, 0)
        result = converter_without_numpy.convert_numpy_types(dt)
        assert result == dt

        # Special floats are also returned unchanged
        data_with_nan = float('nan')
        result = converter_without_numpy.convert_numpy_types(data_with_nan)
        # Can't compare NaN directly, but it should be returned as-is
        assert math.isnan(result)


# ==================== json_serializer Tests ====================


class TestJsonSerializer:
    """Tests for json_serializer method"""

    def test_serialize_datetime(self, converter):
        """Test serializing datetime"""
        dt = datetime.datetime(2025, 6, 15, 14, 30, 0)
        result = converter.json_serializer(dt)

        assert result == '2025-06-15T14:30:00'

    def test_serialize_date(self, converter):
        """Test serializing date"""
        dt = datetime.date(2025, 6, 15)
        result = converter.json_serializer(dt)

        assert result == '2025-06-15'

    def test_serialize_nan(self, converter):
        """Test serializing NaN"""
        result = converter.json_serializer(float('nan'))
        assert result is None

    def test_serialize_inf(self, converter):
        """Test serializing Inf"""
        result = converter.json_serializer(float('inf'))
        assert result is None

    def test_serialize_negative_inf(self, converter):
        """Test serializing -Inf"""
        result = converter.json_serializer(float('-inf'))
        assert result is None

    def test_serialize_unsupported_type_raises_error(self, converter):
        """Test that unsupported types raise TypeError"""
        class CustomClass:
            pass

        with pytest.raises(TypeError, match='not serializable'):
            converter.json_serializer(CustomClass())

    def test_serialize_int_raises_error(self, converter):
        """Test that int raises TypeError (not handled by serializer)"""
        with pytest.raises(TypeError):
            converter.json_serializer(42)

    def test_serialize_string_raises_error(self, converter):
        """Test that string raises TypeError (not handled by serializer)"""
        with pytest.raises(TypeError):
            converter.json_serializer("test")


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_complex_nested_structure(self, converter):
        """Test converting complex nested structure"""
        data = {
            'metadata': {
                'timestamp': datetime.datetime(2025, 1, 1, 0, 0, 0),
                'version': '1.0'
            },
            'results': [
                {'score': 0.85, 'name': 'Model1'},
                {'score': 0.92, 'name': 'Model2'},
                {'score': float('nan'), 'name': 'Model3'}
            ],
            'stats': {
                'mean': 0.885,
                'max': 0.92,
                'min': 0.85,
                'invalid': float('inf')
            },
            'config': (1, 2, 3)
        }

        result = converter.convert_numpy_types(data)

        assert result['metadata']['timestamp'] == '2025-01-01T00:00:00'
        assert result['metadata']['version'] == '1.0'
        assert len(result['results']) == 3
        assert result['results'][2]['score'] is None  # NaN converted
        assert result['stats']['invalid'] is None  # Inf converted
        assert result['config'] == (1, 2, 3)

    def test_deeply_nested_containers(self, converter):
        """Test converting deeply nested containers"""
        data = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'value': 42,
                            'list': [1, 2, float('nan')],
                            'tuple': (float('inf'), 5, 6)
                        }
                    }
                }
            }
        }

        result = converter.convert_numpy_types(data)

        deep_level = result['level1']['level2']['level3']['level4']
        assert deep_level['value'] == 42
        assert deep_level['list'] == [1, 2, None]
        assert deep_level['tuple'] == (None, 5, 6)

    def test_mixed_types_with_special_values(self, converter):
        """Test converting mixed types including special values"""
        data = {
            'integers': [1, 2, 3],
            'floats': [1.5, float('nan'), 2.5, float('inf')],
            'strings': ['a', 'b', 'c'],
            'bools': [True, False, True],
            'dates': [datetime.date(2025, 1, 1), datetime.date(2025, 12, 31)],
            'nested': {
                'tuple': (float('nan'), 'test', 42),
                'list': [float('inf'), None, 3.14]
            }
        }

        result = converter.convert_numpy_types(data)

        assert result['integers'] == [1, 2, 3]
        assert result['floats'] == [1.5, None, 2.5, None]
        assert result['strings'] == ['a', 'b', 'c']
        assert result['bools'] == [True, False, True]
        assert result['dates'] == ['2025-01-01', '2025-12-31']
        assert result['nested']['tuple'] == (None, 'test', 42)
        assert result['nested']['list'] == [None, None, 3.14]


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_empty_dict(self, converter):
        """Test converting empty dict"""
        result = converter.convert_numpy_types({})
        assert result == {}

    def test_empty_list(self, converter):
        """Test converting empty list"""
        result = converter.convert_numpy_types([])
        assert result == []

    def test_empty_tuple(self, converter):
        """Test converting empty tuple"""
        result = converter.convert_numpy_types(())
        assert result == ()

    def test_single_element_containers(self, converter):
        """Test converting single element containers"""
        assert converter.convert_numpy_types([42]) == [42]
        assert converter.convert_numpy_types((42,)) == (42,)
        assert converter.convert_numpy_types({'key': 42}) == {'key': 42}

    def test_dict_with_none_values(self, converter):
        """Test dict with None values"""
        data = {'a': None, 'b': 1, 'c': None}
        result = converter.convert_numpy_types(data)
        assert result == {'a': None, 'b': 1, 'c': None}

    def test_list_with_all_special_values(self, converter):
        """Test list with only special float values"""
        data = [float('nan'), float('inf'), float('-inf')]
        result = converter.convert_numpy_types(data)
        assert result == [None, None, None]

    def test_very_large_float(self, converter):
        """Test very large float value"""
        data = 1e308
        result = converter.convert_numpy_types(data)
        assert result == 1e308

    def test_very_small_float(self, converter):
        """Test very small float value"""
        data = 1e-308
        result = converter.convert_numpy_types(data)
        assert result == 1e-308

    def test_negative_zero(self, converter):
        """Test negative zero"""
        data = -0.0
        result = converter.convert_numpy_types(data)
        assert result == -0.0

    def test_unicode_strings_in_dict(self, converter):
        """Test dict with unicode strings"""
        data = {'name': 'José', 'city': '北京', 'emoji': '🎉'}
        result = converter.convert_numpy_types(data)
        assert result == data

    def test_nested_datetime_in_tuple_in_list(self, converter):
        """Test deeply nested datetime"""
        dt = datetime.datetime(2025, 1, 1, 0, 0, 0)
        data = [(dt, 'value'), (dt, 'value2')]
        result = converter.convert_numpy_types(data)

        assert result == [('2025-01-01T00:00:00', 'value'), ('2025-01-01T00:00:00', 'value2')]


def test_convert_numpy_types_float_with_specific_types():
    """Test convert_numpy_types with specific float types (float16, float32, float64)."""
    import numpy as np
    from deepbridge.core.experiment.report.utils.converters import DataTypeConverter
    
    converter = DataTypeConverter()
    
    # Test float32
    val32 = np.float32(3.14159)
    result = converter.convert_numpy_types(val32)
    assert isinstance(result, float)
    assert abs(result - 3.14159) < 0.001
    
    # Test float64
    val64 = np.float64(2.71828)
    result = converter.convert_numpy_types(val64)
    assert isinstance(result, float)
    assert abs(result - 2.71828) < 0.001


def test_convert_numpy_types_float_with_inf():
    """Test convert_numpy_types with Inf values in specific float types."""
    import numpy as np
    from deepbridge.core.experiment.report.utils.converters import DataTypeConverter
    
    converter = DataTypeConverter()
    
    # Test float32 with Inf
    inf32 = np.float32(np.inf)
    result = converter.convert_numpy_types(inf32)
    assert result is None
    
    # Test float64 with NaN
    nan64 = np.float64(np.nan)
    result = converter.convert_numpy_types(nan64)
    assert result is None


def test_convert_numpy_types_int_types():
    """Test convert_numpy_types with specific int types."""
    import numpy as np
    from deepbridge.core.experiment.report.utils.converters import DataTypeConverter
    
    converter = DataTypeConverter()
    
    # Test int32
    val32 = np.int32(42)
    result = converter.convert_numpy_types(val32)
    assert result == 42
    assert isinstance(result, int)
    
    # Test int64
    val64 = np.int64(100)
    result = converter.convert_numpy_types(val64)
    assert result == 100
    assert isinstance(result, int)


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
