"""
Tests for DataTypeConverter utility.

Coverage Target: Cover numpy type conversion and JSON serialization.
"""

import pytest
import datetime
import math


class TestDataTypeConverter:
    """Tests for DataTypeConverter class."""

    @pytest.fixture
    def converter(self):
        """Create DataTypeConverter instance."""
        from deepbridge.core.experiment.report.utils.converters import DataTypeConverter
        return DataTypeConverter()

    def test_initialization(self, converter):
        """Test converter initialization."""
        assert converter.np is not None

    def test_convert_dict(self, converter):
        """Test converting dictionary with numpy types."""
        import numpy as np

        data = {
            'int_val': np.int32(42),
            'float_val': np.float64(3.14),
            'nested': {'inner': np.int64(100)}
        }

        result = converter.convert_numpy_types(data)
        assert result['int_val'] == 42
        assert isinstance(result['int_val'], int)
        assert result['float_val'] == 3.14
        assert isinstance(result['float_val'], float)
        assert result['nested']['inner'] == 100

    def test_convert_list(self, converter):
        """Test converting list with numpy types."""
        import numpy as np

        data = [np.int32(1), np.float64(2.5), np.int64(3)]
        result = converter.convert_numpy_types(data)

        assert result == [1, 2.5, 3]
        assert all(isinstance(x, (int, float)) for x in result)

    def test_convert_tuple(self, converter):
        """Test converting tuple with numpy types."""
        import numpy as np

        data = (np.int32(1), np.float64(2.5), np.int64(3))
        result = converter.convert_numpy_types(data)

        assert result == (1, 2.5, 3)
        assert isinstance(result, tuple)

    def test_convert_datetime(self, converter):
        """Test converting datetime objects."""
        dt = datetime.datetime(2024, 1, 15, 10, 30)
        result = converter.convert_numpy_types(dt)

        assert isinstance(result, str)
        assert '2024-01-15' in result

    def test_convert_date(self, converter):
        """Test converting date objects."""
        d = datetime.date(2024, 1, 15)
        result = converter.convert_numpy_types(d)

        assert isinstance(result, str)
        assert result == '2024-01-15'

    def test_convert_numpy_integer_types(self, converter):
        """Test converting various numpy integer types."""
        import numpy as np

        int_types = [
            np.int8(10),
            np.int16(20),
            np.int32(30),
            np.int64(40),
        ]

        for val in int_types:
            result = converter.convert_numpy_types(val)
            assert isinstance(result, int)

    def test_convert_numpy_float_types(self, converter):
        """Test converting various numpy float types."""
        import numpy as np

        float_types = [
            np.float16(1.5),
            np.float32(2.5),
            np.float64(3.5),
        ]

        for val in float_types:
            result = converter.convert_numpy_types(val)
            assert isinstance(result, float)

    def test_convert_numpy_nan(self, converter):
        """Test converting numpy NaN values."""
        import numpy as np

        result = converter.convert_numpy_types(np.float64(np.nan))
        assert result is None

    def test_convert_numpy_inf(self, converter):
        """Test converting numpy infinity values."""
        import numpy as np

        result = converter.convert_numpy_types(np.float64(np.inf))
        assert result is None

    def test_convert_python_nan(self, converter):
        """Test converting Python NaN values."""
        result = converter.convert_numpy_types(float('nan'))
        assert result is None

    def test_convert_python_inf(self, converter):
        """Test converting Python infinity values."""
        result = converter.convert_numpy_types(float('inf'))
        assert result is None

    def test_convert_numpy_array(self, converter):
        """Test converting numpy arrays."""
        import numpy as np

        arr = np.array([1, 2, 3, 4, 5])
        result = converter.convert_numpy_types(arr)

        assert result == [1, 2, 3, 4, 5]
        assert isinstance(result, list)

    def test_convert_numpy_array_with_nan(self, converter):
        """Test converting numpy arrays containing NaN."""
        import numpy as np

        arr = np.array([1.0, 2.0, np.nan, 4.0])
        result = converter.convert_numpy_types(arr)

        assert result[0] == 1.0
        assert result[1] == 2.0
        assert result[2] is None
        assert result[3] == 4.0

    def test_convert_nested_structure(self, converter):
        """Test converting deeply nested structures."""
        import numpy as np

        data = {
            'level1': {
                'level2': {
                    'array': np.array([1, 2, 3]),
                    'value': np.int32(42)
                }
            },
            'list': [np.float64(1.5), {'inner': np.int64(100)}]
        }

        result = converter.convert_numpy_types(data)
        assert result['level1']['level2']['array'] == [1, 2, 3]
        assert result['level1']['level2']['value'] == 42
        assert result['list'][0] == 1.5
        assert result['list'][1]['inner'] == 100

    def test_json_serializer_datetime(self, converter):
        """Test JSON serializer with datetime."""
        dt = datetime.datetime(2024, 1, 15, 10, 30)
        result = converter.json_serializer(dt)

        assert isinstance(result, str)
        assert '2024-01-15' in result

    def test_json_serializer_date(self, converter):
        """Test JSON serializer with date."""
        d = datetime.date(2024, 1, 15)
        result = converter.json_serializer(d)

        assert result == '2024-01-15'

    def test_json_serializer_nan(self, converter):
        """Test JSON serializer with NaN."""
        result = converter.json_serializer(float('nan'))
        assert result is None

    def test_json_serializer_inf(self, converter):
        """Test JSON serializer with infinity."""
        result = converter.json_serializer(float('inf'))
        assert result is None

    def test_json_serializer_unsupported_type(self, converter):
        """Test JSON serializer raises error for unsupported types."""
        with pytest.raises(TypeError, match='not serializable'):
            converter.json_serializer(object())

    def test_convert_regular_types_unchanged(self, converter):
        """Test that regular Python types pass through unchanged."""
        data = {
            'string': 'hello',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'none': None
        }

        result = converter.convert_numpy_types(data)
        assert result == data

    def test_convert_empty_containers(self, converter):
        """Test converting empty containers."""
        assert converter.convert_numpy_types({}) == {}
        assert converter.convert_numpy_types([]) == []
        assert converter.convert_numpy_types(()) == ()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
