"""
Comprehensive tests for DataTransformer base class.

This test suite validates:
1. __init__ - initialization with/without numpy
2. transform - base transformation logic
3. _deep_copy - deep copy with special type handling
4. convert_numpy_types - numpy type conversion

Coverage Target: ~95%+
"""

import pytest
import datetime
from unittest.mock import patch, Mock
from decimal import Decimal

from deepbridge.core.experiment.report.base import DataTransformer


# ==================== Fixtures ====================


@pytest.fixture
def transformer():
    """Create a DataTransformer instance"""
    return DataTransformer()


@pytest.fixture
def transformer_no_numpy():
    """Create a DataTransformer instance without numpy"""
    # Create transformer and manually set np to None
    transformer = DataTransformer()
    transformer.np = None
    return transformer


@pytest.fixture
def sample_results():
    """Sample experiment results"""
    return {
        'accuracy': 0.95,
        'precision': 0.92,
        'recall': 0.89,
        'metadata': {
            'dataset': 'test_data',
            'params': {'learning_rate': 0.01}
        }
    }


# ==================== Initialization Tests ====================


class TestInitialization:
    """Tests for DataTransformer initialization"""

    def test_init_with_numpy_available(self, transformer):
        """Test initialization when numpy is available"""
        # Should import numpy
        assert transformer.np is not None
        assert hasattr(transformer.np, 'ndarray')

    def test_init_without_numpy(self, transformer_no_numpy):
        """Test initialization when numpy is not available"""
        assert transformer_no_numpy.np is None


# ==================== transform Tests ====================


class TestTransform:
    """Tests for transform method"""

    def test_transform_basic(self, transformer, sample_results):
        """Test basic transform with model name"""
        result = transformer.transform(
            results=sample_results,
            model_name='test_model'
        )

        assert result['model_name'] == 'test_model'
        assert 'timestamp' in result
        assert result['accuracy'] == 0.95

    def test_transform_with_timestamp(self, transformer, sample_results):
        """Test transform with provided timestamp"""
        custom_timestamp = '2026-01-01 12:00:00'

        result = transformer.transform(
            results=sample_results,
            model_name='test_model',
            timestamp=custom_timestamp
        )

        assert result['timestamp'] == custom_timestamp

    def test_transform_generates_timestamp_when_none(self, transformer, sample_results):
        """Test that transform generates timestamp when not provided"""
        result = transformer.transform(
            results=sample_results,
            model_name='test_model',
            timestamp=None
        )

        assert 'timestamp' in result
        assert result['timestamp'] is not None
        # Verify format
        datetime.datetime.strptime(result['timestamp'], '%Y-%m-%d %H:%M:%S')

    def test_transform_preserves_existing_model_name(self, transformer):
        """Test that existing model_name in results is preserved"""
        results = {
            'model_name': 'existing_model',
            'accuracy': 0.9
        }

        result = transformer.transform(
            results=results,
            model_name='new_model'
        )

        # Should keep existing model_name
        assert result['model_name'] == 'existing_model'

    def test_transform_preserves_existing_timestamp(self, transformer):
        """Test that existing timestamp in results is preserved"""
        existing_timestamp = '2025-12-31 23:59:59'
        results = {
            'timestamp': existing_timestamp,
            'accuracy': 0.9
        }

        result = transformer.transform(
            results=results,
            model_name='test_model',
            timestamp='2026-01-01 00:00:00'
        )

        # Should keep existing timestamp
        assert result['timestamp'] == existing_timestamp

    def test_transform_creates_deep_copy(self, transformer, sample_results):
        """Test that transform creates deep copy of results"""
        result = transformer.transform(
            results=sample_results,
            model_name='test_model'
        )

        # Modify original
        sample_results['accuracy'] = 0.5
        sample_results['metadata']['dataset'] = 'modified'

        # Result should be unchanged
        assert result['accuracy'] == 0.95
        assert result['metadata']['dataset'] == 'test_data'

    def test_transform_calls_convert_numpy_types(self, transformer, sample_results):
        """Test that transform calls convert_numpy_types"""
        with patch.object(transformer, 'convert_numpy_types', side_effect=lambda x: x) as mock_convert:
            result = transformer.transform(
                results=sample_results,
                model_name='test_model'
            )

            mock_convert.assert_called_once()


# ==================== _deep_copy Tests ====================


class TestDeepCopy:
    """Tests for _deep_copy method"""

    def test_deep_copy_dict(self, transformer):
        """Test deep copy of dictionary"""
        original = {'a': 1, 'b': {'c': 2}}

        copied = transformer._deep_copy(original)

        assert copied == original
        assert copied is not original
        assert copied['b'] is not original['b']

    def test_deep_copy_list(self, transformer):
        """Test deep copy of list"""
        original = [1, 2, [3, 4]]

        copied = transformer._deep_copy(original)

        assert copied == original
        assert copied is not original
        assert copied[2] is not original[2]

    def test_deep_copy_tuple(self, transformer):
        """Test deep copy of tuple"""
        original = (1, 2, (3, 4))

        copied = transformer._deep_copy(original)

        assert copied == original

    def test_deep_copy_set(self, transformer):
        """Test deep copy of set"""
        original = {1, 2, 3}

        copied = transformer._deep_copy(original)

        assert copied == original
        assert copied is not original

    def test_deep_copy_nested_structures(self, transformer):
        """Test deep copy of nested structures"""
        original = {
            'list': [1, 2, {'nested': 'dict'}],
            'tuple': (1, [2, 3]),
            'set': {4, 5}
        }

        copied = transformer._deep_copy(original)

        assert copied == original
        # Modify original
        original['list'][2]['nested'] = 'modified'
        # Copied should be unchanged
        assert copied['list'][2]['nested'] == 'dict'

    def test_deep_copy_with_numpy_array(self, transformer):
        """Test deep copy of numpy array"""
        import numpy as np

        original = {'data': np.array([1, 2, 3])}

        copied = transformer._deep_copy(original)

        # Deep copy preserves type (array), convert_numpy_types handles conversion
        assert isinstance(copied['data'], np.ndarray)
        assert list(copied['data']) == [1, 2, 3]

    def test_deep_copy_fallback_on_error(self, transformer):
        """Test fallback when deepcopy fails"""
        # Create object that can't be deep copied
        class UncopiableDict(dict):
            def __deepcopy__(self, memo):
                raise TypeError("Cannot deepcopy")

        original = UncopiableDict({'a': 1, 'b': 2})

        copied = transformer._deep_copy(original)

        # Should use manual copy
        assert copied == {'a': 1, 'b': 2}

    def test_deep_copy_primitives(self, transformer):
        """Test deep copy of primitive types"""
        assert transformer._deep_copy(42) == 42
        assert transformer._deep_copy(3.14) == 3.14
        assert transformer._deep_copy('hello') == 'hello'
        assert transformer._deep_copy(True) is True
        assert transformer._deep_copy(None) is None


# ==================== convert_numpy_types Tests ====================


class TestConvertNumpyTypes:
    """Tests for convert_numpy_types method"""

    def test_convert_dict(self, transformer):
        """Test converting dict with numpy types"""
        import numpy as np

        data = {
            'int_val': np.int64(42),
            'float_val': np.float64(3.14),
            'nested': {
                'array': np.array([1, 2, 3])
            }
        }

        result = transformer.convert_numpy_types(data)

        assert isinstance(result['int_val'], int)
        assert result['int_val'] == 42
        assert isinstance(result['float_val'], float)
        assert result['float_val'] == 3.14
        assert isinstance(result['nested']['array'], list)
        assert result['nested']['array'] == [1, 2, 3]

    def test_convert_list(self, transformer):
        """Test converting list with numpy types"""
        import numpy as np

        data = [np.int32(1), np.float32(2.5), np.array([3, 4])]

        result = transformer.convert_numpy_types(data)

        assert isinstance(result[0], int)
        assert isinstance(result[1], float)
        assert isinstance(result[2], list)

    def test_convert_tuple(self, transformer):
        """Test converting tuple with numpy types"""
        import numpy as np

        data = (np.int16(10), np.float16(20.5))

        result = transformer.convert_numpy_types(data)

        assert isinstance(result, tuple)
        assert isinstance(result[0], int)
        assert isinstance(result[1], float)

    def test_convert_numpy_integer(self, transformer):
        """Test converting numpy integer types"""
        import numpy as np

        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            value = dtype(42)
            result = transformer.convert_numpy_types(value)
            assert isinstance(result, int)
            assert result == 42

    def test_convert_numpy_float(self, transformer):
        """Test converting numpy float types"""
        import numpy as np

        for dtype in [np.float16, np.float32, np.float64]:
            value = dtype(3.14)
            result = transformer.convert_numpy_types(value)
            assert isinstance(result, float)
            assert abs(result - 3.14) < 0.01

    def test_convert_numpy_nan(self, transformer):
        """Test converting numpy NaN to None"""
        import numpy as np

        result = transformer.convert_numpy_types(np.nan)

        assert result is None

    def test_convert_numpy_inf(self, transformer):
        """Test converting numpy Inf to None"""
        import numpy as np

        result_pos = transformer.convert_numpy_types(np.inf)
        result_neg = transformer.convert_numpy_types(-np.inf)

        assert result_pos is None
        assert result_neg is None

    def test_convert_numpy_array_with_nan(self, transformer):
        """Test converting numpy array containing NaN"""
        import numpy as np

        data = np.array([1.0, np.nan, 3.0, np.inf])

        result = transformer.convert_numpy_types(data)

        assert result == [1.0, None, 3.0, None]

    def test_convert_datetime(self, transformer):
        """Test converting datetime objects"""
        dt = datetime.datetime(2026, 1, 15, 10, 30, 45)
        date = datetime.date(2026, 1, 15)

        result_dt = transformer.convert_numpy_types(dt)
        result_date = transformer.convert_numpy_types(date)

        assert result_dt == '2026-01-15T10:30:45'
        assert result_date == '2026-01-15'

    def test_convert_python_nan_inf(self, transformer):
        """Test converting Python float NaN/Inf"""
        result_nan = transformer.convert_numpy_types(float('nan'))
        result_inf = transformer.convert_numpy_types(float('inf'))
        result_ninf = transformer.convert_numpy_types(float('-inf'))

        assert result_nan is None
        assert result_inf is None
        assert result_ninf is None

    def test_convert_without_numpy(self, transformer_no_numpy):
        """Test convert_numpy_types when numpy not available"""
        data = {
            'int': 42,
            'float': 3.14,
            'list': [1, 2, 3]
        }

        result = transformer_no_numpy.convert_numpy_types(data)

        # Should return data as-is
        assert result == data

    def test_convert_primitives(self, transformer):
        """Test that primitives pass through unchanged"""
        assert transformer.convert_numpy_types(42) == 42
        assert transformer.convert_numpy_types(3.14) == 3.14
        assert transformer.convert_numpy_types('hello') == 'hello'
        assert transformer.convert_numpy_types(True) is True
        assert transformer.convert_numpy_types(None) is None


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow_with_numpy_data(self, transformer):
        """Test complete workflow with numpy data"""
        import numpy as np

        results = {
            'accuracy': np.float64(0.95),
            'predictions': np.array([0, 1, 1, 0]),
            'confusion_matrix': np.array([[10, 2], [1, 15]]),
            'metrics': {
                'precision': np.float32(0.92),
                'recall': np.float64(0.89)
            },
            'timestamp_obj': datetime.datetime(2026, 1, 15, 12, 0, 0)
        }

        result = transformer.transform(
            results=results,
            model_name='test_model'
        )

        # All numpy types should be converted
        assert isinstance(result['accuracy'], float)
        assert isinstance(result['predictions'], list)
        assert isinstance(result['confusion_matrix'], list)
        assert isinstance(result['metrics']['precision'], float)
        assert isinstance(result['metrics']['recall'], float)
        assert result['timestamp_obj'] == '2026-01-15T12:00:00'

    def test_workflow_with_nan_inf_values(self, transformer):
        """Test workflow with NaN and Inf values"""
        import numpy as np

        results = {
            'valid_score': 0.85,
            'invalid_score': np.nan,
            'infinite_score': np.inf,
            'scores_array': np.array([0.9, np.nan, 0.7, np.inf])
        }

        result = transformer.transform(
            results=results,
            model_name='test_model'
        )

        assert result['valid_score'] == 0.85
        assert result['invalid_score'] is None
        assert result['infinite_score'] is None
        assert result['scores_array'] == [0.9, None, 0.7, None]

    def test_workflow_preserves_non_numpy_data(self, transformer):
        """Test that non-numpy data is preserved correctly"""
        results = {
            'string': 'test',
            'int': 42,
            'float': 3.14,
            'bool': True,
            'none': None,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'}
        }

        result = transformer.transform(
            results=results,
            model_name='test_model'
        )

        assert result['string'] == 'test'
        assert result['int'] == 42
        assert result['float'] == 3.14
        assert result['bool'] is True
        assert result['none'] is None
        assert result['list'] == [1, 2, 3]
        assert result['dict'] == {'nested': 'value'}


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_empty_results(self, transformer):
        """Test with empty results dict"""
        result = transformer.transform(
            results={},
            model_name='test_model'
        )

        assert result['model_name'] == 'test_model'
        assert 'timestamp' in result

    def test_deeply_nested_structure(self, transformer):
        """Test with deeply nested data structure"""
        import numpy as np

        results = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'value': np.int64(42)
                        }
                    }
                }
            }
        }

        result = transformer.transform(
            results=results,
            model_name='test_model'
        )

        assert isinstance(result['level1']['level2']['level3']['level4']['value'], int)

    def test_mixed_numpy_and_python_array(self, transformer):
        """Test list containing both numpy and python types"""
        import numpy as np

        data = [
            1,  # Python int
            np.int32(2),  # Numpy int
            3.14,  # Python float
            np.float64(2.71),  # Numpy float
            [np.array([1, 2])],  # Nested numpy array
        ]

        result = transformer.convert_numpy_types(data)

        assert all(isinstance(x, (int, float, list)) for x in result)

    def test_transform_with_special_characters_in_model_name(self, transformer):
        """Test with special characters in model name"""
        result = transformer.transform(
            results={'accuracy': 0.9},
            model_name='model_v1.0-beta (test)'
        )

        assert result['model_name'] == 'model_v1.0-beta (test)'

    @patch('deepbridge.core.experiment.report.base.logger')
    def test_init_without_numpy_logs_warning(self, mock_logger):
        """Test that initialization without numpy logs warning"""
        with patch.dict('sys.modules', {'numpy': None}):
            with patch('builtins.__import__', side_effect=ImportError("No numpy")):
                transformer = DataTransformer()

                assert transformer.np is None
                mock_logger.warning.assert_called_once()
                assert 'NumPy not available' in str(mock_logger.warning.call_args)

    def test_deep_copy_with_list(self, transformer):
        """Test _deep_copy with list in fallback path"""
        import numpy as np

        # Create an object that fails deepcopy
        class FailDeepCopy:
            def __deepcopy__(self, memo):
                raise TypeError("Cannot deepcopy")

        obj = {'data': [FailDeepCopy(), 'string', 123]}

        # Should fall back to manual copy
        with patch('copy.deepcopy', side_effect=TypeError("Cannot deepcopy")):
            result = transformer._deep_copy(obj)

            assert isinstance(result, dict)
            assert 'data' in result
            assert isinstance(result['data'], list)

    def test_deep_copy_with_tuple(self, transformer):
        """Test _deep_copy with tuple in fallback path"""
        obj = {'data': (1, 2, 3)}

        with patch('copy.deepcopy', side_effect=TypeError("Cannot deepcopy")):
            result = transformer._deep_copy(obj)

            assert isinstance(result, dict)
            assert isinstance(result['data'], tuple)

    def test_deep_copy_with_set(self, transformer):
        """Test _deep_copy with set in fallback path"""
        obj = {'data': {1, 2, 3}}

        with patch('copy.deepcopy', side_effect=TypeError("Cannot deepcopy")):
            result = transformer._deep_copy(obj)

            assert isinstance(result, dict)
            assert isinstance(result['data'], set)

    def test_deep_copy_with_numpy_array_fallback(self, transformer):
        """Test _deep_copy with numpy array in fallback path"""
        import numpy as np

        arr = np.array([1, 2, 3])
        obj = {'data': arr}

        with patch('copy.deepcopy', side_effect=TypeError("Cannot deepcopy")):
            result = transformer._deep_copy(obj)

            assert isinstance(result, dict)
            assert isinstance(result['data'], list)
            assert result['data'] == [1, 2, 3]

    def test_deep_copy_with_uncopyable_object(self, transformer):
        """Test _deep_copy with object that can't be copied"""
        class Uncopyable:
            pass

        obj = Uncopyable()

        with patch('copy.deepcopy', side_effect=TypeError("Cannot deepcopy")):
            result = transformer._deep_copy(obj)

            # Should return the object as is
            assert result is obj

    def test_convert_numpy_types_with_nan_float(self, transformer):
        """Test convert_numpy_types with NaN floating point"""
        import numpy as np

        # Create data with NaN
        data = {'value': np.float64(np.nan)}

        result = transformer.convert_numpy_types(data)

        assert result['value'] is None
