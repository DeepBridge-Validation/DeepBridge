"""
Comprehensive tests for DataFormatter.

This test suite validates:
1. format_percentage - percentage formatting with decimal places
2. format_number - number formatting with decimal places
3. format_list - list to string conversion
4. format_object_list - extracting and formatting object lists

Coverage Target: ~95%+
"""

import pytest
from unittest.mock import patch

from deepbridge.core.experiment.report.utils.formatters import DataFormatter


# ==================== Fixtures ====================


@pytest.fixture
def formatter():
    """Create DataFormatter instance (though all methods are static)"""
    return DataFormatter()


# ==================== format_percentage Tests ====================


class TestFormatPercentage:
    """Tests for format_percentage method"""

    def test_format_valid_percentage(self):
        """Test formatting valid percentage value"""
        result = DataFormatter.format_percentage(0.95)
        assert result == '95.00%'

    def test_format_percentage_with_custom_decimals(self):
        """Test formatting with custom decimal places"""
        result = DataFormatter.format_percentage(0.8567, decimal_places=3)
        assert result == '85.670%'

    def test_format_percentage_with_zero_decimals(self):
        """Test formatting with zero decimal places"""
        result = DataFormatter.format_percentage(0.75, decimal_places=0)
        assert result == '75%'

    def test_format_percentage_with_one_decimal(self):
        """Test formatting with one decimal place"""
        result = DataFormatter.format_percentage(0.456, decimal_places=1)
        assert result == '45.6%'

    def test_format_none_percentage(self):
        """Test formatting None returns N/A"""
        result = DataFormatter.format_percentage(None)
        assert result == 'N/A'

    def test_format_zero_percentage(self):
        """Test formatting zero"""
        result = DataFormatter.format_percentage(0.0)
        assert result == '0.00%'

    def test_format_one_percentage(self):
        """Test formatting 1.0 (100%)"""
        result = DataFormatter.format_percentage(1.0)
        assert result == '100.00%'

    def test_format_percentage_greater_than_one(self):
        """Test formatting value > 1.0"""
        result = DataFormatter.format_percentage(1.5)
        assert result == '150.00%'

    def test_format_negative_percentage(self):
        """Test formatting negative value"""
        result = DataFormatter.format_percentage(-0.25)
        assert result == '-25.00%'

    def test_format_very_small_percentage(self):
        """Test formatting very small value"""
        result = DataFormatter.format_percentage(0.001, decimal_places=4)
        assert result == '0.1000%'

    def test_format_string_percentage_error(self):
        """Test that string value returns N/A with warning"""
        with patch('deepbridge.core.experiment.report.utils.formatters.logger') as mock_logger:
            result = DataFormatter.format_percentage('invalid')

            assert result == 'N/A'
            mock_logger.warning.assert_called_once()

    def test_format_list_percentage_error(self):
        """Test that list value returns N/A with warning"""
        with patch('deepbridge.core.experiment.report.utils.formatters.logger') as mock_logger:
            result = DataFormatter.format_percentage([1, 2, 3])

            assert result == 'N/A'
            mock_logger.warning.assert_called_once()


# ==================== format_number Tests ====================


class TestFormatNumber:
    """Tests for format_number method"""

    def test_format_valid_number(self):
        """Test formatting valid number"""
        result = DataFormatter.format_number(123.456)
        assert result == '123.456'

    def test_format_number_with_custom_decimals(self):
        """Test formatting with custom decimal places"""
        result = DataFormatter.format_number(3.14159, decimal_places=2)
        assert result == '3.14'

    def test_format_number_with_zero_decimals(self):
        """Test formatting with zero decimal places"""
        result = DataFormatter.format_number(42.7, decimal_places=0)
        assert result == '43'

    def test_format_number_with_many_decimals(self):
        """Test formatting with many decimal places"""
        result = DataFormatter.format_number(1.23456789, decimal_places=6)
        assert result == '1.234568'

    def test_format_none_number(self):
        """Test formatting None returns N/A"""
        result = DataFormatter.format_number(None)
        assert result == 'N/A'

    def test_format_zero_number(self):
        """Test formatting zero"""
        result = DataFormatter.format_number(0.0)
        assert result == '0.000'

    def test_format_negative_number(self):
        """Test formatting negative number"""
        result = DataFormatter.format_number(-99.999, decimal_places=2)
        assert result == '-100.00'

    def test_format_very_large_number(self):
        """Test formatting very large number"""
        result = DataFormatter.format_number(1e10, decimal_places=1)
        assert result == '10000000000.0'

    def test_format_very_small_number(self):
        """Test formatting very small number"""
        result = DataFormatter.format_number(0.000001, decimal_places=8)
        assert result == '0.00000100'

    def test_format_integer_as_number(self):
        """Test formatting integer"""
        result = DataFormatter.format_number(42, decimal_places=2)
        assert result == '42.00'

    def test_format_string_number_error(self):
        """Test that string value returns N/A with warning"""
        with patch('deepbridge.core.experiment.report.utils.formatters.logger') as mock_logger:
            result = DataFormatter.format_number('not a number')

            assert result == 'N/A'
            mock_logger.warning.assert_called_once()

    def test_format_dict_number_error(self):
        """Test that dict value returns N/A with warning"""
        with patch('deepbridge.core.experiment.report.utils.formatters.logger') as mock_logger:
            result = DataFormatter.format_number({'key': 'value'})

            assert result == 'N/A'
            mock_logger.warning.assert_called_once()


# ==================== format_list Tests ====================


class TestFormatList:
    """Tests for format_list method"""

    def test_format_simple_list(self):
        """Test formatting simple list"""
        result = DataFormatter.format_list([1, 2, 3])
        assert result == '1, 2, 3'

    def test_format_string_list(self):
        """Test formatting list of strings"""
        result = DataFormatter.format_list(['apple', 'banana', 'cherry'])
        assert result == 'apple, banana, cherry'

    def test_format_mixed_list(self):
        """Test formatting list with mixed types"""
        result = DataFormatter.format_list([1, 'two', 3.0, True])
        assert result == '1, two, 3.0, True'

    def test_format_list_custom_separator(self):
        """Test formatting with custom separator"""
        result = DataFormatter.format_list([1, 2, 3], separator=' | ')
        assert result == '1 | 2 | 3'

    def test_format_list_newline_separator(self):
        """Test formatting with newline separator"""
        result = DataFormatter.format_list(['a', 'b', 'c'], separator='\n')
        assert result == 'a\nb\nc'

    def test_format_empty_list(self):
        """Test formatting empty list returns empty string"""
        result = DataFormatter.format_list([])
        assert result == ''

    def test_format_none_list(self):
        """Test formatting None list returns empty string"""
        result = DataFormatter.format_list(None)
        assert result == ''

    def test_format_single_item_list(self):
        """Test formatting list with single item"""
        result = DataFormatter.format_list([42])
        assert result == '42'

    def test_format_list_with_none_items(self):
        """Test formatting list with None items"""
        result = DataFormatter.format_list([1, None, 3])
        assert result == '1, None, 3'

    def test_format_list_exception_handling(self):
        """Test that exceptions are caught and logged"""
        # Create object that will fail when converted to string
        class FailStr:
            def __str__(self):
                raise ValueError("Cannot convert to string")

        with patch('deepbridge.core.experiment.report.utils.formatters.logger') as mock_logger:
            result = DataFormatter.format_list([FailStr()])

            # Should return str representation of list
            assert 'FailStr' in result or 'object' in result
            mock_logger.warning.assert_called_once()

    def test_format_list_with_empty_strings(self):
        """Test formatting list with empty strings"""
        result = DataFormatter.format_list(['', 'b', ''])
        assert result == ', b, '


# ==================== format_object_list Tests ====================


class TestFormatObjectList:
    """Tests for format_object_list method"""

    def test_format_simple_object_list(self):
        """Test formatting simple list of objects"""
        objects = [
            {'name': 'Alice', 'age': 30},
            {'name': 'Bob', 'age': 25},
            {'name': 'Charlie', 'age': 35}
        ]
        result = DataFormatter.format_object_list(objects, 'name')
        assert result == 'Alice, Bob, Charlie'

    def test_format_object_list_different_key(self):
        """Test formatting with different key"""
        objects = [
            {'id': 1, 'value': 100},
            {'id': 2, 'value': 200},
            {'id': 3, 'value': 300}
        ]
        result = DataFormatter.format_object_list(objects, 'value')
        assert result == '100, 200, 300'

    def test_format_object_list_custom_separator(self):
        """Test formatting with custom separator"""
        objects = [
            {'name': 'A'},
            {'name': 'B'},
            {'name': 'C'}
        ]
        result = DataFormatter.format_object_list(objects, 'name', separator=' - ')
        assert result == 'A - B - C'

    def test_format_empty_object_list(self):
        """Test formatting empty list returns empty string"""
        result = DataFormatter.format_object_list([], 'name')
        assert result == ''

    def test_format_none_object_list(self):
        """Test formatting None list returns empty string"""
        result = DataFormatter.format_object_list(None, 'name')
        assert result == ''

    def test_format_object_list_missing_key(self):
        """Test formatting when objects are missing the key"""
        objects = [
            {'name': 'Alice'},
            {'age': 25},  # Missing 'name'
            {'name': 'Charlie'}
        ]
        result = DataFormatter.format_object_list(objects, 'name')
        # Empty string from missing key should be filtered out
        assert result == 'Alice, Charlie'

    def test_format_object_list_all_missing_key(self):
        """Test formatting when all objects are missing the key"""
        objects = [
            {'age': 30},
            {'age': 25},
            {'age': 35}
        ]
        result = DataFormatter.format_object_list(objects, 'name')
        assert result == ''

    def test_format_object_list_with_none_values(self):
        """Test formatting when key values are None"""
        objects = [
            {'name': 'Alice'},
            {'name': None},
            {'name': 'Charlie'}
        ]
        result = DataFormatter.format_object_list(objects, 'name')
        # None converts to 'None' which is truthy when filtered
        assert result == 'Alice, None, Charlie'

    def test_format_object_list_with_numeric_values(self):
        """Test formatting when key values are numbers"""
        objects = [
            {'score': 95},
            {'score': 87},
            {'score': 92}
        ]
        result = DataFormatter.format_object_list(objects, 'score')
        assert result == '95, 87, 92'

    def test_format_object_list_exception_handling(self):
        """Test that exceptions are caught and logged"""
        # Create object that will fail
        not_a_dict = "not a dictionary"

        with patch('deepbridge.core.experiment.report.utils.formatters.logger') as mock_logger:
            result = DataFormatter.format_object_list([not_a_dict], 'name')

            # Should return empty string on exception
            assert result == ''
            mock_logger.warning.assert_called_once()

    def test_format_object_list_single_object(self):
        """Test formatting single object"""
        objects = [{'name': 'Single'}]
        result = DataFormatter.format_object_list(objects, 'name')
        assert result == 'Single'

    def test_format_object_list_with_empty_string_values(self):
        """Test that empty string values are filtered out"""
        objects = [
            {'name': 'Alice'},
            {'name': ''},  # Empty string
            {'name': 'Charlie'}
        ]
        result = DataFormatter.format_object_list(objects, 'name')
        # Empty strings should be filtered
        assert result == 'Alice, Charlie'


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_format_metrics_workflow(self):
        """Test formatting multiple types of metrics"""
        # Simulate formatting various metrics
        accuracy = DataFormatter.format_percentage(0.92, decimal_places=2)
        loss = DataFormatter.format_number(0.345, decimal_places=4)
        classes = DataFormatter.format_list(['cat', 'dog', 'bird'])

        assert accuracy == '92.00%'
        assert loss == '0.3450'
        assert classes == 'cat, dog, bird'

    def test_format_model_results_workflow(self):
        """Test formatting model results"""
        models = [
            {'name': 'Model A', 'accuracy': 0.95},
            {'name': 'Model B', 'accuracy': 0.87},
            {'name': 'Model C', 'accuracy': 0.91}
        ]

        model_names = DataFormatter.format_object_list(models, 'name')

        # Format each accuracy
        accuracies = [
            DataFormatter.format_percentage(m['accuracy'], 1)
            for m in models
        ]

        assert model_names == 'Model A, Model B, Model C'
        assert accuracies == ['95.0%', '87.0%', '91.0%']

    def test_error_handling_workflow(self):
        """Test that errors don't crash the formatting"""
        with patch('deepbridge.core.experiment.report.utils.formatters.logger'):
            # All should return safe defaults
            result1 = DataFormatter.format_percentage('bad')
            result2 = DataFormatter.format_number({'bad': 'data'})
            result3 = DataFormatter.format_list(None)
            result4 = DataFormatter.format_object_list(None, 'key')

            assert result1 == 'N/A'
            assert result2 == 'N/A'
            assert result3 == ''
            assert result4 == ''


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_format_percentage_with_infinity(self):
        """Test formatting infinity"""
        with patch('deepbridge.core.experiment.report.utils.formatters.logger'):
            result = DataFormatter.format_percentage(float('inf'))
            # May either format or return N/A depending on implementation
            assert result == 'inf%' or result == 'N/A'

    def test_format_number_with_nan(self):
        """Test formatting NaN"""
        with patch('deepbridge.core.experiment.report.utils.formatters.logger'):
            result = DataFormatter.format_number(float('nan'))
            # NaN should format or return N/A
            assert 'nan' in result.lower() or result == 'N/A'

    def test_format_very_long_list(self):
        """Test formatting very long list"""
        long_list = list(range(1000))
        result = DataFormatter.format_list(long_list)

        # Should handle without error
        assert result.startswith('0, 1, 2')
        assert result.endswith('999')

    def test_format_list_with_unicode(self):
        """Test formatting list with unicode characters"""
        result = DataFormatter.format_list(['hello', '世界', 'мир', '🌍'])
        assert '世界' in result
        assert 'мир' in result
        assert '🌍' in result

    def test_format_object_list_deeply_nested(self):
        """Test formatting with nested object values"""
        objects = [
            {'name': 'A', 'data': {'nested': 'value'}},
            {'name': 'B', 'data': {'nested': 'value'}}
        ]
        result = DataFormatter.format_object_list(objects, 'name')
        assert result == 'A, B'

    def test_format_with_boolean_values(self):
        """Test formatting boolean values"""
        result = DataFormatter.format_list([True, False, True])
        assert result == 'True, False, True'

    def test_format_percentage_rounding(self):
        """Test that rounding works correctly"""
        result = DataFormatter.format_percentage(0.12345, decimal_places=2)
        assert result == '12.35%'  # Should round up

        result = DataFormatter.format_percentage(0.12344, decimal_places=2)
        assert result == '12.34%'  # Should round down
