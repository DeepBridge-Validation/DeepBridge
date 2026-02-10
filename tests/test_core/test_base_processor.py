"""
Comprehensive tests for BaseProcessor.

This test suite validates:
1. __init__ - initialization
2. process - abstract method
3. log - verbose logging
4. validate_dataframe - DataFrame validation
5. _is_compatible_dtype - dtype compatibility checking
6. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
import pandas as pd
from unittest.mock import patch

from deepbridge.core.base_processor import BaseProcessor


# ==================== Fixtures ====================


# Create a concrete implementation for testing
class ConcreteProcessor(BaseProcessor):
    """Concrete implementation of BaseProcessor for testing"""

    def process(self, data, **kwargs):
        """Concrete process implementation"""
        self.log("Processing data...")
        return data


@pytest.fixture
def processor():
    """Create ConcreteProcessor instance"""
    return ConcreteProcessor()


@pytest.fixture
def verbose_processor():
    """Create ConcreteProcessor instance with verbose=True"""
    return ConcreteProcessor(verbose=True)


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing"""
    return pd.DataFrame({
        'int_col': [1, 2, 3],
        'float_col': [1.0, 2.0, 3.0],
        'str_col': ['a', 'b', 'c'],
        'bool_col': [True, False, True],
    })


# ==================== Initialization Tests ====================


class TestInitialization:
    """Tests for __init__ method"""

    def test_init_default_verbose_false(self):
        """Test initialization with default verbose=False"""
        proc = ConcreteProcessor()

        assert proc.verbose is False

    def test_init_with_verbose_true(self):
        """Test initialization with verbose=True"""
        proc = ConcreteProcessor(verbose=True)

        assert proc.verbose is True

    def test_init_stores_verbose_parameter(self):
        """Test that verbose parameter is stored"""
        proc = ConcreteProcessor(verbose=False)

        assert hasattr(proc, 'verbose')


# ==================== Process Method Tests ====================


class TestProcessMethod:
    """Tests for process method"""

    def test_process_is_abstract(self):
        """Test that BaseProcessor cannot be instantiated directly"""
        with pytest.raises(TypeError):
            BaseProcessor()

    def test_process_works_in_concrete_class(self, processor):
        """Test that process works in concrete implementation"""
        data = [1, 2, 3]
        result = processor.process(data)

        assert result == data


# ==================== Log Method Tests ====================


class TestLogMethod:
    """Tests for log method"""

    def test_log_prints_when_verbose_true(self, verbose_processor):
        """Test that log prints message when verbose=True"""
        with patch('builtins.print') as mock_print:
            verbose_processor.log('Test message')

            mock_print.assert_called_once_with('Test message')

    def test_log_does_not_print_when_verbose_false(self, processor):
        """Test that log doesn't print when verbose=False"""
        with patch('builtins.print') as mock_print:
            processor.log('Test message')

            mock_print.assert_not_called()

    def test_log_accepts_any_string(self, verbose_processor):
        """Test that log accepts any string message"""
        with patch('builtins.print') as mock_print:
            verbose_processor.log('Complex message with 123 numbers!')

            assert mock_print.called


# ==================== validate_dataframe Tests ====================


class TestValidateDataframe:
    """Tests for validate_dataframe method"""

    def test_validate_with_no_requirements_returns_true(self, processor, sample_dataframe):
        """Test validation with no requirements returns True"""
        result = processor.validate_dataframe(sample_dataframe)

        assert result is True

    def test_validate_with_valid_required_columns(self, processor, sample_dataframe):
        """Test validation with valid required columns"""
        result = processor.validate_dataframe(
            sample_dataframe,
            required_columns=['int_col', 'float_col']
        )

        assert result is True

    def test_validate_raises_error_for_missing_columns(self, processor, sample_dataframe):
        """Test validation raises error for missing columns"""
        with pytest.raises(ValueError, match='Missing required columns'):
            processor.validate_dataframe(
                sample_dataframe,
                required_columns=['missing_col']
            )

    def test_validate_with_multiple_missing_columns(self, processor, sample_dataframe):
        """Test validation with multiple missing columns"""
        with pytest.raises(ValueError, match='Missing required columns'):
            processor.validate_dataframe(
                sample_dataframe,
                required_columns=['missing1', 'missing2']
            )

    def test_validate_with_valid_required_types(self, processor, sample_dataframe):
        """Test validation with valid required types"""
        result = processor.validate_dataframe(
            sample_dataframe,
            required_types={'int_col': int, 'float_col': float}
        )

        assert result is True

    def test_validate_raises_error_for_incorrect_type(self, processor, sample_dataframe):
        """Test validation raises error for incorrect type"""
        with pytest.raises(ValueError, match='has type.*expected'):
            processor.validate_dataframe(
                sample_dataframe,
                required_types={'int_col': float}  # int_col is int, not float
            )

    def test_validate_with_both_columns_and_types(self, processor, sample_dataframe):
        """Test validation with both required columns and types"""
        result = processor.validate_dataframe(
            sample_dataframe,
            required_columns=['int_col', 'float_col'],
            required_types={'int_col': int, 'float_col': float}
        )

        assert result is True

    def test_validate_skips_type_check_for_missing_column(self, processor, sample_dataframe):
        """Test validation skips type check for columns not in DataFrame"""
        # Should not raise error even though column doesn't exist
        result = processor.validate_dataframe(
            sample_dataframe,
            required_types={'missing_col': int}
        )

        assert result is True

    def test_validate_with_empty_required_columns_list(self, processor, sample_dataframe):
        """Test validation with empty required columns list"""
        result = processor.validate_dataframe(
            sample_dataframe,
            required_columns=[]
        )

        assert result is True

    def test_validate_with_empty_required_types_dict(self, processor, sample_dataframe):
        """Test validation with empty required types dict"""
        result = processor.validate_dataframe(
            sample_dataframe,
            required_types={}
        )

        assert result is True


# ==================== _is_compatible_dtype Tests ====================


class TestIsCompatibleDtype:
    """Tests for _is_compatible_dtype method"""

    def test_is_compatible_float_dtype(self, processor):
        """Test float dtype compatibility"""
        df = pd.DataFrame({'col': [1.0, 2.0, 3.0]})

        result = processor._is_compatible_dtype(df['col'].dtype, float)

        assert result is True

    def test_is_compatible_int_dtype(self, processor):
        """Test int dtype compatibility"""
        df = pd.DataFrame({'col': [1, 2, 3]})

        result = processor._is_compatible_dtype(df['col'].dtype, int)

        assert result is True

    def test_is_compatible_bool_dtype(self, processor):
        """Test bool dtype compatibility"""
        df = pd.DataFrame({'col': [True, False, True]})

        result = processor._is_compatible_dtype(df['col'].dtype, bool)

        assert result is True

    def test_is_compatible_str_dtype_with_object(self, processor):
        """Test str dtype compatibility with object dtype"""
        df = pd.DataFrame({'col': ['a', 'b', 'c']})

        result = processor._is_compatible_dtype(df['col'].dtype, str)

        assert result is True

    def test_is_compatible_categorical_dtype(self, processor):
        """Test categorical dtype compatibility"""
        df = pd.DataFrame({'col': pd.Categorical(['a', 'b', 'c'])})

        result = processor._is_compatible_dtype(df['col'].dtype, 'categorical')

        assert result is True

    def test_is_compatible_returns_true_for_unknown_types(self, processor):
        """Test that unknown types return True (permissive)"""
        df = pd.DataFrame({'col': [1, 2, 3]})

        result = processor._is_compatible_dtype(df['col'].dtype, dict)

        assert result is True

    def test_is_compatible_int_not_float(self, processor):
        """Test that int dtype is not compatible with float requirement"""
        df = pd.DataFrame({'col': [1, 2, 3]})

        result = processor._is_compatible_dtype(df['col'].dtype, float)

        assert result is False

    def test_is_compatible_float_not_int(self, processor):
        """Test that float dtype is not compatible with int requirement"""
        df = pd.DataFrame({'col': [1.0, 2.0, 3.0]})

        result = processor._is_compatible_dtype(df['col'].dtype, int)

        assert result is False


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_validation_workflow(self, processor, sample_dataframe):
        """Test complete validation workflow"""
        # Validate columns exist
        processor.validate_dataframe(
            sample_dataframe,
            required_columns=['int_col', 'float_col', 'str_col']
        )

        # Validate types
        processor.validate_dataframe(
            sample_dataframe,
            required_types={
                'int_col': int,
                'float_col': float,
                'str_col': str,
                'bool_col': bool,
            }
        )

        # Validate both together
        result = processor.validate_dataframe(
            sample_dataframe,
            required_columns=['int_col', 'float_col'],
            required_types={'int_col': int, 'float_col': float}
        )

        assert result is True

    def test_process_with_validation(self, processor, sample_dataframe):
        """Test processing with validation"""
        # Validate first
        processor.validate_dataframe(
            sample_dataframe,
            required_columns=['int_col']
        )

        # Then process
        result = processor.process(sample_dataframe)

        assert result is not None


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_validate_empty_dataframe(self, processor):
        """Test validation with empty DataFrame"""
        empty_df = pd.DataFrame()

        result = processor.validate_dataframe(empty_df)

        assert result is True

    def test_validate_single_column_dataframe(self, processor):
        """Test validation with single column DataFrame"""
        df = pd.DataFrame({'col': [1, 2, 3]})

        result = processor.validate_dataframe(
            df,
            required_columns=['col'],
            required_types={'col': int}
        )

        assert result is True

    def test_validate_with_none_parameters(self, processor, sample_dataframe):
        """Test validation with explicit None parameters"""
        result = processor.validate_dataframe(
            sample_dataframe,
            required_columns=None,
            required_types=None
        )

        assert result is True

    def test_log_with_empty_string(self, verbose_processor):
        """Test log with empty string"""
        with patch('builtins.print') as mock_print:
            verbose_processor.log('')

            mock_print.assert_called_once_with('')

    def test_log_with_unicode_characters(self, verbose_processor):
        """Test log with unicode characters"""
        with patch('builtins.print') as mock_print:
            verbose_processor.log('José 北京 café 🎉')

            mock_print.assert_called_once()

    def test_validate_with_all_column_types(self, processor):
        """Test validation with all different column types"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.0, 2.0, 3.0],
            'str_col': ['a', 'b', 'c'],
            'bool_col': [True, False, True],
            'cat_col': pd.Categorical(['x', 'y', 'z']),
        })

        result = processor.validate_dataframe(
            df,
            required_types={
                'int_col': int,
                'float_col': float,
                'str_col': str,
                'bool_col': bool,
                'cat_col': 'categorical',
            }
        )

        assert result is True
