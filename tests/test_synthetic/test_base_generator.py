"""
Comprehensive tests for BaseGenerator.

This test suite validates:
1. __init__ - initialization with various parameters
2. fit - abstract method with concrete logic
3. generate - abstract method enforcement
4. _infer_column_types - column type inference
5. _restore_dtypes - dtype restoration
6. log - verbose logging
7. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from deepbridge.synthetic.base_generator import BaseGenerator


# ==================== Fixtures ====================


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing"""
    return pd.DataFrame({
        'cat_col': ['A', 'B', 'C', 'A'],
        'num_col': [1.0, 2.0, 3.0, 4.0],
        'bool_col': [True, False, True, False],
        'int_col': [1, 2, 3, 4],
        'target': [0, 1, 0, 1],
    })


@pytest.fixture
def categorical_data():
    """Create DataFrame with categorical dtype"""
    df = pd.DataFrame({
        'cat1': pd.Categorical(['X', 'Y', 'Z', 'X']),
        'cat2': ['A', 'B', 'A', 'B'],
        'num': [1, 2, 3, 4],
    })
    return df


# Create a concrete implementation for testing
class ConcreteGenerator(BaseGenerator):
    """Concrete implementation of BaseGenerator for testing"""

    def fit(self, data, target_column=None, categorical_columns=None,
            numerical_columns=None, **kwargs):
        """Concrete fit implementation"""
        # Call parent fit method
        super().fit(data, target_column, categorical_columns,
                   numerical_columns, **kwargs)
        return self

    def generate(self, num_samples, **kwargs):
        """Concrete generate implementation"""
        # Call parent generate to check fitted status
        super().generate(num_samples, **kwargs)

        # Generate simple synthetic data
        data = {}
        if self.numerical_columns:
            for col in self.numerical_columns:
                data[col] = self.rng.randn(num_samples)
        if self.categorical_columns:
            for col in self.categorical_columns:
                data[col] = self.rng.choice(['A', 'B', 'C'], num_samples)

        return pd.DataFrame(data)


@pytest.fixture
def generator():
    """Create ConcreteGenerator instance"""
    return ConcreteGenerator(random_state=42)


# ==================== Initialization Tests ====================


class TestInitialization:
    """Tests for __init__ method"""

    def test_init_default_parameters(self):
        """Test initialization with default parameters"""
        gen = ConcreteGenerator()

        assert gen.random_state is None
        assert gen.verbose is False
        assert gen.preserve_dtypes is True
        assert gen.fitted is False
        assert gen.categorical_columns is None
        assert gen.numerical_columns is None
        assert gen.target_column is None
        assert gen.dtypes is None

    def test_init_with_random_state(self):
        """Test initialization with random state"""
        gen = ConcreteGenerator(random_state=42)

        assert gen.random_state == 42
        assert gen.rng is not None
        assert isinstance(gen.rng, np.random.RandomState)

    def test_init_with_verbose(self):
        """Test initialization with verbose=True"""
        gen = ConcreteGenerator(verbose=True)

        assert gen.verbose is True

    def test_init_with_preserve_dtypes_false(self):
        """Test initialization with preserve_dtypes=False"""
        gen = ConcreteGenerator(preserve_dtypes=False)

        assert gen.preserve_dtypes is False

    def test_init_creates_rng(self):
        """Test that initialization creates random number generator"""
        gen = ConcreteGenerator(random_state=123)

        # Should have rng attribute
        assert hasattr(gen, 'rng')
        # RNG should be reproducible
        val1 = gen.rng.rand()
        gen2 = ConcreteGenerator(random_state=123)
        val2 = gen2.rng.rand()
        assert val1 == val2


# ==================== Fit Method Tests ====================


class TestFitMethod:
    """Tests for fit method"""

    def test_fit_stores_target_column(self, generator, sample_data):
        """Test that fit stores target column"""
        generator.fit(sample_data, target_column='target')

        assert generator.target_column == 'target'

    def test_fit_with_explicit_columns(self, generator, sample_data):
        """Test fit with explicit categorical and numerical columns"""
        generator.fit(
            sample_data,
            categorical_columns=['cat_col', 'bool_col'],
            numerical_columns=['num_col', 'int_col']
        )

        assert generator.categorical_columns == ['cat_col', 'bool_col']
        assert generator.numerical_columns == ['num_col', 'int_col']

    def test_fit_infers_columns_when_not_provided(self, generator, sample_data):
        """Test that fit infers columns when not provided"""
        generator.fit(sample_data)

        assert generator.categorical_columns is not None
        assert generator.numerical_columns is not None

    def test_fit_preserves_dtypes_when_enabled(self, generator, sample_data):
        """Test that fit preserves dtypes when preserve_dtypes=True"""
        generator.fit(sample_data)

        assert generator.dtypes is not None
        assert isinstance(generator.dtypes, dict)
        assert len(generator.dtypes) == len(sample_data.columns)

    def test_fit_does_not_preserve_dtypes_when_disabled(self, sample_data):
        """Test that fit doesn't preserve dtypes when preserve_dtypes=False"""
        gen = ConcreteGenerator(preserve_dtypes=False)
        gen.fit(sample_data)

        assert gen.dtypes is None

    def test_fit_sets_fitted_flag(self, generator, sample_data):
        """Test that fit sets fitted flag to True"""
        assert generator.fitted is False

        generator.fit(sample_data)

        assert generator.fitted is True

    def test_fit_returns_self(self, generator, sample_data):
        """Test that fit returns self for chaining"""
        result = generator.fit(sample_data)

        assert result is generator

    def test_fit_with_only_categorical_columns(self, generator, sample_data):
        """Test fit when only categorical_columns is provided"""
        generator.fit(sample_data, categorical_columns=['cat_col'])

        assert generator.categorical_columns == ['cat_col']
        # numerical_columns should be inferred
        assert generator.numerical_columns is not None

    def test_fit_with_only_numerical_columns(self, generator, sample_data):
        """Test fit when only numerical_columns is provided"""
        generator.fit(sample_data, numerical_columns=['num_col'])

        assert generator.numerical_columns == ['num_col']
        # categorical_columns should be inferred
        assert generator.categorical_columns is not None


# ==================== Generate Method Tests ====================


class TestGenerateMethod:
    """Tests for generate method"""

    def test_generate_raises_error_when_not_fitted(self, generator):
        """Test that generate raises error when not fitted"""
        with pytest.raises(ValueError, match='Generator not fitted'):
            generator.generate(10)

    def test_generate_works_after_fit(self, generator, sample_data):
        """Test that generate works after fitting"""
        generator.fit(sample_data)

        result = generator.generate(10)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_generate_respects_num_samples(self, generator, sample_data):
        """Test that generate creates correct number of samples"""
        generator.fit(sample_data)

        result = generator.generate(100)

        assert len(result) == 100


# ==================== _infer_column_types Tests ====================


class TestInferColumnTypes:
    """Tests for _infer_column_types method"""

    def test_infer_object_columns_as_categorical(self, generator, sample_data):
        """Test that object columns are inferred as categorical"""
        generator._infer_column_types(sample_data)

        assert 'cat_col' in generator.categorical_columns

    def test_infer_boolean_columns_as_categorical(self, generator, sample_data):
        """Test that boolean columns are inferred as categorical"""
        generator._infer_column_types(sample_data)

        assert 'bool_col' in generator.categorical_columns

    def test_infer_numeric_columns_as_numerical(self, generator, sample_data):
        """Test that numeric columns are inferred as numerical"""
        generator._infer_column_types(sample_data)

        assert 'num_col' in generator.numerical_columns
        assert 'int_col' in generator.numerical_columns

    def test_infer_excludes_target_column(self, generator, sample_data):
        """Test that target column is excluded from inference"""
        generator.target_column = 'target'
        generator._infer_column_types(sample_data)

        assert 'target' not in generator.categorical_columns
        assert 'target' not in generator.numerical_columns

    def test_infer_with_categorical_dtype(self, generator, categorical_data):
        """Test inference with pandas Categorical dtype"""
        generator._infer_column_types(categorical_data)

        assert 'cat1' in generator.categorical_columns
        assert 'cat2' in generator.categorical_columns
        assert 'num' in generator.numerical_columns

    def test_infer_prints_message_when_verbose(self, sample_data):
        """Test that inference prints message when verbose=True"""
        gen = ConcreteGenerator(verbose=True)

        with patch('builtins.print') as mock_print:
            gen._infer_column_types(sample_data)

            mock_print.assert_called_once()
            call_args = mock_print.call_args[0][0]
            assert 'categorical columns' in call_args
            assert 'numerical columns' in call_args

    def test_infer_does_not_print_when_not_verbose(self, generator, sample_data):
        """Test that inference doesn't print when verbose=False"""
        with patch('builtins.print') as mock_print:
            generator._infer_column_types(sample_data)

            mock_print.assert_not_called()


# ==================== _restore_dtypes Tests ====================


class TestRestoreDtypes:
    """Tests for _restore_dtypes method"""

    def test_restore_returns_copy(self, generator, sample_data):
        """Test that restore returns a copy, not original"""
        generator.dtypes = {'col': 'int64'}

        test_df = pd.DataFrame({'col': [1, 2, 3]})
        result = generator._restore_dtypes(test_df)

        assert result is not test_df

    def test_restore_when_preserve_dtypes_false(self, sample_data):
        """Test restore returns unchanged data when preserve_dtypes=False"""
        gen = ConcreteGenerator(preserve_dtypes=False)

        result = gen._restore_dtypes(sample_data)

        assert result is sample_data  # Should return original

    def test_restore_when_dtypes_none(self, generator, sample_data):
        """Test restore returns unchanged data when dtypes is None"""
        generator.dtypes = None

        result = generator._restore_dtypes(sample_data)

        assert result is sample_data

    def test_restore_converts_dtypes(self, generator):
        """Test that restore converts column dtypes"""
        generator.dtypes = {'col1': 'int32', 'col2': 'float32'}

        test_df = pd.DataFrame({
            'col1': [1.0, 2.0, 3.0],
            'col2': [1, 2, 3]
        })

        result = generator._restore_dtypes(test_df)

        assert result['col1'].dtype == 'int32'
        assert result['col2'].dtype == 'float32'

    def test_restore_handles_categorical_dtype(self, generator):
        """Test that restore handles categorical dtypes"""
        cat_dtype = pd.CategoricalDtype(categories=['A', 'B', 'C'])
        generator.dtypes = {'col': cat_dtype}

        test_df = pd.DataFrame({'col': ['A', 'B', 'C']})

        result = generator._restore_dtypes(test_df)

        assert pd.api.types.is_categorical_dtype(result['col'])

    def test_restore_skips_missing_columns(self, generator):
        """Test that restore skips columns not in data"""
        generator.dtypes = {'col1': 'int32', 'col2': 'float32'}

        test_df = pd.DataFrame({'col1': [1, 2, 3]})  # col2 missing

        result = generator._restore_dtypes(test_df)

        assert 'col2' not in result.columns
        assert result['col1'].dtype == 'int32'

    def test_restore_handles_conversion_errors(self, generator):
        """Test that restore handles conversion errors gracefully"""
        generator.dtypes = {'col': 'int32'}

        # Create data that can't be converted to int
        test_df = pd.DataFrame({'col': ['not', 'a', 'number']})

        # Should not raise error, just keep original type
        result = generator._restore_dtypes(test_df)

        assert result is not None

    def test_restore_prints_message_on_error_when_verbose(self):
        """Test that restore prints message on conversion error when verbose"""
        gen = ConcreteGenerator(verbose=True, preserve_dtypes=True)
        gen.dtypes = {'col': 'int32'}

        test_df = pd.DataFrame({'col': ['not', 'a', 'number']})

        with patch('builtins.print') as mock_print:
            gen._restore_dtypes(test_df)

            # Should have printed error message
            assert mock_print.called


# ==================== Log Method Tests ====================


class TestLogMethod:
    """Tests for log method"""

    def test_log_prints_when_verbose(self):
        """Test that log prints message when verbose=True"""
        gen = ConcreteGenerator(verbose=True)

        with patch('builtins.print') as mock_print:
            gen.log('Test message')

            mock_print.assert_called_once_with('Test message')

    def test_log_does_not_print_when_not_verbose(self, generator):
        """Test that log doesn't print when verbose=False"""
        with patch('builtins.print') as mock_print:
            generator.log('Test message')

            mock_print.assert_not_called()


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for complete workflows"""

    def test_full_workflow_fit_and_generate(self, generator, sample_data):
        """Test complete workflow: fit then generate"""
        # Fit
        generator.fit(sample_data, target_column='target')

        # Generate
        result = generator.generate(20)

        assert len(result) == 20
        assert generator.fitted is True

    def test_workflow_with_dtype_preservation(self, sample_data):
        """Test workflow with dtype preservation"""
        gen = ConcreteGenerator(random_state=42, preserve_dtypes=True)

        # Fit
        gen.fit(sample_data)

        # Generate
        result = gen.generate(10)

        assert result is not None

    def test_workflow_with_explicit_columns(self, generator, sample_data):
        """Test workflow with explicit column specification"""
        generator.fit(
            sample_data,
            target_column='target',
            categorical_columns=['cat_col'],
            numerical_columns=['num_col', 'int_col']
        )

        result = generator.generate(15)

        assert len(result) == 15

    def test_reproducibility_with_random_state(self, sample_data):
        """Test that same random_state produces same results"""
        gen1 = ConcreteGenerator(random_state=42)
        gen1.fit(sample_data)
        result1 = gen1.generate(10)

        gen2 = ConcreteGenerator(random_state=42)
        gen2.fit(sample_data)
        result2 = gen2.generate(10)

        # Should produce same random values
        pd.testing.assert_frame_equal(result1, result2)


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_fit_with_empty_dataframe(self, generator):
        """Test fit with empty DataFrame"""
        empty_df = pd.DataFrame()

        generator.fit(empty_df)

        assert generator.categorical_columns == []
        assert generator.numerical_columns == []

    def test_fit_with_single_column(self, generator):
        """Test fit with single column DataFrame"""
        single_col_df = pd.DataFrame({'col': [1, 2, 3]})

        generator.fit(single_col_df)

        assert len(generator.numerical_columns) == 1

    def test_generate_with_zero_samples(self, generator, sample_data):
        """Test generate with num_samples=0"""
        generator.fit(sample_data)

        result = generator.generate(0)

        assert len(result) == 0

    def test_fit_with_all_same_type_columns(self, generator):
        """Test fit with all columns of same type"""
        all_numeric_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': [4.0, 5.0, 6.0],
            'col3': [7, 8, 9]
        })

        generator.fit(all_numeric_df)

        assert len(generator.categorical_columns) == 0
        assert len(generator.numerical_columns) == 3

    def test_restore_dtypes_with_empty_dtypes_dict(self, generator, sample_data):
        """Test restore with empty dtypes dictionary"""
        generator.dtypes = {}

        result = generator._restore_dtypes(sample_data)

        assert result is not None
