"""
Comprehensive tests for synthetic data validators.

This test suite validates:
1. validate_dataset - dataset basic validation
2. validate_columns - column existence validation
3. validate_generator_params - generation parameter validation
4. validate_file_path - file path validation
5. Edge cases and error handling

Coverage Target: ~95%+
"""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
import os

from deepbridge.synthetic.utils.validators import (
    validate_dataset,
    validate_columns,
    validate_generator_params,
    validate_file_path,
)


# ==================== Fixtures ====================


@pytest.fixture
def valid_dataframe():
    """Create valid DataFrame for testing"""
    return pd.DataFrame({
        'num_col1': [1, 2, 3, 4, 5],
        'num_col2': [1.0, 2.0, 3.0, 4.0, 5.0],
        'cat_col1': ['A', 'B', 'C', 'D', 'E'],
        'cat_col2': ['X', 'Y', 'Z', 'X', 'Y'],
        'target': [0, 1, 0, 1, 0],
    })


@pytest.fixture
def temp_dir():
    """Create temporary directory for testing"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ==================== validate_dataset Tests ====================


class TestValidateDataset:
    """Tests for validate_dataset function"""

    def test_validate_valid_dataset(self, valid_dataframe):
        """Test validation with valid dataset"""
        result = validate_dataset(valid_dataframe)
        assert result is True

    def test_validate_raises_error_for_none(self):
        """Test validation raises error for None"""
        with pytest.raises(ValueError, match='empty or None'):
            validate_dataset(None)

    def test_validate_raises_error_for_empty_dataframe(self):
        """Test validation raises error for empty DataFrame"""
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match='empty or None'):
            validate_dataset(empty_df)

    def test_validate_raises_error_for_zero_rows(self):
        """Test validation raises error for DataFrame with zero rows"""
        df = pd.DataFrame(columns=['col1', 'col2'])

        # DataFrame with no rows is considered empty by pandas
        with pytest.raises(ValueError, match='empty or None'):
            validate_dataset(df)

    def test_validate_raises_error_for_zero_columns(self):
        """Test validation raises error for DataFrame with zero columns"""
        # Create DataFrame with rows but no columns
        df = pd.DataFrame(index=[0, 1, 2])

        # DataFrame with no columns is considered empty by pandas
        with pytest.raises(ValueError, match='empty or None'):
            validate_dataset(df)

    def test_validate_raises_error_for_duplicate_columns(self):
        """Test validation raises error for duplicate column names"""
        df = pd.DataFrame([[1, 2], [3, 4]], columns=['col', 'col'])

        with pytest.raises(ValueError, match='duplicate column names'):
            validate_dataset(df)

    def test_validate_passes_for_single_row(self):
        """Test validation passes for single row DataFrame"""
        df = pd.DataFrame({'col': [1]})

        result = validate_dataset(df)
        assert result is True

    def test_validate_passes_for_single_column(self):
        """Test validation passes for single column DataFrame"""
        df = pd.DataFrame({'col': [1, 2, 3]})

        result = validate_dataset(df)
        assert result is True


# ==================== validate_columns Tests ====================


class TestValidateColumns:
    """Tests for validate_columns function"""

    def test_validate_with_no_specified_columns(self, valid_dataframe):
        """Test validation with no columns specified"""
        result = validate_columns(valid_dataframe)
        assert result is True

    def test_validate_with_valid_numerical_columns(self, valid_dataframe):
        """Test validation with valid numerical columns"""
        result = validate_columns(
            valid_dataframe,
            numerical_columns=['num_col1', 'num_col2']
        )
        assert result is True

    def test_validate_with_valid_categorical_columns(self, valid_dataframe):
        """Test validation with valid categorical columns"""
        result = validate_columns(
            valid_dataframe,
            categorical_columns=['cat_col1', 'cat_col2']
        )
        assert result is True

    def test_validate_with_valid_target_column(self, valid_dataframe):
        """Test validation with valid target column"""
        result = validate_columns(
            valid_dataframe,
            target_column='target'
        )
        assert result is True

    def test_validate_with_all_columns_specified(self, valid_dataframe):
        """Test validation with all column types specified"""
        result = validate_columns(
            valid_dataframe,
            numerical_columns=['num_col1', 'num_col2'],
            categorical_columns=['cat_col1', 'cat_col2'],
            target_column='target'
        )
        assert result is True

    def test_validate_raises_error_for_missing_numerical_columns(self, valid_dataframe):
        """Test validation raises error for missing numerical columns"""
        with pytest.raises(ValueError, match='Numerical columns not found'):
            validate_columns(
                valid_dataframe,
                numerical_columns=['missing_col']
            )

    def test_validate_raises_error_for_missing_categorical_columns(self, valid_dataframe):
        """Test validation raises error for missing categorical columns"""
        with pytest.raises(ValueError, match='Categorical columns not found'):
            validate_columns(
                valid_dataframe,
                categorical_columns=['missing_col']
            )

    def test_validate_raises_error_for_missing_target_column(self, valid_dataframe):
        """Test validation raises error for missing target column"""
        with pytest.raises(ValueError, match='Target column.*not found'):
            validate_columns(
                valid_dataframe,
                target_column='missing_target'
            )

    def test_validate_raises_error_for_overlapping_columns(self, valid_dataframe):
        """Test validation raises error for overlapping num/cat columns"""
        with pytest.raises(ValueError, match='cannot be both numerical and categorical'):
            validate_columns(
                valid_dataframe,
                numerical_columns=['num_col1'],
                categorical_columns=['num_col1']  # Same column
            )

    def test_validate_with_multiple_missing_numerical_columns(self, valid_dataframe):
        """Test validation with multiple missing numerical columns"""
        with pytest.raises(ValueError, match='Numerical columns not found'):
            validate_columns(
                valid_dataframe,
                numerical_columns=['missing1', 'missing2']
            )

    def test_validate_with_multiple_missing_categorical_columns(self, valid_dataframe):
        """Test validation with multiple missing categorical columns"""
        with pytest.raises(ValueError, match='Categorical columns not found'):
            validate_columns(
                valid_dataframe,
                categorical_columns=['missing1', 'missing2']
            )

    def test_validate_with_mixed_valid_and_invalid_numerical(self, valid_dataframe):
        """Test validation with mix of valid and invalid numerical columns"""
        with pytest.raises(ValueError, match='Numerical columns not found'):
            validate_columns(
                valid_dataframe,
                numerical_columns=['num_col1', 'missing_col']
            )

    def test_validate_with_empty_column_lists(self, valid_dataframe):
        """Test validation with empty column lists"""
        result = validate_columns(
            valid_dataframe,
            numerical_columns=[],
            categorical_columns=[]
        )
        assert result is True


# ==================== validate_generator_params Tests ====================


class TestValidateGeneratorParams:
    """Tests for validate_generator_params function"""

    def test_validate_gaussian_method_with_positive_samples(self):
        """Test validation with gaussian method and positive samples"""
        result = validate_generator_params('gaussian', 100)
        assert result is True

    def test_validate_raises_error_for_unknown_method(self):
        """Test validation raises error for unknown method"""
        with pytest.raises(ValueError, match='Unknown method'):
            validate_generator_params('unknown_method', 100)

    def test_validate_raises_error_for_zero_samples(self):
        """Test validation raises error for zero samples"""
        with pytest.raises(ValueError, match='must be positive'):
            validate_generator_params('gaussian', 0)

    def test_validate_raises_error_for_negative_samples(self):
        """Test validation raises error for negative samples"""
        with pytest.raises(ValueError, match='must be positive'):
            validate_generator_params('gaussian', -10)

    def test_validate_gaussian_with_large_num_samples(self):
        """Test validation with gaussian and large num_samples"""
        result = validate_generator_params('gaussian', 1000000)
        assert result is True

    def test_validate_ctgan_raises_not_implemented(self):
        """Test validation raises NotImplementedError for ctgan"""
        with pytest.raises(NotImplementedError, match='CTGAN'):
            validate_generator_params('ctgan', 100)

    def test_validate_tvae_raises_not_implemented(self):
        """Test validation raises NotImplementedError for tvae"""
        with pytest.raises(NotImplementedError, match='TVAE'):
            validate_generator_params('tvae', 100)

    def test_validate_smote_raises_not_implemented(self):
        """Test validation raises NotImplementedError for smote"""
        with pytest.raises(NotImplementedError, match='SMOTE'):
            validate_generator_params('smote', 100)

    def test_validate_with_additional_kwargs(self):
        """Test validation with additional keyword arguments"""
        result = validate_generator_params(
            'gaussian', 100,
            random_state=42,
            custom_param='value'
        )
        assert result is True

    def test_validate_error_message_shows_valid_methods(self):
        """Test that error message shows list of valid methods"""
        with pytest.raises(ValueError) as exc_info:
            validate_generator_params('invalid', 100)

        error_msg = str(exc_info.value)
        assert 'gaussian' in error_msg
        assert 'ctgan' in error_msg
        assert 'tvae' in error_msg
        assert 'smote' in error_msg


# ==================== validate_file_path Tests ====================


class TestValidateFilePath:
    """Tests for validate_file_path function"""

    def test_validate_path_string_to_path_object(self, temp_dir):
        """Test validation converts string to Path object"""
        path_str = os.path.join(temp_dir, 'test.txt')

        result = validate_file_path(path_str)

        assert isinstance(result, Path)

    def test_validate_path_object_remains_path_object(self, temp_dir):
        """Test validation with Path object input"""
        path_obj = Path(temp_dir) / 'test.txt'

        result = validate_file_path(path_obj)

        assert isinstance(result, Path)

    def test_validate_with_existing_parent_directory(self, temp_dir):
        """Test validation with existing parent directory"""
        path = os.path.join(temp_dir, 'new_file.txt')

        result = validate_file_path(path)

        assert result == Path(path)

    def test_validate_raises_error_for_nonexistent_parent(self):
        """Test validation raises error for nonexistent parent directory"""
        path = '/nonexistent/directory/file.txt'

        with pytest.raises(ValueError, match='Parent directory does not exist'):
            validate_file_path(path)

    def test_validate_with_must_exist_true_for_existing_file(self, temp_dir):
        """Test validation with must_exist=True for existing file"""
        # Create a temporary file
        file_path = os.path.join(temp_dir, 'existing.txt')
        Path(file_path).touch()

        result = validate_file_path(file_path, must_exist=True)

        assert result == Path(file_path)

    def test_validate_with_must_exist_true_for_nonexistent_file(self, temp_dir):
        """Test validation with must_exist=True for nonexistent file"""
        path = os.path.join(temp_dir, 'nonexistent.txt')

        with pytest.raises(FileNotFoundError, match='File not found'):
            validate_file_path(path, must_exist=True)

    def test_validate_with_must_exist_false_for_nonexistent_file(self, temp_dir):
        """Test validation with must_exist=False for nonexistent file"""
        path = os.path.join(temp_dir, 'new_file.txt')

        result = validate_file_path(path, must_exist=False)

        assert result == Path(path)

    def test_validate_with_nested_directory(self, temp_dir):
        """Test validation with nested directory structure"""
        nested_dir = os.path.join(temp_dir, 'subdir')
        os.makedirs(nested_dir)
        path = os.path.join(nested_dir, 'file.txt')

        result = validate_file_path(path)

        assert result == Path(path)


# ==================== Integration Tests ====================


class TestIntegration:
    """Integration tests for validator combinations"""

    def test_validate_dataset_and_columns_together(self, valid_dataframe):
        """Test dataset and column validation together"""
        # First validate dataset
        validate_dataset(valid_dataframe)

        # Then validate columns
        result = validate_columns(
            valid_dataframe,
            numerical_columns=['num_col1'],
            categorical_columns=['cat_col1'],
            target_column='target'
        )

        assert result is True

    def test_full_validation_workflow(self, valid_dataframe, temp_dir):
        """Test complete validation workflow"""
        # 1. Validate dataset
        validate_dataset(valid_dataframe)

        # 2. Validate columns
        validate_columns(
            valid_dataframe,
            numerical_columns=['num_col1', 'num_col2'],
            categorical_columns=['cat_col1', 'cat_col2'],
            target_column='target'
        )

        # 3. Validate generator params
        validate_generator_params('gaussian', 100)

        # 4. Validate file path
        output_path = os.path.join(temp_dir, 'output.csv')
        result = validate_file_path(output_path)

        assert isinstance(result, Path)


# ==================== Edge Cases ====================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_validate_dataset_with_single_value(self):
        """Test validation with single cell DataFrame"""
        df = pd.DataFrame({'col': [1]})

        result = validate_dataset(df)
        assert result is True

    def test_validate_columns_with_unicode_names(self):
        """Test validation with unicode column names"""
        df = pd.DataFrame({
            'José': [1, 2, 3],
            '北京': [4, 5, 6],
            'café': [7, 8, 9],
        })

        result = validate_columns(
            df,
            numerical_columns=['José', '北京', 'café']
        )
        assert result is True

    def test_validate_generator_params_with_one_sample(self):
        """Test validation with num_samples=1"""
        result = validate_generator_params('gaussian', 1)
        assert result is True

    def test_validate_file_path_with_relative_path(self):
        """Test validation with relative path"""
        # Use current directory as base
        path = './test_file.txt'

        result = validate_file_path(path)

        assert isinstance(result, Path)

    def test_validate_columns_all_none(self, valid_dataframe):
        """Test validation when all column parameters are None"""
        result = validate_columns(
            valid_dataframe,
            numerical_columns=None,
            categorical_columns=None,
            target_column=None
        )
        assert result is True

    def test_validate_dataset_with_many_columns(self):
        """Test validation with DataFrame with many columns"""
        df = pd.DataFrame({f'col{i}': [1, 2, 3] for i in range(100)})

        result = validate_dataset(df)
        assert result is True

    def test_validate_dataset_with_many_rows(self):
        """Test validation with DataFrame with many rows"""
        df = pd.DataFrame({'col': range(10000)})

        result = validate_dataset(df)
        assert result is True
