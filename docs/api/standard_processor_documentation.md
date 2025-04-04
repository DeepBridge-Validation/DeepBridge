# StandardProcessor Class Documentation

## Overview

The `StandardProcessor` class implements the `BaseProcessor` interface to provide a comprehensive solution for common data preprocessing tasks in machine learning workflows. It handles numerical and categorical data transformations, including scaling, missing value imputation, outlier handling, and categorical encoding.

## Key Features

- **Automated Column Type Detection**: Automatically distinguishes between numerical and categorical columns
- **Flexible Scaling Options**: Supports multiple scaling techniques for numerical data
- **Missing Value Handling**: Implements imputation strategies for both numerical and categorical data
- **Outlier Treatment**: Provides IQR-based outlier capping functionality
- **Categorical Encoding**: Offers multiple encoding strategies for categorical variables
- **Framework Compatibility**: Works with both pandas DataFrames and NumPy arrays

## Class Structure

### Initialization

```python
StandardProcessor(
    verbose: bool = False,
    scaler_type: str = 'standard',
    handle_missing: bool = True,
    handle_outliers: bool = False,
    categorical_encoding: str = 'onehot'
)
```

Parameters:
- `verbose`: Whether to print progress information
- `scaler_type`: Type of scaler to use ('standard', 'minmax', 'robust', or None)
- `handle_missing`: Whether to handle missing values
- `handle_outliers`: Whether to handle outliers
- `categorical_encoding`: Type of encoding for categorical features ('onehot', 'label', 'ordinal', or None)

### Main Processing Method

#### `process()`

```python
def process(
    self, 
    data: Union[pd.DataFrame, np.ndarray], 
    **kwargs
) -> Union[pd.DataFrame, np.ndarray]
```

The main method that orchestrates the entire data processing workflow.

Parameters:
- `data`: Data to process (DataFrame or NumPy array)
- `**kwargs`: Additional processing parameters:
  - `fit`: Whether to fit transformers (default: False)
  - `numerical_columns`: List of numerical column names
  - `categorical_columns`: List of categorical column names
  - `target_column`: Target column name (excluded from processing)

Returns:
- Processed data in the same format as the input (DataFrame or NumPy array)

### Helper Methods

#### `_process_numerical()`

```python
def _process_numerical(
    self, 
    data: pd.DataFrame, 
    columns: List[str], 
    fit: bool
) -> pd.DataFrame
```

Handles numerical data preprocessing including missing value imputation, outlier handling, and scaling.

#### `_process_categorical()`

```python
def _process_categorical(
    self, 
    data: pd.DataFrame, 
    columns: List[str], 
    fit: bool
) -> pd.DataFrame
```

Handles categorical data preprocessing including missing value handling and encoding.

#### `_infer_column_types()`

```python
def _infer_column_types(
    self, 
    data: pd.DataFrame, 
    columns: List[str], 
    target_column: Optional[str] = None
) -> None
```

Automatically distinguishes between numerical and categorical columns based on data types and cardinality.

## Data Processing Capabilities

### Numerical Data Processing

1. **Missing Value Imputation**:
   - Uses scikit-learn's `SimpleImputer` with mean strategy for numerical data
   - Applied when `handle_missing=True`

2. **Outlier Handling**:
   - Uses the Interquartile Range (IQR) method to detect outliers
   - Caps outliers at Q1 - 1.5*IQR and Q3 + 1.5*IQR
   - Applied when `handle_outliers=True`

3. **Scaling**:
   - Supports multiple scaling techniques through scikit-learn:
     - `StandardScaler`: Standardizes features to zero mean and unit variance
     - `MinMaxScaler`: Scales features to a specified range (default 0-1)
     - `RobustScaler`: Scales features using statistics robust to outliers
   - Selected via the `scaler_type` parameter

### Categorical Data Processing

1. **Missing Value Handling**:
   - Fills missing values with the most common value in the column
   - Applied when `handle_missing=True`

2. **Categorical Encoding**:
   - Supports multiple encoding strategies:
     - `onehot`: Creates binary columns for each category (one-hot encoding)
     - `label`: Assigns an integer to each category (label encoding)
   - Selected via the `categorical_encoding` parameter

## Column Type Inference

The class can automatically detect column types based on:

1. **Data Type**: 
   - Categorical: Objects, categorical types, and boolean types
   - Numerical: Integer and float types

2. **Cardinality**: 
   - Numerical columns with low cardinality (≤10 unique values) are treated as categorical
   - This helps identify categorical features stored as integers

## Usage Examples

### Basic Usage

```python
from deepbridge.core.standard_processor import StandardProcessor
import pandas as pd

# Sample data
data = pd.DataFrame({
    'age': [25, 30, 35, None, 45],
    'income': [50000, 60000, 75000, 90000, 120000],
    'gender': ['M', 'F', 'M', 'F', None],
    'category': [1, 2, 3, 2, 1]
})

# Create processor
processor = StandardProcessor(
    scaler_type='standard',
    handle_missing=True,
    handle_outliers=True,
    categorical_encoding='onehot'
)

# Process data (fit and transform)
processed_data = processor.process(data, fit=True)

print(processed_data)
```

### Using Pre-specified Column Types

```python
# Define column types explicitly
numerical_cols = ['age', 'income']
categorical_cols = ['gender', 'category']

# Process with explicit column types
processed_data = processor.process(
    data,
    fit=True,
    numerical_columns=numerical_cols,
    categorical_columns=categorical_cols
)
```

### Processing New Data After Fitting

```python
# New data with the same structure
new_data = pd.DataFrame({
    'age': [22, 40, 33],
    'income': [45000, 80000, 65000],
    'gender': ['F', 'M', 'F'],
    'category': [3, 1, 2]
})

# Process new data (transform only, no fitting)
processed_new_data = processor.process(new_data, fit=False)
```

### Working with NumPy Arrays

```python
import numpy as np

# Sample array data
array_data = np.array([
    [25, 50000, 0, 1],
    [30, 60000, 1, 2],
    [35, 75000, 0, 3]
])

# Process array data
processed_array = processor.process(
    array_data,
    fit=True,
    numerical_columns=[0, 1],
    categorical_columns=[2, 3]
)
```

### Excluding Target Column

```python
# Process data while excluding the target column from transformations
processed_data = processor.process(
    data,
    fit=True,
    target_column='category'
)
```

## Implementation Notes

1. **Transformation Sequence**:
   - For numerical data: missing value imputation → outlier handling → scaling
   - For categorical data: missing value handling → encoding

2. **Fit vs. Transform**:
   - When `fit=True`, the processor learns parameters from the data
   - When `fit=False`, it applies previously learned parameters to new data

3. **Column Type Detection**:
   - Happens automatically if column types are not explicitly provided
   - Can be overridden by explicitly passing `numerical_columns` and `categorical_columns`

4. **Format Preservation**:
   - Input as DataFrame → Output as DataFrame
   - Input as NumPy array → Output as NumPy array

5. **Target Column Handling**:
   - Target column is excluded from all transformations when specified

## Best Practices

1. **Initial Fitting**:
   - Always use `fit=True` on the first call with training data
   - Use `fit=False` for subsequent transformation of test/validation data

2. **Column Specification**:
   - Explicitly specify column types when working with test data to ensure consistency

3. **Outlier Handling**:
   - Use with caution as it modifies the data distribution
   - Consider whether domain knowledge suggests outliers are errors or important signals

4. **Categorical Encoding Choice**:
   - Use `onehot` for models sensitive to ordinal relationships (random forests, etc.)
   - Use `label` for models that can handle ordinal features or when dimensionality is a concern

5. **Error Cases**:
   - May encounter errors when new categorical values appear in test data that weren't in training data
   - The processor attempts to handle this by assigning -1 to unknown categories