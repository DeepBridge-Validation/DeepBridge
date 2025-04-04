# DBDataset Class Documentation

## Overview

The `DBDataset` class is a core component of the DeepBridge framework that manages training and test datasets along with optional models and predictions. It provides a unified interface for handling data regardless of its source format, supporting both pandas DataFrames and scikit-learn datasets.

## Key Features

- Handles both unified datasets and pre-split train/test datasets
- Supports scikit-learn dataset objects and pandas DataFrames
- Manages features, target variables, and categorical features
- Provides model integration and prediction generation
- Validates data consistency and input parameters

## Initialization

The `DBDataset` class can be initialized in several ways:

```python
DBDataset(
    data=None,                      # Single dataset to be split into train/test
    train_data=None,                # Pre-split training data
    test_data=None,                 # Pre-split test data
    target_column=None,             # Target variable column name
    features=None,                  # List of feature column names
    model_path=None,                # Path to a saved model file
    model=None,                     # Pre-loaded model object
    train_predictions=None,         # Prediction data for training set
    test_predictions=None,          # Prediction data for test set
    prob_cols=None,                 # Probability column names
    categorical_features=None,      # List of categorical feature names
    max_categories=None,            # Maximum number of categories for auto-detection
    dataset_name=None,              # Name identifier for the dataset
    test_size=0.2,                  # Test split ratio when using a unified dataset
    random_state=None               # Random seed for reproducibility
)
```

## Data Input Options

The class supports three main ways to provide data:

1. **Unified Dataset**: Provide a single dataset through the `data` parameter, which will be automatically split into train and test sets.
2. **Pre-split Datasets**: Provide separate `train_data` and `test_data` parameters.
3. **sklearn Datasets**: Provide scikit-learn dataset objects, which will be automatically converted to pandas DataFrames.

## Model Integration Options

There are three mutually exclusive ways to integrate models:

1. **Model Path**: Provide a file path to a saved model via `model_path`.
2. **Model Object**: Provide a pre-loaded model object via `model`.
3. **Prediction Columns**: If no model is available, provide prediction data via `prob_cols`.

## Key Methods

### Data Access

- `X`: Returns the features dataset
- `target`: Returns the target column values
- `original_prob`: Returns predictions DataFrame
- `train_data`: Returns the training dataset
- `test_data`: Returns the test dataset
- `features`: Returns list of feature names
- `categorical_features`: Returns list of categorical feature names
- `numerical_features`: Returns list of numerical feature names
- `target_name`: Returns name of target column
- `model`: Returns the loaded model if available

### Data Retrieval

- `get_feature_data(dataset='train')`: Get feature columns from the specified dataset
- `get_target_data(dataset='train')`: Get target column from the specified dataset

### Model Management

- `set_model(model_or_path)`: Load and set a model from file or directly set a model object

## Internal Processing

The class performs several key internal processing steps:

1. **Data Validation**: Checks that input data is valid and consistent
2. **Data Processing**: Handles unified or pre-split data formats
3. **Feature Management**: Manages feature lists and identifies categorical features
4. **Model Handling**: Loads models and generates predictions when possible
5. **Formatting**: Provides formatted information about the dataset

## Usage Examples

### Creating a Dataset from a Single DataFrame

```python
import pandas as pd
from deepbridge.core.db_data import DBDataset

# Create a sample DataFrame
df = pd.DataFrame({
    'feature1': [1, 2, 3, 4, 5],
    'feature2': ['a', 'b', 'c', 'd', 'e'],
    'target': [0, 1, 0, 1, 0]
})

# Create a DBDataset instance
dataset = DBDataset(
    data=df,
    target_column='target',
    test_size=0.2
)

# Access data
X_train = dataset.get_feature_data('train')
y_train = dataset.get_target_data('train')
```

### Creating a Dataset with Pre-split Data

```python
import pandas as pd
from deepbridge.core.db_data import DBDataset

# Create sample train and test DataFrames
train_df = pd.DataFrame({
    'feature1': [1, 2, 3],
    'feature2': ['a', 'b', 'c'],
    'target': [0, 1, 0]
})

test_df = pd.DataFrame({
    'feature1': [4, 5],
    'feature2': ['d', 'e'],
    'target': [1, 0]
})

# Create a DBDataset instance
dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target'
)
```

### Loading a Model

```python
from deepbridge.core.db_data import DBDataset
from sklearn.ensemble import RandomForestClassifier

# Create a dataset
dataset = DBDataset(
    data=df,
    target_column='target'
)

# Train a model
model = RandomForestClassifier()
model.fit(dataset.get_feature_data(), dataset.get_target_data())

# Set the model in the dataset
dataset.set_model(model)
```

## Error Handling

The class handles various error conditions:

- Invalid data input combinations
- Missing target columns
- Empty datasets
- Invalid categorical features
- Model loading failures
- Prediction generation failures

Each error provides a descriptive message to help diagnose the issue.