# DataManager Documentation

## Overview

The `DataManager` class in the DeepBridge framework is responsible for data preparation, organization, and access throughout the experiment workflow. It provides a consistent interface for working with datasets, handling the splitting of data into training and testing sets, and managing probability data when available.

## Class Purpose

The primary purpose of the `DataManager` is to:

1. Centralize data preparation logic in a single component
2. Ensure consistent data splitting across experiment phases
3. Provide standardized access to data subsets
4. Handle probability data alongside features and targets

## Key Features

### Data Preparation

The `DataManager` handles the initial preparation of data for experiments:

- Performs train-test splitting using scikit-learn's `train_test_split` function
- Maintains consistent random state for reproducibility
- Preserves dataset indices to ensure proper alignment with probability data
- Supports customizable test size proportion

### Data Access

The class provides standardized methods for accessing different data components:

- Features and targets for both training and testing sets
- Probability data when available (for models that provide probabilities)
- Combined data access through a single method call

### Probability Data Management

For experiments involving probability predictions:

- Aligns probability data with feature and target data using indices
- Splits probability data to match train-test splits
- Provides methods for converting probabilities to binary predictions

## Class Structure

### Initialization

```python
DataManager(dataset, test_size, random_state)
```

Parameters:
- `dataset`: The DBDataset instance containing features, target, and optional probabilities
- `test_size`: The proportion of data to use for testing (between 0 and 1)
- `random_state`: Seed for random number generation to ensure reproducible splits

### Main Methods

#### `prepare_data()`

```python
prepare_data() -> None
```

Prepares the data by performing train-test split on features and target, and on probability data if available. This method is typically called during initialization of the `DataManager`.

#### `get_dataset_split()`

```python
get_dataset_split(dataset: str = 'train') -> tuple
```

Gets the features, target, and probabilities (if available) for the specified dataset split.

Parameters:
- `dataset`: Either 'train' or 'test', specifying which split to retrieve

Returns:
- A tuple containing (X, y, probabilities) for the specified split

#### `get_binary_predictions()`

```python
get_binary_predictions(probabilities: pd.DataFrame, threshold: float = 0.5) -> pd.Series
```

Converts probability predictions to binary predictions using a threshold.

Parameters:
- `probabilities`: DataFrame containing probability values
- `threshold`: Probability threshold for binary classification (default: 0.5)

Returns:
- Series of binary predictions (0 or 1)

### Attributes

- `X_train`, `X_test`: Feature data for training and testing
- `y_train`, `y_test`: Target data for training and testing
- `prob_train`, `prob_test`: Probability data for training and testing (if available)

## Usage Example

```python
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.data_manager import DataManager

# Create a dataset
dataset = DBDataset(
    data=my_dataframe,
    target_column='target'
)

# Initialize data manager
data_manager = DataManager(
    dataset=dataset,
    test_size=0.2,
    random_state=42
)

# Prepare data
data_manager.prepare_data()

# Get training data
X_train, y_train, prob_train = data_manager.get_dataset_split('train')

# Get testing data
X_test, y_test, prob_test = data_manager.get_dataset_split('test')

# Convert probabilities to binary predictions
if prob_test is not None:
    y_pred = data_manager.get_binary_predictions(prob_test, threshold=0.7)
```

## Integration Points

The `DataManager` integrates with other DeepBridge components:

1. **Input**: Takes a `DBDataset` containing features, target, and optional probability data
2. **Output**: Provides split data for training and testing to other components
3. **Usage**: Used by the `Experiment` class for initial data preparation and access

## Implementation Notes

- The class maintains the original indices from the input DataFrame to ensure proper alignment
- It handles the case where probability data may not be available
- Random state is used consistently to ensure reproducible splits
- The data preparation follows scikit-learn's conventions for train-test splitting

## Design Decisions

1. **Delegation Pattern**: The `DataManager` represents a specialized component extracted from the `Experiment` class for better separation of concerns.

2. **Stateful Design**: The class maintains state (the split data) to avoid redundant splitting operations.

3. **Minimal Functionality**: The class focuses specifically on data preparation and access, delegating other responsibilities to specialized components.

4. **Direct Property Access**: While getter methods are provided, the class also exposes data directly through properties for convenience and backward compatibility.

The `DataManager` class provides a clean, focused interface for data handling in the DeepBridge framework, ensuring consistent data preparation and access throughout the experiment workflow.