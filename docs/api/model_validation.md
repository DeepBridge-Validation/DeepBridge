# Model Validation API Reference

This document provides detailed information about the `ModelValidation` class and its methods.

## ModelValidation

::: deepbridge.model_validation.ModelValidation
    handler: python
    selection:
      members:
        - __init__
        - add_data
        - add_model
        - save_model
        - load_model
        - save_metrics
        - get_experiment_info

## Class Overview

```python
from deepbridge.model_validation import ModelValidation

# Create an instance
validation = ModelValidation(
    experiment_name="my_experiment",
    save_path="./experiments"
)
```

## Constructor Parameters

### `__init__(experiment_name: str = "default_experiment", save_path: Optional[str] = None)`

Create a new ModelValidation instance.

**Parameters:**

- `experiment_name` (str, optional):
    - Name of the experiment
    - Used for identification and organization
    - Default: "default_experiment"

- `save_path` (str, optional):
    - Path where experiment files will be saved
    - If None, uses "./experiments/{experiment_name}"
    - Default: None

**Returns:**
- ModelValidation instance

**Example:**
```python
# Basic usage
validation = ModelValidation("experiment1")

# Custom save path
validation = ModelValidation(
    experiment_name="experiment1",
    save_path="./my_experiments"
)
```

## Methods

### `add_data(X_train, y_train, X_test=None, y_test=None)`

Add training and test data to the experiment.

**Parameters:**

- `X_train` (Union[pd.DataFrame, np.ndarray]):
    - Training features
    - Can be DataFrame or numpy array

- `y_train` (Union[pd.Series, np.ndarray]):
    - Training labels
    - Can be Series or numpy array

- `X_test` (Union[pd.DataFrame, np.ndarray], optional):
    - Test features
    - Default: None

- `y_test` (Union[pd.Series, np.ndarray], optional):
    - Test labels
    - Default: None

**Example:**
```python
# Add only training data
validation.add_data(X_train, y_train)

# Add both training and test data
validation.add_data(X_train, y_train, X_test, y_test)
```

### `add_model(model: BaseEstimator, model_name: str, is_surrogate: bool = False)`

Add a model to the experiment.

**Parameters:**

- `model` (BaseEstimator):
    - Scikit-learn compatible model
    - Must implement fit/predict interface

- `model_name` (str):
    - Unique identifier for the model
    - Used for saving and loading

- `is_surrogate` (bool, optional):
    - Whether this is a surrogate model
    - Default: False

**Example:**
```python
# Add main model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
validation.add_model(model, "random_forest_v1")

# Add surrogate model
surrogate = LogisticRegression()
validation.add_model(surrogate, "surrogate_v1", is_surrogate=True)
```

### `save_model(model_name: str, is_surrogate: bool = False)`

Save a model to disk.

**Parameters:**

- `model_name` (str):
    - Name of the model to save
    - Must match name used in add_model

- `is_surrogate` (bool, optional):
    - Whether this is a surrogate model
    - Default: False

**Example:**
```python
# Save main model
validation.save_model("random_forest_v1")

# Save surrogate model
validation.save_model("surrogate_v1", is_surrogate=True)
```

### `load_model(model_name: str, is_surrogate: bool = False)`

Load a saved model.

**Parameters:**

- `model_name` (str):
    - Name of the model to load
    - Must exist in experiment directory

- `is_surrogate` (bool, optional):
    - Whether this is a surrogate model
    - Default: False

**Returns:**
- BaseEstimator: The loaded model

**Example:**
```python
# Load main model
model = validation.load_model("random_forest_v1")

# Load surrogate model
surrogate = validation.load_model("surrogate_v1", is_surrogate=True)
```

### `save_metrics(metrics: Dict, model_name: str)`

Save metrics for a specific model.

**Parameters:**

- `metrics` (Dict):
    - Dictionary of metrics to save
    - Can include any serializable values

- `model_name` (str):
    - Name of the model these metrics belong to

**Example:**
```python
# Save performance metrics
metrics = {
    "accuracy": 0.95,
    "roc_auc": 0.98,
    "f1_score": 0.96
}
validation.save_metrics(metrics, "random_forest_v1")
```

### `get_experiment_info()`

Get information about the experiment.

**Returns:**
- Dict containing experiment metadata and metrics

**Example:**
```python
# Get experiment information
info = validation.get_experiment_info()
print(f"Number of models: {info['n_models']}")
print(f"Metrics: {info['metrics']}")
```

## Properties

- `experiment_name` (str): Name of the experiment
- `save_path` (Path): Path where experiment files are saved
- `models` (Dict[str, BaseEstimator]): Dictionary of main models
- `surrogate_models` (Dict[str, BaseEstimator]): Dictionary of surrogate models
- `metrics` (Dict[str, Dict]): Dictionary of model metrics

## Example Usage

```python
from deepbridge.model_validation import ModelValidation
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create experiment
validation = ModelValidation("my_experiment")

# Add data
validation.add_data(X_train, y_train, X_test, y_test)

# Train and add model
model = RandomForestClassifier()
model.fit(X_train, y_train)
validation.add_model(model, "rf_v1")

# Save model
validation.save_model("rf_v1")

# Calculate and save metrics
y_pred = model.predict(X_test)
metrics = {"accuracy": accuracy_score(y_test, y_pred)}
validation.save_metrics(metrics, "rf_v1")

# Get experiment info
info = validation.get_experiment_info()
```