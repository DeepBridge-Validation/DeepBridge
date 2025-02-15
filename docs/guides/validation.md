# Model Validation Guide

This guide covers the Model Validation module of DeepBridge, which provides tools for managing and validating machine learning models.

## Overview

The Model Validation module helps you:
- Organize and track experiments
- Manage model versions
- Store and analyze metrics
- Compare model performances
- Handle surrogate models

## Basic Concepts

### Experiments

An experiment in DeepBridge is a container that holds:
- Training and test data
- Models
- Performance metrics
- Experiment metadata

### Model Types

The validation module supports:
- Standard models (main models)
- Surrogate models (simplified versions)
- Any scikit-learn compatible estimator

## Getting Started

### Creating an Experiment

```python
from deepbridge.model_validation import ModelValidation

# Basic setup
experiment = ModelValidation(
    experiment_name="customer_churn_prediction",
    save_path="./experiments"
)
```

### Adding Data

```python
# Prepare your data
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and split data
data = pd.read_csv("churn_data.csv")
X = data.drop("churn", axis=1)
y = data["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Add to experiment
experiment.add_data(
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    y_test=y_test
)
```

### Managing Models

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Train models
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Add models to experiment
experiment.add_model(rf_model, "random_forest_v1")
experiment.add_model(lr_model, "logistic_regression_v1")

# Save models
experiment.save_model("random_forest_v1")
experiment.save_model("logistic_regression_v1")
```

### Working with Metrics

```python
from sklearn.metrics import accuracy_score, roc_auc_score

# Calculate metrics
def calculate_metrics(model, X, y):
    predictions = model.predict(X)
    probas = model.predict_proba(X)[:, 1]
    
    return {
        "accuracy": accuracy_score(y, predictions),
        "roc_auc": roc_auc_score(y, probas)
    }

# Add metrics for each model
for model_name in ["random_forest_v1", "logistic_regression_v1"]:
    model = experiment.models[model_name]
    train_metrics = calculate_metrics(model, X_train, y_train)
    test_metrics = calculate_metrics(model, X_test, y_test)
    
    experiment.save_metrics({
        "train": train_metrics,
        "test": test_metrics
    }, model_name)
```

## Advanced Usage

### Working with Surrogate Models

```python
# Create and train a surrogate model
surrogate_model = LogisticRegression()
surrogate_model.fit(X_train, y_train)

# Add surrogate model
experiment.add_model(
    model=surrogate_model,
    model_name="surrogate_v1",
    is_surrogate=True
)

# Save surrogate model
experiment.save_model("surrogate_v1", is_surrogate=True)
```

### Experiment Analysis

```python
# Get comprehensive experiment information
info = experiment.get_experiment_info()

# Print summary
print(f"Experiment: {info['experiment_name']}")
print(f"Number of models: {info['n_models']}")
print(f"Number of surrogate models: {info['n_surrogate_models']}")

# Analyze data shapes
for name, shape in info['data_shapes'].items():
    if shape:
        print(f"{name}: {shape}")

# Compare model metrics
for model_name, metrics in info['metrics'].items():
    print(f"\nModel: {model_name}")
    print("Train metrics:", metrics['train'])
    print("Test metrics:", metrics['test'])
```

### Model Loading and Reuse

```python
# Load a saved model
loaded_model = experiment.load_model("random_forest_v1")

# Make predictions
predictions = loaded_model.predict(X_test)

# Add new metrics
new_metrics = calculate_metrics(loaded_model, X_test, y_test)
experiment.save_metrics(new_metrics, "random_forest_v1")
```

## Best Practices

### 1. Experiment Organization

```python
# Use descriptive names and structured paths
experiment = ModelValidation(
    experiment_name="churn_prediction/random_forest/v1",
    save_path="./experiments/churn"
)

# Add metadata
experiment.save_metrics({
    "description": "Random Forest model for churn prediction",
    "features": list(X_train.columns),
    "parameters": rf_model.get_params(),
    "data_version": "2024-02-14"
}, "experiment_metadata")
```

### 2. Data Validation

```python
def validate_data(X, y):
    """Validate input data before adding to experiment"""
    assert not X.isna().any().any(), "Data contains missing values"
    assert len(X) == len(y), "X and y must have same length"
    assert X.shape[1] == expected_features, "Unexpected number of features"
    
# Use in experiment
validate_data(X_train, y_train)
validate_data(X_test, y_test)
```

### 3. Model Versioning

```python
def create_model_version(experiment, model, version):
    """Add a new version of a model"""
    model_name = f"model_v{version}"
    experiment.add_model(model, model_name)
    experiment.save_model(model_name)
    
    metrics = calculate_metrics(model, X_test, y_test)
    experiment.save_metrics(metrics, model_name)
    
    return model_name
```

### 4. Performance Monitoring

```python
def monitor_performance(experiment, model_name):
    """Monitor model performance over time"""
    metrics = experiment.metrics[model_name]
    
    # Check for performance degradation
    if metrics['test']['accuracy'] < 0.8:
        print(f"Warning: Model {model_name} performance below threshold")
    
    return metrics
```

## Common Issues and Solutions

1. **Data Management**
   - Always validate data shapes and types
   - Handle missing values before adding to experiment
   - Document data preprocessing steps

2. **Model Storage**
   - Use consistent naming conventions
   - Include model metadata
   - Regularly clean up unused models

3. **Metric Tracking**
   - Track multiple metrics
   - Compare train vs test performance
   - Monitor changes over time

## Next Steps

- Learn about [Model Distillation](distillation.md)
- Explore the [CLI Usage](cli.md)
- Check the [API Reference](../api/model_validation.md)