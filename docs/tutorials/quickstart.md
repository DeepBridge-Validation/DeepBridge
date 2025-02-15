# Quick Start Guide

This guide will help you get started with DeepBridge by walking through common use cases and basic functionality.

## Installation

First, install DeepBridge using pip:

```bash
pip install deepbridge
```

## Basic Usage

Let's walk through some common use cases.

### 1. Model Validation

#### Setting Up an Experiment

```python
from deepbridge.model_validation import ModelValidation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Create a new experiment
experiment = ModelValidation(
    experiment_name="first_experiment",
    save_path="./experiments"
)

# Prepare your data
data = pd.read_csv("your_data.csv")
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Add data to the experiment
experiment.add_data(X_train, y_train, X_test, y_test)
```

#### Adding and Validating Models

```python
# Train a model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Add model to experiment
experiment.add_model(model, "rf_model_v1")

# Save the model
experiment.save_model("rf_model_v1")

# Get experiment information
info = experiment.get_experiment_info()
print(info)
```

### 2. Model Distillation

#### Basic Distillation

```python
from deepbridge.model_distiller import ModelDistiller
import numpy as np

# Create a distiller with default settings
distiller = ModelDistiller(model_type="gbm")

# Get predictions from your complex model
complex_model_probs = complex_model.predict_proba(X)

# Train the distilled model
distiller.fit(
    X=X_train,
    probas=complex_model_probs,
    test_size=0.2
)

# Make predictions with the distilled model
predictions = distiller.predict(X_test)
```

#### Customizing the Distillation

```python
# Custom model parameters
model_params = {
    'n_estimators': 150,
    'learning_rate': 0.05,
    'max_depth': 4
}

# Create a configured distiller
distiller = ModelDistiller(
    model_type="xgb",
    model_params=model_params,
    save_path="./models/distilled"
)

# Train with custom settings
distiller.fit(
    X=X_train,
    probas=complex_model_probs,
    test_size=0.3,
    verbose=True
)

# Save the distilled model
distiller.save("distilled_model_v1")
```

### 3. Using the CLI

The CLI provides easy access to common operations.

#### Model Validation Commands

```bash
# Create a new experiment
deepbridge validation create my_experiment --path ./experiments

# Add data to the experiment
deepbridge validation add-data \
    ./experiments/my_experiment \
    train_data.csv \
    --test-data test_data.csv \
    --target-column target
```

#### Model Distillation Commands

```bash
# Train a distilled model
deepbridge distill train gbm \
    predictions.csv \
    features.csv \
    --save-path ./models \
    --test-size 0.2

# Make predictions
deepbridge distill predict \
    ./models/model.joblib \
    new_data.csv \
    --output predictions.csv
```

## Common Patterns

### 1. Experiment Organization

```python
# Create structured experiments
experiment = ModelValidation(
    experiment_name="project_name/experiment_type/version",
    save_path="./experiments"
)

# Add experiment metadata
experiment.save_metrics({
    "model_type": "random_forest",
    "hyperparameters": model.get_params(),
    "training_date": "2024-02-14"
}, "rf_model_v1")
```

### 2. Model Comparison

```python
# Add multiple models
experiment.add_model(model1, "model_v1")
experiment.add_model(model2, "model_v2")

# Compare performances
info = experiment.get_experiment_info()
for model_name, metrics in info["metrics"].items():
    print(f"Model: {model_name}")
    print(f"Performance: {metrics}")
```

### 3. Efficient Distillation

```python
# Start with simpler models
distiller = ModelDistiller(
    model_type="gbm",
    model_params={"n_estimators": 50}
)

# Gradually increase complexity if needed
if performance_not_sufficient:
    distiller = ModelDistiller(
        model_type="xgb",
        model_params={
            "n_estimators": 100,
            "max_depth": 5
        }
    )
```

## Next Steps

- Learn more about [Model Validation](../guides/validation.md)
- Explore [Model Distillation](../guides/distillation.md)
- Check out the [CLI Guide](../guides/cli.md)
- See the [API Reference](../api/model_validation.md)

## Tips and Best Practices

1. **Data Management**
   - Always validate your data before adding to experiments
   - Keep consistent data formats
   - Document data preprocessing steps

2. **Model Organization**
   - Use clear naming conventions
   - Version your models systematically
   - Document model configurations

3. **Performance Optimization**
   - Start with simple models
   - Monitor training metrics
   - Validate results thoroughly

4. **Resource Management**
   - Clean up unused experiments
   - Monitor disk usage
   - Use appropriate model complexity