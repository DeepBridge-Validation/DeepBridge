# Model Distillation Guide

This guide covers the Model Distillation module of DeepBridge, which helps you create simpler, more efficient versions of complex models while maintaining performance.

## What is Model Distillation?

Model distillation is a technique where a simpler model (student) learns to mimic the behavior of a more complex model (teacher). Benefits include:
- Reduced computational complexity
- Faster inference time
- Lower memory footprint
- Easier deployment

## Supported Models

DeepBridge supports several types of student models:

- **Gradient Boosting (GBM)**
  - Fast training and inference
  - Good performance on structured data
  - Highly interpretable

- **XGBoost (XGB)**
  - Advanced gradient boosting
  - High performance
  - Extensive optimization options

- **Multi-layer Perceptron (MLP)**
  - Neural network architecture
  - Flexible model capacity
  - Good for complex patterns

## Basic Usage

### Creating a Distiller

```python
from deepbridge.model_distiller import ModelDistiller

# Create a basic distiller
distiller = ModelDistiller(model_type="gbm")

# Create with custom parameters
distiller = ModelDistiller(
    model_type="xgb",
    model_params={
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5
    }
)
```

### Training Process

```python
import numpy as np
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2
)

# Get teacher model predictions
teacher_probas = teacher_model.predict_proba(X_train)

# Train the distilled model
distiller.fit(
    X=X_train,
    probas=teacher_probas,
    test_size=0.2,
    verbose=True
)

# Make predictions
predictions = distiller.predict(X_test)
```

### Evaluating Performance

```python
# Calculate detailed metrics
metrics = distiller.calculate_detailed_metrics(
    original_probas=teacher_model.predict_proba(X_test)[:, 1],
    distilled_probas=predictions,
    y_true=y_test
)

print("Performance Metrics:")
print(f"Original ROC-AUC: {metrics['original_roc_auc']:.4f}")
print(f"Distilled ROC-AUC: {metrics['distilled_roc_auc']:.4f}")
```

## Advanced Configuration

### Model-Specific Settings

#### Gradient Boosting
```python
gbm_params = {
    'n_estimators': 150,
    'learning_rate': 0.05,
    'max_depth': 4,
    'min_samples_split': 5,
    'subsample': 0.8
}

distiller = ModelDistiller(
    model_type="gbm",
    model_params=gbm_params
)
```

#### XGBoost
```python
xgb_params = {
    'n_estimators': 100,
    'learning_rate': 0.1,
    'max_depth': 5,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'objective': 'binary:logistic'
}

distiller = ModelDistiller(
    model_type="xgb",
    model_params=xgb_params
)
```

#### Multi-layer Perceptron
```python
mlp_params = {
    'hidden_layer_sizes': (100, 50),
    'activation': 'relu',
    'learning_rate': 'adaptive',
    'max_iter': 1000,
    'early_stopping': True
}

distiller = ModelDistiller(
    model_type="mlp",
    model_params=mlp_params
)
```

## Best Practices

### 1. Model Selection

Choose your student model based on:

```python
def select_student_model(data_size, feature_dim, complexity):
    if data_size < 10000:
        return ModelDistiller(model_type="gbm")
    elif complexity == "high":
        return ModelDistiller(
            model_type="xgb",
            model_params={'n_estimators': 200}
        )
    else:
        return ModelDistiller(model_type="mlp")
```

### 2. Training Strategy

```python
def train_with_validation(distiller, X, teacher_probas):
    # Split data for validation
    X_train, X_val, prob_train, prob_val = train_test_split(
        X, teacher_probas, test_size=0.2
    )
    
    # Train with validation monitoring
    distiller.fit(
        X=X_train,
        probas=prob_train,
        test_size=0.2,
        verbose=True
    )
    
    # Validate performance
    val_predictions = distiller.predict(X_val)
    return calculate_metrics(prob_val, val_predictions)
```

### 3. Model Persistence

```python
# Save and load models
def save_distilled_model(distiller, path, metadata=None):
    """Save distilled model with metadata"""
    distiller.save(path)
    
    if metadata:
        with open(f"{path}/metadata.json", "w") as f:
            json.dump(metadata, f)

def load_distilled_model(path):
    """Load distilled model and metadata"""
    distiller = ModelDistiller.load(path)
    
    metadata_path = f"{path}/metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path) "r") as f:
            metadata = json.load(f)
        return distiller, metadata
    
    return distiller
```

## Performance Optimization

### 1. Parameter Tuning

```python
from sklearn.model_selection import GridSearchCV

def optimize_gbm_params(X, probas):
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 4, 5]
    }
    
    base_model = GradientBoostingRegressor()
    grid_search = GridSearchCV(
        base_model, 
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error'
    )
    
    grid_search.fit(X, probas)
    return grid_search.best_params_
```

### 2. Model Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

def calibrate_distilled_model(distiller, X, y):
    """Calibrate model probabilities"""
    calibrated_model = CalibratedClassifierCV(
        distiller.model,
        cv=5,
        method='isotonic'
    )
    calibrated_model.fit(X, y)
    return calibrated_model
```

## Troubleshooting

Common issues and solutions:

1. **Poor Performance**
   - Increase model complexity
   - Try different model types
   - Check data quality
   - Validate teacher model predictions

2. **Slow Training**
   - Reduce number of estimators
   - Use simpler model architecture
   - Sample training data
   - Optimize hyperparameters

3. **Memory Issues**
   - Batch process large datasets
   - Reduce model complexity
   - Use memory-efficient parameters
   - Clean up unused objects

## Next Steps

- Explore [Model Validation](validation.md)
- Check the [CLI Guide](cli.md)
- See the [API Reference](../api/model_distiller.md)