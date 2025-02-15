# Custom Models in DeepBridge

This guide explains how to integrate custom models into DeepBridge for both model validation and distillation tasks.

## Overview

DeepBridge supports any model that follows the scikit-learn estimator interface. This guide will show you how to:
- Create custom model classes
- Integrate them with DeepBridge
- Use them in experiments
- Create custom distillation architectures

## Creating Custom Models

### Basic Custom Model

```python
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

class CustomModel(BaseEstimator, ClassifierMixin):
    """Example custom model implementation"""
    
    def __init__(self, hidden_size=100, learning_rate=0.01):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit the model to data."""
        # Your training logic here
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        # Your prediction logic here
        return predictions
    
    def predict_proba(self, X):
        """Return probability estimates."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        # Your probability prediction logic here
        return probabilities
```

### Custom Ensemble Model

```python
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class CustomEnsemble(BaseEstimator, ClassifierMixin):
    """Custom ensemble combining multiple models"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.models = {
            'rf': RandomForestClassifier(n_estimators=n_estimators),
            'lr': LogisticRegression()
        }
    
    def fit(self, X, y):
        """Fit all models in ensemble"""
        for name, model in self.models.items():
            model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Weighted average of model probabilities"""
        probas = []
        weights = {'rf': 0.7, 'lr': 0.3}
        
        for name, model in self.models.items():
            prob = model.predict_proba(X)
            probas.append(prob * weights[name])
        
        return np.sum(probas, axis=0)
```

## Integration with DeepBridge

### Using Custom Models in Validation

```python
from deepbridge.model_validation import ModelValidation

# Create experiment
experiment = ModelValidation("custom_model_experiment")

# Create and train custom model
custom_model = CustomModel(hidden_size=200)
custom_model.fit(X_train, y_train)

# Add to experiment
experiment.add_model(
    model=custom_model,
    model_name="custom_v1"
)

# Save model
experiment.save_model("custom_v1")
```

### Custom Distillation Models

```python
from deepbridge.model_distiller import ModelDistiller

class CustomDistiller(ModelDistiller):
    """Custom distillation implementation"""
    
    def __init__(self, model_params=None):
        super().__init__(model_type="custom")
        self.model = CustomModel(**(model_params or {}))
    
    def fit(self, X, probas, test_size=0.2, random_state=None):
        """Custom fitting process"""
        # Your custom distillation logic here
        self.model.fit(X, probas)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Custom prediction process"""
        return self.model.predict_proba(X)
```

## Advanced Usage

### Custom Model with Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

class CalibratedCustomModel(BaseEstimator, ClassifierMixin):
    """Custom model with probability calibration"""
    
    def __init__(self, base_model, cv=5):
        self.base_model = base_model
        self.cv = cv
        self.calibrated_model = None
    
    def fit(self, X, y):
        """Fit and calibrate model"""
        self.calibrated_model = CalibratedClassifierCV(
            self.base_model,
            cv=self.cv,
            method='isotonic'
        )
        self.calibrated_model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Return calibrated probabilities"""
        return self.calibrated_model.predict_proba(X)
```

### Custom Feature Transformation

```python
class TransformingModel(BaseEstimator, ClassifierMixin):
    """Model with custom feature transformation"""
    
    def __init__(self, base_model):
        self.base_model = base_model
        self.feature_means = None
    
    def transform_features(self, X):
        """Custom feature transformation"""
        # Example: Standardization with moving average
        if self.feature_means is None:
            self.feature_means = X.mean(axis=0)
        
        X_transformed = (X - self.feature_means) / X.std(axis=0)
        return X_transformed
    
    def fit(self, X, y):
        """Fit with transformed features"""
        X_transformed = self.transform_features(X)
        self.base_model.fit(X_transformed, y)
        return self
    
    def predict_proba(self, X):
        """Predict with transformed features"""
        X_transformed = self.transform_features(X)
        return self.base_model.predict_proba(X_transformed)
```

## Best Practices

### 1. Model Validation

```python
def validate_custom_model(model, X, y):
    """Validate custom model implementation"""
    try:
        # Check fit method
        model.fit(X, y)
        assert hasattr(model, 'is_fitted'), "Model missing is_fitted attribute"
        
        # Check predictions
        preds = model.predict(X)
        assert preds.shape[0] == X.shape[0], "Wrong prediction shape"
        
        # Check probabilities
        probas = model.predict_proba(X)
        assert probas.shape[1] == len(np.unique(y)), "Wrong probability shape"
        
        return True
    except Exception as e:
        print(f"Validation failed: {str(e)}")
        return False
```

### 2. Serialization Support

```python
class SerializableCustomModel(BaseEstimator, ClassifierMixin):
    """Custom model with proper serialization"""
    
    def __init__(self):
        self.parameters = {}
        self.is_fitted = False
    
    def __getstate__(self):
        """Support for pickling"""
        state = self.__dict__.copy()
        # Remove unpicklable entries
        return state
    
    def __setstate__(self, state):
        """Support for unpickling"""
        self.__dict__.update(state)
```

### 3. Performance Monitoring

```python
class MonitoredCustomModel(BaseEstimator, ClassifierMixin):
    """Custom model with performance monitoring"""
    
    def __init__(self):
        self.training_history = []
        self.prediction_times = []
    
    def fit(self, X, y):
        """Fit with performance tracking"""
        start_time = time.time()
        # Training logic here
        train_time = time.time() - start_time
        
        self.training_history.append({
            'train_time': train_time,
            'data_shape': X.shape
        })
        return self
    
    def predict_proba(self, X):
        """Predictions with timing"""
        start_time = time.time()
        probas = # Prediction logic
        self.prediction_times.append(time.time() - start_time)
        return probas
```

## Testing Custom Models

```python
import pytest
import numpy as np

def test_custom_model():
    """Test suite for custom model"""
    # Create test data
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    
    # Initialize model
    model = CustomModel()
    
    # Test fitting
    model.fit(X, y)
    assert model.is_fitted
    
    # Test predictions
    preds = model.predict(X)
    assert preds.shape == (100,)
    
    # Test probabilities
    probas = model.predict_proba(X)
    assert probas.shape == (100, 2)
```

## Next Steps

- Check [Model Validation](../guides/validation.md) for more on experiment management
- See [Model Distillation](../guides/distillation.md) for distillation techniques
- Review the [API Reference](../api/model_validation.md) for detailed documentation