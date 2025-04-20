![DeepBridge Logo](/assets/images/logo.svg)

# DeepBridge: Advanced Machine Learning Model Validation and Distillation

[![PyPI version](https://badge.fury.io/py/deepbridge.svg)](https://badge.fury.io/py/deepbridge)
[![Documentation Status](https://readthedocs.org/projects/deepbridge/badge/?version=latest)](https://deepbridge.readthedocs.io/en/latest/)

## Why DeepBridge?

DeepBridge is a comprehensive Python library that revolutionizes machine learning model validation and distillation. Our key features include:

- 🚀 **Automated Model Validation**: Streamline experiment tracking and model comparison
- 🧠 **Intelligent Model Distillation**: Create lightweight, high-performance models
- 📊 **Advanced Metrics**: In-depth performance analysis
- 🔬 **Flexible Experiment Management**: Support for multiple model types and use cases

## Key Differentiators

- Seamless integration with scikit-learn
- Support for complex model architectures
- Automated hyperparameter optimization
- Robust performance tracking

## Installation

Install DeepBridge using pip:

```bash
pip install deepbridge
```

## Quick Start

### Model Validation

```python
from deepbridge.model_validation import ModelValidation

# Create a new experiment
experiment = ModelValidation("my_experiment")

# Add training and test data
experiment.add_data(X_train, y_train, X_test, y_test)

# Add a model to the experiment
experiment.add_model(model, "model_v1")

# Save the model
experiment.save_model("model_v1")
```

### Model Distillation

```python
from deepbridge.model_distiller import ModelDistiller

# Create a distiller
distiller = ModelDistiller(model_type="gbm")

# Train the distilled model
distiller.fit(X=features, probas=predictions)

# Make predictions
new_predictions = distiller.predict(X_test)
```

## Key Features

### Model Validation

- Experiment management and tracking
- Model versioning and storage
- Performance metrics calculation
- Support for surrogate models

### Model Distillation

- Support for multiple model types:
  - Gradient Boosting (GBM)
  - XGBoost
  - Multi-layer Perceptron (MLP)
- Automated model training
- Comprehensive performance metrics
- Easy model persistence

### Command Line Interface

```bash
# Create a new experiment
deepbridge validation create my_experiment --path ./experiments

# Train a distilled model
deepbridge distill train gbm predictions.csv features.csv -s ./models

# Make predictions
deepbridge distill predict ./models/model.joblib new_data.csv -o predictions.csv
```

## Next Steps

- Check out the [Quick Start Guide](tutorials/quickstart.md) for a detailed introduction
- Learn about [Model Validation](guides/validation.md)
- Explore [Model Distillation](guides/distillation.md)
- See the [API Reference](api/model_validation.md) for detailed documentation

## Contributing

We welcome contributions! Please see our [Contributing Guide](contributing.md) for details.

## License

DeepBridge is released under the MIT License. See the [License](license.md) file for more details.

## Support

- Report bugs and request features on [GitHub Issues](https://github.com/DeepBridge-Validation/DeepBridge/issues)
- Join our community discussions
- Check out our [documentation](https://deepbridge.readthedocs.io/)