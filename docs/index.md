# DeepBridge Documentation

Welcome to DeepBridge's documentation. DeepBridge is a comprehensive Python library designed to simplify machine learning model validation and distillation processes.

## Overview

DeepBridge provides tools and utilities for:
- Managing and validating machine learning models
- Implementing model distillation techniques
- Streamlining experiment workflows
- Optimizing model performance

## Getting Started

### Installation

```bash
pip install deepb
```

### Basic Usage

```python
from deepbridge.model_validation import ModelValidation
from deepbridge.model_distiller import ModelDistiller

# Create and manage experiments
experiment = ModelValidation("my_experiment")
experiment.add_data(X_train, y_train, X_test, y_test)

# Perform model distillation
distiller = ModelDistiller(model_type="gbm")
distiller.fit(X=features, probas=predictions)
```

## Core Components

### Model Validation Module
The validation module provides tools for:
- Experiment management
- Model versioning
- Performance tracking
- Surrogate model support

[Learn more about Model Validation →](./model_validation.md)

### Model Distillation Module
The distillation module includes:
- Multiple model architectures
- Performance optimization
- Automated training workflows
- Detailed metrics

[Learn more about Model Distillation →](./model_distiller.md)

### Command Line Interface
The CLI offers:
- Intuitive commands
- Rich output formatting
- Batch processing capabilities
- Experiment management

[Learn more about CLI →](./cli.md)

## User Guide

1. [Quick Start Tutorial](./tutorials/quickstart.md)
2. [Model Validation Guide](./guides/validation.md)
3. [Model Distillation Guide](./guides/distillation.md)
4. [CLI Usage Guide](./guides/cli.md)

## API Reference

- [ModelValidation Class](./api/model_validation.md)
- [ModelDistiller Class](./api/model_distiller.md)
- [CLI Reference](./api/cli.md)

## Examples

### Model Validation
```python
# Create experiment
experiment = ModelValidation("my_experiment")

# Add data and model
experiment.add_data(X_train, y_train, X_test, y_test)
experiment.add_model(model, "model_v1")

# Save and analyze
experiment.save_model("model_v1")
info = experiment.get_experiment_info()
```

### Model Distillation
```python
# Configure distiller
distiller = ModelDistiller(
    model_type="xgb",
    model_params={
        'n_estimators': 100,
        'max_depth': 5
    }
)

# Train and evaluate
distiller.fit(X=features, probas=predictions)
predictions = distiller.predict(X_new)
```

### Using the CLI
```bash
# Create experiment
deepbridge validation create my_experiment --path ./experiments

# Train model
deepbridge distill train gbm predictions.csv features.csv -s ./models

# Make predictions
deepbridge distill predict ./models/model.joblib new_data.csv -o predictions.csv
```

## Best Practices

### Experiment Management
- Use meaningful experiment names
- Maintain consistent directory structure
- Document model configurations
- Track experiment metrics

### Model Distillation
- Start with default parameters
- Monitor training progress
- Compare different architectures
- Validate results thoroughly

### Data Handling
- Validate input data
- Use appropriate data formats
- Maintain data versioning
- Handle missing values

## Advanced Topics

- [Custom Model Integration](./advanced/custom_models.md)
- [Performance Optimization](./advanced/optimization.md)
- [Experiment Tracking](./advanced/tracking.md)
- [Model Deployment](./advanced/deployment.md)

## Contributing

We welcome contributions! See our [Contributing Guide](./contributing.md) for details on:
- Setting up development environment
- Code style guidelines
- Pull request process
- Bug reporting

## Support

- GitHub Issues: For bug reports and feature requests
- Documentation: Comprehensive guides and API reference
- Examples: Practical use cases and tutorials
- Community: Discussion forums and chat

## License

DeepBridge is released under the MIT License. See [LICENSE](./LICENSE) for details.