# DeepBridge

[![Documentation Status](https://deepbridge.readthedocs.io/en/latest/)](https://deepbridge.readthedocs.io/en/latest/)
[![CI](https://github.com/DeepBridge-Validation/DeepBridge/actions/workflows/pipeline.yaml)](https://github.com/DeepBridge-Validation/DeepBridge/actions/workflows/pipeline.yaml)


DeepBridge is a Python library for streamlining machine learning model validation and distillation processes. It provides tools to manage experiments, validate models, and create more efficient versions of complex models.

## Installation

You can install DeepBridge using pip:

```bash
pip install deepb
```

Or install from source:

```bash
git clone https://github.com/yourusername/deepbridge.git
cd deepbridge
pip install -e .
```

## Quick Start

### Model Validation
```python
from deepbridge.model_validation import ModelValidation

# Create experiment
experiment = ModelValidation("my_experiment")

# Add data
experiment.add_data(X_train, y_train, X_test, y_test)

# Add and save model
experiment.add_model(model, "model_v1")
experiment.save_model("model_v1")
```

### Model Distillation
```python
from deepbridge.model_distiller import ModelDistiller

# Create and train distilled model
distiller = ModelDistiller(model_type="gbm")
distiller.fit(X=features, probas=predictions)

# Make predictions
predictions = distiller.predict(X_new)
```

### Using the CLI
```bash
# Create experiment
deepbridge validation create my_experiment --path ./experiments

# Train distilled model
deepbridge distill train gbm predictions.csv features.csv -s ./models
```

## Features

- **Model Validation**
  - Experiment management
  - Metric tracking
  - Model versioning
  - Surrogate model support

- **Model Distillation**
  - Multiple model types (GBM, XGBoost, MLP)
  - Performance metrics
  - Optimization options
  - Easy model persistence

- **Command Line Interface**
  - Intuitive commands
  - Rich output formatting
  - Multiple data format support

## Requirements

- Python 3.8+
- Dependencies:
  ```
  numpy>=1.24.0
  pandas>=2.0.0
  scikit-learn>=1.2.0
  xgboost>=1.7.0
  scipy>=1.10.0
  typer[all]>=0.9.0
  rich>=13.0.0
  ```

## Documentation

For detailed documentation, visit [our documentation page](https://deepbridge.readthedocs.io/).

### Example Notebooks

Check out our [example notebooks](examples/) for detailed usage scenarios:
- Basic Model Validation
- Model Distillation Techniques
- CLI Usage Examples

## Contributing

We welcome contributions! Here's how you can help:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/deepbridge.git
cd deepbridge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install development dependencies
pip install -r requirements-dev.txt
```

## Running Tests

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use DeepBridge in your research, please cite:

```bibtex
@software{deepbridge2024,
  title = {DeepBridge: A Python Library for Model Validation and Distillation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/deepbridge}
}
```

## Acknowledgments

- Thanks to all contributors
- Inspired by best practices in model optimization
- Built with modern Python tools and libraries

## Contact

- GitHub Issues: For bugs and feature requests
- Email: your.email@example.com
- Twitter: [@YourHandle](https://twitter.com/YourHandle)