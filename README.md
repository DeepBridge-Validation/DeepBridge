# DeepBridge

[![Documentation Status](https://readthedocs.org/projects/deepbridge/badge/?version=latest)](https://deepbridge.readthedocs.io/en/latest/)
[![CI](https://github.com/DeepBridge-Validation/DeepBridge/actions/workflows/pipeline.yaml/badge.svg)](https://github.com/DeepBridge-Validation/DeepBridge/actions/workflows/pipeline.yaml)
[![PyPI version](https://badge.fury.io/py/deepbridge.svg)](https://badge.fury.io/py/deepbridge)

DeepBridge is a comprehensive Python library for advanced machine learning model validation, distillation, and performance analysis. It provides powerful tools to manage experiments, validate models, create more efficient model versions, and conduct in-depth performance evaluations.

## Installation

You can install DeepBridge using pip:

```bash
pip install deepbridge
```

Or install from source:

```bash
git clone https://github.com/DeepBridge-Validation/DeepBridge.git
cd deepbridge
pip install -e .
```

## Key Features

- **Model Validation**
  - Experiment tracking and management
  - Comprehensive model performance analysis
  - Advanced metric tracking
  - Model versioning support

- **Model Distillation**
  - Knowledge distillation across multiple model types
  - Advanced configuration options
  - Performance optimization
  - Probabilistic model compression

- **Advanced Analytics**
  - Detailed performance metrics
  - Distribution analysis
  - Visualization of model performance
  - Precision-recall trade-off analysis

## Quick Start

### Model Distillation
```python
from deepbridge.model_distiller import ModelDistiller

# Create and train distilled model
distiller = ModelDistiller(model_type="gbm")
distiller.fit(X=features, probas=predictions)

# Make predictions
predictions = distiller.predict(X_new)
```

### Automated Distillation
```python
from deepbridge.auto_distiller import AutoDistiller
from deepbridge.db_data import DBDataset

# Create dataset
dataset = DBDataset(
    data=df,
    target_column='target',
    features=features,
    prob_cols=['prob_class_0', 'prob_class_1']
)

# Run automated distillation
distiller = AutoDistiller(
    dataset=dataset,
    output_dir='results',
    test_size=0.2,
    n_trials=10
)
results = distiller.run(use_probabilities=True)
```

## Command-Line Interface
```bash
# Create experiment
deepbridge validation create my_experiment --path ./experiments

# Train distilled model
deepbridge distill train gbm predictions.csv features.csv -s ./models
```

## Requirements

- Python 3.8+
- Key Dependencies:
  - numpy
  - pandas
  - scikit-learn
  - xgboost
  - scipy
  - matplotlib

## Documentation

Full documentation available at: [DeepBridge Documentation](https://deepbridge.readthedocs.io/)

## Contributing

We welcome contributions! Please see our contribution guidelines for details on how to submit pull requests, report issues, and contribute to the project.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/DeepBridge-Validation/DeepBridge.git
cd deepbridge

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running Tests

```bash
pytest tests/
```

## License

MIT License

## Citation

If you use DeepBridge in your research, please cite:

```bibtex
@software{deepbridge2025,
  title = {DeepBridge: Advanced Model Validation and Distillation Library},
  author = {Gustavo Haase, Paulo Dourado},
  year = {2025},
  url = {https://github.com/DeepBridge-Validation/DeepBridge}
}
```

## Contact

- GitHub Issues: [DeepBridge Issues](https://github.com/DeepBridge-Validation/DeepBridge/issues)
- Email: gustavo.haase@gmail.com