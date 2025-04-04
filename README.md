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

### Experiment Framework
- **Modular Architecture**: Component-based design with specialized managers
- **Comprehensive Testing**: Robustness, uncertainty, and resilience evaluation
- **Visualization System**: Integrated visualization capabilities
- **Reporting Engine**: Detailed HTML report generation

### Model Validation
- **Multi-faceted Evaluation**: Assess models across multiple dimensions
- **Alternative Model Comparison**: Generate and compare different model types
- **Metrics Analysis**: Comprehensive performance metrics
- **Visualization Tools**: Interactive plots for model analysis

### Model Distillation
- **Knowledge Distillation**: Transfer knowledge from complex to simpler models
- **Surrogate Modeling**: Create lightweight approximations of complex models
- **Hyperparameter Optimization**: Automated tuning of student models
- **Distribution Matching**: Ensure student models faithfully reproduce teacher distributions

### Advanced Analytics
- **Robustness Testing**: Evaluate model stability under perturbations
- **Uncertainty Quantification**: Assess model confidence and calibration
- **Resilience Analysis**: Test models under adverse conditions
- **Hyperparameter Importance**: Identify critical hyperparameters

## Architecture Overview

DeepBridge has been redesigned with a modular, component-based architecture:

```
┌─────────────────┐
│   Experiment    │
└───────┬─────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  ┌───────────┐  ┌────────────┐  ┌──────────────┐        │
│  │DataManager│  │ModelManager│  │TestRunner    │        │
│  └───────────┘  └────────────┘  └──────────────┘        │
│                                                         │
│  ┌───────────┐  ┌─────────────┐  ┌───────────────┐      │
│  │ModelEval  │  │ReportGen    │  │VisualizationMgr│     │
│  └───────────┘  └─────────────┘  └───────────────┘      │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Key Components
- **Experiment**: Central coordinator managing the experiment workflow
- **DataManager**: Handles data preparation and splitting
- **ModelManager**: Creates and manages models and distillation
- **TestRunner**: Coordinates test execution across managers
- **ModelEvaluation**: Calculates metrics and evaluates models
- **ReportGenerator**: Creates HTML reports with results
- **VisualizationManager**: Coordinates visualization creation

## Quick Start

### Comprehensive Experiment
```python
from deepbridge.core.experiment import Experiment
from deepbridge.core.db_data import DBDataset

# Create dataset
dataset = DBDataset(
    data=df,
    target_column='target',
    features=features
)

# Initialize experiment with tests
experiment = Experiment(
    dataset=dataset,
    experiment_type='binary_classification',
    tests=['robustness', 'uncertainty']
)

# Train a distilled model
experiment.fit(
    student_model_type='random_forest',
    distillation_method='knowledge_distillation',
    temperature=2.0
)

# Run tests and generate visualizations
experiment.run_tests(config_name='medium')
robustness_plot = experiment.plot_robustness_comparison()

# Save comprehensive report
experiment.save_report('experiment_report.html')
```

### Direct Model Distillation
```python
from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation

# Create distiller directly
distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_model_type='gbm',
    temperature=2.0,
    alpha=0.5
)

# Train the distilled model
distiller.fit(X_train, y_train)

# Make predictions
predictions = distiller.predict(X_test)
```

### Automated Distillation
```python
from deepbridge.auto_distiller import AutoDistiller
from deepbridge.db_data import DBDataset

# Create dataset with probabilities
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

# Run robustness tests
deepbridge validation test robustness my_experiment --config medium
```

## Documentation

Full documentation available at: [DeepBridge Documentation](https://deepbridge.readthedocs.io/)

The documentation includes:
- API Reference
- Architecture Guides
- Tutorial Notebooks
- Examples

## Requirements

- Python 3.8+
- Key Dependencies:
  - numpy
  - pandas
  - scikit-learn
  - xgboost
  - scipy
  - plotly
  - optuna

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