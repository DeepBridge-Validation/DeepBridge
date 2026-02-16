<div align="center">
  <img src="https://raw.githubusercontent.com/DeepBridge-Validation/DeepBridge/master/.github/assets/logo.png" alt="DeepBridge Logo" width="500"/>

  # DeepBridge

  [![Documentation Status](https://readthedocs.org/projects/deepbridge/badge/?version=latest)](https://deepbridge.readthedocs.io/en/latest/)
  [![CI](https://github.com/DeepBridge-Validation/DeepBridge/actions/workflows/pipeline.yaml/badge.svg)](https://github.com/DeepBridge-Validation/DeepBridge/actions/workflows/pipeline.yaml)
  [![PyPI version](https://badge.fury.io/py/deepbridge.svg)](https://badge.fury.io/py/deepbridge)
  [![PyPI Downloads](https://static.pepy.tech/badge/deepbridge)](https://pepy.tech/projects/deepbridge)
  [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Development Status](https://img.shields.io/badge/status-beta-orange.svg)](https://github.com/DeepBridge-Validation/DeepBridge)
  [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
  [![codecov](https://codecov.io/gh/DeepBridge-Validation/DeepBridge/branch/master/graph/badge.svg?token=3cdc73fd-4db4-442c-86f8-5650e2fc114a)](https://codecov.io/gh/DeepBridge-Validation/DeepBridge)
</div>

> **⚠️ BREAKING CHANGES - DeepBridge v2.0**
>
> DeepBridge v2.0 has been refactored to focus on **Model Validation**.
>
> **Moved to separate packages:**
> - 📦 **Distillation** → [`deepbridge-distillation`](https://github.com/DeepBridge-Validation/deepbridge-distillation)
> - 📦 **Synthetic Data** → [`deepbridge-synthetic`](https://github.com/DeepBridge-Validation/deepbridge-synthetic)
>
> See [Migration Guide](desenvolvimento/refatoracao/GUIA_RAPIDO_MIGRACAO.md) for details.

DeepBridge is a comprehensive Python library for machine learning model validation and performance analysis. It provides powerful tools to manage experiments, validate models, and conduct in-depth performance evaluations.

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

- **Comprehensive Testing Framework**
  - Robustness testing with perturbation analysis
  - Uncertainty quantification using conformal prediction
  - Resilience testing under distribution shifts
  - Hyperparameter importance analysis
  - **Fairness testing and bias detection** (NEW!)
    - 15 fairness metrics (pre-training and post-training)
    - Auto-detection of sensitive attributes
    - EEOC compliance verification (80% rule)
    - Threshold analysis for fairness optimization
    - Interactive HTML reports with visualizations

- **Model Validation**
  - Experiment tracking and management
  - Comprehensive model performance analysis
  - Advanced metric tracking
  - Model versioning support

- **Model Distillation** → Moved to [`deepbridge-distillation`](https://github.com/DeepBridge-Validation/deepbridge-distillation)
  - Knowledge distillation across multiple model types
  - Automated distillation with hyperparameter optimization
  - Support for GBM, XGBoost, and neural networks
  - Performance optimization and model compression

- **Advanced Analytics & Reporting**
  - Interactive HTML reports with Plotly visualizations
  - Static reports for documentation
  - Detailed performance metrics and analysis
  - Multi-model comparison capabilities

- **Synthetic Data Generation** → Moved to [`deepbridge-synthetic`](https://github.com/DeepBridge-Validation/deepbridge-synthetic)
  - Gaussian Copula method
  - Privacy-preserving data synthesis
  - Quality metrics and validation
  - Standalone package (no dependencies on deepbridge)

## Quick Start

### Model Validation
```python
from deepbridge.core.experiment import Experiment
from deepbridge.db_data import DBDataset

# Create dataset
dataset = DBDataset(
    data=df,
    target_column='target',
    features=['feature1', 'feature2', 'feature3']
)

# Create experiment
experiment = Experiment(
    name='model_validation',
    dataset=dataset,
    models={'my_model': trained_model}
)

# Run validation tests
robustness_results = experiment.run_test('robustness', config='medium')
uncertainty_results = experiment.run_test('uncertainty', config='medium')

# Generate comprehensive report
experiment.generate_report('robustness', output_dir='./reports')
```

### Model Distillation

> **Note:** Distillation has moved to [`deepbridge-distillation`](https://github.com/DeepBridge-Validation/deepbridge-distillation)
>
> ```bash
> pip install deepbridge-distillation
> ```

```python
from deepbridge import DBDataset
from deepbridge_distillation import AutoDistiller

# Create dataset with predictions
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

### Fairness Testing
```python
from deepbridge.core.experiment import Experiment
from deepbridge.db_data import DBDataset

# Create dataset (model already trained)
dataset = DBDataset(
    data=df,
    target_column='approved',
    model=trained_model
)

# Create experiment with protected attributes
experiment = Experiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["fairness"],
    protected_attributes=['gender', 'race', 'age_group']
)

# Run fairness tests
fairness_result = experiment.run_fairness_tests(config='full')

# Check results
print(f"Overall Fairness Score: {fairness_result.overall_fairness_score:.3f}")
print(f"Critical Issues: {len(fairness_result.critical_issues)}")
print(f"EEOC Compliant: {fairness_result.overall_fairness_score >= 0.80}")

# Generate interactive HTML report
fairness_result.save_html('fairness_report.html', model_name='My Model')
```

## Command-Line Interface
```bash
# Run model validation
deepbridge validate --dataset data.csv --model model.pkl --tests all

# Generate reports
deepbridge report --results ./results --output ./reports --format interactive

# Train distilled model (requires deepbridge-distillation)
deepbridge distill train gbm predictions.csv features.csv -s ./models

# Generate synthetic data (requires deepbridge-synthetic)
deepbridge synthetic generate --data original.csv --method gaussian_copula --samples 10000
```

## Requirements

- Python 3.10-3.12
- Key Dependencies:
  - numpy >= 2.2.3
  - pandas >= 2.2.3
  - scikit-learn >= 1.6.1
  - xgboost >= 2.1.4
  - scipy >= 1.15.1
  - matplotlib >= 3.10.0
  - seaborn >= 0.13.2
  - plotly >= 6.0.0
  - optuna >= 4.2.1
  - jinja2 >= 3.1.5

## Documentation

Full documentation is available at: [DeepBridge Documentation](https://deepbridge.readthedocs.io/)

### Key Documentation Sections

- **[Getting Started](https://deepbridge.readthedocs.io/en/latest/tutorials/install/)** - Installation and basic examples
- **[User Guide](https://deepbridge.readthedocs.io/en/latest/guides/validation/)** - Core concepts and tutorials
- **[Technical Reference](https://deepbridge.readthedocs.io/en/latest/technical/implementation_guide/)** - Architecture and implementation details
- **[API Reference](https://deepbridge.readthedocs.io/en/latest/api/complete_reference/)** - Complete API documentation

### Quick Links

- [Installation Guide](https://deepbridge.readthedocs.io/en/latest/tutorials/install/)
- [Basic Examples](https://deepbridge.readthedocs.io/en/latest/tutorials/basic_examples/)
- [Complete Workflow](https://deepbridge.readthedocs.io/en/latest/tutorials/complete_workflow/)
- [FAQ](https://deepbridge.readthedocs.io/en/latest/resources/faq/)
- [Troubleshooting](https://deepbridge.readthedocs.io/en/latest/resources/troubleshooting/)

### Fairness Documentation

- [Fairness Tutorial (Step-by-Step)](docs/FAIRNESS_TUTORIAL.md) - Complete tutorial from basics to production
- [Best Practices Guide](docs/FAIRNESS_BEST_PRACTICES.md) - Guidelines for ethical ML and fairness
- [FAQ](docs/FAIRNESS_FAQ.md) - Common questions and troubleshooting
- [Complete Example](examples/fairness_complete_example.py) - End-to-end executable example

## Contributing

We welcome contributions! Please see our contribution guidelines for details on how to submit pull requests, report issues, and contribute to the project.

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

### Recent Updates

- **2025-11-03**: **NEW Fairness Module** - Complete fairness testing framework with 15 metrics, auto-detection of sensitive attributes, EEOC compliance checks, threshold analysis, and interactive HTML reports. Includes comprehensive documentation, tutorial, and examples.
- **2025-07-02**: Added comprehensive documentation including Implementation Guide, Testing Framework, Report Generation, and complete API Reference
- **2025-05-15**: Fixed static report chart URLs to properly use relative paths with `./` prefix for improved portability across different environments

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