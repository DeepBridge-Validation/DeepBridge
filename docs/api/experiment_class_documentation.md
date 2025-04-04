# Experiment Class Documentation

## Overview

The `Experiment` class serves as the central coordinator for machine learning experiments in the DeepBridge framework. It integrates various specialized components to orchestrate the entire experimentation workflow, from data management and model training to testing, evaluation, and visualization.

## Class Purpose

The `Experiment` class is designed to provide a high-level interface for conducting comprehensive machine learning experiments with an emphasis on model evaluation, distillation, and robust testing. It serves as the main entry point for users of the DeepBridge framework and coordinates the interactions between various specialized components.

## Key Components and Responsibilities

### Experiment Configuration

- Validates and maintains experiment configuration parameters
- Supports different experiment types (binary classification, regression, forecasting)
- Manages random seeds and test splits for reproducibility
- Controls verbosity and logging behavior

### Delegation Architecture

The class follows a delegation pattern, distributing responsibilities to specialized components:

- `DataManager`: Handles data preparation and splitting
- `ModelManager`: Manages model creation and distillation
- `TestRunner`: Coordinates execution of various test suites
- `ModelEvaluation`: Provides metrics calculation and evaluation
- `ReportGenerator`: Generates comprehensive reports
- `VisualizationManager`: Centralizes visualization capabilities

### Model Training and Distillation

- Supports model distillation to create lightweight models
- Provides different distillation approaches (surrogate, knowledge distillation)
- Handles hyperparameter optimization during model training
- Evaluates performance of distilled models against original models

### Test Coordination

- Manages execution of various test suites:
  - Robustness: Testing model stability under data perturbations
  - Uncertainty: Evaluating model uncertainty estimation
  - Resilience: Testing model performance under adverse conditions
  - Hyperparameter importance: Analyzing model sensitivity to parameter changes
- Offers different test configurations (quick, medium, full)
- Compares test results across alternative models

### Visualization and Reporting

- Provides methods for generating various visualizations:
  - Robustness comparison and distribution plots
  - Feature importance visualizations
  - Uncertainty estimation plots
  - Model comparison visualizations
- Generates comprehensive HTML reports summarizing experiment results
- Maintains backward compatibility through proxy properties

## Key Methods

### Initialization

```python
Experiment(
    dataset,
    experiment_type,
    test_size=0.2,
    random_state=42,
    config=None,
    auto_fit=None,
    tests=None
)
```

Creates a new experiment with specified configuration and dataset.

### Model Training

```python
fit(
    student_model_type=ModelType.LOGISTIC_REGRESSION,
    student_params=None,
    temperature=1.0,
    alpha=0.5,
    use_probabilities=True,
    n_trials=50,
    validation_split=0.2,
    verbose=True,
    distillation_method="surrogate",
    **kwargs
)
```

Trains a model using distillation techniques on the provided dataset.

### Test Execution

```python
run_tests(config_name='quick')
```

Runs all specified tests with the given configuration level.

### Visualization Methods

The class provides numerous methods for generating visualizations, delegating to the `VisualizationManager`:

- `plot_robustness_comparison()`: Shows robustness across models
- `plot_robustness_distribution()`: Displays distribution of robustness scores
- `plot_feature_importance_robustness()`: Shows feature importance for robustness
- `plot_uncertainty_alpha_comparison()`: Compares uncertainty across alpha levels
- And many more specific visualization methods for different test types

### Report Generation

```python
save_report(report_path)
```

Generates and saves an HTML report with comprehensive experiment results.

## Usage Example

```python
from deepbridge.core.experiment import Experiment
from deepbridge.core.db_data import DBDataset

# Create a dataset
dataset = DBDataset(
    data=my_dataframe,
    target_column='target'
)

# Initialize experiment with robustness and uncertainty tests
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

# Run more detailed tests
results = experiment.run_tests(config_name='medium')

# Generate visualizations
robustness_plot = experiment.plot_robustness_comparison()

# Save comprehensive report
experiment.save_report('experiment_report.html')
```

## Integration Points

The `Experiment` class integrates with other DeepBridge components:

1. **Input**: Accepts a `DBDataset` containing data and optional models
2. **Output**: Produces visualizations, metrics, test results, and reports
3. **Extension**: Can be extended with new test types by updating the test lists

## Implementation Notes

- The class has been refactored to follow a delegation pattern, improving maintainability
- It maintains backward compatibility through careful property management
- Logging is configured to control verbosity, especially for optimization libraries
- Auto-fit capability detects when a model should be automatically created

## Advanced Features

- **Alternative Models**: Creates and tests alternative models alongside the primary model
- **Comprehensive Metrics**: Calculates extensive metrics for model evaluation
- **Configurable Testing**: Supports different testing depths to balance speed and thoroughness
- **Optimization Integration**: Handles hyperparameter optimization with Optuna

The `Experiment` class serves as the orchestration layer that brings together the various specialized components of the DeepBridge framework, providing a unified interface for conducting machine learning experiments with a focus on model robustness, uncertainty, and resilience.