# Component Integration Guide

## Overview

DeepBridge has been redesigned with a modular, component-based architecture that allows for clean separation of concerns and flexible extension. This guide explains how the various components in the framework integrate with each other, providing a comprehensive understanding of the system's internal workings.

## Core Architecture

The architecture follows a delegation pattern, with the `Experiment` class serving as the central coordinator:

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

## Component Responsibilities and Integration

### Experiment Class

The `Experiment` class serves as the main entry point and coordinator for all components. It:

1. Initializes all necessary components during construction
2. Delegates specific tasks to specialized components
3. Provides a unified interface for users
4. Coordinates the interaction between components

```python
def __init__(self, dataset, experiment_type, test_size=0.2, ...):
    # Initialize components
    self.data_manager = DataManager(dataset, test_size, random_state)
    self.model_manager = ModelManager(dataset, self.experiment_type, self.verbose)
    self.model_evaluation = ModelEvaluation(self.experiment_type, self.metrics_calculator)
    self.report_generator = ReportGenerator()
    
    # Initialize test runner with necessary components
    self.test_runner = TestRunner(
        self.dataset,
        self.alternative_models,
        self.tests,
        self.X_train, self.X_test,
        self.y_train, self.y_test,
        self.verbose
    )
    
    # Initialize visualization manager
    self.visualization_manager = VisualizationManager(self.test_runner)
```

### DataManager Integration

The `DataManager` handles all data-related operations:

1. It receives the dataset from the `Experiment` constructor
2. It prepares train/test splits based on configuration
3. It provides access to data subsets (X_train, y_test, etc.) to other components
4. It handles any necessary data transformations

Integration points:
- `Experiment` uses DataManager's prepared data for model training and evaluation
- `ModelManager` uses DataManager's data to train models
- `TestRunner` uses DataManager's data to execute tests

```python
# In Experiment.__init__
self.data_manager = DataManager(dataset, test_size, random_state)
self.data_manager.prepare_data()
self.X_train, self.X_test = self.data_manager.X_train, self.data_manager.X_test
self.y_train, self.y_test = self.data_manager.y_train, self.data_manager.y_test
self.prob_train, self.prob_test = self.data_manager.prob_train, self.data_manager.prob_test
```

### ModelManager Integration

The `ModelManager` handles model creation and configuration:

1. It creates alternative models for comparison during initialization
2. It provides the `create_distillation_model()` method for model distillation
3. It interfaces with the model registry to instantiate appropriate model types

Integration points:
- `Experiment` calls ModelManager to create distillation models in `fit()`
- `Experiment` stores alternative models created by ModelManager
- `TestRunner` uses models created by ModelManager for testing

```python
# In Experiment.fit()
self.distillation_model = self.model_manager.create_distillation_model(
    distillation_method, 
    student_model_type, 
    student_params,
    temperature, 
    alpha, 
    use_probabilities, 
    n_trials, 
    validation_split
)
```

### TestRunner Integration

The `TestRunner` coordinates test execution across different test types:

1. It handles test execution based on configuration (quick, medium, full)
2. It provides a consistent interface for testing different aspects (robustness, uncertainty, etc.)
3. It stores and provides access to test results

Integration points:
- `Experiment` delegates test execution to TestRunner in `run_tests()`
- `VisualizationManager` retrieves test results from TestRunner
- Specialized managers (like RobustnessManager) are created by TestRunner

```python
# In Experiment.run_tests()
def run_tests(self, config_name: str = 'quick') -> dict:
    """Run all tests specified during initialization with the given configuration."""
    results = self.test_runner.run_tests(config_name)
    self.test_results.update(results)
    return results
```

### ModelEvaluation Integration

The `ModelEvaluation` class handles metric calculation and model comparison:

1. It receives experiment type and metrics calculator during initialization
2. It provides methods for evaluating models and calculating metrics
3. It supports comparison between models

Integration points:
- `Experiment` uses ModelEvaluation to assess distillation in `fit()`
- `Experiment` uses ModelEvaluation for model comparison in `compare_all_models()`

```python
# In Experiment.fit()
train_metrics = self.model_evaluation.evaluate_distillation(
    self.distillation_model, 'train', 
    self.X_train, self.y_train, self.prob_train
)
```

### VisualizationManager Integration

The `VisualizationManager` centralizes visualization capabilities:

1. It retrieves test results from the TestRunner
2. It provides access to visualizations through specialized methods
3. It delegates to appropriate specialized managers for visualization generation

Integration points:
- `Experiment` provides access to VisualizationManager methods
- VisualizationManager integrates with TestRunner to run tests if needed
- Specialized visualizer classes are used by VisualizationManager

```python
# In Experiment - delegation methods
def plot_robustness_comparison(self):
    """Get the plotly figure showing the comparison of robustness across models."""
    return self.visualization_manager.plot_robustness_comparison()
```

### ReportGenerator Integration

The `ReportGenerator` creates comprehensive reports:

1. It aggregates results from different components
2. It formats results into structured HTML reports
3. It saves reports to disk

Integration points:
- `Experiment` calls ReportGenerator in `save_report()`
- ReportGenerator uses results from various components to generate reports

```python
# In Experiment.save_report()
def save_report(self, report_path: str) -> str:
    """Generate and save an HTML report with all experiment results."""
    return self.report_generator.save_report(
        report_path,
        self.get_comprehensive_results()
    )
```

## Specialized Managers

The framework includes specialized manager classes that implement the `BaseManager` interface:

### BaseManager

The `BaseManager` defines the common interface for all manager components:

```python
class BaseManager(ABC):
    def __init__(self, dataset, alternative_models=None, verbose=False):
        self.dataset = dataset
        self.alternative_models = alternative_models or {}
        self.verbose = verbose
    
    @abstractmethod
    def run_tests(self, config_name: str = 'quick', **kwargs) -> dict:
        """Run standard tests on the primary model."""
        pass
    
    @abstractmethod
    def compare_models(self, config_name: str = 'quick', **kwargs) -> dict:
        """Compare test results across all models."""
        pass
```

Integration points:
- `TestRunner` creates and uses specialized managers to run tests
- Specialized managers produce structured results that follow a consistent format
- VisualizationManager retrieves results from TestRunner and generates visualizations

### Manager-Visualizer Integration

Each specialized manager integrates with corresponding visualizer classes:

1. Managers generate raw test results with a specific structure
2. Managers may include visualization data in their results
3. Visualizer classes consume these results to create standardized visualizations
4. VisualizationManager provides a consistent interface to access these visualizations

```
┌─────────────────┐     ┌──────────────────┐
│  RobustnessManager │────> │ RobustnessResults │
└────────┬────────┘     └────────┬─────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌──────────────────┐
│  RobustnessVisualizer │<───┤ VisualizationManager │
└─────────────────┘     └──────────────────┘
```

## Data Flow in Common Operations

### Model Training and Evaluation

1. `Experiment.fit()` requests model creation from ModelManager
2. ModelManager creates the appropriate distillation model
3. Experiment trains the model on data from DataManager
4. Experiment uses ModelEvaluation to evaluate model performance
5. Results are stored in the Experiment for later use

### Test Execution

1. `Experiment.run_tests()` delegates to TestRunner
2. TestRunner creates appropriate specialized managers
3. Specialized managers run tests and generate results
4. Results are returned to Experiment and stored
5. VisualizationManager can access these results for visualization

### Report Generation

1. `Experiment.save_report()` delegates to ReportGenerator
2. Experiment collects comprehensive results from all components
3. ReportGenerator formats these results into an HTML report
4. The report includes visualizations from VisualizationManager
5. The generated report is saved to disk

## Extending the Framework

The component-based architecture makes it easy to extend the framework:

### Adding a New Test Type

1. Create a new specialized manager implementing the BaseManager interface
2. Create a corresponding visualizer class
3. Update TestRunner to handle the new test type
4. Add methods to VisualizationManager to access the new visualizations
5. Add delegation methods to Experiment for the new visualization methods

### Adding a New Distillation Method

1. Implement the new distillation method
2. Update ModelManager to support creation of models with the new method
3. Ensure the new method produces results compatible with ModelEvaluation

### Adding a New Visualization Type

1. Update the appropriate visualizer class to support the new visualization
2. Add methods to VisualizationManager to access the new visualization
3. Add delegation methods to Experiment for the new visualization methods

## Testing Component Integration

When testing component integration, focus on these key interactions:

1. **Data Flow Testing**: Ensure data flows correctly between components
2. **Interface Compliance**: Verify components adhere to expected interfaces
3. **Error Handling**: Check that errors in one component are properly handled by others
4. **Configuration Propagation**: Confirm configuration parameters are correctly passed between components

## Best Practices for Working with Components

1. **Follow the Delegation Pattern**: Components should delegate specific responsibilities to specialized components
2. **Maintain Consistent Interfaces**: Ensure components adhere to expected interfaces
3. **Use Clear Result Structures**: Structure results consistently across components
4. **Follow Naming Conventions**: Use consistent naming for similar concepts across components
5. **Document Integration Points**: Clearly document how components integrate with each other

This comprehensive component integration guide provides a foundation for understanding how DeepBridge's architecture works and how to effectively extend it for new capabilities.