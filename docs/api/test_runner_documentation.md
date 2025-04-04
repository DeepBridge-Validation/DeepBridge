# TestRunner Documentation

## Overview

The `TestRunner` class in the DeepBridge framework is responsible for coordinating the execution of various test suites across models. It acts as a central manager for running tests like robustness, uncertainty, resilience, and hyperparameter importance evaluations, ensuring consistent testing workflows and result organization.

## Class Definition

```python
class TestRunner:
    """
    Responsible for running various tests on models.
    Extracted from Experiment class to separate test execution responsibilities.
    """
    
    def __init__(
        self,
        dataset: 'DBDataset',
        alternative_models: dict,
        tests: t.List[str],
        X_train,
        X_test,
        y_train,
        y_test,
        verbose: bool = False
    ):
        """
        Initialize the test runner with dataset and model information.
        
        Args:
            dataset: The DBDataset containing model and data
            alternative_models: Dictionary of alternative models
            tests: List of tests to run
            X_train: Training features
            X_test: Testing features
            y_train: Training target
            y_test: Testing target
            verbose: Whether to print verbose output
        """
```

## Key Responsibilities

The `TestRunner` has the following primary responsibilities:

1. **Test Coordination**: Manages the execution of different test types across models
2. **Results Collection**: Aggregates and organizes test results from various test suites
3. **Alternative Model Testing**: Handles testing across both primary and alternative models
4. **Configuration Management**: Supports different testing configurations (quick, medium, full)

## Integration with Manager Classes

The `TestRunner` integrates with specialized manager classes that implement specific test types:

1. **RobustnessManager**: For testing model stability under perturbations
2. **UncertaintyManager**: For evaluating uncertainty estimation capabilities
3. **ResilienceManager**: For testing performance under adverse conditions
4. **HyperparameterManager**: For analyzing hyperparameter sensitivity

Each manager is instantiated with the appropriate dataset and configuration, and its `run_tests()` method is called to execute specific tests.

## Key Methods

### run_initial_tests()

```python
def run_initial_tests(self) -> dict:
    """Run the tests specified in self.tests using manager classes."""
```

This method runs quick versions of requested tests during initialization. It:
- Checks if a model is available in the dataset
- Creates appropriate manager instances for each requested test type
- Calls `run_tests()` on each manager with default settings
- Collects and organizes results in a structured dictionary
- Stores results for future reference

### run_tests(config_name)

```python
def run_tests(self, config_name: str = 'quick') -> dict:
    """
    Run all tests specified during initialization with the given configuration.
    
    Parameters:
    -----------
    config_name : str
        Name of the configuration to use: 'quick', 'medium', or 'full'
        
    Returns:
    --------
    dict : Dictionary with test results
    """
```

This method provides more control over test execution:
- Allows specifying the test intensity via `config_name`
- Supports more detailed test configurations
- Runs tests on both primary and alternative models
- Creates comprehensive result structures for each test type

### get_test_results(test_type)

```python
def get_test_results(self, test_type: str = None):
    """
    Get test results for a specific test type or all results.
    
    Args:
        test_type: The type of test to get results for. If None, returns all results.
        
    Returns:
        dict: Dictionary with test results
    """
```

This method provides access to previously run test results:
- Retrieves results for a specific test type if requested
- Returns all results if no specific type is specified
- Returns None if the requested test type has not been run

## Test Result Structure

The test results are organized in a consistent structure:

```
{
    'test_type': {
        'primary_model': {
            # Test-specific results for the primary model
        },
        'alternative_models': {
            'model_name_1': {
                # Test-specific results for alternative model 1
            },
            'model_name_2': {
                # Test-specific results for alternative model 2
            }
        }
    }
}
```

This structured format makes it easy to:
- Compare results across models
- Generate visualizations from results
- Include results in comprehensive reports

## Alternative Model Testing

For testing alternative models, the `TestRunner` uses a helper method to create consistent datasets:

```python
def _create_alternative_dataset(self, model):
    """
    Helper method to create a dataset with an alternative model.
    Uses DBDatasetFactory to ensure consistent dataset creation.
    """
    return DBDatasetFactory.create_for_alternative_model(
        original_dataset=self.dataset,
        model=model
    )
```

This ensures that alternative models are tested under the same conditions as the primary model.

## Usage Example

The `TestRunner` is typically used within the `Experiment` class, but can also be used independently:

```python
from deepbridge.core.experiment.test_runner import TestRunner
from deepbridge.core.db_data import DBDataset

# Create dataset with model
dataset = DBDataset(
    data=my_dataframe,
    target_column='target',
    model=my_model
)

# Initialize test runner with tests to run
test_runner = TestRunner(
    dataset=dataset,
    alternative_models={'random_forest': rf_model, 'xgboost': xgb_model},
    tests=['robustness', 'uncertainty'],
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    verbose=True
)

# Run tests with medium configuration
results = test_runner.run_tests(config_name='medium')

# Get specific test results
robustness_results = test_runner.get_test_results('robustness')
```

## Integration with Experiment Class

Within the `Experiment` class, the `TestRunner` is initialized during experiment creation:

```python
# From Experiment.__init__
self.test_runner = TestRunner(
    self.dataset,
    self.alternative_models,
    self.tests,
    self.X_train,
    self.X_test,
    self.y_train,
    self.y_test,
    self.verbose
)
```

The `Experiment` class then delegates test execution to the `TestRunner`:

```python
# From Experiment.run_tests
def run_tests(self, config_name: str = 'quick') -> dict:
    """
    Run all tests specified during initialization with the given configuration.
    """
    results = self.test_runner.run_tests(config_name)
    self.test_results.update(results)
    return results
```

## Integration with VisualizationManager

The `TestRunner` provides results to the `VisualizationManager`, which uses them to generate visualizations:

```python
# From VisualizationManager.get_robustness_results
def get_robustness_results(self):
    """
    Get the robustness test results.
    """
    results = self.test_runner.get_test_results('robustness')
    if results is None and "robustness" in self.test_runner.tests:
        # Run robustness tests if they were requested but not run yet
        robustness_manager = RobustnessManager(
            self.test_runner.dataset, 
            self.test_runner.alternative_models, 
            self.test_runner.verbose
        )
        results = robustness_manager.run_tests()
        self.test_runner.test_results['robustness'] = results
    
    return results
```

## Implementation Notes

- The `TestRunner` follows the Mediator pattern, coordinating between test managers and result consumers
- It uses lazy initialization of test results to avoid unnecessary computation
- The class supports extensibility through the addition of new test types
- It maintains a consistent result structure across different test types
- The implementation separates test execution from result visualization and reporting

## Extension Points

To extend the `TestRunner` with a new test type:

1. Add the new test type to the list of available tests
2. Create a specialized manager class for the new test type
3. Implement the test execution logic in the manager's `run_tests()` method
4. Update the `run_tests()` method in `TestRunner` to handle the new test type
5. Add result retrieval support in `get_test_results()`

The `TestRunner` provides a flexible and extensible architecture for running various tests on machine learning models, making it easy to evaluate model quality across multiple dimensions.