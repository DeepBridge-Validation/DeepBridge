# VisualizationManager Documentation

## Overview

The `VisualizationManager` class serves as a central coordinator for visualization operations within the DeepBridge framework. It provides a unified interface for accessing and managing visualizations across different test types, delegating visualization requests to the appropriate specialized visualizers. This component simplifies the generation and retrieval of visualizations from test results, making it easier to create comprehensive reports and interactive dashboards.

## Class Definition

```python
class VisualizationManager:
    """
    Manages visualization and retrieval of test results.
    Extracted from Experiment class to centralize visualization responsibilities.
    """
    
    def __init__(self, test_runner: TestRunner):
        """
        Initialize the visualization manager with a test runner.
        
        Args:
            test_runner: The TestRunner instance containing test results
        """
        self.test_runner = test_runner
```

## Key Responsibilities

The `VisualizationManager` has the following key responsibilities:

1. **Test Result Access**: Provides methods to access test results for different test types
2. **Visualization Generation**: Offers specialized methods for creating different types of visualizations
3. **Lazy Execution**: Runs tests on-demand if results are not already available
4. **Consistent Interface**: Maintains a consistent API for accessing visualizations across test types

## Integration with Test Managers

The `VisualizationManager` integrates with the specialized manager classes through the `TestRunner`:

1. It retrieves test results from the `TestRunner`
2. If results don't exist yet, it initializes the appropriate manager and runs tests
3. It stores the new results back in the `TestRunner` for future access
4. It extracts visualization data from the standard results structure

```python
def get_robustness_results(self):
    """
    Get the robustness test results.
    
    Returns:
        dict: Dictionary containing robustness test results for main model and alternatives.
              Returns None if robustness tests haven't been run.
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

## Visualization Categories

The `VisualizationManager` provides visualization methods organized by test category:

### Robustness Visualizations

Methods for working with robustness test visualizations:

- `get_robustness_results()`: Retrieves robustness test results
- `get_robustness_visualizations()`: Gets all robustness visualizations
- `plot_robustness_comparison()`: Gets comparison plot of model robustness
- `plot_robustness_distribution()`: Gets boxplot of robustness score distributions
- `plot_feature_importance_robustness()`: Gets feature importance for robustness
- `plot_perturbation_methods_comparison()`: Gets comparison of perturbation methods

### Uncertainty Visualizations

Methods for working with uncertainty test visualizations:

- `get_uncertainty_results()`: Retrieves uncertainty test results
- `get_uncertainty_visualizations()`: Gets all uncertainty visualizations
- `plot_uncertainty_alpha_comparison()`: Gets comparison of alpha levels
- `plot_uncertainty_width_distribution()`: Gets distribution of interval widths
- `plot_feature_importance_uncertainty()`: Gets feature importance for uncertainty
- `plot_coverage_vs_width()`: Gets coverage vs width trade-off plot

### Resilience Visualizations

Methods for working with resilience test results:

- `get_resilience_results()`: Retrieves resilience test results

### Hyperparameter Visualizations

Methods for working with hyperparameter test results:

- `get_hyperparameter_results()`: Retrieves hyperparameter test results
- `get_hyperparameter_importance()`: Gets hyperparameter importance scores
- `get_hyperparameter_tuning_order()`: Gets recommended hyperparameter tuning order

## Integration with Visualizers

While the `VisualizationManager` doesn't directly implement visualization logic, it works with the specialized visualizer classes:

1. Managers (like `RobustnessManager`) generate visualizations using their `generate_visualizations()` methods
2. These visualizations follow the interface defined by their respective visualizer classes (like `BaseRobustnessVisualizer`)
3. The `VisualizationManager` provides a consistent API for retrieving these visualizations

This architecture separates visualization retrieval from visualization generation:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Experiment    │────>│VisualizationMgr │────>│   TestRunner    │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Visualizer    │<────│     Manager     │<────│  Test Results   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

## Usage Example

```python
from deepbridge.core.experiment import Experiment
from deepbridge.core.db_data import DBDataset

# Create dataset and experiment
dataset = DBDataset(data=my_data, target_column='target')
experiment = Experiment(dataset=dataset, tests=['robustness', 'uncertainty'])

# Run tests
experiment.run_tests()

# Use visualization manager to get plots
robustness_plot = experiment.visualization_manager.plot_robustness_comparison()
uncertainty_plot = experiment.visualization_manager.plot_uncertainty_alpha_comparison()

# Show plots
robustness_plot.show()
uncertainty_plot.show()
```

## Implementation Notes

- The `VisualizationManager` follows the Facade pattern, simplifying access to test results and visualizations
- It implements lazy loading of test results to minimize unnecessary computation
- The class provides a standardized interface for all visualization types
- Each visualization method returns a Plotly figure that can be displayed or saved
- Methods gracefully handle missing visualizations, returning None instead of raising exceptions

## Extension

To extend the `VisualizationManager` with a new test type:

1. Create a new specialized manager class for the test type
2. Implement visualization generation in that manager
3. Add corresponding getter methods to the `VisualizationManager`:
   - Method to retrieve test results
   - Method to retrieve all visualizations
   - Specific methods for individual visualization types

```python
# Example: Adding support for a new test type
def get_new_test_results(self):
    """
    Get the new test results.
    
    Returns:
        dict: Dictionary containing new test results.
    """
    results = self.test_runner.get_test_results('new_test')
    if results is None and "new_test" in self.test_runner.tests:
        new_test_manager = NewTestManager(
            self.test_runner.dataset, 
            self.test_runner.alternative_models, 
            self.test_runner.verbose
        )
        results = new_test_manager.run_tests()
        self.test_runner.test_results['new_test'] = results
    
    return results

def plot_new_test_visualization(self):
    """
    Get the new test visualization.
    
    Returns:
        plotly.graph_objects.Figure: New test visualization.
    """
    results = self.get_new_test_results()
    if results and 'visualizations' in results:
        return results['visualizations'].get('new_visualization')
    return None
```

The `VisualizationManager` centralizes the visualization capabilities of the DeepBridge framework, providing a consistent interface for retrieving and displaying visualizations from various test types. Its design supports extensibility while maintaining a clean, easy-to-use API for visualization access.