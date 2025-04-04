# Specialized Managers Documentation

## Overview

The specialized managers in the DeepBridge framework implement the `BaseManager` interface to provide specific testing capabilities for different aspects of model evaluation. Each manager focuses on a particular property of machine learning models, conducting tests and generating visualizations to help understand model behavior.

## BaseManager Interface

All specialized managers implement the common `BaseManager` abstract base class, which defines:

- Standard initialization with dataset, alternative models, and verbosity
- Abstract `run_tests()` method for executing tests on the primary model
- Abstract `compare_models()` method for comparing results across models
- Utility methods for logging and result retrieval

```python
class BaseManager(ABC):
    """
    Abstract base class for all manager components.
    Defines the common interface that all managers should implement.
    """
    
    def __init__(self, dataset, alternative_models=None, verbose=False):
        """
        Initialize the base manager.
        
        Args:
            dataset: DBDataset instance containing the primary model
            alternative_models: Dictionary of alternative models for comparison
            verbose: Whether to print progress information
        """
        self.dataset = dataset
        self.alternative_models = alternative_models or {}
        self.verbose = verbose
    
    @abstractmethod
    def run_tests(self, config_name: str = 'quick', **kwargs) -> dict:
        """
        Run standard tests on the primary model.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            **kwargs: Additional test parameters
            
        Returns:
            dict: Results of the tests
        """
        pass
    
    @abstractmethod
    def compare_models(self, config_name: str = 'quick', **kwargs) -> dict:
        """
        Compare test results across all models.
        
        Args:
            config_name: Configuration profile ('quick', 'medium', 'full')
            **kwargs: Additional test parameters
            
        Returns:
            dict: Comparison results for all models
        """
        pass
```

## Manager-Visualizer Integration

Managers integrate with the visualization system through a consistent pattern:

1. Each manager has methods to generate visualizations based on test results
2. Managers produce structured results that can be consumed by visualization components
3. Specialized visualization classes exist for each manager type (e.g., `BaseRobustnessVisualizer`)
4. The `VisualizationManager` coordinates between managers and visualizers

### Integration Flow

```
┌────────────┐     ┌─────────────────┐     ┌─────────────┐
│ BaseManager├────>│TestRunner       │────>│Test Results │
└────────────┘     └─────────────────┘     └──────┬──────┘
      │                                           │
      │                                           ▼
      │                                    ┌─────────────┐
      │                                    │Visualizations│
      │                                    └──────┬──────┘
      ▼                                           │
┌────────────┐     ┌─────────────────┐     ┌─────▼──────┐
│Visualizer  │<────┤VisualizationMgr │<────┤Report      │
└────────────┘     └─────────────────┘     └────────────┘
```

## Specialized Manager Components

### RobustnessManager

Evaluates model stability in the face of data perturbations and adversarial inputs.

#### Key Responsibilities:
- Testing model performance under different types of data perturbations
- Evaluating sensitivity to feature-level changes
- Comparing robustness across different models
- Generating visualizations of robustness metrics

#### Main Methods:
- `run_tests()`: Executes robustness tests with configurable intensity
- `compare_models_robustness()`: Compares robustness across different models
- `generate_visualizations()`: Creates plots showing robustness characteristics

#### Test Configurations:
- Quick: Basic perturbation tests with limited variants
- Medium: Moderate perturbation tests with multiple perturbation types
- Full: Comprehensive robustness evaluation with extensive perturbations

#### Visualization Integration:
The RobustnessManager generates visualizations that are consumed by `BaseRobustnessVisualizer` implementations:

```python
def generate_visualizations(self, robustness_results):
    """
    Generate Plotly visualizations for robustness tests.
    """
    visualizations = {}
    
    # Models Comparison - Performance vs Perturbation
    # Feature Importance Visualization  
    # Additional visualizations (distribution, boxplots, etc.)
            
    return visualizations
```

### UncertaintyManager

Assesses a model's ability to quantify prediction uncertainty and provide calibrated probability estimates.

#### Key Responsibilities:
- Evaluating prediction interval coverage
- Testing calibration of probability estimates
- Measuring uncertainty across different data segments
- Generating visualizations of uncertainty metrics

#### Main Methods:
- `run_tests()`: Executes uncertainty quantification tests
- `compare_models_uncertainty()`: Compares uncertainty estimation across models
- `generate_calibration_plots()`: Creates visualizations of model calibration

#### Test Configurations:
- Quick: Basic calibration and interval tests
- Medium: More comprehensive uncertainty evaluation with multiple metrics
- Full: Extensive uncertainty testing across many confidence levels

#### Visualization Integration:
The UncertaintyManager integrates with `BaseUncertaintyVisualizer` implementations to create calibration plots, reliability diagrams, and other uncertainty visualizations.

### ResilienceManager

Tests model performance under adverse conditions such as missing data, noisy inputs, or computational constraints.

#### Key Responsibilities:
- Evaluating model behavior with incomplete or corrupted data
- Testing resilience to missing feature values
- Measuring performance degradation under stressful conditions
- Comparing resilience across different models

#### Main Methods:
- `run_tests()`: Executes resilience tests with configurable intensity
- `compare_models_resilience()`: Compares resilience metrics across models
- `generate_degradation_plots()`: Visualizes performance degradation patterns

#### Test Configurations:
- Quick: Basic resilience tests with few scenarios
- Medium: Moderate resilience testing with multiple degradation types
- Full: Comprehensive resilience evaluation with extensive test cases

### HyperparameterManager

Analyzes the sensitivity of model performance to changes in hyperparameters, identifying the most critical parameters.

#### Key Responsibilities:
- Measuring the impact of hyperparameter changes on model performance
- Identifying the most important hyperparameters for tuning
- Suggesting optimal hyperparameter tuning order
- Comparing hyperparameter sensitivity across models

#### Main Methods:
- `run_tests()`: Executes hyperparameter importance analysis
- `compare_models_hyperparameters()`: Compares hyperparameter importance across models
- `generate_importance_plots()`: Creates visualizations of hyperparameter importance

#### Test Configurations:
- Quick: Basic hyperparameter sensitivity with limited parameter space
- Medium: Moderate analysis with more parameters and variations
- Full: Comprehensive hyperparameter analysis with extensive sampling

## Integration Architecture

The specialized managers are integrated with the `Experiment` class through the `TestRunner`, which:

1. Creates the appropriate manager instance based on test type
2. Provides the necessary dataset and model information
3. Configures the test intensity based on user requirements
4. Collects and organizes test results for reporting

## Usage Example

```python
from deepbridge.core.experiment.managers import RobustnessManager

# Create the manager with your dataset and alternative models
robustness_manager = RobustnessManager(
    dataset=my_dataset,
    alternative_models={'random_forest': rf_model, 'xgboost': xgb_model},
    verbose=True
)

# Run quick tests on the primary model
results = robustness_manager.run_tests(config_name='quick')

# Compare robustness across all models
comparison = robustness_manager.compare_models(config_name='medium')

# Get visualizations
visualizations = results.get('visualizations', {})
```

## Implementation Notes

- Each manager can operate independently or as part of the integrated experiment workflow
- Managers store their results and can be queried for specific result types
- All managers support three standard configuration levels (quick, medium, full)
- Each manager implements domain-specific visualization techniques
- Results are organized in a consistent structure for easy integration with reporting

## Extension Points

The manager architecture is designed for extensibility:

1. New manager types can be created by implementing the `BaseManager` interface
2. Existing managers can be extended with additional test methods
3. Custom visualization types can be added to each manager
4. New test configurations can be defined for different use cases

The specialized manager components provide a flexible and extensible architecture for comprehensive model evaluation, allowing users to assess different aspects of model quality through a consistent interface.