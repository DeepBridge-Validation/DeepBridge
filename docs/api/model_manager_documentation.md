# ModelManager Documentation

## Overview

The `ModelManager` class in the DeepBridge framework is responsible for creating, configuring, and managing models within experiments. It handles the creation of distillation models, generation of alternative models for comparison, and interaction with the framework's model registry. This component centralizes model-related operations and provides a consistent interface for model creation across the experiment workflow.

## Class Definition

```python
class ModelManager:
    """
    Manages creation and handling of different models in the experiment.
    """
    
    def __init__(self, dataset, experiment_type, verbose=False):
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.verbose = verbose
```

## Key Responsibilities

The `ModelManager` has the following primary responsibilities:

1. **Model Creation**: Creates and configures models for experiments
2. **Distillation Support**: Provides interfaces for surrogate and knowledge distillation model creation
3. **Alternative Model Generation**: Creates complementary models for comparison purposes
4. **Model Registry Integration**: Interfaces with the framework's model registry to retrieve model implementations

## Creating Alternative Models

One of the key responsibilities of the `ModelManager` is to generate alternative models for comparison with the primary model:

```python
def create_alternative_models(self, X_train, y_train):
    """
    Create 3 alternative models different from the original model,
    using ModelRegistry directly without SurrogateModel.
    """
```

This method:
1. Identifies the original model type from the dataset
2. Creates up to 3 different model types for comparison
3. Trains each model on the provided training data
4. Returns a dictionary of trained models, keyed by model type

The alternative models provide important benchmarks for evaluating the primary model's performance and for assessing the quality of distilled models.

## Distillation Model Creation

The `ModelManager` supports two main approaches to distillation:

### 1. Creating Models from Pre-calculated Probabilities

```python
def _create_model_from_probabilities(self,
                               distillation_method: str,
                               student_model_type: ModelType,
                               student_params: t.Optional[dict],
                               temperature: float,
                               alpha: float,
                               n_trials: int,
                               validation_split: float) -> object:
    """Create distillation model from pre-calculated probabilities"""
```

This method creates a distilled model using pre-calculated probability outputs from a teacher model. This approach is useful when:
- The original teacher model is not available
- You want to distill from an ensemble or averaged prediction
- The teacher model's probabilities have been collected over time

### 2. Creating Models from a Teacher Model

```python
def _create_model_from_teacher(self,
                        distillation_method: str,
                        student_model_type: ModelType,
                        student_params: t.Optional[dict],
                        temperature: float,
                        alpha: float,
                        n_trials: int,
                        validation_split: float) -> object:
    """Create distillation model from teacher model"""
```

This method creates a distilled model using a teacher model directly. This approach is useful when:
- The teacher model is available for making predictions
- You want to explore different temperature settings dynamically
- You need to use the full model for additional functionality beyond prediction

## Distillation Method Selection

The `create_distillation_model` method serves as the main entry point for creating distilled models:

```python
def create_distillation_model(self, 
                        distillation_method: str,
                        student_model_type: ModelType,
                        student_params: t.Optional[dict],
                        temperature: float,
                        alpha: float,
                        use_probabilities: bool,
                        n_trials: int,
                        validation_split: float) -> object:
    """Create appropriate distillation model based on method and available data"""
```

This method:
1. Determines whether to use pre-calculated probabilities or a teacher model
2. Selects the appropriate distillation technique based on the provided method
3. Creates and configures the distillation model with the specified parameters
4. Returns the configured (but not yet trained) distillation model

The supported distillation methods are:
- **Surrogate**: Directly learns the probability outputs of the teacher model
- **Knowledge Distillation**: Uses soft targets with temperature to transfer knowledge

## Integration with Model Registry

The `ModelManager` integrates with the framework's `ModelRegistry` to access model implementations:

```python
# Example usage within create_alternative_models
model = ModelRegistry.get_model(
    model_type=model_type,
    custom_params=None,  # Use default parameters
    mode=mode  # Use classification or regression mode
)
```

This integration ensures:
1. Consistent model creation across the framework
2. Access to the latest model implementations
3. Standardized parameter handling
4. Proper model configuration based on the task type

## Default Model Selection

The `ModelManager` provides a method to select a reasonable default model:

```python
def get_default_model_type(self):
    """Get a default model type for XGBoost or similar"""
    # Try to find XGBoost or fallback to first model type
    for model_type in ModelType:
        if 'XGB' in model_type.name:
            return model_type
    # Fallback to first model type
    return next(iter(ModelType))
```

This method:
1. Attempts to find an XGBoost model type (preferred default)
2. Falls back to the first available model type if XGBoost is not available
3. Returns a `ModelType` enum value that can be used with the registry

## Usage Example

The `ModelManager` is typically used within the `Experiment` class:

```python
from deepbridge.core.experiment import Experiment
from deepbridge.core.db_data import DBDataset

# Create dataset
dataset = DBDataset(
    data=my_dataframe,
    target_column='target'
)

# Initialize experiment
experiment = Experiment(
    dataset=dataset,
    experiment_type='binary_classification'
)

# The experiment internally uses ModelManager
# Create a distilled model
experiment.fit(
    student_model_type='random_forest',
    distillation_method='knowledge_distillation',
    temperature=2.0,
    use_probabilities=True
)
```

It can also be used independently:

```python
from deepbridge.core.experiment.managers.model_manager import ModelManager
from deepbridge.utils.model_registry import ModelType

# Create manager
model_manager = ModelManager(
    dataset=my_dataset,
    experiment_type='binary_classification',
    verbose=True
)

# Create alternative models
alternative_models = model_manager.create_alternative_models(X_train, y_train)

# Create a distillation model
distillation_model = model_manager.create_distillation_model(
    distillation_method='surrogate',
    student_model_type=ModelType.RANDOM_FOREST,
    student_params=None,
    temperature=1.0,
    alpha=0.5,
    use_probabilities=True,
    n_trials=20,
    validation_split=0.2
)

# Train the model
distillation_model.fit(X_train, y_train)
```

## Implementation Notes

- The `ModelManager` uses late imports to avoid circular dependencies
- It gracefully handles failures when creating models
- The manager supports both classification and regression tasks
- Alternative model creation is limited to 3 models to manage computational resources
- The manager provides verbose logging to help debug model creation issues

## Integration with Experiment

Within the `Experiment` class, the `ModelManager` is initialized during experiment creation:

```python
# From Experiment.__init__
self.model_manager = ModelManager(dataset, self.experiment_type, self.verbose)
```

The `Experiment` class delegates model creation to the `ModelManager`:

```python
# From Experiment.fit
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

This delegation pattern ensures:
1. Clear separation of responsibilities
2. Centralized model management
3. Consistent model creation across the framework
4. Easy extension with new model types or distillation techniques