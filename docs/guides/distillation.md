# Knowledge Distillation Guide

## Overview

Knowledge Distillation is a powerful technique for creating smaller, more efficient models that preserve the performance of larger, more complex models. DeepBridge provides a comprehensive framework for knowledge distillation with its component-based architecture.

## Distillation Architecture

DeepBridge implements distillation through a modular component system:

```
┌───────────────────┐
│  Experiment       │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  ModelManager     │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐      ┌───────────────────┐
│  Distillation     │──────│  ModelEvaluation  │
│  Techniques       │      └───────────────────┘
└───────────────────┘
```

## Distillation Methods

DeepBridge supports multiple distillation approaches:

### 1. Surrogate Model Distillation

Directly learns to mimic the output probabilities of a teacher model:

```python
from deepbridge.distillation.techniques.surrogate import SurrogateModel
from deepbridge.utils.model_registry import ModelType
import numpy as np

# Create and configure surrogate model
surrogate = SurrogateModel(
    student_model_type=ModelType.LOGISTIC_REGRESSION,
    student_params={'C': 1.0},
    n_trials=20  # Number of hyperparameter optimization trials
)

# Train using teacher model's probability outputs
teacher_probas = teacher_model.predict_proba(X_train)
surrogate.fit(X=X_train, probas=teacher_probas)

# Make predictions with the surrogate model
predictions = surrogate.predict(X_test)
probabilities = surrogate.predict_proba(X_test)
```

### 2. Knowledge Distillation

Uses temperature scaling to transfer knowledge from teacher to student:

```python
from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
from deepbridge.utils.model_registry import ModelType

# Create and configure knowledge distillation
distiller = KnowledgeDistillation(
    teacher_model=complex_teacher_model,  # Original complex model
    student_model_type=ModelType.GBM,     # Lighter student model type
    temperature=2.0,                      # Softening temperature
    alpha=0.5,                            # Balance between soft and hard targets
    n_trials=30                           # Hyperparameter optimization trials
)

# Train using both teacher model and actual labels
distiller.fit(X_train, y_train)

# Make predictions
predictions = distiller.predict(X_test)
probabilities = distiller.predict_proba(X_test)
```

## Using Distillation in Experiments

The `Experiment` class provides a high-level interface for distillation:

```python
from deepbridge.core.experiment import Experiment
from deepbridge.core.db_data import DBDataset
from deepbridge.utils.model_registry import ModelType

# Create dataset
dataset = DBDataset(
    data=X_train, 
    target=y_train,
    test_data=X_test,
    test_target=y_test,
    model=teacher_model  # Provide the complex teacher model
)

# Initialize experiment
experiment = Experiment(
    dataset=dataset,
    experiment_type='binary_classification',
    tests=['robustness']  # Optionally test distilled model
)

# Perform model distillation
experiment.fit(
    student_model_type=ModelType.GBM,  # Choose student model type
    distillation_method='knowledge_distillation',  # Or 'surrogate'
    temperature=2.0,  # Temperature for softening probabilities
    alpha=0.5,        # Balance between soft and hard targets
    n_trials=20       # Hyperparameter optimization trials
)

# Evaluate the distilled model
metrics = experiment.model_evaluation.evaluate_model(
    experiment.distillation_model, 
    "Distilled Model",
    "distilled", 
    X_test, 
    y_test
)

print(f"Distilled model accuracy: {metrics.get('accuracy', 'N/A')}")
```

## Automated Distillation

For even simpler model distillation, use the `AutoDistiller`:

```python
from deepbridge.auto_distiller import AutoDistiller
from deepbridge.core.db_data import DBDataset
from deepbridge.utils.model_registry import ModelType

# Create dataset with probabilities
dataset = DBDataset(
    data=df,
    target_column='target',
    features=feature_columns,
    prob_cols=['prob_class_0', 'prob_class_1']  # Pre-calculated probabilities
)

# Configure and run automated distillation
distiller = AutoDistiller(
    dataset=dataset,
    output_dir='results',
    test_size=0.2,
    n_trials=10
)

# Customize model configurations to test
distiller.customize_config(
    model_types=[
        ModelType.LOGISTIC_REGRESSION, 
        ModelType.GBM, 
        ModelType.RANDOM_FOREST
    ],
    temperatures=[1.0, 2.0, 5.0],
    alphas=[0.3, 0.5, 0.7]
)

# Run the distillation process
results = distiller.run(use_probabilities=True)

# Get the best model based on test accuracy
best_model = distiller.get_best_model(metric='test_accuracy')
```

## Working with ModelManager Directly

For more advanced control, you can use the `ModelManager` component directly:

```python
from deepbridge.core.experiment.managers.model_manager import ModelManager
from deepbridge.utils.model_registry import ModelType

# Create model manager
model_manager = ModelManager(
    dataset=dataset,
    experiment_type='binary_classification',
    verbose=True
)

# Create distillation model
distillation_model = model_manager.create_distillation_model(
    distillation_method='knowledge_distillation',
    student_model_type=ModelType.GBM,
    student_params={'max_depth': 4},
    temperature=2.0,
    alpha=0.5,
    use_probabilities=True,
    n_trials=20,
    validation_split=0.2
)

# Train the model
distillation_model.fit(X_train, y_train)
```

## Key Parameters for Distillation

### Temperature

The temperature parameter controls the "softness" of probability distributions:

```python
# High temperature (softer probabilities)
distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_model_type=ModelType.GBM,
    temperature=5.0,  # Higher temperature = softer probabilities
    alpha=0.5
)

# Low temperature (sharper probabilities)
distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_model_type=ModelType.GBM,
    temperature=1.0,  # Lower temperature = sharper probabilities
    alpha=0.5
)
```

### Alpha

The alpha parameter balances between mimicking the teacher (soft targets) and learning from actual labels (hard targets):

```python
# Focus more on teacher probabilities
distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_model_type=ModelType.GBM,
    temperature=2.0,
    alpha=0.8  # Higher alpha = more focus on soft targets
)

# Balance between teacher and true labels
distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_model_type=ModelType.GBM,
    temperature=2.0,
    alpha=0.5  # Equal balance
)

# Focus more on true labels
distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_model_type=ModelType.GBM,
    temperature=2.0,
    alpha=0.2  # Lower alpha = more focus on hard targets
)
```

## Evaluating Distilled Models

DeepBridge provides comprehensive tools for evaluating distilled models:

```python
from deepbridge.core.experiment import Experiment
from deepbridge.core.db_data import DBDataset

# Create dataset with both models
dataset = DBDataset(
    data=X_train, 
    target=y_train,
    test_data=X_test,
    test_target=y_test,
    model=teacher_model
)

# Initialize experiment
experiment = Experiment(
    dataset=dataset,
    experiment_type='binary_classification'
)

# Perform distillation
experiment.fit(
    student_model_type='gbm',
    distillation_method='knowledge_distillation'
)

# Compare all models
comparison = experiment.compare_all_models(dataset='test')
print(comparison)

# Run robustness tests to compare stability
experiment.tests = ['robustness']
results = experiment.run_tests()

# Visualize comparison
robustness_plot = experiment.plot_robustness_comparison()
```

## Distribution Matching Analysis

You can analyze how well the student model matches the teacher's probability distributions:

```python
from deepbridge.core.experiment.model_evaluation import ModelEvaluation
from sklearn.metrics import r2_score
import numpy as np

# Create model evaluation component
evaluator = ModelEvaluation(
    experiment_type='binary_classification',
    metrics_calculator=None  # Or provide metrics calculator
)

# Get model predictions
teacher_probs = teacher_model.predict_proba(X_test)
student_probs = student_model.predict_proba(X_test)

# Calculate distribution metrics
results = evaluator._calculate_distribution_metrics(
    teacher_probs[:, 1],  # Probability of positive class from teacher
    student_probs[:, 1]   # Probability of positive class from student
)

ks_stat, ks_pvalue, r2 = results
print(f"KS Statistic: {ks_stat:.4f}")
print(f"KS p-value: {ks_pvalue:.4f}")
print(f"R² Score: {r2:.4f}")
```

## Best Practices

### 1. Model Selection

Choose student models based on your deployment constraints:

```python
# For maximum accuracy and acceptable size
distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_model_type=ModelType.GBM,
    temperature=2.0
)

# For smallest model size and fast inference
distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_model_type=ModelType.LOGISTIC_REGRESSION,
    temperature=2.0
)

# For balanced size-performance trade-off
distiller = KnowledgeDistillation(
    teacher_model=teacher_model,
    student_model_type=ModelType.DECISION_TREE,
    temperature=2.0
)
```

### 2. Parameter Experimentation

Test multiple configurations to find the optimal settings:

```python
# Create dataset
dataset = DBDataset(data=X_train, target=y_train, test_data=X_test, test_target=y_test)

# Initialize experiment
experiment = Experiment(dataset=dataset, experiment_type='binary_classification')

# Compare different temperatures
results = {}
for temp in [1.0, 2.0, 5.0, 10.0]:
    experiment.fit(
        student_model_type='gbm',
        distillation_method='knowledge_distillation',
        temperature=temp,
        alpha=0.5
    )
    
    # Evaluate on test set
    accuracy = experiment.model_evaluation.evaluate_model(
        experiment.distillation_model, 
        f"Temp_{temp}", 
        "distilled", 
        X_test, 
        y_test
    ).get('accuracy')
    
    results[f"Temperature {temp}"] = accuracy

print("Temperature comparison:", results)
```

### 3. Validation After Distillation

Always validate distilled models thoroughly:

```python
# Create experiment with distilled model
experiment = Experiment(
    dataset=dataset,
    experiment_type='binary_classification',
    tests=['robustness', 'uncertainty', 'resilience']
)

# Fit distilled model
experiment.fit(
    student_model_type='gbm',
    distillation_method='knowledge_distillation'
)

# Run comprehensive validation
experiment.run_tests(config_name='full')

# Generate report
experiment.save_report('distilled_model_validation.html')
```

## Conclusion

DeepBridge's component-based architecture provides flexible and powerful tools for knowledge distillation. By choosing the right distillation method and parameters, you can create efficient models that maintain most of the performance of complex teacher models while being much lighter and faster.

## Additional Resources

- [Knowledge Distillation Theory](../concepts/knowledge_distillation.md)
- [How Student Models Learn](../concepts/model_learns.md)
- [AutoDistiller Guide](../tutorials/AutoDistiller.md)
- [ModelManager Documentation](../api/model_manager_documentation.md)