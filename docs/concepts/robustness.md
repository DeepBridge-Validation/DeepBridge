# Model Robustness Testing

## Overview

Model robustness refers to the ability of a machine learning model to maintain its performance when input data is perturbed or slightly modified. DeepBridge provides comprehensive tools to evaluate and visualize model robustness, helping you build more reliable and trustworthy machine learning models.

## Key Concepts

### Perturbation Methods

- **Raw Perturbation** - Adds Gaussian noise proportional to feature variance
- **Quantile Perturbation** - Transforms data into quantile space, adds noise, and transforms back
- **Categorical Perturbation** - Randomly replaces categorical values based on their frequency distribution

### Robustness Metrics

- **Performance Under Perturbation** - How model performance changes as perturbation size increases
- **Robustness Index** - Area under the curve of performance vs perturbation magnitude
- **Feature Importance** - Impact of perturbing individual features on model performance

## Getting Started

### Basic Robustness Testing

```python
from deepbridge.validation import RobustnessTest
from deepbridge.visualization import RobustnessViz

# Initialize robustness test
robustness_test = RobustnessTest()

# Evaluate model robustness
results = robustness_test.evaluate_robustness(
    models={'Model A': model_a, 'Model B': model_b},
    X=X_test,
    y=y_test,
    perturb_method='raw',
    perturb_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    metric='AUC',
    n_iterations=10
)

# Visualize results
fig = RobustnessViz.plot_models_comparison(
    results=results,
    metric_name='AUC Score'
)
```

### Feature Importance Analysis

```python
# Analyze which features impact robustness the most
feature_importance = robustness_test.analyze_feature_importance(
    model=model,
    X=X_test,
    y=y_test,
    perturb_method='raw',
    perturb_size=0.5,
    metric='AUC'
)

# Visualize feature importance
fig = RobustnessViz.plot_feature_importance(
    feature_importance_results=feature_importance
)
```

### Comparing Perturbation Methods

```python
# Compare different perturbation methods
methods_results = robustness_test.compare_perturbation_methods(
    model=model,
    X=X_test,
    y=y_test,
    perturb_methods=['raw', 'quantile'],
    perturb_sizes=[0.1, 0.3, 0.5, 0.7, 1.0],
    metric='AUC'
)

# Visualize comparison
fig = RobustnessViz.plot_perturbation_methods_comparison(
    methods_comparison_results=methods_results
)
```

## Robustness Score

To calculate an overall robustness score for model comparison:

```python
from deepbridge.validation import RobustnessScore

# Calculate robustness indices
robustness_indices = RobustnessScore.calculate_robustness_index(
    results=results,
    metric='AUC'
)

# Visualize robustness indices
fig = RobustnessViz.plot_robustness_index(
    results=results,
    robustness_indices=robustness_indices
)
```

## Best Practices

1. **Start with Raw Perturbation** - It's the most straightforward method and provides a good baseline.

2. **Test Multiple Perturbation Sizes** - A range of 0.1 to 1.0 typically covers mild to severe perturbations.

3. **Use Domain-Appropriate Metrics** - For classification, use AUC or F1; for regression, use MSE or MAE.

4. **Repeat with Multiple Iterations** - Due to the randomness in perturbations, use at least 10 iterations for stable results.

5. **Compare Multiple Models** - Robustness testing is most valuable when comparing different model architectures.

## Advanced Configuration

### Custom Perturbation Methods

You can create custom perturbation methods by extending the `BasePerturbation` class:

```python
from deepbridge.validation.perturbation import BasePerturbation

class MyCustomPerturbation(BasePerturbation):
    def perturb(self, X, perturb_size, **kwargs):
        # Implement your custom perturbation logic
        return perturbed_X
```

### Working with Categorical Features

For datasets with categorical features, specify them explicitly:

```python
results = robustness_test.evaluate_robustness(
    models=models,
    X=X_test,
    y=y_test,
    perturb_method='raw',
    cat_features=['feature_a', 'feature_b'],
    perturb_sizes=[0.1, 0.3, 0.5]
)
```

## Conclusion

Robustness testing is an essential step in model validation, helping you build models that perform reliably in real-world conditions. By systematically evaluating how your models respond to perturbations in the input data, you can identify weaknesses, compare model architectures, and select the most robust models for deployment.