# DeepBridge Testing Framework

## Overview

The DeepBridge testing framework provides comprehensive model validation through four specialized test suites: Robustness, Uncertainty, Resilience, and Hyperparameter testing. Each suite is designed to evaluate different aspects of model behavior and performance under various conditions.

## Test Types

### 1. Robustness Testing

Robustness testing evaluates how well a model maintains its performance when input data is perturbed. This helps identify model vulnerabilities and feature sensitivities.

#### Test Methods

**Raw Perturbation**
- Adds Gaussian noise directly to feature values
- Noise level scales from 0.1 to 1.0 (10% to 100% of feature standard deviation)
- Tests overall model stability

**Quantile Perturbation**
- Applies noise based on feature quantiles
- More sophisticated than raw perturbation
- Better handles features with different scales

**Adversarial Perturbation**
- Creates targeted perturbations to maximize prediction error
- Identifies worst-case scenarios
- Useful for security-critical applications

#### Configuration Levels

```python
# Quick configuration (3 trials, 3 levels)
robustness_config = {
    'n_trials': 3,
    'perturbation_levels': [0.1, 0.5, 1.0],
    'methods': ['raw', 'quantile']
}

# Medium configuration (5 trials, 5 levels)
robustness_config = {
    'n_trials': 5,
    'perturbation_levels': [0.1, 0.3, 0.5, 0.7, 1.0],
    'methods': ['raw', 'quantile', 'adversarial']
}

# Full configuration (10 trials, 10 levels)
robustness_config = {
    'n_trials': 10,
    'perturbation_levels': np.linspace(0.1, 1.0, 10),
    'methods': ['raw', 'quantile', 'adversarial'],
    'feature_subset': None  # Test all features
}
```

#### Usage Example

```python
from deepbridge.validation.wrappers import RobustnessSuite

# Create and run robustness test
suite = RobustnessSuite(
    dataset=dataset,
    model=model,
    config='medium'  # or custom config dict
)

results = suite.run()

# Analyze results
print(f"Base Score: {results['base_score']}")
print(f"Average Impact: {results['avg_impact']}")
print(f"Most Sensitive Features: {results['feature_importance']}")
```

#### Result Structure

```python
{
    'base_score': 0.95,  # Original model performance
    'raw': {
        'by_level': {
            0.1: {'mean_score': 0.93, 'std': 0.02},
            0.5: {'mean_score': 0.87, 'std': 0.04},
            1.0: {'mean_score': 0.75, 'std': 0.08}
        },
        'overall': {
            'avg_degradation': 0.15,
            'worst_case': 0.65
        }
    },
    'quantile': {...},  # Similar structure
    'adversarial': {...},
    'feature_importance': {
        'feature1': 0.35,  # Impact score
        'feature2': 0.28,
        'feature3': 0.15
    },
    'avg_impact': 0.18,
    'model_type': 'RandomForestClassifier'
}
```

### 2. Uncertainty Testing

Uncertainty testing quantifies the reliability of model predictions using conformal prediction techniques. It provides prediction intervals and evaluates calibration quality.

#### Methods

**CRQR (Conformalized Residual Quantile Regression)**
- State-of-the-art uncertainty quantification
- Provides prediction intervals with guaranteed coverage
- Works for both classification and regression

#### Key Parameters

```python
uncertainty_config = {
    'alpha_levels': [0.01, 0.05, 0.1, 0.2],  # Confidence levels
    'test_size': 0.3,  # Holdout for calibration
    'calibration_ratio': 0.33,  # Portion for calibration
    'n_bootstraps': 100  # For stability
}
```

#### Usage Example

```python
from deepbridge.validation.wrappers import UncertaintySuite

suite = UncertaintySuite(
    dataset=dataset,
    model=model,
    config='medium'
)

results = suite.run()

# Check calibration quality
print(f"Coverage Error: {results['coverage_error']}")
print(f"Average Interval Width: {results['avg_interval_width']}")
print(f"Uncertainty Quality Score: {results['uncertainty_quality_score']}")
```

#### Result Structure

```python
{
    'crqr': {
        'by_alpha': {
            0.05: {
                'coverage': 0.948,  # Actual coverage
                'expected': 0.95,   # Expected coverage
                'width': 0.15,      # Average interval width
                'error': 0.002      # Coverage error
            },
            0.1: {...},
            0.2: {...}
        },
        'by_feature': {
            'feature1': {'importance': 0.25},
            'feature2': {'importance': 0.18}
        },
        'all_results': [...]  # Detailed results
    },
    'coverage_error': 0.012,  # Overall calibration error
    'avg_interval_width': 0.18,
    'uncertainty_quality_score': 0.88  # 0-1 score
}
```

### 3. Resilience Testing

Resilience testing measures model performance under distribution shifts, simulating real-world scenarios where deployment data differs from training data.

#### Drift Types

**Covariate Shift**
- Changes in feature distributions
- Common in real deployments
- Tests feature robustness

**Label Shift**
- Changes in target distribution
- Tests model calibration
- Important for imbalanced problems

**Concept Drift**
- Changes in feature-target relationships
- Most challenging scenario
- Tests fundamental model assumptions

**Temporal Drift**
- Time-based distribution changes
- Simulates model aging
- Important for production monitoring

#### Distance Metrics

```python
distance_metrics = [
    'psi',         # Population Stability Index
    'ks',          # Kolmogorov-Smirnov
    'wasserstein', # Wasserstein distance
    'kl',          # Kullback-Leibler divergence
    'js'           # Jensen-Shannon divergence
]
```

#### Usage Example

```python
from deepbridge.validation.wrappers import ResilienceSuite

suite = ResilienceSuite(
    dataset=dataset,
    model=model,
    config='medium'
)

results = suite.run()

# Analyze resilience
print(f"Resilience Score: {results['resilience_score']}")
print(f"Worst Drift Impact: {results['worst_case_performance']}")
print(f"Feature Distances: {results['feature_distances']}")
```

#### Result Structure

```python
{
    'distribution_shift': {
        'by_alpha': {
            0.05: {
                'performance_gap': 0.02,
                'drift_detected': False,
                'confidence': 0.95
            },
            0.15: {
                'performance_gap': 0.08,
                'drift_detected': True,
                'confidence': 0.87
            }
        },
        'by_distance_metric': {
            'psi': {'avg_distance': 0.12},
            'ks': {'avg_distance': 0.15},
            'wasserstein': {'avg_distance': 0.18}
        }
    },
    'resilience_score': 0.82,  # 0-1 score
    'feature_distances': {
        'feature1': 0.22,
        'feature2': 0.15,
        'feature3': 0.08
    },
    'critical_features': ['feature1', 'feature2']
}
```

### 4. Hyperparameter Testing

Hyperparameter testing identifies which model parameters have the most impact on performance, helping prioritize tuning efforts.

#### Analysis Methods

**Subsampling Analysis**
- Tests parameter importance via data subsampling
- Robust to overfitting
- Provides confidence intervals

**Cross-Validation Analysis**
- Uses k-fold CV for stability
- Accounts for data variability
- More computationally intensive

#### Configuration

```python
hyperparameter_config = {
    'n_subsamples': 10,      # Number of subsamples
    'subsample_size': 0.7,   # Size of each subsample
    'cv_folds': 5,           # Cross-validation folds
    'scoring': 'accuracy',   # Metric to optimize
    'n_jobs': -1            # Parallel processing
}
```

#### Usage Example

```python
from deepbridge.validation.wrappers import HyperparameterSuite

suite = HyperparameterSuite(
    dataset=dataset,
    model=model,
    config='medium'
)

results = suite.run()

# Get tuning recommendations
print(f"Importance Scores: {results['importance_scores']}")
print(f"Tuning Order: {results['tuning_order']}")
print(f"Expected Improvement: {results['expected_improvement']}")
```

#### Result Structure

```python
{
    'importance': {
        'by_config': {
            'n_estimators': {
                'score': 0.45,
                'std': 0.05,
                'rank': 1
            },
            'max_depth': {
                'score': 0.32,
                'std': 0.04,
                'rank': 2
            },
            'min_samples_split': {
                'score': 0.15,
                'std': 0.03,
                'rank': 3
            }
        },
        'all_results': [...]  # Detailed results
    },
    'importance_scores': {
        'n_estimators': 0.45,
        'max_depth': 0.32,
        'min_samples_split': 0.15,
        'learning_rate': 0.08
    },
    'tuning_order': [
        'n_estimators',
        'max_depth',
        'min_samples_split',
        'learning_rate'
    ],
    'expected_improvement': {
        'n_estimators': 0.03,  # Expected performance gain
        'max_depth': 0.02
    }
}
```

## Advanced Features

### Feature Subset Testing

Test specific features to identify vulnerabilities:

```python
# Test only critical features
critical_features = ['age', 'income', 'credit_score']

results = suite.run(feature_subset=critical_features)
```

### Model Comparison

Compare multiple models simultaneously:

```python
from deepbridge.core.experiment import Experiment

experiment = Experiment(
    name='model_comparison',
    dataset=dataset,
    models={
        'baseline': LogisticRegression(),
        'rf': RandomForestClassifier(),
        'xgb': XGBClassifier()
    }
)

# Run all tests on all models
results = experiment.run_all_tests(config='medium')

# Generate comparative report
experiment.generate_report(
    test_type='robustness',
    output_dir='./reports',
    format='interactive'
)
```

### Custom Test Strategies

Create custom test strategies:

```python
from deepbridge.core.experiment.test_strategies import TestStrategy

class CustomStrategy(TestStrategy):
    def run(self, dataset, model, config):
        # Custom test implementation
        results = {}
        
        # Your test logic here
        
        return results

# Register and use
from deepbridge.core.experiment.test_runner import TestRunner

runner = TestRunner(strategy=CustomStrategy())
results = runner.run(dataset, model)
```

### Progressive Testing

Start with quick tests and progressively increase thoroughness:

```python
# Step 1: Quick scan
quick_results = experiment.run_test('robustness', config='quick')

if quick_results['avg_impact'] > 0.1:
    # Step 2: Detailed analysis
    medium_results = experiment.run_test('robustness', config='medium')
    
    if medium_results['avg_impact'] > 0.2:
        # Step 3: Full investigation
        full_results = experiment.run_test('robustness', config='full')
```

## Best Practices

### 1. Test Selection

- **Robustness**: Essential for production models
- **Uncertainty**: Critical for high-stakes decisions
- **Resilience**: Important for long-term deployments
- **Hyperparameter**: Useful during model development

### 2. Configuration Guidelines

- Start with 'quick' for initial assessment
- Use 'medium' for standard validation
- Reserve 'full' for final validation or critical models

### 3. Result Interpretation

**Robustness Scores**
- \> 0.9: Excellent robustness
- 0.7-0.9: Good robustness
- 0.5-0.7: Moderate vulnerability
- < 0.5: High vulnerability

**Uncertainty Quality**
- \> 0.9: Well-calibrated
- 0.8-0.9: Good calibration
- 0.7-0.8: Acceptable
- < 0.7: Needs recalibration

**Resilience Scores**
- \> 0.85: Highly resilient
- 0.7-0.85: Moderately resilient
- 0.5-0.7: Vulnerable to shifts
- < 0.5: High drift sensitivity

### 4. Performance Optimization

```python
# Enable parallel processing
import os
os.environ['DEEPBRIDGE_N_JOBS'] = '-1'

# Use caching for repeated tests
from deepbridge.utils.cache import enable_caching
enable_caching()

# Batch processing for large datasets
from deepbridge.utils.batch import BatchProcessor
processor = BatchProcessor(batch_size=10000)
results = processor.run_tests(dataset, model)
```

## Integration Examples

### With MLflow

```python
import mlflow
from deepbridge.core.experiment import Experiment

with mlflow.start_run():
    # Log parameters
    mlflow.log_params(model.get_params())
    
    # Run tests
    experiment = Experiment('mlflow_exp', dataset, {'model': model})
    results = experiment.run_all_tests()
    
    # Log metrics
    mlflow.log_metric('robustness_score', results['robustness']['avg_impact'])
    mlflow.log_metric('uncertainty_quality', results['uncertainty']['uncertainty_quality_score'])
    mlflow.log_metric('resilience_score', results['resilience']['resilience_score'])
    
    # Save report
    experiment.generate_report('all', './reports')
    mlflow.log_artifacts('./reports')
```

### With Weights & Biases

```python
import wandb
from deepbridge.validation.wrappers import RobustnessSuite

wandb.init(project='model-validation')

# Run tests
suite = RobustnessSuite(dataset, model)
results = suite.run()

# Log to W&B
wandb.log({
    'robustness/base_score': results['base_score'],
    'robustness/avg_impact': results['avg_impact'],
    'robustness/feature_importance': results['feature_importance']
})

# Log charts
import matplotlib.pyplot as plt
fig = suite.plot_results()
wandb.log({'robustness_chart': wandb.Image(fig)})
```

## Troubleshooting

### Common Issues

1. **Memory errors with large datasets**
   - Use batch processing
   - Reduce test configurations
   - Enable data sampling

2. **Slow test execution**
   - Enable parallel processing
   - Use 'quick' configuration first
   - Profile code to find bottlenecks

3. **Inconsistent results**
   - Set random seeds
   - Increase number of trials
   - Check for data leakage

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
results = suite.run(verbose=True, debug=True)

# Save intermediate results
results = suite.run(save_intermediate=True, output_dir='./debug')
```