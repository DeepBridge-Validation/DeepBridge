# DeepBridge API Reference

This document provides a comprehensive API reference for all DeepBridge modules and classes.

## Core Modules

### deepbridge.db_data

#### DBDataset

The fundamental data container for DeepBridge operations.

```python
class DBDataset:
    """
    Container for managing datasets with features, targets, and predictions.
    
    Parameters
    ----------
    data : pd.DataFrame
        The complete dataset
    target_column : str
        Name of the target column
    features : List[str], optional
        List of feature columns. If None, all columns except target are used
    prob_cols : List[str], optional
        List of probability columns for multi-class problems
    task : str, optional
        Task type: 'classification' or 'regression'. Auto-detected if None
    
    Attributes
    ----------
    data : pd.DataFrame
        Original dataset
    features : List[str]
        Feature column names
    target : pd.Series
        Target values
    task : str
        Task type
    n_classes : int
        Number of classes (for classification)
    
    Examples
    --------
    >>> dataset = DBDataset(df, target_column='label', features=['f1', 'f2'])
    >>> X_train, X_test, y_train, y_test = dataset.train_test_split()
    """
    
    def __init__(self, data, target_column, features=None, prob_cols=None, task=None)
    
    def train_test_split(self, test_size=0.2, random_state=None, stratify=None):
        """Split data into train and test sets."""
        
    def get_features(self):
        """Return feature matrix."""
        
    def get_target(self):
        """Return target values."""
        
    def add_predictions(self, predictions, model_name='model'):
        """Add model predictions to the dataset."""
        
    def subset(self, indices):
        """Create a subset of the dataset."""
```

### deepbridge.core.experiment

#### Experiment

Main class for managing ML experiments and validation workflows.

```python
class Experiment:
    """
    Orchestrates model validation experiments.
    
    Parameters
    ----------
    name : str
        Experiment name
    dataset : DBDataset
        Dataset for the experiment
    models : Dict[str, Any]
        Dictionary of models to evaluate
    output_dir : str, default='./experiments'
        Directory for saving results
    config : Dict[str, Any], optional
        Additional configuration options
    
    Methods
    -------
    run_test(test_type, config='medium', feature_subset=None)
        Run a specific validation test
    run_all_tests(config='medium')
        Run all available tests
    generate_report(test_type, output_dir=None, format='interactive')
        Generate HTML report for test results
    save_results()
        Save experiment results to disk
    load_results(path)
        Load saved experiment results
    
    Examples
    --------
    >>> exp = Experiment('my_exp', dataset, {'model1': clf1, 'model2': clf2})
    >>> results = exp.run_test('robustness', config='full')
    >>> exp.generate_report('robustness', './reports')
    """
    
    def __init__(self, name, dataset, models, output_dir='./experiments', config=None)
    
    def run_test(self, test_type, config='medium', feature_subset=None):
        """Execute validation test."""
        
    def compare_models(self, metric='accuracy'):
        """Compare model performances."""
```

### deepbridge.validation.wrappers

#### RobustnessSuite

```python
class RobustnessSuite:
    """
    Robustness testing for ML models.
    
    Parameters
    ----------
    dataset : DBDataset
        Dataset for testing
    model : Any
        Model to evaluate
    config : str or dict
        Configuration level ('quick', 'medium', 'full') or custom config
    
    Methods
    -------
    run()
        Execute robustness tests
    get_feature_importance()
        Extract feature importance from results
    
    Returns
    -------
    Dict containing:
        - base_score: Original model performance
        - raw: Raw perturbation results
        - quantile: Quantile perturbation results
        - adversarial: Adversarial perturbation results
        - feature_importance: Impact by feature
        - avg_impact: Average performance degradation
    """
    
    def __init__(self, dataset, model, config='medium')
```

#### UncertaintySuite

```python
class UncertaintySuite:
    """
    Uncertainty quantification for predictions.
    
    Parameters
    ----------
    dataset : DBDataset
        Dataset for testing
    model : Any
        Model to evaluate
    config : str or dict
        Configuration level or custom config
    
    Methods
    -------
    run()
        Execute uncertainty tests
    get_coverage_by_alpha()
        Get coverage rates for different confidence levels
    
    Returns
    -------
    Dict containing:
        - crqr: Conformal prediction results
        - coverage_error: Difference from expected coverage
        - avg_interval_width: Average prediction interval size
        - uncertainty_quality_score: Overall uncertainty calibration
    """
```

#### ResilienceSuite

```python
class ResilienceSuite:
    """
    Test model resilience to distribution shifts.
    
    Parameters
    ----------
    dataset : DBDataset
        Dataset for testing
    model : Any
        Model to evaluate
    config : str or dict
        Configuration level or custom config
    
    Methods
    -------
    run()
        Execute resilience tests
    analyze_drift(drift_type='covariate')
        Analyze specific drift type
    
    Returns
    -------
    Dict containing:
        - distribution_shift: Results by shift intensity
        - resilience_score: Overall resilience metric
        - feature_distances: Feature-level drift measures
        - performance_gaps: Performance degradation
    """
```

#### HyperparameterSuite

```python
class HyperparameterSuite:
    """
    Hyperparameter importance analysis.
    
    Parameters
    ----------
    dataset : DBDataset
        Dataset for testing
    model : Any
        Model to evaluate
    config : str or dict
        Configuration level or custom config
    
    Returns
    -------
    Dict containing:
        - importance_scores: Score for each hyperparameter
        - tuning_order: Recommended tuning sequence
        - sensitivity_analysis: Parameter sensitivity metrics
    """
```

### deepbridge.distillation

#### AutoDistiller

```python
class AutoDistiller:
    """
    Automated model distillation with hyperparameter optimization.
    
    Parameters
    ----------
    dataset : DBDataset
        Training dataset
    output_dir : str
        Directory for saving results
    model_types : List[str], default=['gbm', 'mlp', 'xgboost']
        Student model types to try
    n_trials : int, default=50
        Number of optimization trials
    test_size : float, default=0.2
        Test set proportion
    
    Methods
    -------
    run(use_probabilities=True)
        Execute automated distillation
    get_best_model()
        Return best student model
    compare_results()
        Compare all student models
    
    Examples
    --------
    >>> distiller = AutoDistiller(dataset, './results', n_trials=100)
    >>> results = distiller.run()
    >>> best_model = distiller.get_best_model()
    """
    
    def __init__(self, dataset, output_dir, model_types=None, n_trials=50, test_size=0.2)
```

#### KnowledgeDistillation

```python
class KnowledgeDistillation:
    """
    Traditional knowledge distillation from teacher to student.
    
    Parameters
    ----------
    teacher_model : Any
        Teacher model
    student_type : str
        Type of student model ('mlp', 'gbm', 'xgboost')
    temperature : float, default=3.0
        Distillation temperature
    alpha : float, default=0.7
        Weight for distillation loss
    
    Methods
    -------
    fit(X, y, X_val=None, y_val=None)
        Train student model
    predict(X)
        Make predictions
    evaluate(X, y)
        Evaluate student performance
    """
```

### deepbridge.synthetic

#### StandardGenerator

```python
class StandardGenerator:
    """
    Standard synthetic data generator.
    
    Parameters
    ----------
    method : str, default='gaussian_copula'
        Generation method
    n_samples : int, optional
        Number of samples to generate
    random_state : int, optional
        Random seed
    
    Methods
    -------
    fit(data)
        Learn data distribution
    generate(n_samples=None)
        Generate synthetic samples
    fit_generate(data, n_samples=None)
        Fit and generate in one step
    
    Examples
    --------
    >>> generator = StandardGenerator(method='gaussian_copula')
    >>> synthetic = generator.fit_generate(real_data, n_samples=1000)
    """
```

#### SyntheticMetrics

```python
class SyntheticMetrics:
    """
    Evaluate synthetic data quality.
    
    Methods
    -------
    evaluate(real_data, synthetic_data)
        Comprehensive quality assessment
    statistical_similarity(real_data, synthetic_data)
        Statistical tests
    privacy_assessment(real_data, synthetic_data)
        Privacy preservation metrics
    utility_score(real_data, synthetic_data, task='classification')
        ML utility evaluation
    
    Returns
    -------
    Dict containing:
        - statistical: Statistical similarity scores
        - privacy: Privacy metrics
        - utility: ML performance comparison
        - overall_score: Combined quality metric
    """
```

### deepbridge.metrics

#### ClassificationMetrics

```python
class ClassificationMetrics:
    """
    Comprehensive classification metrics.
    
    Methods
    -------
    calculate(y_true, y_pred, y_proba=None)
        Calculate all metrics
    get_summary()
        Get key metric summary
    plot_confusion_matrix()
        Generate confusion matrix plot
    plot_roc_curve()
        Generate ROC curve
    
    Available Metrics
    ----------------
    - accuracy, precision, recall, f1_score
    - auc_roc, auc_pr, log_loss
    - matthews_corrcoef, cohen_kappa
    - balanced_accuracy
    """
```

#### RegressionMetrics

```python
class RegressionMetrics:
    """
    Comprehensive regression metrics.
    
    Methods
    -------
    calculate(y_true, y_pred)
        Calculate all metrics
    residual_analysis()
        Analyze prediction residuals
    plot_predictions()
        Scatter plot of predictions
    
    Available Metrics
    ----------------
    - mse, rmse, mae, mape
    - r2, adjusted_r2
    - explained_variance
    - max_error, median_absolute_error
    """
```

### deepbridge.utils

#### ModelRegistry

```python
class ModelRegistry:
    """
    Central registry for model information.
    
    Methods
    -------
    register_model(model, name, metadata=None)
        Register a model
    get_model_info(model)
        Get model type and properties
    list_supported_models()
        List all supported model types
    
    Supported Models
    ---------------
    - Scikit-learn: All classifiers and regressors
    - XGBoost: XGBClassifier, XGBRegressor
    - LightGBM: LGBMClassifier, LGBMRegressor
    - CatBoost: CatBoostClassifier, CatBoostRegressor
    - PyTorch: Neural network models
    - TensorFlow/Keras: Deep learning models
    """
```

#### FeatureManager

```python
class FeatureManager:
    """
    Feature analysis and management.
    
    Methods
    -------
    identify_features(data)
        Identify feature types
    get_feature_stats(data)
        Calculate feature statistics
    detect_correlations(data, threshold=0.9)
        Find highly correlated features
    suggest_transformations(data)
        Recommend feature transformations
    """
```

#### DataValidator

```python
class DataValidator:
    """
    Data quality validation.
    
    Methods
    -------
    validate(data)
        Comprehensive validation
    check_missing_values(data)
        Identify missing data patterns
    check_duplicates(data)
        Find duplicate rows
    check_outliers(data, method='iqr')
        Detect outliers
    generate_report()
        Create validation report
    """
```

### deepbridge.cli

#### CLI Commands

```python
# Main command groups
deepbridge dataset    # Dataset operations
deepbridge validate   # Model validation
deepbridge distill    # Model distillation
deepbridge synthetic  # Synthetic data generation
deepbridge report     # Report generation

# Common options
--config PATH         # Configuration file
--output PATH         # Output directory
--verbose            # Verbose output
--parallel INT       # Number of parallel jobs
```

## Configuration

### Test Configurations

```python
# Predefined configurations
CONFIGS = {
    'quick': {
        'robustness': {'n_trials': 3, 'perturbation_levels': [0.1, 0.5, 1.0]},
        'uncertainty': {'alpha_levels': [0.05, 0.1, 0.2]},
        'resilience': {'drift_intensities': [0.05, 0.15, 0.3]},
        'hyperparameter': {'n_subsamples': 5, 'cv_folds': 3}
    },
    'medium': {
        'robustness': {'n_trials': 5, 'perturbation_levels': [0.1, 0.3, 0.5, 0.7, 1.0]},
        'uncertainty': {'alpha_levels': [0.01, 0.05, 0.1, 0.15, 0.2]},
        'resilience': {'drift_intensities': [0.01, 0.05, 0.1, 0.2, 0.3]},
        'hyperparameter': {'n_subsamples': 10, 'cv_folds': 5}
    },
    'full': {
        'robustness': {'n_trials': 10, 'perturbation_levels': np.linspace(0.1, 1.0, 10)},
        'uncertainty': {'alpha_levels': np.linspace(0.01, 0.3, 10)},
        'resilience': {'drift_intensities': np.linspace(0.01, 0.3, 10)},
        'hyperparameter': {'n_subsamples': 20, 'cv_folds': 5}
    }
}
```

### Custom Configuration

```python
# Create custom configuration
custom_config = {
    'n_trials': 7,
    'perturbation_levels': [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
    'methods': ['raw', 'quantile'],
    'feature_subset': ['important_feature1', 'important_feature2']
}

suite = RobustnessSuite(dataset, model, config=custom_config)
```

## Type Definitions

```python
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd

# Common type aliases
Model = Any  # Scikit-learn compatible model
Dataset = Union[pd.DataFrame, np.ndarray]
Features = Union[List[str], np.ndarray]
Predictions = np.ndarray
Config = Union[str, Dict[str, Any]]
Results = Dict[str, Any]
```

## Error Handling

### Common Exceptions

```python
class DeepBridgeError(Exception):
    """Base exception for DeepBridge."""

class DataError(DeepBridgeError):
    """Raised for data-related issues."""

class ModelError(DeepBridgeError):
    """Raised for model-related issues."""

class ConfigError(DeepBridgeError):
    """Raised for configuration issues."""

class ValidationError(DeepBridgeError):
    """Raised for validation failures."""
```

### Error Handling Example

```python
from deepbridge.exceptions import DataError, ModelError

try:
    dataset = DBDataset(df, target_column='label')
except DataError as e:
    logger.error(f"Data error: {e}")
    # Handle missing target column
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

## Best Practices

### Memory Management

```python
# Use generators for large datasets
def process_in_batches(dataset, batch_size=1000):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]

# Enable garbage collection
import gc
gc.collect()
```

### Performance Optimization

```python
# Use parallel processing
from joblib import Parallel, delayed

results = Parallel(n_jobs=-1)(
    delayed(process_model)(model, data) 
    for model in models
)

# Enable caching
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(params):
    # Computation here
    pass
```

### Logging

```python
from deepbridge.utils.logger import get_logger

logger = get_logger(__name__)

logger.info("Starting experiment")
logger.debug(f"Dataset shape: {dataset.shape}")
logger.warning("Model not fitted")
logger.error("Validation failed")
```

## Version Information

```python
import deepbridge

print(deepbridge.__version__)  # Current version
print(deepbridge.__author__)   # Author information
print(deepbridge.__license__)  # License type
```