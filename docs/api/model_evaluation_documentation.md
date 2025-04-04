# ModelEvaluation Documentation

## Overview

The `ModelEvaluation` class is a specialized component in the DeepBridge framework responsible for assessing model performance and calculating metrics. It provides comprehensive evaluation capabilities for various model types, with particular emphasis on distillation model evaluation and model comparison. This component centralizes metric calculation logic and ensures consistent evaluation across the framework.

## Class Definition

```python
class ModelEvaluation:
    """
    Handles model evaluation, metric calculation, and model comparison.
    """
    
    def __init__(self, experiment_type, metrics_calculator):
        self.experiment_type = experiment_type
        self.metrics_calculator = metrics_calculator
```

## Key Responsibilities

The `ModelEvaluation` class has several key responsibilities:

1. **Metric Calculation**: Computes performance metrics for models based on experiment type
2. **Distillation Evaluation**: Specializes in evaluating teacher-student model relationships
3. **Distribution Comparison**: Analyzes similarities between probability distributions
4. **Model Comparison**: Provides tools for comparing multiple models' performance
5. **Prediction Generation**: Creates structured output of model predictions

## Core Methods

### calculate_metrics

```python
def calculate_metrics(self, 
                     y_true: t.Union[np.ndarray, pd.Series],
                     y_pred: t.Union[np.ndarray, pd.Series],
                     y_prob: t.Optional[t.Union[np.ndarray, pd.Series]] = None,
                     teacher_prob: t.Optional[t.Union[np.ndarray, pd.Series]] = None) -> dict:
    """
    Calculate metrics based on experiment type.
    """
```

This method serves as a delegator to type-specific metric calculators, routing the calculation based on the experiment type (e.g., binary classification, regression). It provides a consistent interface regardless of the underlying model type or experiment configuration.

### evaluate_distillation

```python
def evaluate_distillation(self, model, dataset, X, y, prob=None):
    """
    Evaluate the distillation model for the specified dataset.
    """
```

This method specializes in evaluating distillation models by:
1. Generating predictions from the student model
2. Calculating standard performance metrics
3. Computing distribution comparison metrics between teacher and student
4. Organizing the results into a structured format with predictions

The method handles various probability formats and ensures consistent evaluation across different distillation approaches.

### compare_all_models

```python
def compare_all_models(self, dataset, original_model, alternative_models, distilled_model, X, y):
    """Compare all models on the specified dataset"""
```

This method provides a comprehensive comparison between different models:
1. The original/teacher model
2. Alternative models (e.g., different algorithms or configurations)
3. The distilled student model

It returns a DataFrame with standardized metrics for easy comparison, making it simple to assess the trade-offs between models.

## Distribution Metrics

The `ModelEvaluation` class includes specialized methods for comparing probability distributions between teacher and student models:

```python
def _calculate_distribution_metrics(self, teacher_probs, student_probs):
    """Calculate statistical metrics comparing distributions"""
```

This method computes:
1. **Kolmogorov-Smirnov (KS) statistic**: Measures the maximum difference between cumulative distributions
2. **KS p-value**: Indicates the statistical significance of the KS test
3. **R² score**: Measures how well the student distribution aligns with the teacher distribution

```python
def _add_distribution_metrics(self, metrics, teacher_probs, student_probs, ks_stat, ks_pvalue, r2):
    """Add distribution comparison metrics to the metrics dictionary"""
```

This method enhances the metrics dictionary with:
1. The computed KS statistics
2. The R² score for distribution alignment
3. **KL divergence**: A measure of how the student probability distribution differs from the teacher distribution

These distribution metrics are crucial for evaluating knowledge distillation quality beyond just accuracy metrics.

## Model-specific Evaluation

The `ModelEvaluation` class handles the complexities of different model types:

```python
def evaluate_model(self, model, model_name, model_type, X, y):
    """Evaluate a single model"""
```

This method:
1. Detects whether the model is a standard classifier or a regression-based surrogate
2. Handles probability extraction appropriately for each model type
3. Adapts evaluation to the model's prediction interface
4. Calculates metrics while accounting for model-specific characteristics

This flexibility allows consistent evaluation across different algorithm families and distillation approaches.

## Working with Predictions

The class provides methods for generating and formatting predictions:

```python
def get_predictions(self, model, X, y_true):
    """Get predictions from a model"""
```

This method returns a DataFrame containing:
1. True labels (`y_true`)
2. Predicted labels (`y_pred`)
3. Probability estimates for each class (`prob_0`, `prob_1` for binary classification)

This structured output is useful for further analysis, visualization, or error examination.

## Integration with Experiment

Within the `Experiment` class, the `ModelEvaluation` component is initialized during experiment creation:

```python
# From Experiment.__init__
self.model_evaluation = ModelEvaluation(self.experiment_type, self.metrics_calculator)
```

The `Experiment` class delegates evaluation tasks to this component:

```python
# From Experiment.fit
train_metrics = self.model_evaluation.evaluate_distillation(
    self.distillation_model, 'train', 
    self.X_train, self.y_train, self.prob_train
)
```

This delegation ensures:
1. Consistent evaluation across the framework
2. Clear separation of evaluation logic from experiment management
3. Specialized handling of different evaluation scenarios

## Usage Example

The `ModelEvaluation` class can be used independently:

```python
from deepbridge.core.experiment.model_evaluation import ModelEvaluation
from deepbridge.metrics.classification import Classification

# Create evaluator for binary classification
evaluator = ModelEvaluation(
    experiment_type='binary_classification',
    metrics_calculator=Classification()
)

# Evaluate distilled model
evaluation_results = evaluator.evaluate_distillation(
    model=student_model,
    dataset='test',
    X=X_test,
    y=y_test,
    prob=teacher_probabilities
)

# Compare multiple models
comparison = evaluator.compare_all_models(
    dataset='test',
    original_model=teacher_model,
    alternative_models={'random_forest': rf_model, 'xgboost': xgb_model},
    distilled_model=student_model,
    X=X_test,
    y=y_test
)

# Display comparison results
print(comparison)
```

## Implementation Notes

- The evaluator handles both probability-based and label-based metrics
- It automatically detects and adapts to different model interfaces
- The class implements graceful error handling to ensure robustness
- Distribution metrics provide deeper insights into distillation quality
- The component works with both pandas and numpy data structures

## Integration with Metrics Calculators

The `ModelEvaluation` class is designed to work with specialized metrics calculators:

1. It receives a metrics calculator during initialization based on the experiment type
2. It delegates the detailed metric computation to these specialized calculators
3. It enhances the basic metrics with distribution comparison metrics

This separation allows:
- Easy extension with new metric types
- Specialization for different problem domains
- Clear responsibility boundaries

## Extension Points

To extend the `ModelEvaluation` class for new experiment types:

1. Implement a new metrics calculator for the experiment type
2. Update the `calculate_metrics` method to handle the new experiment type
3. Add specialized evaluation logic if needed for the new experiment type

The class is designed to be extensible while maintaining a consistent interface for the rest of the framework.