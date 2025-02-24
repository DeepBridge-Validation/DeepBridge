# AutoDistiller Guide

## What is AutoDistiller?

AutoDistiller is a central component of the DeepBridge library that automates the knowledge distillation process for machine learning models. Knowledge distillation is a technique where a simpler model (the "student") learns to mimic the behavior of a more complex model (the "teacher"), resulting in smaller and more efficient models that preserve most of the original performance.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DeepBridge-Validation/DeepBridge/blob/master/examples/quickstart.ipynb)


## Why use AutoDistiller?

- **Automation**: Automated testing of multiple student model configurations
- **Optimization**: Finds the best combination of model type, temperature, and alpha value
- **Simplicity**: Unified and easy-to-use interface for the distillation process
- **Visualization**: Automatically generates visualizations and performance reports
- **Reproducibility**: Ensures experiments are reproducible through random seeds

## AutoDistiller Architecture

AutoDistiller is designed with a modular architecture that separates different responsibilities:

1. **Configuration** (`DistillationConfig`): Manages experiment configuration parameters
2. **Experiment Execution** (`ExperimentRunner`): Runs distillation experiments
3. **Metrics Evaluation** (`MetricsEvaluator`): Analyzes and evaluates results
4. **Visualization** (`Visualizer`): Creates graphs and visualizations
5. **Report Generation** (`ReportGenerator`): Produces detailed reports

## How AutoDistiller Works

### Workflow

1. **Initialization**: The user creates an AutoDistiller instance with a dataset and configurations
2. **Configuration**: Customization of model types, temperatures, and alpha values to be tested
3. **Execution**: Training of multiple models with different configurations
4. **Evaluation**: Calculation and analysis of performance metrics for each model
5. **Visualization**: Automatic generation of comparative graphs
6. **Report**: Creation of a complete report with results and recommendations

### Distillation Process

The distillation process occurs for each combination of:
- **Student model type**: Algorithms such as Logistic Regression, Decision Tree, GBM, XGBoost, etc.
- **Temperature**: A hyperparameter that controls the "softness" of the teacher's probabilities
- **Alpha**: A parameter that controls the balance between mimicking the teacher and adapting to the actual labels

For each combination, AutoDistiller:
1. Creates a student model of the specified type
2. Trains the model using the teacher's predictions and the temperature and alpha values
3. Evaluates the model's performance using multiple metrics
4. Stores the results for comparative analysis

## Key Components

### DBDataset

Before using AutoDistiller, you need to prepare your data using the `DBDataset` class, which encapsulates:
- Training and test data
- Labels/targets
- Teacher model predictions
- Dataset metadata

```python
dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    features=feature_list,
    train_predictions=train_probs_df,
    test_predictions=test_probs_df,
    prob_cols=probability_column_names
)
```

### DistillationConfig

This class manages the configuration of distillation experiments:

```python
config = DistillationConfig(
    output_dir="./results",
    test_size=0.2,
    random_state=42,
    n_trials=10,
    validation_split=0.2
)

# Customize configurations
config.customize(
    model_types=[ModelType.LOGISTIC_REGRESSION, ModelType.GBM],
    temperatures=[0.5, 1.0, 2.0],
    alphas=[0.3, 0.5, 0.7]
)
```

### ExperimentRunner

Executes distillation experiments with the provided configurations:

```python
runner = ExperimentRunner(dataset=dataset, config=config)
results_df = runner.run_experiments()
```

### MetricsEvaluator

Analyzes experiment results:

```python
evaluator = MetricsEvaluator(results_df=results_df, config=config)
best_model = evaluator.find_best_model(metric='test_accuracy')
```

## Important Parameters

### Student Model Types

AutoDistiller supports various model types for distillation:

- `ModelType.LOGISTIC_REGRESSION`: Simple and efficient linear models
- `ModelType.DECISION_TREE`: Easy-to-interpret decision trees
- `ModelType.GBM`: Gradient Boosting Machines for better performance
- `ModelType.XGB`: XGBoost for cases requiring high performance
- `ModelType.MLP`: Multi-Layer Perceptron (simple neural networks)

### Temperature

Temperature (T) controls the "softness" of the teacher's probabilities:

- **T < 1.0**: Makes distributions sharper (more confident)
- **T = 1.0**: Maintains original probabilities
- **T > 1.0**: Makes distributions smoother (less confident)

Higher temperatures generally make learning easier for the student, especially for complex tasks.

### Alpha

Alpha (α) controls the balance between:

- **α = 1.0**: Total focus on mimicking the teacher's probabilities
- **α = 0.0**: Total focus on predicting actual labels
- **0.0 < α < 1.0**: Combination of both approaches

Typical values are between 0.3 and 0.7, with 0.5 being a good starting point.

## Using AutoDistiller

### Initialization and Configuration

```python
from deepbridge.auto_distiller import AutoDistiller
from deepbridge.distillation.classification.model_registry import ModelType

# Create instance
auto_distiller = AutoDistiller(
    dataset=dataset,
    output_dir="./distillation_results",
    test_size=0.2,
    random_state=42,
    n_trials=10,
    validation_split=0.2,
    verbose=True
)

# Customize configuration
auto_distiller.customize_config(
    model_types=[
        ModelType.LOGISTIC_REGRESSION,
        ModelType.DECISION_TREE,
        ModelType.GBM
    ],
    temperatures=[0.5, 1.0, 2.0],
    alphas=[0.3, 0.5, 0.7]
)
```

### Execution

```python
# Run the distillation process
results = auto_distiller.run(use_probabilities=True, verbose_output=True)
```

### Results Analysis

```python
# Find the best model based on a metric
best_config = auto_distiller.find_best_model(metric='test_accuracy', minimize=False)

# Get the trained model
best_model = auto_distiller.get_trained_model(
    model_type=best_config['model_type'],
    temperature=best_config['temperature'],
    alpha=best_config['alpha']
)

# Generate complete report
report = auto_distiller.generate_report()
```

### Saving the Best Model

```python
# Save the best model
model_path = auto_distiller.save_best_model(
    metric='test_accuracy',
    minimize=False,
    file_path='./models/best_distilled_model.pkl'
)
```

## Best Practices

1. **Data Preparation**:
   - Normalize/standardize features before use
   - Ensure there's no data leakage between training/testing

2. **Model Selection**:
   - Start with simpler models (LogisticRegression, DecisionTree)
   - Move to more complex models (GBM, XGB) if necessary

3. **Hyperparameters**:
   - Test different temperatures (0.5, 1.0, 2.0, 5.0)
   - Try various alpha values (0.3, 0.5, 0.7, 0.9)
   - Increase the number of trials for better optimization (n_trials=20+)

4. **Evaluation**:
   - Consider multiple metrics (accuracy, auc_roc, kl_divergence)
   - Compare performance with the teacher model

5. **Visualization**:
   - Examine generated visualizations to understand trade-offs
   - Compare probability distributions to verify fidelity

## Output and Results

AutoDistiller automatically generates:

1. **Results DataFrame**: Containing metrics for each tested configuration
2. **Visualizations**: Comparative graphs of different models and configurations
3. **Markdown Report**: Detailed summary of experiments and results
4. **Optimized Model**: The best distilled model ready for use

## Complete Workflow Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from deepbridge.db_data import DBDataset
from deepbridge.auto_distiller import AutoDistiller
from deepbridge.distillation.classification.model_registry import ModelType

# 1. Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Train teacher model
teacher_model = RandomForestClassifier(n_estimators=500)
teacher_model.fit(X_train, y_train)

# 3. Generate teacher probabilities
train_probs = teacher_model.predict_proba(X_train)
test_probs = teacher_model.predict_proba(X_test)

# 4. Create probability DataFrames
train_probs_df = pd.DataFrame(
    train_probs, 
    columns=[f'prob_class_{i}' for i in range(train_probs.shape[1])],
    index=X_train.index
)

test_probs_df = pd.DataFrame(
    test_probs, 
    columns=[f'prob_class_{i}' for i in range(test_probs.shape[1])],
    index=X_test.index
)

# 5. Create DBDataset
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

dataset = DBDataset(
    train_data=train_data,
    test_data=test_data,
    target_column='target',
    features=X.columns.tolist(),
    train_predictions=train_probs_df,
    test_predictions=test_probs_df,
    prob_cols=[f'prob_class_{i}' for i in range(train_probs.shape[1])]
)

# 6. Initialize AutoDistiller
auto_distiller = AutoDistiller(
    dataset=dataset,
    output_dir="./results",
    test_size=0.2,
    random_state=42,
    n_trials=10,
    validation_split=0.2,
    verbose=True
)

# 7. Configure experiments
auto_distiller.customize_config(
    model_types=[
        ModelType.LOGISTIC_REGRESSION,
        ModelType.DECISION_TREE,
        ModelType.GBM,
        ModelType.XGB
    ],
    temperatures=[0.5, 1.0, 2.0, 5.0],
    alphas=[0.3, 0.5, 0.7, 0.9]
)

# 8. Run distillation
results = auto_distiller.run(use_probabilities=True)

# 9. Analyze results
best_config = auto_distiller.find_best_model(metric='test_accuracy')
print(f"Best model: {best_config['model_type']}")
print(f"Temperature: {best_config['temperature']}")
print(f"Alpha: {best_config['alpha']}")
print(f"Accuracy: {best_config.get('test_accuracy', 'N/A')}")

# 10. Save the best model
auto_distiller.save_best_model(metric='test_accuracy', file_path='./best_model.pkl')
```

## Conclusion

AutoDistiller significantly simplifies the knowledge distillation process, automating the complex tasks of testing multiple configurations and finding the optimal model. It allows you to obtain smaller and more efficient models without sacrificing much performance, facilitating model implementation in environments with limited resources.

When using AutoDistiller, you don't need to worry about the implementation details of knowledge distillation, and can focus on adjusting high-level parameters to achieve the best results for your specific use case.