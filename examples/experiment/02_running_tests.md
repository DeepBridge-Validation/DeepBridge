# Running Tests with Experiment

This example demonstrates how to run various model validation tests using the `Experiment` class.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

# Step 1: Prepare a dataset and model
# ---------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
df = pd.concat([X, y], axis=1)

# Train a model on the entire dataset for simplicity
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create a DBDataset
db_dataset = DBDataset(
    data=df,
    target_column='target',
    model=model,
    test_size=0.2,
    random_state=42
)

# Step 2: Create an Experiment with specific tests
# ----------------------------------------------
# Specify which tests you want to run
# Available tests: "robustness", "uncertainty", "resilience", "hyperparameters"
experiment = Experiment(
    dataset=db_dataset,
    experiment_type="binary_classification",
    tests=["robustness", "uncertainty"],  # We'll run these tests
    random_state=42,
    # Optionally specify a subset of features to focus on
    feature_subset=['mean radius', 'mean texture', 'mean perimeter']
)

# Step 3: Run all specified tests
# -----------------------------
# Run tests with "quick" configuration (fastest but less comprehensive)
# Other options: "medium", "full" (most comprehensive but slowest)
results = experiment.run_tests(config_name="quick")

print("Tests completed:")
for test_name in results.results.keys():
    if test_name != 'initial_results':
        print(f"- {test_name}")

# Step 4: Run a specific test
# -------------------------
# Run just the robustness test with "medium" configuration
robustness_result = experiment.run_test("robustness", config_name="medium")

# Print robustness scores for each feature
print("\nRobustness Scores by Feature:")
if hasattr(robustness_result, 'feature_robustness'):
    for feature, score in robustness_result.feature_robustness.items():
        print(f"{feature}: {score:.4f}")

# Step 5: Access test results
# -------------------------
# Access robustness results
robustness_results = experiment.get_robustness_results()
print("\nRobustness Test Summary:")
if 'overall_robustness_score' in robustness_results:
    print(f"Overall robustness score: {robustness_results['overall_robustness_score']:.4f}")

# Access uncertainty results
uncertainty_results = experiment.get_uncertainty_results()
print("\nUncertainty Test Summary:")
if 'calibration_error' in uncertainty_results:
    print(f"Calibration error: {uncertainty_results['calibration_error']:.4f}")

# Step 6: Run a test with a different configuration
# ----------------------------------------------
# Create a new experiment with different tests
experiment2 = Experiment(
    dataset=db_dataset,
    experiment_type="binary_classification",
    tests=["hyperparameters"],
    random_state=42
)

# Run the hyperparameter test with custom parameters
hyperparameter_result = experiment2.run_test(
    "hyperparameters", 
    config_name="quick",
    # Additional parameters
    n_trials=10,  # Number of hyperparameter combinations to try
    max_features=3  # Maximum number of features to use
)

print("\nHyperparameter Test Results:")
if hasattr(hyperparameter_result, 'best_params'):
    print(f"Best hyperparameters: {hyperparameter_result.best_params}")
```

## Key Points

- The `Experiment` class supports various types of model validation tests
- Tests are specified when creating the experiment but are only run when explicitly called
- Available tests include "robustness", "uncertainty", "resilience", and "hyperparameters"
- Test configurations ("quick", "medium", "full") control the depth and runtime of the tests
- You can run all specified tests with `run_tests()` or a specific test with `run_test()`
- Test results can be accessed through getter methods like `get_robustness_results()`
- You can pass additional parameters to tests through the `run_test()` method