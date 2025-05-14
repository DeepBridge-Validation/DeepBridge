# Generating Reports from Experiment Results

This example demonstrates how to generate HTML reports from `Experiment` results, which provide interactive visualizations and detailed analysis of model performance and validation tests.

```python
import pandas as pd
import numpy as np
import os
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

# Step 1: Prepare a dataset with a trained model
# --------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
df = pd.concat([X, y], axis=1)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create a DBDataset
db_dataset = DBDataset(
    data=df,
    target_column='target',
    model=model,
    test_size=0.2,
    random_state=42,
    dataset_name="Breast Cancer"  # Name to display in reports
)

# Step 2: Create an Experiment with all tests
# -----------------------------------------
experiment = Experiment(
    dataset=db_dataset,
    experiment_type="binary_classification",
    # Specify all test types
    tests=["robustness", "uncertainty", "resilience", "hyperparameters"],
    random_state=42
)

# Step 3: Run the tests
# -------------------
# Run all tests with 'quick' configuration
results = experiment.run_tests(config_name="quick")

# Step 4: Generate HTML reports for each test
# -----------------------------------------
# Define output directory for reports
output_dir = os.path.join(os.getcwd(), "reports")
os.makedirs(output_dir, exist_ok=True)

# Generate a report for each test type
for test_type in ["robustness", "uncertainty", "resilience", "hyperparameters"]:
    # Generate file path for the report
    report_path = os.path.join(output_dir, f"{test_type}_report.html")
    
    # Generate and save the report
    try:
        output_path = experiment.save_html(
            test_type=test_type,
            file_path=report_path,
            model_name="Breast Cancer Classifier"  # Custom name for the report
        )
        print(f"{test_type.capitalize()} report saved to: {output_path}")
    except ValueError as e:
        print(f"Error generating {test_type} report: {str(e)}")

# Step 5: Generate a report from experiment results directly
# -------------------------------------------------------
# The results object from run_tests() also has a save_html method
robustness_report_path = os.path.join(output_dir, "robustness_report_direct.html")
try:
    output_path = results.save_html(
        test_type="robustness",
        file_path=robustness_report_path
    )
    print(f"\nDirect robustness report saved to: {output_path}")
except Exception as e:
    print(f"Error generating direct report: {str(e)}")

# Step 6: Run a specific test and generate its report
# -------------------------------------------------
# Run just the robustness test with custom parameters
robustness_result = experiment.run_test(
    "robustness",
    config_name="medium",
    n_perturbations=5,  # Custom number of perturbations
    perturbation_magnitude=0.2  # Custom perturbation strength
)

# Generate report for just this test
custom_report_path = os.path.join(output_dir, "custom_robustness_report.html")
try:
    output_path = robustness_result.save_html(
        file_path=custom_report_path
    )
    print(f"\nCustom robustness report saved to: {output_path}")
except Exception as e:
    print(f"Error generating custom report: {str(e)}")
```

## Key Points

- The `Experiment` class and test result objects provide methods to generate HTML reports
- Reports provide interactive visualizations of test results
- Two ways to generate reports:
  1. Use `experiment.save_html()` to generate a report for a test that has been run
  2. Use `results.save_html()` on the object returned by `run_tests()`
- Reports are customizable with parameters like `model_name`
- Each test type has its own report format with appropriate visualizations:
  - Robustness: Feature perturbation effects and model stability
  - Uncertainty: Calibration plots and confidence intervals
  - Resilience: Model performance under distribution shifts
  - Hyperparameters: Feature and parameter importance
- Reports are saved as HTML files that can be viewed in any web browser