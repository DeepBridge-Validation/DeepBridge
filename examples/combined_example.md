# Complete DeepBridge Example Workflow

This example demonstrates a complete workflow using both `DBDataset` and `Experiment` classes together.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.utils.model_registry import ModelType
import os

# Step 1: Load and prepare data
# ---------------------------
# Load a dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
df = pd.concat([X, y], axis=1)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create train and test DataFrames
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# Step 2: Train a complex model
# ---------------------------
# Train a complex random forest model
complex_model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42
)
complex_model.fit(X_train, y_train)

# Step 3: Create a DBDataset with the model
# --------------------------------------
db_dataset = DBDataset(
    train_data=train_df,
    test_data=test_df,
    target_column='target',
    model=complex_model,
    categorical_features=[],  # All features are numerical in this dataset
    dataset_name="Breast Cancer"
)

print(f"Dataset created with {len(db_dataset.features)} features")
print(f"Dataset has {len(db_dataset.train_data)} training samples and {len(db_dataset.test_data)} test samples")

# Step 4: Create an Experiment for model validation
# ----------------------------------------------
experiment = Experiment(
    dataset=db_dataset,
    experiment_type="binary_classification",
    tests=["robustness", "uncertainty"],
    random_state=42
)

# Step 5: Run validation tests
# -------------------------
print("\nRunning validation tests...")
results = experiment.run_tests(config_name="quick")

# Print summary of initial results
initial_metrics = results.results['initial_results']['models']['primary_model']['metrics']
print("\nOriginal model performance:")
for metric, value in initial_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Print robustness summary
if 'robustness' in results.results:
    rob_results = results.results['robustness']
    if 'overall_robustness_score' in rob_results:
        print(f"\nOverall robustness score: {rob_results['overall_robustness_score']:.4f}")
    
    # Print feature robustness scores
    if 'feature_robustness' in rob_results:
        print("\nTop 5 most robust features:")
        sorted_features = sorted(
            rob_results['feature_robustness'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        for feature, score in sorted_features:
            print(f"  {feature}: {score:.4f}")

# Step 6: Create a distilled model
# -----------------------------
print("\nCreating a distilled model...")
experiment.fit(
    student_model_type=ModelType.LOGISTIC_REGRESSION,
    temperature=1.5,
    alpha=0.3,
    use_probabilities=True,
    n_trials=20,
    verbose=False
)

# Step 7: Compare original and distilled models
# ------------------------------------------
comparison = experiment.compare_all_models(dataset='test')
print("\nModel comparison:")
for model_name, metrics in comparison.items():
    print(f"{model_name}: accuracy={metrics.get('accuracy', 'N/A'):.4f}, roc_auc={metrics.get('roc_auc', 'N/A'):.4f}")

# Step 8: Access feature importance
# ------------------------------
# Get feature importance from the original model
try:
    feature_imp = experiment.get_feature_importance()
    print("\nTop 5 most important features:")
    for i, (feature, importance) in enumerate(list(feature_imp.items())[:5]):
        print(f"  {i+1}. {feature}: {importance:.4f}")
except ValueError as e:
    print(f"\nFeature importance not available: {str(e)}")

# Step 9: Generate reports
# ---------------------
# Create directory for reports
output_dir = os.path.join(os.getcwd(), "reports")
os.makedirs(output_dir, exist_ok=True)

# Generate robustness report
rob_report_path = os.path.join(output_dir, "robustness_report.html")
experiment.save_html(
    test_type="robustness",
    file_path=rob_report_path,
    model_name="Breast Cancer Model"
)
print(f"\nRobustness report saved to: {rob_report_path}")

# Step 10: Make predictions with the distilled model
# -----------------------------------------------
# Get a few samples to predict
new_samples = X_test.iloc[:5]

# Original model predictions
original_preds = complex_model.predict(new_samples)
original_probs = complex_model.predict_proba(new_samples)[:, 1]

# Distilled model predictions
distilled_model = experiment.distillation_model
distilled_preds = distilled_model.predict(new_samples)
distilled_probs = distilled_model.predict_proba(new_samples)[:, 1]

print("\nPrediction comparison for 5 test samples:")
print(f"{'Sample':^10}{'Original':^15}{'Distilled':^15}{'Original Prob':^15}{'Distilled Prob':^15}")
print("-" * 70)
for i in range(5):
    print(f"{i:^10}{original_preds[i]:^15}{distilled_preds[i]:^15}{original_probs[i]:.4f:^15}{distilled_probs[i]:.4f:^15}")
```

## Key Points

- This example demonstrates a complete workflow integrating `DBDataset` and `Experiment`
- Steps include:
  1. Loading and preprocessing data
  2. Training an initial complex model
  3. Creating a `DBDataset` with the model
  4. Creating an `Experiment` for model validation
  5. Running validation tests (robustness, uncertainty)
  6. Creating a distilled model using `fit()`
  7. Comparing the original and distilled models
  8. Analyzing feature importance
  9. Generating HTML reports
  10. Making predictions with both models
- This integrated workflow shows how the two classes work together to provide a comprehensive model development and validation pipeline