# Basic Usage of Experiment

This example demonstrates how to create and use the `Experiment` class to evaluate machine learning models.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment

# Step 1: Prepare a dataset with a trained model
# ---------------------------------------------
# Load a sample dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
df = pd.concat([X, y], axis=1)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create a DBDataset with the model
db_dataset = DBDataset(
    train_data=pd.concat([X_train, y_train], axis=1),
    test_data=pd.concat([X_test, y_test], axis=1),
    target_column='target',
    model=model
)

# Step 2: Create an Experiment
# ---------------------------
# Initialize an experiment for binary classification
experiment = Experiment(
    dataset=db_dataset,
    experiment_type="binary_classification",  # One of: "binary_classification", "regression", "forecasting"
    test_size=0.2,  # This is used if dataset doesn't have pre-split data
    random_state=42
)

# Step 3: Access basic experiment information
# ------------------------------------------
# Initial results are available immediately after initialization
initial_results = experiment.initial_results

# Print basic metrics for the primary model
primary_model_metrics = initial_results['models']['primary_model']['metrics']
print("Primary Model Metrics:")
for metric, value in primary_model_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Print metrics for alternative models (created automatically)
if 'models' in initial_results:
    for model_name, model_data in initial_results['models'].items():
        if model_name != 'primary_model':
            print(f"\n{model_name} Metrics:")
            for metric, value in model_data['metrics'].items():
                print(f"  {metric}: {value:.4f}")

# Step 4: Use the model from the experiment
# ----------------------------------------
# Access the model
model = experiment.model

# Make new predictions
new_data = X_test.iloc[:5]  # Just as an example
predictions = model.predict(new_data)
print(f"\nPredictions for new data: {predictions}")

# Compare metrics across different models
comparison = experiment.compare_all_models(dataset='test')
print("\nModel Comparison:")
for model_name, metrics in comparison.items():
    print(f"{model_name}: accuracy={metrics.get('accuracy', 'N/A'):.4f}, roc_auc={metrics.get('roc_auc', 'N/A'):.4f}")

# Access feature importance (if available)
try:
    feature_imp = experiment.get_feature_importance()
    print("\nTop 5 features by importance:")
    for i, (feature, importance) in enumerate(list(feature_imp.items())[:5]):
        print(f"{i+1}. {feature}: {importance:.4f}")
except ValueError as e:
    print(f"\nFeature importance not available: {str(e)}")
```

## Key Points

- The `Experiment` class works with a `DBDataset` that already contains a model
- The experiment automatically creates and evaluates alternative models for comparison
- Initial metrics are calculated during initialization
- The experiment provides methods to access the model and make predictions
- Feature importance is automatically calculated when available