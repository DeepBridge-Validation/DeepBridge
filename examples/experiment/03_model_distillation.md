# Model Distillation with Experiment

This example demonstrates how to use the `Experiment` class for model distillation, which allows you to create simpler, more interpretable models that maintain the performance of complex models.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from deepbridge.core.db_data import DBDataset
from deepbridge.core.experiment.experiment import Experiment
from deepbridge.utils.model_registry import ModelType

# Step 1: Prepare a dataset with a complex model
# --------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')
df = pd.concat([X, y], axis=1)

# Train a complex model (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create a DBDataset with the model
db_dataset = DBDataset(
    data=df,
    target_column='target',
    model=model,  # This is our "teacher" model
    test_size=0.2,
    random_state=42
)

# Step 2: Create an Experiment
# --------------------------
experiment = Experiment(
    dataset=db_dataset,
    experiment_type="binary_classification",
    random_state=42
)

# Step 3: Fit a surrogate model (default distillation method)
# ---------------------------------------------------------
# This creates a simpler model that mimics the complex model
experiment.fit(
    # Choose which type of model to use as the student
    student_model_type=ModelType.LOGISTIC_REGRESSION,
    # Set hyperparameters for the student model (optional)
    student_params={'C': 1.0, 'max_iter': 1000},
    # Temperature controls softness of probabilities (higher = softer)
    temperature=1.0,
    # Alpha controls weight between hard labels and soft probabilities 
    # (1.0 = only hard labels, 0.0 = only soft probabilities)
    alpha=0.5,  
    # Whether to use teacher model's probability outputs
    use_probabilities=True,
    # Number of hyperparameter optimization trials
    n_trials=20,
    # Portion of training data used for validation during distillation
    validation_split=0.2,
    # Whether to display optimization progress
    verbose=True,
    # Distillation method: "surrogate" or "knowledge_distillation"
    distillation_method="surrogate"
)

# Step 4: Evaluate and compare the distilled model
# ----------------------------------------------
# Compare metrics between original and distilled model
comparison = experiment.compare_all_models(dataset='test')
print("Model Comparison:")
for model_name, metrics in comparison.items():
    print(f"{model_name}: accuracy={metrics.get('accuracy', 'N/A'):.4f}, roc_auc={metrics.get('roc_auc', 'N/A'):.4f}")

# Access the distilled model
distilled_model = experiment.distillation_model
print(f"\nDistilled model type: {type(distilled_model).__name__}")

# Get predictions from the distilled model
student_predictions = experiment.get_student_predictions(dataset='test')
print(f"\nDistilled model predictions shape: {student_predictions.shape}")
print(f"Prediction columns: {student_predictions.columns.tolist()}")

# Step 5: Try knowledge distillation with a different student model
# --------------------------------------------------------------
# Note: If we had provided only probabilities without the original model,
# a surrogate model would have been automatically created during initialization
# due to the auto_fit parameter defaulting to True when no model is present.
#
# For example:
# db_dataset_probs_only = DBDataset(
#     data=df,
#     target_column='target',
#     prob_cols=['prob_0', 'prob_1'],  # Only probabilities, no model
#     train_predictions=train_probs_df,
#     test_predictions=test_probs_df
# )
# auto_experiment = Experiment(
#     dataset=db_dataset_probs_only,
#     experiment_type="binary_classification",
#     # auto_fit=True is the default when probabilities exist but model doesn't
# )
# At this point, auto_experiment would already have a distillation_model created automatically

experiment2 = Experiment(
    dataset=db_dataset,
    experiment_type="binary_classification",
    random_state=42
)

# Fit using knowledge distillation (explicitly trains on both labels and probabilities)
experiment2.fit(
    student_model_type=ModelType.DECISION_TREE,
    temperature=2.0,  # Higher temperature makes probabilities softer
    alpha=0.3,  # More weight on soft probabilities (0.3 on hard labels, 0.7 on soft probabilities)
    use_probabilities=True,
    n_trials=20,
    verbose=True,
    distillation_method="knowledge_distillation"  # Explicitly use knowledge distillation
)

# Compare the two distillation approaches
comparison2 = experiment2.compare_all_models(dataset='test')
print("\nKnowledge Distillation Model Comparison:")
for model_name, metrics in comparison2.items():
    if 'distilled' in model_name.lower():
        print(f"{model_name}: accuracy={metrics.get('accuracy', 'N/A'):.4f}, roc_auc={metrics.get('roc_auc', 'N/A'):.4f}")
```

## Key Points

- The `Experiment` class provides model distillation capabilities through the `fit()` method
- Distillation allows you to create simpler models (students) that mimic complex models (teachers)
- Two main distillation methods are supported:
  - `surrogate`: Trains a student model to match the teacher's predictions
  - `knowledge_distillation`: Trains on both true labels and teacher predictions with a weighted loss
- You can control various aspects of distillation:
  - Student model type (logistic regression, decision tree, etc.)
  - Temperature parameter (controls probability "softness")
  - Alpha parameter (balance between hard labels and soft probabilities)
  - Hyperparameter optimization with `n_trials`
- The distilled model is accessible via `experiment.distillation_model`
- Use `compare_all_models()` to evaluate how the distilled model performs against the original
- **Automatic Surrogate Creation**: When you provide a dataset with probabilities but no model to the `Experiment` class, it will automatically create a surrogate model during initialization (via `auto_fit=True`)