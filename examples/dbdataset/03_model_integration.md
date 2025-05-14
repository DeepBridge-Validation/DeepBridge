# Working with Models in DBDataset

This example demonstrates how to create a `DBDataset` with a model, load models from files, and work with model predictions.

```python
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from deepbridge.core.db_data import DBDataset
from pathlib import Path

# Load a sample dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')
data = pd.concat([X, y], axis=1)

# Method 1: Create dataset first, then add model later
db_dataset = DBDataset(
    data=data,
    target_column='species',
    test_size=0.3,
    random_state=42
)

# Create and train a model
X_train = db_dataset.get_feature_data('train')
y_train = db_dataset.get_target_data('train')
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Add the model to the dataset
db_dataset.set_model(model)
print(f"Model type: {type(db_dataset.model).__name__}")

# Method 2: Provide model directly when creating dataset
# This is useful when you already have a trained model
model2 = RandomForestClassifier(n_estimators=50, random_state=42)
model2.fit(X, y)  # Train on the entire dataset for simplicity

db_dataset2 = DBDataset(
    data=data,
    target_column='species',
    model=model2,  # Provide the model directly
    test_size=0.3,
    random_state=42
)

# Method 3: Provide model path (file needs to exist)
# First, let's save a model to a file
import os
model_path = os.path.join(os.getcwd(), 'iris_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

# Now create a dataset with model path
try:
    db_dataset3 = DBDataset(
        data=data,
        target_column='species',
        model_path=model_path,  # Path to the saved model
        test_size=0.3,
        random_state=42
    )
    print(f"Model loaded from file: {type(db_dataset3.model).__name__}")
except Exception as e:
    print(f"Error loading model from file: {str(e)}")

# Method 4: Working with pre-computed predictions
# This is useful when you don't have the original model
# but have the predictions it generated

# Generate predictions with our existing model
X_test = db_dataset.get_feature_data('test')
train_probs = model.predict_proba(X_train)
test_probs = model.predict_proba(X_test)

# Convert to DataFrame
prob_columns = [f'prob_class_{i}' for i in range(3)]  # Iris has 3 classes
train_prob_df = pd.DataFrame(train_probs, columns=prob_columns)
test_prob_df = pd.DataFrame(test_probs, columns=prob_columns)

# Create dataset with just the probabilities (no model)
db_dataset4 = DBDataset(
    train_data=pd.concat([X_train.reset_index(drop=True), 
                         db_dataset.get_target_data('train').reset_index(drop=True)], axis=1),
    test_data=pd.concat([X_test.reset_index(drop=True), 
                        db_dataset.get_target_data('test').reset_index(drop=True)], axis=1),
    target_column='species',
    prob_cols=prob_columns,
    train_predictions=train_prob_df,
    test_predictions=test_prob_df
)

print(f"Dataset created with predictions only (no model object)")

# Clean up the file we created
os.remove(model_path)
```

## Key Points

- `DBDataset` offers multiple ways to work with models:
  1. Create the dataset first, then add a model with `set_model()`
  2. Provide a trained model when creating the dataset
  3. Provide a path to a saved model file
  4. Provide pre-computed predictions when you don't have the original model

- When providing a model or model path, `DBDataset` automatically generates predictions
- The `prob_cols` parameter allows working with pre-computed probabilities when you don't have the model