# Working with scikit-learn Datasets and Models

`DBDataset` can seamlessly work with scikit-learn datasets and models. This example demonstrates how to use `DBDataset` with scikit-learn's datasets and train a model.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from deepbridge.core.db_data import DBDataset

# Load a scikit-learn dataset
breast_cancer = load_breast_cancer()

# Create a DBDataset directly from the scikit-learn dataset
# The DBDataset constructor can handle scikit-learn's Bunch objects
db_dataset = DBDataset(
    data=breast_cancer,        # Pass the scikit-learn dataset directly
    target_column='target',    # The target column name (will be created)
    test_size=0.2,             # 20% of data will be used for testing
    random_state=42            # For reproducibility
)

# The feature names will be automatically extracted from the scikit-learn dataset
print(f"Feature names from breast cancer dataset: {db_dataset.features}")
print(f"Number of features: {len(db_dataset.features)}")

# Train a scikit-learn model using the data from DBDataset
X_train = db_dataset.get_feature_data('train')
y_train = db_dataset.get_target_data('train')

# Create and train a logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Set the model to the DBDataset
# This will automatically generate predictions for both train and test sets
db_dataset.set_model(model)

# Now the dataset has a trained model associated with it
print(f"Model type: {type(db_dataset.model).__name__}")

# You can also load pre-split data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create train and test DataFrames
import pandas as pd
feature_names = breast_cancer.feature_names
train_df = pd.DataFrame(X_train, columns=feature_names)
train_df['target'] = y_train
test_df = pd.DataFrame(X_test, columns=feature_names)
test_df['target'] = y_test

# Create DBDataset from separate train and test DataFrames
db_dataset2 = DBDataset(
    train_data=train_df,       # Pre-split training data
    test_data=test_df,         # Pre-split test data
    target_column='target'     # Target column name
)
```

## Key Points

- `DBDataset` can handle scikit-learn's Bunch objects directly
- Feature names are automatically extracted from the scikit-learn dataset
- You can add a trained model to a `DBDataset` using the `set_model` method
- The class can also be initialized with pre-split train and test data