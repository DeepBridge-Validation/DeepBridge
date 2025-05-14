# Feature Management in DBDataset

This example demonstrates how to work with features in `DBDataset`, including specifying features, handling categorical features, and feature subsets.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from deepbridge.core.db_data import DBDataset

# Create a sample dataset with mixed types
np.random.seed(42)
n_samples = 1000

# Generate numerical features
X_num, y = make_classification(n_samples=n_samples, n_features=4, 
                              n_informative=2, n_redundant=1, 
                              random_state=42)

# Convert to DataFrame
df = pd.DataFrame(X_num, columns=['num1', 'num2', 'num3', 'num4'])

# Add categorical features
df['cat1'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
df['cat2'] = np.random.choice(['X', 'Y'], size=n_samples)
df['cat3'] = np.random.randint(1, 10, size=n_samples)  # Numeric categorical
df['target'] = y

# Example 1: Let DBDataset automatically infer features and types
db_dataset1 = DBDataset(
    data=df,
    target_column='target',
    test_size=0.2,
    random_state=42,
    # max_categories parameter controls categorical inference
    # Features with unique values <= max_categories will be treated as categorical
    max_categories=5  # cat1, cat2, and cat3 will be detected as categorical
)

print("Example 1: Automatic feature inference")
print(f"All features: {db_dataset1.features}")
print(f"Categorical features: {db_dataset1.categorical_features}")
print(f"Numerical features: {db_dataset1.numerical_features}")

# Example 2: Explicitly specify which features to use
# This is useful when you want to use only a subset of available features
selected_features = ['num1', 'num3', 'cat1']
db_dataset2 = DBDataset(
    data=df,
    target_column='target',
    features=selected_features,  # Only use these features
    test_size=0.2,
    random_state=42
)

print("\nExample 2: Explicitly specified features")
print(f"Selected features: {db_dataset2.features}")
print(f"Categorical features: {db_dataset2.categorical_features}")
print(f"Numerical features: {db_dataset2.numerical_features}")

# Example 3: Explicitly specify categorical features
# This is useful when you want to override automatic inference
db_dataset3 = DBDataset(
    data=df,
    target_column='target',
    # No features specified, so all columns except target will be used
    test_size=0.2,
    random_state=42,
    # Explicitly specify which features are categorical
    categorical_features=['cat1', 'cat2', 'cat3', 'num4']  # Treating num4 as categorical
)

print("\nExample 3: Explicitly specified categorical features")
print(f"All features: {db_dataset3.features}")
print(f"Categorical features: {db_dataset3.categorical_features}")
print(f"Numerical features: {db_dataset3.numerical_features}")

# Using get_feature_data to access only the feature columns
X_train = db_dataset3.get_feature_data('train')
print(f"\nTraining data shape: {X_train.shape}")
print(f"Feature columns: {X_train.columns.tolist()}")

# Access specific data subsets
print("\nAccessing specific data:")
print(f"First 5 rows of training features:\n{db_dataset3.get_feature_data('train').head()}")
print(f"First 5 values of training target:\n{db_dataset3.get_target_data('train').head()}")
```

## Key Points

- `DBDataset` can automatically infer which features are categorical and numerical
- The `max_categories` parameter controls categorical feature inference
- You can explicitly specify which features to use with the `features` parameter
- You can explicitly specify which features are categorical with the `categorical_features` parameter
- The class provides methods to access feature data for both training and testing sets