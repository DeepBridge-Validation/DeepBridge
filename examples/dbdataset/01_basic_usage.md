# Basic Usage of DBDataset

This example demonstrates the most basic usage of the `DBDataset` class with a pandas DataFrame.

```python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from deepbridge.core.db_data import DBDataset

# Create a simple synthetic dataset
def create_sample_data(n_samples=1000):
    """Create a synthetic dataset for demonstration."""
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df

# Create sample data
data = create_sample_data()

# Create a DBDataset instance
# This will automatically split the data into train and test sets
db_dataset = DBDataset(
    data=data,                 # The DataFrame containing both features and target
    target_column='target',    # The name of the target column
    test_size=0.2,             # 20% of data will be used for testing
    random_state=42            # For reproducibility
)

# Now you can access various components of the dataset
print(f"Dataset size: {len(db_dataset)}")
print(f"Number of features: {len(db_dataset.features)}")
print(f"Feature names: {db_dataset.features}")

# Access the train and test data
X_train = db_dataset.get_feature_data('train')
y_train = db_dataset.get_target_data('train')
X_test = db_dataset.get_feature_data('test')
y_test = db_dataset.get_target_data('test')

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

# Properties for working with feature types
print(f"Numerical features: {db_dataset.numerical_features}")
print(f"Categorical features: {db_dataset.categorical_features}")
```

## Key Points

- `DBDataset` can be created with a single DataFrame containing both features and target
- It automatically handles train/test splitting
- The class provides convenient methods to access various aspects of the data
- Feature types (categorical vs numerical) are automatically inferred by default