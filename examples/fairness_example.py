"""
Example: Fairness Testing

This example shows how to test model fairness across protected attributes.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from deepbridge import DBDataset, Experiment


# Generate sample data with protected attributes
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.exponential(50000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'gender': np.random.choice(['M', 'F'], n_samples),
    'race': np.random.choice(['A', 'B', 'C'], n_samples),
    'approved': np.random.randint(0, 2, n_samples),
})

# Train model (only on non-protected features)
features = ['age', 'income', 'credit_score']
X = df[features]
y = df['approved']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Create dataset - DBDataset will automatically use all columns in the dataframe
# The model was trained only on non-protected features, ensuring fairness
# But we need all features available for fairness testing
dataset = DBDataset(
    data=df,  # Full dataframe including protected attributes
    target_column='approved',
    model=model
)

# Create experiment with protected attributes
experiment = Experiment(
    dataset=dataset,
    experiment_type='binary_classification',
    tests=['fairness'],
    protected_attributes=['gender', 'race']
)

# Run fairness test
print("Running fairness test...")
results = experiment.run_tests(config_name='full')
print(f"✅ Fairness test completed!")

# Generate report
print("\n📊 Generating report...")
report_path = experiment.save_html('fairness', './reports/fairness_report.html')
print(f"Report saved to: {report_path}")
