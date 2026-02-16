"""
Example: Robustness Testing with WeakSpot Detection

This example shows how to test model robustness and identify weak regions.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deepbridge import DBDataset, Experiment


# Generate sample data
np.random.seed(42)
n_samples = 1000

df = pd.DataFrame({
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.randn(n_samples),
    'feature3': np.random.uniform(0, 100, n_samples),
    'target': np.random.randint(0, 2, n_samples),
})

# Train a model
X_train, X_test, y_train, y_test = train_test_split(
    df[['feature1', 'feature2', 'feature3']],
    df['target'],
    test_size=0.3,
    random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create dataset with model
test_df = X_test.copy()
test_df['target'] = y_test
test_df = test_df.reset_index(drop=True)  # Reset indices for consistency

dataset = DBDataset(
    data=test_df,
    target_column='target',
    model=model
)

# Create experiment
experiment = Experiment(
    dataset=dataset,
    experiment_type='binary_classification',
    tests=['robustness']
)

# Run robustness test
print("Running robustness test...")
results = experiment.run_tests(config_name='medium')
print(f"✅ Robustness test completed!")

# Generate report
print("\n📊 Generating report...")
report_path = experiment.save_html('robustness', './reports/robustness_report.html')
print(f"Report saved to: {report_path}")
