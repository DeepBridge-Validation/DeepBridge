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

# Train model
features = ['age', 'income', 'credit_score']
X = df[features]
y = df['approved']

model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Create dataset
dataset = DBDataset(
    data=df,
    target_column='approved',
    features=features,
    model=model
)

# Create experiment with protected attributes
experiment = Experiment(
    name='fairness_test',
    dataset=dataset,
    experiment_type='binary_classification',
    tests=['fairness'],
    protected_attributes=['gender', 'race']
)

# Run fairness test
print("Running fairness test...")
result = experiment.run_test('fairness', config='full')

print(f"\n✅ Fairness score: {result.overall_fairness_score:.3f}")
print(f"Critical issues: {len(result.critical_issues)}")
print(f"EEOC compliant: {result.overall_fairness_score >= 0.80}")

# Generate report
report_path = experiment.generate_report('fairness', output_dir='./reports')
print(f"\n📊 Report saved to: {report_path}")
