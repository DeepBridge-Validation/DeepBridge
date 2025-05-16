"""Script to run uncertainty test and analyze results"""
import os
import json
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from deepbridge.core.experiment import Experiment
from deepbridge.core.db_data import DBData

# Create synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create dataset
train_data = DBData(X_train, y_train)
test_data = DBData(X_test, y_test)

# Create and train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create experiment
experiment = Experiment(name="uncertainty_test")
experiment.add_model(model, "RandomForest")
experiment.set_data(train_data, test_data)

# Run uncertainty test
print("Running uncertainty test...")
results = experiment.run_uncertainty_test()

# Save results structure to file
results_structure = {}
for model_name, model_results in results.model_results.items():
    results_structure[model_name] = {}
    for key, value in model_results.uncertainty.items():
        if hasattr(value, 'shape'):
            results_structure[model_name][key] = f"Array with shape: {value.shape}"
        elif isinstance(value, dict):
            results_structure[model_name][key] = {k: str(type(v)) for k, v in value.items()}
        else:
            results_structure[model_name][key] = str(type(value))

# Save the structure
with open('uncertainty_results_structure.json', 'w') as f:
    json.dump(results_structure, f, indent=2)

print("Saved results structure to uncertainty_results_structure.json")

# Now let's look at the renderer expectations
print("\nAnalyzing static uncertainty renderer requirements...")

# Extract renderer file path for reference
renderer_file = "/home/guhaase/projetos/DeepBridge/deepbridge/core/experiment/report/renderers/static/static_uncertainty_renderer.py"
print(f"Renderer file: {renderer_file}")