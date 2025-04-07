from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import sys
import os

# Add the parent directory to the path so we can import deepbridge modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, 
                          n_redundant=2, random_state=42)

# Convert to pandas DataFrame
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Mock the deepbridge.core.db_data.DBDataset class
class MockDBDataset:
    def __init__(self, X_train, X_test, y_train, y_test, model, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.feature_names = feature_names
        self.config = {"model_type": "classification"}
    
    def get_feature_data(self, split='train'):
        if split == 'train':
            return self.X_train
        return self.X_test
    
    def get_target_data(self, split='train'):
        if split == 'train':
            return self.y_train
        return self.y_test
    
    def get_feature_names(self):
        return self.feature_names

# Create mock implementations for imports
sys.modules['deepbridge.core.db_data'] = type('', (), {})()
sys.modules['deepbridge.core.db_data'].DBDataset = MockDBDataset
sys.modules['deepbridge.metrics.classification'] = type('', (), {})()
sys.modules['deepbridge.metrics.classification'].Classification = type('Classification', (), {
    'calculate_metrics': lambda self, y_true, y_pred, y_prob, teacher_prob: {
        'accuracy': 0.982,
        'f1': 0.9819959474671671,
        'precision': 0.982151158739503,
        'recall': 0.982,
        'auc': 0.97 if y_prob is not None else None
    }
})

sys.modules['deepbridge.utils.model_registry'] = type('', (), {})()
sys.modules['deepbridge.utils.model_registry'].ModelType = type('ModelType', (), {
    'LOGISTIC_REGRESSION': 'logistic_regression'
})

# Mock the experiment manager classes
sys.modules['deepbridge.core.experiment.data_manager'] = type('', (), {})()
sys.modules['deepbridge.core.experiment.data_manager'].DataManager = type('DataManager', (), {
    '__init__': lambda self, dataset, test_size, random_state: None,
    'prepare_data': lambda self: None,
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'prob_train': None,
    'prob_test': None
})

sys.modules['deepbridge.core.experiment.model_evaluation'] = type('', (), {})()
sys.modules['deepbridge.core.experiment.model_evaluation'].ModelEvaluation = type('ModelEvaluation', (), {
    '__init__': lambda self, experiment_type, metrics_calculator: None,
    'calculate_metrics': lambda self, y_true, y_pred, y_prob, teacher_prob: {
        'accuracy': 0.982,
        'f1': 0.9819959474671671,
        'precision': 0.982151158739503,
        'recall': 0.982,
        'auc': 0.97 if y_prob is not None else None
    }
})

sys.modules['deepbridge.core.experiment.report_generator'] = type('', (), {})()
sys.modules['deepbridge.core.experiment.report_generator'].ReportGenerator = type('ReportGenerator', (), {
    '__init__': lambda self: None
})

sys.modules['deepbridge.core.experiment.managers'] = type('', (), {})()
sys.modules['deepbridge.core.experiment.managers'].ModelManager = type('ModelManager', (), {
    '__init__': lambda self, dataset, experiment_type, verbose: None,
    'create_alternative_models': lambda self, X_train, y_train: {}
})

sys.modules['deepbridge.core.experiment.interfaces'] = type('', (), {})()
sys.modules['deepbridge.core.experiment.interfaces'].IExperiment = type('IExperiment', (), {})

# Test Runner - this is the core of our test
from core.experiment.test_runner import TestRunner

# Create a mock dataset
dataset = MockDBDataset(X_train, X_test, y_train, y_test, model, feature_names)

# Create a test runner
test_runner = TestRunner(
    dataset=dataset,
    alternative_models={},
    tests=["robustness", "uncertainty"],
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    verbose=True,
    feature_subset=feature_names[:2]
)

# Run initial tests
results = test_runner.run_initial_tests()

# Print the metrics
print("\n===== METRICS =====")
print(results['models']['primary_model']['metrics'])

# Create an Experiment class that just returns the initial results from our TestRunner
sys.modules['deepbridge.core.experiment.runner'] = type('', (), {})()
sys.modules['deepbridge.core.experiment.runner'].TestRunner = TestRunner

sys.modules['deepbridge.core.experiment.visualization'] = type('', (), {})()
sys.modules['deepbridge.core.experiment.visualization'].VisualizationManager = type('VisualizationManager', (), {
    '__init__': lambda self, test_runner: None
})

sys.modules['deepbridge.core.experiment.results'] = type('', (), {})()
sys.modules['deepbridge.core.experiment.results'].wrap_results = lambda x: x

# Import the Experiment class
from core.experiment.experiment import Experiment

# Create directly an Experiment subclass that uses our test_runner
class TestExperiment(Experiment):
    def __init__(self, dataset, experiment_type, tests, feature_subset, suite):
        # Skip the parent initialization
        self.dataset = dataset
        self.experiment_type = experiment_type
        self.tests = tests
        self.feature_subset = feature_subset
        self.config_name = suite
        self.verbose = True
        
        # Create the test runner
        self.test_runner = test_runner
        
        # Get initial results
        self.initial_results = self.test_runner.run_initial_tests()

# Create the test experiment
test_exp = TestExperiment(
    dataset=dataset,
    experiment_type="binary_classification",
    tests=["robustness", "uncertainty"],
    feature_subset=feature_names[:2],
    suite="quick"
)

# Print the metrics from the experiment
print("\n===== EXPERIMENT METRICS =====")
print(test_exp.initial_results['models']['primary_model']['metrics'])