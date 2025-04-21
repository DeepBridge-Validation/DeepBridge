"""
Test script for generating a robustness report with the new template structure.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Add the parent directory to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import DeepBridge modules
from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite

class SimpleDataset:
    """Simple dataset wrapper for testing."""
    
    def __init__(self):
        # Create synthetic data
        X, y = make_classification(
            n_samples=1000, 
            n_features=10,
            n_informative=6,
            n_redundant=2,
            random_state=42
        )
        
        # Create feature names
        self.features = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        
        # Convert to pandas DataFrame/Series
        self.train_data = pd.DataFrame(X_train, columns=self.features)
        self.train_target = pd.Series(y_train, name="target")
        
        self.test_data = pd.DataFrame(X_test, columns=self.features)
        self.test_target = pd.Series(y_test, name="target")
        
        # Train a simple model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.train_data, self.train_target)
        
        # Set problem type
        self.problem_type = "classification"
    
    def get_feature_data(self, data_type="test"):
        """Get feature data."""
        if data_type == "train":
            return self.train_data
        else:
            return self.test_data
    
    def get_target_data(self, data_type="test"):
        """Get target data."""
        if data_type == "train":
            return self.train_target
        else:
            return self.test_target


def generate_test_report():
    """Generate a test robustness report."""
    
    # Create a simple dataset
    dataset = SimpleDataset()
    
    # Create a robustness suite
    suite = RobustnessSuite(
        dataset,
        verbose=True,
        metric='AUC',
        n_iterations=3
    )
    
    # Configure with medium preset
    suite.config('medium')
    
    # Run the tests
    results = suite.run()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the report
    output_path = os.path.join(output_dir, "test_robustness_report.html")
    report_path = suite.save_report(output_path, "Test Random Forest")
    
    print(f"\nReport generated at: {report_path}")
    return report_path


if __name__ == "__main__":
    # Generate the test report
    report_path = generate_test_report()
    
    # Print instructions
    print("\nTo view the report, open the file in a web browser:")
    print(f"  file://{os.path.abspath(report_path)}")