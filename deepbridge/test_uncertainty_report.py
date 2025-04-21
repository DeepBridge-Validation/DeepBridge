"""
Test script for generating an uncertainty report with the new template structure.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Add the parent directory to the system path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules directly using importlib to avoid package dependencies
import importlib.util

def import_report_manager():
    """Import the ReportManager class directly from the file."""
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                           "core/experiment/report_manager.py")
    module_name = "report_manager"
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module.ReportManager

class SimpleDataset:
    """Simple dataset wrapper for testing."""
    
    def __init__(self):
        # Create synthetic data
        X, y = make_regression(
            n_samples=1000, 
            n_features=10,
            n_informative=6,
            noise=0.1,
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
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(self.train_data, self.train_target)
        
        # Set problem type
        self.problem_type = "regression"
    
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


def generate_test_uncertainty_report():
    """Generate a test uncertainty report using placeholder data."""
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_path = os.path.join(output_dir, "test_uncertainty_report.html")
    
    # Create dummy data for the uncertainty report
    results = {
        "model_name": "Test Regression Model",
        "model_type": "RandomForestRegressor",
        "timestamp": "2025-04-21 10:00:00",
        "metric": "RMSE",
        "method": "quantile_regression",
        
        # Main metrics
        "uncertainty_score": 0.85,
        "avg_coverage": 0.92,  # This will be used as coverage_score
        "calibration_error": 0.05,
        "avg_width": 0.18,     # This will be used as sharpness
        "consistency": 0.88,
        
        # Alpha levels
        "alpha_levels": [0.05, 0.1, 0.2],
        
        # Mock data structure for CRQR
        "crqr": {
            "by_alpha": {
                "0.05": {
                    "overall_result": {
                        "coverage": 0.94,
                        "expected_coverage": 0.95,
                        "mean_width": 0.25
                    }
                },
                "0.1": {
                    "overall_result": {
                        "coverage": 0.91,
                        "expected_coverage": 0.9,
                        "mean_width": 0.18
                    }
                },
                "0.2": {
                    "overall_result": {
                        "coverage": 0.78,
                        "expected_coverage": 0.8,
                        "mean_width": 0.12
                    }
                }
            }
        }
    }
    
    try:
        # Import the ReportManager directly
        ReportManager = import_report_manager()
        
        # Create report manager instance
        report_manager = ReportManager()
        
        # Generate the report
        report_path = report_manager.generate_uncertainty_report(
            results, 
            output_path, 
            "Test Regression Model"
        )
        
        print(f"\nUncertainty report generated at: {report_path}")
        return report_path
    
    except Exception as e:
        print(f"Error generating uncertainty report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Generate the test report
    report_path = generate_test_uncertainty_report()
    
    if report_path:
        # Print instructions
        print("\nTo view the report, open the file in a web browser:")
        print(f"  file://{os.path.abspath(report_path)}")