"""
Test script for generating a resilience report with the new template structure.
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


def generate_test_resilience_report():
    """Generate a test resilience report using placeholder data."""
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file path
    output_path = os.path.join(output_dir, "test_resilience_report.html")
    
    # Create dummy data for the resilience report
    results = {
        "model_name": "Test Classification Model",
        "model_type": "RandomForestClassifier",
        "timestamp": "2025-04-21 10:00:00",
        "metric": "AUC",
        
        # Main metrics
        "resilience_score": 0.82,
        "avg_performance_gap": 0.12,
        "base_score": 0.91,
        "dataset_info": {
            "baseline_name": "Original Dataset",
            "target_name": "Shifted Dataset"
        },
        
        # Detailed metrics
        "metrics": {
            "Accuracy": 0.88,
            "F1 Score": 0.85,
            "AUC": 0.91,
            "Precision": 0.83,
            "Recall": 0.87
        },
        
        # Distribution shift data
        "distribution_shift": {
            "by_distance_metric": {
                "PSI": {
                    "avg_feature_distances": {
                        "feature_0": 0.15,
                        "feature_1": 0.22,
                        "feature_2": 0.08,
                        "feature_3": 0.12,
                        "feature_4": 0.05
                    },
                    "top_features": {
                        "feature_1": 0.22,
                        "feature_0": 0.15,
                        "feature_3": 0.12,
                        "feature_2": 0.08,
                        "feature_4": 0.05
                    }
                },
                "KS": {
                    "avg_feature_distances": {
                        "feature_0": 0.18,
                        "feature_1": 0.25,
                        "feature_2": 0.1,
                        "feature_3": 0.15,
                        "feature_4": 0.08
                    }
                }
            },
            "all_results": [
                {
                    "name": "Low PSI Shift",
                    "alpha": 0.1,
                    "distance_metric": "PSI",
                    "performance_gap": 0.08,
                    "worst_metric": 0.15,
                    "remaining_metric": 0.12,
                    "metric": "AUC",
                    "baseline_performance": 0.91,
                    "target_performance": 0.83,
                    "features": ["feature_1", "feature_0", "feature_3"],
                    "metrics": {
                        "Accuracy": {"baseline": 0.88, "target": 0.81, "gap": 0.07},
                        "F1 Score": {"baseline": 0.85, "target": 0.77, "gap": 0.08},
                        "AUC": {"baseline": 0.91, "target": 0.83, "gap": 0.08}
                    }
                },
                {
                    "name": "Medium PSI Shift",
                    "alpha": 0.2,
                    "distance_metric": "PSI",
                    "performance_gap": 0.12,
                    "worst_metric": 0.22,
                    "remaining_metric": 0.18,
                    "metric": "AUC",
                    "baseline_performance": 0.91,
                    "target_performance": 0.79,
                    "features": ["feature_1", "feature_0", "feature_3"],
                    "metrics": {
                        "Accuracy": {"baseline": 0.88, "target": 0.77, "gap": 0.11},
                        "F1 Score": {"baseline": 0.85, "target": 0.74, "gap": 0.11},
                        "AUC": {"baseline": 0.91, "target": 0.79, "gap": 0.12}
                    }
                },
                {
                    "name": "Low KS Shift",
                    "alpha": 0.1,
                    "distance_metric": "KS",
                    "performance_gap": 0.09,
                    "worst_metric": 0.18,
                    "remaining_metric": 0.15,
                    "metric": "AUC",
                    "baseline_performance": 0.91,
                    "target_performance": 0.82,
                    "features": ["feature_1", "feature_0", "feature_3"],
                    "metrics": {
                        "Accuracy": {"baseline": 0.88, "target": 0.80, "gap": 0.08},
                        "F1 Score": {"baseline": 0.85, "target": 0.78, "gap": 0.07},
                        "AUC": {"baseline": 0.91, "target": 0.82, "gap": 0.09}
                    }
                },
                {
                    "name": "Medium KS Shift",
                    "alpha": 0.2,
                    "distance_metric": "KS",
                    "performance_gap": 0.15,
                    "worst_metric": 0.25,
                    "remaining_metric": 0.21,
                    "metric": "AUC",
                    "baseline_performance": 0.91,
                    "target_performance": 0.76,
                    "features": ["feature_1", "feature_0", "feature_3"],
                    "metrics": {
                        "Accuracy": {"baseline": 0.88, "target": 0.75, "gap": 0.13},
                        "F1 Score": {"baseline": 0.85, "target": 0.72, "gap": 0.13},
                        "AUC": {"baseline": 0.91, "target": 0.76, "gap": 0.15}
                    }
                },
                {
                    "name": "High PSI Shift",
                    "alpha": 0.3,
                    "distance_metric": "PSI",
                    "performance_gap": 0.22,
                    "worst_metric": 0.35,
                    "remaining_metric": 0.28,
                    "metric": "AUC",
                    "baseline_performance": 0.91,
                    "target_performance": 0.69,
                    "features": ["feature_1", "feature_0", "feature_3", "feature_4"],
                    "metrics": {
                        "Accuracy": {"baseline": 0.88, "target": 0.68, "gap": 0.20},
                        "F1 Score": {"baseline": 0.85, "target": 0.66, "gap": 0.19},
                        "AUC": {"baseline": 0.91, "target": 0.69, "gap": 0.22}
                    }
                },
                {
                    "name": "Feature 2 Shift",
                    "alpha": 0.15,
                    "distance_metric": "PSI",
                    "performance_gap": 0.10,
                    "worst_metric": 0.20,
                    "remaining_metric": 0.16,
                    "metric": "AUC",
                    "baseline_performance": 0.91,
                    "target_performance": 0.81,
                    "features": ["feature_2", "feature_4", "feature_1"],
                    "metrics": {
                        "Accuracy": {"baseline": 0.88, "target": 0.79, "gap": 0.09},
                        "F1 Score": {"baseline": 0.85, "target": 0.76, "gap": 0.09},
                        "AUC": {"baseline": 0.91, "target": 0.81, "gap": 0.10}
                    }
                },
                {
                    "name": "Combined Feature Shift",
                    "alpha": 0.25,
                    "distance_metric": "KS",
                    "performance_gap": 0.18,
                    "worst_metric": 0.30,
                    "remaining_metric": 0.25,
                    "metric": "AUC",
                    "baseline_performance": 0.91,
                    "target_performance": 0.73,
                    "features": ["feature_1", "feature_0", "feature_2", "feature_3", "feature_4"],
                    "metrics": {
                        "Accuracy": {"baseline": 0.88, "target": 0.71, "gap": 0.17},
                        "F1 Score": {"baseline": 0.85, "target": 0.70, "gap": 0.15},
                        "AUC": {"baseline": 0.91, "target": 0.73, "gap": 0.18}
                    }
                }
            ]
        },
        "distance_metrics": ["PSI", "KS"],
        "alphas": [0.1, 0.2],
        
        # Add sample data for detailed analysis
        "samples": [
            {
                "id": "sample_001",
                "baseline_prediction": 0.82,
                "target_prediction": 0.43,
                "prediction_change": -0.39,
                "key_features": ["feature_1", "feature_3"]
            },
            {
                "id": "sample_002",
                "baseline_prediction": 0.75,
                "target_prediction": 0.51,
                "prediction_change": -0.24,
                "key_features": ["feature_0", "feature_2"]
            },
            {
                "id": "sample_003",
                "baseline_prediction": 0.91,
                "target_prediction": 0.62,
                "prediction_change": -0.29,
                "key_features": ["feature_1", "feature_4"]
            }
        ],
        
        # Include outlier sensitivity for the metrics card
        "outlier_sensitivity": 0.18
    }
    
    try:
        # Import the ReportManager directly
        ReportManager = import_report_manager()
        
        # Create report manager instance
        report_manager = ReportManager()
        
        # Generate the report
        report_path = report_manager.generate_resilience_report(
            results, 
            output_path, 
            "Test Classification Model"
        )
        
        print(f"\nResilience report generated at: {report_path}")
        return report_path
    
    except Exception as e:
        print(f"Error generating resilience report: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Generate the test report
    report_path = generate_test_resilience_report()
    
    if report_path:
        # Print instructions
        print("\nTo view the report, open the file in a web browser:")
        print(f"  file://{os.path.abspath(report_path)}")