"""
Test script for the updated report templates.
This script directly accesses the report_manager module to avoid import issues.
"""

import os
import sys
import json
import importlib.util

# Manually import the report_manager module to avoid package import issues
def import_report_manager():
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                            "core/experiment/report_manager.py")
    module_name = "report_manager"
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    
    return module.ReportManager

def create_test_report(report_type, output_path):
    """Create a test report with dummy data."""
    # Create a dummy result data structure
    dummy_results = {
        "model_name": "Test Model",
        "model_type": "Test Type",
        "base_score": 0.85,
        "metric": "accuracy"
    }
    
    # Add specific data for each report type
    if report_type == "robustness":
        dummy_results.update({
            "robustness_score": 0.82,
            "raw_impact": 0.12,
            "quantile_impact": 0.08,
            "feature_importance": {
                "feature1": 0.8,
                "feature2": 0.6,
                "feature3": 0.4,
                "feature4": 0.2,
                "feature5": 0.1
            },
            "model_feature_importance": {
                "feature1": 0.7,
                "feature2": 0.5,
                "feature3": 0.3,
                "feature4": 0.1
            },
            "has_alternative_models": False,
            "raw": {
                "by_level": {
                    "0.1": {
                        "overall_result": {
                            "all_features": {
                                "mean_score": 0.82,
                                "worst_score": 0.75,
                                "impact": 0.03
                            }
                        }
                    },
                    "0.2": {
                        "overall_result": {
                            "all_features": {
                                "mean_score": 0.80,
                                "worst_score": 0.70,
                                "impact": 0.05
                            }
                        }
                    },
                    "0.3": {
                        "overall_result": {
                            "all_features": {
                                "mean_score": 0.78,
                                "worst_score": 0.65,
                                "impact": 0.07
                            }
                        }
                    }
                }
            }
        })
    elif report_type == "uncertainty":
        dummy_results.update({
            "uncertainty_score": 0.76,
            "coverage_score": 0.82,
            "calibration_error": 0.09,
            "sharpness": 0.15,
            "consistency": 0.88,
            "uncertainty_method": "CRQR",
            "alpha_levels": [0.1, 0.2, 0.3, 0.4, 0.5],
            "alpha_coverage": {
                "0.1": {
                    "coverage": 0.91,
                    "width": 0.25,
                    "expected_coverage": 0.9
                },
                "0.2": {
                    "coverage": 0.82,
                    "width": 0.20,
                    "expected_coverage": 0.8
                },
                "0.3": {
                    "coverage": 0.72,
                    "width": 0.17,
                    "expected_coverage": 0.7
                }
            },
            "uncertainty_distribution": {
                "bins": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "frequencies": [5, 8, 12, 20, 25, 18, 10, 8, 7, 3]
            },
            "calibration": {
                "confidence_bins": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                "observed_frequencies": [0.12, 0.22, 0.29, 0.38, 0.52, 0.61, 0.69, 0.78, 0.88, 0.95]
            }
        })
    elif report_type == "resilience":
        dummy_results.update({
            "resilience_score": 0.79,
            "avg_performance_gap": 0.15,
            "distance_metrics": ["PSI", "KS", "WD1"],
            "distribution_shift_results": [
                {"alpha": 0.1, "distance_metric": "PSI", "performance_gap": 0.05, "metric": "accuracy"},
                {"alpha": 0.2, "distance_metric": "PSI", "performance_gap": 0.1, "metric": "accuracy"},
                {"alpha": 0.3, "distance_metric": "PSI", "performance_gap": 0.15, "metric": "accuracy"}
            ],
            "dataset_info": {
                "baseline_name": "Training Dataset",
                "target_name": "Test Dataset with Distribution Shift"
            },
            "distribution_shift": {
                "all_results": [
                    {"alpha": 0.1, "distance_metric": "PSI", "performance_gap": 0.05, "metric": "accuracy"},
                    {"alpha": 0.2, "distance_metric": "PSI", "performance_gap": 0.1, "metric": "accuracy"},
                    {"alpha": 0.3, "distance_metric": "PSI", "performance_gap": 0.15, "metric": "accuracy"}
                ],
                "by_distance_metric": {
                    "PSI": {
                        "avg_feature_distances": {
                            "feature1": 0.15,
                            "feature2": 0.08,
                            "feature3": 0.05
                        },
                        "top_features": {
                            "feature1": 0.15,
                            "feature2": 0.08,
                            "feature3": 0.05
                        }
                    }
                }
            }
        })
    
    # Initialize report manager
    ReportManager = import_report_manager()
    report_manager = ReportManager()
    
    # Generate report
    try:
        output_file = report_manager.generate_report(report_type, dummy_results, output_path, "Test Model")
        print(f"Successfully generated {report_type} report: {output_file}")
        return output_file
    except Exception as e:
        print(f"Error generating {report_type} report: {str(e)}")
        return None

if __name__ == "__main__":
    # Create reports directory if it doesn't exist
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Test robustness report
    robustness_report = create_test_report(
        "robustness", 
        os.path.join(reports_dir, "test_robustness_report.html")
    )
    
    # Test uncertainty report
    uncertainty_report = create_test_report(
        "uncertainty", 
        os.path.join(reports_dir, "test_uncertainty_report.html")
    )
    
    # Test resilience report
    resilience_report = create_test_report(
        "resilience", 
        os.path.join(reports_dir, "test_resilience_report.html")
    )
    
    print("\nTest complete!")
    print(f"Robustness report: {robustness_report}")
    print(f"Uncertainty report: {uncertainty_report}")
    print(f"Resilience report: {resilience_report}")