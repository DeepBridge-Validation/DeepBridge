"""
Example of generating an HTML report directly from a results dictionary.
This demonstrates how to use the utility function to convert test results into HTML.
"""

import os
import sys
from pathlib import Path

# Import the necessary function - use relative import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.report_from_results import generate_report_from_results

# Add support for specialized report types
def generate_specialized_report(results_dict, report_type, output_path):
    """Generate a specialized report (robustness or uncertainty) from results dictionary."""
    if report_type == 'robustness':
        from reporting.plots.robustness.robustness_report_generator import generate_robustness_report
        robustness_data = results_dict.get('robustness', {})
        return generate_robustness_report(
            robustness_data,
            output_path,
            model_name="Primary Model",
            experiment_info=results_dict
        )
    elif report_type == 'uncertainty':
        from reporting.plots.uncertainty.uncertainty_report_generator import generate_uncertainty_report
        uncertainty_data = results_dict.get('uncertainty', {})
        return generate_uncertainty_report(
            uncertainty_data,
            output_path,
            model_name="Primary Model",
            experiment_info=results_dict
        )
    else:
        return f"Unsupported report type: {report_type}"

def main():
    """Demonstrate how to generate an HTML report from a results dictionary."""
    print("DeepBridge - Simple Report Generation Example")
    print("============================================")
    
    # Create output directory
    os.makedirs("reports", exist_ok=True)
    
    # Create a sample results dictionary (simulating experiment.run_tests('quick') output)
    results_dict = {
        'experiment_type': 'binary_classification',
        'tests_performed': ['robustness', 'uncertainty'],
        'dataset_size': 1000,
        
        # Include initial_results with model metrics
        'initial_results': {
            'models': {
                'primary_model': {
                    'name': 'primary_model',
                    'type': 'RandomForestClassifier',
                    'metrics': {
                        'accuracy': 0.982,
                        'f1': 0.9819959474671671,
                        'precision': 0.982151158739503,
                        'recall': 0.982,
                        'roc_auc': 0.9951927693489533
                    },
                    'hyperparameters': {
                        'n_estimators': 100,
                        'max_depth': 10
                    }
                },
                'DECISION_TREE': {
                    'name': 'DECISION_TREE',
                    'type': 'DecisionTreeClassifier',
                    'metrics': {
                        'accuracy': 0.935,
                        'f1': 0.9349988946805626,
                        'precision': 0.9349996598898043,
                        'recall': 0.935,
                        'roc_auc': 0.9647254697202485
                    }
                },
                'LOGISTIC_REGRESSION': {
                    'name': 'LOGISTIC_REGRESSION',
                    'type': 'LogisticRegression',
                    'metrics': {
                        'accuracy': 0.892,
                        'f1': 0.891998163998164,
                        'precision': 0.892512069910651,
                        'recall': 0.892,
                        'roc_auc': 0.950924436655784
                    }
                },
                'GBM': {
                    'name': 'GBM',
                    'type': 'GradientBoostingClassifier',
                    'metrics': {
                        'accuracy': 0.9375,
                        'f1': 0.9374750876950666,
                        'precision': 0.9377924782923626,
                        'recall': 0.9375,
                        'roc_auc': 0.9727560255425389
                    }
                }
            }
        },
        
        # Robustness test results with nested primary_model structure (matches actual output format)
        'robustness': {
            'primary_model': {
                'base_score': 0.85,
                'avg_overall_impact': 0.12,
                'avg_raw_impact': 0.15,
                'avg_quantile_impact': 0.09,
                'feature_importance': {
                    'feature_1': 0.35,
                    'feature_2': 0.25,
                    'feature_3': 0.20,
                    'feature_4': 0.15,
                    'feature_5': 0.05
                },
                'raw': {
                    'by_level': {
                        '0.1': {'overall_result': {'mean_score': 0.83, 'std_score': 0.02}},
                        '0.3': {'overall_result': {'mean_score': 0.79, 'std_score': 0.03}},
                        '0.5': {'overall_result': {'mean_score': 0.74, 'std_score': 0.04}}
                    }
                },
                'quantile': {
                    'by_level': {
                        '0.1': {'overall_result': {'mean_score': 0.84, 'std_score': 0.01}},
                        '0.3': {'overall_result': {'mean_score': 0.81, 'std_score': 0.02}},
                        '0.5': {'overall_result': {'mean_score': 0.78, 'std_score': 0.03}}
                    }
                }
            }
        },
        
        # Uncertainty test results with nested primary_model structure
        'uncertainty': {
            'primary_model': {
                'expected_calibration_error': 0.08,
                'brier_score': 0.15,
                'coverage_stats': {
                    '0.1': {'coverage': 0.92, 'avg_width': 0.45},
                    '0.2': {'coverage': 0.82, 'avg_width': 0.30}
                },
                'calibration_metrics': {
                    'max_calibration_error': 0.12,
                    'mean_calibration_error': 0.07
                }
            }
        }
    }
    
    # Generate the HTML report
    print("\nGenerating HTML report from results dictionary...")
    report_path = generate_report_from_results(
        results_dict,
        "reports/simple_report.html",
        "Simple Test Results Report"
    )
    print(f"Report saved to: {report_path}")
    
    # Generate specialized robustness report
    try:
        print("\nGenerating specialized robustness report...")
        robustness_path = generate_report_from_results(
            results_dict,
            "reports/robustness_report.html",
            "Robustness Test Results Report",
            report_type="robustness"
        )
        print(f"Robustness report saved to: {robustness_path}")
    except Exception as e:
        print(f"Error generating robustness report: {e}")
        
    # Generate specialized uncertainty report
    try:
        print("\nGenerating specialized uncertainty report...")
        uncertainty_path = generate_report_from_results(
            results_dict,
            "reports/uncertainty_report.html",
            "Uncertainty Test Results Report",
            report_type="uncertainty"
        )
        print(f"Uncertainty report saved to: {uncertainty_path}")
    except Exception as e:
        print(f"Error generating uncertainty report: {e}")
    
    print("\nExample completed. Please check the generated HTML files in the 'reports' directory.")

if __name__ == "__main__":
    main()