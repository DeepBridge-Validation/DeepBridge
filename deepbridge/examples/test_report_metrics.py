"""
Test script to verify report metrics are properly displayed.
This is a standalone script that creates a report using the real metrics.
"""

import os
import sys
import json
import datetime
from pathlib import Path

# Define the function to generate HTML directly in this file
def generate_test_report(results, output_path, experiment_name="Test Report"):
    """Generate an HTML report from test results with real metrics."""
    # Ensure directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build report HTML
    html = f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{experiment_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .header {{ background: linear-gradient(135deg, #0062cc 0%, #1e88e5 100%); color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; }}
            .section {{ margin-bottom: 30px; padding: 20px; background: #fff; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{experiment_name}</h1>
                <p>Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
    """
    
    # Add model metrics section if initial_results is available
    if 'initial_results' in results and 'models' in results['initial_results']:
        models_data = results['initial_results']['models']
        
        html += """
        <div class="section">
            <h2>Model Performance Metrics</h2>
            <div class="test-result">
        """
        
        # Primary model metrics
        if 'primary_model' in models_data:
            primary_model = models_data['primary_model']
            primary_metrics = primary_model.get('metrics', {})
            
            html += '<h3>Primary Model Metrics</h3>'
            html += '<table><tr><th>Metric</th><th>Value</th></tr>'
            
            for metric, value in primary_metrics.items():
                # Handle numpy float types
                try:
                    value_str = f"{float(value):.4f}"
                except (TypeError, ValueError):
                    value_str = str(value)
                
                html += f'<tr><td>{metric.replace("_", " ").title()}</td><td>{value_str}</td></tr>'
            
            html += '</table>'
        
        # Alternative models metrics
        alt_models = {k: v for k, v in models_data.items() if k != 'primary_model'}
        if alt_models:
            html += '<h3>Alternative Models Metrics</h3>'
            html += '<table><tr><th>Model</th><th>Accuracy</th><th>F1</th><th>Precision</th><th>Recall</th><th>ROC AUC</th></tr>'
            
            for model_name, model_data in alt_models.items():
                metrics = model_data.get('metrics', {})
                html += f'<tr><td>{model_name}</td>'
                
                # Add each metric for the model
                for metric in ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
                    value = metrics.get(metric, metrics.get('auc', '-'))
                    if value != '-':
                        try:
                            value_str = f"{float(value):.4f}"
                        except (TypeError, ValueError):
                            value_str = str(value)
                    else:
                        value_str = '-'
                    html += f'<td>{value_str}</td>'
                
                html += '</tr>'
            
            html += '</table>'
        
        html += """
            </div>
        </div>
        """
    
    # Close the HTML
    html += """
        </div>
    </body>
    </html>
    """
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return str(output_path)

def main():
    """Create a sample report with real metrics."""
    print("DeepBridge - Test Report Metrics Example")
    print("========================================")
    
    # Create output directory
    os.makedirs("reports", exist_ok=True)
    
    # Create a sample results dictionary with metrics
    results_dict = {
        'experiment_type': 'binary_classification',
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
        }
    }
    
    # Generate a test report
    report_path = generate_test_report(
        results_dict,
        "reports/test_metrics_report.html",
        "Test Metrics Report"
    )
    
    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    main()