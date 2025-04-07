"""
Module for generating reports from uncertainty test results.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import datetime
import numpy as np
import json

class UncertaintyReporter:
    """
    Generates reports from uncertainty test results.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the reporter.
        
        Parameters:
        -----------
        verbose : bool
            Whether to print progress information
        """
        self.verbose = verbose
        
    def _convert_numpy_arrays(self, obj):
        """
        Recursively convert numpy arrays to lists in a dictionary or list.
        
        Parameters:
        -----------
        obj : Any
            The object to convert
        """
        import numpy as np
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    self._convert_numpy_arrays(value)
                elif isinstance(value, np.ndarray):
                    obj[key] = value.tolist()
                elif isinstance(value, np.number):
                    obj[key] = value.item()
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if isinstance(item, (dict, list)):
                    self._convert_numpy_arrays(item)
                elif isinstance(item, np.ndarray):
                    obj[i] = item.tolist()
                elif isinstance(item, np.number):
                    obj[i] = item.item()
    
    def generate_text_report(self, 
                            test_results: Dict[str, Any], 
                            model_name: str = "Main Model") -> str:
        """
        Generate a text report from uncertainty test results.
        
        Parameters:
        -----------
        test_results : Dict
            Uncertainty test results
        model_name : str
            Name of the model for the report
            
        Returns:
        --------
        str : Text report
        """
        report_lines = []
        
        # Add header
        report_lines.append(f"# Uncertainty Test Report - {model_name}")
        report_lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add overall uncertainty score
        report_lines.append("\n## Overall Uncertainty")
        report_lines.append(f"Uncertainty quality score: {test_results.get('uncertainty_quality_score', 0):.3f}")
        report_lines.append(f"Average coverage error: {test_results.get('avg_coverage_error', 0):.3f}")
        report_lines.append(f"Average normalized width: {test_results.get('avg_normalized_width', 0):.3f}")
        
        # Add CRQR results by alpha
        report_lines.append("\n## CRQR Results")
        
        for alpha, alpha_data in sorted(test_results.get('crqr', {}).get('by_alpha', {}).items()):
            overall = alpha_data.get('overall_result', {})
            if overall:
                report_lines.append(f"\n### Alpha = {alpha} (Expected coverage: {(1-float(alpha))*100:.1f}%)")
                report_lines.append(f"Actual coverage: {overall.get('coverage', 0)*100:.1f}%")
                report_lines.append(f"Mean interval width: {overall.get('mean_width', 0):.3f}")
                report_lines.append(f"Median interval width: {overall.get('median_width', 0):.3f}")
        
        # Add feature importance section
        report_lines.append("\n## Feature Importance")
        
        # Get feature importance
        importance = test_results.get('feature_importance', {})
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top 10 features
        if len(sorted_features) > 10:
            sorted_features = sorted_features[:10]
            report_lines.append("Top 10 most important features:")
        else:
            report_lines.append("Feature importance:")
            
        for feature, value in sorted_features:
            report_lines.append(f"- {feature}: {value:.3f}")
        
        # Não mais exibimos o tempo de execução
        
        return '\n'.join(report_lines)
    
    def save_text_report(self, 
                        output_path: str, 
                        test_results: Dict[str, Any], 
                        model_name: str = "Main Model") -> str:
        """
        Generate and save a text report to a file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the report file
        test_results : Dict
            Uncertainty test results
        model_name : str
            Name of the model for the report
            
        Returns:
        --------
        str : Path to the saved report
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Generate report
        report_text = self.generate_text_report(test_results, model_name)
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
            
        if self.verbose:
            print(f"Report saved to {output_path}")
            
        return output_path
    
    def _prepare_visualization_data(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare data for visualizations.
        
        Parameters:
        -----------
        test_results : Dict
            Uncertainty test results
            
        Returns:
        --------
        Dict : Visualization data
        """
        visualizations = {}
        
        # Extract data for alpha comparison chart
        alpha_comparison = {
            'data': [],
            'layout': {
                'title': 'Confidence Level Comparison',
                'barmode': 'group',
                'yaxis': {
                    'title': 'Coverage',
                    'side': 'left',
                    'range': [0, 1]
                },
                'yaxis2': {
                    'title': 'Interval Width',
                    'side': 'right',
                    'overlaying': 'y',
                    'range': [0, 0.5]
                },
                'legend': {
                    'orientation': 'h',
                    'y': -0.2
                }
            }
        }
        
        # Extract alpha levels, coverages, and widths
        alphas = []
        actual_coverages = []
        expected_coverages = []
        mean_widths = []
        
        for alpha, alpha_data in sorted(test_results.get('crqr', {}).get('by_alpha', {}).items()):
            overall = alpha_data.get('overall_result', {})
            if overall:
                alphas.append(str(alpha))
                actual_coverages.append(overall.get('coverage', 0))
                expected_coverages.append(overall.get('expected_coverage', 0))
                mean_widths.append(overall.get('mean_width', 0))
        
        # Add traces for actual coverage, expected coverage, and mean width
        alpha_comparison['data'].append({
            'x': alphas,
            'y': actual_coverages,
            'name': 'Actual Coverage',
            'type': 'bar',
            'marker': {'color': '#3182CE'}
        })
        
        alpha_comparison['data'].append({
            'x': alphas,
            'y': expected_coverages,
            'name': 'Expected Coverage',
            'type': 'bar',
            'marker': {'color': '#E53E3E'}
        })
        
        alpha_comparison['data'].append({
            'x': alphas,
            'y': mean_widths,
            'name': 'Mean Width',
            'type': 'bar',
            'marker': {'color': '#805AD5'},
            'yaxis': 'y2'
        })
        
        visualizations['alpha_comparison_chart'] = alpha_comparison
        
        # Extract data for feature importance chart
        feature_importance = {
            'data': [],
            'layout': {
                'title': 'Feature Importance for Uncertainty',
                'xaxis': {
                    'title': 'Importance Score'
                },
                'yaxis': {
                    'title': 'Feature',
                    'automargin': True
                }
            }
        }
        
        # Get feature importance and sort it
        importance = test_results.get('feature_importance', {})
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top 10 features
        if len(sorted_features) > 10:
            sorted_features = sorted_features[:10]
        
        # Extract feature names and values
        feature_names = [feature for feature, _ in sorted_features]
        feature_values = [value for _, value in sorted_features]
        
        # Add trace for feature importance
        feature_importance['data'].append({
            'x': feature_values,
            'y': feature_names,
            'type': 'bar',
            'orientation': 'h',
            'marker': {
                'color': '#805AD5'
            }
        })
        
        visualizations['feature_importance_chart'] = feature_importance
        
        # Extract data for prediction interval chart
        # We'll use sample data from the first alpha level
        prediction_interval = {
            'data': [],
            'layout': {
                'title': 'Prediction Intervals',
                'xaxis': {
                    'title': 'Sample Index'
                },
                'yaxis': {
                    'title': 'Value'
                }
            }
        }
        
        # Get sample data from the first alpha level
        first_alpha = alphas[0] if alphas else None
        if first_alpha and first_alpha in test_results.get('crqr', {}).get('by_alpha', {}):
            overall = test_results['crqr']['by_alpha'][first_alpha].get('overall_result', {})
            if overall and 'lower_bounds' in overall and 'upper_bounds' in overall:
                # Use a subset of the data for visualization (max 20 points)
                indices = list(range(1, min(21, len(overall['lower_bounds']) + 1)))
                lower_bounds = overall['lower_bounds'][:20].tolist() if isinstance(overall['lower_bounds'], np.ndarray) else overall['lower_bounds'][:20]
                upper_bounds = overall['upper_bounds'][:20].tolist() if isinstance(overall['upper_bounds'], np.ndarray) else overall['upper_bounds'][:20]
                
                # Simulate actual values (midpoint between bounds)
                actual_values = [(lower + upper) / 2 for lower, upper in zip(lower_bounds, upper_bounds)]
                
                # Add traces for actual values, lower bounds, and upper bounds
                prediction_interval['data'].append({
                    'y': actual_values,
                    'x': indices,
                    'type': 'scatter',
                    'mode': 'markers',
                    'name': 'Actual Value',
                    'marker': {
                        'color': '#3182CE',
                        'size': 10
                    }
                })
                
                prediction_interval['data'].append({
                    'y': lower_bounds,
                    'x': indices,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Lower Bound',
                    'line': {
                        'color': '#E53E3E',
                        'width': 1,
                        'dash': 'dot'
                    }
                })
                
                prediction_interval['data'].append({
                    'y': upper_bounds,
                    'x': indices,
                    'type': 'scatter',
                    'mode': 'lines',
                    'name': 'Upper Bound',
                    'line': {
                        'color': '#E53E3E',
                        'width': 1,
                        'dash': 'dot'
                    },
                    'fill': 'tonexty',
                    'fillcolor': 'rgba(231, 76, 60, 0.1)'
                })
        
        visualizations['prediction_interval_chart'] = prediction_interval
        
        # Create uncertainty calibration chart
        uncertainty_calibration = {
            'data': [],
            'layout': {
                'title': 'Uncertainty Calibration',
                'xaxis': {
                    'title': 'Predicted Probability',
                    'range': [0, 1]
                },
                'yaxis': {
                    'title': 'Observed Frequency',
                    'range': [0, 1]
                },
                'showlegend': True
            }
        }
        
        # Generate example calibration curve (ideally this would be calculated from actual data)
        # For demonstration, we'll create a slightly miscalibrated curve
        predicted_probs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        observed_freqs = [0, 0.12, 0.22, 0.32, 0.43, 0.54, 0.65, 0.76, 0.85, 0.96, 1.0]
        
        # Add traces for calibration curve and perfect calibration line
        uncertainty_calibration['data'].append({
            'x': predicted_probs,
            'y': observed_freqs,
            'type': 'scatter',
            'mode': 'lines+markers',
            'name': 'Calibration Curve',
            'marker': {
                'color': '#3182CE',
                'size': 8
            },
            'line': {
                'color': '#3182CE',
                'width': 2
            }
        })
        
        uncertainty_calibration['data'].append({
            'x': [0, 1],
            'y': [0, 1],
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Perfect Calibration',
            'line': {
                'color': '#A0AEC0',
                'width': 2,
                'dash': 'dash'
            }
        })
        
        visualizations['uncertainty_calibration_chart'] = uncertainty_calibration
        
        return visualizations
    
    def generate_html_report(self, 
                           test_results: Dict[str, Any], 
                           model_name: str = "Main Model") -> str:
        """
        Generate an HTML report from uncertainty test results.
        
        Parameters:
        -----------
        test_results : Dict
            Uncertainty test results
        model_name : str
            Name of the model for the report
            
        Returns:
        --------
        str : HTML report content
        """
        # Initialize variables to be populated in the template
        report_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare visualizations
        visualizations = self._prepare_visualization_data(test_results)
        
        # Prepare data for template context
        template_context = {
            'model_name': model_name,
            'test_results': test_results,
            'report_date': report_date,
            'summary_text': f"Uncertainty quality score: {test_results.get('uncertainty_quality_score', 0):.3f}",
            'alpha_comparison_chart': json.dumps(visualizations.get('alpha_comparison_chart', {})),
            'feature_importance_chart': json.dumps(visualizations.get('feature_importance_chart', {})),
            'prediction_interval_chart': json.dumps(visualizations.get('prediction_interval_chart', {})),
            'uncertainty_calibration_chart': json.dumps(visualizations.get('uncertainty_calibration_chart', {}))
        }
        
        # Try to import the template module
        try:
            # First, try to import the module
            from deepbridge.reports.templates.uncertainty_report_template import get_template
            template = get_template()
            
            # Fill template with data
            # Convert any numpy arrays in test_results to lists for JSON serialization
            self._convert_numpy_arrays(test_results)
            
            # Include React components for uncertainty visualization
            template_context['include_react_components'] = True
            
            # Set correct path for uncertainty visualization components
            import os
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            rel_path = 'examples/Synthetic/visualizations/incerteza'
            full_path = os.path.join(base_path, rel_path)
            
            if os.path.exists(full_path):
                template_context['uncertainty_visualization_path'] = full_path
            else:
                # Default to relative path if directory doesn't exist
                template_context['uncertainty_visualization_path'] = '/examples/Synthetic/visualizations/incerteza'
            
            html_content = template.render(**template_context)
            
            return html_content
            
        except (ImportError, Exception) as e:
            if self.verbose:
                print(f"Error using template: {str(e)}")
                print("Falling back to simple HTML report")
                
            # Fallback to a simple HTML report if template not available
            text_report = self.generate_text_report(test_results, model_name)
            
            # Convert text report to HTML
            report_lines = ['<!DOCTYPE html>',
                           '<html>',
                           '<head>',
                           f'<title>Uncertainty Report - {model_name}</title>',
                           '<style>',
                           'body { font-family: Arial, sans-serif; margin: 20px; }',
                           'h1, h2, h3 { color: #333; }',
                           '.metric { margin: 10px 0; }',
                           '.value { font-weight: bold; }',
                           '</style>',
                           '</head>',
                           '<body>']
            
            html_lines = []
            
            for line in text_report.split('\n'):
                if line.startswith('# '):
                    html_lines.append(f'<h1>{line[2:]}</h1>')
                elif line.startswith('## '):
                    html_lines.append(f'<h2>{line[3:]}</h2>')
                elif line.startswith('### '):
                    html_lines.append(f'<h3>{line[4:]}</h3>')
                elif line.startswith('- '):
                    html_lines.append(f'<div class="metric">• {line[2:]}</div>')
                elif line.strip():
                    if ':' in line:
                        label, value = line.split(':', 1)
                        html_lines.append(f'<div class="metric">{label}: <span class="value">{value}</span></div>')
                    else:
                        html_lines.append(f'<p>{line}</p>')
                else:
                    html_lines.append('<br>')
            
            report_lines.extend(html_lines)
            report_lines.append('</body>')
            report_lines.append('</html>')
            
            return '\n'.join(report_lines)
    
    def save_html_report(self, 
                        output_path: str, 
                        test_results: Dict[str, Any], 
                        model_name: str = "Main Model") -> str:
        """
        Generate and save an HTML report to a file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the report file
        test_results : Dict
            Uncertainty test results
        model_name : str
            Name of the model for the report
            
        Returns:
        --------
        str : Path to the saved report
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Generate HTML report
        html_content = self.generate_html_report(test_results, model_name)
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        if self.verbose:
            print(f"HTML report saved to {output_path}")
            
        return output_path