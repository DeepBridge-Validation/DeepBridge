"""
Module for generating reports from robustness test results.
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Union, Any
import datetime

class RobustnessReporter:
    """
    Generates reports from robustness test results.
    Extracted from RobustnessSuite to separate reporting responsibilities.
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
    
    def generate_text_report(self, 
                            test_results: Dict[str, Any], 
                            model_name: str = "Main Model") -> str:
        """
        Generate a text report from robustness test results.
        
        Parameters:
        -----------
        test_results : Dict
            Robustness test results
        model_name : str
            Name of the model for the report
            
        Returns:
        --------
        str : Text report
        """
        report_lines = []
        
        # Add header
        report_lines.append(f"# Robustness Test Report - {model_name}")
        report_lines.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Add overall robustness score
        report_lines.append("\n## Overall Robustness")
        report_lines.append(f"Average impact: {test_results.get('avg_overall_impact', 0):.3f}")
        report_lines.append(f"Baseline score: {test_results.get('base_score', 0):.3f}")
        
        # Add Raw perturbation results
        report_lines.append("\n### Gaussian Noise Perturbation")
        report_lines.append(f"Average impact: {test_results.get('avg_raw_impact', 0):.3f}")
        
        # Add results by level
        for level, level_data in sorted(test_results.get('raw', {}).get('by_level', {}).items()):
            overall = level_data.get('overall_result', {})
            if overall:
                report_lines.append(f"- Level: {level}, Mean Score: {overall.get('mean_score', 0):.3f}, Std: {overall.get('std_score', 0):.3f}")
        
        # Add Quantile perturbation results
        report_lines.append("\n### Quantile Perturbation")
        report_lines.append(f"Average impact: {test_results.get('avg_quantile_impact', 0):.3f}")
        
        # Add results by level
        for level, level_data in sorted(test_results.get('quantile', {}).get('by_level', {}).items()):
            overall = level_data.get('overall_result', {})
            if overall:
                report_lines.append(f"- Level: {level}, Mean Score: {overall.get('mean_score', 0):.3f}, Std: {overall.get('std_score', 0):.3f}")
        
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
            Robustness test results
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
    
    def generate_html_report(self, 
                           test_results: Dict[str, Any], 
                           visualizations: Dict[str, Any],
                           model_name: str = "Main Model") -> str:
        """
        Generate an HTML report from robustness test results.
        
        Parameters:
        -----------
        test_results : Dict
            Robustness test results
        visualizations : Dict
            Dictionary of generated visualizations
        model_name : str
            Name of the model for the report
            
        Returns:
        --------
        str : HTML report content
        """
        # Initialize variables to be populated in the template
        report_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare data for template context
        # Set defaults for required template variables
        template_context = {
            'model_name': model_name,
            'test_results': test_results,
            'visualizations': visualizations,
            'generation_time': report_date,
            'report_date': report_date,
            'models_evaluated': model_name,
            'key_finding': f"Model shows {test_results.get('avg_overall_impact', 0):.3f} impact under perturbation",
            'summary_text': f"Baseline performance: {test_results.get('base_score', 0):.3f}",
            'best_model': model_name,
            'model_metrics': [
                {'name': model_name, 
                 'robustness_index': f"{(1 - test_results.get('avg_overall_impact', 0)):.3f}",
                 'color': '#3a6ea5',
                 'baseline': f"{test_results.get('base_score', 0):.3f}",
                 'perturbed': '0.000', 
                 'drop': f"{test_results.get('avg_overall_impact', 0)*100:.1f}"}
            ],
            'perturbation_levels': [],
            'boxplot_chart': '{"data":[], "layout":{}}',
            'feature_importance_chart': '{"data":[], "layout":{}}',
            'recommendations': [
                {'title': 'Improve Robustness Training',
                 'content': 'Consider adding noise to training data or using adversarial training techniques.'}
            ]
        }
        
        # Extract perturbation levels used
        if 'raw' in test_results and 'by_level' in test_results['raw']:
            levels = sorted(test_results['raw']['by_level'].keys())
            template_context['perturbation_levels'] = levels
            
            # Create detailed model results
            model_detailed_results = [
                {'name': model_name, 
                 'robustness_index': f"{(1 - test_results.get('avg_overall_impact', 0)):.3f}", 
                 'scores': []}
            ]
            
            # Add scores by level
            for level in levels:
                level_data = test_results['raw']['by_level'].get(level, {})
                overall = level_data.get('overall_result', {})
                if overall:
                    model_detailed_results[0]['scores'].append(f"{overall.get('mean_score', 0):.3f}")
                else:
                    model_detailed_results[0]['scores'].append("N/A")
                    
            template_context['model_detailed_results'] = model_detailed_results
        
        # Extract top features
        top_features = []
        if 'feature_importance' in test_results:
            # Sort features by importance
            sorted_features = sorted(test_results['feature_importance'].items(), 
                                    key=lambda x: x[1], reverse=True)
            
            # Get top 5 or less
            for i, (feature, value) in enumerate(sorted_features[:5]):
                # Calculate percentage for visualization
                percentage = min(100, max(5, value * 100))  # Ensure it's at least 5% for visibility
                top_features.append({
                    'name': feature,
                    'value': f"{value:.3f}",
                    'percentage': percentage
                })
                
            template_context['top_features'] = top_features
            
        # Try to import the template module
        try:
            # First, try to import the module
            from deepbridge.reports.templates.robustness_report_template import get_template
            template = get_template()
            
            # Fill template with data
            html_content = template.render(**template_context)
            
            return html_content
            
        except (ImportError, Exception) as e:
            if self.verbose:
                print(f"Error using template: {str(e)}")
                print("Falling back to simple HTML report")
                
            # Fallback to a simple HTML report if template not available
            report_lines = ['<!DOCTYPE html>',
                           '<html>',
                           '<head>',
                           f'<title>Robustness Report - {model_name}</title>',
                           '<style>',
                           'body { font-family: Arial, sans-serif; margin: 20px; }',
                           'h1, h2, h3 { color: #333; }',
                           '.metric { margin: 10px 0; }',
                           '.value { font-weight: bold; }',
                           '</style>',
                           '</head>',
                           '<body>']
            
            # Convert text report to HTML
            text_report = self.generate_text_report(test_results, model_name)
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
                        visualizations: Dict[str, Any],
                        model_name: str = "Main Model") -> str:
        """
        Generate and save an HTML report to a file.
        
        Parameters:
        -----------
        output_path : str
            Path to save the report file
        test_results : Dict
            Robustness test results
        visualizations : Dict
            Dictionary of generated visualizations
        model_name : str
            Name of the model for the report
            
        Returns:
        --------
        str : Path to the saved report
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Generate HTML report
        html_content = self.generate_html_report(test_results, visualizations, model_name)
        
        # Write report to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        if self.verbose:
            print(f"HTML report saved to {output_path}")
            
        return output_path