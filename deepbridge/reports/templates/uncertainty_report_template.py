"""
Template provider for uncertainty report HTML.
This module provides the template for the uncertainty report.
"""

import os
import jinja2
from pathlib import Path
import json
import datetime

def get_template():
    """
    Get the uncertainty report template.
    
    Returns:
        jinja2.Template: A Jinja2 template object ready for rendering
    """
    # Get the directory containing this file
    template_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up Jinja2 environment
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )
    
    # Add custom filters to help with JSON data
    env.filters['json'] = lambda obj: json.dumps(obj)
    env.filters['string'] = lambda obj: str(obj)
    
    # Load the template
    try:
        template = env.get_template('uncertainty_report_template.html')
        return template
    except jinja2.exceptions.TemplateNotFound:
        # Return a simple template instead
        return SimpleTemplate()


class SimpleTemplate:
    """
    A simple template class that renders basic HTML when the template file is not found.
    This provides basic compatibility with the Jinja2 Template class.
    """
    
    def __init__(self):
        """Initialize the simple template."""
        self.name = "SimpleTemplate (Fallback)"
    
    def render(self, **kwargs):
        """
        Render a basic HTML template with the given context.
        
        Args:
            **kwargs: Template context variables
            
        Returns:
            str: Rendered HTML
        """
        model_name = kwargs.get('model_name', 'Main Model')
        test_results = kwargs.get('test_results', {})
        visualizations = kwargs.get('visualizations', {})
        generation_time = kwargs.get('generation_time', datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        report_lines = [
            '<!DOCTYPE html>',
            '<html>',
            '<head>',
            f'<title>Uncertainty Report - {model_name}</title>',
            '<style>',
            'body { font-family: Arial, sans-serif; margin: 20px; }',
            'h1, h2, h3 { color: #333; }',
            '.metric { margin: 10px 0; }',
            '.value { font-weight: bold; }',
            '.container { max-width: 1200px; margin: 0 auto; padding: 20px; }',
            '.section { margin-bottom: 30px; padding: 15px; border: 1px solid #eee; border-radius: 5px; }',
            '</style>',
            '</head>',
            '<body>',
            '<div class="container">',
            f'<h1>Uncertainty Test Report - {model_name}</h1>',
            f'<div class="metric">Generated on: <span class="value"> {generation_time}</span></div>',
            '<div class="section">',
            '<h2>Overall Uncertainty Quality</h2>',
            f'<div class="metric">Uncertainty quality score: <span class="value"> {test_results.get("uncertainty_quality_score", 0):.3f}</span></div>',
            f'<div class="metric">Average coverage error: <span class="value"> {test_results.get("avg_coverage_error", 0):.3f}</span></div>',
            f'<div class="metric">Average normalized width: <span class="value"> {test_results.get("avg_normalized_width", 0):.3f}</span></div>',
            '</div>',
        ]
        
        # CRQR Section
        report_lines.extend([
            '<div class="section">',
            '<h3>CRQR Results</h3>'
        ])
        
        # Add CRQR results by alpha
        for alpha, alpha_data in sorted(test_results.get('crqr', {}).get('by_alpha', {}).items()):
            overall = alpha_data.get('overall_result', {})
            if overall:
                report_lines.append(
                    f'<div class="section">'
                    f'<h4>Alpha = {alpha} (Expected coverage: {(1-float(alpha))*100:.1f}%)</h4>'
                    f'<div class="metric">Actual coverage: <span class="value"> {overall.get("coverage", 0)*100:.1f}%</span></div>'
                    f'<div class="metric">Mean interval width: <span class="value"> {overall.get("mean_width", 0):.3f}</span></div>'
                    f'<div class="metric">Median interval width: <span class="value"> {overall.get("median_width", 0):.3f}</span></div>'
                    f'</div>'
                )
        
        report_lines.append('</div>')  # Close CRQR section
        
        # Feature Importance Section
        report_lines.extend([
            '<div class="section">',
            '<h2>Feature Importance</h2>'
        ])
        
        # Get feature importance
        importance = test_results.get('feature_importance', {})
        
        # Sort features by importance
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        # Limit to top 10 features
        if sorted_features:
            if len(sorted_features) > 10:
                sorted_features = sorted_features[:10]
                report_lines.append('<div class="metric">Top 10 most important features:</div>')
            else:
                report_lines.append('<div class="metric">Feature importance:</div>')
                
            for feature, value in sorted_features:
                report_lines.append(f'<div class="metric">• {feature}: {value:.3f}</div>')
        else:
            report_lines.append('<div class="metric">No feature importance data available</div>')
        
        report_lines.append('</div>')  # Close feature importance section
        
        # Add execution time
        if 'execution_time' in test_results:
            report_lines.append(f'<div class="metric">Execution time: <span class="value"> {test_results["execution_time"]:.2f} seconds</span></div>')
        
        # Close HTML
        report_lines.extend([
            '</div>', # Close container
            '</body>',
            '</html>'
        ])
        
        return '\n'.join(report_lines)


if __name__ == "__main__":
    # Simple test to verify the template can be loaded
    try:
        template = get_template()
        print(f"Successfully loaded template: {template.name}")
        
        # Test rendering with sample data
        html = template.render(
            model_name="Test Model",
            test_results={
                "uncertainty_quality_score": 0.85,
                "avg_coverage_error": 0.05,
                "avg_normalized_width": 0.3,
                "crqr": {
                    "by_alpha": {
                        "0.1": {
                            "overall_result": {
                                "coverage": 0.95, 
                                "mean_width": 0.45, 
                                "median_width": 0.4
                            }
                        }
                    }
                },
                "feature_importance": {"feature1": 0.5, "feature2": 0.3},
                "execution_time": 1.5
            },
            visualizations={},
            generation_time="2023-01-01 12:00:00"
        )
        print("Sample HTML output generated successfully")
        
    except Exception as e:
        print(f"Error working with template: {str(e)}")