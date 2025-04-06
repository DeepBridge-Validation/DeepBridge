"""
Uncertainty report generator module for DeepBridge.
Generates HTML reports from uncertainty test results.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from jinja2 import Template

class UncertaintyReportGenerator:
    """
    Generates uncertainty analysis reports based on test results.
    Uses Plotly.js for interactive visualizations.
    """
    
    def __init__(self):
        """Initialize the uncertainty report generator."""
        self.template_path = os.path.join(
            os.path.dirname(__file__),
            "uncertainty_report_template.html"
        )
    
    def generate_report(self, 
                       results: Dict[str, Any], 
                       output_path: str, 
                       model_name: str = "Primary Model",
                       experiment_info: Dict[str, Any] = None) -> str:
        """
        Generate an uncertainty report based on test results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            The uncertainty test results
        output_path : str
            Path to save the generated report
        model_name : str
            Name of the model being analyzed
        experiment_info : Dict[str, Any], optional
            Additional experiment information from experiment.experiment_info
            
        Returns:
        --------
        str : Path to the saved report
        """
        # Extract results for all models - handle both flat and nested structures
        if 'primary_model' in results:
            # Nested structure from experiment.run_tests()
            primary_results = results.get('primary_model', {})
            alternative_models = results.get('alternative_models', {})
        else:
            # Flat structure from direct run_uncertainty_tests() call
            primary_results = results
            alternative_models = {}
        
        # Prepare template data
        template_data = self._prepare_template_data(
            primary_results, 
            model_name,
            alternative_models,
            experiment_info
        )
        
        # Render template
        report_html = self._render_template(template_data)
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
            
        return output_path
    
    def _prepare_template_data(
        self, 
        results: Dict[str, Any], 
        model_name: str,
        alternative_models: Dict[str, Dict[str, Any]] = None,
        experiment_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Prepare data for the report template.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            The uncertainty test results for the primary model
        model_name : str
            Name of the model being analyzed
        alternative_models : Dict[str, Dict[str, Any]], optional
            Results for alternative models, keyed by model name
        experiment_info : Dict[str, Any], optional
            Additional experiment information
            
        Returns:
        --------
        Dict[str, Any] : Data for the template
        """
        # Initialize template data with basic information
        template_data = {
            'model_name': model_name,
            'model_type': results.get('model_type', 'Unknown'),
            'base_score': results.get('base_score', 0),
            'calibration_score': results.get('calibration_score', 0),
            'coverage_rate': results.get('coverage_rate', 0),
            'avg_interval_width': results.get('average_width', 0),
            'metric_name': results.get('metric', 'ROC AUC')
        }
        
        # Add experiment configuration information if available
        if experiment_info:
            # Try to extract config information
            config = {}
            dataset_info = {}
            
            # Config might be directly at the top level or inside a 'config' key
            if 'config' in experiment_info:
                config = experiment_info['config']
                if 'dataset_info' in config:
                    dataset_info = config['dataset_info']
            
            # Also check for configuration at the top level
            experiment_type = config.get('experiment_type', experiment_info.get('experiment_type', 'binary_classification'))
            test_size = config.get('test_size', experiment_info.get('test_size', 0.2))
            random_state = config.get('random_state', experiment_info.get('random_state', 42))
            
            # Update template with extracted information
            template_data.update({
                'experiment_type': experiment_type,
                'test_size': test_size,
                'random_state': random_state,
                'n_samples': dataset_info.get('n_samples', 0),
                'n_features': dataset_info.get('n_features', 0)
            })
            
            # Get metrics for primary model
            primary_model_metrics = {}
            
            # Option 1: metrics in experiment_info['models']['primary_model']['metrics']
            if 'models' in experiment_info and 'primary_model' in experiment_info['models']:
                model_info = experiment_info['models']['primary_model']
                if isinstance(model_info, dict) and 'metrics' in model_info:
                    primary_model_metrics = model_info['metrics']
            
            # Option 2: check in model data - first model might be primary
            elif 'models' in experiment_info:
                for model_name, model_data in experiment_info['models'].items():
                    if isinstance(model_data, dict) and 'metrics' in model_data:
                        primary_model_metrics = model_data['metrics']
                        break
                        
            # Option 3: metrics might be directly in primary_model
            elif 'primary_model' in experiment_info:
                pm = experiment_info['primary_model']
                if isinstance(pm, dict) and 'metrics' in pm:
                    primary_model_metrics = pm['metrics']
            
            # Use available metrics if found
            if primary_model_metrics:
                template_data['primary_metrics'] = primary_model_metrics
            else:
                template_data['primary_metrics'] = {}
                
                # Use base_score from results if available
                if 'base_score' in results:
                    template_data['primary_metrics']['roc_auc'] = results['base_score']
        else:
            # Default experiment info
            template_data.update({
                'experiment_type': 'binary_classification',
                'test_size': 0.2,
                'random_state': 42,
                'n_samples': 0,
                'n_features': 0,
                'primary_metrics': {}
            })
        
        # Prepare alpha level analysis
        alpha_levels = []
        coverage_rates = []
        interval_widths = []
        
        if 'by_alpha' in results:
            for alpha, alpha_data in sorted(results['by_alpha'].items()):
                alpha_level = float(alpha)
                alpha_levels.append(alpha_level)
                
                # Get coverage rate
                coverage = alpha_data.get('coverage', alpha_level)  # Default to alpha if not found
                coverage_rates.append(coverage)
                
                # Get interval width
                width = alpha_data.get('average_width', 0)
                interval_widths.append(width)
        else:
            # Provide some reasonable defaults if no alpha level data
            alpha_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
            coverage_rates = alpha_levels
            interval_widths = [0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Update template data with alpha analysis
        template_data.update({
            'alpha_levels': alpha_levels,
            'coverage_rates': coverage_rates,
            'interval_widths': interval_widths
        })
        
        # Process feature importance data
        feature_names = []
        feature_importance_values = []
        feature_importance_items = []
        feature_importance_pct_values = []
        feature_importance_pct = []
        
        if 'feature_importance' in results:
            feature_importance = results['feature_importance']
            if isinstance(feature_importance, dict):
                # Filter out any special keys if present
                feature_importance = {k: v for k, v in feature_importance.items() if not k.startswith('_')}
                
                # Sort by absolute importance (highest first)
                sorted_features = sorted(
                    feature_importance.items(), 
                    key=lambda x: abs(x[1]), 
                    reverse=True
                )
                
                feature_names = [f[0] for f in sorted_features]
                feature_importance_values = [f[1] for f in sorted_features]
                feature_importance_items = sorted_features
                
                # Calculate percentage of total importance
                total_abs_importance = sum(abs(v[1]) for v in sorted_features)
                if total_abs_importance > 0:
                    feature_importance_pct_values = [(100 * abs(v) / total_abs_importance) for v in feature_importance_values]
                    
                    # Create percentage items list for the table
                    feature_importance_pct = [(name, 100 * abs(val) / total_abs_importance) 
                                             for name, val in sorted_features]
        
        # Process alternative models data
        model_names = []
        alt_models = []
        alt_metrics = []
        alt_metrics_values = []
        
        if alternative_models:
            for alt_model_name, alt_model_results in alternative_models.items():
                model_names.append(alt_model_name)
                
                # Prepare alternative model data
                alt_model_data = {
                    'name': alt_model_name,
                    'calibration_score': alt_model_results.get('calibration_score', 0),
                    'coverage_rate': alt_model_results.get('coverage_rate', 0),
                    'avg_interval_width': alt_model_results.get('average_width', 0),
                    'base_score': alt_model_results.get('base_score', 0)
                }
                
                alt_models.append(alt_model_data)
                
                # Extract model metrics
                model_metrics = {}
                
                # Check for metrics in experiment_info if available
                found_metrics = False
                if experiment_info and 'models' in experiment_info:
                    models_dict = experiment_info['models']
                    if alt_model_name in models_dict and isinstance(models_dict[alt_model_name], dict):
                        model_info = models_dict[alt_model_name]
                        if 'metrics' in model_info:
                            model_metrics = model_info['metrics']
                            found_metrics = True
                
                # Make sure roc_auc is set if base_score is available
                if 'roc_auc' not in model_metrics and 'base_score' in alt_model_results:
                    model_metrics['roc_auc'] = alt_model_results['base_score']
                
                # Prepare metrics values for radar chart
                metrics_values = [
                    model_metrics.get('accuracy', 0),
                    model_metrics.get('roc_auc', model_metrics.get('auc', 0)),  # Prefer roc_auc but fall back to auc
                    model_metrics.get('f1', 0),
                    model_metrics.get('precision', 0),
                    model_metrics.get('recall', 0)
                ]
                
                alt_metrics.append(model_metrics)
                alt_metrics_values.append(metrics_values)
        
        # Make sure we have roc_auc in primary metrics
        primary_metrics = template_data.get('primary_metrics', {})
        if 'auc' in primary_metrics and 'roc_auc' not in primary_metrics:
            primary_metrics['roc_auc'] = primary_metrics['auc']
        
        # Prepare metrics values for the radar chart
        primary_metrics_values = [
            primary_metrics.get('accuracy', 0),
            primary_metrics.get('roc_auc', primary_metrics.get('auc', 0)),  # Prefer roc_auc but fall back to auc
            primary_metrics.get('f1', 0),
            primary_metrics.get('precision', 0),
            primary_metrics.get('recall', 0)
        ]
        
        # Prepare calibration curve data
        calibration_curve = {
            'expected': [0, 0.2, 0.4, 0.6, 0.8, 1.0],  # Default values if not available
            'actual': [0, 0.2, 0.4, 0.6, 0.8, 1.0]     # Default values (perfect calibration)
        }
        
        if 'calibration_curve' in results:
            curve_data = results['calibration_curve']
            if isinstance(curve_data, dict):
                if 'expected' in curve_data and 'actual' in curve_data:
                    calibration_curve = curve_data
            elif isinstance(curve_data, list):
                # Handle list format [expected, actual]
                if len(curve_data) >= 2:
                    calibration_curve = {
                        'expected': curve_data[0],
                        'actual': curve_data[1]
                    }
        
        # Prepare calibration histogram data (default empty if not available)
        calibration_histogram = results.get('calibration_histogram', [])
        
        # Update template data with all prepared values
        template_data.update({
            'feature_names': feature_names,
            'feature_importance_values': feature_importance_values,
            'feature_importance': feature_importance_items,
            'feature_importance_pct_values': feature_importance_pct_values,
            'feature_importance_pct': feature_importance_pct,
            'model_names': model_names,
            'alt_models': alt_models,
            'primary_metrics_values': primary_metrics_values,
            'alt_metrics': alt_metrics,
            'alt_metrics_values': alt_metrics_values,
            'calibration_curve': calibration_curve,
            'calibration_histogram': calibration_histogram
        })
        
        return template_data
    
    def _render_template(self, template_data: Dict[str, Any]) -> str:
        """
        Render the HTML template with the provided data.
        
        Parameters:
        -----------
        template_data : Dict[str, Any]
            Data for the template
            
        Returns:
        --------
        str : Rendered HTML
        """
        with open(self.template_path, 'r', encoding='utf-8') as f:
            template_content = f.read()
            
        template = Template(template_content)
        return template.render(**template_data)


def generate_uncertainty_report(
    results: Dict[str, Any], 
    output_path: str,
    model_name: str = "Primary Model",
    experiment_info: Dict[str, Any] = None
) -> str:
    """
    Generate an uncertainty report from test results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Uncertainty test results
    output_path : str
        Path to save the report
    model_name : str, optional
        Name of the model
    experiment_info : Dict[str, Any], optional
        Additional experiment information
        
    Returns:
    --------
    str : Path to the saved report
    """
    generator = UncertaintyReportGenerator()
    return generator.generate_report(results, output_path, model_name, experiment_info)