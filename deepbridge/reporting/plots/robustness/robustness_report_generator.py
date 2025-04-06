"""
Robustness report generator module for DeepBridge.
Generates HTML reports from robustness test results.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from jinja2 import Template

class RobustnessReportGenerator:
    """
    Generates robustness analysis reports based on test results.
    Uses Plotly.js for interactive visualizations.
    """
    
    def __init__(self):
        """Initialize the robustness report generator."""
        self.template_path = os.path.join(
            os.path.dirname(__file__),
            "robustness_report_template.html"
        )
    
    def generate_report(self, 
                       results: Dict[str, Any], 
                       output_path: str, 
                       model_name: str = "Primary Model",
                       experiment_info: Dict[str, Any] = None) -> str:
        """
        Generate a robustness report based on test results.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            The robustness test results
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
            # Flat structure from direct run_robustness_tests() call
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
            The robustness test results for the primary model
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
            'robustness_score': 1.0 - results.get('avg_overall_impact', 0),
            'avg_raw_impact': results.get('avg_raw_impact', 0),
            'avg_quantile_impact': results.get('avg_quantile_impact', 0),
            'n_iterations': results.get('n_iterations', 1),
            'feature_subset': results.get('feature_subset', None),
            'metric_name': results.get('metric', 'AUC')
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
                'n_samples': dataset_info.get('n_samples'),
                'n_features': dataset_info.get('n_features')
            })
            
            # Get metrics for primary model - check all possible locations
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
            
            # If we found metrics, use them
            if primary_model_metrics:
                template_data['primary_metrics'] = primary_model_metrics
            else:
                template_data['primary_metrics'] = {
                    'accuracy': 0, 'auc': 0, 'f1': 0, 'precision': 0, 'recall': 0
                }
        else:
            # Default experiment info
            template_data.update({
                'experiment_type': 'binary_classification',
                'test_size': 0.2,
                'random_state': 42,
                'primary_metrics': {
                    'accuracy': 0, 'auc': 0, 'f1': 0, 'precision': 0, 'recall': 0
                }
            })
        
        # Prepare perturbation data
        perturbation_levels = []
        raw_mean_scores = []
        raw_worst_scores = []
        quantile_mean_scores = []
        quantile_worst_scores = []
        raw_results = []
        quantile_results = []
        raw_distributions = {}
        quantile_distributions = {}
        
        # Process raw perturbation results
        if 'raw' in results and 'by_level' in results['raw']:
            for level, level_data in sorted(results['raw']['by_level'].items(), key=lambda x: float(x[0])):
                perturbation_levels.append(float(level))
                
                # Get overall results - check for all_features first
                overall_result = level_data.get('overall_result', {})
                run_key = 'all_features'
                
                # Fall back to first key in overall_result if all_features doesn't exist
                if run_key not in overall_result and overall_result:
                    run_key = next(iter(overall_result))
                
                if run_key in overall_result:
                    mean_score = overall_result[run_key].get('mean_score', 0)
                    worst_score = overall_result[run_key].get('worst_score', 0)
                    std_score = overall_result[run_key].get('std_score', 0)
                    impact = overall_result[run_key].get('impact', 0)
                    
                    raw_mean_scores.append(mean_score)
                    raw_worst_scores.append(worst_score)
                    raw_results.append((level, {
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'worst_score': worst_score,
                        'impact': impact
                    }))
                    
                    # Extract score distributions for box plots
                    if 'runs' in level_data and run_key in level_data['runs']:
                        runs = level_data['runs'][run_key]
                        if runs and 'iterations' in runs[0]:
                            scores = runs[0]['iterations'].get('scores', [])
                            if scores:
                                raw_distributions[level] = scores
        
        # Process quantile perturbation results
        if 'quantile' in results and 'by_level' in results['quantile']:
            for level, level_data in sorted(results['quantile']['by_level'].items(), key=lambda x: float(x[0])):
                # Get overall results - check for all_features first
                overall_result = level_data.get('overall_result', {})
                run_key = 'all_features'
                
                # Fall back to first key in overall_result if all_features doesn't exist
                if run_key not in overall_result and overall_result:
                    run_key = next(iter(overall_result))
                
                if run_key in overall_result:
                    mean_score = overall_result[run_key].get('mean_score', 0)
                    worst_score = overall_result[run_key].get('worst_score', 0)
                    std_score = overall_result[run_key].get('std_score', 0)
                    impact = overall_result[run_key].get('impact', 0)
                    
                    quantile_mean_scores.append(mean_score)
                    quantile_worst_scores.append(worst_score)
                    quantile_results.append((level, {
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'worst_score': worst_score,
                        'impact': impact
                    }))
                    
                    # Extract score distributions for box plots
                    if 'runs' in level_data and run_key in level_data['runs']:
                        runs = level_data['runs'][run_key]
                        if runs and 'iterations' in runs[0]:
                            scores = runs[0]['iterations'].get('scores', [])
                            if scores:
                                quantile_distributions[level] = scores
        
        # Prepare feature subset comparison data if available
        feature_subset_comparison = None
        if template_data['feature_subset'] and 'raw' in results and 'by_level' in results['raw']:
            all_features_scores = []
            subset_features_scores = []
            
            for level, level_data in sorted(results['raw']['by_level'].items(), key=lambda x: float(x[0])):
                overall_result = level_data.get('overall_result', {})
                
                if 'all_features' in overall_result:
                    all_features_scores.append(overall_result['all_features'].get('mean_score', 0))
                    
                if 'feature_subset' in overall_result:
                    subset_features_scores.append(overall_result['feature_subset'].get('mean_score', 0))
            
            if all_features_scores and subset_features_scores:
                feature_subset_comparison = {
                    'all_features': all_features_scores,
                    'feature_subset': subset_features_scores
                }
        
        # Process feature importance data and convert to percentages
        feature_names = []
        feature_importance_values = []
        feature_importance_items = []
        feature_importance_pct_values = []
        feature_importance_pct = []
        
        if 'feature_importance' in results:
            feature_importance = results['feature_importance']
            if isinstance(feature_importance, dict):
                # Filter out the _detailed_results key if present
                feature_importance = {k: v for k, v in feature_importance.items() if k != '_detailed_results'}
                
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
        
        # Process alternative models data if available
        model_names = []
        raw_model_scores = []
        quantile_model_scores = []
        alternative_models_impact = []
        
        if alternative_models:
            for alt_model_name, alt_model_results in alternative_models.items():
                model_names.append(alt_model_name)
                
                # Extract raw perturbation scores
                alt_raw_scores = []
                if 'raw' in alt_model_results and 'by_level' in alt_model_results['raw']:
                    for level in perturbation_levels:
                        level_str = str(level)
                        if level_str in alt_model_results['raw']['by_level']:
                            level_data = alt_model_results['raw']['by_level'][level_str]
                            overall_result = level_data.get('overall_result', {})
                            run_key = 'all_features'
                            
                            # Fall back to first key in overall_result if all_features doesn't exist
                            if run_key not in overall_result and overall_result:
                                run_key = next(iter(overall_result))
                                
                            if run_key in overall_result:
                                alt_raw_scores.append(overall_result[run_key].get('mean_score', 0))
                            else:
                                alt_raw_scores.append(None)
                        else:
                            alt_raw_scores.append(None)
                            
                raw_model_scores.append(alt_raw_scores)
                
                # Extract quantile perturbation scores
                alt_quantile_scores = []
                if 'quantile' in alt_model_results and 'by_level' in alt_model_results['quantile']:
                    for level in perturbation_levels:
                        level_str = str(level)
                        if level_str in alt_model_results['quantile']['by_level']:
                            level_data = alt_model_results['quantile']['by_level'][level_str]
                            overall_result = level_data.get('overall_result', {})
                            run_key = 'all_features'
                            
                            # Fall back to first key in overall_result if all_features doesn't exist
                            if run_key not in overall_result and overall_result:
                                run_key = next(iter(overall_result))
                                
                            if run_key in overall_result:
                                alt_quantile_scores.append(overall_result[run_key].get('mean_score', 0))
                            else:
                                alt_quantile_scores.append(None)
                        else:
                            alt_quantile_scores.append(None)
                            
                quantile_model_scores.append(alt_quantile_scores)
                
                # Calculate average impact
                avg_raw_impact = alt_model_results.get('avg_raw_impact', 0)
                avg_quantile_impact = alt_model_results.get('avg_quantile_impact', 0)
                avg_impact = (avg_raw_impact + avg_quantile_impact) / 2
                alternative_models_impact.append(avg_impact)
        
        # Prepare metrics for models comparison
        primary_metrics = template_data['primary_metrics']
        
        # Look for metrics in standard field names to ensure consistency
        auc_field_names = ['auc', 'roc_auc', 'ROC_AUC', 'AUC']
        auc_value = 0
        
        # Try to find AUC in any of the field names
        for field in auc_field_names:
            if field in primary_metrics:
                auc_value = primary_metrics[field]
                break
                
        # If AUC is still 0, and we have a base_score in the results, use that
        # In robustness tests, base_score is typically the AUC value
        if auc_value == 0 and 'base_score' in results:
            auc_value = results['base_score']
        
        # Directly check template_data for base_score which we already extracted above
        if auc_value == 0 and 'base_score' in template_data:
            auc_value = template_data['base_score']
                
        # If AUC value is still 0, and we have alternative metrics like ROC in the results,
        # try to get the AUC value from the results
        if auc_value == 0:
            # Try to get AUC from primary_results metrics if it exists
            if results and isinstance(results, dict):
                # Check primary_model results first
                if 'primary_model' in results:
                    primary_model = results['primary_model']
                    
                    # Check various locations where AUC might be stored
                    if isinstance(primary_model, dict):
                        if 'metrics' in primary_model:
                            metrics = primary_model['metrics']
                            for field in auc_field_names:
                                if field in metrics:
                                    auc_value = metrics[field]
                                    break
                        elif 'base_score' in primary_model:
                            # base_score is sometimes the AUC
                            auc_value = primary_model['base_score']
                
                # Check if experiment_info has the metric value
                if auc_value == 0 and experiment_info:
                    # Look for metrics in experiment_info['primary_model']
                    if 'primary_model' in experiment_info:
                        pm = experiment_info['primary_model']
                        if isinstance(pm, dict) and 'metrics' in pm:
                            metrics = pm['metrics']
                            for field in auc_field_names:
                                if field in metrics:
                                    auc_value = metrics[field]
                                    break
                    
                    # Look for metrics in experiment_info['models']['primary_model']
                    if auc_value == 0 and 'models' in experiment_info:
                        if 'primary_model' in experiment_info['models']:
                            pm = experiment_info['models']['primary_model']
                            if isinstance(pm, dict) and 'metrics' in pm:
                                metrics = pm['metrics']
                                for field in auc_field_names:
                                    if field in metrics:
                                        auc_value = metrics[field]
                                        break
                
                # If still not found, check results directly
                if auc_value == 0:
                    for field in auc_field_names:
                        if field in results:
                            auc_value = results[field]
                            break
                            
        # If we absolutely cannot find AUC, try to extract it from robustness results
        if auc_value == 0:
            # Calculate metrics for all models
            metrics = {}
            # Extract metric from metric_name and base_score
            metric_name = template_data.get('metric_name', 'AUC').lower()
            base_score = template_data.get('base_score', 0)
            
            # If metric name is AUC, we should use the base_score
            if metric_name.lower() in ['auc', 'roc_auc']:
                auc_value = base_score
                # Use available metrics or calculate reasonable defaults
                metrics = {
                    'accuracy': primary_metrics.get('accuracy', 0),
                    'roc_auc': auc_value,
                    'f1': primary_metrics.get('f1', 0),
                    'precision': primary_metrics.get('precision', 0),
                    'recall': primary_metrics.get('recall', 0)
                }
                # Update primary metrics with these values
                primary_metrics.update(metrics)
            else:
                # Nothing to do - use existing metrics
        
        # Make sure we use roc_auc as the standard for AUC metrics
        if 'auc' in primary_metrics and 'roc_auc' not in primary_metrics:
            primary_metrics['roc_auc'] = primary_metrics['auc']
        
        primary_metrics_values = [
            primary_metrics.get('accuracy', 0),
            primary_metrics.get('roc_auc', primary_metrics.get('auc', 0)),  # Prefer roc_auc but fall back to auc
            primary_metrics.get('f1', 0),
            primary_metrics.get('precision', 0),
            primary_metrics.get('recall', 0)
        ]
        
        # Prepare metrics for alternative models
        alt_metrics = []
        alt_metrics_values = []
        
        # Process alternative models
        
        if alternative_models:
            # Extract alternative model metrics if available
            for model_name in model_names:
                model_metrics = {}
                metrics_values = [0, 0, 0, 0, 0]  # Default metrics values
                
                # Check for metrics in several possible locations
                found_metrics = False
                
                if experiment_info:
                    # Option 1: Check in experiment_info['models']
                    if 'models' in experiment_info:
                        models_dict = experiment_info['models']
                        if model_name in models_dict and isinstance(models_dict[model_name], dict):
                            model_info = models_dict[model_name]
                            if 'metrics' in model_info:
                                model_metrics = model_info['metrics']
                                found_metrics = True
                    
                    # Option 2: Check in alternative_models
                    if not found_metrics and model_name in alternative_models:
                        alt_model = alternative_models[model_name]
                        if isinstance(alt_model, dict) and 'metrics' in alt_model:
                            model_metrics = alt_model['metrics']
                            found_metrics = True
                
                # Look for AUC in various possible field names
                auc_field_names = ['auc', 'roc_auc', 'ROC_AUC', 'AUC']
                auc_value = 0
                
                # If AUC is not found but we have the model in alternative_models
                if auc_value == 0 and model_name in alternative_models:
                    alt_model_results = alternative_models[model_name]
                    if isinstance(alt_model_results, dict):
                        # Check various locations where AUC might be stored
                        if 'metrics' in alt_model_results:
                            metrics = alt_model_results['metrics']
                            for field in auc_field_names:
                                if field in metrics:
                                    auc_value = metrics[field]
                                    break
                        
                        # If AUC is still not found, use base_score
                        if auc_value == 0 and 'base_score' in alt_model_results:
                            # base_score is sometimes the AUC
                            auc_value = alt_model_results['base_score']
                
                # Make sure we have roc_auc field
                if 'roc_auc' not in model_metrics and auc_value > 0:
                    model_metrics['roc_auc'] = auc_value
                
                # Use real metrics when available
                metrics_values = [
                    model_metrics.get('accuracy', 0),
                    model_metrics.get('roc_auc', model_metrics.get('auc', 0)),  # Prefer roc_auc but fall back to auc
                    model_metrics.get('f1', 0),
                    model_metrics.get('precision', 0),
                    model_metrics.get('recall', 0)
                ]
                
                # Add robustness score
                idx = model_names.index(model_name)
                if idx < len(alternative_models_impact):
                    robustness = 1.0 - alternative_models_impact[idx]
                    model_metrics['robustness'] = robustness
                else:
                    model_metrics['robustness'] = 0
                
                alt_metrics.append(model_metrics)
                alt_metrics_values.append(metrics_values)
        
        # Ensure all data has default values to prevent template rendering errors
        if not feature_subset_comparison:
            feature_subset_comparison = {'all_features': [], 'feature_subset': []}
        
        # Add all data to template
        template_data.update({
            'perturbation_levels': perturbation_levels,
            'raw_mean_scores': raw_mean_scores,
            'raw_worst_scores': raw_worst_scores,
            'quantile_mean_scores': quantile_mean_scores,
            'quantile_worst_scores': quantile_worst_scores,
            'raw_results': raw_results,
            'quantile_results': quantile_results,
            'raw_distributions': raw_distributions,
            'quantile_distributions': quantile_distributions,
            'feature_names': feature_names,
            'feature_importance_values': feature_importance_values,
            'feature_importance': feature_importance_items,
            'feature_importance_pct_values': feature_importance_pct_values,
            'feature_importance_pct': feature_importance_pct,
            'feature_subset_comparison': feature_subset_comparison,
            'model_names': model_names,
            'raw_model_scores': raw_model_scores,
            'quantile_model_scores': quantile_model_scores,
            'alternative_models_impact': alternative_models_impact,
            'primary_metrics_values': primary_metrics_values,
            'alt_metrics': alt_metrics,
            'alt_metrics_values': alt_metrics_values
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


def generate_robustness_report(
    results: Dict[str, Any], 
    output_path: str,
    model_name: str = "Primary Model",
    experiment_info: Dict[str, Any] = None
) -> str:
    """
    Generate a robustness report from test results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Robustness test results
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
    generator = RobustnessReportGenerator()
    return generator.generate_report(results, output_path, model_name, experiment_info)