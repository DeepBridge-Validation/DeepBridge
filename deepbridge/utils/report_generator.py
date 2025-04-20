import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import jinja2
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

class HTMLReportGenerator:
    """
    Generates HTML reports for distillation experiments.
    
    This class creates interactive HTML reports to visualize and compare 
    distillation results, including model performance metrics and distribution comparisons.
    """
    
    def __init__(
        self,
        results_df: pd.DataFrame,
        output_dir: str,
        metrics_evaluator: Any = None,
        template_dir: Optional[str] = None
    ):
        """
        Initialize the HTML report generator.
        
        Args:
            results_df: DataFrame containing experiment results
            output_dir: Directory to save the generated reports
            metrics_evaluator: Optional metrics evaluator instance to calculate additional metrics
            template_dir: Optional directory containing custom HTML templates
        """
        self.results_df = results_df
        self.output_dir = output_dir
        self.metrics_evaluator = metrics_evaluator
        
        # Setup Jinja2 templates
        self.template_dir = template_dir
        if not self.template_dir:
            # Use default templates embedded in the package
            self.template_dir = os.path.join(os.path.dirname(__file__), 'templates')
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize templating environment
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
    def generate_reports(self) -> Dict[str, str]:
        """
        Generate all HTML reports.
        
        Returns:
            Dictionary with report names and their file paths
        """
        reports = {}
        
        # Generate comparison report
        comparison_report_path = self.generate_comparison_report()
        reports['comparison'] = comparison_report_path
        
        # Generate best model report
        best_model_report_path = self.generate_best_model_report()
        reports['best_model'] = best_model_report_path
        
        return reports
        
    def generate_comparison_report(self) -> str:
        """
        Generate HTML report comparing all models.
        
        Returns:
            Path to the generated HTML report
        """
        # Prepare data for the report
        best_models_by_metric = self._get_best_models_by_metric()
        model_comparison_data = self._prepare_model_comparison_data()
        
        # Load the template
        template = self.env.get_template('comparison_report_template.html')
        
        # CHANGE THIS PART - Add distribution_data
        
        # Create placeholder distribution data
        distribution_data = self._prepare_distribution_data(self._get_best_model())
        
        # Render the template with the data
        html_content = template.render(
            title="Model Distillation - Best Models Comparison",
            date=datetime.now().strftime("%Y-%m-%d"),
            best_models=best_models_by_metric,
            model_comparison=model_comparison_data,
            summary=self._generate_summary(),
            distribution_data=distribution_data  # Add this line
        )
        
        # Save the rendered HTML to a file
        output_path = os.path.join(self.output_dir, "model_comparison_report.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return output_path
        
    def generate_best_model_report(self) -> str:
        """
        Generate detailed HTML report for the best model.
        
        Returns:
            Path to the generated HTML report
        """
        # Get the best model by accuracy (or other metric if specified)
        best_model = self._get_best_model()
        
        # Load the template
        template = self.env.get_template('best_model_report_template.html')
        
        # Render the template with the data
        html_content = template.render(
            title="Best Distilled Model - Detailed Report",
            date=datetime.now().strftime("%Y-%m-%d"),
            model=best_model,
            metrics=self._prepare_detailed_metrics(best_model),
            distribution_data=self._prepare_distribution_data(best_model),
            feature_importance=self._get_feature_importance(best_model)
        )
        
        # Save the rendered HTML to a file
        output_path = os.path.join(self.output_dir, "best_model_report.html")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return output_path

    def _get_best_models_by_metric(self) -> Dict[str, Dict]:
        """
        Find the best model for each evaluation metric.
        
        Returns:
            Dictionary mapping metrics to their best model configurations
        """
        best_models = {}
        
        # Metrics to evaluate (metric_name, minimize_flag, display_name)
        metrics = [
            ('test_accuracy', False, 'Accuracy'),
            ('test_precision', False, 'Precision'),
            ('test_recall', False, 'Recall'),
            ('test_f1', False, 'F1 Score'),
            ('test_auc_roc', False, 'AUC-ROC'),
            ('test_auc_pr', False, 'AUC-PR'),
            ('test_kl_divergence', True, 'KL Divergence'),
            ('test_ks_statistic', True, 'KS Statistic'),
            ('test_r2_score', False, 'R² Score')
        ]
        
        # Find the best model for each metric
        for metric_name, minimize, display_name in metrics:
            if metric_name in self.results_df.columns:
                # Skip if all values are NaN
                if self.results_df[metric_name].isna().all():
                    continue
                    
                # Find the best row
                if minimize:
                    best_idx = self.results_df[metric_name].idxmin()
                else:
                    best_idx = self.results_df[metric_name].idxmax()
                
                # Get the row as a dictionary
                best_row = self.results_df.loc[best_idx].to_dict()
                
                # Store the best model configuration
                best_models[metric_name] = {
                    'metric_name': metric_name,
                    'display_name': display_name,
                    'minimize': minimize,
                    'model_type': best_row['model_type'],
                    'temperature': best_row['temperature'],
                    'alpha': best_row['alpha'],
                    'value': best_row[metric_name],
                    'teacher_value': self._get_teacher_value(metric_name),
                    'best_params': self._parse_best_params(best_row.get('best_params', '{}'))
                }
                
                # Calculate the difference if teacher value exists
                if best_models[metric_name]['teacher_value'] is not None:
                    teacher_val = best_models[metric_name]['teacher_value']
                    student_val = best_models[metric_name]['value']
                    
                    best_models[metric_name]['difference'] = student_val - teacher_val
                    
                    # Calculate retention percentage for metrics where it makes sense
                    if not minimize and teacher_val != 0:
                        best_models[metric_name]['retention'] = (student_val / teacher_val) * 100
                    elif minimize and teacher_val != 0:
                        # For metrics like KL divergence, lower is better and teacher is usually close to 0
                        # So retention doesn't make sense in the same way
                        best_models[metric_name]['retention'] = None
                
        return best_models
        
    def _get_best_model(self, metric: str = 'test_accuracy') -> Dict:
        """
        Get the best model configuration based on a specific metric.
        
        Args:
            metric: Metric name to use for finding the best model
            
        Returns:
            Dictionary containing the best model configuration
        """
        # For metrics like KL divergence, lower is better
        minimize = metric in ['test_kl_divergence', 'test_ks_statistic']
        
        # Skip if all values are NaN
        if self.results_df[metric].isna().all():
            raise ValueError(f"No valid values for metric {metric}")
            
        # Find the best row
        if minimize:
            best_idx = self.results_df[metric].idxmin()
        else:
            best_idx = self.results_df[metric].idxmax()
        
        # Get the row as a dictionary
        best_model = self.results_df.loc[best_idx].to_dict()
        
        # Add parsed hyperparameters
        best_model['parsed_params'] = self._parse_best_params(best_model.get('best_params', '{}'))
        
        # Add teacher values for comparison
        for col in self.results_df.columns:
            if col.startswith('test_'):
                best_model[f'teacher_{col}'] = self._get_teacher_value(col)
        
        return best_model
        
    def _prepare_model_comparison_data(self) -> Dict:
        """
        Prepare model comparison data for the report.
        
        Returns:
            Dictionary with model comparison data
        """
        # If we have a metrics evaluator, use it to get model comparison metrics
        if self.metrics_evaluator:
            try:
                model_metrics = self.metrics_evaluator.get_model_comparison_metrics()
                if not model_metrics.empty:
                    return model_metrics.to_dict(orient='records')
            except Exception as e:
                print(f"Error getting model comparison metrics: {e}")
        
        # Fallback to direct calculation if metrics_evaluator is not available
        # or encountered an error
        model_comparison = []
        
        # Group by model_type and calculate metrics
        grouped = self.results_df.groupby('model_type')
        
        for model_type, group in grouped:
            model_data = {'model': model_type}
            
            # Calculate metrics for each model type
            for col in self.results_df.columns:
                if col.startswith('test_'):
                    # Skip if all values are NaN
                    if group[col].isna().all():
                        continue
                        
                    # For metrics like KL divergence, lower is better
                    minimize = col in ['test_kl_divergence', 'test_ks_statistic']
                    
                    if minimize:
                        model_data[f'min_{col.replace("test_", "")}'] = group[col].min()
                    else:
                        model_data[f'max_{col.replace("test_", "")}'] = group[col].max()
                    
                    model_data[f'avg_{col.replace("test_", "")}'] = group[col].mean()
            
            model_comparison.append(model_data)
        
        return model_comparison
        
    def _prepare_detailed_metrics(self, best_model: Dict) -> Dict:
        """
        Prepare detailed metrics for the best model.
        
        Args:
            best_model: Best model configuration
            
        Returns:
            Dictionary with detailed metrics
        """
        metrics = {}
        
        # Extract test metrics from the best model
        for key, value in best_model.items():
            if key.startswith('test_'):
                metric_name = key.replace('test_', '')
                
                # Get teacher value if available
                teacher_value = best_model.get(f'teacher_{key}')
                
                metrics[metric_name] = {
                    'name': metric_name,
                    'display_name': metric_name.replace('_', ' ').title(),
                    'value': value,
                    'teacher_value': teacher_value
                }
                
                # Calculate the difference if teacher value exists
                if teacher_value is not None:
                    metrics[metric_name]['difference'] = value - teacher_value
                    
                    # Calculate retention percentage for most metrics
                    minimize = metric_name in ['kl_divergence', 'ks_statistic']
                    if not minimize and teacher_value != 0:
                        metrics[metric_name]['retention'] = (value / teacher_value) * 100
        
        return metrics
        
    def _prepare_distribution_data(self, best_model: Dict) -> Dict:
        """
        Prepare distribution data for visualizations.
        
        Args:
            best_model: Best model configuration
            
        Returns:
            Dictionary with distribution data
        """
        # In a real implementation, this would extract actual distribution data
        # from saved model predictions. For now, we'll return placeholder data.
        
        return {
            'teacher_probs': list(np.linspace(0, 1, 100)),
            'student_probs': list(np.linspace(0.05, 0.95, 100)),
            'teacher_errors': list(np.random.normal(0, 0.15, 100)),
            'student_errors': list(np.random.normal(0, 0.17, 100))
        }
        
    def _get_feature_importance(self, best_model: Dict) -> List[Dict]:
        """
        Get feature importance data if available.
        
        Args:
            best_model: Best model configuration
            
        Returns:
            List of dictionaries with feature names and importance values
        """
        # In a real implementation, this would extract feature importance
        # from the saved model. For now, we'll return placeholder data.
        
        return [
            {'feature': f'Feature {i}', 'importance': 0.2 - (i * 0.02)} 
            for i in range(10)
        ]
        
    def _get_teacher_value(self, metric: str) -> Optional[float]:
        """
        Get the teacher value for a specific metric.
        
        In a real implementation, this would retrieve the actual teacher 
        model's performance on this metric from stored results.
        
        Args:
            metric: Metric name
            
        Returns:
            Teacher value or None if not available
        """
        # Default teacher values for common metrics
        if metric == 'test_accuracy':
            return 0.891
        elif metric == 'test_precision':
            return 0.887
        elif metric == 'test_recall':
            return 0.879
        elif metric == 'test_f1':
            return 0.882
        elif metric == 'test_auc_roc':
            return 0.953
        elif metric == 'test_auc_pr':
            return 0.934
        elif metric == 'test_kl_divergence':
            return 0.000  # Perfect match for KL divergence
        elif metric == 'test_ks_statistic':
            return 0.000  # Perfect match for KS statistic
        elif metric == 'test_r2_score':
            return 1.000  # Perfect match for R² score
        
        return None
        
    def _parse_best_params(self, params_str: str) -> Dict:
        """
        Parse best parameters string to dictionary.
        
        Args:
            params_str: String representation of parameters
            
        Returns:
            Dictionary of parameters
        """
        try:
            if params_str and params_str != '{}':
                return json.loads(params_str.replace("'", '"'))
            return {}
        except Exception as e:
            print(f"Error parsing best params: {e}")
            return {}
            
    def _generate_summary(self) -> Dict:
        """
        Generate a summary of the results.
        
        Returns:
            Dictionary with summary information
        """
        # Extract key findings
        best_models = self._get_best_models_by_metric()
        
        # Find which model type appears most frequently
        model_counts = {}
        for metric, model in best_models.items():
            model_type = model['model_type']
            model_counts[model_type] = model_counts.get(model_type, 0) + 1
            
        best_overall_model = max(model_counts.items(), key=lambda x: x[1])[0]
        
        # Find best temperature and alpha values
        temp_values = [model['temperature'] for model in best_models.values()]
        alpha_values = [model['alpha'] for model in best_models.values()]
        
        avg_temp = sum(temp_values) / len(temp_values) if temp_values else 0
        avg_alpha = sum(alpha_values) / len(alpha_values) if alpha_values else 0
        
        # Find models with best distribution matching
        distribution_metrics = ['test_kl_divergence', 'test_r2_score', 'test_ks_statistic']
        dist_models = [best_models[m] for m in distribution_metrics if m in best_models]
        
        if dist_models:
            # Count model types for distribution metrics
            dist_model_counts = {}
            for model in dist_models:
                model_type = model['model_type']
                dist_model_counts[model_type] = dist_model_counts.get(model_type, 0) + 1
                
            best_dist_model = max(dist_model_counts.items(), key=lambda x: x[1])[0]
        else:
            best_dist_model = best_overall_model
        
        return {
            'best_overall_model': best_overall_model,
            'best_dist_model': best_dist_model,
            'avg_temperature': avg_temp,
            'avg_alpha': avg_alpha,
            'recommended_model': best_dist_model,
            'recommended_temp': 2.0 if avg_temp > 1.5 else 1.0,
            'recommended_alpha': round(avg_alpha * 2) / 2  # Round to nearest 0.5
        }