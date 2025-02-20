import os
import pandas as pd
from typing import List, Dict, Optional

from deepbridge.auto.config import DistillationConfig
from deepbridge.auto.metrics import MetricsEvaluator

class ReportGenerator:
    """
    Generates comprehensive reports for distillation experiments.
    
    Creates markdown and text reports summarizing experiment results,
    best configurations, and performance metrics.
    """
    
    def __init__(
        self,
        results_df: pd.DataFrame,
        config: DistillationConfig,
        metrics_evaluator: MetricsEvaluator
    ):
        """
        Initialize the report generator.
        
        Args:
            results_df: DataFrame containing experiment results
            config: Configuration for report generation
            metrics_evaluator: Metrics evaluator instance
        """
        self.results_df = results_df
        self.config = config
        self.metrics_evaluator = metrics_evaluator
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report of the distillation results.
        
        Returns:
            String containing the report
        """
        valid_results = self.metrics_evaluator.get_valid_results()
        
        if valid_results.empty:
            return "No valid results to generate report"
        
        report = ["# Knowledge Distillation Report\n"]
        
        # Add general information
        report.append("## General Information")
        report.append(f"- Number of models tested: {len(self.config.model_types)}")
        report.append(f"- Temperatures tested: {self.config.temperatures}")
        report.append(f"- Alpha values tested: {self.config.alphas}")
        report.append(f"- Total configurations: {self.config.get_total_configurations()}")
        report.append(f"- Valid results: {len(valid_results)}")
        report.append("")
        
        # Add best configurations for each metric
        report.append("## Best Configurations")

        # Define all metrics to find best models for
        metrics = [
            ('test_accuracy', False, 'Test Accuracy'),
            ('test_precision', False, 'Precision'),
            ('test_recall', False, 'Recall'),
            ('test_f1', False, 'F1 Score'),
            ('test_auc_roc', False, 'AUC-ROC'),
            ('test_auc_pr', False, 'AUC-PR'),
            ('test_kl_divergence', True, 'KL Divergence'),
            ('test_ks_statistic', True, 'KS Statistic'),
            ('test_r2_score', False, 'R² Score')
        ]

        for metric, minimize, metric_name in metrics:
            try:
                best = self.metrics_evaluator.find_best_model(metric=metric, minimize=minimize)
                report.append(f"### Best Model by {metric_name}")
                report.append(f"- Model Type: {best['model_type']}")
                report.append(f"- Temperature: {best['temperature']}")
                report.append(f"- Alpha: {best['alpha']}")
                report.append(f"- Test Accuracy: {best.get('test_accuracy', 'N/A')}")
                report.append(f"- Test Precision: {best.get('test_precision', 'N/A')}")
                report.append(f"- Test Recall: {best.get('test_recall', 'N/A')}")
                report.append(f"- Test F1: {best.get('test_f1', 'N/A')}")
                report.append(f"- Test AUC-ROC: {best.get('test_auc_roc', 'N/A')}")
                report.append(f"- Test AUC-PR: {best.get('test_auc_pr', 'N/A')}")
                report.append(f"- KL Divergence (Test): {best.get('test_kl_divergence', 'N/A')}")
                report.append(f"- KS Statistic (Test): {best.get('test_ks_statistic', 'N/A')}")
                report.append(f"- KS p-value (Test): {best.get('test_ks_pvalue', 'N/A')}")
                report.append(f"- R² Score (Test): {best.get('test_r2_score', 'N/A')}")
                report.append(f"- Parameters: {best.get('best_params', 'N/A')}")
                report.append("")
            except (ValueError, KeyError) as e:
                report.append(f"### Best Model by {metric_name}")
                report.append(f"Unable to find best model: {str(e)}")
                report.append("")
        
        # Add model comparison
        report.append("## Model Comparison")
        model_comparison = valid_results.groupby('model_type').agg({
            'test_accuracy': ['mean', 'max', 'std'],
            'train_accuracy': ['mean', 'max', 'std'],
            'test_precision': ['mean', 'max', 'std'] if 'test_precision' in valid_results.columns else None,
            'test_recall': ['mean', 'max', 'std'] if 'test_recall' in valid_results.columns else None,
            'test_f1': ['mean', 'max', 'std'] if 'test_f1' in valid_results.columns else None,
            'test_auc_roc': ['mean', 'max', 'std'] if 'test_auc_roc' in valid_results.columns else None,
            'test_auc_pr': ['mean', 'max', 'std'] if 'test_auc_pr' in valid_results.columns else None,
            'test_kl_divergence': ['mean', 'min', 'std']
        }).reset_index()
        
        # Filter out None values from the aggregation
        model_comparison = model_comparison.dropna(axis=1, how='all')
        
        model_comparison_str = model_comparison.to_string()
        report.append("```")
        report.append(model_comparison_str)
        report.append("```")
        report.append("")
        
        # Add temperature impact
        report.append("## Impact of Temperature")
        metrics_columns = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 
                          'test_auc_roc', 'test_auc_pr', 'test_kl_divergence']
        available_metrics = [col for col in metrics_columns if col in valid_results.columns]
        
        temp_impact = self.metrics_evaluator.get_factor_impact('temperature')
        
        temp_impact_str = temp_impact.to_string()
        report.append("```")
        report.append(temp_impact_str)
        report.append("```")
        report.append("")
        
        # Add alpha impact
        report.append("## Impact of Alpha")
        alpha_impact = self.metrics_evaluator.get_factor_impact('alpha')
        
        alpha_impact_str = alpha_impact.to_string()
        report.append("```")
        report.append(alpha_impact_str)
        report.append("```")
        report.append("")
        
        return '\n'.join(report)
    
    def save_report(self):
        """Generate and save the report to a file."""
        report = self.generate_report()
        
        # Save report
        report_path = os.path.join(self.config.output_dir, "distillation_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.config.log_info(f"Report saved to {report_path}")
        
        return report_path
    
    def generate_summary(self) -> str:
        """
        Generate a brief summary of the distillation results.
        
        Returns:
            String containing a summary
        """
        valid_results = self.metrics_evaluator.get_valid_results()
        
        if valid_results.empty:
            return "No valid results to generate summary"
        
        try:
            best_accuracy = self.metrics_evaluator.find_best_model(metric='test_accuracy')
            best_precision = self.metrics_evaluator.find_best_model(metric='test_precision')
            best_recall = self.metrics_evaluator.find_best_model(metric='test_recall')
            best_kl = self.metrics_evaluator.find_best_model(metric='test_kl_divergence', minimize=True)
            best_ks = self.metrics_evaluator.find_best_model(metric='test_ks_statistic', minimize=True)
            best_r2 = self.metrics_evaluator.find_best_model(metric='test_r2_score')
            
            summary = ["# Knowledge Distillation Summary\n"]
            
            summary.append("## Best Accuracy Configuration")
            summary.append(f"- Model: {best_accuracy['model_type']}")
            summary.append(f"- Temperature: {best_accuracy['temperature']}")
            summary.append(f"- Alpha: {best_accuracy['alpha']}")
            summary.append(f"- Test Accuracy: {best_accuracy.get('test_accuracy', 'N/A')}")
            summary.append(f"- KL Divergence: {best_accuracy.get('test_kl_divergence', 'N/A')}")
            summary.append(f"- KS Statistic: {best_accuracy.get('test_ks_statistic', 'N/A')}")
            summary.append(f"- R² Score: {best_accuracy.get('test_r2_score', 'N/A')}")
            summary.append("")
            
            summary.append("## Best Distribution Match by KL Divergence")
            summary.append(f"- Model: {best_kl['model_type']}")
            summary.append(f"- Temperature: {best_kl['temperature']}")
            summary.append(f"- Alpha: {best_kl['alpha']}")
            summary.append(f"- Test Accuracy: {best_kl.get('test_accuracy', 'N/A')}")
            summary.append(f"- KL Divergence: {best_kl.get('test_kl_divergence', 'N/A')}")
            summary.append(f"- KS Statistic: {best_kl.get('test_ks_statistic', 'N/A')}")
            summary.append(f"- R² Score: {best_kl.get('test_r2_score', 'N/A')}")
            summary.append("")
            
            summary.append("## Best Distribution Match by KS Statistic")
            summary.append(f"- Model: {best_ks['model_type']}")
            summary.append(f"- Temperature: {best_ks['temperature']}")
            summary.append(f"- Alpha: {best_ks['alpha']}")
            summary.append(f"- Test Accuracy: {best_ks.get('test_accuracy', 'N/A')}")
            summary.append(f"- KL Divergence: {best_ks.get('test_kl_divergence', 'N/A')}")
            summary.append(f"- KS Statistic: {best_ks.get('test_ks_statistic', 'N/A')}")
            summary.append(f"- R² Score: {best_ks.get('test_r2_score', 'N/A')}")
            summary.append("")
            
            summary.append("## Best Distribution Match by R² Score")
            summary.append(f"- Model: {best_r2['model_type']}")
            summary.append(f"- Temperature: {best_r2['temperature']}")
            summary.append(f"- Alpha: {best_r2['alpha']}")
            summary.append(f"- Test Accuracy: {best_r2.get('test_accuracy', 'N/A')}")
            summary.append(f"- KL Divergence: {best_r2.get('test_kl_divergence', 'N/A')}")
            summary.append(f"- KS Statistic: {best_r2.get('test_ks_statistic', 'N/A')}")
            summary.append(f"- R² Score: {best_r2.get('test_r2_score', 'N/A')}")
            summary.append("")
            
            # Add experiment info
            summary.append("## Experiment Overview")
            summary.append(f"- Total configurations tested: {self.config.get_total_configurations()}")
            summary.append(f"- Valid results: {len(valid_results)}")
            summary.append(f"- Models tested: {', '.join([str(m) for m in self.config.model_types])}")
            summary.append(f"- Results directory: {self.config.output_dir}")
            
            return '\n'.join(summary)
            
        except (ValueError, KeyError) as e:
            return f"Unable to generate summary: {e}"