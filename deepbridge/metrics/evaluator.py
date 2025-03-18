import pandas as pd
from typing import Dict, Optional, Union, List

from deepbridge.utils.model_registry import ModelType
from deepbridge.config.settings import DistillationConfig

class MetricsEvaluator:
    """
    Evaluates and analyzes metrics from distillation experiments.
    
    Provides functionality to find best models, analyze results,
    and extract metrics information.
    """
    
    def __init__(self, results_df: pd.DataFrame, config: DistillationConfig):
        """
        Initialize the metrics evaluator.
        
        Args:
            results_df: DataFrame containing experiment results
            config: Configuration information
        """
        self.results_df = results_df
        self.config = config
    
    def find_best_model(self, metric: str = 'test_accuracy', minimize: bool = False) -> Dict:
        """
        Find the best model configuration based on a specific metric with robust error handling.
        
        Args:
            metric: Metric to use for finding the best model (default: 'test_accuracy')
            minimize: Whether the metric should be minimized (default: False)
        
        Returns:
            Dictionary containing the best model configuration
        """
        try:
            # Verify that the metric exists in results
            if metric not in self.results_df.columns:
                available_metrics = [col for col in self.results_df.columns 
                                  if col.startswith('test_') or col.startswith('train_')]
                self.config.log_info(f"Metric '{metric}' not found. Available metrics: {available_metrics}")
                raise ValueError(f"Metric '{metric}' not found in results")
                
            # Get valid results (non-NaN for this metric)
            valid_results = self.results_df.dropna(subset=[metric])
            
            if valid_results.empty:
                self.config.log_info(f"No valid results for metric: {metric}")
                raise ValueError(f"No valid results for metric: {metric}")
            
            # Find best index
            if minimize:
                best_idx = valid_results[metric].idxmin()
                best_value = valid_results.loc[best_idx, metric]
                self.config.log_info(f"Found minimum {metric} = {best_value}")
            else:
                best_idx = valid_results[metric].idxmax()
                best_value = valid_results.loc[best_idx, metric]
                self.config.log_info(f"Found maximum {metric} = {best_value}")
            
            # Get best configuration
            best_config = valid_results.loc[best_idx].to_dict()
            
            # Log useful information
            self.config.log_info(f"Best model configuration based on {metric}:")
            for key in ['model_type', 'temperature', 'alpha', metric]:
                if key in best_config:
                    self.config.log_info(f"  {key}: {best_config[key]}")
            
            return best_config
            
        except Exception as e:
            self.config.log_info(f"Error finding best model for {metric}: {str(e)}")
            import traceback
            self.config.log_info(traceback.format_exc())
            raise ValueError(f"Error finding best model: {str(e)}")
    
    def get_valid_results(self, metric: Optional[str] = 'test_accuracy') -> pd.DataFrame:
        """
        Get valid results, optionally filtered by a specific metric.
        
        Args:
            metric: Metric to validate results against (default: 'test_accuracy')
        
        Returns:
            DataFrame with valid results
        """
        try:
            if metric:
                if metric not in self.results_df.columns:
                    self.config.log_info(f"Metric {metric} not found in results. Available columns: {self.results_df.columns.tolist()}")
                    return pd.DataFrame()
                return self.results_df.dropna(subset=[metric])
                
            return self.results_df
            
        except Exception as e:
            self.config.log_info(f"Error getting valid results: {str(e)}")
            return pd.DataFrame()
    
    def get_model_comparison_metrics(self) -> pd.DataFrame:
        """
        Get model comparison metrics across all experiments with robust error handling.
        
        Returns:
            DataFrame with model comparison metrics
        """
        try:
            valid_results = self.get_valid_results()
            
            if valid_results.empty:
                self.config.log_info("No valid results for model comparison metrics")
                return pd.DataFrame()
            
            model_types = valid_results['model_type'].unique()
            model_metrics = []
            
            # Determine available metrics
            metrics_map = {
                'test_accuracy': ('avg_accuracy', 'max_accuracy'),
                'test_precision': ('avg_precision', 'max_precision'),
                'test_recall': ('avg_recall', 'max_recall'),
                'test_f1': ('avg_f1', 'max_f1'),
                'test_auc_roc': ('avg_auc_roc', 'max_auc_roc'),
                'test_auc_pr': ('avg_auc_pr', 'max_auc_pr'),
                'test_kl_divergence': ('avg_kl_div', 'min_kl_div'),
                # Add new metrics
                'test_ks_statistic': ('avg_ks_stat', 'min_ks_stat'),
                'test_r2_score': ('avg_r2', 'max_r2')
            }
            
            # Check which metrics are actually available in the results
            available_metrics = {}
            for metric_col, (avg_name, max_name) in metrics_map.items():
                if metric_col in valid_results.columns and not valid_results[metric_col].isna().all():
                    available_metrics[metric_col] = (avg_name, max_name)
                    
            self.config.log_info(f"Available metrics for comparison: {list(available_metrics.keys())}")
            
            # If no metrics available, return empty DataFrame
            if not available_metrics:
                self.config.log_info("No metrics available for model comparison")
                return pd.DataFrame()
            
            for model in model_types:
                model_data = valid_results[valid_results['model_type'] == model]
                
                # Start with model name
                model_metric = {'model': model}
                
                # Add available metrics
                for metric_col, (avg_name, max_or_min_name) in available_metrics.items():
                    if metric_col in ['test_kl_divergence', 'test_ks_statistic']:
                        # For these metrics, we want minimum (lower is better)
                        model_metric[avg_name] = model_data[metric_col].mean()
                        model_metric[max_or_min_name] = model_data[metric_col].min()
                    else:
                        # For other metrics, we want maximum (higher is better)
                        model_metric[avg_name] = model_data[metric_col].mean()
                        model_metric[max_or_min_name] = model_data[metric_col].max()
                
                model_metrics.append(model_metric)
            
            # Convert to DataFrame
            result_df = pd.DataFrame(model_metrics)
            
            # Log the columns for debugging
            self.config.log_info(f"Created comparison metrics with columns: {result_df.columns.tolist()}")
            
            return result_df
            
        except Exception as e:
            self.config.log_info(f"Error in get_model_comparison_metrics: {str(e)}")
            import traceback
            self.config.log_info(traceback.format_exc())
            return pd.DataFrame()
    
    def get_factor_impact(self, factor: str) -> pd.DataFrame:
        """
        Analyze the impact of a specific factor (temperature or alpha) with robust error handling.
        
        Args:
            factor: Factor to analyze ('temperature' or 'alpha')
        
        Returns:
            DataFrame with factor impact analysis
        """
        try:
            valid_results = self.get_valid_results()
            
            if valid_results.empty:
                self.config.log_info(f"No valid results for {factor} impact analysis")
                return pd.DataFrame()
            
            # Check if factor exists in the results
            if factor not in valid_results.columns:
                self.config.log_info(f"Factor '{factor}' not found in results columns: {valid_results.columns.tolist()}")
                return pd.DataFrame()
            
            # Define all possible metrics columns
            metrics_columns = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1', 
                             'test_auc_roc', 'test_auc_pr', 'test_kl_divergence']
            
            # Filter to only include columns that actually exist in the results
            available_metrics = [col for col in metrics_columns if col in valid_results.columns]
            
            if not available_metrics:
                self.config.log_info("No metrics columns available for factor impact analysis")
                return pd.DataFrame()
            
            # Group by model type and factor, calculating mean for each available metric
            try:
                impact = valid_results.groupby(['model_type', factor])[available_metrics].agg('mean').reset_index()
                self.config.log_info(f"Created impact analysis with metrics: {available_metrics}")
                return impact
            except Exception as e:
                self.config.log_info(f"Error in groupby operation: {str(e)}")
                return pd.DataFrame()
        
        except Exception as e:
            self.config.log_info(f"Error in get_factor_impact for {factor}: {str(e)}")
            import traceback
            self.config.log_info(traceback.format_exc())
            return pd.DataFrame()
    
    def get_available_metrics(self) -> List[str]:
        """
        Get list of available metrics in the results with robust error checking.
        
        Returns:
            List of metric names
        """
        try:
            if self.results_df is None or self.results_df.empty:
                self.config.log_info("No results available for metrics analysis")
                return []
                
            base_metrics = [
                'accuracy', 'precision', 'recall', 'f1', 'auc_roc', 'auc_pr', 
                'kl_divergence', 'ks_statistic', 'r2_score'
            ]
            available = []
            
            for metric in base_metrics:
                test_metric = f'test_{metric}'
                if test_metric in self.results_df.columns and not self.results_df[test_metric].isna().all():
                    available.append(metric)
                    
            self.config.log_info(f"Available metrics found: {available}")
            return available
            
        except Exception as e:
            self.config.log_info(f"Error getting available metrics: {str(e)}")
            return []