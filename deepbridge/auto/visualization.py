import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

from deepbridge.auto.config import DistillationConfig
from deepbridge.auto.metrics import MetricsEvaluator

class Visualizer:
    """
    Creates visualizations for distillation experiment results.
    
    Generates various plots to analyze the performance of different
    model configurations, temperature and alpha impacts.
    """
    
    def __init__(
        self,
        results_df: pd.DataFrame,
        config: DistillationConfig,
        metrics_evaluator: MetricsEvaluator
    ):
        """
        Initialize the visualizer.
        
        Args:
            results_df: DataFrame containing experiment results
            config: Configuration for visualization
            metrics_evaluator: Metrics evaluator instance
        """
        self.results_df = results_df
        self.config = config
        self.metrics_evaluator = metrics_evaluator
    
    def create_all_visualizations(self):
        """
        Create and save all visualizations with robust error handling.
        Each visualization function is called in a try-except block to prevent
        failures in one visualization from affecting others.
        """
        valid_results = self.metrics_evaluator.get_valid_results()
        
        if valid_results.empty:
            self.config.log_info("No valid results to visualize")
            return
        
        self.config.log_info(f"Creating visualizations with {len(valid_results)} valid results")
        self.config.log_info(f"Columns available: {valid_results.columns.tolist()}")
        
        # 1. KL Divergence by Temperature
        try:
            if 'test_kl_divergence' in valid_results.columns and not valid_results['test_kl_divergence'].isna().all():
                self._plot_kl_divergence_by_temperature(valid_results)
                self.config.log_info("Created KL divergence by temperature plot")
            else:
                self.config.log_info("Skipping KL divergence plot: metric not available")
        except Exception as e:
            self.config.log_info(f"Error creating KL divergence plot: {str(e)}")
            import traceback
            self.config.log_info(traceback.format_exc())
        
        # 2. Accuracy by Alpha
        try:
            if 'test_accuracy' in valid_results.columns and not valid_results['test_accuracy'].isna().all():
                self._plot_accuracy_by_alpha(valid_results)
                self.config.log_info("Created accuracy by alpha plot")
            else:
                self.config.log_info("Skipping accuracy by alpha plot: metric not available")
        except Exception as e:
            self.config.log_info(f"Error creating accuracy by alpha plot: {str(e)}")
        
        # 3. Model Comparison (using our robust implementation)
        try:
            self._plot_model_comparison(valid_results)
            self.config.log_info("Created model comparison plots")
        except Exception as e:
            self.config.log_info(f"Error creating model comparison plot: {str(e)}")
        
        # 4. Individual Metric Comparisons
        available_metrics = self.metrics_evaluator.get_available_metrics()
        self.config.log_info(f"Available metrics for individual plots: {available_metrics}")
        
        for metric in ['precision', 'recall', 'f1', 'auc_roc', 'auc_pr']:
            if metric in available_metrics:
                try:
                    metric_col = f'test_{metric}'
                    if metric_col in valid_results.columns and not valid_results[metric_col].isna().all():
                        self._plot_metric_comparison(valid_results, metric)
                        self.config.log_info(f"Created {metric} comparison plot")
                    else:
                        self.config.log_info(f"Skipping {metric} plot: data not available")
                except Exception as e:
                    self.config.log_info(f"Error creating {metric} comparison plot: {str(e)}")
        
        # 5. Precision-Recall Trade-off
        if 'precision' in available_metrics and 'recall' in available_metrics:
            try:
                if ('test_precision' in valid_results.columns and 
                    'test_recall' in valid_results.columns and 
                    not valid_results['test_precision'].isna().all() and
                    not valid_results['test_recall'].isna().all()):
                    
                    self._plot_precision_recall_tradeoff(valid_results)
                    self.config.log_info("Created precision-recall trade-off plot")
                else:
                    self.config.log_info("Skipping precision-recall trade-off: data not available")
            except Exception as e:
                self.config.log_info(f"Error creating precision-recall trade-off plot: {str(e)}")
        
        self.config.log_info(f"Visualization process completed. Results saved to {self.config.output_dir}")
    
    def _plot_kl_divergence_by_temperature(self, results_df: pd.DataFrame):
        """
        Plot KL divergence by temperature for each model type.
        
        Args:
            results_df: DataFrame containing valid results
        """
        try:
            plt.figure(figsize=(15, 10))
            model_types = results_df['model_type'].unique()
            
            for i, temp in enumerate(self.config.temperatures):
                if i >= 4:  # Limit to 4 subplots
                    break
                    
                plt.subplot(2, 2, i+1)
                
                temp_data = results_df[results_df['temperature'] == temp]
                models = []
                kl_means = []
                kl_stds = []
                
                for model in model_types:
                    model_data = temp_data[temp_data['model_type'] == model]['test_kl_divergence']
                    if not model_data.empty and not model_data.isna().all():
                        models.append(model)
                        kl_means.append(model_data.mean())
                        kl_stds.append(model_data.std())
                
                if models:  # Only plot if there are valid data
                    x = range(len(models))
                    plt.bar(x, kl_means, yerr=kl_stds, capsize=10)
                    plt.xlabel('Model')
                    plt.ylabel('KL Divergence (Test)')
                    plt.title(f'KL Divergence with Temperature = {temp}')
                    plt.xticks(x, models, rotation=45)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                else:
                    plt.text(0.5, 0.5, 'No valid data for this temperature',
                            horizontalalignment='center', verticalalignment='center',
                            transform=plt.gca().transAxes)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_dir, 'kl_divergence_by_temperature.png'))
            plt.close()
        except Exception as e:
            self.config.log_info(f"Error in _plot_kl_divergence_by_temperature: {str(e)}")
            plt.close()
    
    def _plot_accuracy_by_alpha(self, results_df: pd.DataFrame):
        """
        Plot accuracy by alpha for each model type.
        
        Args:
            results_df: DataFrame containing valid results
        """
        try:
            plt.figure(figsize=(15, 10))
            model_types = results_df['model_type'].unique()
            
            for i, a in enumerate(self.config.alphas):
                if i >= 4:  # Limit to 4 subplots
                    break
                    
                plt.subplot(2, 2, i+1)
                
                alpha_data = results_df[results_df['alpha'] == a]
                models = []
                acc_means = []
                acc_stds = []
                
                for model in model_types:
                    model_data = alpha_data[alpha_data['model_type'] == model]['test_accuracy']
                    if not model_data.empty and not model_data.isna().all():
                        models.append(model)
                        acc_means.append(model_data.mean())
                        acc_stds.append(model_data.std())
                
                if models:  # Only plot if there are valid data
                    x = range(len(models))
                    plt.bar(x, acc_means, yerr=acc_stds, capsize=10)
                    plt.xlabel('Model')
                    plt.ylabel('Accuracy (Test)')
                    plt.title(f'Accuracy with Alpha = {a}')
                    plt.xticks(x, models, rotation=45)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                else:
                    plt.text(0.5, 0.5, 'No valid data for this alpha',
                           horizontalalignment='center', verticalalignment='center',
                           transform=plt.gca().transAxes)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_dir, 'accuracy_by_alpha.png'))
            plt.close()
        except Exception as e:
            self.config.log_info(f"Error in _plot_accuracy_by_alpha: {str(e)}")
            plt.close()
    
    def _plot_metric_comparison(self, results_df: pd.DataFrame, metric_name: str):
        """
        Plot comparison of a specific metric across models.
        
        Args:
            results_df: DataFrame containing valid results
            metric_name: Name of the metric to plot (without 'train_' or 'test_' prefix)
        """
        try:
            plt.figure(figsize=(15, 10))
            model_types = results_df['model_type'].unique()
            test_metric = f"test_{metric_name}"
            
            if test_metric not in results_df.columns:
                self.config.log_info(f"Metric {test_metric} not found in results")
                return
                
            for i, temp in enumerate(self.config.temperatures):
                if i >= len(self.config.temperatures) or i >= 3:  # Limit to 3 subplots
                    break
                    
                plt.subplot(min(len(self.config.temperatures), 3), 1, i+1)
                
                temp_data = results_df[results_df['temperature'] == temp]
                models = []
                metric_values = []
                
                for model in model_types:
                    model_data = temp_data[temp_data['model_type'] == model][test_metric]
                    if not model_data.empty and not model_data.isna().all():
                        models.append(model)
                        alpha_values = []
                        for alpha in self.config.alphas:
                            alpha_data = temp_data[(temp_data['model_type'] == model) & 
                                                  (temp_data['alpha'] == alpha)][test_metric]
                            if not alpha_data.empty and not alpha_data.isna().all():
                                alpha_values.append(alpha_data.mean())
                        metric_values.append(alpha_values)
                
                if models and metric_values and any(len(vals) > 0 for vals in metric_values):
                    x = range(len(models))
                    width = 0.2
                    
                    for j, alpha in enumerate(self.config.alphas):
                        if j >= len(self.config.alphas):
                            continue
                            
                        alpha_vals = []
                        for vals in metric_values:
                            if j < len(vals):
                                alpha_vals.append(vals[j])
                                
                        if alpha_vals:
                            plt.bar([pos + j*width for pos in x[:len(alpha_vals)]], 
                                    alpha_vals, 
                                    width=width, 
                                    label=f'Alpha={alpha}')
                    
                    plt.xlabel('Model')
                    plt.ylabel(f'{metric_name.upper()} (Test)')
                    plt.title(f'{metric_name.upper()} with Temperature = {temp}')
                    
                    if models:
                        plt.xticks([pos + width for pos in x[:len(models)]], models, rotation=45)
                        
                    plt.legend()
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                else:
                    plt.text(0.5, 0.5, f'No valid {metric_name} data for this temperature',
                            horizontalalignment='center', verticalalignment='center',
                            transform=plt.gca().transAxes)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_dir, f'{metric_name}_comparison.png'))
            plt.close()
        except Exception as e:
            self.config.log_info(f"Error in _plot_metric_comparison: {str(e)}")
            plt.close()
    
    def _plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Plot overall model comparison with robust error handling.
        
        Args:
            results_df: DataFrame containing valid results
        """
        try:
            # Get model metrics
            model_metrics_df = self.metrics_evaluator.get_model_comparison_metrics()
            
            # Basic validation
            if model_metrics_df is None or model_metrics_df.empty:
                self.config.log_info("No valid metrics for model comparison")
                return
                
            if 'model' not in model_metrics_df.columns:
                self.config.log_info("'model' column not found in metrics dataframe")
                return
                
            # Log available columns for debugging
            self.config.log_info(f"Available columns in model_metrics_df: {model_metrics_df.columns.tolist()}")
            
            # Verify required columns exist
            required_columns = ['model', 'max_accuracy', 'min_kl_div']
            missing_columns = [col for col in required_columns if col not in model_metrics_df.columns]
            if missing_columns:
                self.config.log_info(f"Missing required columns for basic comparison: {missing_columns}")
                return
                
            model_types = model_metrics_df['model'].tolist()
            
            # PLOT 1: Accuracy and KL divergence comparison
            # ==============================================
            x = range(len(model_types))
            fig, ax1 = plt.subplots(figsize=(14, 8))
            
            # Plot accuracy bars
            bars = ax1.bar(x, model_metrics_df['max_accuracy'], color='royalblue', alpha=0.7)
            ax1.set_xlabel('Model Type')
            ax1.set_ylabel('Max Accuracy', color='royalblue')
            ax1.tick_params(axis='y', labelcolor='royalblue')
            ax1.set_xticks(x)
            ax1.set_xticklabels(model_metrics_df['model'], rotation=45)
            
            # Add accuracy values on top of bars
            for bar, value in zip(bars, model_metrics_df['max_accuracy']):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', color='royalblue', fontweight='bold')
            
            # Create second y-axis for KL divergence
            ax2 = ax1.twinx()
            line = ax2.plot(x, model_metrics_df['min_kl_div'], 'ro-', linewidth=2, markersize=8)
            ax2.set_ylabel('Min KL Divergence', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Add KL divergence values
            for i, value in enumerate(model_metrics_df['min_kl_div']):
                ax2.text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom', color='red')
            
            plt.title('Model Comparison: Maximum Accuracy and Minimum KL Divergence')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_dir, 'model_comparison.png'))
            plt.close()
            
            # PLOT 2: Additional metrics comparison (if available)
            # ===================================================
            # Get available metrics from the metrics evaluator
            available_metrics = self.metrics_evaluator.get_available_metrics()
            self.config.log_info(f"Available metrics for visualization: {available_metrics}")
            
            # Check which metrics have corresponding columns in the DataFrame
            valid_metrics = []
            for metric in available_metrics:
                column_name = f'max_{metric}' if metric != 'kl_divergence' else 'min_kl_div'
                if column_name in model_metrics_df.columns:
                    valid_metrics.append((metric, column_name))
                    
            if not valid_metrics:
                self.config.log_info("No valid metric columns found for detailed comparison")
                return
                
            # Create subplots for valid metrics only
            metric_count = len(valid_metrics)
            if metric_count == 0:
                return
                
            # Calculate grid layout
            rows = min(3, (metric_count + 1) // 2)
            cols = min(2, metric_count)
            
            plt.figure(figsize=(cols * 9, rows * 4))
            
            for i, (metric, column) in enumerate(valid_metrics):
                if i >= rows * cols:  # Limit number of subplots
                    break
                    
                plt.subplot(rows, cols, i+1)
                
                if metric == 'kl_divergence':
                    title = f'Minimum {metric.upper()} by Model'
                else:
                    title = f'Maximum {metric.upper()} by Model'
                
                # Get values safely
                values = model_metrics_df[column].values
                
                # Create bar chart
                bars = plt.bar(range(len(model_types)), values, color='steelblue')
                plt.xlabel('Model Type')
                plt.ylabel(metric.upper())
                plt.title(title)
                plt.xticks(range(len(model_types)), model_metrics_df['model'], rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add values on top of bars
                for bar, value in zip(bars, values):
                    if pd.notnull(value):  # Check if the value is not NaN
                        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                                f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_dir, 'all_metrics_comparison.png'))
            plt.close()
            
        except Exception as e:
            self.config.log_info(f"Error in plot_model_comparison: {str(e)}")
            import traceback
            self.config.log_info(traceback.format_exc())
            plt.close()
    
    def _plot_precision_recall_tradeoff(self, results_df: pd.DataFrame):
        """
        Plot trade-off between precision and recall with robust error handling.
        
        Args:
            results_df: DataFrame containing valid results
        """
        try:
            # Verify required columns exist
            required_cols = ['test_precision', 'test_recall']
            for col in required_cols:
                if col not in results_df.columns:
                    self.config.log_info(f"Required column {col} not found in results DataFrame")
                    return
                    
            # Check if we have enough non-null values
            valid_data = results_df.dropna(subset=required_cols)
            if len(valid_data) < 2:
                self.config.log_info(f"Insufficient valid data points for precision-recall trade-off plot")
                return
                
            # Get available temperatures and models
            temperatures = sorted(results_df['temperature'].unique())
            model_types = results_df['model_type'].unique()
            
            # Create figure with appropriate size
            num_temps = min(len(temperatures), 4)  # Limit to 4 subplots max
            plt.figure(figsize=(15, 5 * num_temps))
            
            # Plot for each temperature
            for i, temp in enumerate(temperatures[:num_temps]):
                plt.subplot(num_temps, 1, i+1)
                
                # Filter data for this temperature
                temp_data = results_df[results_df['temperature'] == temp]
                temp_valid_data = temp_data.dropna(subset=required_cols)
                
                if len(temp_valid_data) < 2:
                    plt.text(0.5, 0.5, f'Insufficient valid data for temperature={temp}',
                            horizontalalignment='center', verticalalignment='center',
                            transform=plt.gca().transAxes)
                    continue
                
                # Plot each model
                plot_added = False
                for model in model_types:
                    model_data = temp_data[temp_data['model_type'] == model]
                    model_valid_data = model_data.dropna(subset=required_cols)
                    
                    if len(model_valid_data) > 0:
                        plt.scatter(
                            model_valid_data['test_recall'], 
                            model_valid_data['test_precision'],
                            label=model, 
                            s=80, 
                            alpha=0.7
                        )
                        plot_added = True
                        
                        # Add alpha annotations
                        for _, row in model_valid_data.iterrows():
                            if pd.notnull(row['test_precision']) and pd.notnull(row['test_recall']):
                                try:
                                    plt.annotate(
                                        f"α={row['alpha']}", 
                                        (row['test_recall'], row['test_precision']),
                                        textcoords="offset points",
                                        xytext=(0,10),
                                        ha='center'
                                    )
                                except Exception as e:
                                    self.config.log_info(f"Error adding annotation: {str(e)}")
                
                if not plot_added:
                    plt.text(0.5, 0.5, f'No valid data to plot for temperature={temp}',
                            horizontalalignment='center', verticalalignment='center',
                            transform=plt.gca().transAxes)
                    continue
                    
                # Add plot elements
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title(f'Precision-Recall Trade-off (Temperature = {temp})')
                plt.grid(True, alpha=0.3)
                plt.legend(title='Model Type')
                
                # Add reference line if we have valid data
                try:
                    min_val = max(0, min(temp_valid_data['test_precision'].min(), 
                                         temp_valid_data['test_recall'].min()))
                    max_val = min(1, max(temp_valid_data['test_precision'].max(), 
                                         temp_valid_data['test_recall'].max()))
                    if not (pd.isna(min_val) or pd.isna(max_val)):
                        plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3)
                except Exception as e:
                    self.config.log_info(f"Error adding reference line: {str(e)}")
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.output_dir, 'precision_recall_tradeoff.png'))
            plt.close()
            
        except Exception as e:
            self.config.log_info(f"Error in plot_precision_recall_tradeoff: {str(e)}")
            import traceback
            self.config.log_info(traceback.format_exc())
            plt.close()