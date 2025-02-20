import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .base_visualizer import BaseVisualizer

class ModelComparisonVisualizer(BaseVisualizer):
    """
    Specialized visualizer for creating model comparison plots.
    """
    
    def create_model_comparison_plot(self, model_metrics):
        """
        Create bar chart comparing model performance across metrics.
        
        Args:
            model_metrics: DataFrame with model comparison metrics
        """
        try:
            if model_metrics is not None and not model_metrics.empty:
                plt.figure(figsize=(14, 10))
                
                # Number of metrics to display
                metric_keys = []
                metric_display_names = {
                    'max_accuracy': 'Accuracy',
                    'max_precision': 'Precision',
                    'max_recall': 'Recall',
                    'max_f1': 'F1 Score',
                    'min_kl_div': 'KL Divergence',
                    'min_ks_stat': 'KS Statistic',
                    'max_r2': 'R² Score'
                }
                
                for metric in ['max_accuracy', 'min_kl_div', 'min_ks_stat', 'max_r2']:
                    if metric in model_metrics.columns:
                        metric_keys.append(metric)
                
                if metric_keys:
                    models = model_metrics['model'].tolist()
                    x = np.arange(len(models))
                    n_metrics = len(metric_keys)
                    width = 0.8 / n_metrics
                    
                    # Color map for better visualization
                    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860']
                    
                    for i, metric in enumerate(metric_keys):
                        display_name = metric_display_names.get(metric, metric)
                        is_minimize = 'min_' in metric
                        
                        values = model_metrics[metric].values
                        offset = (i - n_metrics/2 + 0.5) * width
                        
                        bars = plt.bar(x + offset, values, width, 
                                label=display_name,
                                color=colors[i % len(colors)])
                        
                        # Add value labels
                        for bar in bars:
                            height = bar.get_height()
                            plt.annotate(f'{height:.3f}',
                                       xy=(bar.get_x() + bar.get_width()/2, height),
                                       xytext=(0, 3),
                                       textcoords="offset points",
                                       ha='center', va='bottom',
                                       fontsize=9)
                    
                    plt.xlabel('Model Type', fontsize=12)
                    plt.ylabel('Metric Value', fontsize=12)
                    plt.title('Model Performance Comparison Across Metrics', fontsize=14, fontweight='bold')
                    plt.xticks(x, models, rotation=45)
                    plt.legend(title='Metrics')
                    plt.grid(axis='y', alpha=0.3)
                    
                    # Add explanatory notes
                    note_text = "Note: "
                    for metric in metric_keys:
                        if 'min_' in metric:
                            display_name = metric_display_names.get(metric, metric)
                            note_text += f"For {display_name}, lower is better. "
                        else:
                            display_name = metric_display_names.get(metric, metric)
                            note_text += f"For {display_name}, higher is better. "
                    
                    plt.figtext(0.5, 0.01, note_text, ha="center", fontsize=10, 
                               bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
                    
                    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                    output_path = os.path.join(self.output_dir, 'model_performance_comparison.png')
                    plt.savefig(output_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Created model comparison bar chart: {output_path}")
                    
        except Exception as e:
            print(f"Error creating model comparison visualization: {str(e)}")
    
    def create_performance_metrics_grid(self, results_df, metrics=None):
        """
        Create a grid of performance metrics for different models.
        
        Args:
            results_df: DataFrame containing experiment results
            metrics: List of metrics to include (default: accuracy, precision, recall, f1)
        """
        if metrics is None:
            metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        
        try:
            # Filter for metrics that actually exist in the dataframe
            available_metrics = [m for m in metrics if m in results_df.columns]
            
            if not available_metrics:
                print("No available metrics to plot")
                return
                
            # Get unique model types
            models = results_df['model_type'].unique()
            
            if len(models) == 0:
                print("No models to compare")
                return
                
            # Set up the plot grid
            n_metrics = len(available_metrics)
            n_cols = min(3, n_metrics)
            n_rows = (n_metrics + n_cols - 1) // n_cols
                
            plt.figure(figsize=(n_cols * 6, n_rows * 5))
                
            # Plot each metric
            for i, metric in enumerate(available_metrics):
                plt.subplot(n_rows, n_cols, i+1)
                    
                # Plot metric for each model
                for model in models:
                    model_data = results_df[results_df['model_type'] == model]
                    valid_data = model_data.dropna(subset=[metric])
                        
                    if not valid_data.empty:
                        # Group by temperature and calculate mean
                        grouped = valid_data.groupby('temperature')[metric].mean()
                        plt.plot(grouped.index, grouped.values, 'o-', 
                                label=model, linewidth=2, markersize=8)
                            
                # Add labels and title
                display_name = metric.replace('test_', '').capitalize()
                plt.xlabel('Temperature', fontsize=12)
                plt.ylabel(display_name, fontsize=12)
                plt.title(f'{display_name} by Temperature', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend(title='Model Type')
                    
            plt.tight_layout()
            output_path = os.path.join(self.output_dir, 'performance_metrics_grid.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
                
            print(f"Created performance metrics grid: {output_path}")
                
        except Exception as e:
            print(f"Error creating performance metrics grid: {str(e)}")
            
    def create_alpha_impact_plot(self, results_df, metric='test_accuracy'):
        """
        Create visualization showing the impact of alpha on a specific metric.
        
        Args:
            results_df: DataFrame containing experiment results
            metric: Metric to analyze
        """
        try:
            if metric in results_df.columns and not results_df[metric].isna().all():
                plt.figure(figsize=(12, 8))
                
                # Plot for each model
                for model in results_df['model_type'].unique():
                    model_data = results_df[results_df['model_type'] == model]
                    alphas = sorted(model_data['alpha'].unique())
                    
                    metric_values = []
                    valid_alphas = []
                    
                    for alpha in alphas:
                        alpha_data = model_data[model_data['alpha'] == alpha][metric]
                        if not alpha_data.empty and not alpha_data.isna().all():
                            valid_alphas.append(alpha)
                            metric_values.append(alpha_data.mean())
                    
                    if valid_alphas and metric_values:
                        plt.plot(valid_alphas, metric_values, 'o-', 
                                linewidth=2, markersize=8,
                                label=f"{model}")
                
                # Format the plot
                plt.xlabel('Alpha (α)', fontsize=12)
                display_name = metric.replace('test_', '').capitalize()
                plt.ylabel(display_name, fontsize=12)
                plt.title(f'Effect of Alpha on {display_name}', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(title="Model Type")
                
                # Add explanatory note
                is_minimize = metric in ['test_kl_divergence', 'test_ks_statistic']
                direction = "lower" if is_minimize else "higher"
                plt.figtext(0.5, 0.01,
                          f"This plot shows how the weighting parameter alpha affects {display_name}.\n"
                          f"Alpha controls the balance between teacher prediction matching and ground truth prediction.\n"
                          f"For {display_name}, {direction} values are better.",
                          ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
                
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                output_path = os.path.join(self.output_dir, f'alpha_impact_on_{metric.replace("test_", "")}.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Created alpha impact plot: {output_path}")
                
        except Exception as e:
            print(f"Error creating alpha impact visualization: {str(e)}")