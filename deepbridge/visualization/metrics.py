import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deepbridge.visualization.base import BaseVisualizer

class MetricsVisualizer(BaseVisualizer):
    """
    Specialized visualizer for creating metrics-related plots and model comparisons.
    This class consolidates functionality from both metrics and model comparison visualizers.
    """
    
    #
    # Precision-Recall and Distribution Metrics Visualizations
    #
    
    def create_precision_recall_plot(self, results_df):
        """
        Create precision-recall trade-off plot.
        
        Args:
            results_df: DataFrame containing experiment results
        """
        try:
            if ('test_precision' in results_df.columns and not results_df['test_precision'].isna().all() and
                'test_recall' in results_df.columns and not results_df['test_recall'].isna().all()):
                
                plt.figure(figsize=(12, 8))
                
                # Plot scatter points for each model
                for model in results_df['model_type'].unique():
                    model_data = results_df[results_df['model_type'] == model]
                    valid_data = model_data.dropna(subset=['test_precision', 'test_recall'])
                    
                    if not valid_data.empty:
                        # Create scatter plot
                        scatter = plt.scatter(
                            valid_data['test_recall'], 
                            valid_data['test_precision'],
                            label=model,
                            alpha=0.7,
                            s=80
                        )
                        
                        # Add alpha annotations
                        for _, row in valid_data.iterrows():
                            plt.annotate(
                                f"α={row['alpha']}, T={row['temperature']}", 
                                (row['test_recall'], row['test_precision']),
                                textcoords="offset points",
                                xytext=(0, 5),
                                ha='center',
                                fontsize=8
                            )

                # Add reference line if we have valid data
                valid_results = results_df.dropna(subset=['test_precision', 'test_recall'])
                if not valid_results.empty:
                    max_val = max(
                        valid_results['test_precision'].max(),
                        valid_results['test_recall'].max()
                    )
                    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.3)
                    
                plt.xlabel('Recall', fontsize=12)
                plt.ylabel('Precision', fontsize=12)
                plt.title('Precision vs Recall Trade-off', fontsize=14, fontweight='bold')
                plt.grid(True, alpha=0.3)
                plt.legend(title='Model Type')
                
                # Add explanatory text
                plt.figtext(0.01, 0.01, 
                           "Points above the diagonal line indicate better precision at the expense of recall.\n"
                           "Points closer to (1,1) show better overall performance.",
                           ha="left", fontsize=9, style='italic')
                
                output_path = os.path.join(self.output_dir, 'precision_recall_tradeoff.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Created precision-recall tradeoff plot: {output_path}")
                
        except Exception as e:
            print(f"Error creating precision-recall visualization: {str(e)}")
    
    def create_distribution_metrics_by_temperature_plot(self, results_df):
        """
        Create visualization showing distribution metrics by temperature.
        
        Args:
            results_df: DataFrame containing experiment results
        """
        try:
            if ('test_ks_statistic' in results_df.columns and not results_df['test_ks_statistic'].isna().all() and
                'test_r2_score' in results_df.columns and not results_df['test_r2_score'].isna().all()):
                
                plt.figure(figsize=(15, 12))
                
                # Plot KS statistic (lower is better)
                plt.subplot(2, 1, 1)
                for model in results_df['model_type'].unique():
                    model_data = results_df[results_df['model_type'] == model]
                    temps = sorted(model_data['temperature'].unique())
                    
                    for alpha in sorted(model_data['alpha'].unique()):
                        alpha_data = model_data[model_data['alpha'] == alpha]
                        if not alpha_data.empty:
                            ks_values = []
                            valid_temps = []
                            
                            for temp in temps:
                                temp_data = alpha_data[alpha_data['temperature'] == temp]['test_ks_statistic']
                                if not temp_data.empty and not temp_data.isna().all():
                                    valid_temps.append(temp)
                                    ks_values.append(temp_data.mean())
                            
                            if valid_temps and ks_values:
                                plt.plot(valid_temps, ks_values, 'o-', 
                                        linewidth=2, markersize=8,
                                        label=f"{model} (α={alpha})")
                
                plt.xlabel('Temperature', fontsize=12)
                plt.ylabel('KS Statistic (lower is better)', fontsize=12)
                plt.title('Effect of Temperature on Distribution Similarity (KS Statistic)', 
                         fontsize=14, fontweight='bold')
                
                # Highlight that lower is better
                ymin, ymax = plt.ylim()
                plt.annotate('Better', xy=(0.02, 0.1), xycoords='axes fraction',
                           xytext=(0.02, 0.25), 
                           arrowprops=dict(arrowstyle='->', color='green'),
                           color='green', fontweight='bold')
                
                plt.grid(True, alpha=0.3)
                plt.legend(title="Model & Alpha")
                
                # Plot R² score (higher is better)
                plt.subplot(2, 1, 2)
                for model in results_df['model_type'].unique():
                    model_data = results_df[results_df['model_type'] == model]
                    temps = sorted(model_data['temperature'].unique())
                    
                    for alpha in sorted(model_data['alpha'].unique()):
                        alpha_data = model_data[model_data['alpha'] == alpha]
                        if not alpha_data.empty:
                            r2_values = []
                            valid_temps = []
                            
                            for temp in temps:
                                temp_data = alpha_data[alpha_data['temperature'] == temp]['test_r2_score']
                                if not temp_data.empty and not temp_data.isna().all():
                                    valid_temps.append(temp)
                                    r2_values.append(temp_data.mean())
                            
                            if valid_temps and r2_values:
                                plt.plot(valid_temps, r2_values, 'o-', 
                                        linewidth=2, markersize=8,
                                        label=f"{model} (α={alpha})")
                
                plt.xlabel('Temperature', fontsize=12)
                plt.ylabel('R² Score (higher is better)', fontsize=12)
                plt.title('Effect of Temperature on Distribution Similarity (R² Score)', 
                         fontsize=14, fontweight='bold')
                
                # Highlight that higher is better
                ymin, ymax = plt.ylim()
                plt.annotate('Better', xy=(0.02, 0.9), xycoords='axes fraction',
                           xytext=(0.02, 0.75), 
                           arrowprops=dict(arrowstyle='->', color='green'),
                           color='green', fontweight='bold')
                
                plt.grid(True, alpha=0.3)
                plt.legend(title="Model & Alpha")
                
                # Add explanatory notes
                plt.figtext(0.5, 0.01,
                          "These plots show how temperature affects the similarity between teacher and student probability distributions.\n"
                          "KS Statistic: Measures maximum difference between distributions (lower is better).\n"
                          "R² Score: Measures how well the distributions align (higher is better).",
                          ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
                
                plt.tight_layout(rect=[0, 0.03, 1, 0.97])
                output_path = os.path.join(self.output_dir, 'distribution_metrics_by_temperature.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Created distribution metrics by temperature plot: {output_path}")
                
        except Exception as e:
            print(f"Error creating distribution metrics visualization: {str(e)}")
    
    def create_kl_divergence_plot(self, results_df):
        """
        Create visualization showing KL divergence by temperature and alpha.
        
        Args:
            results_df: DataFrame containing experiment results
        """
        try:
            if 'test_kl_divergence' in results_df.columns and not results_df['test_kl_divergence'].isna().all():
                plt.figure(figsize=(14, 8))
                
                for model in results_df['model_type'].unique():
                    model_data = results_df[results_df['model_type'] == model]
                    temps = sorted(model_data['temperature'].unique())
                    
                    # Different line style for each model
                    for alpha in sorted(model_data['alpha'].unique()):
                        alpha_data = model_data[model_data['alpha'] == alpha]
                        kl_values = []
                        valid_temps = []
                        
                        for temp in temps:
                            temp_data = alpha_data[alpha_data['temperature'] == temp]['test_kl_divergence']
                            if not temp_data.empty and not temp_data.isna().all():
                                valid_temps.append(temp)
                                kl_values.append(temp_data.mean())
                        
                        if valid_temps and kl_values:
                            plt.plot(valid_temps, kl_values, 'o-', 
                                    linewidth=2, markersize=8,
                                    label=f"{model} (α={alpha})")
                
                plt.xlabel('Temperature', fontsize=12)
                plt.ylabel('KL Divergence (lower is better)', fontsize=12)
                plt.title('KL Divergence by Temperature', fontsize=14, fontweight='bold')
                
                # Highlight that lower is better
                plt.annotate('Better', xy=(0.02, 0.1), xycoords='axes fraction',
                           xytext=(0.02, 0.25), 
                           arrowprops=dict(arrowstyle='->', color='green'),
                           color='green', fontweight='bold')
                
                plt.grid(True, alpha=0.3)
                plt.legend(title="Model & Alpha")
                
                # Add explanatory notes
                plt.figtext(0.5, 0.01,
                         "KL Divergence measures how much one probability distribution diverges from another.\n"
                         "Lower values indicate better matching between teacher and student distributions.",
                         ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
                
                plt.tight_layout(rect=[0, 0.05, 1, 0.95])
                output_path = os.path.join(self.output_dir, 'kl_divergence_by_temperature.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"Created KL divergence plot: {output_path}")
                
        except Exception as e:
            print(f"Error creating KL divergence visualization: {str(e)}")
    
    def create_metrics_heatmap(self, results_df, metric='test_kl_divergence', model_type=None):
        """
        Create a heatmap visualization of metrics by temperature and alpha.
        
        Args:
            results_df: DataFrame containing experiment results
            metric: Metric to visualize
            model_type: Specific model type to filter for (if None, uses first available)
        """
        try:
            if metric not in results_df.columns or results_df[metric].isna().all():
                print(f"Metric {metric} not available in results")
                return
                
            # Filter for the specified model or use the first one
            if model_type is None:
                model_types = results_df['model_type'].unique()
                if len(model_types) == 0:
                    print("No models available in results")
                    return
                model_type = model_types[0]
            
            model_data = results_df[results_df['model_type'] == model_type]
            if model_data.empty:
                print(f"No data available for model type {model_type}")
                return
                
            # Get unique temperatures and alphas
            temps = sorted(model_data['temperature'].unique())
            alphas = sorted(model_data['alpha'].unique())
            
            if not temps or not alphas:
                print("No temperature or alpha values available")
                return
                
            # Create 2D grid for heatmap
            values = np.zeros((len(alphas), len(temps)))
            for i, alpha in enumerate(alphas):
                for j, temp in enumerate(temps):
                    filtered = model_data[(model_data['alpha'] == alpha) & (model_data['temperature'] == temp)]
                    if not filtered.empty and not filtered[metric].isna().all():
                        values[i, j] = filtered[metric].mean()
                    else:
                        values[i, j] = np.nan
            
            # Create heatmap with proper labels
            plt.figure(figsize=(12, 8))
            
            # Determine colormap based on metric
            is_minimize = metric in ['test_kl_divergence', 'test_ks_statistic']
            cmap = 'viridis_r' if is_minimize else 'viridis'
            
            # Create heatmap with filled NaN values (in grey)
            masked_values = np.ma.masked_invalid(values)
            im = plt.imshow(masked_values, cmap=cmap, aspect='auto')
            plt.colorbar(im, label=metric.replace('test_', '').replace('_', ' ').title())
            
            # Add x and y labels
            plt.xticks(range(len(temps)), [f"{t:.1f}" for t in temps])
            plt.yticks(range(len(alphas)), [f"{a:.2f}" for a in alphas])
            plt.xlabel('Temperature', fontsize=12)
            plt.ylabel('Alpha (α)', fontsize=12)
            
            # Add title with model name
            display_metric = metric.replace('test_', '').replace('_', ' ').title()
            plt.title(f'{display_metric} for {model_type} by Temperature and Alpha', 
                    fontsize=14, fontweight='bold')
            
            # Add text annotations with actual values
            for i in range(len(alphas)):
                for j in range(len(temps)):
                    if not np.isnan(values[i, j]):
                        text_color = 'white' if im.norm(values[i, j]) > 0.5 else 'black'
                        plt.text(j, i, f'{values[i, j]:.3f}', 
                                ha='center', va='center', color=text_color)
            
            # Add note about interpretation
            direction = "lower" if is_minimize else "higher"
            plt.figtext(0.5, 0.01,
                     f"This heatmap shows how different combinations of temperature and alpha affect {display_metric}.\n"
                     f"For this metric, {direction} values (shown in {'darker' if is_minimize else 'brighter'} color) are better.",
                     ha="center", fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            metric_name = metric.replace('test_', '').replace('_', '')
            output_path = os.path.join(self.output_dir, f'{model_type}_{metric_name}_heatmap.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created metrics heatmap: {output_path}")
            
        except Exception as e:
            print(f"Error creating metrics heatmap: {str(e)}")

    #
    # Model Comparison Visualizations
    #
    
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