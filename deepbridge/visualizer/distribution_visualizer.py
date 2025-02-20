import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score
from pathlib import Path
import os

class DistributionVisualizer:
    """
    A specialized class for visualizing and comparing probability distributions
    between teacher and student models in knowledge distillation.
    """
    
    def __init__(self, output_dir: str = "distribution_plots"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set Seaborn styling
        sns.set_theme(style="darkgrid")
    
    def visualize_all(self, distiller, best_metric='test_kl_divergence', minimize=True):
        """
        Generate all visualizations in one call.
        
        Args:
            distiller: Trained AutoDistiller instance
            best_metric: Metric to use for finding the best model
            minimize: Whether the metric should be minimized
        """
        print("Generating all visualizations...")
        
        # 1. Generate distribution visualizations for best model
        self.visualize_distillation_results(distiller, best_metric, minimize)
        
        # 2. Generate precision-recall tradeoff plot
        self.create_precision_recall_plot(distiller.results_df)
        
        # 3. Generate distribution metrics by temperature
        self.create_distribution_metrics_by_temperature_plot(distiller.results_df)
        
        # 4. Generate model comparison plot
        model_metrics = distiller.metrics_evaluator.get_model_comparison_metrics()
        self.create_model_comparison_plot(model_metrics)
        
        print(f"All visualizations saved to {self.output_dir}")
    
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
    
    def compare_distributions(self,
                             teacher_probs,
                             student_probs,
                             title="Teacher vs Student Probability Distribution",
                             filename="probability_distribution_comparison.png",
                             show_metrics=True):
        """
        Create a visualization comparing teacher and student probability distributions.
        
        Args:
            teacher_probs: Teacher model probabilities
            student_probs: Student model probabilities
            title: Plot title
            filename: Output filename
            show_metrics: Whether to display distribution similarity metrics on the plot
            
        Returns:
            Dictionary containing calculated distribution metrics
        """
        # Process probabilities to correct format
        teacher_probs_processed = self._process_probabilities(teacher_probs)
        student_probs_processed = self._process_probabilities(student_probs)
            
        # Calculate distribution similarity metrics
        metrics = self._calculate_metrics(teacher_probs_processed, student_probs_processed)
        
        # Create the plot
        plt.figure(figsize=(12, 7))
        
        # Plot density curves
        sns.kdeplot(teacher_probs_processed, fill=True, color="royalblue", alpha=0.5, 
                   label="Teacher Model", linewidth=2)
        sns.kdeplot(student_probs_processed, fill=True, color="crimson", alpha=0.5, 
                   label="Student Model", linewidth=2)
        
        # Add histogram for additional clarity (normalized)
        plt.hist(teacher_probs_processed, bins=30, density=True, alpha=0.3, color="blue")
        plt.hist(student_probs_processed, bins=30, density=True, alpha=0.3, color="red")
        
        # Add titles and labels
        plt.xlabel("Probability Value", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        
        # Add metrics to the plot if requested
        if show_metrics:
            metrics_text = (
                f"KL Divergence: {metrics['kl_divergence']:.4f}\n"
                f"KS Statistic: {metrics['ks_statistic']:.4f} (p={metrics['ks_pvalue']:.4f})\n"
                f"R² Score: {metrics['r2_score']:.4f}\n"
                f"Jensen-Shannon: {metrics['jensen_shannon']:.4f}"
            )
            plt.annotate(metrics_text, xy=(0.02, 0.96), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                        va='top', fontsize=10)
        
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save and close the figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created distribution comparison: {output_path}")
        return metrics
    
    def compare_cumulative_distributions(self,
                                        teacher_probs,
                                        student_probs,
                                        title="Cumulative Distribution Comparison",
                                        filename="cumulative_distribution_comparison.png"):
        """
        Create a visualization comparing cumulative distributions.
        
        Args:
            teacher_probs: Teacher model probabilities
            student_probs: Student model probabilities
            title: Plot title
            filename: Output filename
        """
        # Process probabilities to correct format
        teacher_probs_processed = self._process_probabilities(teacher_probs)
        student_probs_processed = self._process_probabilities(student_probs)
        
        # Create CDF plot
        plt.figure(figsize=(12, 7))
        
        # Compute empirical CDFs
        x_teacher = np.sort(teacher_probs_processed)
        y_teacher = np.arange(1, len(x_teacher) + 1) / len(x_teacher)
        
        x_student = np.sort(student_probs_processed)
        y_student = np.arange(1, len(x_student) + 1) / len(x_student)
        
        # Plot CDFs
        plt.plot(x_teacher, y_teacher, '-', linewidth=2, color='royalblue', label='Teacher Model')
        plt.plot(x_student, y_student, '-', linewidth=2, color='crimson', label='Student Model')
        
        # Calculate KS statistic and visualize it
        ks_stat, ks_pvalue = stats.ks_2samp(teacher_probs_processed, student_probs_processed)
        
        # Find the point of maximum difference between the CDFs
        # This requires a bit of interpolation since the x-values may not align
        all_x = np.sort(np.unique(np.concatenate([x_teacher, x_student])))
        teacher_cdf_interp = np.interp(all_x, x_teacher, y_teacher)
        student_cdf_interp = np.interp(all_x, x_student, y_student)
        differences = np.abs(teacher_cdf_interp - student_cdf_interp)
        max_diff_idx = np.argmax(differences)
        max_diff_x = all_x[max_diff_idx]
        max_diff_y1 = teacher_cdf_interp[max_diff_idx]
        max_diff_y2 = student_cdf_interp[max_diff_idx]
        
        # Plot the KS statistic visualization
        plt.plot([max_diff_x, max_diff_x], [max_diff_y1, max_diff_y2], 'k--', linewidth=1.5)
        plt.scatter([max_diff_x], [max_diff_y1], s=50, color='royalblue')
        plt.scatter([max_diff_x], [max_diff_y2], s=50, color='crimson')
        
        ks_text = f"KS statistic: {ks_stat:.4f}\np-value: {ks_pvalue:.4f}"
        plt.annotate(ks_text, xy=(max_diff_x, (max_diff_y1 + max_diff_y2) / 2),
                    xytext=(max_diff_x + 0.1, (max_diff_y1 + max_diff_y2) / 2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Add labels and title
        plt.xlabel('Probability Value', fontsize=12)
        plt.ylabel('Cumulative Probability', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save and close the figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created cumulative distribution comparison: {output_path}")
    
    def create_quantile_plot(self,
                            teacher_probs,
                            student_probs,
                            title="Q-Q Plot: Teacher vs Student",
                            filename="qq_plot_comparison.png"):
        """
        Create a quantile-quantile plot to compare distributions.
        
        Args:
            teacher_probs: Teacher model probabilities
            student_probs: Student model probabilities
            title: Plot title
            filename: Output filename
        """
        # Process probabilities to correct format
        teacher_probs_processed = self._process_probabilities(teacher_probs)
        student_probs_processed = self._process_probabilities(student_probs)
        
        plt.figure(figsize=(10, 10))
        
        # Create Q-Q plot
        teacher_quantiles = np.quantile(teacher_probs_processed, np.linspace(0, 1, 100))
        student_quantiles = np.quantile(student_probs_processed, np.linspace(0, 1, 100))
        
        plt.scatter(teacher_quantiles, student_quantiles, color='purple', alpha=0.7)
        
        # Add reference line (perfect match)
        min_val = min(teacher_probs_processed.min(), student_probs_processed.min())
        max_val = max(teacher_probs_processed.max(), student_probs_processed.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, 
                label='Perfect Match Reference')
        
        # Calculate and display R² for the Q-Q line
        r2 = r2_score(teacher_quantiles, student_quantiles)
        plt.annotate(f"R² = {r2:.4f}", xy=(0.05, 0.95), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        plt.xlabel('Teacher Model Quantiles', fontsize=12)
        plt.ylabel('Student Model Quantiles', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add reference diagonal guides
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
        plt.axvline(x=0.5, color='gray', linestyle=':', alpha=0.5)
        
        # Save and close the figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created quantile plot: {output_path}")
    
    def visualize_distillation_results(self,
                                     auto_distiller,
                                     best_model_metric='test_kl_divergence',
                                     minimize=True):
        """
        Generate comprehensive distribution visualizations for the best distilled model.
        
        Args:
            auto_distiller: AutoDistiller instance with completed experiments
            best_model_metric: Metric to use for finding the best model
            minimize: Whether the metric should be minimized
        """
        try:
            # Find the best model configuration
            best_config = auto_distiller.find_best_model(metric=best_model_metric, minimize=minimize)
            
            model_type = best_config['model_type']
            temperature = best_config['temperature']
            alpha = best_config['alpha']
            
            # Log the best configuration
            print(f"Generating visualizations for best model:")
            print(f"  Model Type: {model_type}")
            print(f"  Temperature: {temperature}")
            print(f"  Alpha: {alpha}")
            print(f"  {best_model_metric}: {best_config.get(best_model_metric, 'N/A')}")
            
            # Get student model and predictions
            best_model = auto_distiller.get_trained_model(model_type, temperature, alpha)
            
            # Get test set from experiment_runner
            X_test = auto_distiller.experiment_runner.experiment.X_test
            y_test = auto_distiller.experiment_runner.experiment.y_test
            
            # Get student predictions
            student_probs = best_model.predict_proba(X_test)
            
            # Get teacher probabilities
            teacher_probs = auto_distiller.experiment_runner.experiment.prob_test
            
            # Create various distribution visualizations
            model_desc = f"{model_type}_t{temperature}_a{alpha}"
            
            # Distribution comparison
            self.compare_distributions(
                teacher_probs=teacher_probs,
                student_probs=student_probs,
                title=f"Probability Distribution: Teacher vs Best Student Model\n({model_desc})",
                filename=f"best_model_{model_desc}_distribution.png"
            )
            
            # Cumulative distribution
            self.compare_cumulative_distributions(
                teacher_probs=teacher_probs,
                student_probs=student_probs,
                title=f"Cumulative Distribution: Teacher vs Best Student Model\n({model_desc})",
                filename=f"best_model_{model_desc}_cdf.png"
            )
            
            # Q-Q plot
            self.create_quantile_plot(
                teacher_probs=teacher_probs,
                student_probs=student_probs,
                title=f"Q-Q Plot: Teacher vs Best Student Model\n({model_desc})",
                filename=f"best_model_{model_desc}_qq_plot.png"
            )
            
        except Exception as e:
            print(f"Error visualizing distillation results: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _process_probabilities(self, probs):
        """
        Process probabilities to extract positive class probabilities and ensure correct format.
        
        Args:
            probs: Input probabilities (DataFrame, Series, or ndarray)
            
        Returns:
            numpy.ndarray: Processed probability array for the positive class
        """
        # Handle pandas DataFrame
        if isinstance(probs, pd.DataFrame):
            # Check for specific probability columns
            if 'prob_class_1' in probs.columns:
                return probs['prob_class_1'].values
            elif 'prob_1' in probs.columns:
                return probs['prob_1'].values
            elif 'class_1_prob' in probs.columns:
                return probs['class_1_prob'].values
            # If no specific columns found, use the last column
            return probs.iloc[:, -1].values
        
        # Handle pandas Series
        if isinstance(probs, pd.Series):
            return probs.values
        
        # Handle numpy arrays
        if isinstance(probs, np.ndarray):
            # Extract positive class for 2D arrays
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                return probs[:, 1]
            return probs
        
        # If we get here, input format is not recognized
        raise ValueError(f"Unrecognized probability format: {type(probs)}")
    
    def _calculate_metrics(self, teacher_probs, student_probs):
        """
        Calculate distribution similarity metrics.
        
        Args:
            teacher_probs: Teacher model probabilities
            student_probs: Student model probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # KL Divergence
        try:
            epsilon = 1e-10
            teacher_probs_clip = np.clip(teacher_probs, epsilon, 1-epsilon)
            student_probs_clip = np.clip(student_probs, epsilon, 1-epsilon)
            
            # Create histograms with same bins for both distributions
            bins = np.linspace(0, 1, 50)
            teacher_hist, _ = np.histogram(teacher_probs_clip, bins=bins, density=True)
            student_hist, _ = np.histogram(student_probs_clip, bins=bins, density=True)
            
            # Add small epsilon to avoid division by zero
            teacher_hist = teacher_hist + epsilon
            student_hist = student_hist + epsilon
            
            # Normalize
            teacher_hist = teacher_hist / teacher_hist.sum()
            student_hist = student_hist / student_hist.sum()
            
            # Calculate KL divergence
            kl_div = np.sum(teacher_hist * np.log(teacher_hist / student_hist))
            metrics['kl_divergence'] = float(kl_div)
            
            # Calculate Jensen-Shannon divergence (symmetric)
            m = 0.5 * (teacher_hist + student_hist)
            js_div = 0.5 * np.sum(teacher_hist * np.log(teacher_hist / m)) + \
                     0.5 * np.sum(student_hist * np.log(student_hist / m))
            metrics['jensen_shannon'] = float(js_div)
            
        except Exception as e:
            print(f"Error calculating KL divergence: {str(e)}")
            metrics['kl_divergence'] = float('nan')
            metrics['jensen_shannon'] = float('nan')
            
        except Exception as e:
            print(f"Error calculating KL divergence: {str(e)}")
            metrics['kl_divergence'] = float('nan')
            metrics['jensen_shannon'] = float('nan')