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
    
    def compare_distributions(self,
                            teacher_probs: np.ndarray,
                            student_probs: np.ndarray,
                            title: str = "Teacher vs Student Probability Distribution",
                            filename: str = "probability_distribution_comparison.png",
                            show_metrics: bool = True) -> dict:
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
        # Ensure we're working with the right format of probabilities
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
        plt.xlabel("Probability Value")
        plt.ylabel("Density")
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
        
        return metrics
    
    def compare_cumulative_distributions(self,
                                        teacher_probs: np.ndarray,
                                        student_probs: np.ndarray,
                                        title: str = "Cumulative Distribution Comparison",
                                        filename: str = "cumulative_distribution_comparison.png") -> None:
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
        plt.xlabel('Probability Value')
        plt.ylabel('Cumulative Probability')
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save and close the figure
        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def create_quantile_plot(self,
                            teacher_probs: np.ndarray,
                            student_probs: np.ndarray,
                            title: str = "Q-Q Plot: Teacher vs Student",
                            filename: str = "qq_plot_comparison.png") -> None:
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
        
        plt.xlabel('Teacher Model Quantiles')
        plt.ylabel('Student Model Quantiles')
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

    def _calculate_metrics(self, teacher_probs: np.ndarray, student_probs: np.ndarray) -> dict:
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
        
        # KS statistic
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(teacher_probs, student_probs)
            metrics['ks_statistic'] = float(ks_stat)
            metrics['ks_pvalue'] = float(ks_pvalue)
        except Exception as e:
            print(f"Error calculating KS statistic: {str(e)}")
            metrics['ks_statistic'] = float('nan')
            metrics['ks_pvalue'] = float('nan')
        
        # R² score (using sorted distributions)
        try:
            # Sort both distributions to compare shape rather than correlation
            teacher_sorted = np.sort(teacher_probs)
            student_sorted = np.sort(student_probs)
            
            # Ensure equal length by sampling or truncating
            if len(teacher_sorted) != len(student_sorted):
                min_len = min(len(teacher_sorted), len(student_sorted))
                teacher_sorted = teacher_sorted[:min_len]
                student_sorted = student_sorted[:min_len]
                
            metrics['r2_score'] = float(r2_score(teacher_sorted, student_sorted))
        except Exception as e:
            print(f"Error calculating R² score: {str(e)}")
            metrics['r2_score'] = float('nan')
            
        return metrics
    
    def visualize_distillation_results(self,
                                      auto_distiller,
                                      best_model_metric: str = 'test_kl_divergence',
                                      minimize: bool = True) -> None:
        """
        Generate comprehensive distribution visualizations for the best distilled model.
        
        Args:
            auto_distiller: AutoDistiller instance with completed experiments
            best_model_metric: Metric to use for finding the best model
            minimize: Whether the metric should be minimized
        """
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
        
        # Get student model probabilities
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
        
        print(f"Visualizations saved to: {self.output_dir}")