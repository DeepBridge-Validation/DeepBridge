import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score

from deepbridge.visualizer.base_visualizer import BaseVisualizer

class DistributionPlotsVisualizer(BaseVisualizer):
    """
    Specialized visualizer for creating distribution comparison plots.
    """
    
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