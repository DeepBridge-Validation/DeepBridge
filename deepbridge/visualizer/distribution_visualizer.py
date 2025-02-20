import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from deepbridge.visualizer.base_visualizer import BaseVisualizer
from deepbridge.visualizer.distribution_plots import DistributionPlotsVisualizer
from deepbridge.visualizer.metrics_plots import MetricsPlotsVisualizer
from deepbridge.visualizer.model_comparison_plots import ModelComparisonVisualizer



class DistributionVisualizer(BaseVisualizer):
    """
    Main visualization class that coordinates all visualization types.
    This class serves as the main entry point for generating visualizations.
    """
    
    def __init__(self, output_dir: str = "distribution_plots"):
        """
        Initialize the visualizer.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        super().__init__(output_dir)
        
        # Initialize specialized visualizers
        self.distribution_viz = DistributionPlotsVisualizer(output_dir)
        self.metrics_viz = MetricsPlotsVisualizer(output_dir)
        self.model_comparison_viz = ModelComparisonVisualizer(output_dir)
    
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
            
            # Use the specialized distribution visualizer
            self.distribution_viz.compare_distributions(
                teacher_probs=teacher_probs,
                student_probs=student_probs,
                title=f"Probability Distribution: Teacher vs Best Student Model\n({model_desc})",
                filename=f"best_model_{model_desc}_distribution.png"
            )
            
            self.distribution_viz.compare_cumulative_distributions(
                teacher_probs=teacher_probs,
                student_probs=student_probs,
                title=f"Cumulative Distribution: Teacher vs Best Student Model\n({model_desc})",
                filename=f"best_model_{model_desc}_cdf.png"
            )
            
            self.distribution_viz.create_quantile_plot(
                teacher_probs=teacher_probs,
                student_probs=student_probs,
                title=f"Q-Q Plot: Teacher vs Best Student Model\n({model_desc})",
                filename=f"best_model_{model_desc}_qq_plot.png"
            )
            
        except Exception as e:
            print(f"Error visualizing distillation results: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_precision_recall_plot(self, results_df):
        """Create precision-recall trade-off plot."""
        self.metrics_viz.create_precision_recall_plot(results_df)
    
    def create_distribution_metrics_by_temperature_plot(self, results_df):
        """Create visualization showing distribution metrics by temperature."""
        self.metrics_viz.create_distribution_metrics_by_temperature_plot(results_df)
        
    def create_model_comparison_plot(self, model_metrics):
        """Create bar chart comparing model performance across metrics."""
        self.model_comparison_viz.create_model_comparison_plot(model_metrics)
        
    # Convenience methods that delegate to the distribution visualizer
    def compare_distributions(self, teacher_probs, student_probs, title=None, filename=None, show_metrics=True):
        """Delegate to distribution visualizer."""
        return self.distribution_viz.compare_distributions(
            teacher_probs, student_probs, title, filename, show_metrics
        )
    
    def compare_cumulative_distributions(self, teacher_probs, student_probs, title=None, filename=None):
        """Delegate to distribution visualizer."""
        return self.distribution_viz.compare_cumulative_distributions(
            teacher_probs, student_probs, title, filename
        )
    
    def create_quantile_plot(self, teacher_probs, student_probs, title=None, filename=None):
        """Delegate to distribution visualizer."""
        return self.distribution_viz.create_quantile_plot(
            teacher_probs, student_probs, title, filename
        )