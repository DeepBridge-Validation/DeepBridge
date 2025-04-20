"""
Module for calculating quality metrics for synthetic data.
"""

import pandas as pd
import numpy as np
import typing as t
import psutil

class MetricsCalculator:
    """
    Calculates quality metrics for synthetic data.
    Extracted from Synthesize class to separate metrics calculation responsibilities.
    """
    
    def __init__(
        self,
        verbose: bool = True,
        random_state: t.Optional[int] = None
    ):
        """
        Initialize the metrics calculator.
        
        Args:
            verbose: Whether to print progress information
            random_state: Seed for reproducibility
        """
        self.verbose = verbose
        self.random_state = random_state
        self.metrics = None
        self.metrics_calculator = None
    
    def calculate_metrics(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        generator: t.Any,
        dask_manager=None
    ) -> t.Dict[str, t.Any]:
        """
        Calculate quality metrics for the synthetic data.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset to evaluate
            generator: Generator object with column information
            dask_manager: Optional DaskManager for distributed computation
            
        Returns:
            Dictionary of quality metrics
        """
        from deepbridge.synthetic.metrics.synthetic_metrics import SyntheticMetrics
        
        self.log("Calculating quality metrics...")
        
        # Check memory before metrics calculation
        pre_metrics_memory = psutil.Process().memory_info().rss
        self.log(f"Memory usage before metrics calculation: {pre_metrics_memory / (1024**3):.2f} GB")
        
        try:
            # Limit sample size for metrics calculation to avoid memory issues
            max_metrics_samples = min(5000, len(original_data), len(synthetic_data))
            
            # Check if we have a DaskManager and if it's active
            use_dask = dask_manager is not None and dask_manager.is_active
            dask_client = dask_manager.client if use_dask else None
            
            # Use SyntheticMetrics class to calculate comprehensive metrics
            metrics_calculator = SyntheticMetrics(
                real_data=original_data,
                synthetic_data=synthetic_data,
                numerical_columns=generator.numerical_columns,
                categorical_columns=generator.categorical_columns,
                target_column=generator.target_column,
                sample_size=max_metrics_samples,
                random_state=self.random_state,
                verbose=self.verbose,
                use_dask=use_dask,
                dask_client=dask_client
            )
            
            # Store the metrics instance and get the metrics dictionary
            self.metrics_calculator = metrics_calculator
            self.metrics = metrics_calculator.get_metrics()
            
            # Check memory after metrics calculation
            post_metrics_memory = psutil.Process().memory_info().rss
            self.log(f"Memory usage after metrics calculation: {post_metrics_memory / (1024**3):.2f} GB")
            self.log(f"Memory increase: {(post_metrics_memory - pre_metrics_memory) / (1024**3):.2f} GB")
            
            return self.metrics
            
        except Exception as e:
            self.log(f"Error calculating quality metrics: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            
            self.metrics = {"error": str(e)}
            return self.metrics
    
    def print_summary(self) -> None:
        """Print a summary of the calculated metrics."""
        if self.metrics_calculator is not None:
            self.metrics_calculator.print_summary()
        elif self.metrics is not None:
            # Print a simple summary if full calculator is not available
            self.log("\n=== Synthetic Data Quality Metrics ===")
            
            if 'overall' in self.metrics:
                overall = self.metrics['overall']
                self.log(f"Overall quality score: {overall.get('quality_score', 0.0):.4f}")
            
            if 'statistical' in self.metrics:
                stats = self.metrics['statistical']
                self.log(f"Statistical similarity: {stats.get('overall_score', 0.0):.4f}")
            
            if 'privacy' in self.metrics:
                privacy = self.metrics['privacy']
                self.log(f"Privacy score: {privacy.get('overall_score', 0.0):.4f}")
            
            if 'utility' in self.metrics:
                utility = self.metrics['utility']
                self.log(f"Utility score: {utility.get('overall_score', 0.0):.4f}")
        else:
            self.log("No metrics available to display")
    
    def overall_quality(self) -> float:
        """
        Get the overall quality score of the synthetic data.
        
        Returns:
            Float between 0 and 1, where higher indicates better quality
        """
        if self.metrics_calculator is not None:
            return self.metrics_calculator.overall_quality()
        elif self.metrics and 'overall' in self.metrics:
            return self.metrics['overall'].get('quality_score', 0.0)
        return 0.0
    
    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)