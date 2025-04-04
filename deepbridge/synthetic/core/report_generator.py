"""
Module for generating reports on synthetic data quality.
"""

import pandas as pd
import typing as t
from pathlib import Path

class SyntheticReporter:
    """
    Generates reports on synthetic data quality.
    Extracted from Synthesize class to separate reporting responsibilities.
    """
    
    def __init__(
        self,
        verbose: bool = True
    ):
        """
        Initialize the report generator.
        
        Args:
            verbose: Whether to print progress information
        """
        self.verbose = verbose
        self.report_file = None
    
    def generate_report(
        self,
        original_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metrics: t.Dict[str, t.Any],
        generator_info: str,
        report_path: t.Optional[t.Union[str, Path]] = None,
        report_format: str = 'html',
        include_data_samples: bool = True,
        include_visualizations: bool = True,
        **kwargs
    ) -> t.Optional[str]:
        """
        Generate a quality report for the synthetic data.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset to evaluate
            metrics: Quality metrics from MetricsCalculator
            generator_info: Information about the generator
            report_path: Path to save the report
            report_format: Format of the report ('html' or 'text')
            include_data_samples: Whether to include data samples in the report
            include_visualizations: Whether to include visualizations
            **kwargs: Additional parameters for report customization
            
        Returns:
            Path to the generated report file or None if failed
        """
        from deepbridge.synthetic.reports.report_generator import generate_quality_report
        
        self.log("Generating quality report...")
        
        try:
            # Generate the report
            report_file = generate_quality_report(
                real_data=original_data,
                synthetic_data=synthetic_data,
                quality_metrics=metrics,
                report_path=report_path,
                generator_info=generator_info,
                include_data_samples=include_data_samples,
                report_format=report_format,
                include_visualizations=include_visualizations,
                **kwargs
            )
            
            # Store the report path
            self.report_file = report_file
            
            self.log(f"Quality report generated: {report_file}")
            
            return report_file
            
        except Exception as e:
            self.log(f"Error generating quality report: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            
            return None
    
    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)