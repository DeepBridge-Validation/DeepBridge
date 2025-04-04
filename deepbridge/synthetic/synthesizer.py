import pandas as pd
import numpy as np
import typing as t
from pathlib import Path
import gc
import psutil
import warnings
from datetime import datetime
import traceback

from deepbridge.synthetic.core import (
    DaskManager,
    SyntheticDataProcessor,
    DataGenerator,
    MetricsCalculator,
    SyntheticReporter
)

class Synthesize:
    """
    A unified interface for generating synthetic data based on real datasets.
    
    This class provides a simple interface to the synthetic data generation
    functionality, with configurable parameters for different generation methods,
    quality metrics, and memory optimization.
    
    This class has been refactored to delegate responsibilities to specialized components.
    
    Example:
        from synthetic import Synthesize
        
        # Generate synthetic data with default parameters
        synthetic_df = Synthesize(
            dataset=my_dataset,
            method='gaussian',
            num_samples=1000,
            random_state=42,
            print_metrics=True
        )
        
        # Access the synthetic data and quality metrics
        synthetic_data = synthetic_df.data
        quality_metrics = synthetic_df.metrics
    """
    
    def save_report(self, output_path: t.Union[str, Path], **kwargs) -> str:
        """
        Generate and save an HTML report analyzing the synthetic data quality.
        
        Args:
            output_path: Path where the HTML report should be saved
            **kwargs: Additional parameters for report customization
            
        Returns:
            Path to the generated HTML report
        """
        if not hasattr(self, 'data') or len(self.data) == 0:
            raise ValueError("No synthetic data available to generate report")
        
        # Create generator info
        generator_info = f"Method: {self.method}, Samples: {self.num_samples}, Random State: {self.random_state}"
        
        # Generate the report
        report_path = self.reporter.generate_report(
            original_data=self.original_data,
            synthetic_data=self.data,
            metrics=self.metrics if hasattr(self, 'metrics') and self.metrics else {},
            generator_info=generator_info,
            report_path=output_path,
            include_data_samples=kwargs.get('include_data_samples', True),
            report_format='html',
            include_visualizations=kwargs.get('include_visualizations', True),
            **kwargs
        )
        
        return report_path
    
    def overall_quality(self) -> float:
        """
        Get the overall quality score of the synthetic data.
        
        Returns:
            Float between 0 and 1, where higher indicates better quality
        """
        return self.metrics_calculator.overall_quality()
    
    def resample(self, num_samples: int = None, **kwargs) -> pd.DataFrame:
        """
        Generate a new batch of synthetic data without refitting the model.
        
        Args:
            num_samples: Number of samples to generate (defaults to original amount)
            **kwargs: Additional generation parameters
            
        Returns:
            DataFrame with newly generated synthetic data
        """
        if not hasattr(self, 'original_data') or self.original_data is None:
            raise ValueError("No original data available for resampling")
            
        # Use original number of samples if not specified
        if num_samples is None:
            num_samples = self.num_samples
            
        self.log(f"Resampling {num_samples} synthetic samples...")
        
        # Get original target and features
        data, target_column, categorical_features, numerical_features = self.data_processor.process_dataset(self.dataset)
        
        # Generate new synthetic data
        synthetic_data, _ = self.data_generator.generate_data(
            data=data,
            target_column=target_column,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            num_samples=num_samples,
            chunk_size=self.data_processor.chunk_size,
            dask_manager=self.dask_manager if self.use_dask else None
        )
        
        # Apply similarity filtering if needed
        if self.data_processor.similarity_threshold is not None:
            synthetic_data = self.data_processor.apply_similarity_filtering(
                data, 
                synthetic_data, 
                self.dask_manager if self.use_dask else None, 
                self.n_jobs
            )
            
        return synthetic_data
    
    def __repr__(self):
        """String representation of the object."""
        status = "completed" if hasattr(self, 'data') and len(self.data) > 0 else "not completed"
        sample_count = len(self.data) if hasattr(self, 'data') else 0
        quality_score = f", quality={self.overall_quality():.4f}" if hasattr(self, 'metrics_calculator') else ""
        return f"Synthesize(method='{self.method}', samples={sample_count}{quality_score}, {status})"
        
    def __init__(
        self,
        dataset: t.Any,
        method: str = 'gaussian',
        num_samples: int = 1000,
        random_state: t.Optional[int] = None,
        chunk_size: t.Optional[int] = None,
        similarity_threshold: t.Optional[float] = None,
        return_quality_metrics: bool = False,
        print_metrics: bool = True,
        verbose: bool = True,
        generate_report: bool = False,
        report_path: t.Optional[t.Union[str, Path]] = None,
        fit_sample_size: int = 5000,
        n_jobs: int = -1,
        memory_limit_percentage: float = 70.0,
        use_dask: bool = True,
        dask_temp_directory: t.Optional[str] = None,
        dask_n_workers: t.Optional[int] = None,
        dask_threads_per_worker: int = 2,
        **kwargs
    ):
        """
        Initialize and run the synthetic data generation process.
        
        Args:
            dataset: A DBDataset or a pandas DataFrame
            method: Method to use for generation ('gaussian', 'ctgan', etc.)
            num_samples: Number of synthetic samples to generate
            random_state: Seed for reproducibility
            chunk_size: Size of chunks for memory-efficient generation
            similarity_threshold: Threshold for filtering similar samples (0.0-1.0)
            return_quality_metrics: Whether to calculate and return quality metrics
            print_metrics: Whether to print quality metrics summary
            verbose: Whether to print progress information
            generate_report: Whether to generate a detailed quality report
            report_path: Path to save the generated report
            fit_sample_size: Maximum number of samples to use for fitting the model
            n_jobs: Number of parallel jobs (-1 uses all cores)
            memory_limit_percentage: Maximum memory usage percentage
            use_dask: Whether to use Dask for distributed processing
            dask_temp_directory: Directory for Dask to store temporary files
            dask_n_workers: Number of Dask workers (None = auto)
            dask_threads_per_worker: Number of threads per Dask worker
            **kwargs: Additional parameters for the specific generator
        """
        # Store configuration parameters
        self.dataset = dataset
        self.method = method
        self.num_samples = num_samples
        self.random_state = random_state
        self.return_quality_metrics = return_quality_metrics
        self.print_metrics = print_metrics
        self.verbose = verbose
        self.generate_report = generate_report
        self.report_path = report_path
        self.fit_sample_size = fit_sample_size
        self.n_jobs = n_jobs
        self.memory_limit_percentage = memory_limit_percentage
        self.use_dask = use_dask
        self.kwargs = kwargs
        
        # Initialize components
        self.dask_manager = DaskManager(
            use_dask=use_dask,
            dask_temp_directory=dask_temp_directory,
            dask_n_workers=dask_n_workers,
            dask_threads_per_worker=dask_threads_per_worker,
            memory_limit_percentage=memory_limit_percentage,
            verbose=verbose
        )
        
        self.data_processor = SyntheticDataProcessor(
            verbose=verbose,
            memory_limit_percentage=memory_limit_percentage,
            similarity_threshold=similarity_threshold,
            chunk_size=chunk_size,
            random_state=random_state
        )
        
        self.data_generator = DataGenerator(
            method=method,
            random_state=random_state,
            verbose=verbose,
            fit_sample_size=fit_sample_size,
            n_jobs=n_jobs,
            memory_limit_percentage=memory_limit_percentage,
            **kwargs
        )
        
        self.metrics_calculator = MetricsCalculator(
            verbose=verbose,
            random_state=random_state
        )
        
        self.reporter = SyntheticReporter(
            verbose=verbose
        )
        
        # Initialize data placeholders
        self.metrics = None
        self.report_file = None
        self.data = pd.DataFrame()  # Initialize with empty DataFrame
        
        # Initialize Dask client if using Dask
        if self.use_dask:
            self.dask_manager.initialize_client()
        
        # Generate the synthetic data
        try:
            self._generate()
        except Exception as e:
            print(f"Error during synthetic data generation: {str(e)}")
            print(traceback.format_exc())
            # Data remains as empty DataFrame
            raise
        finally:
            # Close Dask client if it was created
            self.dask_manager.close_client()
            
    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def _generate(self):
        """Run the synthetic data generation process with memory monitoring."""
        # Process input dataset
        data, target_column, categorical_features, numerical_features = self.data_processor.process_dataset(self.dataset)
        
        # Store original data for metrics calculation
        self.original_data = data
        
        # Determine optimal chunk size if not specified
        chunk_size = self.data_processor.determine_optimal_chunk_size(data)
        
        # Generate synthetic data
        synthetic_data, generator = self.data_generator.generate_data(
            data=data,
            target_column=target_column,
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            num_samples=self.num_samples,
            chunk_size=chunk_size,
            dask_manager=self.dask_manager if self.use_dask else None
        )
        
        # Apply similarity filtering if threshold is provided
        if self.data_processor.similarity_threshold is not None:
            pre_filter_count = len(synthetic_data)
            synthetic_data = self.data_processor.apply_similarity_filtering(
                data, 
                synthetic_data, 
                self.dask_manager if self.use_dask else None, 
                self.n_jobs
            )
            self.log(f"Applied similarity filtering: {pre_filter_count} → {len(synthetic_data)} samples")
        
        # Store the generated synthetic data
        self.data = synthetic_data
        
        # Calculate quality metrics if requested
        if self.return_quality_metrics or self.print_metrics or self.generate_report:
            self.metrics = self.metrics_calculator.calculate_metrics(
                original_data=data,
                synthetic_data=synthetic_data,
                generator=generator,
                dask_manager=self.dask_manager if self.use_dask else None
            )
            
            # Print metrics if requested
            if self.print_metrics:
                self.metrics_calculator.print_summary()
            
            # Generate report if requested
            if self.generate_report:
                self.reporter.generate_report(
                    original_data=data,
                    synthetic_data=synthetic_data,
                    metrics=self.metrics,
                    generator_info=str(generator),
                    report_path=self.report_path
                )
            
        # Clean up to free memory
        self.data_processor.clean_memory()
        
        self.log("Synthetic data generation completed successfully")