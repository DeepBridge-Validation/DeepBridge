import pandas as pd
import numpy as np
import typing as t
from pathlib import Path
import gc
import psutil
import warnings
from datetime import datetime
import traceback

class Synthesize:
    """
    A unified interface for generating synthetic data based on real datasets.
    
    This class provides a simple interface to the synthetic data generation
    functionality, with configurable parameters for different generation methods,
    quality metrics, and memory optimization.
    
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
        
        # Import report generator here to avoid circular imports
        from .reports.report_generator import generate_quality_report
        
        # Create generator info
        generator_info = f"Method: {self.method}, Samples: {self.num_samples}, Random State: {self.random_state}"
        
        # Generate the report
        report_path = generate_quality_report(
            real_data=self.original_data,
            synthetic_data=self.data,
            quality_metrics=self.metrics if hasattr(self, 'metrics') and self.metrics else {},
            report_path=output_path,
            generator_info=generator_info,
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
        if hasattr(self, 'metrics_calculator') and self.metrics_calculator is not None:
            return self.metrics_calculator.overall_quality()
        elif hasattr(self, 'metrics') and self.metrics and 'overall' in self.metrics:
            return self.metrics['overall'].get('quality_score', 0.0)
        return 0.0
    
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
        
        # Initialize a new generator with the same parameters
        generator = self._initialize_generator()
        
        # Get original target and features
        data, target_column, categorical_features, numerical_features = self._process_dataset()
        
        # Fit the generator
        generator.fit(
            data=data,
            target_column=target_column,
            categorical_columns=categorical_features,
            numerical_columns=numerical_features,
            max_fit_samples=self.fit_sample_size,
            **self.kwargs
        )
        
        # Generate new synthetic data
        synthetic_data = generator.generate(
            num_samples=num_samples,
            chunk_size=self.chunk_size,
            memory_efficient=True,
            dynamic_chunk_sizing=True,
            **kwargs
        )
        
        # Apply similarity filtering if needed
        if self.similarity_threshold is not None:
            synthetic_data = self._apply_similarity_filtering(data, synthetic_data)
            
        return synthetic_data
    
    def __repr__(self):
        """String representation of the object."""
        status = "completed" if hasattr(self, 'data') and len(self.data) > 0 else "not completed"
        sample_count = len(self.data) if hasattr(self, 'data') else 0
        quality_score = f", quality={self.overall_quality():.4f}" if hasattr(self, 'metrics') else ""
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
            **kwargs: Additional parameters for the specific generator
        """
        self.dataset = dataset
        self.method = method
        self.num_samples = num_samples
        self.random_state = random_state
        self.chunk_size = chunk_size
        self.similarity_threshold = similarity_threshold
        self.return_quality_metrics = return_quality_metrics
        self.print_metrics = print_metrics
        self.verbose = verbose
        self.generate_report = generate_report
        self.report_path = report_path
        self.fit_sample_size = fit_sample_size
        self.n_jobs = n_jobs
        self.memory_limit_percentage = memory_limit_percentage
        self.kwargs = kwargs
        
        # Memory management
        self._total_system_memory = psutil.virtual_memory().total
        self._memory_limit = (self.memory_limit_percentage / 100.0) * self._total_system_memory
        
        if self.verbose:
            print(f"System memory: {self._total_system_memory / (1024**3):.2f} GB")
            print(f"Memory limit: {self._memory_limit / (1024**3):.2f} GB ({memory_limit_percentage}%)")
        
        # Initialize metrics and data placeholders
        self.metrics = None
        self.metrics_calculator = None
        self.report_file = None
        self.data = pd.DataFrame()  # Initialize with empty DataFrame
        
        # Generate the synthetic data
        try:
            self._generate()
        except Exception as e:
            print(f"Error during synthetic data generation: {str(e)}")
            print(traceback.format_exc())
            # Data remains as empty DataFrame
            raise
            
    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
        
    def _process_dataset(self):
        """Process the input dataset to extract necessary information."""
        # Handle input dataset - could be DBDataset or pandas DataFrame
        if hasattr(self.dataset, 'X') and hasattr(self.dataset, 'target') and hasattr(self.dataset, 'target_name'):
            # This is a DBDataset
            self.log("Using DBDataset as input")
            
            data = pd.concat([
                self.dataset.X, 
                self.dataset.target.to_frame(name=self.dataset.target_name)
            ], axis=1)
            
            target_column = self.dataset.target_name
            categorical_features = (self.dataset.categorical_features 
                                  if hasattr(self.dataset, 'categorical_features') else None)
            numerical_features = (self.dataset.numerical_features 
                                if hasattr(self.dataset, 'numerical_features') else None)
        
        elif isinstance(self.dataset, pd.DataFrame):
            # This is a pandas DataFrame
            self.log("Using pandas DataFrame as input")
            
            data = self.dataset
            target_column = self.kwargs.get('target_column')
            categorical_features = self.kwargs.get('categorical_features')
            numerical_features = self.kwargs.get('numerical_features')
        
        else:
            raise ValueError("Dataset must be either a DBDataset or a pandas DataFrame")
            
        # Log dataset information
        self.log(f"Dataset shape: {data.shape}")
        self.log(f"Target column: {target_column}")
        self.log(f"Missing values: {data.isna().sum().sum()} ({data.isna().sum().sum() / data.size:.2%})")
        
        # Ensure categorical and numerical features are valid
        if categorical_features is None and numerical_features is None:
            self.log("No feature types specified. Will be inferred by the generator.")
        
        return data, target_column, categorical_features, numerical_features
    
    def _initialize_generator(self):
        """Initialize the appropriate generator based on the chosen method."""
        from .methods.gaussian_copula import GaussianCopulaGenerator
        
        if self.method.lower() == 'gaussian':
            return GaussianCopulaGenerator(
                random_state=self.random_state,
                preserve_dtypes=self.kwargs.get('preserve_dtypes', True),
                preserve_constraints=self.kwargs.get('preserve_constraints', True),
                verbose=self.verbose,
                fit_sample_size=self.fit_sample_size,
                n_jobs=self.n_jobs,
                memory_limit_percentage=self.memory_limit_percentage
            )
        # Add other methods as they are implemented
        # elif self.method.lower() == 'ctgan':
        #     from .methods.future_methods.ctgan import CTGANGenerator
        #     return CTGANGenerator(...)
        else:
            raise ValueError(f"Unknown method: {self.method}. Supported methods: 'gaussian'")
    
    def _apply_similarity_filtering(self, original_data, synthetic_data):
        """Apply similarity filtering to remove too-similar samples."""
        from .metrics.similarity import filter_by_similarity
        
        self.log(f"Filtering synthetic data with similarity threshold: {self.similarity_threshold}")
        
        original_count = len(synthetic_data)
        filtered_data = filter_by_similarity(
            original_data=original_data,
            synthetic_data=synthetic_data,
            threshold=self.similarity_threshold,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose
        )
        
        removed_count = original_count - len(filtered_data)
        removed_percentage = removed_count / original_count * 100 if original_count > 0 else 0
        self.log(f"Removed {removed_count} samples ({removed_percentage:.2f}%) with similarity ≥ {self.similarity_threshold}")
        
        return filtered_data
    
    def _calculate_quality_metrics(self, original_data, synthetic_data, generator):
        """Calculate quality metrics for the synthetic data."""
        from .metrics.synthetic_metrics import SyntheticMetrics
        
        self.log("Calculating quality metrics...")
        
        # Check memory before metrics calculation
        pre_metrics_memory = psutil.Process().memory_info().rss
        self.log(f"Memory usage before metrics calculation: {pre_metrics_memory / (1024**3):.2f} GB")
        
        try:
            # Limit sample size for metrics calculation to avoid memory issues
            max_metrics_samples = min(5000, len(original_data), len(synthetic_data))
            
            # Use SyntheticMetrics class to calculate comprehensive metrics
            metrics_calculator = SyntheticMetrics(
                real_data=original_data,
                synthetic_data=synthetic_data,
                numerical_columns=generator.numerical_columns,
                categorical_columns=generator.categorical_columns,
                target_column=generator.target_column,
                sample_size=max_metrics_samples,
                random_state=self.random_state,
                verbose=self.verbose
            )
            
            # Store the metrics instance and get the metrics dictionary
            self.metrics_calculator = metrics_calculator
            self.metrics = metrics_calculator.get_metrics()
            
            # Print metrics if requested
            if self.print_metrics:
                metrics_calculator.print_summary()
            
            # Generate report if requested
            if self.generate_report:
                self._generate_quality_report(original_data, synthetic_data, metrics_calculator, generator)
                
            # Check memory after metrics calculation
            post_metrics_memory = psutil.Process().memory_info().rss
            self.log(f"Memory usage after metrics calculation: {post_metrics_memory / (1024**3):.2f} GB")
            self.log(f"Memory increase: {(post_metrics_memory - pre_metrics_memory) / (1024**3):.2f} GB")
            
        except Exception as e:
            self.log(f"Error calculating quality metrics: {str(e)}")
            self.log(traceback.format_exc())
            self.metrics = {"error": str(e)}
    
    def _generate_quality_report(self, original_data, synthetic_data, metrics_calculator, generator):
        """Generate a quality report for the synthetic data."""
        from .reports.report_generator import generate_quality_report
        
        self.log("Generating quality report...")
        
        try:
            report_file = generate_quality_report(
                real_data=original_data,
                synthetic_data=synthetic_data,
                quality_metrics=metrics_calculator.get_metrics(),
                report_path=self.report_path,
                generator_info=generator.__repr__(),
                include_data_samples=self.kwargs.get('include_data_samples', True),
                report_format=self.kwargs.get('report_format', 'html'),
                include_visualizations=self.kwargs.get('include_visualizations', True),
                **self.kwargs
            )
            
            # Store the report path
            self.report_file = report_file
            
            self.log(f"Quality report generated: {report_file}")
        except Exception as e:
            self.log(f"Error generating quality report: {str(e)}")
            self.log(traceback.format_exc())
    
    def _generate(self):
        """Run the synthetic data generation process with memory monitoring."""
        self.log(f"Starting synthetic data generation using {self.method} method")
        self.log(f"Generating {self.num_samples} synthetic samples")
        
        # Monitor initial memory usage
        initial_memory = psutil.Process().memory_info().rss
        self.log(f"Initial memory usage: {initial_memory / (1024**3):.2f} GB")
        
        # Process input dataset - could be DBDataset or pandas DataFrame
        data, target_column, categorical_features, numerical_features = self._process_dataset()
        
        # Store original data for metrics calculation
        self.original_data = data
        
        # Check if we need to dynamically determine chunk size
        if self.chunk_size is None:
            estimated_row_size = data.memory_usage(deep=True).sum() / len(data)
            available_memory = 0.6 * self._memory_limit  # Use 60% of memory limit
            suggested_chunk_size = int(available_memory / estimated_row_size / 2)  # Divide by 2 for safety
            
            # Set a reasonable min/max
            self.chunk_size = max(min(suggested_chunk_size, 10000), 500)
            self.log(f"Dynamically set chunk size to {self.chunk_size} based on available memory")
        
        # Initialize the generator based on the chosen method
        generator = self._initialize_generator()
        
        # Fit the generator
        self.log("Fitting the generator...")
        
        try:
            # Clear memory before fitting
            gc.collect()
            
            # Fit the model
            generator.fit(
                data=data,
                target_column=target_column,
                categorical_columns=categorical_features,
                numerical_columns=numerical_features,
                max_fit_samples=self.fit_sample_size,
                **self.kwargs
            )
        except Exception as e:
            self.log(f"Error during model fitting: {str(e)}")
            raise RuntimeError(f"Failed to fit model: {str(e)}")
        
        # Generate synthetic data
        self.log("Generating synthetic data...")
        
        try:
            # Monitor memory before generation
            pre_gen_memory = psutil.Process().memory_info().rss
            self.log(f"Memory usage before generation: {pre_gen_memory / (1024**3):.2f} GB")
            
            # Generate data with memory-efficient options
            synthetic_data = generator.generate(
                num_samples=self.num_samples,
                chunk_size=self.chunk_size,
                memory_efficient=True,
                dynamic_chunk_sizing=True,
                post_process_method='enhanced',
                **self.kwargs
            )
            
            # Apply similarity filtering if threshold is provided
            if self.similarity_threshold is not None:
                pre_filter_count = len(synthetic_data)
                synthetic_data = self._apply_similarity_filtering(data, synthetic_data)
                self.log(f"Applied similarity filtering: {pre_filter_count} → {len(synthetic_data)} samples")
            
            # Monitor memory after generation
            post_gen_memory = psutil.Process().memory_info().rss
            self.log(f"Memory usage after generation: {post_gen_memory / (1024**3):.2f} GB")
            self.log(f"Memory increase: {(post_gen_memory - pre_gen_memory) / (1024**3):.2f} GB")
            
            self.log(f"Generated {len(synthetic_data)} synthetic samples")
            
            # Store the generated synthetic data
            self.data = synthetic_data
            
            # Calculate quality metrics if requested
            if self.return_quality_metrics or self.print_metrics or self.generate_report:
                self._calculate_quality_metrics(data, synthetic_data, generator)
                
            # Clean up to free memory
            gc.collect()
            
            self.log("Synthetic data generation completed successfully")
        
        except Exception as e:
            self.log(f"Error during synthetic data generation: {str(e)}")
            self.log(traceback.format_exc())
            raise