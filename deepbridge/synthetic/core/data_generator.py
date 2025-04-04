"""
Module for generating synthetic data.
"""

import pandas as pd
import typing as t
import psutil
import gc

class DataGenerator:
    """
    Generates synthetic data from real datasets.
    Extracted from Synthesize class to separate data generation responsibilities.
    """
    
    def __init__(
        self,
        method: str = 'gaussian',
        random_state: t.Optional[int] = None,
        verbose: bool = True,
        fit_sample_size: int = 5000,
        n_jobs: int = -1,
        memory_limit_percentage: float = 70.0,
        **kwargs
    ):
        """
        Initialize the data generator.
        
        Args:
            method: Method to use for generation ('gaussian', 'ctgan', etc.)
            random_state: Seed for reproducibility
            verbose: Whether to print progress information
            fit_sample_size: Maximum number of samples to use for fitting the model
            n_jobs: Number of parallel jobs (-1 uses all cores)
            memory_limit_percentage: Maximum memory usage percentage
            **kwargs: Additional parameters for the specific generator
        """
        self.method = method
        self.random_state = random_state
        self.verbose = verbose
        self.fit_sample_size = fit_sample_size
        self.n_jobs = n_jobs
        self.memory_limit_percentage = memory_limit_percentage
        self.kwargs = kwargs
        
        # Memory management
        self._total_system_memory = psutil.virtual_memory().total
        self._memory_limit = (self.memory_limit_percentage / 100.0) * self._total_system_memory
    
    def initialize_generator(self, dask_manager=None) -> t.Any:
        """
        Initialize the appropriate generator based on the chosen method.
        
        Args:
            dask_manager: Optional DaskManager for distributed computation
            
        Returns:
            Generator instance
        """
        from deepbridge.synthetic.methods.gaussian_copula import GaussianCopulaGenerator
        
        # Determine if we should use Dask
        use_dask = dask_manager is not None and dask_manager.is_active
        
        # Get Dask parameters if available
        dask_client = dask_manager.client if use_dask else None
        dask_temp_directory = dask_manager.dask_temp_directory if use_dask else None
        dask_n_workers = dask_manager.dask_n_workers if use_dask else None
        dask_threads_per_worker = dask_manager.dask_threads_per_worker if use_dask else 2
        
        if self.method.lower() == 'gaussian':
            return GaussianCopulaGenerator(
                random_state=self.random_state,
                preserve_dtypes=self.kwargs.get('preserve_dtypes', True),
                preserve_constraints=self.kwargs.get('preserve_constraints', True),
                verbose=self.verbose,
                fit_sample_size=self.fit_sample_size,
                n_jobs=self.n_jobs,
                memory_limit_percentage=self.memory_limit_percentage,
                use_dask=use_dask,
                dask_temp_directory=dask_temp_directory,
                dask_n_workers=dask_n_workers,
                dask_threads_per_worker=dask_threads_per_worker
            )
        # Add other methods as they are implemented
        # elif self.method.lower() == 'ctgan':
        #     from deepbridge.synthetic.methods.future_methods.ctgan import CTGANGenerator
        #     return CTGANGenerator(...)
        else:
            raise ValueError(f"Unknown method: {self.method}. Supported methods: 'gaussian'")
    
    def generate_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        categorical_features: t.Optional[t.List[str]],
        numerical_features: t.Optional[t.List[str]],
        num_samples: int,
        chunk_size: int,
        dask_manager=None
    ) -> t.Tuple[pd.DataFrame, t.Any]:
        """
        Generate synthetic data using the specified method.
        
        Args:
            data: Original dataset
            target_column: Name of the target column
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            num_samples: Number of samples to generate
            chunk_size: Size of chunks for memory-efficient generation
            dask_manager: Optional DaskManager for distributed computation
            
        Returns:
            Tuple of synthetic data DataFrame and generator instance
        """
        self.log(f"Starting synthetic data generation using {self.method} method")
        self.log(f"Generating {num_samples} synthetic samples")
        
        # Monitor initial memory usage
        initial_memory = psutil.Process().memory_info().rss
        self.log(f"Initial memory usage: {initial_memory / (1024**3):.2f} GB")
        
        # Initialize the generator based on the chosen method
        generator = self.initialize_generator(dask_manager)
        
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
                num_samples=num_samples,
                chunk_size=chunk_size,
                memory_efficient=True,
                dynamic_chunk_sizing=True,
                post_process_method='enhanced',
                **self.kwargs
            )
            
            # Monitor memory after generation
            post_gen_memory = psutil.Process().memory_info().rss
            self.log(f"Memory usage after generation: {post_gen_memory / (1024**3):.2f} GB")
            self.log(f"Memory increase: {(post_gen_memory - pre_gen_memory) / (1024**3):.2f} GB")
            
            self.log(f"Generated {len(synthetic_data)} synthetic samples")
            
            # Clean up to free memory
            gc.collect()
            
            return synthetic_data, generator
        
        except Exception as e:
            self.log(f"Error during synthetic data generation: {str(e)}")
            import traceback
            self.log(traceback.format_exc())
            raise
    
    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)