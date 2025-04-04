"""
Module for processing and manipulating data for synthetic data generation.
"""

import pandas as pd
import numpy as np
import typing as t
from pathlib import Path
import psutil
import gc

class SyntheticDataProcessor:
    """
    Processes and manipulates data for synthetic data generation.
    Extracted from Synthesize class to separate data processing responsibilities.
    """
    
    def __init__(
        self,
        verbose: bool = True,
        memory_limit_percentage: float = 70.0,
        similarity_threshold: t.Optional[float] = None,
        chunk_size: t.Optional[int] = None,
        random_state: t.Optional[int] = None
    ):
        """
        Initialize the data processor.
        
        Args:
            verbose: Whether to print progress information
            memory_limit_percentage: Maximum memory usage percentage
            similarity_threshold: Threshold for filtering similar samples (0.0-1.0)
            chunk_size: Size of chunks for memory-efficient generation
            random_state: Seed for reproducibility
        """
        self.verbose = verbose
        self.memory_limit_percentage = memory_limit_percentage
        self.similarity_threshold = similarity_threshold
        self.chunk_size = chunk_size
        self.random_state = random_state
        
        # Memory management
        self._total_system_memory = psutil.virtual_memory().total
        self._memory_limit = (self.memory_limit_percentage / 100.0) * self._total_system_memory
    
    def process_dataset(self, dataset: t.Any) -> t.Tuple[pd.DataFrame, str, t.Optional[t.List[str]], t.Optional[t.List[str]]]:
        """
        Process the input dataset to extract necessary information.
        
        Args:
            dataset: A DBDataset or a pandas DataFrame
            
        Returns:
            Tuple[DataFrame, str, List[str], List[str]]: 
                Data, target column, categorical features, numerical features
        """
        # Handle input dataset - could be DBDataset or pandas DataFrame
        if hasattr(dataset, 'X') and hasattr(dataset, 'target') and hasattr(dataset, 'target_name'):
            # This is a DBDataset
            self.log("Using DBDataset as input")
            
            data = pd.concat([
                dataset.X, 
                dataset.target.to_frame(name=dataset.target_name)
            ], axis=1)
            
            target_column = dataset.target_name
            categorical_features = (dataset.categorical_features 
                                 if hasattr(dataset, 'categorical_features') else None)
            numerical_features = (dataset.numerical_features 
                               if hasattr(dataset, 'numerical_features') else None)
        
        elif isinstance(dataset, pd.DataFrame):
            # This is a pandas DataFrame
            self.log("Using pandas DataFrame as input")
            
            data = dataset
            target_column = None  # Will need to be provided externally
            categorical_features = None
            numerical_features = None
        
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
    
    def apply_similarity_filtering(
        self, 
        original_data: pd.DataFrame, 
        synthetic_data: pd.DataFrame,
        dask_manager=None,
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        Apply similarity filtering to remove too-similar samples.
        
        Args:
            original_data: Original dataset
            synthetic_data: Synthetic dataset to filter
            dask_manager: Optional DaskManager for distributed computation
            n_jobs: Number of parallel jobs (-1 uses all cores)
            
        Returns:
            Filtered synthetic dataset
        """
        from deepbridge.synthetic.metrics.similarity.similarity_core import filter_by_similarity
        
        self.log(f"Filtering synthetic data with similarity threshold: {self.similarity_threshold}")
        
        original_count = len(synthetic_data)
        
        # Check if we have a DaskManager and if it's active
        use_dask = dask_manager is not None and dask_manager.is_active
        dask_client = dask_manager.client if use_dask else None
        
        # If using Dask and the datasets are large, convert to Dask DataFrames
        if use_dask and len(original_data) > 10000 and len(synthetic_data) > 10000:
            # Convert to Dask DataFrames for more efficient processing
            try:
                self.log("Using Dask for similarity filtering")
                
                # Filter with Dask support
                filtered_data = filter_by_similarity(
                    original_data=original_data,  # Keep original as pandas for reference calculations
                    synthetic_data=synthetic_data,  # Keep synthetic as pandas for filtering logic
                    threshold=self.similarity_threshold,
                    random_state=self.random_state,
                    n_jobs=n_jobs,
                    verbose=self.verbose,
                    use_dask=True,
                    dask_client=dask_client
                )
                
            except Exception as e:
                self.log(f"Error using Dask for similarity filtering: {str(e)}. Falling back to standard method.")
                filtered_data = filter_by_similarity(
                    original_data=original_data,
                    synthetic_data=synthetic_data,
                    threshold=self.similarity_threshold,
                    random_state=self.random_state,
                    n_jobs=n_jobs,
                    verbose=self.verbose
                )
        else:
            # Use standard similarity filtering
            filtered_data = filter_by_similarity(
                original_data=original_data,
                synthetic_data=synthetic_data,
                threshold=self.similarity_threshold,
                random_state=self.random_state,
                n_jobs=n_jobs,
                verbose=self.verbose
            )
        
        removed_count = original_count - len(filtered_data)
        removed_percentage = removed_count / original_count * 100 if original_count > 0 else 0
        self.log(f"Removed {removed_count} samples ({removed_percentage:.2f}%) with similarity ≥ {self.similarity_threshold}")
        
        return filtered_data
    
    def determine_optimal_chunk_size(self, data: pd.DataFrame) -> int:
        """
        Determine optimal chunk size for memory-efficient processing.
        
        Args:
            data: Data to analyze for chunk sizing
            
        Returns:
            Optimal chunk size
        """
        if self.chunk_size is not None:
            return self.chunk_size
        
        estimated_row_size = data.memory_usage(deep=True).sum() / len(data)
        available_memory = 0.6 * self._memory_limit  # Use 60% of memory limit
        suggested_chunk_size = int(available_memory / estimated_row_size / 2)  # Divide by 2 for safety
        
        # Set a reasonable min/max
        optimal_chunk_size = max(min(suggested_chunk_size, 10000), 500)
        self.log(f"Dynamically set chunk size to {optimal_chunk_size} based on available memory")
        
        self.chunk_size = optimal_chunk_size
        return optimal_chunk_size
    
    def clean_memory(self) -> None:
        """Force garbage collection to free memory."""
        gc.collect()
    
    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)