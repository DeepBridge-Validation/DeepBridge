import numpy as np
import pandas as pd
import typing as t
import gc
import psutil
import warnings
from tqdm.auto import tqdm

# Import Dask for parallel processing
import dask
import dask.dataframe as dd
import dask.array as da
from dask.distributed import Client, wait, progress

# Import GaussianMultivariate from copulas
from copulas.multivariate import GaussianMultivariate

from ..base import BaseSyntheticGenerator

class GaussianCopulaGenerator(BaseSyntheticGenerator):
    """
    Synthetic data generator using Gaussian Copula method.
    
    This generator models the dependencies between variables using a Gaussian copula,
    which is a statistical model that represents a multivariate distribution by
    capturing the dependence structure using Gaussian functions.
    
    Uses the copulas.multivariate.GaussianMultivariate implementation with optimized
    memory management and Dask for better performance and scalability with large datasets.
    """
    
    def __init__(
        self,
        random_state: t.Optional[int] = None,
        preserve_dtypes: bool = True,
        preserve_constraints: bool = True,
        verbose: bool = True,
        fit_sample_size: int = 10000,
        n_jobs: int = -1,
        memory_limit_percentage: float = 70.0,
        use_dask: bool = True,
        dask_temp_directory: t.Optional[str] = None,
        dask_n_workers: t.Optional[int] = None,
        dask_threads_per_worker: int = 2
    ):
        """
        Initialize the Gaussian Copula generator.
        
        Args:
            random_state: Seed for random number generation to ensure reproducibility
            preserve_dtypes: Whether to preserve the original data types in synthetic data
            preserve_constraints: Whether to enforce constraints from original data
            verbose: Whether to print progress and information during processing
            fit_sample_size: Maximum number of samples to use for fitting the model
            n_jobs: Number of parallel jobs for processing chunks (-1 uses all cores)
            memory_limit_percentage: Maximum memory usage as percentage of system memory
            use_dask: Whether to use Dask for distributed computing
            dask_temp_directory: Directory for Dask to store temporary files
            dask_n_workers: Number of Dask workers to use (None = auto)
            dask_threads_per_worker: Number of threads per Dask worker
        """
        super().__init__(
            random_state=random_state,
            preserve_dtypes=preserve_dtypes,
            preserve_constraints=preserve_constraints,
            verbose=verbose,
            use_dask=use_dask,
            dask_temp_directory=dask_temp_directory,
            dask_n_workers=dask_n_workers,
            dask_threads_per_worker=dask_threads_per_worker,
            memory_limit_percentage=memory_limit_percentage
        )
        self.fit_sample_size = fit_sample_size
        self.original_data_sample = None
        self.copula_model = None
        self.dtypes = {}
        self.n_jobs = n_jobs
        self.memory_limit_percentage = memory_limit_percentage

    # This function is extracted from _generate_with_dask for serialization
    def _process_chunk_for_dask(self, size, chunk_id):
        """
        Generate and process a chunk of synthetic data.
        
        Args:
            size: Size of the chunk
            chunk_id: ID of the chunk
            
        Returns:
            Processed chunk of synthetic data
        """
        if self.verbose:
            print(f"Generating chunk {chunk_id+1} with {size} samples")
        
        # Generate samples for this chunk
        chunk_df = self._generate_copula_samples(size)
        
        # Apply post-processing
        chunk_df = self._post_process_chunk(chunk_df)
        
        return chunk_df
    
    def fit(
        self,
        data: pd.DataFrame,
        target_column: t.Optional[str] = None,
        categorical_columns: t.Optional[t.List[str]] = None,
        numerical_columns: t.Optional[t.List[str]] = None,
        **kwargs
    ) -> None:
        """
        Fit the Gaussian Copula generator to the input data.
        
        Args:
            data: The dataset to fit the generator on
            target_column: The name of the target variable column
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            **kwargs: Additional parameters specific to the implementation
                - max_fit_samples: Maximum number of samples to use for fitting (default: self.fit_sample_size)
                - stratify_by: Column to use for stratified sampling during fitting
        """
        self.log(f"Fitting Gaussian Copula generator on dataset with {len(data)} rows...")
        
        # Store target column name
        self.target_column = target_column
        
        # Get stratification column if provided
        stratify_by = kwargs.get('stratify_by', None)
        stratify = data[stratify_by] if stratify_by and stratify_by in data.columns else None
        
        # Determine the number of samples to use for fitting
        max_fit_samples = kwargs.get('max_fit_samples', self.fit_sample_size)
        if len(data) > max_fit_samples:
            self.log(f"Dataset is large ({len(data)} rows). Using {max_fit_samples} samples for fitting.")
            if stratify is not None:
                self.log(f"Using stratified sampling by column: {stratify_by}")
                fit_data = data.sample(max_fit_samples, random_state=self.random_state, stratify=stratify)
            else:
                fit_data = data.sample(max_fit_samples, random_state=self.random_state)
        else:
            fit_data = data
        
        # Store a small sample of original data for validation and constraint enforcement
        sample_size = min(1000, len(data))
        self.original_data_sample = data.sample(sample_size, random_state=self.random_state)
        
        # Validate and infer column types
        self.categorical_columns, self.numerical_columns = self._validate_columns(
            fit_data, categorical_columns, numerical_columns
        )
        
        self.log(f"Identified {len(self.categorical_columns)} categorical columns and {len(self.numerical_columns)} numerical columns")
        
        # Store original data types and ranges
        self.dtypes = {col: fit_data[col].dtype for col in fit_data.columns}
        
        # Store distribution parameters for numerical columns to use in post-processing
        self.num_column_stats = {}
        for col in self.numerical_columns:
            if col in fit_data.columns:
                col_data = fit_data[col].dropna()
                if len(col_data) > 0:
                    self.num_column_stats[col] = {
                        'min': col_data.min(),
                        'max': col_data.max(),
                        'mean': col_data.mean(),
                        'std': col_data.std(),
                        'median': col_data.median(),
                        'q1': col_data.quantile(0.25),
                        'q3': col_data.quantile(0.75)
                    }
        
        # Store categorical value distributions for post-processing
        self.cat_column_stats = {}
        for col in self.categorical_columns:
            if col in fit_data.columns:
                value_counts = fit_data[col].value_counts(normalize=True)
                self.cat_column_stats[col] = {
                    'values': value_counts.index.tolist(),
                    'frequencies': value_counts.values.tolist()
                }
        
        # Handle categorical features by encoding them
        encoded_data = self._encode_categorical_features(fit_data)
        
        try:
            # Clear memory before fitting
            gc.collect()
            
            # Initialize the copula model
            self.log("Initializing GaussianMultivariate copula model...")
            self.copula_model = GaussianMultivariate(random_state=self.random_state)
            
            # Fit the copula model
            self.log("Fitting copula model to data...")
            self.copula_model.fit(encoded_data)
            
            self.log("Copula model fitting completed successfully")
            
            # Clean up to free memory
            del encoded_data
            gc.collect()
            
        except Exception as e:
            self.log(f"Error fitting copula model: {str(e)}")
            raise RuntimeError(f"Failed to fit copula model: {str(e)}")
        
        self._is_fitted = True
        self.log("Gaussian Copula model fitting completed successfully")
    
    def generate(
        self, 
        num_samples: int = 1000,
        chunk_size: t.Optional[int] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic data based on the fitted Gaussian Copula model.
        
        Args:
            num_samples: Number of synthetic samples to generate
            chunk_size: Size of chunks to use when generating large datasets
            **kwargs: Additional parameters
                - memory_efficient: Whether to use memory-efficient generation (default: True)
                - dynamic_chunk_sizing: Adjust chunk size based on available memory (default: True)
                - post_process_method: How to post-process data ('standard', 'enhanced', or 'minimal')
            
        Returns:
            DataFrame containing the generated synthetic data
        """
        if not self._is_fitted:
            raise ValueError("Generator must be fitted before generating data")
        
        self.log(f"Generating {num_samples} synthetic samples...")
        
        # Determine post-processing method
        self.post_process_method = kwargs.get('post_process_method', 'enhanced')
        
        # Determine chunk size for memory-efficient generation
        memory_efficient = kwargs.get('memory_efficient', True)
        dynamic_chunk_sizing = kwargs.get('dynamic_chunk_sizing', True)
        
        if memory_efficient:
            # If no chunk size is provided or dynamic sizing is requested
            if chunk_size is None or dynamic_chunk_sizing:
                # Estimate memory per sample
                test_samples = min(100, num_samples)
                test_data = self._generate_copula_samples(test_samples)
                memory_per_sample = test_data.memory_usage(deep=True).sum() / test_samples
                
                # Calculate safe chunk size based on available memory
                available_memory = max(0.5 * self._memory_limit - psutil.virtual_memory().used, 
                                     0.2 * self._memory_limit)  # At least 20% of limit
                
                # Use at most 50% of available memory for chunk generation
                safe_chunk_size = int(0.5 * available_memory / memory_per_sample)
                
                # Set a reasonable minimum and maximum
                safe_chunk_size = max(min(safe_chunk_size, 10000), 100)
                
                if chunk_size is None or (dynamic_chunk_sizing and safe_chunk_size < chunk_size):
                    chunk_size = safe_chunk_size
                    
                self.log(f"Dynamically determined chunk size: {chunk_size} samples")
                
                # Clean up test data
                del test_data
                gc.collect()
            
            # Check if we should use Dask
            if self.use_dask and self._dask_client is not None:
                try:
                    return self._generate_with_dask(num_samples, chunk_size)
                except Exception as e:
                    self.log(f"Error using Dask for generation: {str(e)}")
                    self.log("Falling back to standard chunk processing")
                    return self._generate_in_chunks(num_samples, chunk_size)
            else:
                # Use standard chunk processing
                return self._generate_in_chunks(num_samples, chunk_size)
        else:
            # Generate all data at once
            return self._generate_batch(num_samples)
    
    def _generate_with_dask(self, num_samples: int, chunk_size: int) -> pd.DataFrame:
        """
        Generate synthetic data using Dask for distributed computing.
        
        Args:
            num_samples: Total number of samples to generate
            chunk_size: Size of each chunk
            
        Returns:
            DataFrame with generated synthetic data
        """
        # Calculate chunk sizes
        chunk_sizes = []
        remaining = num_samples
        
        while remaining > 0:
            size = min(chunk_size, remaining)
            chunk_sizes.append(size)
            remaining -= size
        
        self.log(f"Generating {len(chunk_sizes)} chunks with sizes: {chunk_sizes}")
        
        # Create list of tasks by using pure function (not method)
        tasks = []
        for i, size in enumerate(chunk_sizes):
            # Pass only the necessary arguments to the worker-friendly function
            tasks.append(dask.delayed(self._process_chunk_for_dask)(size, i))
        
        # Compute chunks in parallel
        self.log("Computing chunks in parallel with Dask...")
        
        # Use progress visualization if in verbose mode
        if self.verbose:
            # Create a progress bar
            with tqdm(total=len(chunk_sizes), desc="Generating chunks") as pbar:
                # Compute all tasks and gather results
                chunks = dask.compute(*tasks)
                # Update progress bar after completion
                pbar.update(len(chunk_sizes))
        else:
            # Compute without progress visualization
            chunks = dask.compute(*tasks)
        
        # Combine all chunks
        try:
            # Use pandas concat for combining the results
            result = pd.concat(chunks, ignore_index=True)
        except Exception as e:
            self.log(f"Error combining chunks: {str(e)}")
            # Attempt alternative approach if the first fails
            result = pd.DataFrame()
            for chunk in chunks:
                result = pd.concat([result, chunk], ignore_index=True)
        
        # Clean up to free memory
        del chunks
        gc.collect()
        
        self.log(f"Successfully generated {len(result)} samples using Dask")
        return result
    
    def _generate_in_chunks(self, num_samples: int, chunk_size: int) -> pd.DataFrame:
        """
        Generate synthetic data in chunks to optimize memory usage.
        
        Args:
            num_samples: Total number of samples to generate
            chunk_size: Size of each chunk
            
        Returns:
            DataFrame with generated synthetic data
        """
        # Calculate chunk sizes
        chunk_sizes = []
        remaining = num_samples
        
        while remaining > 0:
            size = min(chunk_size, remaining)
            chunk_sizes.append(size)
            remaining -= size
        
        self.log(f"Generating {len(chunk_sizes)} chunks with sizes: {chunk_sizes}")
        
        # Check if we can use parallel processing
        use_parallel = self.n_jobs != 1 and len(chunk_sizes) > 1
        
        if use_parallel:
            from joblib import Parallel, delayed
            self.log(f"Using parallel processing with {self.n_jobs} jobs")
            
            # Process chunks in parallel
            try:
                n_jobs = self.n_jobs if self.n_jobs > 0 else None  # None means all cores
                
                # Create a progress bar
                with tqdm(total=len(chunk_sizes), desc="Generating chunks", disable=not self.verbose) as pbar:
                    # Generate chunks in parallel with progress tracking
                    chunks = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
                        delayed(self._process_chunk_for_dask)(size, i) for i, size in enumerate(chunk_sizes)
                    )
                    pbar.update(len(chunk_sizes))
                
                # Combine all chunks
                result = pd.concat(chunks, ignore_index=True)
                
                # Clean up to free memory
                del chunks
                gc.collect()
                
                self.log(f"Successfully generated {len(result)} samples using parallel processing")
                return result
                
            except Exception as e:
                self.log(f"Error in parallel processing: {str(e)}")
                self.log("Falling back to sequential processing")
                use_parallel = False
        
        if not use_parallel:
            # Process chunks sequentially
            all_chunks = []
            
            with tqdm(total=len(chunk_sizes), desc="Generating chunks", disable=not self.verbose) as pbar:
                for i, size in enumerate(chunk_sizes):
                    self.log(f"Generating chunk {i+1}/{len(chunk_sizes)} with {size} samples")
                    
                    chunk_df = self._generate_copula_samples(size)
                    
                    # Apply post-processing
                    chunk_df = self._post_process_chunk(chunk_df)
                    
                    all_chunks.append(chunk_df)
                    
                    # Clean up to free memory after each chunk
                    gc.collect()
                    
                    pbar.update(1)
            
            # Combine all chunks
            result = pd.concat(all_chunks, ignore_index=True)
            
            # Clean up to free memory
            del all_chunks
            gc.collect()
            
            self.log(f"Successfully generated {len(result)} samples using sequential processing")
            return result
    
    def _post_process_chunk(self, chunk_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply post-processing to a generated chunk based on the selected method.
        
        Args:
            chunk_df: Generated data chunk
            
        Returns:
            Post-processed data chunk
        """
        # Apply constraints if required
        if self.preserve_constraints:
            chunk_df = self._enforce_constraints(chunk_df, self.original_data_sample)
        
        # Enhanced post-processing to improve data quality
        if self.post_process_method == 'enhanced':
            # Adjust numerical columns to better match original distributions
            for col in self.numerical_columns:
                if col in chunk_df.columns and col in self.num_column_stats:
                    stats = self.num_column_stats[col]
                    
                    # Handle outliers by clipping to a reasonable range
                    iqr = stats['q3'] - stats['q1']
                    lower_bound = stats['q1'] - 1.5 * iqr
                    upper_bound = stats['q3'] + 1.5 * iqr
                    
                    # Clip values but keep some variability
                    chunk_df[col] = chunk_df[col].clip(
                        lower=max(lower_bound, stats['min']),
                        upper=min(upper_bound, stats['max'])
                    )
            
            # Correct categorical distributions
            for col in self.categorical_columns:
                if col in chunk_df.columns and col in self.cat_column_stats:
                    # Only apply if current distribution deviates significantly
                    synth_dist = chunk_df[col].value_counts(normalize=True)
                    orig_dist = pd.Series(
                        self.cat_column_stats[col]['frequencies'],
                        index=self.cat_column_stats[col]['values']
                    )
                    
                    # Measure distribution difference
                    common_cats = set(synth_dist.index) & set(orig_dist.index)
                    if len(common_cats) > 0:
                        common_synth = synth_dist.loc[list(common_cats)]
                        common_orig = orig_dist.loc[list(common_cats)]
                        dist_diff = np.abs(common_synth - common_orig).mean()
                        
                        # If difference is significant, adjust the distribution
                        if dist_diff > 0.1:  # threshold for adjustment
                            self._adjust_categorical_distribution(chunk_df, col, orig_dist)
        
        # Convert dtypes back to original if required
        if self.preserve_dtypes:
            for col, dtype in self.dtypes.items():
                if col in chunk_df.columns:
                    try:
                        chunk_df[col] = chunk_df[col].astype(dtype)
                    except (ValueError, TypeError):
                        # If conversion fails, keep as is
                        pass
        
        # Optimize memory usage
        chunk_df = self._memory_optimize(chunk_df)
        
        return chunk_df
    
    def _adjust_categorical_distribution(self, df: pd.DataFrame, column: str, target_dist: pd.Series) -> None:
        """
        Adjust categorical distribution to match target distribution.
        
        Args:
            df: DataFrame to adjust
            column: Column to adjust
            target_dist: Target distribution
        """
        current_dist = df[column].value_counts(normalize=True)
        
        # For each category that needs adjustment
        for cat, target_freq in target_dist.items():
            if cat in current_dist.index:
                current_freq = current_dist[cat]
                
                # Calculate how many values need to change
                diff = target_freq - current_freq
                if abs(diff) < 0.01:  # Skip small adjustments
                    continue
                
                n_samples = len(df)
                n_changes = int(abs(diff) * n_samples)
                
                if diff > 0:  # Need to increase this category
                    # Find other categories to decrease
                    other_cats = [c for c in current_dist.index if current_dist[c] > target_dist.get(c, 0)]
                    if not other_cats:
                        continue
                        
                    # Select random samples from other categories to change
                    for _ in range(min(n_changes, 100)):  # Limit changes to avoid overfitting
                        other_cat = np.random.choice(other_cats)
                        idx = df[df[column] == other_cat].index
                        if len(idx) > 0:
                            change_idx = np.random.choice(idx)
                            df.loc[change_idx, column] = cat
                
                elif diff < 0:  # Need to decrease this category
                    # Find other categories to increase
                    other_cats = [c for c in target_dist.index 
                                if c in current_dist.index and current_dist[c] < target_dist[c]]
                    if not other_cats:
                        continue
                        
                    # Select random samples from this category to change
                    idx = df[df[column] == cat].index
                    for _ in range(min(n_changes, 100)):  # Limit changes to avoid overfitting
                        if len(idx) > 0:
                            change_idx = np.random.choice(idx)
                            other_cat = np.random.choice(other_cats)
                            df.loc[change_idx, column] = other_cat
    
    def _generate_batch(self, num_samples: int) -> pd.DataFrame:
        """
        Generate a batch of synthetic data.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            DataFrame with generated synthetic data
        """
        # Generate samples using the copula model
        synthetic_data = self._generate_copula_samples(num_samples)
        
        # Apply post-processing
        synthetic_data = self._post_process_chunk(synthetic_data)
        
        return synthetic_data
    
    def _generate_copula_samples(self, num_samples: int) -> pd.DataFrame:
        """
        Generate samples using the fitted copula model.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            DataFrame with generated samples
        """
        try:
            # Use the copula model to generate samples
            synthetic_data = self.copula_model.sample(num_samples)
            return synthetic_data
        except Exception as e:
            self.log(f"Error generating samples: {str(e)}")
            raise RuntimeError(f"Failed to generate samples: {str(e)}")
    
    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features to numerical values for copula fitting.
        
        Args:
            data: DataFrame with categorical and numerical features
            
        Returns:
            DataFrame with all features encoded as numerical
        """
        # Create a copy to avoid modifying the original data
        encoded_data = data.copy()
        
        for col in self.categorical_columns:
            if col in encoded_data.columns:
                # For categorical features, use improved encoding method
                # Frequency-based encoding can better preserve distribution
                value_counts = encoded_data[col].value_counts(normalize=True)
                
                # Map categories to their frequency (preserves distribution better)
                freq_map = value_counts.to_dict()
                encoded_data[col] = encoded_data[col].map(freq_map).fillna(0)
                
                # Add small random noise to avoid identical values
                np.random.seed(self.random_state)
                encoded_data[col] += np.random.normal(0, 0.01, len(encoded_data))
                
                # Normalize to [0, 1] range
                min_val = encoded_data[col].min()
                max_val = encoded_data[col].max()
                if max_val > min_val:
                    encoded_data[col] = (encoded_data[col] - min_val) / (max_val - min_val)
        
        return encoded_data
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Calculate feature importance based on correlation structure.
        
        Returns:
            DataFrame with feature importance scores
        """
        if not self._is_fitted:
            raise ValueError("Generator must be fitted before calculating feature importance")
        
        try:
            # Get correlation matrix from the fitted model
            corr_matrix = self.copula_model.covariance
            
            # Convert to DataFrame with column names
            columns = list(self.dtypes.keys())
            corr_df = pd.DataFrame(corr_matrix, index=columns, columns=columns)
            
            # Calculate importance as the sum of absolute correlations for each feature
            importance = corr_df.abs().sum() / (len(columns) - 1)
            
            # Normalize to 0-100 scale
            importance = 100 * importance / importance.max()
            
            # Create result DataFrame
            result = pd.DataFrame({
                'feature': importance.index,
                'importance': importance.values
            }).sort_values('importance', ascending=False)
            
            return result
            
        except Exception as e:
            self.log(f"Error calculating feature importance: {str(e)}")
            return pd.DataFrame(columns=['feature', 'importance'])