import gc
import typing as t
import warnings

# Import Dask for parallel processing
import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
import psutil

# Import GaussianMultivariate from copulas
from copulas.multivariate import GaussianMultivariate
from dask.distributed import Client, progress, wait
from sklearn.preprocessing import OrdinalEncoder
from tqdm.auto import tqdm

from ..base import BaseSyntheticGenerator
from ..utils.bernoulli_custom import BernoulliCustom


# Define a standalone function for Dask serialization
def _process_chunk_for_dask_standalone(
    size: int,
    chunk_id: int,
    random_state: int,
    scaler_fit_data: np.ndarray,
    copula_model_params: dict,
    categorical_columns: list,
    numerical_columns: list,
    dtypes: dict,
    column_constraints: dict,
    preserve_dtypes: bool,
    preserve_constraints: bool,
    original_data_sample_dict: dict,
    post_process_method: str,
    num_column_stats: dict,
    cat_column_stats: dict,
):
    """
    Generate and process a chunk of synthetic data - Standalone function for Dask compatibility.

    Args:
        size: Size of the chunk
        chunk_id: ID of the chunk
        random_state: Random seed
        scaler_fit_data: Scaled data used to fit the copula
        copula_model_params: Parameters of the fitted copula model
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        dtypes: Dictionary of column data types
        column_constraints: Dictionary of column constraints
        preserve_dtypes: Whether to preserve original data types
        preserve_constraints: Whether to enforce constraints
        original_data_sample_dict: Dictionary version of original data sample
        post_process_method: Method for post-processing
        num_column_stats: Statistics for numerical columns
        cat_column_stats: Statistics for categorical columns

    Returns:
        Processed chunk of synthetic data
    """
    np.random.seed(
        random_state + chunk_id if random_state is not None else chunk_id
    )

    # Print message
    print(f'Generating chunk {chunk_id+1} with {size} samples')

    # Recreate copula model from parameters
    copula_model = GaussianMultivariate(random_state=random_state)

    # Restore model from parameters
    copula_model.covariance = copula_model_params.get('covariance')
    copula_model.columns = copula_model_params.get('columns')
    copula_model.univariates = copula_model_params.get('univariates')

    # Generate samples
    synthetic_data = copula_model.sample(size)

    # Convert original data sample back from dict
    original_data_sample = pd.DataFrame.from_dict(original_data_sample_dict)

    # Post-process the chunk
    # Apply constraints if required
    if preserve_constraints:
        # Enforce constraints from original data
        for col in numerical_columns:
            if col in synthetic_data.columns and col in column_constraints:
                constraints = column_constraints[col]
                # Clip values to enforce range constraints
                synthetic_data[col] = synthetic_data[col].clip(
                    constraints.get('min'), constraints.get('max')
                )

        # Handle categorical values
        for col in categorical_columns:
            if col in synthetic_data.columns and col in column_constraints:
                # Get allowed values from constraints
                allowed_values = column_constraints[col].get('values', [])

                # Map values outside of the allowed set to values within the set
                mask = ~synthetic_data[col].isin(allowed_values)
                if mask.any():
                    # Replace with random values from allowed set
                    replacement_values = np.random.choice(
                        allowed_values, size=mask.sum(), replace=True
                    )
                    synthetic_data.loc[mask, col] = replacement_values

    # Enhanced post-processing to improve data quality
    if post_process_method == 'enhanced':
        # Adjust numerical columns to better match original distributions
        for col in numerical_columns:
            if col in synthetic_data.columns and col in num_column_stats:
                stats = num_column_stats[col]

                # Handle outliers by clipping to a reasonable range
                if 'q1' in stats and 'q3' in stats:
                    iqr = stats['q3'] - stats['q1']
                    lower_bound = stats['q1'] - 1.5 * iqr
                    upper_bound = stats['q3'] + 1.5 * iqr

                    # Clip values but keep some variability
                    synthetic_data[col] = synthetic_data[col].clip(
                        lower=max(lower_bound, stats.get('min', lower_bound)),
                        upper=min(upper_bound, stats.get('max', upper_bound)),
                    )

        # Correct categorical distributions
        for col in categorical_columns:
            if col in synthetic_data.columns and col in cat_column_stats:
                # Only apply if current distribution deviates significantly
                synth_dist = synthetic_data[col].value_counts(normalize=True)

                cat_stats = cat_column_stats[col]
                if 'values' in cat_stats and 'frequencies' in cat_stats:
                    orig_dist = pd.Series(
                        cat_stats['frequencies'], index=cat_stats['values']
                    )

                    # Measure distribution difference
                    common_cats = set(synth_dist.index) & set(orig_dist.index)
                    if len(common_cats) > 0:
                        common_synth = synth_dist.loc[list(common_cats)]
                        common_orig = orig_dist.loc[list(common_cats)]
                        dist_diff = np.abs(common_synth - common_orig).mean()

                        # If difference is significant, adjust the distribution
                        if dist_diff > 0.1:  # threshold for adjustment
                            # For each category that needs adjustment
                            for cat, target_freq in orig_dist.items():
                                if cat in synth_dist.index:
                                    current_freq = synth_dist[cat]

                                    # Calculate how many values need to change
                                    diff = target_freq - current_freq
                                    if (
                                        abs(diff) < 0.01
                                    ):  # Skip small adjustments
                                        continue

                                    n_samples = len(synthetic_data)
                                    n_changes = int(abs(diff) * n_samples)

                                    if (
                                        diff > 0
                                    ):  # Need to increase this category
                                        # Find other categories to decrease
                                        other_cats = [
                                            c
                                            for c in synth_dist.index
                                            if synth_dist[c]
                                            > orig_dist.get(c, 0)
                                        ]
                                        if not other_cats:
                                            continue

                                        # Select random samples from other categories to change
                                        for _ in range(
                                            min(n_changes, 100)
                                        ):  # Limit changes to avoid overfitting
                                            other_cat = np.random.choice(
                                                other_cats
                                            )
                                            idx = synthetic_data[
                                                synthetic_data[col]
                                                == other_cat
                                            ].index
                                            if len(idx) > 0:
                                                change_idx = np.random.choice(
                                                    idx
                                                )
                                                synthetic_data.loc[
                                                    change_idx, col
                                                ] = cat

                                    elif (
                                        diff < 0
                                    ):  # Need to decrease this category
                                        # Find other categories to increase
                                        other_cats = [
                                            c
                                            for c in orig_dist.index
                                            if c in synth_dist.index
                                            and synth_dist[c] < orig_dist[c]
                                        ]
                                        if not other_cats:
                                            continue

                                        # Select random samples from this category to change
                                        idx = synthetic_data[
                                            synthetic_data[col] == cat
                                        ].index
                                        for _ in range(
                                            min(n_changes, 100)
                                        ):  # Limit changes to avoid overfitting
                                            if len(idx) > 0:
                                                change_idx = np.random.choice(
                                                    idx
                                                )
                                                other_cat = np.random.choice(
                                                    other_cats
                                                )
                                                synthetic_data.loc[
                                                    change_idx, col
                                                ] = other_cat

    # Convert dtypes back to original if required
    if preserve_dtypes:
        for col, dtype in dtypes.items():
            if col in synthetic_data.columns:
                try:
                    synthetic_data[col] = synthetic_data[col].astype(dtype)
                except (ValueError, TypeError):
                    # If conversion fails, keep as is
                    pass

    return synthetic_data


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
        dask_threads_per_worker: int = 2,
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
            memory_limit_percentage=memory_limit_percentage,
        )
        self.fit_sample_size = fit_sample_size
        self.original_data_sample = None
        self.copula_model = None
        self.dtypes = {}
        self.n_jobs = n_jobs
        self.memory_limit_percentage = memory_limit_percentage
        # Store column constraints
        self.column_constraints = {}
        self.num_column_stats = {}
        self.cat_column_stats = {}
        self.post_process_method = 'standard'

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
            self.log(f'Error generating samples: {str(e)}')
            raise RuntimeError(f'Failed to generate samples: {str(e)}')

    def _encode_categorical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features to numerical values for copula fitting.

        Args:
            data: DataFrame with categorical and numerical features

        Returns:
            DataFrame with categorical features encoded numerically
        """
        # Cria uma cópia dos dados para não alterar o original
        encoded_data = data.copy()

        if self.categorical_columns:
            encoder = OrdinalEncoder()
            encoded_data[self.categorical_columns] = encoder.fit_transform(
                encoded_data[self.categorical_columns]
            )

            # Opcional: Salvar o encoder para uso posterior (geração inversa)
            self._categorical_encoder = encoder

        return encoded_data

    # This is now a method that prepares parameters for the standalone function
    def _prepare_chunk_data(self, size, chunk_id):
        """
        Prepare data needed for chunk processing

        Args:
            size: Size of the chunk
            chunk_id: ID of the chunk

        Returns:
            Dictionary with parameters for standalone processing
        """
        if self.verbose:
            print(f'Preparing data for chunk {chunk_id+1} with {size} samples')

        # Extract model parameters that can be serialized
        copula_model_params = {
            'covariance': self.copula_model.covariance
            if hasattr(self.copula_model, 'covariance')
            else None,
            'columns': self.copula_model.columns
            if hasattr(self.copula_model, 'columns')
            else None,
            'univariates': self.copula_model.univariates
            if hasattr(self.copula_model, 'univariates')
            else None,
        }

        # Prepare original data sample as dict for serialization
        original_data_sample_dict = (
            self.original_data_sample.to_dict()
            if self.original_data_sample is not None
            else {}
        )

        # Extract column constraints
        column_constraints = {}

        # For numerical columns
        for col in self.numerical_columns:
            if col in self.num_column_stats:
                column_constraints[col] = {
                    'min': self.num_column_stats[col].get('min'),
                    'max': self.num_column_stats[col].get('max'),
                }

        # For categorical columns
        for col in self.categorical_columns:
            if col in self.cat_column_stats:
                column_constraints[col] = {
                    'values': self.cat_column_stats[col].get('values', [])
                }

        return dict(
            size=size,
            chunk_id=chunk_id,
            random_state=self.random_state,
            scaler_fit_data=None,  # This would be too large, omit it
            copula_model_params=copula_model_params,
            categorical_columns=self.categorical_columns,
            numerical_columns=self.numerical_columns,
            dtypes=self.dtypes,
            column_constraints=column_constraints,
            preserve_dtypes=self.preserve_dtypes,
            preserve_constraints=self.preserve_constraints,
            original_data_sample_dict=original_data_sample_dict,
            post_process_method=self.post_process_method,
            num_column_stats=self.num_column_stats,
            cat_column_stats=self.cat_column_stats,
        )

    def fit(
        self,
        data: pd.DataFrame,
        target_column: t.Optional[str] = None,
        categorical_columns: t.Optional[t.List[str]] = None,
        numerical_columns: t.Optional[t.List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Fit the Gaussian Copula generator to the input data.

        Args:
            data: The dataset to fit the generator on
            target_column: The name of the target variable column
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            **kwargs: Additional parameters specific to the implementation
        """
        self.log(
            f'Fitting Gaussian Copula generator on dataset with {len(data)} rows...'
        )

        # Store target column name
        self.target_column = target_column

        # Get stratification column if provided
        stratify_by = kwargs.get('stratify_by', None)
        stratify = (
            data[stratify_by]
            if stratify_by and stratify_by in data.columns
            else None
        )

        # Determine the number of samples to use for fitting
        max_fit_samples = kwargs.get('max_fit_samples', self.fit_sample_size)
        if len(data) > max_fit_samples:
            self.log(
                f'Dataset is large ({len(data)} rows). Using {max_fit_samples} samples for fitting.'
            )
            if stratify is not None:
                self.log(f'Using stratified sampling by column: {stratify_by}')
                fit_data = data.sample(
                    max_fit_samples,
                    random_state=self.random_state,
                    stratify=stratify,
                )
            else:
                fit_data = data.sample(
                    max_fit_samples, random_state=self.random_state
                )
        else:
            fit_data = data

        # Store a small sample of original data for validation and constraint enforcement
        sample_size = min(1000, len(data))
        self.original_data_sample = data.sample(
            sample_size, random_state=self.random_state
        )

        # Validate and infer column types
        (
            self.categorical_columns,
            self.numerical_columns,
        ) = self._validate_columns(
            fit_data, categorical_columns, numerical_columns
        )

        self.log(
            f'Identified {len(self.categorical_columns)} categorical columns and {len(self.numerical_columns)} numerical columns'
        )

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
                        'q3': col_data.quantile(0.75),
                    }

        # Store categorical value distributions for post-processing
        self.cat_column_stats = {}
        for col in self.categorical_columns:
            if col in fit_data.columns:
                value_counts = fit_data[col].value_counts(normalize=True)
                self.cat_column_stats[col] = {
                    'values': value_counts.index.tolist(),
                    'frequencies': value_counts.values.tolist(),
                }

        # Handle categorical features by encoding them
        encoded_data = self._encode_categorical_features(fit_data)

        # Define binary columns explicitly
        binary_columns = []
        if target_column and fit_data[target_column].nunique() == 2:
            binary_columns.append(target_column)

        # Fit the copula model
        try:
            gc.collect()
            self.log('Initializing GaussianMultivariate copula model...')
            self.copula_model = GaussianMultivariate(
                random_state=self.random_state
            )
            self.log('Fitting copula model to data...')
            self.copula_model.fit(encoded_data)
            self.log('Copula model fitting completed successfully')

            # Explicitly redefine binary columns with Bernoulli distribution
            for column in binary_columns:
                bernoulli_custom = BernoulliCustom()
                bernoulli_custom.fit(encoded_data[column])

                # Obter índice correto da coluna para substituição
                column_index = self.copula_model.columns.index(column)

                # Substituir pelo índice
                self.copula_model.univariates[column_index] = bernoulli_custom

                self.log(
                    f"Explicitly set '{column}' as BernoulliCustom with parameter p = {bernoulli_custom.p:.4f}"
                )

            # Explicitly log marginal distributions
            self.log('Identified marginal distributions for each column:')
            for column, distribution in self.copula_model.univariates.items():
                dist_name = type(distribution).__name__
                params = distribution.to_dict()
                self.log(
                    f"Column '{column}': Distribution '{dist_name}' with parameters {params}"
                )

            del encoded_data
            gc.collect()

        except Exception as e:
            self.log(f'Error fitting copula model: {str(e)}')
            raise RuntimeError(f'Failed to fit copula model: {str(e)}')

        self._is_fitted = True
        self.log('Gaussian Copula model fitting completed successfully')

    def generate(
        self,
        num_samples: int = 1000,
        chunk_size: t.Optional[int] = None,
        **kwargs,
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
            raise ValueError('Generator must be fitted before generating data')

        self.log(f'Generating {num_samples} synthetic samples...')

        # Determine post-processing method
        self.post_process_method = kwargs.get(
            'post_process_method', 'enhanced'
        )

        # Determine chunk size for memory-efficient generation
        memory_efficient = kwargs.get('memory_efficient', True)
        dynamic_chunk_sizing = kwargs.get('dynamic_chunk_sizing', True)

        if memory_efficient:
            # If no chunk size is provided or dynamic sizing is requested
            if chunk_size is None or dynamic_chunk_sizing:
                # Estimate memory per sample
                test_samples = min(100, num_samples)
                test_data = self._generate_copula_samples(test_samples)
                memory_per_sample = (
                    test_data.memory_usage(deep=True).sum() / test_samples
                )

                # Calculate safe chunk size based on available memory
                available_memory = max(
                    0.5 * self._memory_limit - psutil.virtual_memory().used,
                    0.2 * self._memory_limit,
                )  # At least 20% of limit

                # Use at most 50% of available memory for chunk generation
                safe_chunk_size = int(
                    0.5 * available_memory / memory_per_sample
                )

                # Set a reasonable minimum and maximum
                safe_chunk_size = max(min(safe_chunk_size, 10000), 100)

                if chunk_size is None or (
                    dynamic_chunk_sizing and safe_chunk_size < chunk_size
                ):
                    chunk_size = safe_chunk_size

                self.log(
                    f'Dynamically determined chunk size: {chunk_size} samples'
                )

                # Clean up test data
                del test_data
                gc.collect()

            # Check if we should use Dask
            if self.use_dask and self._dask_client is not None:
                try:
                    return self._generate_with_dask(num_samples, chunk_size)
                except Exception as e:
                    self.log(f'Error using Dask for generation: {str(e)}')
                    self.log('Falling back to standard chunk processing')
                    return self._generate_in_chunks(num_samples, chunk_size)
            else:
                # Use standard chunk processing
                return self._generate_in_chunks(num_samples, chunk_size)
        else:
            # Generate all data at once
            return self._generate_batch(num_samples)

    def _generate_with_dask(
        self, num_samples: int, chunk_size: int
    ) -> pd.DataFrame:
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

        self.log(
            f'Generating {len(chunk_sizes)} chunks with sizes: {chunk_sizes}'
        )

        # Create list of tasks using standalone function
        tasks = []
        for i, size in enumerate(chunk_sizes):
            # Prepare parameters
            params = self._prepare_chunk_data(size, i)

            # Create a delayed task with the standalone function
            task = dask.delayed(_process_chunk_for_dask_standalone)(**params)
            tasks.append(task)

        # Compute chunks in parallel
        self.log('Computing chunks in parallel with Dask...')

        # Use progress visualization if in verbose mode
        if self.verbose:
            # Create a progress bar
            with tqdm(
                total=len(chunk_sizes), desc='Generating chunks'
            ) as pbar:
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
            self.log(f'Error combining chunks: {str(e)}')
            # Attempt alternative approach if the first fails
            result = pd.DataFrame()
            for chunk in chunks:
                result = pd.concat([result, chunk], ignore_index=True)

        # Clean up to free memory
        del chunks
        gc.collect()

        self.log(f'Successfully generated {len(result)} samples using Dask')
        return result

    def _generate_in_chunks(
        self, num_samples: int, chunk_size: int
    ) -> pd.DataFrame:
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

        self.log(
            f'Generating {len(chunk_sizes)} chunks with sizes: {chunk_sizes}'
        )

        # Process chunks sequentially
        all_chunks = []

        with tqdm(
            total=len(chunk_sizes),
            desc='Generating chunks',
            disable=not self.verbose,
        ) as pbar:
            for i, size in enumerate(chunk_sizes):
                self.log(
                    f'Generating chunk {i+1}/{len(chunk_sizes)} with {size} samples'
                )

                # Generate samples using the copula model
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

        self.log(
            f'Successfully generated {len(result)} samples using sequential processing'
        )
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
            chunk_df = self._enforce_constraints(
                chunk_df, self.original_data_sample
            )

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
                        upper=min(upper_bound, stats['max']),
                    )

            # Correct categorical distributions
            for col in self.categorical_columns:
                if col in chunk_df.columns and col in self.cat_column_stats:
                    # Only apply if current distribution deviates significantly
                    synth_dist = chunk_df[col].value_counts(normalize=True)
                    orig_dist = pd.Series(
                        self.cat_column_stats[col]['frequencies'],
                        index=self.cat_column_stats[col]['values'],
                    )

                    # Measure distribution difference
                    common_cats = set(synth_dist.index) & set(orig_dist.index)
                    if len(common_cats) > 0:
                        common_synth = synth_dist.loc[list(common_cats)]
                        common_orig = orig_dist.loc[list(common_cats)]
                        dist_diff = np.abs(common_synth - common_orig).mean()

                        # If difference is significant, adjust the distribution
                        if dist_diff > 0.1:  # threshold for adjustment
                            self._adjust_categorical_distribution(
                                chunk_df, col, orig_dist
                            )

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

    def _adjust_categorical_distribution(
        self, df: pd.DataFrame, column: str, target_dist: pd.Series
    ) -> None:
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
                    other_cats = [
                        c
                        for c in current_dist.index
                        if current_dist[c] > target_dist.get(c, 0)
                    ]
                    if not other_cats:
                        continue

                    # Select random samples from other categories to change
                    for _ in range(
                        min(n_changes, 100)
                    ):  # Limit changes to avoid overfitting
                        other_cat = np.random.choice(other_cats)
                        idx = df[df[column] == other_cat].index
                        if len(idx) > 0:
                            change_idx = np.random.choice(idx)
                            df.loc[change_idx, column] = cat

                elif diff < 0:  # Need to decrease this category
                    # Find other categories to increase
                    other_cats = [
                        c
                        for c in target_dist.index
                        if c in current_dist.index
                        and current_dist[c] < target_dist[c]
                    ]
                    if not other_cats:
                        continue

                    # Select random samples from this category to change
                    idx = df[df[column] == cat].index
                    for _ in range(
                        min(n_changes, 100)
                    ):  # Limit changes to avoid overfitting
                        if len(idx) > 0:
                            change_idx = np.random.choice(idx)
                            other_cat = np.random.choice(other_cats)
                            df.loc[change_idx, column] = other_cat

    def save_model(self, path: str) -> None:
        """
        Save the fitted model to disk.

        Args:
            path: Path to save the model
        """
        if not self._is_fitted:
            raise ValueError('Generator must be fitted before saving')

        import os
        import pickle

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Prepare model data for serialization
        model_data = {
            'copula_parameters': {
                'covariance': self.copula_model.covariance
                if hasattr(self.copula_model, 'covariance')
                else None,
                'columns': self.copula_model.columns
                if hasattr(self.copula_model, 'columns')
                else None,
                'univariates': self.copula_model.univariates
                if hasattr(self.copula_model, 'univariates')
                else None,
            },
            'categorical_columns': self.categorical_columns,
            'numerical_columns': self.numerical_columns,
            'dtypes': self.dtypes,
            'num_column_stats': self.num_column_stats,
            'cat_column_stats': self.cat_column_stats,
            'random_state': self.random_state,
            'version': '0.2.0',  # Add version info for compatibility checks
        }

        # Save to file
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        self.log(f'Model saved to {path}')

    def load_model(self, path: str) -> None:
        """
        Load a saved model from disk.

        Args:
            path: Path to the saved model
        """
        import pickle

        # Load from file
        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        # Check version compatibility
        if 'version' not in model_data:
            self.log('Warning: Loading model saved with an older version')

        # Restore model parameters
        self.categorical_columns = model_data.get('categorical_columns', [])
        self.numerical_columns = model_data.get('numerical_columns', [])
        self.dtypes = model_data.get('dtypes', {})
        self.num_column_stats = model_data.get('num_column_stats', {})
        self.cat_column_stats = model_data.get('cat_column_stats', {})

        # Initialize the copula model
        from copulas.multivariate import GaussianMultivariate

        self.copula_model = GaussianMultivariate(
            random_state=self.random_state
        )

        # Restore copula parameters
        copula_params = model_data.get('copula_parameters', {})
        self.copula_model.covariance = copula_params.get('covariance')
        self.copula_model.columns = copula_params.get('columns')
        self.copula_model.univariates = copula_params.get('univariates')

        self._is_fitted = True
        self.log(f'Model loaded from {path}')

    def evaluate_quality(
        self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame = None
    ) -> dict:
        """
        Evaluate the quality of synthetic data compared to real data.

        Args:
            real_data: Original real dataset
            synthetic_data: Generated synthetic dataset (if None, uses self-generated data)

        Returns:
            Dictionary with quality metrics
        """
        # If synthetic data is not provided, generate it
        if synthetic_data is None:
            if not hasattr(self, 'data') or len(self.data) == 0:
                synthetic_data = self.generate(len(real_data))
            else:
                synthetic_data = self.data

        # Import evaluation function
        from ..metrics.statistical import evaluate_synthetic_quality

        # Evaluate quality
        metrics = evaluate_synthetic_quality(
            real_data=real_data,
            synthetic_data=synthetic_data,
            numerical_columns=self.numerical_columns,
            categorical_columns=self.categorical_columns,
            target_column=self.target_column,
            verbose=self.verbose,
        )

        return metrics

    def plot_comparison(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame = None,
        columns: t.List[str] = None,
        save_path: str = None,
    ) -> None:
        """
        Plot comparison visualizations between real and synthetic data.

        Args:
            real_data: Original real dataset
            synthetic_data: Generated synthetic dataset (if None, uses self-generated data)
            columns: Columns to include in the visualization (if None, selects automatically)
            save_path: Path to save the visualization (if None, displays it)
        """
        # If synthetic data is not provided, generate it
        if synthetic_data is None:
            if not hasattr(self, 'data') or len(self.data) == 0:
                synthetic_data = self.generate(len(real_data))
            else:
                synthetic_data = self.data

        # Import visualization functions
        from ..visualization.comparison import plot_distributions

        # If columns are not specified, select some
        if columns is None:
            # Get common columns
            common_cols = list(
                set(real_data.columns) & set(synthetic_data.columns)
            )

            # Prioritize specified categorical and numerical columns
            priority_cols = list(
                set(self.categorical_columns + self.numerical_columns)
                & set(common_cols)
            )

            # Limit to 10 columns
            if len(priority_cols) > 10:
                columns = priority_cols[:10]
            else:
                columns = priority_cols

        # Create visualization
        fig = plot_distributions(
            real_data=real_data,
            synthetic_data=synthetic_data,
            columns=columns,
            numerical_columns=self.numerical_columns,
            categorical_columns=self.categorical_columns,
            save_path=save_path,
        )

        # Display if not saving
        if save_path is None and fig is not None:
            import matplotlib.pyplot as plt

            plt.show()

    def generate_report(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame = None,
        report_path: str = None,
    ) -> str:
        """
        Generate a detailed quality report for synthetic data.

        Args:
            real_data: Original real dataset
            synthetic_data: Generated synthetic dataset (if None, uses self-generated data)
            report_path: Path to save the report (if None, uses default)

        Returns:
            Path to the generated report
        """
        # If synthetic data is not provided, generate it
        if synthetic_data is None:
            if not hasattr(self, 'data') or len(self.data) == 0:
                synthetic_data = self.generate(len(real_data))
            else:
                synthetic_data = self.data

        # Import report generator
        from ..reports.report_generator import generate_quality_report

        # Evaluate quality metrics
        metrics = self.evaluate_quality(real_data, synthetic_data)

        # Generate report
        generator_info = (
            f'GaussianCopulaGenerator(random_state={self.random_state})'
        )
        report_file = generate_quality_report(
            real_data=real_data,
            synthetic_data=synthetic_data,
            quality_metrics=metrics,
            generator_info=generator_info,
            report_path=report_path,
        )

        return report_file
