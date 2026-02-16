import gc
import typing as t
from abc import ABC, abstractmethod
from pathlib import Path

# Import Dask
import dask
import dask.dataframe as dd
import numpy as np
import pandas as pd
import psutil
from dask.distributed import Client, LocalCluster


class BaseSyntheticGenerator(ABC):
    """
    Base abstract class for all synthetic data generation methods.

    This serves as the foundation for all data generation techniques
    and defines the common interface that all implementations must follow.
    """

    def __init__(
        self,
        random_state: t.Optional[int] = None,
        preserve_dtypes: bool = True,
        preserve_constraints: bool = True,
        verbose: bool = True,
        use_dask: bool = False,
        dask_temp_directory: t.Optional[str] = None,
        dask_n_workers: t.Optional[int] = None,
        dask_threads_per_worker: int = 2,
        memory_limit_percentage: float = 70.0,
    ):
        """
        Initialize the base synthetic data generator.

        Args:
            random_state: Seed for random number generation to ensure reproducibility
            preserve_dtypes: Whether to preserve the original data types in synthetic data
            preserve_constraints: Whether to enforce constraints from original data (ranges, unique values, etc.)
            verbose: Whether to print progress and information during processing
            use_dask: Whether to use Dask for distributed processing
            dask_temp_directory: Directory for Dask to store temporary files
            dask_n_workers: Number of Dask workers to use (None = auto)
            dask_threads_per_worker: Number of threads per Dask worker
            memory_limit_percentage: Maximum memory usage as percentage of system memory
        """
        self.random_state = random_state
        self.preserve_dtypes = preserve_dtypes
        self.preserve_constraints = preserve_constraints
        self.verbose = verbose
        self._is_fitted = False
        self.numerical_columns = []
        self.categorical_columns = []
        self.target_column = None

        # Dask configuration
        self.use_dask = use_dask
        self.dask_temp_directory = dask_temp_directory
        self.dask_n_workers = dask_n_workers
        self.dask_threads_per_worker = dask_threads_per_worker
        self._dask_client = None
        self._dask_cluster = None
        self._dask_initialized = False

        # Memory management
        self.memory_limit_percentage = memory_limit_percentage
        self._total_system_memory = psutil.virtual_memory().total
        self._memory_limit = (
            self.memory_limit_percentage / 100.0
        ) * self._total_system_memory

        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

        if verbose:
            self.log(
                f'System memory: {self._total_system_memory / (1024**3):.2f} GB'
            )
            self.log(
                f'Memory limit: {self._memory_limit / (1024**3):.2f} GB ({memory_limit_percentage}%)'
            )

        # Initialize Dask if needed
        if self.use_dask:
            self._initialize_dask()

    def _initialize_dask(self):
        """Initialize Dask for distributed computing."""
        if self.use_dask and self._dask_client is None:
            try:
                self.log('Initializing Dask client...')

                # Configure client parameters
                memory_limit = int(
                    self._memory_limit / (self.dask_n_workers or 4)
                )

                # Create local cluster
                cluster_kwargs = {
                    'processes': True,
                    'threads_per_worker': self.dask_threads_per_worker,
                    'memory_limit': f'{memory_limit}B',
                }

                if self.dask_n_workers is not None:
                    cluster_kwargs['n_workers'] = self.dask_n_workers

                if self.dask_temp_directory:
                    cluster_kwargs[
                        'local_directory'
                    ] = self.dask_temp_directory

                # Create cluster and client
                self._dask_cluster = LocalCluster(**cluster_kwargs)
                self._dask_client = Client(self._dask_cluster)
                self._dask_initialized = True

                self.log(
                    f"Dask client initialized with {len(self._dask_client.scheduler_info()['workers'])} workers"
                )
                self.log(f'Dashboard link: {self._dask_client.dashboard_link}')

            except Exception as e:
                self.log(
                    f'Error initializing Dask client: {str(e)}. Falling back to non-Dask mode.'
                )
                self.use_dask = False
                self._dask_client = None
                self._dask_cluster = None

    def _close_dask(self):
        """Close Dask client and cluster if they exist."""
        if self._dask_initialized:
            try:
                if self._dask_client is not None:
                    self.log('Closing Dask client...')
                    self._dask_client.close()
                    self._dask_client = None

                if self._dask_cluster is not None:
                    self.log('Shutting down Dask cluster...')
                    self._dask_cluster.close()
                    self._dask_cluster = None

                self._dask_initialized = False

            except Exception as e:
                self.log(f'Error closing Dask resources: {str(e)}')

    @abstractmethod
    def fit(
        self,
        data: pd.DataFrame,
        target_column: t.Optional[str] = None,
        categorical_columns: t.Optional[t.List[str]] = None,
        numerical_columns: t.Optional[t.List[str]] = None,
        **kwargs,
    ) -> None:
        """
        Fit the generator to the input data.

        Args:
            data: The dataset to fit the generator on
            target_column: The name of the target variable column
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            **kwargs: Additional parameters specific to the implementation
        """
        pass

    @abstractmethod
    def generate(
        self,
        num_samples: int = 1000,
        chunk_size: t.Optional[int] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Generate synthetic data based on the fitted model.

        Args:
            num_samples: Number of synthetic samples to generate
            chunk_size: Size of chunks to use when generating large datasets
            **kwargs: Additional parameters specific to the implementation

        Returns:
            DataFrame containing the generated synthetic data
        """
        pass

    def _validate_columns(
        self,
        data: pd.DataFrame,
        categorical_columns: t.Optional[t.List[str]] = None,
        numerical_columns: t.Optional[t.List[str]] = None,
    ) -> t.Tuple[t.List[str], t.List[str]]:
        """
        Validate and infer column types if not explicitly provided.

        Args:
            data: The dataset to validate
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names

        Returns:
            Tuple of (categorical_columns, numerical_columns)
        """
        all_columns = data.columns.tolist()

        # If categorical columns are provided, validate them
        if categorical_columns is not None:
            invalid_cols = set(categorical_columns) - set(all_columns)
            if invalid_cols:
                raise ValueError(
                    f'Categorical columns {invalid_cols} not found in data'
                )
        else:
            # Infer categorical columns
            categorical_columns = []
            for col in all_columns:
                if data[
                    col
                ].dtype == 'object' or pd.api.types.is_categorical_dtype(
                    data[col]
                ):
                    categorical_columns.append(col)
                elif pd.api.types.is_bool_dtype(data[col]):
                    categorical_columns.append(col)
                elif (
                    pd.api.types.is_integer_dtype(data[col])
                    and data[col].nunique() < 20
                ):
                    # Treat integers with few unique values as categorical
                    categorical_columns.append(col)

        # If numerical columns are provided, validate them
        if numerical_columns is not None:
            invalid_cols = set(numerical_columns) - set(all_columns)
            if invalid_cols:
                raise ValueError(
                    f'Numerical columns {invalid_cols} not found in data'
                )
        else:
            # Infer numerical columns (all columns that are not categorical)
            numerical_columns = [
                col for col in all_columns if col not in categorical_columns
            ]

        return categorical_columns, numerical_columns

    def _memory_optimize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage of DataFrame by downcasting data types.

        Args:
            df: DataFrame to optimize

        Returns:
            Memory-optimized DataFrame
        """
        if not self.preserve_dtypes:
            # Downcast numeric columns
            for col in self.numerical_columns:
                if col in df.columns:
                    if pd.api.types.is_integer_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], downcast='integer')
                    elif pd.api.types.is_float_dtype(df[col]):
                        df[col] = pd.to_numeric(df[col], downcast='float')

            # Convert categorical columns to appropriate types
            for col in self.categorical_columns:
                if (
                    col in df.columns
                    and not pd.api.types.is_categorical_dtype(df[col])
                ):
                    df[col] = df[col].astype('category')

        return df

    def _enforce_constraints(
        self, df: pd.DataFrame, original_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enforce constraints from original data to synthetic data.

        Args:
            df: Synthetic DataFrame to apply constraints to
            original_data: Original DataFrame to derive constraints from

        Returns:
            DataFrame with constraints enforced
        """
        if not self.preserve_constraints:
            return df

        # Handle value ranges for numerical columns
        for col in self.numerical_columns:
            if col in df.columns:
                min_val = original_data[col].min()
                max_val = original_data[col].max()

                # Clip values to enforce range constraints
                df[col] = df[col].clip(min_val, max_val)

        # Handle categorical values
        for col in self.categorical_columns:
            if col in df.columns:
                # Get allowed values from original data
                allowed_values = set(original_data[col].unique())

                # Map values outside of the allowed set to values within the set
                mask = ~df[col].isin(allowed_values)
                if mask.any():
                    # Replace with random values from allowed set
                    replacement_values = np.random.choice(
                        list(allowed_values), size=mask.sum(), replace=True
                    )
                    df.loc[mask, col] = replacement_values

        return df

    def _enforce_constraints_dask(
        self, df: dd.DataFrame, original_data: pd.DataFrame
    ) -> dd.DataFrame:
        """
        Enforce constraints from original data to synthetic data using Dask.

        Args:
            df: Synthetic Dask DataFrame to apply constraints to
            original_data: Original DataFrame to derive constraints from

        Returns:
            Dask DataFrame with constraints enforced
        """
        if not self.preserve_constraints:
            return df

        # Function to apply constraints to a partition
        def apply_constraints(partition):
            # Handle numerical columns
            for col in self.numerical_columns:
                if col in partition.columns:
                    min_val = original_data[col].min()
                    max_val = original_data[col].max()
                    partition[col] = partition[col].clip(min_val, max_val)

            # Handle categorical columns
            for col in self.categorical_columns:
                if col in partition.columns:
                    allowed_values = set(original_data[col].unique())
                    mask = ~partition[col].isin(allowed_values)
                    if mask.any():
                        # Replace with random values from allowed set - need to set seed for reproducibility
                        np.random.seed(self.random_state)
                        replacement_values = np.random.choice(
                            list(allowed_values), size=mask.sum(), replace=True
                        )
                        partition.loc[mask, col] = replacement_values

            return partition

        # Apply constraints function to each partition
        return df.map_partitions(apply_constraints)

    def _to_dask_dataframe(
        self, df: pd.DataFrame, npartitions: t.Optional[int] = None
    ) -> dd.DataFrame:
        """
        Convert pandas DataFrame to Dask DataFrame with optimal partitioning.

        Args:
            df: Pandas DataFrame to convert
            npartitions: Number of partitions (if None, calculated based on size)

        Returns:
            Dask DataFrame
        """
        if not self.use_dask or self._dask_client is None:
            raise ValueError('Dask client is not initialized')

        # Determine optimal number of partitions if not specified
        if npartitions is None:
            # Get number of workers
            try:
                n_workers = len(self._dask_client.scheduler_info()['workers'])
                # Calculate partitions based on workers and memory
                memory_per_row = df.memory_usage(deep=True).sum() / len(df)
                target_partition_size = (
                    100 * 1024 * 1024
                )  # Target 100MB per partition
                partition_size = max(
                    int(target_partition_size / memory_per_row), 1000
                )

                # Calculate partitions with at least 2 per worker
                npartitions = max(
                    n_workers * 2, min(len(df) // partition_size, 100)
                )
            except:
                # Fallback to reasonable default
                npartitions = min(max(len(df) // 10000, 1), 100)

        # Convert to Dask DataFrame
        dask_df = dd.from_pandas(df, npartitions=npartitions)

        if self.verbose:
            self.log(
                f'Converted DataFrame to Dask DataFrame with {npartitions} partitions'
            )

        return dask_df

    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)

    def __repr__(self) -> str:
        """Return string representation of the generator."""
        status = 'fitted' if self._is_fitted else 'not fitted'
        dask_status = (
            'with Dask'
            if self.use_dask and self._dask_client is not None
            else ''
        )
        return f'{self.__class__.__name__}(random_state={self.random_state}, {status} {dask_status})'

    def __del__(self):
        """Clean up resources when object is destroyed."""
        self._close_dask()
