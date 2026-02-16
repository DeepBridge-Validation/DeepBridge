"""
Module for managing Dask distributed computing for synthetic data generation.
"""

import typing as t
from pathlib import Path

import dask
import dask.dataframe as dd
import psutil
from dask.distributed import Client, progress, wait


class DaskManager:
    """
    Manages Dask distributed computing for synthetic data operations.
    Extracted from Synthesize class to separate Dask responsibilities.
    """

    def __init__(
        self,
        use_dask: bool = True,
        dask_temp_directory: t.Optional[str] = None,
        dask_n_workers: t.Optional[int] = None,
        dask_threads_per_worker: int = 2,
        memory_limit_percentage: float = 70.0,
        verbose: bool = True,
    ):
        """
        Initialize the Dask manager.

        Args:
            use_dask: Whether to use Dask for distributed processing
            dask_temp_directory: Directory for Dask to store temporary files
            dask_n_workers: Number of Dask workers (None = auto)
            dask_threads_per_worker: Number of threads per Dask worker
            memory_limit_percentage: Maximum memory usage percentage
            verbose: Whether to print progress information
        """
        self.use_dask = use_dask
        self.dask_temp_directory = dask_temp_directory
        self.dask_n_workers = dask_n_workers
        self.dask_threads_per_worker = dask_threads_per_worker
        self.memory_limit_percentage = memory_limit_percentage
        self.verbose = verbose

        # Memory management
        self._total_system_memory = psutil.virtual_memory().total
        self._memory_limit = (
            self.memory_limit_percentage / 100.0
        ) * self._total_system_memory

        # Dask client
        self._dask_client = None

        if self.verbose:
            print(
                f'System memory: {self._total_system_memory / (1024**3):.2f} GB'
            )
            print(
                f'Memory limit: {self._memory_limit / (1024**3):.2f} GB ({memory_limit_percentage}%)'
            )
            if self.use_dask:
                print(
                    f"Dask enabled with {dask_n_workers or 'auto'} workers, {dask_threads_per_worker} threads per worker"
                )

    def initialize_client(self) -> t.Optional[Client]:
        """
        Initialize the Dask client for distributed computing.

        Returns:
            Optional[Client]: The initialized Dask client or None if failed or disabled
        """
        if not self.use_dask:
            return None

        try:
            self.log('Initializing Dask client...')

            # Configure client parameters
            client_kwargs = {
                'processes': True,
                'threads_per_worker': self.dask_threads_per_worker,
                'memory_limit': f'{int(self._memory_limit / (self.dask_n_workers or 4))}B',
            }

            if self.dask_n_workers is not None:
                client_kwargs['n_workers'] = self.dask_n_workers

            if self.dask_temp_directory:
                client_kwargs['local_directory'] = self.dask_temp_directory

            # Create client
            self._dask_client = Client(**client_kwargs)

            self.log(
                f'Dask client initialized: {self._dask_client.dashboard_link}'
            )

            return self._dask_client

        except Exception as e:
            self.log(
                f'Error initializing Dask client: {str(e)}. Falling back to non-Dask mode.'
            )
            self.use_dask = False
            self._dask_client = None
            return None

    def close_client(self) -> None:
        """Close the Dask client if it exists."""
        if self._dask_client is not None:
            try:
                self.log('Closing Dask client...')
                self._dask_client.close()
                self._dask_client = None
            except Exception as e:
                self.log(f'Error closing Dask client: {str(e)}')

    def to_dask_dataframe(
        self, df, num_partitions: t.Optional[int] = None
    ) -> dd.DataFrame:
        """
        Convert a pandas DataFrame to a Dask DataFrame with optimized partitioning.

        Args:
            df: Pandas DataFrame to convert
            num_partitions: Number of partitions to use (None for auto)

        Returns:
            Dask DataFrame
        """
        if not self.use_dask or self._dask_client is None:
            raise ValueError('Dask client not initialized or Dask is disabled')

        try:
            # Calculate optimal partition size (aim for ~100MB per partition)
            memory_per_row = df.memory_usage(deep=True).sum() / len(df)
            partition_size = max(int(100 * 1024 * 1024 / memory_per_row), 1000)

            if num_partitions is None:
                num_partitions = max(1, len(df) // partition_size)

            self.log(
                f'Converting DataFrame to Dask DataFrame with {num_partitions} partitions'
            )

            # Convert to Dask DataFrame
            dask_df = dd.from_pandas(df, npartitions=num_partitions)

            return dask_df

        except Exception as e:
            self.log(f'Error converting to Dask DataFrame: {str(e)}')
            raise

    @property
    def client(self) -> t.Optional[Client]:
        """Get the current Dask client."""
        return self._dask_client

    @property
    def is_active(self) -> bool:
        """Check if Dask is enabled and client is available."""
        return self.use_dask and self._dask_client is not None

    def log(self, message: str) -> None:
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
