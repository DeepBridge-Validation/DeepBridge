"""
Core functions for calculating similarity between original and synthetic data.
"""

import gc
import typing as t

# Import Dask modules
import dask
import dask.array as da
import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask.distributed import Client, progress, wait
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Define standalone functions for Dask compatibility


def _create_preprocessor(
    numerical_columns: t.List[str], categorical_columns: t.List[str]
) -> ColumnTransformer:
    """
    Create column transformer for preprocessing data.

    Args:
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names

    Returns:
        ColumnTransformer for preprocessing
    """
    transformers = []

    if numerical_columns:
        transformers.append(('num', StandardScaler(), numerical_columns))

    if categorical_columns:
        transformers.append(
            (
                'cat',
                OneHotEncoder(handle_unknown='ignore'),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder='drop')


def _preprocess_data(
    data: pd.DataFrame, preprocessor: ColumnTransformer
) -> np.ndarray:
    """
    Preprocess data using a fitted transformer.

    Args:
        data: DataFrame to preprocess
        preprocessor: Fitted ColumnTransformer

    Returns:
        Preprocessed data array
    """
    # Transform data
    transformed = preprocessor.transform(data)

    # Handle sparse matrices
    if hasattr(transformed, 'toarray'):
        transformed = transformed.toarray()

    return transformed


def _compute_nn_distances(
    original_transformed: np.ndarray,
    synthetic_transformed: np.ndarray,
    metric: str = 'euclidean',
    n_neighbors: int = 5,
    n_jobs: int = -1,
) -> np.ndarray:
    """
    Compute nearest neighbor distances.

    Args:
        original_transformed: Preprocessed original data
        synthetic_transformed: Preprocessed synthetic data
        metric: Distance metric
        n_neighbors: Number of neighbors
        n_jobs: Number of parallel jobs

    Returns:
        Distances array
    """
    nn_model = NearestNeighbors(
        n_neighbors=min(n_neighbors, len(original_transformed)),
        metric=metric,
        n_jobs=n_jobs,
    )
    nn_model.fit(original_transformed)
    distances, _ = nn_model.kneighbors(synthetic_transformed)
    return distances


def _calculate_similarity_scores(distances: np.ndarray) -> np.ndarray:
    """
    Calculate similarity scores from distances.

    Args:
        distances: Array of distances to nearest neighbors

    Returns:
        Array of similarity scores
    """
    if len(distances) == 0:
        return np.array([])

    max_dist = np.max(distances)
    if max_dist > 0:
        normalized_distances = distances / max_dist
        similarities = 1 - normalized_distances.mean(axis=1)
    else:
        similarities = np.ones(len(distances))

    return similarities


# Standalone function for Dask serialization
def process_similarity_chunk_standalone(
    original_data_dict: dict,
    synthetic_chunk_dict: dict,
    categorical_columns: t.List[str],
    numerical_columns: t.List[str],
    metric: str = 'euclidean',
    n_neighbors: int = 5,
    chunk_id: int = 0,
    verbose: bool = False,
) -> t.Tuple[list, list]:
    """
    Process a single chunk for similarity calculation - Standalone version for Dask.

    Args:
        original_data_dict: Dictionary representation of original data
        synthetic_chunk_dict: Dictionary representation of synthetic chunk
        categorical_columns: List of categorical columns
        numerical_columns: List of numerical columns
        metric: Distance metric
        n_neighbors: Number of neighbors
        chunk_id: ID of the chunk
        verbose: Whether to show verbose output

    Returns:
        Tuple of (chunk_indices, similarity_scores)
    """
    if verbose:
        print(
            f'Processing chunk {chunk_id} with {len(synthetic_chunk_dict)} samples'
        )

    # Convert dictionaries back to DataFrames
    original_data = pd.DataFrame.from_dict(original_data_dict)
    synthetic_chunk = pd.DataFrame.from_dict(synthetic_chunk_dict)

    # Create and fit preprocessor
    preprocessor = _create_preprocessor(numerical_columns, categorical_columns)
    preprocessor.fit(original_data)

    # Preprocess data
    original_transformed = _preprocess_data(original_data, preprocessor)
    synthetic_transformed = _preprocess_data(synthetic_chunk, preprocessor)

    # Compute distances
    distances = _compute_nn_distances(
        original_transformed,
        synthetic_transformed,
        metric=metric,
        n_neighbors=n_neighbors,
        n_jobs=1,  # Use single thread within each Dask task
    )

    # Calculate similarity scores
    similarity_scores = _calculate_similarity_scores(distances)

    # Return indices and scores as lists for serialization
    return synthetic_chunk.index.tolist(), similarity_scores.tolist()


# Standalone function for Dask similarity filtering
def filter_chunk_by_similarity_standalone(
    original_data_dict: dict,
    synthetic_chunk_dict: dict,
    threshold: float,
    categorical_columns: t.List[str],
    numerical_columns: t.List[str],
    metric: str = 'euclidean',
    n_neighbors: int = 5,
    chunk_id: int = 0,
    verbose: bool = False,
) -> list:
    """
    Filter a chunk based on similarity threshold - Standalone version for Dask.

    Args:
        original_data_dict: Dictionary representation of original data
        synthetic_chunk_dict: Dictionary representation of synthetic chunk
        threshold: Similarity threshold
        categorical_columns: List of categorical columns
        numerical_columns: List of numerical columns
        metric: Distance metric
        n_neighbors: Number of neighbors
        chunk_id: ID of the chunk
        verbose: Whether to show verbose output

    Returns:
        List of indices to keep
    """
    # Convert dictionaries back to DataFrames
    original_data = pd.DataFrame.from_dict(original_data_dict)
    synthetic_chunk = pd.DataFrame.from_dict(synthetic_chunk_dict)

    # Get chunk indices and similarity scores
    chunk_indices, similarity_scores = process_similarity_chunk_standalone(
        original_data_dict,
        synthetic_chunk_dict,
        categorical_columns,
        numerical_columns,
        metric,
        n_neighbors,
        chunk_id,
        verbose,
    )

    # Convert back to numpy arrays
    chunk_indices = np.array(chunk_indices)
    similarity_scores = np.array(similarity_scores)

    # Filter based on threshold
    keep_mask = similarity_scores < threshold
    keep_indices = chunk_indices[keep_mask].tolist()

    return keep_indices


def calculate_similarity(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    categorical_columns: t.Optional[t.List[str]] = None,
    numerical_columns: t.Optional[t.List[str]] = None,
    metric: str = 'euclidean',
    n_neighbors: int = 5,
    sample_size: int = 10000,
    random_state: t.Optional[int] = None,
    n_jobs: int = -1,
    verbose: bool = False,
    use_dask: bool = False,
    dask_client: t.Optional[Client] = None,
    **kwargs,
) -> pd.Series:
    """
    Calculate similarity scores between synthetic samples and nearest original samples.

    Args:
        original_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        metric: Distance metric for nearest neighbors
        n_neighbors: Number of nearest neighbors to consider
        sample_size: Maximum number of samples to use
        random_state: Random seed for sampling
        n_jobs: Number of parallel jobs to use
        verbose: Whether to print progress information
        use_dask: Whether to use Dask for distributed computation
        dask_client: Existing Dask client to use (if None, will not use Dask even if use_dask=True)
        **kwargs: Additional parameters

    Returns:
        Series with similarity scores for each synthetic sample
    """
    if verbose:
        print(f'Calculating similarity between original and synthetic data...')

    # Sample data if it's too large
    if len(original_data) > sample_size:
        original_sample = original_data.sample(
            sample_size, random_state=random_state
        )
    else:
        original_sample = original_data

    if len(synthetic_data) > sample_size:
        synthetic_sample = synthetic_data.sample(
            sample_size, random_state=random_state
        )
    else:
        synthetic_sample = synthetic_data

    # Ensure columns match between datasets
    common_columns = list(
        set(original_sample.columns) & set(synthetic_sample.columns)
    )
    original_sample = original_sample[common_columns]
    synthetic_sample = synthetic_sample[common_columns]

    # Infer column types if not provided
    if categorical_columns is None and numerical_columns is None:
        categorical_columns = []
        numerical_columns = []

        for col in common_columns:
            if (
                pd.api.types.is_numeric_dtype(original_sample[col])
                and original_sample[col].nunique() > 10
            ):
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)

        if verbose:
            print(
                f'Inferred {len(numerical_columns)} numerical columns and {len(categorical_columns)} categorical columns'
            )

    elif categorical_columns is None:
        categorical_columns = [
            col for col in common_columns if col not in numerical_columns
        ]
    elif numerical_columns is None:
        numerical_columns = [
            col for col in common_columns if col not in categorical_columns
        ]

    # Check if we should use Dask
    should_use_dask = (
        use_dask
        and dask_client is not None
        and (len(original_sample) > 5000 or len(synthetic_sample) > 5000)
    )

    if should_use_dask:
        try:
            if verbose:
                print('Using Dask for distributed similarity calculation')

            # Determine optimal chunk size
            n_workers = len(dask_client.scheduler_info()['workers'])
            chunk_size = max(
                100, min(1000, len(synthetic_sample) // (n_workers * 2))
            )
            n_chunks = (len(synthetic_sample) + chunk_size - 1) // chunk_size

            if verbose:
                print(
                    f'Processing data in {n_chunks} chunks of size ~{chunk_size}'
                )

            # Convert DataFrames to dictionaries for serialization
            original_data_dict = original_sample.to_dict()

            # Process in chunks
            all_indices = []
            all_scores = []

            # Create tasks for each chunk
            tasks = []
            for i in range(n_chunks):
                start = i * chunk_size
                end = min(start + chunk_size, len(synthetic_sample))
                chunk = synthetic_sample.iloc[start:end]

                # Convert chunk to dictionary for serialization
                synthetic_chunk_dict = chunk.to_dict()

                # Create a delayed task with standalone function
                task = dask.delayed(process_similarity_chunk_standalone)(
                    original_data_dict,
                    synthetic_chunk_dict,
                    categorical_columns,
                    numerical_columns,
                    metric,
                    n_neighbors,
                    i,
                    verbose,
                )
                tasks.append(task)

            # Compute all tasks
            results = dask_client.compute(tasks)

            # Process results
            for chunk_indices, chunk_scores in results:
                all_indices.extend(chunk_indices)
                all_scores.extend(chunk_scores)

            # Create Series with results
            return pd.Series(all_scores, index=all_indices)

        except Exception as e:
            if verbose:
                print(f'Error using Dask for similarity calculation: {str(e)}')
                print('Falling back to standard method')

    # Standard approach (non-Dask)
    try:
        if verbose:
            print('Using standard approach for similarity calculation')

        # Create preprocessing pipeline
        transformers = []

        if numerical_columns:
            transformers.append(('num', StandardScaler(), numerical_columns))

        if categorical_columns:
            transformers.append(
                (
                    'cat',
                    OneHotEncoder(handle_unknown='ignore'),
                    categorical_columns,
                )
            )

        if not transformers:
            raise ValueError('No valid columns for preprocessing')

        preprocessor = ColumnTransformer(
            transformers=transformers, remainder='drop'
        )

        # Transform data
        original_transformed = preprocessor.fit_transform(original_sample)
        synthetic_transformed = preprocessor.transform(synthetic_sample)

        # Handle sparse matrices
        if hasattr(original_transformed, 'toarray'):
            original_transformed = original_transformed.toarray()
        if hasattr(synthetic_transformed, 'toarray'):
            synthetic_transformed = synthetic_transformed.toarray()

        # Calculate distances
        distances = _compute_nn_distances(
            original_transformed,
            synthetic_transformed,
            metric=metric,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
        )

        # Calculate similarities
        similarities = _calculate_similarity_scores(distances)

        return pd.Series(similarities, index=synthetic_sample.index)

    except Exception as e:
        if verbose:
            print(f'Error in similarity calculation: {str(e)}')

        # Try fallback method with only numerical columns
        if len(numerical_columns) > 0:
            if verbose:
                print('Falling back to numerical-only similarity calculation')

            # Process only numerical columns
            numerical_data_original = (
                original_sample[numerical_columns].fillna(0).values
            )
            numerical_data_synthetic = (
                synthetic_sample[numerical_columns].fillna(0).values
            )

            # Scale numerical data
            scaler = StandardScaler()
            numerical_data_original = scaler.fit_transform(
                numerical_data_original
            )
            numerical_data_synthetic = scaler.transform(
                numerical_data_synthetic
            )

            # Calculate distances
            distances = _compute_nn_distances(
                numerical_data_original,
                numerical_data_synthetic,
                metric=metric,
                n_neighbors=n_neighbors,
                n_jobs=n_jobs,
            )

            # Calculate similarities
            similarities = _calculate_similarity_scores(distances)

            return pd.Series(similarities, index=synthetic_sample.index)
        else:
            # No fallback possible
            raise ValueError(
                'Could not calculate similarity with available columns'
            ) from e


def filter_by_similarity(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    threshold: float = 0.8,
    categorical_columns: t.Optional[t.List[str]] = None,
    numerical_columns: t.Optional[t.List[str]] = None,
    batch_size: int = 5000,
    n_jobs: int = -1,
    random_state: t.Optional[int] = None,
    verbose: bool = True,
    use_dask: bool = False,
    dask_client: t.Optional[Client] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Filter synthetic data to remove samples that are too similar to original data.

    Memory-efficient implementation that processes data in batches with optional Dask support.

    Args:
        original_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        threshold: Similarity threshold (0.0-1.0), higher means more similar
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        batch_size: Size of batches for processing
        n_jobs: Number of parallel jobs
        random_state: Random seed for sampling
        verbose: Whether to print progress information
        use_dask: Whether to use Dask for distributed computation
        dask_client: Existing Dask client to use
        **kwargs: Additional parameters

    Returns:
        Filtered synthetic data
    """
    if verbose:
        print(
            f'Filtering synthetic data with similarity threshold: {threshold}'
        )

    # Determine if we should use Dask
    should_use_dask = (
        use_dask and dask_client is not None and len(synthetic_data) > 10000
    )

    # Ensure columns match between datasets
    common_columns = list(
        set(original_data.columns) & set(synthetic_data.columns)
    )
    original_data_common = original_data[common_columns]

    # Infer column types if not provided
    if categorical_columns is None and numerical_columns is None:
        categorical_columns = []
        numerical_columns = []

        for col in common_columns:
            if (
                pd.api.types.is_numeric_dtype(original_data[col])
                and original_data[col].nunique() > 10
            ):
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)

        if verbose:
            print(
                f'Inferred {len(numerical_columns)} numerical columns and {len(categorical_columns)} categorical columns'
            )

    elif categorical_columns is None:
        categorical_columns = [
            col for col in common_columns if col not in numerical_columns
        ]
    elif numerical_columns is None:
        numerical_columns = [
            col for col in common_columns if col not in categorical_columns
        ]

    if should_use_dask:
        try:
            if verbose:
                print(f'Using Dask for distributed similarity filtering')

            # Calculate optimal batch size based on available memory and workers
            n_workers = len(dask_client.scheduler_info()['workers'])
            optimal_batch_size = min(
                batch_size, max(1000, len(synthetic_data) // (n_workers * 2))
            )

            if verbose:
                print(
                    f'Using {n_workers} Dask workers with batch size {optimal_batch_size}'
                )

            # Calculate the number of batches
            n_batches = (
                len(synthetic_data) + optimal_batch_size - 1
            ) // optimal_batch_size

            if verbose:
                print(f'Processing {n_batches} batches in parallel')

            # Convert original data to dictionary for serialization
            original_data_dict = original_data_common.to_dict()

            # Create tasks for batch processing
            tasks = []
            for i in range(n_batches):
                start_idx = i * optimal_batch_size
                end_idx = min(
                    (i + 1) * optimal_batch_size, len(synthetic_data)
                )
                batch = synthetic_data.iloc[start_idx:end_idx]

                # Convert batch to dictionary for serialization
                synthetic_chunk_dict = batch.to_dict()

                # Create a delayed task with standalone function
                task = dask.delayed(filter_chunk_by_similarity_standalone)(
                    original_data_dict,
                    synthetic_chunk_dict,
                    threshold,
                    categorical_columns,
                    numerical_columns,
                    'euclidean',
                    5,
                    i,
                    False,
                )
                tasks.append(task)

            # Compute all tasks
            if verbose:
                print('Computing tasks...')

            all_keep_indices = dask_client.compute(tasks)

            # Flatten the list of lists
            keep_indices = []
            for indices in all_keep_indices:
                keep_indices.extend(indices)

            # Create filtered dataframe
            filtered_data = synthetic_data.loc[keep_indices]

            if verbose:
                removed_count = len(synthetic_data) - len(filtered_data)
                removed_percentage = (
                    removed_count / len(synthetic_data) * 100
                    if len(synthetic_data) > 0
                    else 0
                )
                print(
                    f'Removed {removed_count} samples ({removed_percentage:.2f}%) with similarity ≥ {threshold}'
                )

            return filtered_data

        except Exception as e:
            if verbose:
                print(f'Error using Dask for similarity filtering: {str(e)}')
                print('Falling back to standard method')

    # Process in batches for memory efficiency
    if len(synthetic_data) > batch_size:
        # Calculate the number of batches
        n_batches = (len(synthetic_data) + batch_size - 1) // batch_size

        if verbose:
            print(f'Processing {n_batches} batches of size {batch_size}')

        # Process each batch
        keep_indices = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(synthetic_data))

            if verbose:
                print(
                    f'Processing batch {i+1}/{n_batches} (samples {start_idx}-{end_idx})'
                )

            # Get the current batch
            batch = synthetic_data.iloc[start_idx:end_idx]

            # Calculate similarity for this batch
            similarity_scores = calculate_similarity(
                original_data=original_data,
                synthetic_data=batch,
                categorical_columns=categorical_columns,
                numerical_columns=numerical_columns,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=False,
                **kwargs,
            )

            # Keep samples below threshold
            batch_keep = similarity_scores < threshold
            batch_indices = batch.index[batch_keep]
            keep_indices.extend(batch_indices)

            # Clear memory
            gc.collect()

        # Create filtered dataframe
        filtered_data = synthetic_data.loc[keep_indices]
    else:
        # Calculate similarity scores for entire dataset
        similarity_scores = calculate_similarity(
            original_data=original_data,
            synthetic_data=synthetic_data,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=verbose,
            **kwargs,
        )

        # Filter based on threshold
        mask = similarity_scores < threshold
        filtered_data = synthetic_data.loc[mask]

    if verbose:
        removed_count = len(synthetic_data) - len(filtered_data)
        removed_percentage = (
            removed_count / len(synthetic_data) * 100
            if len(synthetic_data) > 0
            else 0
        )
        print(
            f'Removed {removed_count} samples ({removed_percentage:.2f}%) with similarity ≥ {threshold}'
        )

    return filtered_data
