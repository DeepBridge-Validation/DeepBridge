"""
Core functions for calculating similarity between original and synthetic data.
"""

import pandas as pd
import numpy as np
import typing as t
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
import gc

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
    **kwargs
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
        **kwargs: Additional parameters
        
    Returns:
        Series with similarity scores for each synthetic sample
    """
    if verbose:
        print(f"Calculating similarity between original and synthetic data...")
        
    # Sample data if it's too large
    if len(original_data) > sample_size:
        original_sample = original_data.sample(sample_size, random_state=random_state)
    else:
        original_sample = original_data
    
    if len(synthetic_data) > sample_size:
        synthetic_sample = synthetic_data.sample(sample_size, random_state=random_state)
    else:
        synthetic_sample = synthetic_data
    
    # Ensure columns match between datasets
    common_columns = list(set(original_sample.columns) & set(synthetic_sample.columns))
    original_sample = original_sample[common_columns]
    synthetic_sample = synthetic_sample[common_columns]
    
    # Infer column types if not provided
    if categorical_columns is None and numerical_columns is None:
        categorical_columns = []
        numerical_columns = []
        
        for col in common_columns:
            if pd.api.types.is_numeric_dtype(original_sample[col]) and \
               original_sample[col].nunique() > 10:
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)
                
        if verbose:
            print(f"Inferred {len(numerical_columns)} numerical columns and {len(categorical_columns)} categorical columns")
            
    elif categorical_columns is None:
        categorical_columns = [col for col in common_columns if col not in numerical_columns]
    elif numerical_columns is None:
        numerical_columns = [col for col in common_columns if col not in categorical_columns]
    
    # Create preprocessor
    transformers = []
    
    if numerical_columns:
        transformers.append(('num', StandardScaler(), numerical_columns))
        
    if categorical_columns:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns))
    
    if not transformers:
        raise ValueError("No valid columns for preprocessing")
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
    
    # Transform data
    try:
        if verbose:
            print("Preprocessing data...")
            
        original_transformed = preprocessor.fit_transform(original_sample)
        synthetic_transformed = preprocessor.transform(synthetic_sample)
        
        # Handle sparse matrices
        if hasattr(original_transformed, 'toarray'):
            original_transformed = original_transformed.toarray()
        if hasattr(synthetic_transformed, 'toarray'):
            synthetic_transformed = synthetic_transformed.toarray()
            
    except Exception as e:
        if verbose:
            print(f"Error in data transformation: {str(e)}")
            print("Falling back to numerical columns only")
            
        # Fallback to simple imputation and scaling for numerical data only
        if not numerical_columns:
            raise ValueError("No numerical columns available for fallback transformation")
            
        numerical_data_original = original_sample[numerical_columns].fillna(0).values
        numerical_data_synthetic = synthetic_sample[numerical_columns].fillna(0).values
        
        # Scale numerical data
        scaler = StandardScaler()
        numerical_data_original = scaler.fit_transform(numerical_data_original)
        numerical_data_synthetic = scaler.transform(numerical_data_synthetic)
        
        # Use only numerical data for similarity
        original_transformed = numerical_data_original
        synthetic_transformed = numerical_data_synthetic
    
    # Initialize and fit nearest neighbors model
    if verbose:
        print(f"Finding {n_neighbors} nearest neighbors...")
        
    nn_model = NearestNeighbors(n_neighbors=min(n_neighbors, len(original_transformed)), 
                               metric=metric, n_jobs=n_jobs)
    nn_model.fit(original_transformed)
    
    # Calculate distances to nearest neighbors
    distances, indices = nn_model.kneighbors(synthetic_transformed)
    
    # Calculate similarity score (inversely related to distance)
    # Normalize distances to [0, 1] range, then invert
    if len(distances) > 0:
        max_dist = np.max(distances)
        if max_dist > 0:
            normalized_distances = distances / max_dist
            similarities = 1 - normalized_distances.mean(axis=1)
        else:
            similarities = np.ones(len(distances))
    else:
        similarities = np.array([])
        
    if verbose:
        print(f"Similarity calculation complete. Average similarity: {similarities.mean():.4f}")
    
    return pd.Series(similarities, index=synthetic_sample.index)

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
    **kwargs
) -> pd.DataFrame:
    """
    Filter synthetic data to remove samples that are too similar to original data.
    
    Memory-efficient implementation that processes data in batches.
    
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
        **kwargs: Additional parameters
        
    Returns:
        Filtered synthetic data
    """
    if verbose:
        print(f"Filtering synthetic data with similarity threshold: {threshold}")
    
    # Process in batches for memory efficiency
    if len(synthetic_data) > batch_size:
        # Calculate the number of batches
        n_batches = (len(synthetic_data) + batch_size - 1) // batch_size
        
        if verbose:
            print(f"Processing {n_batches} batches of size {batch_size}")
        
        # Process each batch
        keep_indices = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(synthetic_data))
            
            if verbose:
                print(f"Processing batch {i+1}/{n_batches} (samples {start_idx}-{end_idx})")
            
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
                **kwargs
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
            **kwargs
        )
        
        # Filter based on threshold
        mask = similarity_scores < threshold
        filtered_data = synthetic_data.loc[mask]
    
    if verbose:
        removed_count = len(synthetic_data) - len(filtered_data)
        removed_percentage = removed_count / len(synthetic_data) * 100 if len(synthetic_data) > 0 else 0
        print(f"Removed {removed_count} samples ({removed_percentage:.2f}%) with similarity ≥ {threshold}")
    
    return filtered_data