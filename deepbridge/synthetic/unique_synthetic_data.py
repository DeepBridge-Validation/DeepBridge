"""
Function to generate unique synthetic data not present in the original dataset.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Any

def generate_unique_data(
    dataset,
    num_samples: int = 100,
    method: str = "gaussian",
    max_iterations: int = 10,
    similarity_threshold: float = 0.98,
    random_state: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generate synthetic data that is distinct from the original dataset.
    
    This function creates synthetic data samples that are not too similar to 
    any samples in the original dataset. It uses a combined approach of checking
    for exact duplicates and measuring similarity between samples.
    
    Args:
        dataset: A DBDataset instance containing the original data
        num_samples: Number of unique samples to generate
        method: Generator method ('gaussian' or 'ctgan')
        max_iterations: Maximum number of iterations to attempt generating unique samples
        similarity_threshold: Threshold for considering samples too similar (0.0-1.0)
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters to pass to the generator
        
    Returns:
        pd.DataFrame: DataFrame containing unique synthetic samples
        
    Raises:
        ValueError: If unable to generate the requested number of unique samples
    """
    from deepbridge.synthetic.synthesizer import synthesize
    
    # Ensure we have the original data
    if not hasattr(dataset, '_original_data'):
        raise ValueError("Dataset must be a DBDataset with _original_data attribute")
    
    # Get the original data
    original_data = dataset._original_data
    
    # Initial batch size - generate more samples than needed to account for duplicates
    batch_size = min(num_samples * 2, num_samples + 1000)
    
    # Get categorical and numerical features
    categorical_features = getattr(dataset, 'categorical_features', [])
    numerical_features = getattr(dataset, 'numerical_features', [])
    
    # If neither is available, infer them
    if not categorical_features and not numerical_features:
        categorical_features = [col for col in original_data.columns 
                               if original_data[col].dtype == 'object' 
                               or original_data[col].dtype == 'category'
                               or (original_data[col].dtype == 'int64' and original_data[col].nunique() < 20)]
        numerical_features = [col for col in original_data.columns if col not in categorical_features]
    
    # Track unique samples
    unique_samples = pd.DataFrame(columns=original_data.columns)
    
    # Set random seed if provided
    if random_state is not None:
        np.random.seed(random_state)
    
    # Try to generate unique samples
    for iteration in range(max_iterations):
        # Generate synthetic samples
        current_batch_size = min(batch_size, num_samples * 3)  # Increase batch size in later iterations
        synthetic_batch = synthesize(
            dataset, 
            method=method, 
            num_samples=current_batch_size,
            random_state=random_state + iteration if random_state is not None else None,
            return_quality_metrics=False,
            print_metrics=False,
            **kwargs
        )
        
        # Remove exact duplicates within synthetic data
        synthetic_batch = synthetic_batch.drop_duplicates().reset_index(drop=True)
        
        # Remove samples that are identical to any in the original data
        synthetic_batch = _remove_exact_duplicates(synthetic_batch, original_data)
        
        # Remove samples that are too similar to the original data
        synthetic_batch = _remove_similar_samples(
            synthetic_batch, 
            original_data, 
            categorical_features, 
            numerical_features, 
            similarity_threshold
        )
        
        # Add the new unique samples to our collection
        unique_samples = pd.concat([unique_samples, synthetic_batch], ignore_index=True)
        
        # Remove any duplicates that might have been added
        unique_samples = unique_samples.drop_duplicates().reset_index(drop=True)
        
        # Check if we have enough samples
        if len(unique_samples) >= num_samples:
            # Return exactly the number of samples requested
            return unique_samples.head(num_samples)
        
        # Increase batch size for next iteration
        batch_size = int(batch_size * 1.5)
    
    # If we get here, we couldn't generate enough unique samples
    if len(unique_samples) > 0:
        print(f"Warning: Could only generate {len(unique_samples)} unique samples out of {num_samples} requested")
        return unique_samples
    else:
        raise ValueError(f"Failed to generate unique samples after {max_iterations} iterations")

def _remove_exact_duplicates(synthetic_data: pd.DataFrame, original_data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove synthetic samples that exactly match any sample in the original data.
    
    Args:
        synthetic_data: DataFrame of synthetic samples
        original_data: DataFrame of original data
        
    Returns:
        DataFrame with duplicates removed
    """
    # For categorical data: convert to tuples and use set operations
    synthetic_tuples = set(map(tuple, synthetic_data.values))
    original_tuples = set(map(tuple, original_data.values))
    
    # Remove any exact duplicates
    unique_tuples = synthetic_tuples - original_tuples
    
    # Convert back to DataFrame
    if not unique_tuples:
        return pd.DataFrame(columns=synthetic_data.columns)
    
    return pd.DataFrame(list(unique_tuples), columns=synthetic_data.columns)

def _remove_similar_samples(
    synthetic_data: pd.DataFrame, 
    original_data: pd.DataFrame,
    categorical_features: List[str],
    numerical_features: List[str],
    similarity_threshold: float
) -> pd.DataFrame:
    """
    Remove synthetic samples that are too similar to any sample in the original data.
    
    Args:
        synthetic_data: DataFrame of synthetic samples
        original_data: DataFrame of original data
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        similarity_threshold: Threshold for considering samples too similar (0.0-1.0)
        
    Returns:
        DataFrame with too-similar samples removed
    """
    if synthetic_data.empty:
        return synthetic_data
    
    # If we have too many rows, use a sampling approach for efficiency
    max_comparisons = 10000
    if len(synthetic_data) * len(original_data) > max_comparisons:
        # Sample some rows from both datasets for comparison
        sample_size_original = min(len(original_data), int(np.sqrt(max_comparisons)))
        sample_size_synthetic = min(len(synthetic_data), int(max_comparisons / sample_size_original))
        
        original_sample = original_data.sample(n=sample_size_original)
        too_similar_indices = []
        
        # Check similarity in batches
        batch_size = min(sample_size_synthetic, 1000)
        for i in range(0, len(synthetic_data), batch_size):
            synthetic_batch = synthetic_data.iloc[i:i+batch_size]
            similar_in_batch = []
            
            for j, row in synthetic_batch.iterrows():
                # Check similarity with samples from original data
                if _is_too_similar(row, original_sample, categorical_features, numerical_features, similarity_threshold):
                    similar_in_batch.append(j)
            
            too_similar_indices.extend(similar_in_batch)
        
        # Remove too similar samples
        return synthetic_data.drop(too_similar_indices).reset_index(drop=True)
    else:
        # For smaller datasets, check every combination
        unique_indices = []
        
        for i, row in synthetic_data.iterrows():
            if not _is_too_similar(row, original_data, categorical_features, numerical_features, similarity_threshold):
                unique_indices.append(i)
        
        return synthetic_data.loc[unique_indices].reset_index(drop=True)

def _is_too_similar(
    sample: pd.Series, 
    dataset: pd.DataFrame,
    categorical_features: List[str],
    numerical_features: List[str],
    threshold: float
) -> bool:
    """
    Check if a sample is too similar to any sample in a dataset.
    
    Args:
        sample: Series representing a data sample
        dataset: DataFrame to compare against
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        threshold: Similarity threshold (0.0-1.0)
        
    Returns:
        bool: True if the sample is too similar to any sample in the dataset
    """
    # For very small datasets, use a more precise approach
    if len(dataset) < 100:
        # Calculate similarity for each row
        for _, row in dataset.iterrows():
            similarity = _calculate_similarity(sample, row, categorical_features, numerical_features)
            if similarity >= threshold:
                return True
        return False
    else:
        # For larger datasets, use a faster approach with vectorized operations
        
        # Compare categorical features
        cat_similarity = 0.0
        if categorical_features:
            # Count matching categorical values for each row
            cat_matches = 0
            cat_count = len(categorical_features)
            
            if cat_count > 0:
                for feature in categorical_features:
                    if feature in sample and feature in dataset:
                        cat_matches += (dataset[feature] == sample[feature]).mean()
                
                cat_similarity = cat_matches / cat_count
        
        # Compare numerical features
        num_similarity = 0.0
        if numerical_features:
            # Calculate similarity based on normalized differences for numerical features
            num_diffs = 0
            num_count = len(numerical_features)
            
            if num_count > 0:
                for feature in numerical_features:
                    if feature in sample and feature in dataset:
                        # Get feature range
                        feature_min = dataset[feature].min()
                        feature_max = dataset[feature].max()
                        feature_range = max(feature_max - feature_min, 1e-10)  # Avoid division by zero
                        
                        # Calculate normalized differences
                        normalized_diffs = (1 - abs(dataset[feature] - sample[feature]) / feature_range)
                        # Clip values to [0, 1] range
                        normalized_diffs = normalized_diffs.clip(0, 1)
                        # Take maximum similarity with any sample
                        num_diffs += normalized_diffs.max()
                
                num_similarity = num_diffs / num_count
        
        # Combine similarities
        total_features = len(categorical_features) + len(numerical_features)
        if total_features > 0:
            overall_similarity = (
                (len(categorical_features) * cat_similarity + 
                 len(numerical_features) * num_similarity) / 
                total_features
            )
            
            return overall_similarity >= threshold
        else:
            return False

def _calculate_similarity(
    sample1: pd.Series, 
    sample2: pd.Series,
    categorical_features: List[str],
    numerical_features: List[str]
) -> float:
    """
    Calculate similarity between two samples.
    
    Args:
        sample1: First sample
        sample2: Second sample
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        
    Returns:
        float: Similarity score (0.0-1.0)
    """
    total_features = len(categorical_features) + len(numerical_features)
    if total_features == 0:
        return 0.0
    
    similarity = 0.0
    
    # Check categorical features - exact match is 1.0, otherwise 0.0
    for feature in categorical_features:
        if feature in sample1 and feature in sample2:
            if sample1[feature] == sample2[feature]:
                similarity += 1.0
    
    # Check numerical features - normalized similarity
    for feature in numerical_features:
        if feature in sample1 and feature in sample2:
            # Get feature range from global stats
            feature_range = 1.0  # Default to 1.0 if no range info
            
            # Calculate similarity as 1 - normalized difference
            diff = abs(float(sample1[feature]) - float(sample2[feature])) / feature_range
            feature_similarity = max(0.0, 1.0 - diff)
            similarity += feature_similarity
    
    # Return average similarity
    return similarity / total_features