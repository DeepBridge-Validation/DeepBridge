"""
Unified interface for synthetic data generation.

This module provides a simplified interface for generating synthetic data
from DeepBridge datasets using different generation methods, with options
for controlling similarity to the original data.
"""

import pandas as pd
import warnings
import numpy as np
from typing import Union, Optional, Dict, Any, List, Tuple

def synthesize(
    dataset,
    method: str = "gaussian",
    num_samples: int = 500,
    random_state: int = 42,
    preserve_dtypes: bool = True,
    return_quality_metrics: bool = True,
    print_metrics: bool = True,
    suppress_warnings: bool = True,
    similarity_threshold: float = 1.0,
    max_iterations: int = 10,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Generate synthetic data from a DBDataset using the specified method.
    
    If similarity_threshold < 1.0, it will generate unique synthetic data
    that is dissimilar from the original dataset, with the similarity being
    controlled by the similarity_threshold parameter.
    
    Args:
        dataset: A DBDataset instance containing the data to synthesize
        method: Generation method ('gaussian' or 'ctgan')
        num_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        preserve_dtypes: Whether to preserve data types of original data
        return_quality_metrics: Whether to return quality evaluation metrics
        print_metrics: Whether to print quality evaluation metrics
        suppress_warnings: Whether to suppress SDV deprecation warnings
        similarity_threshold: Threshold for considering samples too similar (0.0-1.0)
            - If 1.0 (default): Regular synthesis without uniqueness check
            - If < 1.0: Generate unique samples (lower values = more unique)
        max_iterations: Maximum number of iterations for unique data generation
        **kwargs: Additional parameters to pass to the specific generator
    
    Returns:
        If return_quality_metrics is True:
            tuple: (synthetic_data, quality_metrics)
                - synthetic_data (pd.DataFrame): Synthetic data with the same structure as the original data
                - quality_metrics (dict): Dictionary containing quality evaluation metrics
        Else:
            pd.DataFrame: Synthetic data with the same structure as the original data
    
    Raises:
        ValueError: If an invalid method is specified or required data is missing
    
    Examples:
        >>> from deepbridge.core.db_data import DBDataset
        >>> from deepbridge.synthetic import synthesize
        >>> 
        >>> # Create a DBDataset with your data and model
        >>> dataset = DBDataset(data=df, target_column='target', model=model)
        >>> 
        >>> # Generate regular synthetic data using Gaussian Copula with quality metrics
        >>> synthetic_data, quality_metrics = synthesize(dataset, num_samples=500)
        >>> 
        >>> # Generate unique synthetic data (different from original dataset)
        >>> unique_synthetic_data, quality_metrics = synthesize(
        ...     dataset, 
        ...     method="gaussian",
        ...     num_samples=1000,
        ...     similarity_threshold=0.8,  # Lower values mean more unique data
        ...     return_quality_metrics=True
        ... )
    """
    # Validate method
    valid_methods = ['gaussian', 'ctgan']
    if method.lower() not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}, got '{method}'")
    
    # Validate dataset
    if not hasattr(dataset, '_original_data'):
        raise ValueError("Dataset must be a DBDataset with _original_data attribute")
    
    # Extract data and metadata from dataset
    data = dataset._original_data
    
    # Get categorical features if available
    if hasattr(dataset, 'categorical_features'):
        categorical_columns = dataset.categorical_features
    else:
        categorical_columns = None
    
    # Get target column if available
    if hasattr(dataset, 'target_name'):
        target_column = dataset.target_name
    else:
        target_column = None
    
    # Suppress specific warnings if requested
    original_filter_action = None
    if suppress_warnings:
        # Save original warning filter action
        original_filter_action = warnings.filters[0] if warnings.filters else None
        
        # Filter out specific SDV warnings
        warnings.filterwarnings("ignore", category=FutureWarning, 
                               message="The 'SingleTableMetadata' is deprecated")
        warnings.filterwarnings("ignore", category=UserWarning, 
                               message="We strongly recommend saving the metadata")
    
    try:
        # Check if we need to generate unique data (similarity_threshold < 1.0)
        if similarity_threshold < 1.0:
            return _generate_unique_data(
                dataset=dataset,
                num_samples=num_samples,
                method=method,
                similarity_threshold=similarity_threshold,
                max_iterations=max_iterations,
                random_state=random_state,
                return_quality_metrics=return_quality_metrics,
                print_metrics=print_metrics,
                **kwargs
            )
            
        # Otherwise, regular synthetic data generation
        # Import appropriate generator based on method
        if method.lower() == 'gaussian':
            from deepbridge.synthetic.methods import GaussianCopulaGenerator
            
            # Set default parameters for Gaussian Copula
            default_params = {
                'enforce_min_max_values': True
            }
            
            # Update with any user-provided parameters
            default_params.update(kwargs)
            
            # Create generator
            generator = GaussianCopulaGenerator(
                random_state=random_state,
                preserve_dtypes=preserve_dtypes,
                **default_params
            )
        else:  # method == 'ctgan'
            from deepbridge.synthetic.methods import CTGANGenerator
            
            # Set default parameters for CTGAN
            default_params = {
                'epochs': 300,
                'batch_size': 500,
                'verbose': False
            }
            
            # Update with any user-provided parameters
            default_params.update(kwargs)
            
            # Create generator
            generator = CTGANGenerator(
                random_state=random_state,
                preserve_dtypes=preserve_dtypes,
                **default_params
            )
        
        # Generate synthetic data
        synthetic_data = generator.fit_generate(
            data=data,
            num_samples=num_samples,
            target_column=target_column,
            categorical_columns=categorical_columns
        )
        
        # Evaluate quality if requested
        if return_quality_metrics or print_metrics:
            # Check if the generator has an evaluate_quality method
            if hasattr(generator, 'evaluate_quality') and callable(getattr(generator, 'evaluate_quality')):
                # Use the generator's built-in evaluate_quality method
                quality_metrics = generator.evaluate_quality(
                    real_data=data,
                    synthetic_data=synthetic_data
                )
            else:
                # Use our own quality evaluation implementation directly
                # without trying to instantiate BaseSyntheticGenerator (which is abstract)
                quality_metrics = _evaluate_synthetic_quality(
                    real_data=data,
                    synthetic_data=synthetic_data,
                    categorical_columns=categorical_columns
                )
            
            # Print quality metrics if requested
            if print_metrics:
                _print_quality_metrics(quality_metrics, categorical_columns, target_column)
            
            # Return data with quality metrics if requested
            if return_quality_metrics:
                return synthetic_data, quality_metrics
        
        # Return just the data if quality metrics not requested
        return synthetic_data
    
    finally:
        # Restore original warning filters if we modified them
        if suppress_warnings and original_filter_action:
            warnings.filters[0] = original_filter_action


def _generate_unique_data(
    dataset,
    num_samples: int = 100,
    method: str = "gaussian",
    similarity_threshold: float = 0.8,
    max_iterations: int = 10,
    random_state: Optional[int] = None,
    return_quality_metrics: bool = True,
    print_metrics: bool = False,
    **kwargs
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
    """
    Generate synthetic data that is distinct from the original dataset.
    
    This function creates synthetic data samples that are not too similar to 
    any samples in the original dataset. It uses a combined approach of checking
    for exact duplicates and measuring similarity between samples.
    
    Args:
        dataset: A DBDataset instance containing the original data
        num_samples: Number of unique samples to generate
        method: Generator method ('gaussian' or 'ctgan')
        similarity_threshold: Threshold for considering samples too similar (0.0-1.0)
        max_iterations: Maximum number of iterations to attempt generating unique samples
        random_state: Random seed for reproducibility
        return_quality_metrics: Whether to return quality evaluation metrics
        print_metrics: Whether to print quality evaluation metrics
        **kwargs: Additional parameters to pass to the generator
        
    Returns:
        Same as synthesize function
    """
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
            unique_samples = unique_samples.head(num_samples)

            # Calculate quality metrics if requested
            if return_quality_metrics or print_metrics:
                quality_metrics = _evaluate_synthetic_quality(
                    real_data=original_data,
                    synthetic_data=unique_samples,
                    categorical_columns=categorical_features
                )
                
                # Print quality metrics if requested
                if print_metrics:
                    _print_quality_metrics(quality_metrics, categorical_features, 
                                          getattr(dataset, 'target_name', None))
                
                # Return data with quality metrics if requested
                if return_quality_metrics:
                    return unique_samples, quality_metrics
                
            return unique_samples
        
        # Increase batch size for next iteration
        batch_size = int(batch_size * 1.5)
    
    # If we get here, we couldn't generate enough unique samples
    if len(unique_samples) > 0:
        print(f"Warning: Could only generate {len(unique_samples)} unique samples out of {num_samples} requested")

        # Calculate quality metrics if requested
        if return_quality_metrics:
            quality_metrics = _evaluate_synthetic_quality(
                real_data=original_data,
                synthetic_data=unique_samples,
                categorical_columns=categorical_features
            )
            return unique_samples, quality_metrics
        
        return unique_samples
    else:
        raise ValueError(f"Failed to generate unique samples after {max_iterations} iterations")


def _print_quality_metrics(quality_metrics: Dict, categorical_columns: List[str], target_column: Optional[str] = None) -> None:
    """
    Print quality metrics in a formatted way.
    
    Args:
        quality_metrics: Dictionary with quality metrics
        categorical_columns: List of categorical column names
        target_column: Name of the target column
    """
    print("\nSynthetic Data Quality Evaluation:")
    
    # Print overall metrics if available
    if 'overall' in quality_metrics:
        overall = quality_metrics['overall']
        print(f"Overall metrics:")
        for metric, value in overall.items():
            print(f"  - {metric}: {value:.4f}")
    
    # Extract numerical and categorical features
    if categorical_columns:
        numerical_columns = [col for col in quality_metrics if col not in categorical_columns 
                           and col != 'overall' and col != target_column]
        if target_column and target_column not in categorical_columns:
            numerical_columns.append(target_column)
    else:
        numerical_columns = [col for col in quality_metrics if col != 'overall' 
                           and isinstance(quality_metrics[col].get('mean_diff'), (int, float))]
        
    # Print metrics for numerical features
    numerical_metrics = {k: v for k, v in quality_metrics.items() 
                       if k in numerical_columns}
    if numerical_metrics:
        print("\nNumerical features:")
        for feature, metrics in numerical_metrics.items():
            print(f"  {feature}:")
            print(f"    - mean diff: {metrics.get('mean_diff', 'N/A'):.4f}")
            print(f"    - std diff: {metrics.get('std_diff', 'N/A'):.4f}")
            if 'ks_statistic' in metrics:
                print(f"    - KS statistic: {metrics['ks_statistic']:.4f}")
    
    # Print metrics for categorical features
    if categorical_columns:
        categorical_metrics = {k: v for k, v in quality_metrics.items() 
                            if k in categorical_columns}
        if categorical_metrics:
            print("\nCategorical features:")
            for feature, metrics in categorical_metrics.items():
                print(f"  {feature}:")
                print(f"    - distribution difference: {metrics.get('distribution_difference', 'N/A'):.4f}")
                print(f"    - category count real: {metrics.get('category_count_real', 'N/A')}")
                print(f"    - category count synthetic: {metrics.get('category_count_synthetic', 'N/A')}")


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


def _evaluate_synthetic_quality(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    categorical_columns: Optional[List[str]] = None
) -> Dict:
    """
    Evaluate the quality of synthetic data compared to real data.
    
    This is a fallback method when the generator doesn't have its own evaluate_quality method.
    
    Args:
        real_data: Original data
        synthetic_data: Synthetic data
        categorical_columns: List of categorical column names
    
    Returns:
        Dict: Quality metrics
    """
    import numpy as np
    from scipy import stats
    
    metrics = {}
    
    # Identify column types if not provided
    if categorical_columns is None:
        categorical_columns = []
        for col in real_data.columns:
            if real_data[col].dtype == 'object' or real_data[col].dtype == 'category':
                categorical_columns.append(col)
    
    numerical_columns = [col for col in real_data.columns if col not in categorical_columns]
    
    # Compare basic statistics for numerical columns
    for col in numerical_columns:
        if col in synthetic_data.columns and col in real_data.columns:
            real_mean = real_data[col].mean()
            synth_mean = synthetic_data[col].mean()
            real_std = real_data[col].std()
            synth_std = synthetic_data[col].std()
            
            metrics[col] = {
                'mean_real': real_mean,
                'mean_synthetic': synth_mean,
                'mean_diff': abs(real_mean - synth_mean),
                'mean_diff_pct': abs(real_mean - synth_mean) / (abs(real_mean) + 1e-10) * 100,
                'std_real': real_std,
                'std_synthetic': synth_std,
                'std_diff': abs(real_std - synth_std),
            }
            
            # Perform KS test if we have enough samples
            if len(real_data) >= 5 and len(synthetic_data) >= 5:
                try:
                    ks_stat, ks_pval = stats.ks_2samp(real_data[col].dropna(), synthetic_data[col].dropna())
                    metrics[col]['ks_statistic'] = ks_stat
                    metrics[col]['ks_pvalue'] = ks_pval
                except Exception:
                    pass
        
    # Compare distributions for categorical columns
    for col in categorical_columns:
        if col in synthetic_data.columns and col in real_data.columns:
            real_dist = real_data[col].value_counts(normalize=True).sort_index()
            synth_dist = synthetic_data[col].value_counts(normalize=True).sort_index()
            
            # Align distributions
            combined = pd.concat([real_dist, synth_dist], axis=1, keys=['real', 'synthetic']).fillna(0)
            
            metrics[col] = {
                'distribution_difference': np.mean(abs(combined['real'] - combined['synthetic'])),
                'category_count_real': real_data[col].nunique(),
                'category_count_synthetic': synthetic_data[col].nunique(),
            }
    
    # Overall metrics
    if len(numerical_columns) > 0:
        metrics['overall'] = {
            'avg_mean_diff_pct': np.mean([
                metrics[col]['mean_diff_pct'] 
                for col in numerical_columns 
                if col in metrics
            ]),
            'avg_ks_statistic': np.mean([
                metrics[col].get('ks_statistic', 0) 
                for col in numerical_columns 
                if col in metrics and 'ks_statistic' in metrics[col]
            ]),
        }
    
    return metrics