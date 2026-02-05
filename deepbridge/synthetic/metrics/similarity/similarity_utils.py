"""
Utility functions for similarity analysis and operations.
"""

import typing as t

import numpy as np
import pandas as pd
from scipy import stats


def _adjust_pairwise_correlation(
    df, col1, col2, target_corr, random_state=None
):
    """Adjust correlation between two columns to match target correlation."""
    # Get current values
    x = df[col1].values
    y = df[col2].values

    # Standardize
    x_std = (x - np.mean(x)) / np.std(x)
    y_std = (y - np.mean(y)) / np.std(y)

    # Get current correlation
    current_corr = np.corrcoef(x_std, y_std)[0, 1]

    # Calculate orthogonal component
    # y_orth is the part of y that's uncorrelated with x
    y_proj = current_corr * x_std
    y_orth = y_std - y_proj

    # Create new y with target correlation
    y_new_std = target_corr * x_std + np.sqrt(1 - target_corr**2) * y_orth

    # Convert back to original scale
    y_new = y_new_std * np.std(y) + np.mean(y)

    # Update column
    df[col2] = y_new


def _map_quantile(value, quantile_mapping):
    """Map a value according to quantile mapping."""
    # Find closest keys (synthetic quantiles) in mapping
    keys = np.array(list(quantile_mapping.keys()))

    # Handle values outside the range
    if value <= keys[0]:
        return quantile_mapping[keys[0]]
    elif value >= keys[-1]:
        return quantile_mapping[keys[-1]]

    # Find two closest keys for interpolation
    idx = np.searchsorted(keys, value)
    key_lower, key_upper = keys[idx - 1], keys[idx]

    # Interpolate between closest original quantiles
    val_lower, val_upper = (
        quantile_mapping[key_lower],
        quantile_mapping[key_upper],
    )

    # Calculate interpolation ratio
    ratio = (
        (value - key_lower) / (key_upper - key_lower)
        if key_upper > key_lower
        else 0
    )

    # Interpolate
    return val_lower + ratio * (val_upper - val_lower)


def calculate_distribution_divergence(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    column: str,
    method: str = 'js',
    bins: int = 50,
    **kwargs
) -> float:
    """
    Calculate divergence between distributions of original and synthetic data.

    Args:
        original_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        column: Column to analyze
        method: Divergence method ('js', 'kl', 'ks', 'earth_mover')
        bins: Number of bins for histogram (for continuous data)
        **kwargs: Additional parameters

    Returns:
        Divergence score (lower is better)
    """
    from scipy import stats

    # Get column data
    orig_col = original_data[column].dropna()
    synth_col = synthetic_data[column].dropna()

    # Check if column exists and has sufficient data
    if len(orig_col) < 5 or len(synth_col) < 5:
        return np.nan

    # Handle different data types
    if pd.api.types.is_numeric_dtype(orig_col) and orig_col.nunique() > 10:
        # Continuous data - use histogram
        orig_hist, bin_edges = np.histogram(orig_col, bins=bins, density=True)
        synth_hist, _ = np.histogram(synth_col, bins=bin_edges, density=True)

        # Add small constant to avoid log(0)
        orig_hist = orig_hist + 1e-10
        synth_hist = synth_hist + 1e-10

        # Normalize
        orig_hist = orig_hist / orig_hist.sum()
        synth_hist = synth_hist / synth_hist.sum()

        # Calculate divergence based on method
        if method == 'js':
            # Jensen-Shannon divergence
            m = 0.5 * (orig_hist + synth_hist)
            js_div = 0.5 * (
                stats.entropy(orig_hist, m) + stats.entropy(synth_hist, m)
            )
            return np.sqrt(js_div)  # Convert to distance

        elif method == 'kl':
            # Kullback-Leibler divergence
            return stats.entropy(orig_hist, synth_hist)

        elif method == 'ks':
            # Kolmogorov-Smirnov statistic
            ks_stat, _ = stats.ks_2samp(orig_col, synth_col)
            return ks_stat

        elif method == 'earth_mover':
            # Earth mover's distance (Wasserstein)
            return stats.wasserstein_distance(orig_col, synth_col)

        else:
            return np.nan

    else:
        # Categorical data - use value counts
        orig_counts = orig_col.value_counts(normalize=True)
        synth_counts = synth_col.value_counts(normalize=True)

        # Get all unique categories
        all_cats = sorted(set(orig_counts.index) | set(synth_counts.index))

        # Create arrays with frequencies for each category
        p = np.array([orig_counts.get(cat, 0) for cat in all_cats])
        q = np.array([synth_counts.get(cat, 0) for cat in all_cats])

        # Add small constant to avoid log(0)
        p = p + 1e-10
        q = q + 1e-10

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        # Calculate divergence based on method
        if method == 'js':
            # Jensen-Shannon divergence
            m = 0.5 * (p + q)
            js_div = 0.5 * (stats.entropy(p, m) + stats.entropy(q, m))
            return np.sqrt(js_div)  # Convert to distance

        elif method == 'kl':
            # Kullback-Leibler divergence
            return stats.entropy(p, q)

        elif method == 'total_variation':
            # Total variation distance
            return 0.5 * np.sum(np.abs(p - q))

        else:
            return np.nan


def evaluate_pairwise_correlations(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    columns: t.Optional[t.List[str]] = None,
    threshold: float = 0.1,
    **kwargs
) -> pd.DataFrame:
    """
    Evaluate how well pairwise correlations are preserved in synthetic data.

    Args:
        original_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        columns: List of columns to evaluate (if None, use all numeric)
        threshold: Threshold for considering correlation difference significant
        **kwargs: Additional parameters

    Returns:
        DataFrame with correlation differences for each pair
    """
    # Identify numeric columns if not provided
    if columns is None:
        columns = []
        for col in original_data.columns:
            if (
                pd.api.types.is_numeric_dtype(original_data[col])
                and col in synthetic_data.columns
                and pd.api.types.is_numeric_dtype(synthetic_data[col])
            ):
                columns.append(col)

    # Ensure at least 2 columns
    if len(columns) < 2:
        raise ValueError(
            'At least 2 numeric columns required for correlation analysis'
        )

    # Calculate correlation matrices
    orig_corr = original_data[columns].corr()
    synth_corr = synthetic_data[columns].corr()

    # Calculate differences
    diff_corr = orig_corr - synth_corr
    abs_diff_corr = diff_corr.abs()

    # Create result dataframe
    result = []

    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i < j:  # Only include each pair once
                orig_val = orig_corr.loc[col1, col2]
                synth_val = synth_corr.loc[col1, col2]
                diff_val = diff_corr.loc[col1, col2]
                abs_diff = abs_diff_corr.loc[col1, col2]

                result.append(
                    {
                        'column1': col1,
                        'column2': col2,
                        'original_corr': orig_val,
                        'synthetic_corr': synth_val,
                        'difference': diff_val,
                        'abs_difference': abs_diff,
                        'is_significant': abs_diff > threshold,
                    }
                )

    # Convert to dataframe
    result_df = pd.DataFrame(result)

    # Sort by absolute difference
    result_df = result_df.sort_values('abs_difference', ascending=False)

    return result_df


def calculate_diversity(
    synthetic_data: pd.DataFrame,
    categorical_columns: t.Optional[t.List[str]] = None,
    numerical_columns: t.Optional[t.List[str]] = None,
    sample_size: int = 10000,
    random_state: t.Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Calculate diversity metrics within the synthetic dataset.

    Args:
        synthetic_data: Generated synthetic dataset
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        sample_size: Maximum number of samples to use
        random_state: Random seed for sampling
        verbose: Whether to print progress information

    Returns:
        Dictionary with diversity metrics
    """
    if verbose:
        print('Calculating synthetic data diversity...')

    # Sample data if needed
    if len(synthetic_data) > sample_size:
        synth_sample = synthetic_data.sample(
            sample_size, random_state=random_state
        )
    else:
        synth_sample = synthetic_data

    # Infer column types if not provided
    if categorical_columns is None and numerical_columns is None:
        categorical_columns = []
        numerical_columns = []

        for col in synth_sample.columns:
            if (
                pd.api.types.is_numeric_dtype(synth_sample[col])
                and synth_sample[col].nunique() > 10
            ):
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)

    # Numerical diversity metrics
    numerical_metrics = {}
    if numerical_columns:
        for col in numerical_columns:
            col_metrics = {
                'min': synth_sample[col].min(),
                'max': synth_sample[col].max(),
                'range': synth_sample[col].max() - synth_sample[col].min(),
                'std': synth_sample[col].std(),
                'nunique': synth_sample[col].nunique(),
                'unique_percentage': 100
                * synth_sample[col].nunique()
                / len(synth_sample),
            }
            numerical_metrics[col] = col_metrics

    # Categorical diversity metrics
    categorical_metrics = {}
    if categorical_columns:
        for col in categorical_columns:
            value_counts = synth_sample[col].value_counts(normalize=True)
            entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))

            col_metrics = {
                'nunique': synth_sample[col].nunique(),
                'unique_percentage': 100
                * synth_sample[col].nunique()
                / len(synth_sample),
                'entropy': entropy,
                'max_category_percentage': 100 * value_counts.max(),
            }
            categorical_metrics[col] = col_metrics

    # Overall diversity metrics

    # Pairwise distance within synthetic data
    if len(numerical_columns) >= 2:
        # Standardize numerical data
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        numerical_data = synth_sample[numerical_columns].fillna(0)
        numerical_scaled = scaler.fit_transform(numerical_data)

        # Calculate pairwise distances (using a sample if dataset is large)
        max_for_pairwise = 1000
        if len(numerical_scaled) > max_for_pairwise:
            indices = np.random.choice(
                len(numerical_scaled), max_for_pairwise, replace=False
            )
            sample_scaled = numerical_scaled[indices]
        else:
            sample_scaled = numerical_scaled

        # Calculate pairwise distances
        from sklearn.metrics.pairwise import euclidean_distances

        distances = euclidean_distances(sample_scaled)

        # Set diagonal to NaN to exclude self-distances
        np.fill_diagonal(distances, np.nan)

        # Calculate diversity metrics
        avg_distance = np.nanmean(distances)
        min_distance = np.nanmin(distances)

        pairwise_metrics = {
            'average_pairwise_distance': avg_distance,
            'minimum_pairwise_distance': min_distance,
        }
    else:
        pairwise_metrics = {}

    # Combine all metrics
    diversity_metrics = {
        'numerical': numerical_metrics,
        'categorical': categorical_metrics,
        'pairwise': pairwise_metrics,
    }

    if verbose:
        print('Diversity calculation complete')

    return diversity_metrics
