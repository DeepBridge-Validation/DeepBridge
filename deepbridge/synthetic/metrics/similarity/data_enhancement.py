"""
Functions for enhancing synthetic data quality through post-processing.
"""

import gc
import typing as t

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

# Import from the same package
from .similarity_core import calculate_similarity, filter_by_similarity


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


def enhance_synthetic_data_quality(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    categorical_columns: t.Optional[t.List[str]] = None,
    numerical_columns: t.Optional[t.List[str]] = None,
    target_column: t.Optional[str] = None,
    similarity_threshold: float = 0.9,
    minority_boost: bool = True,
    correlation_fix: bool = True,
    n_jobs: int = -1,
    random_state: t.Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Enhance synthetic data quality through post-processing techniques.

    This function applies several techniques to improve synthetic data quality:
    1. Remove samples that are too similar to original data
    2. Boost representation of minority classes/values
    3. Fix correlation structure to better match original data
    4. Enhance marginal distributions to match original data

    Args:
        original_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        target_column: Name of the target column
        similarity_threshold: Threshold for similarity filtering
        minority_boost: Whether to boost representation of minority classes
        correlation_fix: Whether to fix correlation structure
        n_jobs: Number of parallel jobs
        random_state: Random seed
        verbose: Whether to print progress information

    Returns:
        Enhanced synthetic data
    """
    if verbose:
        print('Enhancing synthetic data quality...')

    # Set random seed
    np.random.seed(random_state)

    # Make a copy of synthetic data to avoid modifying original
    enhanced_data = synthetic_data.copy()

    # Step 1: Remove samples that are too similar to original data
    if similarity_threshold < 1.0:
        if verbose:
            print(
                f'Step 1: Filtering by similarity threshold {similarity_threshold}'
            )

        enhanced_data = filter_by_similarity(
            original_data=original_data,
            synthetic_data=enhanced_data,
            threshold=similarity_threshold,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

    # Infer column types if not provided
    if categorical_columns is None and numerical_columns is None:
        categorical_columns = []
        numerical_columns = []

        for col in enhanced_data.columns:
            if (
                pd.api.types.is_numeric_dtype(enhanced_data[col])
                and enhanced_data[col].nunique() > 10
            ):
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)

        if verbose:
            print(
                f'Inferred {len(numerical_columns)} numerical columns and {len(categorical_columns)} categorical columns'
            )

    # Step 2: Boost representation of minority classes/values if needed
    if (
        minority_boost
        and target_column is not None
        and target_column in enhanced_data.columns
    ):
        if verbose:
            print('Step 2: Boosting minority class representation')

        # Check if target is categorical
        is_categorical_target = (
            target_column in categorical_columns
            or enhanced_data[target_column].nunique() <= 10
        )

        if is_categorical_target:
            # Calculate class distribution in original data
            orig_class_dist = original_data[target_column].value_counts(
                normalize=True
            )
            synth_class_dist = enhanced_data[target_column].value_counts(
                normalize=True
            )

            # Find underrepresented classes
            all_classes = set(orig_class_dist.index) | set(
                synth_class_dist.index
            )
            underrepresented = []

            for cls in all_classes:
                orig_freq = orig_class_dist.get(cls, 0)
                synth_freq = synth_class_dist.get(cls, 0)

                # Class is underrepresented if synthetic frequency is less than original
                if synth_freq < orig_freq * 0.9:  # 10% threshold
                    underrepresented.append(
                        (cls, orig_freq / max(synth_freq, 0.001))
                    )

            # Sort by representation ratio (most underrepresented first)
            underrepresented.sort(key=lambda x: x[1], reverse=True)

            if underrepresented and verbose:
                print(
                    f'Found {len(underrepresented)} underrepresented classes'
                )

            # Boost underrepresented classes
            for cls, ratio in underrepresented:
                # Get samples of this class
                class_samples = enhanced_data[
                    enhanced_data[target_column] == cls
                ]

                if len(class_samples) == 0:
                    if verbose:
                        print(
                            f'Warning: Class {cls} has no samples in synthetic data'
                        )
                    continue

                # Calculate how many additional samples needed
                target_count = int(
                    len(enhanced_data) * orig_class_dist.get(cls, 0)
                )
                current_count = len(class_samples)
                additional_needed = max(0, target_count - current_count)

                if additional_needed > 0 and verbose:
                    print(
                        f'Boosting class {cls}: Adding {additional_needed} samples'
                    )

                # Add samples (with small variations)
                if additional_needed > 0:
                    # Sample with replacement if we need more than available
                    samples_to_add = class_samples.sample(
                        n=additional_needed,
                        replace=additional_needed > len(class_samples),
                        random_state=random_state,
                    )

                    # Add small random variations to numerical columns
                    for col in numerical_columns:
                        if col in samples_to_add.columns:
                            # Calculate standard deviation of the column
                            col_std = (
                                original_data[col].std() * 0.05
                            )  # Small variation (5% of std)

                            # Add random noise
                            samples_to_add[col] += np.random.normal(
                                0, col_std, len(samples_to_add)
                            )

                    # Concatenate with enhanced data
                    enhanced_data = pd.concat(
                        [enhanced_data, samples_to_add], ignore_index=True
                    )

    # Step 3: Fix correlation structure
    if correlation_fix and len(numerical_columns) >= 2:
        if verbose:
            print('Step 3: Fixing correlation structure')

        # Calculate correlation matrices
        orig_corr = original_data[numerical_columns].corr().fillna(0)
        synth_corr = enhanced_data[numerical_columns].corr().fillna(0)

        # Calculate correlation difference
        corr_diff = (orig_corr - synth_corr).abs()

        # Find columns with largest correlation discrepancies
        mean_diff_by_col = corr_diff.mean()
        columns_to_fix = mean_diff_by_col[
            mean_diff_by_col > 0.1
        ].index.tolist()

        if columns_to_fix and verbose:
            print(
                f'Found {len(columns_to_fix)} columns with correlation issues'
            )

        if len(columns_to_fix) >= 2:
            # Standardize the columns
            synth_data_std = enhanced_data[numerical_columns].copy()
            for col in numerical_columns:
                if col in synth_data_std.columns:
                    synth_data_std[col] = (
                        synth_data_std[col] - synth_data_std[col].mean()
                    ) / synth_data_std[col].std()

            # Create new correlation structure

            # Extract the columns to fix
            X = synth_data_std[columns_to_fix].values

            try:
                # Apply PCA
                pca = PCA(n_components=len(columns_to_fix))
                X_pca = pca.fit_transform(X)

                # Modify the components to match target correlation
                target_corr = orig_corr.loc[
                    columns_to_fix, columns_to_fix
                ].values

                # Use Cholesky decomposition to get a matrix with target correlation
                try:
                    L = np.linalg.cholesky(target_corr)
                    # Transform data with new correlation
                    X_transformed = np.matmul(X_pca, L.T)

                    # Put back in original scale
                    for i, col in enumerate(columns_to_fix):
                        col_mean = enhanced_data[col].mean()
                        col_std = enhanced_data[col].std()
                        enhanced_data[col] = (
                            X_transformed[:, i] * col_std + col_mean
                        )
                except np.linalg.LinAlgError:
                    # If Cholesky fails, use a simpler approach
                    if verbose:
                        print(
                            'Cholesky decomposition failed, using simpler approach'
                        )

                    # Simply adjust correlations pairwise
                    for i, col1 in enumerate(columns_to_fix):
                        for j, col2 in enumerate(columns_to_fix):
                            if i < j:  # Only process each pair once
                                # Get current correlation
                                current_corr = (
                                    enhanced_data[[col1, col2]]
                                    .corr()
                                    .iloc[0, 1]
                                )
                                # Get target correlation
                                target_corr = orig_corr.loc[col1, col2]

                                # If correlation difference is significant
                                if abs(current_corr - target_corr) > 0.1:
                                    # Adjust second column to match target correlation
                                    _adjust_pairwise_correlation(
                                        enhanced_data,
                                        col1,
                                        col2,
                                        target_corr,
                                        random_state,
                                    )
            except Exception as e:
                if verbose:
                    print(f'Error in correlation fixing: {str(e)}')

    # Step 4: Enhance marginal distributions
    if verbose:
        print('Step 4: Enhancing marginal distributions')

    # Fix numerical distributions
    for col in numerical_columns:
        if col in enhanced_data.columns and col in original_data.columns:
            # Skip if column is already well-distributed (use KS test)

            # Clean values for KS test
            orig_values = original_data[col].dropna().values
            synth_values = enhanced_data[col].dropna().values

            if len(orig_values) > 0 and len(synth_values) > 0:
                ks_stat, p_value = stats.ks_2samp(orig_values, synth_values)

                # If distribution is significantly different
                if ks_stat > 0.1 and p_value < 0.05:
                    if verbose:
                        print(
                            f'Fixing distribution for column {col} (KS={ks_stat:.4f}, p={p_value:.4f})'
                        )

                    # Get quantiles from original data
                    orig_quantiles = np.percentile(
                        orig_values, np.arange(0, 101, 5)
                    )
                    synth_quantiles = np.percentile(
                        synth_values, np.arange(0, 101, 5)
                    )

                    # Create quantile mapping
                    quantile_mapping = {
                        synth_q: orig_q
                        for orig_q, synth_q in zip(
                            orig_quantiles, synth_quantiles
                        )
                    }

                    # Apply quantile mapping transformation
                    enhanced_data[col] = enhanced_data[col].apply(
                        lambda x: _map_quantile(x, quantile_mapping)
                        if pd.notna(x)
                        else x
                    )

    # Fix categorical distributions
    for col in categorical_columns:
        if col in enhanced_data.columns and col in original_data.columns:
            # Get value distributions
            orig_dist = original_data[col].value_counts(normalize=True)
            synth_dist = enhanced_data[col].value_counts(normalize=True)

            # Calculate distribution difference
            dist_diff = 0
            for cat in set(orig_dist.index) | set(synth_dist.index):
                orig_freq = orig_dist.get(cat, 0)
                synth_freq = synth_dist.get(cat, 0)
                dist_diff += abs(orig_freq - synth_freq)

            dist_diff /= 2  # Normalize

            # If distribution difference is significant
            if dist_diff > 0.1:
                if verbose:
                    print(
                        f'Fixing distribution for column {col} (diff={dist_diff:.4f})'
                    )

                # For each category, adjust frequency
                for cat, orig_freq in orig_dist.items():
                    synth_freq = synth_dist.get(cat, 0)

                    # If synthetic frequency is too low
                    if synth_freq < orig_freq * 0.9:
                        # Calculate how many samples to add
                        target_count = int(len(enhanced_data) * orig_freq)
                        current_count = len(
                            enhanced_data[enhanced_data[col] == cat]
                        )
                        to_add = max(0, target_count - current_count)

                        if to_add > 0 and verbose:
                            print(
                                f'  Adding {to_add} samples for category {cat}'
                            )

                        # Find other categories to decrease
                        excess_cats = []
                        for other_cat, other_freq in synth_dist.items():
                            orig_other_freq = orig_dist.get(other_cat, 0)
                            if (
                                other_freq > orig_other_freq * 1.1
                            ):  # 10% threshold
                                excess_cats.append(other_cat)

                        if to_add > 0 and excess_cats:
                            # Change some excess category samples to this category
                            for _ in range(min(to_add, 100)):  # Limit changes
                                # Choose a random excess category
                                excess_cat = np.random.choice(excess_cats)

                                # Find samples with excess category
                                excess_indices = enhanced_data[
                                    enhanced_data[col] == excess_cat
                                ].index

                                if len(excess_indices) > 0:
                                    # Choose a random sample to change
                                    idx = np.random.choice(excess_indices)
                                    enhanced_data.loc[idx, col] = cat

    if verbose:
        print('Synthetic data quality enhancement complete')

    return enhanced_data


def detect_duplicates(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    threshold: float = 0.95,
    categorical_columns: t.Optional[t.List[str]] = None,
    numerical_columns: t.Optional[t.List[str]] = None,
    sample_size: int = 10000,
    random_state: t.Optional[int] = None,
    verbose: bool = True,
) -> dict:
    """
    Detect potential duplicates between original and synthetic data.

    Args:
        original_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        threshold: Similarity threshold above which records are considered duplicates
        categorical_columns: List of categorical column names
        numerical_columns: List of numerical column names
        sample_size: Maximum number of samples to check
        random_state: Random seed for sampling
        verbose: Whether to print progress information

    Returns:
        Dictionary with duplicate detection results
    """
    if verbose:
        print(f'Detecting potential duplicates with threshold: {threshold}')

    # Sample data if needed
    if len(original_data) > sample_size:
        orig_sample = original_data.sample(
            sample_size, random_state=random_state
        )
    else:
        orig_sample = original_data

    if len(synthetic_data) > sample_size:
        synth_sample = synthetic_data.sample(
            sample_size, random_state=random_state
        )
    else:
        synth_sample = synthetic_data

    # Calculate similarity
    similarity_scores = calculate_similarity(
        original_data=orig_sample,
        synthetic_data=synth_sample,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        n_neighbors=1,  # Only care about closest match
        verbose=verbose,
        random_state=random_state,
    )

    # Find potential duplicates
    duplicates_mask = similarity_scores >= threshold
    num_duplicates = duplicates_mask.sum()
    duplicate_percentage = 100 * num_duplicates / len(similarity_scores)

    # Get the potential duplicate records
    potential_duplicates = synth_sample[duplicates_mask].index.tolist()

    result = {
        'threshold': threshold,
        'num_duplicates': num_duplicates,
        'duplicate_percentage': duplicate_percentage,
        'potential_duplicate_indices': potential_duplicates,
    }

    if verbose:
        print(
            f'Found {num_duplicates} potential duplicates ({duplicate_percentage:.2f}%)'
        )

    return result
