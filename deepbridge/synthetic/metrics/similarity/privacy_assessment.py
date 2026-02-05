"""
Functions for assessing privacy risks in synthetic data.
"""

import typing as t

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def calculate_privacy_risk(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    sensitive_columns: t.List[str],
    n_samples: int = 1000,
    k_anonymity: int = 5,
    random_state: int = 42,
    **kwargs,
) -> dict:
    """
    Calculate privacy risk metrics for synthetic data.

    Args:
        original_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        sensitive_columns: List of sensitive column names
        n_samples: Number of samples to use
        k_anonymity: k value for k-anonymity
        random_state: Random seed
        **kwargs: Additional parameters

    Returns:
        Dictionary with privacy risk metrics
    """
    # Sample data if needed
    if len(original_data) > n_samples:
        orig_sample = original_data.sample(
            n_samples, random_state=random_state
        )
    else:
        orig_sample = original_data

    if len(synthetic_data) > n_samples:
        synth_sample = synthetic_data.sample(
            n_samples, random_state=random_state
        )
    else:
        synth_sample = synthetic_data

    # Get common sensitive columns
    common_cols = list(
        set(sensitive_columns)
        & set(orig_sample.columns)
        & set(synth_sample.columns)
    )

    if not common_cols:
        raise ValueError('No common sensitive columns found')

    # Prepare data
    X_orig = orig_sample[common_cols].fillna(0)
    X_synth = synth_sample[common_cols].fillna(0)

    # Preprocessing
    # Identify numeric and categorical columns
    numeric_cols = [
        col
        for col in common_cols
        if pd.api.types.is_numeric_dtype(X_orig[col])
    ]
    cat_cols = [col for col in common_cols if col not in numeric_cols]

    # Create preprocessor
    transformers = []

    if numeric_cols:
        transformers.append(('num', StandardScaler(), numeric_cols))

    if cat_cols:
        transformers.append(
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # Transform data
    X_orig_proc = preprocessor.fit_transform(X_orig)
    X_synth_proc = preprocessor.transform(X_synth)

    # Handle sparse matrices
    if hasattr(X_orig_proc, 'toarray'):
        X_orig_proc = X_orig_proc.toarray()
    if hasattr(X_synth_proc, 'toarray'):
        X_synth_proc = X_synth_proc.toarray()

    # Find k nearest neighbors
    nn = NearestNeighbors(n_neighbors=min(k_anonymity, len(X_orig_proc)))
    nn.fit(X_orig_proc)

    # Get distances and indices
    distances, _ = nn.kneighbors(X_synth_proc)

    # Calculate privacy metrics
    metrics = {
        'min_distance': float(np.min(distances)),
        'mean_distance': float(np.mean(distances)),
        'median_distance': float(np.median(distances)),
        'privacy_risk_pct': float(
            np.mean(distances[:, 0] < 0.1) * 100
        ),  # % of samples with nearest neighbor closer than 0.1
    }

    # Calculate histogram of minimum distances
    hist, bin_edges = np.histogram(distances[:, 0], bins=20, range=(0, 1))
    metrics['distance_histogram'] = {
        'counts': hist.tolist(),
        'bin_edges': bin_edges.tolist(),
    }

    # Calculate risk levels
    risk_levels = {
        'high_risk': float(np.mean(distances[:, 0] < 0.05) * 100),
        'medium_risk': float(
            np.mean((distances[:, 0] >= 0.05) & (distances[:, 0] < 0.2)) * 100
        ),
        'low_risk': float(np.mean(distances[:, 0] >= 0.2) * 100),
    }
    metrics['risk_levels'] = risk_levels

    return metrics


def assess_k_anonymity(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    sensitive_columns: t.List[str],
    k: int = 5,
    sample_size: int = 10000,
    random_state: t.Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """
    Assess the k-anonymity privacy risk of synthetic data.

    K-anonymity measures whether synthetic records could potentially
    re-identify individuals in the original dataset.

    Args:
        real_data: Original dataset
        synthetic_data: Synthetic dataset
        sensitive_columns: Columns to consider for privacy assessment
        k: K value for K-anonymity
        sample_size: Maximum sample size to use
        random_state: Random seed for sampling
        verbose: Whether to print progress information

    Returns:
        Dictionary with k-anonymity metrics
    """
    # Sample data if it's too large
    if len(real_data) > sample_size:
        real_sample = real_data.sample(sample_size, random_state=random_state)
    else:
        real_sample = real_data

    if len(synthetic_data) > sample_size:
        synth_sample = synthetic_data.sample(
            sample_size, random_state=random_state
        )
    else:
        synth_sample = synthetic_data

    # Select only the sensitive columns
    common_cols = list(
        set(sensitive_columns)
        & set(real_sample.columns)
        & set(synth_sample.columns)
    )

    if not common_cols:
        raise ValueError('No common sensitive columns found in both datasets')

    if verbose:
        print(
            f'Assessing k-anonymity with {len(common_cols)} sensitive columns'
        )

    real_sensitive = real_sample[common_cols].copy()
    synth_sensitive = synth_sample[common_cols].copy()

    # Handle missing values
    real_sensitive = real_sensitive.fillna(real_sensitive.mean())
    synth_sensitive = synth_sensitive.fillna(synth_sensitive.mean())

    # Standardize the data
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_sensitive)
    synth_scaled = scaler.transform(synth_sensitive)

    # For each synthetic record, find distances to k nearest real records
    nn_model = NearestNeighbors(n_neighbors=k)
    nn_model.fit(real_scaled)

    # Get distances and indices
    distances, indices = nn_model.kneighbors(synth_scaled)

    # Calculate privacy metrics
    avg_distance = np.mean(distances)
    min_distance = np.min(distances)
    percentile_distance = np.percentile(distances, 10)  # 10th percentile

    # Count records with distance below threshold (potentially identifying)
    threshold = 0.1  # This is a heuristic value
    at_risk_count = np.sum(distances[:, 0] < threshold)
    at_risk_percentage = 100 * at_risk_count / len(synth_scaled)

    # Calculate potential attribute disclosure
    attribute_disclosure_risk = {}

    for col in common_cols:
        # For each synthetic record, check if its value for this attribute
        # is close to the values in its k nearest neighbors
        real_values = real_sample[col].values
        synth_values = synth_sample[col].values

        disclosure_count = 0

        for i, (dist, idx) in enumerate(zip(distances, indices)):
            synth_val = synth_values[i]

            # Get neighboring values
            neighbor_vals = real_values[idx]

            # Check if synthetic value is close to any real value
            if pd.api.types.is_numeric_dtype(real_sample[col]):
                # For numerical columns, check for close values
                if np.any(
                    np.abs(neighbor_vals - synth_val)
                    < 0.01 * np.std(real_values)
                ):
                    disclosure_count += 1
            else:
                # For categorical columns, check for exact matches
                if synth_val in neighbor_vals:
                    disclosure_count += 1

        # Calculate disclosure risk
        disclosure_risk = 100 * disclosure_count / len(synth_scaled)
        attribute_disclosure_risk[col] = disclosure_risk

    # Create metrics dictionary
    metrics = {
        'k_anonymity': k,
        'avg_distance': avg_distance,
        'min_distance': min_distance,
        'percentile_10_distance': percentile_distance,
        'at_risk_count': at_risk_count,
        'at_risk_percentage': at_risk_percentage,
        'attribute_disclosure_risk': attribute_disclosure_risk,
        'overall_disclosure_risk': np.mean(
            list(attribute_disclosure_risk.values())
        ),
    }

    if verbose:
        print(
            f'Privacy assessment complete: {at_risk_percentage:.2f}% records at risk'
        )

    return metrics


def assess_l_diversity(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    sensitive_columns: t.List[str],
    quasi_identifiers: t.List[str],
    l: int = 3,
    verbose: bool = False,
) -> dict:
    """
    Assess the l-diversity privacy risk of synthetic data.

    L-diversity measures the diversity of sensitive attributes within
    quasi-identifier groups.

    Args:
        real_data: Original dataset
        synthetic_data: Synthetic dataset
        sensitive_columns: Columns with sensitive information
        quasi_identifiers: Columns used for identifying individuals
        l: L value for L-diversity
        verbose: Whether to print progress information

    Returns:
        Dictionary with l-diversity metrics
    """
    # Validate columns
    for col_list, name in [
        (sensitive_columns, 'sensitive_columns'),
        (quasi_identifiers, 'quasi_identifiers'),
    ]:
        missing = set(col_list) - set(real_data.columns)
        if missing:
            raise ValueError(
                f'Columns in {name} not found in real data: {missing}'
            )

        missing = set(col_list) - set(synthetic_data.columns)
        if missing:
            raise ValueError(
                f'Columns in {name} not found in synthetic data: {missing}'
            )

    if verbose:
        print(f'Assessing l-diversity with l={l}')
        print(
            f'Using {len(sensitive_columns)} sensitive columns and {len(quasi_identifiers)} quasi-identifiers'
        )

    # Group real data by quasi-identifiers
    real_groups = real_data.groupby(quasi_identifiers)

    # Initialize counts
    group_count = 0
    diverse_count = 0

    # Calculate diversity for each sensitive column
    diversity_metrics = {}

    for sensitive_col in sensitive_columns:
        group_count = 0
        diverse_count = 0

        for name, group in real_groups:
            group_count += 1

            # Count unique values in the sensitive column for this group
            unique_values = group[sensitive_col].nunique()

            # This group satisfies l-diversity if it has at least l distinct values
            if unique_values >= l:
                diverse_count += 1

        # Calculate percentage of groups that satisfy l-diversity
        l_diversity_pct = (
            100 * diverse_count / group_count if group_count > 0 else 0
        )

        diversity_metrics[sensitive_col] = {
            'l_value': l,
            'total_groups': group_count,
            'diverse_groups': diverse_count,
            'l_diversity_percentage': l_diversity_pct,
        }

    # Calculate synthetic data metrics
    synth_groups = synthetic_data.groupby(quasi_identifiers)

    for sensitive_col in sensitive_columns:
        synth_group_count = 0
        synth_diverse_count = 0

        for name, group in synth_groups:
            synth_group_count += 1

            # Count unique values in the sensitive column for this group
            unique_values = group[sensitive_col].nunique()

            # This group satisfies l-diversity if it has at least l distinct values
            if unique_values >= l:
                synth_diverse_count += 1

        # Calculate percentage of groups that satisfy l-diversity
        synth_l_diversity_pct = (
            100 * synth_diverse_count / synth_group_count
            if synth_group_count > 0
            else 0
        )

        diversity_metrics[sensitive_col].update(
            {
                'synthetic_total_groups': synth_group_count,
                'synthetic_diverse_groups': synth_diverse_count,
                'synthetic_l_diversity_percentage': synth_l_diversity_pct,
            }
        )

    # Overall metrics
    real_diversity_pct = np.mean(
        [m['l_diversity_percentage'] for m in diversity_metrics.values()]
    )
    synth_diversity_pct = np.mean(
        [
            m['synthetic_l_diversity_percentage']
            for m in diversity_metrics.values()
        ]
    )

    metrics = {
        'l_value': l,
        'real_data_l_diversity_percentage': real_diversity_pct,
        'synthetic_data_l_diversity_percentage': synth_diversity_pct,
        'column_metrics': diversity_metrics,
    }

    if verbose:
        print(f'L-diversity assessment complete:')
        print(f'  - Real data l-diversity: {real_diversity_pct:.2f}%')
        print(f'  - Synthetic data l-diversity: {synth_diversity_pct:.2f}%')

    return metrics
