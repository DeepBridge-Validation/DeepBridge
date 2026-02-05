import typing as t

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


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
        'distances': distances,  # Store distances for potential further analysis
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


def assess_membership_disclosure(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    sensitive_columns: t.List[str],
    sample_size: int = 5000,
    random_state: t.Optional[int] = None,
    verbose: bool = False,
) -> dict:
    """
    Assess the risk of membership disclosure (whether an attacker can determine
    if a specific record was in the training data).

    Args:
        real_data: Original dataset
        synthetic_data: Synthetic dataset
        sensitive_columns: Columns to consider for privacy assessment
        sample_size: Maximum sample size to use
        random_state: Random seed for sampling
        verbose: Whether to print progress information

    Returns:
        Dictionary with membership disclosure risk metrics
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
            f'Assessing membership disclosure risk with {len(common_cols)} columns'
        )

    # Prepare data
    real_subset = real_sample[common_cols].copy()
    synth_subset = synth_sample[common_cols].copy()

    # Handle missing values
    real_subset = real_subset.fillna(real_subset.mean())
    synth_subset = synth_subset.fillna(synth_subset.mean())

    # Standardize the data
    scaler = StandardScaler()
    real_scaled = scaler.fit_transform(real_subset)

    # For each real record, find the closest synthetic record
    nn_model = NearestNeighbors(n_neighbors=1)
    nn_model.fit(synth_subset)

    # Get distances to the closest synthetic record
    distances, indices = nn_model.kneighbors(real_scaled)

    # Calculate metrics
    avg_distance = np.mean(distances)
    min_distance = np.min(distances)
    max_distance = np.max(distances)

    # Calculate quartiles
    q1 = np.percentile(distances, 25)
    q2 = np.percentile(distances, 50)  # median
    q3 = np.percentile(distances, 75)

    # Calculate risk thresholds
    # Records with very low distance might indicate direct copying (high risk)
    high_risk_threshold = q1 / 2  # Half of first quartile
    high_risk_count = np.sum(distances < high_risk_threshold)
    high_risk_percentage = 100 * high_risk_count / len(distances)

    # Records with distance below first quartile might indicate potential risk
    medium_risk_threshold = q1
    medium_risk_count = np.sum(
        (distances >= high_risk_threshold)
        & (distances < medium_risk_threshold)
    )
    medium_risk_percentage = 100 * medium_risk_count / len(distances)

    metrics = {
        'avg_distance': avg_distance.item(),  # Convert from numpy to Python type
        'min_distance': min_distance.item(),
        'max_distance': max_distance.item(),
        'q1_distance': q1.item(),
        'median_distance': q2.item(),
        'q3_distance': q3.item(),
        'high_risk_count': high_risk_count,
        'high_risk_percentage': high_risk_percentage.item(),
        'medium_risk_count': medium_risk_count,
        'medium_risk_percentage': medium_risk_percentage.item(),
        'low_risk_percentage': 100
        - high_risk_percentage.item()
        - medium_risk_percentage.item(),
    }

    if verbose:
        print(f'Membership disclosure assessment complete:')
        print(f'  - High risk records: {high_risk_percentage:.2f}%')
        print(f'  - Medium risk records: {medium_risk_percentage:.2f}%')

    return metrics
