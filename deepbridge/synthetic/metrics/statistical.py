import gc
import typing as t
import warnings

import numpy as np
import pandas as pd
from scipy import stats

# Filtrar avisos do scipy.stats
warnings.filterwarnings(
    'ignore',
    category=RuntimeWarning,
    message='The iteration is not making good progress',
)
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, message='invalid value encountered in'
)
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, message='divide by zero encountered in'
)
warnings.filterwarnings(
    'ignore', category=RuntimeWarning, module='scipy.stats._continuous_distns'
)


def evaluate_synthetic_quality(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    numerical_columns: t.Optional[t.List[str]] = None,
    categorical_columns: t.Optional[t.List[str]] = None,
    target_column: t.Optional[str] = None,
    sample_size: int = 10000,
    verbose: bool = False,
    **kwargs,
) -> dict:
    """
    Evaluate the quality of synthetic data by comparing it to real data.

    This function provides a comprehensive statistical evaluation, including
    column-level comparisons and correlation analysis.

    Args:
        real_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        target_column: Name of the target column
        sample_size: Maximum number of samples to use for evaluation
        verbose: Whether to print progress information
        **kwargs: Additional parameters

    Returns:
        Dictionary with quality metrics
    """
    # Log function for verbose mode
    def log(message):
        if verbose:
            print(message)

    log('Evaluating synthetic data quality...')

    # Sample data if it's too large for efficient evaluation
    if len(real_data) > sample_size:
        log(f'Sampling {sample_size} rows from real data for evaluation')
        real_sample = real_data.sample(sample_size, random_state=42)
    else:
        real_sample = real_data

    if len(synthetic_data) > sample_size:
        log(f'Sampling {sample_size} rows from synthetic data for evaluation')
        synthetic_sample = synthetic_data.sample(sample_size, random_state=42)
    else:
        synthetic_sample = synthetic_data

    # Infer column types if not provided
    all_columns = set(real_sample.columns) & set(synthetic_sample.columns)

    if numerical_columns is None:
        numerical_columns = []
        for col in all_columns:
            if pd.api.types.is_numeric_dtype(real_sample[col]):
                if real_sample[col].nunique() > 10 or (
                    col == target_column
                    and pd.api.types.is_float_dtype(real_sample[col])
                ):
                    numerical_columns.append(col)

    if categorical_columns is None:
        categorical_columns = []
        for col in all_columns:
            if col not in numerical_columns:
                categorical_columns.append(col)

    log(
        f'Evaluating {len(numerical_columns)} numerical columns and {len(categorical_columns)} categorical columns'
    )

    # Initialize metrics dictionary
    metrics = {
        'overall': {},
        'numerical': {},
        'categorical': {},
    }

    # Add dataset size metrics
    metrics['overall']['real_data_size'] = len(real_data)
    metrics['overall']['synthetic_data_size'] = len(synthetic_data)
    metrics['overall']['size_ratio'] = (
        len(synthetic_data) / len(real_data) if len(real_data) > 0 else 0
    )

    # Evaluate numerical columns
    if numerical_columns:
        for col in numerical_columns:
            if col in real_sample.columns and col in synthetic_sample.columns:
                try:
                    metrics['numerical'][col] = evaluate_numerical_column(
                        real_sample[col], synthetic_sample[col], **kwargs
                    )
                except Exception as e:
                    log(f'Error evaluating numerical column {col}: {str(e)}')
                    metrics['numerical'][col] = {'error': str(e)}

        # Calculate overall metrics for numerical columns
        if len(metrics['numerical']) > 0:
            # Calculate average KS statistic
            ks_values = [
                v.get('ks_statistic', 0)
                for v in metrics['numerical'].values()
                if isinstance(v, dict) and 'ks_statistic' in v
            ]

            if ks_values:
                metrics['overall']['avg_ks_statistic'] = sum(ks_values) / len(
                    ks_values
                )

            # Calculate average JSD
            jsd_values = [
                v.get('jensen_shannon_dist', 0)
                for v in metrics['numerical'].values()
                if isinstance(v, dict) and 'jensen_shannon_dist' in v
            ]

            if jsd_values:
                metrics['overall']['avg_jensen_shannon_dist'] = sum(
                    jsd_values
                ) / len(jsd_values)

            # Calculate average relative error in mean and std
            mean_errors = [
                v.get('mean_relative_error', 0)
                for v in metrics['numerical'].values()
                if isinstance(v, dict) and 'mean_relative_error' in v
            ]

            std_errors = [
                v.get('std_relative_error', 0)
                for v in metrics['numerical'].values()
                if isinstance(v, dict) and 'std_relative_error' in v
            ]

            if mean_errors:
                metrics['overall']['avg_mean_relative_error'] = sum(
                    mean_errors
                ) / len(mean_errors)

            if std_errors:
                metrics['overall']['avg_std_relative_error'] = sum(
                    std_errors
                ) / len(std_errors)

    # Evaluate categorical columns
    if categorical_columns:
        for col in categorical_columns:
            if col in real_sample.columns and col in synthetic_sample.columns:
                try:
                    metrics['categorical'][col] = evaluate_categorical_column(
                        real_sample[col], synthetic_sample[col], **kwargs
                    )
                except Exception as e:
                    log(f'Error evaluating categorical column {col}: {str(e)}')
                    metrics['categorical'][col] = {'error': str(e)}

        # Calculate overall metrics for categorical columns
        if len(metrics['categorical']) > 0:
            # Calculate average chi-square p-value
            chi2_values = [
                v.get('chi2_pvalue', 0)
                for v in metrics['categorical'].values()
                if isinstance(v, dict) and 'chi2_pvalue' in v
            ]

            if chi2_values:
                metrics['overall']['avg_chi2_pvalue'] = sum(chi2_values) / len(
                    chi2_values
                )

            # Calculate average distribution difference
            dist_diff_values = [
                v.get('distribution_difference', 0)
                for v in metrics['categorical'].values()
                if isinstance(v, dict) and 'distribution_difference' in v
            ]

            if dist_diff_values:
                metrics['overall']['avg_distribution_difference'] = sum(
                    dist_diff_values
                ) / len(dist_diff_values)

    # Evaluate pairwise correlations if there are multiple numerical columns
    if len(numerical_columns) >= 2:
        log('Evaluating pairwise correlations...')

        try:
            # Calculate correlation matrices
            real_corr = real_sample[numerical_columns].corr().fillna(0)
            synth_corr = synthetic_sample[numerical_columns].corr().fillna(0)

            # Calculate correlation difference
            corr_diff = (real_corr - synth_corr).abs().values
            avg_corr_diff = np.mean(corr_diff)
            max_corr_diff = np.max(corr_diff)

            metrics['overall']['correlation_mean_difference'] = avg_corr_diff
            metrics['overall']['correlation_max_difference'] = max_corr_diff
        except Exception as e:
            log(f'Error evaluating correlations: {str(e)}')

    # Free memory
    gc.collect()

    log('Quality evaluation completed')
    return metrics


def evaluate_numerical_column(
    real_col: pd.Series, synth_col: pd.Series, bins: int = 30, **kwargs
) -> dict:
    """
    Evaluate the quality of a synthetic numerical column.

    Args:
        real_col: Real data column
        synth_col: Synthetic data column
        bins: Number of bins for histogram comparison
        **kwargs: Additional parameters

    Returns:
        Dictionary with quality metrics for the column
    """
    metrics = {}

    # Clean the data - remove NaNs and infinities
    real_clean = real_col.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    synth_clean = (
        synth_col.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    )

    # Skip if insufficient data
    if len(real_clean) < 5 or len(synth_clean) < 5:
        return {'error': 'Insufficient data for evaluation'}

    # Basic statistics
    real_mean = real_clean.mean()
    synth_mean = synth_clean.mean()
    real_std = real_clean.std()
    synth_std = synth_clean.std()

    metrics['mean_real'] = real_mean
    metrics['mean_synthetic'] = synth_mean
    metrics['mean_absolute_error'] = abs(real_mean - synth_mean)
    metrics['mean_relative_error'] = (
        abs(real_mean - synth_mean) / abs(real_mean)
        if abs(real_mean) > 0
        else 0
    )

    metrics['std_real'] = real_std
    metrics['std_synthetic'] = synth_std
    metrics['std_absolute_error'] = abs(real_std - synth_std)
    metrics['std_relative_error'] = (
        abs(real_std - synth_std) / abs(real_std) if abs(real_std) > 0 else 0
    )

    # Min, max, range
    real_min = real_clean.min()
    synth_min = synth_clean.min()
    real_max = real_clean.max()
    synth_max = synth_clean.max()

    metrics['min_real'] = real_min
    metrics['min_synthetic'] = synth_min
    metrics['max_real'] = real_max
    metrics['max_synthetic'] = synth_max
    metrics['range_real'] = real_max - real_min
    metrics['range_synthetic'] = synth_max - synth_min

    # Percentiles
    for p in [1, 5, 25, 50, 75, 95, 99]:
        real_p = np.percentile(real_clean, p)
        synth_p = np.percentile(synth_clean, p)

        metrics[f'p{p}_real'] = real_p
        metrics[f'p{p}_synthetic'] = synth_p
        metrics[f'p{p}_absolute_error'] = abs(real_p - synth_p)

    # Kolmogorov-Smirnov test
    try:
        ks_stat, ks_pval = stats.ks_2samp(real_clean, synth_clean)
        metrics['ks_statistic'] = ks_stat
        metrics['ks_pvalue'] = ks_pval
    except Exception:
        pass

    # Jensen-Shannon Distance (based on histogram distributions)
    try:
        real_hist, bin_edges = np.histogram(
            real_clean, bins=bins, density=True
        )
        synth_hist, _ = np.histogram(synth_clean, bins=bin_edges, density=True)

        # Add small constant to avoid log(0)
        real_hist = real_hist + 1e-10
        synth_hist = synth_hist + 1e-10

        # Normalize
        real_hist = real_hist / real_hist.sum()
        synth_hist = synth_hist / synth_hist.sum()

        # Calculate JS divergence
        m = 0.5 * (real_hist + synth_hist)
        js_div = 0.5 * (
            stats.entropy(real_hist, m) + stats.entropy(synth_hist, m)
        )

        metrics['jensen_shannon_div'] = js_div
        metrics['jensen_shannon_dist'] = np.sqrt(js_div)
    except Exception:
        pass

    return metrics


def evaluate_categorical_column(
    real_col: pd.Series, synth_col: pd.Series, **kwargs
) -> dict:
    """
    Evaluate the quality of a synthetic categorical column.

    Args:
        real_col: Real data column
        synth_col: Synthetic data column
        **kwargs: Additional parameters

    Returns:
        Dictionary with quality metrics for the column
    """
    metrics = {}

    # Clean the data - handle NaNs
    real_clean = real_col.fillna('__NA__')
    synth_clean = synth_col.fillna('__NA__')

    # Category counts
    real_nunique = real_clean.nunique()
    synth_nunique = synth_clean.nunique()

    metrics['category_count_real'] = real_nunique
    metrics['category_count_synthetic'] = synth_nunique
    metrics['category_count_difference'] = abs(real_nunique - synth_nunique)
    metrics['category_count_ratio'] = (
        synth_nunique / real_nunique if real_nunique > 0 else 0
    )

    # Distribution comparison
    real_dist = real_clean.value_counts(normalize=True)
    synth_dist = synth_clean.value_counts(normalize=True)

    # Align the distributions
    combined = pd.concat(
        [real_dist, synth_dist], axis=1, keys=['real', 'synthetic']
    ).fillna(0)

    # Calculate the mean absolute difference in distributions
    metrics['distribution_difference'] = (
        combined['real'].sub(combined['synthetic']).abs().mean()
    )

    # Category coverage - measure if synthetic data has the same categories
    real_categories = set(real_clean.unique())
    synth_categories = set(synth_clean.unique())

    metrics['category_coverage'] = (
        len(real_categories & synth_categories) / len(real_categories)
        if len(real_categories) > 0
        else 0
    )
    metrics['missing_categories'] = len(real_categories - synth_categories)
    metrics['extra_categories'] = len(synth_categories - real_categories)

    # Jensen-Shannon Divergence for categorical distributions
    try:
        # Get probabilities for all categories
        all_categories = list(real_categories | synth_categories)
        real_probs = np.array(
            [real_dist.get(cat, 0) for cat in all_categories]
        )
        synth_probs = np.array(
            [synth_dist.get(cat, 0) for cat in all_categories]
        )

        # Add small constant to avoid log(0)
        real_probs = real_probs + 1e-10
        synth_probs = synth_probs + 1e-10

        # Normalize
        real_probs = real_probs / real_probs.sum()
        synth_probs = synth_probs / synth_probs.sum()

        # Calculate JS divergence
        m = 0.5 * (real_probs + synth_probs)
        js_div = 0.5 * (
            stats.entropy(real_probs, m) + stats.entropy(synth_probs, m)
        )

        metrics['jensen_shannon_div'] = js_div
        metrics['jensen_shannon_dist'] = np.sqrt(js_div)
    except Exception:
        pass

    return metrics


def print_quality_metrics(metrics: dict, detailed: bool = False) -> None:
    """
    Print synthetic data quality metrics in a human-readable format.

    Args:
        metrics: Dictionary with quality metrics from evaluate_synthetic_quality
        detailed: Whether to print detailed metrics for each column
    """
    print('\n===== SYNTHETIC DATA QUALITY EVALUATION =====')

    # Print overall metrics
    print('\nOVERALL METRICS:')
    for key, value in metrics['overall'].items():
        if isinstance(value, (int, float)):
            print(f'  - {key}: {value:.4f}')
        else:
            print(f'  - {key}: {value}')

    if detailed:
        # Print numerical column metrics
        if 'numerical' in metrics and metrics['numerical']:
            print('\nNUMERICAL COLUMNS:')
            for col, col_metrics in metrics['numerical'].items():
                print(f'\n  Column: {col}')
                if 'error' in col_metrics:
                    print(f"    - Error: {col_metrics['error']}")
                    continue

                # Print most important metrics
                if 'ks_statistic' in col_metrics:
                    print(
                        f"    - KS statistic: {col_metrics['ks_statistic']:.4f}"
                    )
                    print(f"    - KS p-value: {col_metrics['ks_pvalue']:.4f}")

                if 'mean_relative_error' in col_metrics:
                    print(
                        f"    - Mean relative error: {col_metrics['mean_relative_error']:.4f}"
                    )

                if 'std_relative_error' in col_metrics:
                    print(
                        f"    - Std relative error: {col_metrics['std_relative_error']:.4f}"
                    )

                if 'jensen_shannon_dist' in col_metrics:
                    print(
                        f"    - Jensen-Shannon distance: {col_metrics['jensen_shannon_dist']:.4f}"
                    )

        # Print categorical column metrics
        if 'categorical' in metrics and metrics['categorical']:
            print('\nCATEGORICAL COLUMNS:')
            for col, col_metrics in metrics['categorical'].items():
                print(f'\n  Column: {col}')
                if 'error' in col_metrics:
                    print(f"    - Error: {col_metrics['error']}")
                    continue

                # Print most important metrics
                if 'distribution_difference' in col_metrics:
                    print(
                        f"    - Distribution difference: {col_metrics['distribution_difference']:.4f}"
                    )

                if 'category_coverage' in col_metrics:
                    print(
                        f"    - Category coverage: {col_metrics['category_coverage']:.4f}"
                    )

                if 'chi2_pvalue' in col_metrics:
                    print(
                        f"    - Chi-square p-value: {col_metrics['chi2_pvalue']:.4f}"
                    )

    print('\nQuality evaluation summary:')

    # Overall quality assessment
    avg_metrics = []
    if 'avg_ks_statistic' in metrics['overall']:
        ks_quality = (
            'Excellent'
            if metrics['overall']['avg_ks_statistic'] < 0.1
            else 'Good'
            if metrics['overall']['avg_ks_statistic'] < 0.2
            else 'Fair'
            if metrics['overall']['avg_ks_statistic'] < 0.3
            else 'Poor'
        )
        avg_metrics.append(
            f"Distribution similarity: {ks_quality} (KS: {metrics['overall']['avg_ks_statistic']:.4f})"
        )

    if 'correlation_mean_difference' in metrics['overall']:
        corr_quality = (
            'Excellent'
            if metrics['overall']['correlation_mean_difference'] < 0.1
            else 'Good'
            if metrics['overall']['correlation_mean_difference'] < 0.2
            else 'Fair'
            if metrics['overall']['correlation_mean_difference'] < 0.3
            else 'Poor'
        )
        avg_metrics.append(
            f"Correlation preservation: {corr_quality} (Diff: {metrics['overall']['correlation_mean_difference']:.4f})"
        )

    if 'avg_distribution_difference' in metrics['overall']:
        cat_quality = (
            'Excellent'
            if metrics['overall']['avg_distribution_difference'] < 0.1
            else 'Good'
            if metrics['overall']['avg_distribution_difference'] < 0.2
            else 'Fair'
            if metrics['overall']['avg_distribution_difference'] < 0.3
            else 'Poor'
        )
        avg_metrics.append(
            f"Categorical distribution: {cat_quality} (Diff: {metrics['overall']['avg_distribution_difference']:.4f})"
        )

    for metric in avg_metrics:
        print(f'  - {metric}')


def calculate_statistical_distance(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    method: str = 'ks',
    numerical_columns: t.Optional[t.List[str]] = None,
    categorical_columns: t.Optional[t.List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Calculate statistical distance between real and synthetic data for each column.

    Args:
        real_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        method: Distance method ('ks', 'js', or 'chi2')
        numerical_columns: List of numerical column names
        categorical_columns: List of categorical column names
        **kwargs: Additional parameters

    Returns:
        DataFrame with distance metrics for each column
    """
    # Infer column types if not provided
    all_columns = set(real_data.columns) & set(synthetic_data.columns)

    if numerical_columns is None:
        numerical_columns = []
        for col in all_columns:
            if (
                pd.api.types.is_numeric_dtype(real_data[col])
                and real_data[col].nunique() > 10
            ):
                numerical_columns.append(col)

    if categorical_columns is None:
        categorical_columns = []
        for col in all_columns:
            if col not in numerical_columns:
                categorical_columns.append(col)

    # Initialize results
    distances = []

    # Calculate distances for numerical columns
    for col in numerical_columns:
        if col in real_data.columns and col in synthetic_data.columns:
            try:
                # Clean data
                real_clean = (
                    real_data[col]
                    .dropna()
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                synth_clean = (
                    synthetic_data[col]
                    .dropna()
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )

                # Skip if insufficient data
                if len(real_clean) < 5 or len(synth_clean) < 5:
                    continue

                if method == 'ks':
                    # Kolmogorov-Smirnov test
                    ks_stat, ks_pval = stats.ks_2samp(real_clean, synth_clean)
                    distances.append(
                        {
                            'column': col,
                            'type': 'numerical',
                            'distance': ks_stat,
                            'p_value': ks_pval,
                            'method': 'KS',
                        }
                    )

                elif method == 'js':
                    # Jensen-Shannon Distance
                    bins = kwargs.get('bins', 30)
                    real_hist, bin_edges = np.histogram(
                        real_clean, bins=bins, density=True
                    )
                    synth_hist, _ = np.histogram(
                        synth_clean, bins=bin_edges, density=True
                    )

                    # Add small constant to avoid log(0)
                    real_hist = real_hist + 1e-10
                    synth_hist = synth_hist + 1e-10

                    # Normalize
                    real_hist = real_hist / real_hist.sum()
                    synth_hist = synth_hist / synth_hist.sum()

                    # Calculate JS divergence
                    m = 0.5 * (real_hist + synth_hist)
                    js_div = 0.5 * (
                        stats.entropy(real_hist, m)
                        + stats.entropy(synth_hist, m)
                    )
                    js_dist = np.sqrt(js_div)

                    distances.append(
                        {
                            'column': col,
                            'type': 'numerical',
                            'distance': js_dist,
                            'p_value': None,
                            'method': 'JS',
                        }
                    )

                else:
                    # Default to KS
                    ks_stat, ks_pval = stats.ks_2samp(real_clean, synth_clean)
                    distances.append(
                        {
                            'column': col,
                            'type': 'numerical',
                            'distance': ks_stat,
                            'p_value': ks_pval,
                            'method': 'KS',
                        }
                    )

            except Exception as e:
                # Skip columns with errors
                continue

    # Calculate distances for categorical columns
    for col in categorical_columns:
        if col in real_data.columns and col in synthetic_data.columns:
            try:
                # Clean data
                real_clean = real_data[col].fillna('__NA__')
                synth_clean = synthetic_data[col].fillna('__NA__')

                # Get value distributions
                real_dist = real_clean.value_counts(normalize=True)
                synth_dist = synth_clean.value_counts(normalize=True)

                if method == 'chi2':
                    # Chi-square test
                    real_categories = set(real_clean.unique())
                    synth_categories = set(synth_clean.unique())
                    all_categories = list(real_categories | synth_categories)

                    # Create observed and expected frequencies
                    observed = [
                        synth_clean.value_counts().get(cat, 0)
                        for cat in all_categories
                    ]
                    expected = [
                        len(synth_clean) * real_dist.get(cat, 0)
                        for cat in all_categories
                    ]

                    # Filter out categories with expected frequency < 5
                    valid_indices = [
                        i for i, exp in enumerate(expected) if exp >= 5
                    ]
                    if len(valid_indices) >= 2:
                        chi2, pval = stats.chisquare(
                            [observed[i] for i in valid_indices],
                            [expected[i] for i in valid_indices],
                        )
                        distances.append(
                            {
                                'column': col,
                                'type': 'categorical',
                                'distance': chi2,
                                'p_value': pval,
                                'method': 'Chi2',
                            }
                        )

                elif method == 'js':
                    # Jensen-Shannon Distance for categorical
                    all_categories = list(
                        set(real_dist.index) | set(synth_dist.index)
                    )
                    real_probs = np.array(
                        [real_dist.get(cat, 0) for cat in all_categories]
                    )
                    synth_probs = np.array(
                        [synth_dist.get(cat, 0) for cat in all_categories]
                    )

                    # Add small constant to avoid log(0)
                    real_probs = real_probs + 1e-10
                    synth_probs = synth_probs + 1e-10

                    # Normalize
                    real_probs = real_probs / real_probs.sum()
                    synth_probs = synth_probs / synth_probs.sum()

                    # Calculate JS divergence
                    m = 0.5 * (real_probs + synth_probs)
                    js_div = 0.5 * (
                        stats.entropy(real_probs, m)
                        + stats.entropy(synth_probs, m)
                    )
                    js_dist = np.sqrt(js_div)

                    distances.append(
                        {
                            'column': col,
                            'type': 'categorical',
                            'distance': js_dist,
                            'p_value': None,
                            'method': 'JS',
                        }
                    )

                else:
                    # Mean absolute difference for categorical (default)
                    # Align distributions
                    combined = pd.concat(
                        [real_dist, synth_dist],
                        axis=1,
                        keys=['real', 'synthetic'],
                    ).fillna(0)
                    dist_diff = (
                        combined['real']
                        .sub(combined['synthetic'])
                        .abs()
                        .mean()
                    )

                    distances.append(
                        {
                            'column': col,
                            'type': 'categorical',
                            'distance': dist_diff,
                            'p_value': None,
                            'method': 'MAD',
                        }
                    )

            except Exception as e:
                # Skip columns with errors
                continue

    # Convert to DataFrame
    return pd.DataFrame(distances)
