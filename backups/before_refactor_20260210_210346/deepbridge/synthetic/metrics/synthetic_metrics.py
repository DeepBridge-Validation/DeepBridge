import typing as t
import warnings
from pathlib import Path

# Import Dask
import dask
import dask.array as da
import dask.dataframe as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dask.distributed import Client, wait

from .privacy import assess_k_anonymity, assess_l_diversity
from .similarity import calculate_similarity, filter_by_similarity

# Import metrics from other modules
from .statistical import (
    evaluate_categorical_column,
    evaluate_numerical_column,
    evaluate_synthetic_quality,
)


class SyntheticMetrics:
    """
    A comprehensive toolkit for evaluating synthetic data quality.

    This class acts as an orchestrator, bringing together metrics from different
    evaluation aspects (statistical, similarity, privacy) to provide a unified
    interface for synthetic data quality assessment.

    Example:
        from synthetic.metrics import SyntheticMetrics

        # Create metrics evaluator
        metrics = SyntheticMetrics(
            real_data=original_df,
            synthetic_data=synthetic_df
        )

        # Get overall quality score
        quality_score = metrics.overall_quality()

        # Print summary
        metrics.print_summary()
    """

    def __init__(
        self,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        categorical_columns: t.Optional[t.List[str]] = None,
        numerical_columns: t.Optional[t.List[str]] = None,
        target_column: t.Optional[str] = None,
        sensitive_columns: t.Optional[t.List[str]] = None,
        sample_size: int = 10000,
        random_state: t.Optional[int] = None,
        verbose: bool = True,
        use_dask: bool = False,
        dask_client: t.Optional[Client] = None,
        dask_partition_size: t.Optional[int] = None,
    ):
        """
        Initialize the synthetic data metrics evaluator.

        Args:
            real_data: Original real dataset
            synthetic_data: Generated synthetic dataset
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            target_column: Name of the target variable column
            sensitive_columns: List of sensitive columns for privacy assessment
            sample_size: Maximum number of samples to use for evaluation
            random_state: Random seed for reproducibility
            verbose: Whether to print progress and information
            use_dask: Whether to use Dask for distributed computing
            dask_client: Existing Dask client to use (if None, will create new one if use_dask=True)
            dask_partition_size: Size of partitions for Dask DataFrames (if None, will be calculated automatically)
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.target_column = target_column
        self.sample_size = sample_size
        self.random_state = random_state
        self.verbose = verbose

        # Dask configuration
        self.use_dask = use_dask
        self.dask_client = dask_client
        self.dask_partition_size = dask_partition_size
        self._dask_initialized = False

        # Initialize Dask if needed
        if self.use_dask:
            self._initialize_dask()

        # Sample data if it's too large
        if len(real_data) > sample_size:
            self.real_sample = real_data.sample(
                sample_size, random_state=random_state
            )
        else:
            self.real_sample = real_data

        if len(synthetic_data) > sample_size:
            self.synthetic_sample = synthetic_data.sample(
                sample_size, random_state=random_state
            )
        else:
            self.synthetic_sample = synthetic_data

        # Validate and infer column types
        (
            self.categorical_columns,
            self.numerical_columns,
        ) = self._infer_column_types(categorical_columns, numerical_columns)

        # Set sensitive columns for privacy evaluation
        if sensitive_columns is None:
            # Default to using all columns as potentially sensitive
            self.sensitive_columns = (
                self.categorical_columns + self.numerical_columns
            )
        else:
            self.sensitive_columns = sensitive_columns

        # Initialize metrics dictionary
        self.metrics = {
            'overall': {},
            'numerical': {},
            'categorical': {},
            'privacy': {},
            'utility': {},
        }

        # Store dataset properties
        self._store_dataset_properties()

        # Calculate all metrics on initialization
        self.calculate_all_metrics()

        # Close Dask client if we created it
        if self.use_dask and self._dask_initialized:
            self._close_dask()

    def _initialize_dask(self):
        """Initialize Dask for distributed computing."""
        if self.dask_client is None:
            try:
                if self.verbose:
                    print(
                        'Initializing Dask client for metrics calculation...'
                    )
                self.dask_client = Client()
                self._dask_initialized = True
                if self.verbose:
                    print(
                        f'Dask client initialized: {self.dask_client.dashboard_link}'
                    )
            except Exception as e:
                if self.verbose:
                    print(
                        f'Error initializing Dask client: {str(e)}. Falling back to non-Dask mode.'
                    )
                self.use_dask = False
                self.dask_client = None

    def _close_dask(self):
        """Close Dask client if we initialized it."""
        if self._dask_initialized and self.dask_client is not None:
            try:
                if self.verbose:
                    print('Closing Dask client...')
                self.dask_client.close()
                self.dask_client = None
            except Exception as e:
                if self.verbose:
                    print(f'Error closing Dask client: {str(e)}')

    def _infer_column_types(
        self,
        categorical_columns: t.Optional[t.List[str]],
        numerical_columns: t.Optional[t.List[str]],
    ) -> t.Tuple[t.List[str], t.List[str]]:
        """
        Infer column types if not explicitly provided.

        Args:
            categorical_columns: Explicitly provided categorical columns
            numerical_columns: Explicitly provided numerical columns

        Returns:
            Tuple of (categorical_columns, numerical_columns)
        """
        # Get common columns between datasets
        common_columns = set(self.real_data.columns) & set(
            self.synthetic_data.columns
        )

        if categorical_columns is None and numerical_columns is None:
            # Infer both categorical and numerical columns
            inferred_categorical = []
            inferred_numerical = []

            for col in common_columns:
                # Check if column is in both datasets
                if (
                    col in self.real_data.columns
                    and col in self.synthetic_data.columns
                ):
                    # Infer type based on data properties
                    if (
                        pd.api.types.is_numeric_dtype(self.real_data[col])
                        and self.real_data[col].nunique() > 10
                        and col != self.target_column
                    ):
                        inferred_numerical.append(col)
                    else:
                        inferred_categorical.append(col)

            return inferred_categorical, inferred_numerical

        elif categorical_columns is None:
            # Infer only categorical columns
            return [
                col for col in common_columns if col not in numerical_columns
            ], numerical_columns

        elif numerical_columns is None:
            # Infer only numerical columns
            return categorical_columns, [
                col for col in common_columns if col not in categorical_columns
            ]

        else:
            # Both are provided, validate they exist in the data
            cat_cols = [
                col for col in categorical_columns if col in common_columns
            ]
            num_cols = [
                col for col in numerical_columns if col in common_columns
            ]
            return cat_cols, num_cols

    def _store_dataset_properties(self):
        """Store basic properties of the datasets."""
        self.metrics['overall']['real_data_size'] = len(self.real_data)
        self.metrics['overall']['synthetic_data_size'] = len(
            self.synthetic_data
        )
        self.metrics['overall']['size_ratio'] = (
            len(self.synthetic_data) / len(self.real_data)
            if len(self.real_data) > 0
            else 0
        )
        self.metrics['overall']['num_numerical_columns'] = len(
            self.numerical_columns
        )
        self.metrics['overall']['num_categorical_columns'] = len(
            self.categorical_columns
        )

    def calculate_all_metrics(self):
        """Calculate all metrics for synthetic data evaluation."""
        if self.verbose:
            print('Calculating all synthetic data quality metrics...')

        # Statistical metrics - uses the function from statistical.py
        self._calculate_statistical_metrics()

        # Privacy metrics - uses functions from privacy.py
        self._calculate_privacy_metrics()

        # Utility metrics
        self._calculate_utility_metrics()

        # Calculate overall quality score
        self._calculate_overall_quality()

        if self.verbose:
            print('Metrics calculation completed.')

    def _calculate_statistical_metrics(self):
        """Calculate statistical metrics using functions from statistical.py."""
        if self.verbose:
            print('Evaluating statistical metrics...')

        if self.use_dask and self.dask_client is not None:
            try:
                # Convert to Dask DataFrames for distributed processing if large enough
                if (
                    len(self.real_sample) > 10000
                    or len(self.synthetic_sample) > 10000
                ):
                    if self.verbose:
                        print('Using Dask for statistical metrics calculation')

                    # Determine partition size
                    partition_size = (
                        self.dask_partition_size or 2000
                    )  # Default to 2000 rows per partition

                    # Convert to Dask DataFrames
                    real_dask = dd.from_pandas(
                        self.real_sample,
                        npartitions=max(
                            1, len(self.real_sample) // partition_size
                        ),
                    )
                    synth_dask = dd.from_pandas(
                        self.synthetic_sample,
                        npartitions=max(
                            1, len(self.synthetic_sample) // partition_size
                        ),
                    )

                    # For statistical evaluation, we can use a modified version of evaluate_synthetic_quality
                    # that works with Dask DataFrames, or we can compute necessary statistics in a distributed manner

                    # For simplicity, we'll fall back to pandas for now as the statistical metrics implementations
                    # would need significant changes to fully support Dask

                    if self.verbose:
                        print(
                            'Falling back to pandas for statistical metrics (Dask implementation pending)'
                        )
            except Exception as e:
                if self.verbose:
                    print(
                        f'Error using Dask for statistical metrics: {str(e)}. Falling back to standard method.'
                    )

        # Use the evaluate_synthetic_quality function from statistical.py
        statistical_metrics = evaluate_synthetic_quality(
            real_data=self.real_sample,
            synthetic_data=self.synthetic_sample,
            numerical_columns=self.numerical_columns,
            categorical_columns=self.categorical_columns,
            target_column=self.target_column,
            sample_size=self.sample_size,
            verbose=self.verbose,
        )

        # Store the metrics in our metrics dictionary
        self.metrics['overall'].update(statistical_metrics['overall'])
        self.metrics['numerical'].update(statistical_metrics['numerical'])
        self.metrics['categorical'].update(statistical_metrics['categorical'])

    def _calculate_privacy_metrics(self):
        """Calculate privacy risk metrics using functions from privacy.py."""
        if self.verbose:
            print('Evaluating privacy metrics...')

        try:
            # Determine which method to use (Dask or standard)
            use_dask_for_privacy = (
                self.use_dask
                and self.dask_client is not None
                and len(self.real_sample) > 5000
            )

            if use_dask_for_privacy:
                if self.verbose:
                    print('Using Dask for privacy metrics calculation')

                # K-anonymity metrics with Dask
                # For full Dask implementation, privacy assessment functions would need to be modified
                # For now, we'll use the standard functions but prepare data using Dask when possible

                # For example, preprocessing steps could be done with Dask:
                # 1. Standardize numerical data in parallel
                # 2. Calculate nearest neighbors for k-anonymity in chunks

                # Preprocess data with Dask if needed, then call standard functions
                # (implementation details omitted for brevity)

                # For now, fall back to standard implementation
                if self.verbose:
                    print(
                        'Falling back to standard privacy metrics (Dask implementation pending)'
                    )

            # Calculate k-anonymity privacy risk (standard method)
            k_metrics = assess_k_anonymity(
                real_data=self.real_sample,
                synthetic_data=self.synthetic_sample,
                sensitive_columns=self.sensitive_columns,
                k=5,
                sample_size=self.sample_size,
                random_state=self.random_state,
            )
            self.metrics['privacy']['k_anonymity'] = k_metrics

            # Extract key privacy metrics for overall evaluation
            self.metrics['overall']['privacy_risk'] = k_metrics[
                'at_risk_percentage'
            ]

            # Optional: Calculate l-diversity if we have quasi-identifiers
            if hasattr(self, 'quasi_identifiers') and self.quasi_identifiers:
                l_metrics = assess_l_diversity(
                    real_data=self.real_sample,
                    synthetic_data=self.synthetic_sample,
                    sensitive_columns=self.sensitive_columns,
                    quasi_identifiers=self.quasi_identifiers,
                    l=3,
                )
                self.metrics['privacy']['l_diversity'] = l_metrics

        except Exception as e:
            if self.verbose:
                print(f'Error evaluating privacy metrics: {str(e)}')
            self.metrics['privacy']['error'] = str(e)

    def _calculate_utility_metrics(self):
        """Calculate utility preservation metrics."""
        if self.verbose:
            print('Evaluating utility metrics...')

        # For now, we'll derive utility metrics from statistical metrics
        # In a full implementation, additional utility-specific tests could be added using Dask

        # Calculate utility score based on statistical similarity
        utility_components = []

        # Distribution similarity component
        if 'avg_ks_statistic' in self.metrics['overall']:
            ks_utility = 1 - self.metrics['overall']['avg_ks_statistic']
            utility_components.append(ks_utility)
            self.metrics['utility']['distribution_similarity'] = ks_utility

        # Correlation preservation component
        if 'correlation_mean_difference' in self.metrics['overall']:
            corr_utility = (
                1 - self.metrics['overall']['correlation_mean_difference']
            )
            utility_components.append(corr_utility)
            self.metrics['utility']['correlation_preservation'] = corr_utility

        # Categorical distribution similarity
        if 'avg_distribution_difference' in self.metrics['overall']:
            cat_utility = (
                1 - self.metrics['overall']['avg_distribution_difference']
            )
            utility_components.append(cat_utility)
            self.metrics['utility']['categorical_similarity'] = cat_utility

        # Calculate overall utility score (average of components)
        if utility_components:
            self.metrics['overall']['utility_score'] = sum(
                utility_components
            ) / len(utility_components)

    def _calculate_overall_quality(self):
        """Calculate overall quality score."""
        # Compile key metrics for overall score
        components = []

        # Statistical similarity (50% weight)
        stat_components = []

        if 'avg_ks_statistic' in self.metrics['overall']:
            stat_components.append(
                1 - self.metrics['overall']['avg_ks_statistic']
            )

        if 'correlation_mean_difference' in self.metrics['overall']:
            stat_components.append(
                1 - self.metrics['overall']['correlation_mean_difference']
            )

        if 'avg_distribution_difference' in self.metrics['overall']:
            stat_components.append(
                1 - self.metrics['overall']['avg_distribution_difference']
            )

        if stat_components:
            statistical_score = sum(stat_components) / len(stat_components)
            components.append(0.5 * statistical_score)
            self.metrics['overall'][
                'statistical_similarity'
            ] = statistical_score

        # Privacy (25% weight)
        if 'privacy_risk' in self.metrics['overall']:
            # Convert privacy risk to score (lower risk = higher score)
            privacy_score = 1 - (self.metrics['overall']['privacy_risk'] / 100)
            components.append(0.25 * privacy_score)
            self.metrics['overall']['privacy_score'] = privacy_score

        # Utility (25% weight)
        if 'utility_score' in self.metrics['overall']:
            components.append(0.25 * self.metrics['overall']['utility_score'])

        # Calculate final score
        if components:
            self.metrics['overall']['quality_score'] = sum(components)

    def overall_quality(self) -> float:
        """
        Get the overall quality score of the synthetic data.

        Returns:
            Float between 0 and 1, where higher indicates better quality
        """
        return self.metrics['overall'].get('quality_score', 0.0)

    def get_metrics(self) -> dict:
        """
        Get all calculated metrics.

        Returns:
            Dictionary with all metrics
        """
        return self.metrics

    def print_summary(self):
        """Print a summary of the synthetic data quality evaluation."""
        print('\n===== SYNTHETIC DATA QUALITY EVALUATION =====')

        # Print overall metrics
        print('\nOVERALL METRICS:')
        for key in [
            'quality_score',
            'statistical_similarity',
            'privacy_score',
            'utility_score',
        ]:
            if key in self.metrics['overall']:
                value = self.metrics['overall'][key]
                quality_level = (
                    'Excellent'
                    if value > 0.9
                    else 'Good'
                    if value > 0.8
                    else 'Fair'
                    if value > 0.7
                    else 'Poor'
                    if value > 0.6
                    else 'Very Poor'
                )
                print(f'  - {key}: {value:.4f} ({quality_level})')

        # Print statistical metrics
        print('\nSTATISTICAL METRICS:')
        for key in [
            'avg_ks_statistic',
            'avg_jensen_shannon_dist',
            'correlation_mean_difference',
            'avg_distribution_difference',
        ]:
            if key in self.metrics['overall']:
                value = self.metrics['overall'][key]
                print(f'  - {key}: {value:.4f}')

        # Print privacy metrics
        if (
            'privacy' in self.metrics
            and 'k_anonymity' in self.metrics['privacy']
        ):
            print('\nPRIVACY METRICS:')
            at_risk = self.metrics['privacy']['k_anonymity'][
                'at_risk_percentage'
            ]
            risk_level = (
                'Low'
                if at_risk < 1
                else 'Medium'
                if at_risk < 5
                else 'High'
                if at_risk < 10
                else 'Very High'
            )
            print(
                f'  - Records at privacy risk: {at_risk:.2f}% ({risk_level} risk)'
            )
            print(
                f"  - Average distance to nearest record: {self.metrics['privacy']['k_anonymity']['avg_distance']:.4f}"
            )

        # Print utility metrics
        print('\nUTILITY METRICS:')
        for key in [
            'distribution_similarity',
            'correlation_preservation',
            'categorical_similarity',
        ]:
            if key in self.metrics['utility']:
                value = self.metrics['utility'][key]
                print(f'  - {key}: {value:.4f}')

    def export_metrics_to_json(self, filepath: str):
        """
        Export metrics to a JSON file.

        Args:
            filepath: Path to save the JSON file
        """
        import json

        # Convert numpy values to Python native types
        def convert_numpy(obj):
            if isinstance(obj, (np.integer, np.floating, np.bool_)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(i) for i in obj]
            else:
                return obj

        # Convert metrics
        metrics_json = convert_numpy(self.metrics)

        # Write to file
        with open(filepath, 'w') as f:
            json.dump(metrics_json, f, indent=2)

        if self.verbose:
            print(f'Metrics exported to {filepath}')

    def __repr__(self):
        """String representation of the object."""
        score = self.metrics['overall'].get('quality_score', 0)
        return f'SyntheticMetrics(quality_score={score:.4f}, real_size={len(self.real_data)}, synthetic_size={len(self.synthetic_data)})'
