"""
Standard implementation of the BaseGenerator interface.
"""

import typing as t

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import gaussian_kde
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .base_generator import BaseGenerator


class StandardGenerator(BaseGenerator):
    """
    Standard implementation of the BaseGenerator interface.
    Provides methods for generating synthetic data using various techniques.
    """

    def __init__(
        self,
        random_state: t.Optional[int] = None,
        verbose: bool = False,
        preserve_dtypes: bool = True,
        method: str = 'gaussian',
        n_components: int = 5,
        preserve_correlations: bool = True,
        outlier_rate: float = 0.0,
    ):
        """
        Initialize the standard generator.

        Args:
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
            preserve_dtypes: Whether to preserve original data types
            method: Generation method ('gaussian', 'gmm', 'kde', or 'bootstrap')
            n_components: Number of components for GMM method
            preserve_correlations: Whether to preserve correlations between features
            outlier_rate: Rate of outliers to generate (0.0 to 0.2)
        """
        super().__init__(
            random_state=random_state,
            verbose=verbose,
            preserve_dtypes=preserve_dtypes,
        )

        self.method = method
        self.n_components = n_components
        self.preserve_correlations = preserve_correlations
        self.outlier_rate = max(0.0, min(0.2, outlier_rate))  # Limit to 0-20%

        # Model components
        self.gmm = None
        self.kde_models = {}
        self.scaler = StandardScaler()
        self.correlation_matrix = None
        self.isolation_forest = None

        # Store statistics for each column
        self.column_stats = {}

    def fit(
        self,
        data: pd.DataFrame,
        target_column: t.Optional[str] = None,
        categorical_columns: t.Optional[t.List[str]] = None,
        numerical_columns: t.Optional[t.List[str]] = None,
        **kwargs,
    ) -> 'StandardGenerator':
        """
        Fit the generator to data.

        Args:
            data: Training data
            target_column: Name of the target column
            categorical_columns: List of categorical column names
            numerical_columns: List of numerical column names
            **kwargs: Additional fitting parameters
                - correlation_threshold: Minimum correlation to preserve (default: 0.1)

        Returns:
            self: Fitted generator
        """
        # Call parent fit method to handle basics
        super().fit(
            data, target_column, categorical_columns, numerical_columns
        )

        # Store data for bootstrap method if needed
        if self.method == 'bootstrap':
            self.original_data = data.copy()

        # Compute statistics for each column
        self._compute_column_stats(data)

        # Prepare numerical data for model fitting
        numerical_data = data[self.numerical_columns].copy()

        # Store correlation matrix if requested
        if self.preserve_correlations and len(self.numerical_columns) > 1:
            self.correlation_matrix = numerical_data.corr()

            # Filter correlations below threshold
            threshold = kwargs.get('correlation_threshold', 0.1)
            self.correlation_matrix = self.correlation_matrix.where(
                (self.correlation_matrix.abs() >= threshold)
                | (self.correlation_matrix.abs() == 1.0),
                0,
            )

        # Scale numerical data
        if len(self.numerical_columns) > 0:
            self.scaler.fit(numerical_data)
            scaled_data = self.scaler.transform(numerical_data)

            # Fit models based on method
            if self.method == 'gmm' and len(self.numerical_columns) > 0:
                # Adjust n_components if too many for the data
                n_components = min(self.n_components, len(data) // 5, 20)

                self.gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    random_state=self.random_state,
                )
                self.gmm.fit(scaled_data)

            elif self.method == 'kde':
                # Fit KDE for each numerical column
                for i, col in enumerate(self.numerical_columns):
                    # Get column data
                    col_data = scaled_data[:, i]
                    # Fit KDE
                    self.kde_models[col] = gaussian_kde(col_data)

            # Train isolation forest for outlier generation if requested
            if self.outlier_rate > 0:
                self.isolation_forest = IsolationForest(
                    random_state=self.random_state
                )
                self.isolation_forest.fit(scaled_data)

        # Fit categorical models
        for col in self.categorical_columns:
            # For now, just store value counts
            self.column_stats[col]['distribution'] = data[col].value_counts(
                normalize=True
            )

        self.log(f'Fitted {self.__class__.__name__} with method={self.method}')
        return self

    def generate(self, num_samples: int, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic data.

        Args:
            num_samples: Number of samples to generate
            **kwargs: Additional generation parameters
                - noise_level: Level of noise to add (0.0 to 1.0, default: 0.05)

        Returns:
            DataFrame of synthetic data
        """
        # Check if fitted
        if not self.fitted:
            raise ValueError('Generator not fitted. Call fit() first.')

        # Extract parameters
        noise_level = kwargs.get('noise_level', 0.05)

        # Initialize result DataFrame
        result = pd.DataFrame()

        # Generate data based on method
        if self.method == 'bootstrap':
            result = self._generate_bootstrap(num_samples, noise_level)
        elif self.method == 'gmm' and self.gmm is not None:
            result = self._generate_gmm(num_samples, noise_level)
        elif self.method == 'kde' and self.kde_models:
            result = self._generate_kde(num_samples, noise_level)
        else:
            # Default to Gaussian method
            result = self._generate_gaussian(num_samples, noise_level)

        # Add outliers if requested
        if self.outlier_rate > 0 and self.isolation_forest is not None:
            num_outliers = int(num_samples * self.outlier_rate)
            if num_outliers > 0:
                outliers = self._generate_outliers(num_outliers)
                # Replace some samples with outliers
                outlier_indices = self.rng.choice(
                    num_samples, num_outliers, replace=False
                )
                result.iloc[outlier_indices] = outliers.iloc[:num_outliers]

        # Restore original data types if requested
        if self.preserve_dtypes:
            result = self._restore_dtypes(result)

        self.log(
            f'Generated {num_samples} synthetic samples using {self.method} method'
        )
        return result

    def _compute_column_stats(self, data: pd.DataFrame) -> None:
        """
        Compute statistics for each column.

        Args:
            data: DataFrame to analyze
        """
        for col in data.columns:
            self.column_stats[col] = {}

            if col in self.numerical_columns:
                # Compute basic statistics
                self.column_stats[col]['min'] = data[col].min()
                self.column_stats[col]['max'] = data[col].max()
                self.column_stats[col]['mean'] = data[col].mean()
                self.column_stats[col]['std'] = data[col].std()

                # Check distribution
                try:
                    # Test for normality
                    _, pvalue = stats.normaltest(data[col].dropna())
                    self.column_stats[col]['is_normal'] = pvalue > 0.05
                except:
                    self.column_stats[col]['is_normal'] = False

            elif col in self.categorical_columns:
                # Compute value counts
                self.column_stats[col]['values'] = data[col].unique().tolist()
                self.column_stats[col]['value_counts'] = (
                    data[col].value_counts().to_dict()
                )

    def _generate_gaussian(
        self, num_samples: int, noise_level: float
    ) -> pd.DataFrame:
        """
        Generate data using a multivariate Gaussian distribution.

        Args:
            num_samples: Number of samples to generate
            noise_level: Level of noise to add

        Returns:
            Generated DataFrame
        """
        result = pd.DataFrame()

        # Generate numerical features
        if self.numerical_columns:
            # If preserving correlations, use multivariate normal
            if (
                self.preserve_correlations
                and self.correlation_matrix is not None
                and len(self.numerical_columns) > 1
            ):
                # Get means and covariance
                means = np.array(
                    [
                        self.column_stats[col]['mean']
                        for col in self.numerical_columns
                    ]
                )

                # Create covariance matrix from correlations and standard deviations
                stds = np.array(
                    [
                        self.column_stats[col]['std']
                        for col in self.numerical_columns
                    ]
                )
                cov_matrix = (
                    np.outer(stds, stds) * self.correlation_matrix.values
                )

                # Add small noise to diagonal to ensure positive definite
                cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * 1e-6

                # Generate multivariate normal data
                num_data = self.rng.multivariate_normal(
                    means, cov_matrix, num_samples
                )

                # Create DataFrame
                num_df = pd.DataFrame(num_data, columns=self.numerical_columns)

            else:
                # Generate each column independently
                num_df = pd.DataFrame()
                for col in self.numerical_columns:
                    # Generate from normal distribution
                    mean = self.column_stats[col]['mean']
                    std = self.column_stats[col]['std']
                    values = self.rng.normal(mean, std, num_samples)

                    # Add noise
                    if noise_level > 0:
                        noise = self.rng.normal(
                            0, std * noise_level, num_samples
                        )
                        values = values + noise

                    # Ensure values within bounds
                    min_val = self.column_stats[col]['min']
                    max_val = self.column_stats[col]['max']
                    values = np.clip(values, min_val, max_val)

                    num_df[col] = values

            result = pd.concat([result, num_df], axis=1)

        # Generate categorical features
        if self.categorical_columns:
            cat_df = self._generate_categorical(num_samples)
            if not cat_df.empty:
                result = pd.concat([result, cat_df], axis=1)

        return result

    def _generate_gmm(
        self, num_samples: int, noise_level: float
    ) -> pd.DataFrame:
        """
        Generate data using a Gaussian Mixture Model.

        Args:
            num_samples: Number of samples to generate
            noise_level: Level of noise to add

        Returns:
            Generated DataFrame
        """
        result = pd.DataFrame()

        # Generate numerical features using GMM
        if self.numerical_columns and self.gmm is not None:
            # Sample from GMM
            num_data, _ = self.gmm.sample(num_samples)

            # Add noise
            if noise_level > 0:
                noise = self.rng.normal(0, noise_level, num_data.shape)
                num_data = num_data + noise

            # Inverse transform to original scale
            num_data = self.scaler.inverse_transform(num_data)

            # Create DataFrame
            num_df = pd.DataFrame(num_data, columns=self.numerical_columns)

            # Ensure values within bounds
            for col in self.numerical_columns:
                min_val = self.column_stats[col]['min']
                max_val = self.column_stats[col]['max']
                num_df[col] = np.clip(num_df[col], min_val, max_val)

            result = pd.concat([result, num_df], axis=1)

        # Generate categorical features (same as gaussian method)
        cat_df = self._generate_categorical(num_samples)
        if not cat_df.empty:
            result = pd.concat([result, cat_df], axis=1)

        return result

    def _generate_kde(
        self, num_samples: int, noise_level: float
    ) -> pd.DataFrame:
        """
        Generate data using Kernel Density Estimation.

        Args:
            num_samples: Number of samples to generate
            noise_level: Level of noise to add

        Returns:
            Generated DataFrame
        """
        result = pd.DataFrame()

        # Generate numerical features using KDE
        if self.numerical_columns and self.kde_models:
            # Initialize array
            num_data = np.zeros((num_samples, len(self.numerical_columns)))

            # Generate each column
            for i, col in enumerate(self.numerical_columns):
                if col in self.kde_models:
                    # Sample from KDE
                    num_data[:, i] = self.kde_models[col].resample(
                        num_samples
                    )[0]
                else:
                    # Fallback to normal distribution
                    mean = self.column_stats[col]['mean']
                    std = self.column_stats[col]['std']
                    num_data[:, i] = self.rng.normal(mean, std, num_samples)

            # Add noise
            if noise_level > 0:
                noise = self.rng.normal(0, noise_level, num_data.shape)
                num_data = num_data + noise

            # Inverse transform to original scale
            num_data = self.scaler.inverse_transform(num_data)

            # Create DataFrame
            num_df = pd.DataFrame(num_data, columns=self.numerical_columns)

            # Ensure values within bounds
            for col in self.numerical_columns:
                min_val = self.column_stats[col]['min']
                max_val = self.column_stats[col]['max']
                num_df[col] = np.clip(num_df[col], min_val, max_val)

            result = pd.concat([result, num_df], axis=1)

        # Generate categorical features (same as gaussian method)
        cat_df = self._generate_categorical(num_samples)
        if not cat_df.empty:
            result = pd.concat([result, cat_df], axis=1)

        return result

    def _generate_bootstrap(
        self, num_samples: int, noise_level: float
    ) -> pd.DataFrame:
        """
        Generate data using bootstrap sampling with noise.

        Args:
            num_samples: Number of samples to generate
            noise_level: Level of noise to add

        Returns:
            Generated DataFrame
        """
        # Sample with replacement from original data
        indices = self.rng.choice(
            len(self.original_data), num_samples, replace=True
        )
        result = self.original_data.iloc[indices].reset_index(drop=True)

        # Add noise to numerical columns
        if noise_level > 0 and self.numerical_columns:
            for col in self.numerical_columns:
                # Get standard deviation for the column
                std = self.column_stats[col]['std']

                # Generate noise
                noise = self.rng.normal(0, std * noise_level, num_samples)

                # Add noise
                result[col] = result[col] + noise

                # Ensure values within bounds
                min_val = self.column_stats[col]['min']
                max_val = self.column_stats[col]['max']
                result[col] = np.clip(result[col], min_val, max_val)

        return result

    def _generate_categorical(self, num_samples: int) -> pd.DataFrame:
        """
        Generate categorical features.

        Args:
            num_samples: Number of samples to generate

        Returns:
            DataFrame with categorical features
        """
        if not self.categorical_columns:
            return pd.DataFrame()

        cat_df = pd.DataFrame()
        for col in self.categorical_columns:
            if col in self.column_stats:
                if 'distribution' in self.column_stats[col]:
                    # Get distribution
                    dist = self.column_stats[col]['distribution']
                    values = self.rng.choice(
                        dist.index, num_samples, p=dist.values
                    )
                    cat_df[col] = values
                elif 'values' in self.column_stats[col]:
                    # Fallback to random sampling from unique values
                    values = self.column_stats[col]['values']
                    if values:  # Check that we have values to sample from
                        cat_df[col] = self.rng.choice(values, num_samples)
                    else:
                        self.log(
                            f'Warning: No values available for categorical column {col}'
                        )
                        cat_df[col] = np.nan
                else:
                    self.log(
                        f'Warning: No distribution or values for categorical column {col}'
                    )
                    cat_df[col] = np.nan
            else:
                self.log(f'Warning: Column stats not available for {col}')

        return cat_df

    def _generate_outliers(self, num_outliers: int) -> pd.DataFrame:
        """
        Generate outlier samples.

        Args:
            num_outliers: Number of outliers to generate

        Returns:
            DataFrame with outlier samples
        """
        result = pd.DataFrame()

        # Generate extreme values for numerical columns
        if self.numerical_columns:
            num_df = pd.DataFrame()
            for col in self.numerical_columns:
                # Get statistics
                mean = self.column_stats[col]['mean']
                std = self.column_stats[col]['std']
                min_val = self.column_stats[col]['min']
                max_val = self.column_stats[col]['max']

                # Generate extreme values
                extreme_factor = (
                    4.0  # Generate values 4+ std deviations from mean
                )

                # Choose direction (high or low) for each outlier
                directions = self.rng.choice([-1, 1], num_outliers)

                # Generate outliers
                values = (
                    mean
                    + directions
                    * (extreme_factor + self.rng.random(num_outliers))
                    * std
                )

                # Ensure at least 20% beyond min/max, but not too extreme
                max_allowed = max_val * 1.5
                min_allowed = min_val * 0.5 if min_val > 0 else min_val * 1.5

                # Clip values
                values = np.clip(values, min_allowed, max_allowed)

                num_df[col] = values

            result = pd.concat([result, num_df], axis=1)

        # Use normal generation for categorical columns
        cat_df = self._generate_categorical(num_outliers)
        if not cat_df.empty:
            result = pd.concat([result, cat_df], axis=1)

        return result
