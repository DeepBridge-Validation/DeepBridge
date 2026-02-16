import typing as t

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler


class NumericalTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for numerical features that handles missing values,
    scaling, and range constraints.
    """

    def __init__(
        self,
        method: str = 'standard',
        fill_value: t.Optional[float] = None,
        preserve_range: bool = True,
    ):
        """
        Initialize numerical transformer.

        Args:
            method: Scaling method ('standard', 'minmax', or 'none')
            fill_value: Value to fill missing data (None for mean imputation)
            preserve_range: Whether to preserve the original data range
        """
        self.method = method
        self.fill_value = fill_value
        self.preserve_range = preserve_range
        self.statistics_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer on the data.

        Args:
            X: DataFrame with numerical features
            y: Target variable (not used)

        Returns:
            Self
        """
        # Store column information
        self.columns_ = X.columns.tolist()
        self.statistics_ = {}

        # Calculate statistics for each column
        for col in self.columns_:
            # Skip if all values are missing
            if X[col].isna().all():
                continue

            col_stats = {}

            # Min, max for range preservation
            col_stats['min'] = X[col].min()
            col_stats['max'] = X[col].max()

            # Mean for imputation
            col_stats['mean'] = X[col].mean()

            # Store statistics
            self.statistics_[col] = col_stats

        # Set up scaling based on method
        if self.method == 'standard':
            self.scaler_ = StandardScaler()
        elif self.method == 'minmax':
            self.scaler_ = MinMaxScaler()
        else:
            self.scaler_ = None

        # Fit scaler if needed
        if self.scaler_ is not None:
            # Temporarily impute missing values for scaling fit
            X_imputed = X.copy()

            for col in self.columns_:
                if col in self.statistics_:
                    if self.fill_value is not None:
                        X_imputed[col] = X_imputed[col].fillna(self.fill_value)
                    else:
                        X_imputed[col] = X_imputed[col].fillna(
                            self.statistics_[col]['mean']
                        )

            # Fit scaler
            self.scaler_.fit(X_imputed)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data.

        Args:
            X: DataFrame with numerical features

        Returns:
            Transformed DataFrame
        """
        # Check that all columns are present
        missing_cols = set(self.columns_) - set(X.columns)
        if missing_cols:
            raise ValueError(f'Missing columns: {missing_cols}')

        # Make a copy
        X_transformed = X[self.columns_].copy()

        # Impute missing values
        for col in self.columns_:
            if col in self.statistics_:
                if X_transformed[col].isna().any():
                    if self.fill_value is not None:
                        X_transformed[col] = X_transformed[col].fillna(
                            self.fill_value
                        )
                    else:
                        X_transformed[col] = X_transformed[col].fillna(
                            self.statistics_[col]['mean']
                        )

        # Apply scaling if needed
        if self.scaler_ is not None:
            X_scaled = pd.DataFrame(
                self.scaler_.transform(X_transformed),
                columns=self.columns_,
                index=X_transformed.index,
            )

            # Preserve original range if requested
            if self.preserve_range:
                for col in self.columns_:
                    if col in self.statistics_:
                        min_val = self.statistics_[col]['min']
                        max_val = self.statistics_[col]['max']

                        # Clip transformed values to original range
                        X_scaled[col] = X_scaled[col].clip(min_val, max_val)

            return X_scaled
        else:
            return X_transformed

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data.

        Args:
            X: Transformed DataFrame

        Returns:
            DataFrame with inverse transformed values
        """
        # Check that all columns are present
        missing_cols = set(self.columns_) - set(X.columns)
        if missing_cols:
            raise ValueError(f'Missing columns: {missing_cols}')

        # Make a copy
        X_inverse = X[self.columns_].copy()

        # Apply inverse scaling if needed
        if self.scaler_ is not None:
            X_inverse = pd.DataFrame(
                self.scaler_.inverse_transform(X_inverse),
                columns=self.columns_,
                index=X_inverse.index,
            )

        return X_inverse


class CategoricalTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for categorical features that handles missing values,
    encoding, and value constraints.
    """

    def __init__(
        self,
        method: str = 'onehot',
        unknown_value: str = 'unknown',
        handle_unknown: str = 'impute',
    ):
        """
        Initialize categorical transformer.

        Args:
            method: Encoding method ('onehot' or 'ordinal')
            unknown_value: Value to use for unknown categories
            handle_unknown: Strategy for unknown values ('impute' or 'error')
        """
        self.method = method
        self.unknown_value = unknown_value
        self.handle_unknown = handle_unknown
        self.statistics_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        """
        Fit the transformer on the data.

        Args:
            X: DataFrame with categorical features
            y: Target variable (not used)

        Returns:
            Self
        """
        # Store column information
        self.columns_ = X.columns.tolist()
        self.statistics_ = {}

        # Calculate statistics for each column
        for col in self.columns_:
            # Skip if all values are missing
            if X[col].isna().all():
                continue

            col_stats = {}

            # Value counts for imputation
            val_counts = X[col].value_counts(normalize=True)
            col_stats['categories'] = val_counts.index.tolist()
            col_stats['frequencies'] = val_counts.values.tolist()

            # Mode for imputation
            col_stats['mode'] = (
                X[col].mode()[0] if not X[col].mode().empty else None
            )

            # Store statistics
            self.statistics_[col] = col_stats

        # Set up encoding based on method
        if self.method == 'onehot':
            self.encoder_ = OneHotEncoder(
                sparse=False, handle_unknown='ignore'
            )

            # Fit encoder
            # Temporarily impute missing values for encoder fit
            X_imputed = X.copy()

            for col in self.columns_:
                if (
                    col in self.statistics_
                    and self.statistics_[col]['mode'] is not None
                ):
                    X_imputed[col] = X_imputed[col].fillna(
                        self.statistics_[col]['mode']
                    )
                else:
                    X_imputed[col] = X_imputed[col].fillna(self.unknown_value)

            self.encoder_.fit(X_imputed)

            # Store feature names
            self.feature_names_ = self.encoder_.get_feature_names_out(
                self.columns_
            )

        else:  # ordinal encoding
            self.encoder_ = None
            self.feature_names_ = self.columns_

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data.

        Args:
            X: DataFrame with categorical features

        Returns:
            Transformed DataFrame
        """
        # Check that all columns are present
        missing_cols = set(self.columns_) - set(X.columns)
        if missing_cols:
            raise ValueError(f'Missing columns: {missing_cols}')

        # Make a copy
        X_temp = X[self.columns_].copy()

        # Impute missing values
        for col in self.columns_:
            if (
                col in self.statistics_
                and self.statistics_[col]['mode'] is not None
            ):
                X_temp[col] = X_temp[col].fillna(self.statistics_[col]['mode'])
            else:
                X_temp[col] = X_temp[col].fillna(self.unknown_value)

        # Handle unknown values
        if self.handle_unknown == 'impute':
            for col in self.columns_:
                if col in self.statistics_:
                    known_cats = set(self.statistics_[col]['categories'])

                    # Replace unknown values
                    unknown_mask = (
                        ~X_temp[col].isin(known_cats) & ~X_temp[col].isna()
                    )
                    if unknown_mask.any():
                        X_temp.loc[unknown_mask, col] = self.unknown_value

        # Apply encoding if needed
        if self.encoder_ is not None:
            # OneHot encoding
            X_encoded = pd.DataFrame(
                self.encoder_.transform(X_temp),
                columns=self.feature_names_,
                index=X_temp.index,
            )

            return X_encoded
        else:
            # Ordinal encoding
            X_encoded = X_temp.copy()

            for col in self.columns_:
                if col in self.statistics_:
                    # Map categories to integers
                    categories = self.statistics_[col]['categories']

                    # Create mapping with unknown value
                    cat_mapping = {cat: i for i, cat in enumerate(categories)}
                    cat_mapping[self.unknown_value] = len(categories)

                    # Apply mapping
                    X_encoded[col] = (
                        X_encoded[col].map(cat_mapping).fillna(len(categories))
                    )

            return X_encoded

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the data.

        Args:
            X: Transformed DataFrame

        Returns:
            DataFrame with inverse transformed values
        """
        if self.encoder_ is not None:
            # OneHot encoding - need to select the encoded columns
            encoded_cols = [
                col for col in X.columns if col in self.feature_names_
            ]

            if len(encoded_cols) != len(self.feature_names_):
                raise ValueError('Some encoded columns are missing')

            # Extract the relevant columns and reorder them
            X_subset = X[encoded_cols].copy()

            # Inverse transform
            X_inv = self.encoder_.inverse_transform(X_subset)

            # Convert back to DataFrame
            X_inverse = pd.DataFrame(
                X_inv, columns=self.columns_, index=X.index
            )

            return X_inverse
        else:
            # Ordinal encoding
            X_inverse = pd.DataFrame(index=X.index, columns=self.columns_)

            for col in self.columns_:
                if col in self.statistics_:
                    # Get categories
                    categories = self.statistics_[col]['categories'] + [
                        self.unknown_value
                    ]

                    # Map integers back to categories
                    X_inverse[col] = X[col].map(
                        lambda x: categories[int(x)]
                        if pd.notna(x) and int(x) < len(categories)
                        else self.unknown_value
                    )

            return X_inverse
