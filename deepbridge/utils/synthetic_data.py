import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Union, Optional

class SyntheticDataGenerator:
    """
    Class for generating synthetic data based on real data distributions.
    Preserves statistical properties and correlations between features.
    """
    
    def __init__(self, preserve_correlations: bool = True):
        """
        Initialize the synthetic data generator.
        
        Args:
            preserve_correlations (bool): Whether to preserve correlations between columns
        """
        self.preserve_correlations = preserve_correlations
        self.columns = None
        self.dtypes = {}
        self.categorical_columns = []
        self.numerical_columns = []
        self.correlations = None
        self.category_mappings = {}
        self.inverse_category_mappings = {}
        self.stats = {}
        self.target_column = None
        self.model = None
        self._is_fitted = False
        
    def _identify_column_types(self, data: pd.DataFrame) -> None:
        """Identify column types and store metadata"""
        self.columns = data.columns.tolist()
        
        for column in self.columns:
            if data[column].dtype in ['object', 'category']:
                self.categorical_columns.append(column)
                # Create mapping for categories
                unique_values = data[column].unique()
                self.category_mappings[column] = {val: idx for idx, val in enumerate(unique_values)}
                self.inverse_category_mappings[column] = {idx: val for idx, val in enumerate(unique_values)}
            else:
                self.numerical_columns.append(column)
                
            self.dtypes[column] = data[column].dtype

    def _transform_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform categorical variables to numeric"""
        transformed = data.copy()
        for col in self.categorical_columns:
            # Handle potential new categories in test data
            transformed[col] = data[col].map(lambda x: self.category_mappings[col].get(x, -1))
            # Replace any unknown categories with most common category
            if (transformed[col] == -1).any():
                most_common_idx = 0  # Default to first category
                transformed.loc[transformed[col] == -1, col] = most_common_idx
        return transformed

    def _inverse_transform_categorical(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric values back to categories"""
        transformed = data.copy()
        for col in self.categorical_columns:
            # Round to ensure valid indices and handle potential out-of-bounds values
            indices = data[col].round().clip(0, len(self.inverse_category_mappings[col]) - 1).astype(int)
            transformed[col] = indices.map(self.inverse_category_mappings[col])
        return transformed

    def _calculate_statistics(self, data: pd.DataFrame) -> None:
        """Calculate statistics needed for synthesis"""
        for col in self.numerical_columns:
            self.stats[col] = {
                'mean': data[col].mean(),
                'std': max(data[col].std(), 1e-5),  # Avoid zero std
                'min': data[col].min(),
                'max': data[col].max()
            }

    def fit(self, data: pd.DataFrame, target_column: Optional[str] = None, model=None) -> 'SyntheticDataGenerator':
        """
        Train the generator with original data.
        
        Args:
            data (pd.DataFrame): DataFrame with original data
            target_column (str, optional): Name of the target column
            model: Optional model to use for predicting target values
            
        Returns:
            self: Fitted generator
        """
        self.target_column = target_column
        self.model = model
        
        # Make a copy to avoid modifying original data
        fit_data = data.copy()
        
        self._identify_column_types(fit_data)
        transformed_data = self._transform_categorical(fit_data)
        self._calculate_statistics(transformed_data)
        
        if self.preserve_correlations:
            self.correlations = transformed_data.corr()
            
            # Check for NaN values in correlation matrix and replace with zeros
            if np.isnan(self.correlations.values).any():
                self.correlations = self.correlations.fillna(0)
        
        self._is_fitted = True
        return self

    def generate(self, num_samples: int, random_state: Optional[int] = None) -> pd.DataFrame:
        """
        Generate synthetic data.
        
        Args:
            num_samples (int): Number of samples to generate
            random_state (int, optional): Random seed for reproducibility
            
        Returns:
            pd.DataFrame: DataFrame with synthetic data
        """
        if not self._is_fitted:
            raise ValueError("The generator must be fitted before generating data")
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
        
        # Remove target column temporarily from columns list if it exists
        columns_for_generation = self.columns.copy()
        if self.target_column and self.target_column in columns_for_generation:
            columns_for_generation.remove(self.target_column)
        
        if self.preserve_correlations and len(columns_for_generation) > 1:
            # Get correlation matrix for features only (excluding target)
            feature_corr = self.correlations.loc[columns_for_generation, columns_for_generation]
            
            # Ensure correlation matrix is positive semidefinite
            try:
                # Generate data using multivariate normal distribution
                synthetic_data = pd.DataFrame(
                    np.random.multivariate_normal(
                        mean=np.zeros(len(columns_for_generation)),
                        cov=feature_corr,
                        size=num_samples
                    ),
                    columns=columns_for_generation
                )
            except np.linalg.LinAlgError:
                # Fall back to Cholesky decomposition for numerical stability
                # Add small value to diagonal
                feature_corr_adjusted = feature_corr.copy()
                np.fill_diagonal(feature_corr_adjusted.values, np.diag(feature_corr) + 1e-5)
                
                synthetic_data = pd.DataFrame(
                    np.random.multivariate_normal(
                        mean=np.zeros(len(columns_for_generation)),
                        cov=feature_corr_adjusted,
                        size=num_samples
                    ),
                    columns=columns_for_generation
                )
        else:
            # Generate independent data if no correlations to preserve or only one column
            synthetic_data = pd.DataFrame(
                np.random.normal(size=(num_samples, len(columns_for_generation))),
                columns=columns_for_generation
            )

        # Adjust statistics for numerical columns
        for col in self.numerical_columns:
            if col in columns_for_generation:  # Skip target if it's numerical
                stats = self.stats[col]
                synthetic_data[col] = (synthetic_data[col] * stats['std']) + stats['mean']
                # Apply min/max limits
                synthetic_data[col] = synthetic_data[col].clip(stats['min'], stats['max'])

        # Adjust categorical columns
        for col in self.categorical_columns:
            if col in columns_for_generation:  # Skip target if it's categorical
                n_categories = len(self.category_mappings[col])
                synthetic_data[col] = np.random.randint(0, n_categories, size=num_samples)

        # Add target column if needed
        if self.target_column:
            if self.model is not None:
                # Predict target using the model
                try:
                    if hasattr(self.model, 'predict_proba'):
                        # For classification, get class probabilities
                        probs = self.model.predict_proba(synthetic_data)
                        # Sample class based on probabilities
                        if probs.shape[1] == 2:  # Binary classification
                            synthetic_data[self.target_column] = np.random.binomial(1, probs[:, 1])
                        else:  # Multi-class
                            synthetic_data[self.target_column] = np.array([
                                np.random.choice(len(p), p=p) for p in probs
                            ])
                    else:
                        # For regression or models without predict_proba
                        preds = self.model.predict(synthetic_data)
                        synthetic_data[self.target_column] = preds
                        
                        # Add some noise to predictions to avoid deterministic values
                        if self.target_column in self.numerical_columns:
                            noise_std = self.stats[self.target_column]['std'] * 0.05  # 5% noise
                            synthetic_data[self.target_column] += np.random.normal(0, noise_std, size=num_samples)
                            synthetic_data[self.target_column] = synthetic_data[self.target_column].clip(
                                self.stats[self.target_column]['min'], 
                                self.stats[self.target_column]['max']
                            )
                        
                except Exception as e:
                    print(f"Error predicting target column: {str(e)}")
                    # Fall back to statistical generation
                    if self.target_column in self.numerical_columns:
                        stats = self.stats[self.target_column]
                        synthetic_data[self.target_column] = np.random.normal(
                            stats['mean'], stats['std'], size=num_samples
                        ).clip(stats['min'], stats['max'])
                    elif self.target_column in self.categorical_columns:
                        n_categories = len(self.category_mappings[self.target_column])
                        synthetic_data[self.target_column] = np.random.randint(0, n_categories, size=num_samples)
            else:
                # Generate target based on statistics if no model provided
                if self.target_column in self.numerical_columns:
                    stats = self.stats[self.target_column]
                    synthetic_data[self.target_column] = np.random.normal(
                        stats['mean'], stats['std'], size=num_samples
                    ).clip(stats['min'], stats['max'])
                elif self.target_column in self.categorical_columns:
                    n_categories = len(self.category_mappings[self.target_column])
                    synthetic_data[self.target_column] = np.random.randint(0, n_categories, size=num_samples)

        # Convert data types and categories back
        result = self._inverse_transform_categorical(synthetic_data)
        for col, dtype in self.dtypes.items():
            if col in result.columns:
                try:
                    result[col] = result[col].astype(dtype)
                except (ValueError, TypeError):
                    # If conversion fails, keep as is
                    pass

        return result.reset_index(drop=True)
    
    def evaluate_quality(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict:
        """
        Evaluate the quality of synthetic data.
        
        Args:
            real_data (pd.DataFrame): Original data
            synthetic_data (pd.DataFrame): Synthetic data
            
        Returns:
            Dict: Quality metrics
        """
        metrics = {}
        
        # Compare basic statistics for numerical columns
        for col in self.numerical_columns:
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
                        ks_stat, ks_pval = stats.ks_2samp(real_data[col], synthetic_data[col])
                        metrics[col]['ks_statistic'] = ks_stat
                        metrics[col]['ks_pvalue'] = ks_pval
                    except Exception:
                        pass
            
        # Compare distributions for categorical columns
        for col in self.categorical_columns:
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
        if len(self.numerical_columns) > 0:
            metrics['overall'] = {
                'avg_mean_diff_pct': np.mean([
                    metrics[col]['mean_diff_pct'] 
                    for col in self.numerical_columns 
                    if col in metrics
                ]),
                'avg_ks_statistic': np.mean([
                    metrics[col].get('ks_statistic', 0) 
                    for col in self.numerical_columns 
                    if col in metrics and 'ks_statistic' in metrics[col]
                ]),
            }
        
        return metrics