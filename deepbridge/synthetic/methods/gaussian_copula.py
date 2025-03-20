"""
Gaussian Copula synthetic data generation.

This module implements a Gaussian Copula-based synthetic data generator
that uses SDV's GaussianCopulaSynthesizer under the hood.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

from deepbridge.synthetic.base import BaseSyntheticGenerator

class GaussianCopulaGenerator(BaseSyntheticGenerator):
    """
    Gaussian Copula-based synthetic data generator.
    
    This generator uses a statistical approach based on Gaussian Copulas
    to model the dependencies between variables and generate synthetic data.
    It relies on SDV's GaussianCopulaSynthesizer for the implementation.
    """
    
    def __init__(
        self,
        random_state: Optional[int] = None,
        preserve_dtypes: bool = True,
        preserve_constraints: bool = True,
        default_distribution: str = 'gaussian_kde',
        numerical_distributions: Optional[Dict[str, str]] = None,
        categorical_sdtypes: Optional[Dict[str, str]] = None,
        enforce_min_max_values: bool = True,
        **kwargs
    ):
        """
        Initialize the Gaussian Copula generator.
        
        Args:
            random_state: Random seed for reproducibility
            preserve_dtypes: Whether to preserve data types of original data
            preserve_constraints: Whether to preserve constraints (e.g., min/max values)
            default_distribution: Default distribution for numerical columns
            numerical_distributions: Dictionary mapping column names to distributions
            categorical_sdtypes: Dictionary mapping categorical columns to SDV types
            enforce_min_max_values: Whether to enforce min/max values from the original data
            **kwargs: Additional parameters for SDV's GaussianCopulaSynthesizer
        """
        super().__init__(random_state, preserve_dtypes, preserve_constraints)
        self.default_distribution = default_distribution
        self.numerical_distributions = numerical_distributions or {}
        self.categorical_sdtypes = categorical_sdtypes or {}
        self.enforce_min_max_values = enforce_min_max_values
        self.sdv_kwargs = kwargs
        self.synthesizer = None
        
        # Ensure SDV is installed
        try:
            import sdv
        except ImportError:
            raise ImportError(
                "The 'sdv' package is required for GaussianCopulaGenerator. "
                "Please install it with 'pip install sdv'."
            )
    
    def fit(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> 'GaussianCopulaGenerator':
        """
        Fit the Gaussian Copula generator to the input data.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            metadata: Optional metadata dictionary with additional information
            **kwargs: Additional parameters for SDV's GaussianCopulaSynthesizer
            
        Returns:
            self: The fitted generator instance
        """
        # Import SDV components
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer
        
        # Validate and infer column types
        categorical_cols, numerical_cols = self._validate_columns(
            data, categorical_columns, numerical_columns
        )

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

from deepbridge.synthetic.base import BaseSyntheticGenerator

class GaussianCopulaGenerator(BaseSyntheticGenerator):
    """
    Gaussian Copula-based synthetic data generator.
    
    This generator uses a statistical approach based on Gaussian Copulas
    to model the dependencies between variables and generate synthetic data.
    It relies on SDV's GaussianCopulaSynthesizer for the implementation.
    """
    
    def __init__(
        self,
        random_state: Optional[int] = None,
        preserve_dtypes: bool = True,
        preserve_constraints: bool = True,
        default_distribution: str = 'gaussian_kde',
        numerical_distributions: Optional[Dict[str, str]] = None,
        categorical_sdtypes: Optional[Dict[str, str]] = None,
        enforce_min_max_values: bool = True,
        **kwargs
    ):
        """
        Initialize the Gaussian Copula generator.
        
        Args:
            random_state: Random seed for reproducibility
            preserve_dtypes: Whether to preserve data types of original data
            preserve_constraints: Whether to preserve constraints (e.g., min/max values)
            default_distribution: Default distribution for numerical columns
            numerical_distributions: Dictionary mapping column names to distributions
            categorical_sdtypes: Dictionary mapping categorical columns to SDV types
            enforce_min_max_values: Whether to enforce min/max values from the original data
            **kwargs: Additional parameters for SDV's GaussianCopulaSynthesizer
        """
        super().__init__(random_state, preserve_dtypes, preserve_constraints)
        self.default_distribution = default_distribution
        self.numerical_distributions = numerical_distributions or {}
        self.categorical_sdtypes = categorical_sdtypes or {}
        self.enforce_min_max_values = enforce_min_max_values
        self.sdv_kwargs = kwargs
        self.synthesizer = None
        
        # Ensure SDV is installed
        try:
            import sdv
        except ImportError:
            raise ImportError(
                "The 'sdv' package is required for GaussianCopulaGenerator. "
                "Please install it with 'pip install sdv'."
            )
    
    def fit(
        self, 
        data: pd.DataFrame, 
        target_column: Optional[str] = None,
        categorical_columns: Optional[List[str]] = None,
        numerical_columns: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        **kwargs
    ) -> 'GaussianCopulaGenerator':
        """
        Fit the Gaussian Copula generator to the input data.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            metadata: Optional metadata dictionary with additional information
            **kwargs: Additional parameters for SDV's GaussianCopulaSynthesizer
            
        Returns:
            self: The fitted generator instance
        """
        # Import SDV components
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer
        
        # Validate and infer column types
        categorical_cols, numerical_cols = self._validate_columns(
            data, categorical_columns, numerical_columns
        )
        
        # Collect metadata
        self._metadata, self._column_stats = self._collect_metadata(
            data, categorical_cols, numerical_cols, target_column
        )
        
        # Create SDV metadata
        sdv_metadata = SingleTableMetadata()
        
        # Define column types for SDV
        for col in data.columns:
            if col in categorical_cols:
                # Check if we have a specific sdtype defined
                if col in self.categorical_sdtypes:
                    sdtype = self.categorical_sdtypes[col]
                else:
                    # Set a default sdtype based on column name and content
                    if 'id' in col.lower() or 'key' in col.lower():
                        sdtype = 'id'
                    elif 'email' in col.lower():
                        sdtype = 'email'
                    elif 'phone' in col.lower() or 'telephone' in col.lower():
                        sdtype = 'phone_number'
                    elif 'address' in col.lower():
                        sdtype = 'address'
                    elif 'name' in col.lower():
                        sdtype = 'name'
                    elif 'date' in col.lower() or 'time' in col.lower():
                        sdtype = 'datetime'
                    else:
                        sdtype = 'categorical'
                        
                sdv_metadata.add_column(col, sdtype=sdtype)
            else:
                # Numerical column
                sdv_metadata.add_column(col, sdtype='numerical')
        
        # Set primary key if it exists in metadata
        if metadata and 'primary_key' in metadata:
            sdv_metadata.set_primary_key(metadata['primary_key'])
        
        # Customize distributions for numerical columns
        distributions = {}
        
        # Set default distribution for all numerical columns
        for col in numerical_cols:
            distributions[col] = self.numerical_distributions.get(col, self.default_distribution)
        
        # Create the synthesizer
        self.synthesizer = GaussianCopulaSynthesizer(
            metadata=sdv_metadata,
            enforce_min_max_values=self.enforce_min_max_values,
            numerical_distributions=distributions,
            default_distribution=self.default_distribution,
            **self.sdv_kwargs
        )
        
        # Fit the synthesizer
        self.synthesizer.fit(data)
        self.fitted = True
        
        return self
    
    def generate(
        self, 
        num_samples: int, 
        conditions: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic data using the fitted Gaussian Copula model.
        
        Args:
            num_samples: Number of synthetic samples to generate
            conditions: Optional conditions to apply during generation
            **kwargs: Additional parameters for sampling
            
        Returns:
            DataFrame of generated synthetic data
        """
        if not self.fitted:
            raise ValueError("Generator must be fitted before generating data.")
        
        # Generate synthetic data
        if conditions:
            # Convert conditions to SDV format if needed
            from sdv.sampling import Condition
            
            sdv_conditions = []
            for condition_dict in conditions if isinstance(conditions, list) else [conditions]:
                column_values = condition_dict.get('column_values', {})
                num_rows = condition_dict.get('num_rows', num_samples)
                sdv_conditions.append(Condition(column_values=column_values, num_rows=num_rows))
            
            synthetic_data = self.synthesizer.sample_from_conditions(conditions=sdv_conditions)
        else:
            synthetic_data = self.synthesizer.sample(num_rows=num_samples)
        
        # Apply formatting to match original data types
        synthetic_data = self._format_output(synthetic_data)
        
        return synthetic_data
    
    def generate_adversarial(
        self, 
        model,
        num_samples: int,
        target_value: Optional[Any] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic adversarial examples that perform poorly on the given model.
        
        Args:
            model: The model to generate adversarial examples for
            num_samples: Number of adversarial samples to generate
            target_value: Optional target value to fool the model into predicting
            **kwargs: Additional parameters for generation
            
        Returns:
            DataFrame of generated adversarial examples
        """
        if not self.fitted:
            raise ValueError("Generator must be fitted before generating adversarial data.")
        
        # Start by generating a larger pool of synthetic samples
        pool_size = num_samples * 10
        synthetic_pool = self.generate(num_samples=pool_size)
        
        # Remove any target column
        target_col = self._metadata.get('target_column')
        X_pool = synthetic_pool.drop(columns=[target_col]) if target_col in synthetic_pool.columns else synthetic_pool
        
        # Get predictions from the model
        try:
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_pool)
                
                # For classification, find samples with highest uncertainty
                if y_pred.shape[1] == 2:  # Binary classification
                    uncertainty = np.abs(y_pred[:, 1] - 0.5)  # Distance from decision boundary
                else:  # Multi-class
                    uncertainty = 1.0 - np.max(y_pred, axis=1)  # 1 - max probability
                    
                # If target_value is provided, filter for that specific class
                if target_value is not None:
                    # For binary classification
                    if y_pred.shape[1] == 2:
                        if target_value == 1:
                            # Want model to predict 1, but it's predicting 0
                            scores = 1.0 - y_pred[:, 1]
                        else:
                            # Want model to predict 0, but it's predicting 1
                            scores = y_pred[:, 1]
                    else:
                        # For multi-class, want model to predict target_value
                        scores = 1.0 - y_pred[:, target_value]
                else:
                    # Without a target, just use uncertainty
                    scores = uncertainty
            else:
                # For regression or other models, use random selection
                scores = np.random.rand(len(X_pool))
        except Exception as e:
            print(f"Error getting predictions from model: {e}")
            # Fallback to random selection
            scores = np.random.rand(len(X_pool))
        
        # Select the samples with highest scores (most adversarial)
        indices = np.argsort(scores)[-num_samples:]
        adversarial_samples = synthetic_pool.iloc[indices].reset_index(drop=True)
        
        return adversarial_samples
    
    def generate_novel(
        self, 
        num_samples: int,
        novelty_degree: float = 0.5,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic data with novel patterns not present in training data.
        
        Args:
            num_samples: Number of novel samples to generate
            novelty_degree: Degree of novelty (0.0 to 1.0, higher means more novel)
            **kwargs: Additional parameters for generation
            
        Returns:
            DataFrame of generated novel examples
        """
        if not self.fitted:
            raise ValueError("Generator must be fitted before generating novel data.")
        
        # Clamp novelty degree to valid range
        novelty_degree = max(0.0, min(1.0, novelty_degree))
        
        # Start by generating standard synthetic data
        synthetic_data = self.synthesizer.sample(num_rows=num_samples * 2)
        
        # Create a modified synthesizer with more flexible constraints
        from sdv.single_table import GaussianCopulaSynthesizer
        from sdv.metadata import SingleTableMetadata
        
        # Copy the metadata
        meta_dict = self.synthesizer.get_metadata().to_dict()
        modified_metadata = SingleTableMetadata.load_from_dict(meta_dict)
        
        # Create modified synthesizer with more exploration
        novel_synthesizer = GaussianCopulaSynthesizer(
            metadata=modified_metadata,
            enforce_min_max_values=False,  # Allow values outside the original range
            default_distribution=self.default_distribution,
            numerical_distributions=self.numerical_distributions
        )
        
        # Train on original + synthetic data with randomization
        augmented_data = synthetic_data.copy()
        for col in self._metadata['numerical_columns']:
            if col in augmented_data.columns:
                # Add noise to numerical columns based on novelty degree
                stats = self._column_stats[col]
                noise_scale = stats['std'] * novelty_degree * 2
                noise = np.random.normal(0, noise_scale, size=len(augmented_data))
                augmented_data[col] = augmented_data[col] + noise
        
        # Fit modified synthesizer on augmented data
        novel_synthesizer.fit(augmented_data)
        
        # Generate novel data
        novel_data = novel_synthesizer.sample(num_rows=num_samples)
        
        # Format and return
        novel_data = self._format_output(novel_data)
        
        return novel_data
    
    def get_learned_distributions(self) -> Dict:
        """
        Get the distributions learned by the synthesizer.
        
        Returns:
            Dictionary of learned distributions for each column
        """
        if not self.fitted:
            raise ValueError("Generator must be fitted to get learned distributions.")
        
        # Get distributions from the SDV synthesizer
        return self.synthesizer.get_parameters()