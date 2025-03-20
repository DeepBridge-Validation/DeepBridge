"""
CTGAN synthetic data generation.

This module implements a CTGAN-based synthetic data generator
that uses SDV's CTGANSynthesizer under the hood.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

from deepbridge.synthetic.base import BaseSyntheticGenerator

class CTGANGenerator(BaseSyntheticGenerator):
    """
    CTGAN-based synthetic data generator.
    
    This generator uses Conditional Tabular GANs (CTGAN) to generate
    synthetic data. It relies on SDV's CTGANSynthesizer for the implementation.
    
    CTGAN is particularly well-suited for capturing complex patterns in tabular data.
    """
    
    def __init__(
        self,
        random_state: Optional[int] = None,
        preserve_dtypes: bool = True,
        preserve_constraints: bool = True,
        epochs: int = 300,
        batch_size: int = 500,
        embedding_dim: int = 128,
        generator_dim: Optional[List[int]] = None,
        discriminator_dim: Optional[List[int]] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the CTGAN generator.
        
        Args:
            random_state: Random seed for reproducibility
            preserve_dtypes: Whether to preserve data types of original data
            preserve_constraints: Whether to preserve constraints (e.g., min/max values)
            epochs: Number of training epochs
            batch_size: Batch size during training
            embedding_dim: Size of the embedding layer
            generator_dim: Dimensions of the generator network
            discriminator_dim: Dimensions of the discriminator network
            verbose: Whether to show progress during training
            **kwargs: Additional parameters for SDV's CTGANSynthesizer
        """
        super().__init__(random_state, preserve_dtypes, preserve_constraints)
        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim or [256, 256]
        self.discriminator_dim = discriminator_dim or [256, 256]
        self.verbose = verbose
        self.sdv_kwargs = kwargs
        self.synthesizer = None
        
        # Ensure SDV is installed
        try:
            import sdv
        except ImportError:
            raise ImportError(
                "The 'sdv' package is required for CTGANGenerator. "
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
    ) -> 'CTGANGenerator':
        """
        Fit the CTGAN generator to the input data.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            categorical_columns: List of categorical columns
            numerical_columns: List of numerical columns
            metadata: Optional metadata dictionary with additional information
            **kwargs: Additional parameters for SDV's CTGANSynthesizer
            
        Returns:
            self: The fitted generator instance
        """
        # Import SDV components
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import CTGANSynthesizer
        
        # Validate and infer column types
        categorical_cols, numerical_cols = self._validate_columns(
            data, categorical_columns, numerical_columns
        )
        
        # Make a copy of the data to avoid modifying the original
        data_copy = data.copy()
        
        # Ensure categorical columns are properly recognized to avoid numeric operations
        for col in categorical_cols:
            if col in data_copy:
                # Convert to category type to ensure it's recognized as categorical
                data_copy[col] = data_copy[col].astype('category')
        
        # Collect metadata
        self._metadata, self._column_stats = self._collect_metadata(
            data_copy, categorical_cols, numerical_cols, target_column
        )
        
        # Create SDV metadata
        sdv_metadata = SingleTableMetadata()
        
        # Define column types for SDV
        for col in data_copy.columns:
            if col in categorical_cols:
                # Check if it's potentially a sensitive column
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
        
        # Create the synthesizer
        self.synthesizer = CTGANSynthesizer(
            metadata=sdv_metadata,
            epochs=self.epochs,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            verbose=self.verbose,
            **self.sdv_kwargs
        )
        
        # Fit the synthesizer
        self.synthesizer.fit(data_copy)
        self.fitted = True
        
        return self
    
    def generate(
        self, 
        num_samples: int, 
        conditions: Optional[Dict] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate synthetic data using the fitted CTGAN model.
        
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
        
        This method oversamples regions where the model is uncertain.
        
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
        
        # Generate a large pool of synthetic samples
        pool_size = num_samples * 10
        synthetic_pool = self.generate(num_samples=pool_size)
        
        # Remove any target column that might be present
        target_col = self._metadata.get('target_column')
        X_pool = synthetic_pool.drop(columns=[target_col]) if target_col in synthetic_pool.columns else synthetic_pool
        
        # Get predictions from the model
        try:
            if hasattr(model, 'predict_proba'):
                y_pred = model.predict_proba(X_pool)
                
                # For classification, find samples near decision boundary
                if y_pred.shape[1] == 2:  # Binary classification
                    uncertainty = 1 - 2 * np.abs(y_pred[:, 1] - 0.5)  # Higher when close to 0.5
                else:  # Multi-class
                    # Sort probabilities and compute gap between top two classes
                    sorted_probs = np.sort(y_pred, axis=1)
                    uncertainty = 1 - (sorted_probs[:, -1] - sorted_probs[:, -2])
                    
                # If target_value is provided, filter for that specific prediction
                if target_value is not None:
                    if y_pred.shape[1] == 2:  # Binary
                        # For binary, target_value should be 0 or 1
                        target_idx = int(target_value)
                        scores = y_pred[:, target_idx]  # Higher score = more likely to be predicted as target
                    else:  # Multi-class
                        scores = y_pred[:, target_value]  # Higher score = more likely to be predicted as target
                else:
                    # Without target, use uncertainty
                    scores = uncertainty
            else:
                # For models without predict_proba, use random selection
                scores = np.random.rand(len(X_pool))
        except Exception as e:
            print(f"Error getting predictions from model: {e}")
            # Fall back to random selection
            scores = np.random.rand(len(X_pool))
        
        # Select the most adversarial samples
        if target_value is not None:
            # For targeted adversarial, higher score is better
            indices = np.argsort(scores)[-num_samples:]
        else:
            # For uncertainty, higher score is better
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
        
        # Clamp novelty degree
        novelty_degree = max(0.0, min(1.0, novelty_degree))
        
        # Generate base synthetic data
        base_synthetic = self.generate(num_samples=num_samples)
        
        # For low novelty, just return normal synthetic data
        if novelty_degree < 0.1:
            return base_synthetic
        
        # For high novelty, modify the data to be more novel
        from sdv.single_table import CTGANSynthesizer
        from sdv.metadata import SingleTableMetadata
        
        # Create a new metadata object
        meta_dict = self.synthesizer.get_metadata().to_dict()
        novel_metadata = SingleTableMetadata.load_from_dict(meta_dict)
        
        # Create and train a new CTGAN with modified parameters
        # Higher noise makes the model explore more novel regions
        novel_synthesizer = CTGANSynthesizer(
            metadata=novel_metadata,
            epochs=int(self.epochs * 0.5),  # Fewer epochs to avoid overfitting
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            generator_dim=self.generator_dim,
            discriminator_dim=self.discriminator_dim,
            generator_lr=0.0002 * (1 + novelty_degree),  # Higher learning rate for novelty
            discriminator_lr=0.0002 * (1 + novelty_degree),
            discriminator_steps=1,
            verbose=self.verbose
        )
        
        # Create mixed data with some random perturbations
        mixed_data = base_synthetic.copy()
        for col in self._metadata['numerical_columns']:
            if col in mixed_data.columns:
                # Add noise to numerical columns based on novelty degree
                stats = self._column_stats[col]
                noise_scale = stats['std'] * novelty_degree * 3
                noise = np.random.normal(0, noise_scale, size=len(mixed_data))
                mixed_data[col] = mixed_data[col] + noise
        
        # Train the new synthesizer on the mixed data
        novel_synthesizer.fit(mixed_data)
        
        # Generate the novel data
        novel_data = novel_synthesizer.sample(num_rows=num_samples)
        
        # Format and return the novel data
        novel_data = self._format_output(novel_data)
        
        return novel_data