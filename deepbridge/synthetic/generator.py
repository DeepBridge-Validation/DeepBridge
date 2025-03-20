"""
Main synthetic data generator interface class.

This module provides the primary interface for generating synthetic data
in DeepBridge, integrating with DBDataset and other core components.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

from deepbridge.core.db_data import DBDataset
from deepbridge.synthetic.base import BaseSyntheticGenerator, SequentialSyntheticGenerator
from deepbridge.synthetic.synthesizer import synthesize  # Import the unified synthesizer function

class SyntheticDataGenerator:
    """
    Main interface for generating synthetic data in DeepBridge.
    
    This class provides a high-level API for generating synthetic data
    from a DBDataset using various generation methods.
    """
    
    def __init__(
        self,
        method: str = 'gaussian',
        random_state: Optional[int] = None,
        preserve_dtypes: bool = True,
        preserve_constraints: bool = True,
        **kwargs
    ):
        """
        Initialize the synthetic data generator.
        
        Args:
            method: Generation method ('gaussian' or 'ctgan')
            random_state: Random seed for reproducibility
            preserve_dtypes: Whether to preserve data types of original data
            preserve_constraints: Whether to preserve constraints (e.g., min/max values)
            **kwargs: Additional parameters for the specific generator
        """
        self.method = method
        self.random_state = random_state
        self.preserve_dtypes = preserve_dtypes
        self.preserve_constraints = preserve_constraints
        self.generator_kwargs = kwargs
        self.generator = None
        
        # Import the appropriate generator class based on method
        self._init_generator()
    
    def _init_generator(self):
        """Initialize the specific generator based on the chosen method."""
        if self.method == 'gaussian':
            from deepbridge.synthetic.methods.gaussian_copula import GaussianCopulaGenerator
            self.generator_class = GaussianCopulaGenerator
        elif self.method == 'ctgan':
            from deepbridge.synthetic.methods.ctgan import CTGANGenerator
            self.generator_class = CTGANGenerator
        else:
            valid_methods = ['gaussian', 'ctgan']
            raise ValueError(f"Unknown method '{self.method}'. Valid methods: {valid_methods}")
    
    def _create_generator(self, is_sequential: bool = False):
        """Create the generator instance."""
        # Common parameters for all generators
        common_params = {
            'random_state': self.random_state,
            'preserve_dtypes': self.preserve_dtypes,
            'preserve_constraints': self.preserve_constraints,
            **self.generator_kwargs
        }
        
        # Create the generator
        self.generator = self.generator_class(**common_params)
        
        return self.generator
    
    def generate_from_dataset(
        self, 
        dataset: DBDataset,
        num_samples: int,
        use_synthetic: bool = False,
        conditions: Optional[Dict] = None,
        similarity_threshold: float = 1.0,
        max_iterations: int = 10,
        return_quality_metrics: bool = False,
        print_metrics: bool = True,
        **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """
        Generate synthetic data from a DBDataset.
        
        Args:
            dataset: DBDataset instance
            num_samples: Number of synthetic samples to generate
            use_synthetic: Whether to use the synthetic data in dataset if available
            conditions: Optional conditions to apply during generation
            similarity_threshold: Threshold for considering samples too similar (0.0-1.0)
                - If 1.0 (default): Regular synthesis without uniqueness check
                - If < 1.0: Generate unique samples (lower values = more unique)
            max_iterations: Maximum number of iterations for unique data generation
            return_quality_metrics: Whether to return quality metrics
            print_metrics: Whether to print quality metrics
            **kwargs: Additional parameters for generation
            
        Returns:
            DataFrame of generated synthetic data or
            Tuple of (DataFrame, metrics_dict) if return_quality_metrics=True
        """
        # Handle the use_synthetic parameter
        if use_synthetic and hasattr(dataset, 'synthetic_data') and dataset.synthetic_data is not None:
            # Use existing synthetic data from the dataset
            source_dataset = dataset.synthetic_data
        else:
            # Use the original data
            source_dataset = dataset
            
        # Simply forward the call to the unified synthesize function
        return synthesize(
            dataset=source_dataset,
            method=self.method,
            num_samples=num_samples,
            random_state=self.random_state,
            preserve_dtypes=self.preserve_dtypes,
            return_quality_metrics=return_quality_metrics,
            print_metrics=print_metrics,
            suppress_warnings=True,
            similarity_threshold=similarity_threshold,
            max_iterations=max_iterations,
            **{**self.generator_kwargs, **kwargs}  # Combine generator kwargs with any additional kwargs
        )
    
    def generate_adversarial(
        self, 
        dataset: DBDataset,
        num_samples: int,
        target_value: Optional[Any] = None,
        similarity_threshold: float = 1.0,
        return_quality_metrics: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """
        Generate adversarial examples that perform poorly on the dataset's model.
        
        Args:
            dataset: DBDataset instance
            num_samples: Number of adversarial samples to generate
            target_value: Optional target value to fool the model into predicting
            similarity_threshold: Threshold for sample uniqueness
            return_quality_metrics: Whether to return quality metrics
            **kwargs: Additional parameters for generation
            
        Returns:
            DataFrame of generated adversarial examples or
            Tuple of (DataFrame, metrics_dict) if return_quality_metrics=True
        """
        if not hasattr(dataset, 'model') or dataset.model is None:
            raise ValueError("Dataset must have a model to generate adversarial examples.")
        
        # Create new generator if needed
        if self.generator is None:
            self._create_generator()
        
        # Get original data from dataset
        data = dataset._original_data.copy()
        target_column = dataset.target_name if hasattr(dataset, 'target_name') else None
        categorical_columns = dataset.categorical_features if hasattr(dataset, 'categorical_features') else None
        numerical_columns = dataset.numerical_features if hasattr(dataset, 'numerical_features') else None
        
        # Fit the generator on the dataset
        self.generator.fit(
            data=data,
            target_column=target_column,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns
        )
        
        # Generate adversarial examples
        adversarial_data = self.generator.generate_adversarial(
            model=dataset.model,
            num_samples=num_samples,
            target_value=target_value,
            **kwargs
        )
        
        # Handle similarity threshold if not 1.0
        if similarity_threshold < 1.0:
            # Remove samples that are too similar to the original data
            from deepbridge.synthetic.synthesizer import _remove_similar_samples
            adversarial_data = _remove_similar_samples(
                adversarial_data, 
                data, 
                categorical_columns or [], 
                numerical_columns or [], 
                similarity_threshold
            )
            
            # Warn if we lost too many samples
            if len(adversarial_data) < num_samples * 0.5:
                print(f"Warning: Could only generate {len(adversarial_data)} unique adversarial samples"
                      f" after applying similarity threshold {similarity_threshold}")
        
        # Calculate quality metrics if requested
        if return_quality_metrics:
            from deepbridge.synthetic.synthesizer import _evaluate_synthetic_quality
            quality_metrics = _evaluate_synthetic_quality(
                real_data=data,
                synthetic_data=adversarial_data,
                categorical_columns=categorical_columns
            )
            return adversarial_data, quality_metrics
        
        return adversarial_data
    
    def generate_novel(
        self, 
        dataset: DBDataset,
        num_samples: int,
        novelty_degree: float = 0.5,
        similarity_threshold: float = 1.0,
        return_quality_metrics: bool = False,
        **kwargs
    ) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict]]:
        """
        Generate novel examples with patterns not seen in the original data.
        
        Args:
            dataset: DBDataset instance
            num_samples: Number of novel samples to generate
            novelty_degree: Degree of novelty (0-1, higher means more novel)
            similarity_threshold: Threshold for sample uniqueness
            return_quality_metrics: Whether to return quality metrics
            **kwargs: Additional parameters for generation
            
        Returns:
            DataFrame of generated novel examples or
            Tuple of (DataFrame, metrics_dict) if return_quality_metrics=True
        """
        # Create new generator if needed
        if self.generator is None:
            self._create_generator()
        
        # Get original data from dataset
        data = dataset._original_data.copy()
        target_column = dataset.target_name if hasattr(dataset, 'target_name') else None
        categorical_columns = dataset.categorical_features if hasattr(dataset, 'categorical_features') else None
        numerical_columns = dataset.numerical_features if hasattr(dataset, 'numerical_features') else None
        
        # Fit the generator on the dataset
        self.generator.fit(
            data=data,
            target_column=target_column,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns
        )
        
        # Generate novel examples
        novel_data = self.generator.generate_novel(
            num_samples=num_samples,
            novelty_degree=novelty_degree,
            **kwargs
        )
        
        # Handle similarity threshold if not 1.0
        if similarity_threshold < 1.0:
            # Remove samples that are too similar to the original data
            from deepbridge.synthetic.synthesizer import _remove_similar_samples
            novel_data = _remove_similar_samples(
                novel_data, 
                data, 
                categorical_columns or [], 
                numerical_columns or [], 
                similarity_threshold
            )
            
            # Warn if we lost too many samples
            if len(novel_data) < num_samples * 0.5:
                print(f"Warning: Could only generate {len(novel_data)} unique novel samples"
                      f" after applying similarity threshold {similarity_threshold}")
        
        # Calculate quality metrics if requested
        if return_quality_metrics:
            from deepbridge.synthetic.synthesizer import _evaluate_synthetic_quality
            quality_metrics = _evaluate_synthetic_quality(
                real_data=data,
                synthetic_data=novel_data,
                categorical_columns=categorical_columns
            )
            return novel_data, quality_metrics
        
        return novel_data


# Define standalone functions outside the class
def generate_synthetic_data(
    dataset: DBDataset,
    num_samples: int = 100,
    method: str = 'gaussian',
    use_synthetic: bool = False,
    random_state: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generates synthetic data from a DBDataset.
    
    This is a convenience function to quickly generate synthetic data
    from a DBDataset without creating a SyntheticDataGenerator instance.
    
    Args:
        dataset: DBDataset instance
        num_samples: Number of synthetic samples to generate
        method: Generation method ('gaussian' or 'ctgan')
        use_synthetic: Whether to use synthetic data in the dataset if available
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters for the specific generator
        
    Returns:
        DataFrame of generated synthetic data
    """
    generator = SyntheticDataGenerator(
        method=method,
        random_state=random_state,
        **kwargs
    )
    
    return generator.generate_from_dataset(
        dataset=dataset,
        num_samples=num_samples,
        use_synthetic=use_synthetic,
        **kwargs
    )


def generate_unique_synthetic_data(
    dataset: DBDataset,
    num_samples: int = 100,
    method: str = 'gaussian',
    max_iterations: int = 10,
    similarity_threshold: float = 0.8,
    random_state: Optional[int] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Generate synthetic data that doesn't appear in the original dataset.
    
    This is a convenience function to quickly generate unique synthetic data
    from a DBDataset without creating a SyntheticDataGenerator instance.
    
    Args:
        dataset: DBDataset instance
        num_samples: Number of unique samples to generate
        method: Generation method ('gaussian' or 'ctgan')
        max_iterations: Maximum number of iterations to attempt generating unique samples
        similarity_threshold: Threshold for considering samples too similar (0.0-1.0)
        random_state: Random seed for reproducibility
        **kwargs: Additional parameters for the specific generator
        
    Returns:
        DataFrame of unique synthetic data
    """
    generator = SyntheticDataGenerator(
        method=method,
        random_state=random_state,
        **kwargs
    )
    
    return generator.generate_from_dataset(
        dataset=dataset,
        num_samples=num_samples,
        similarity_threshold=similarity_threshold,
        max_iterations=max_iterations,
        **kwargs
    )


# Define the add_method_to_class function outside the class
def _add_method_to_class(cls):
    """Add the generate_synthetic_data method to DBDataset."""
    def generate_synthetic_data_method(self, num_samples=100, method='gaussian', **kwargs):
        return generate_synthetic_data(self, num_samples, method, **kwargs)
    
    def generate_unique_data_method(self, num_samples=100, method='gaussian', **kwargs):
        return generate_unique_synthetic_data(self, num_samples, method, **kwargs)
    
    # Add the methods to the class
    setattr(cls, 'generate_synthetic_data', generate_synthetic_data_method)
    setattr(cls, 'generate_unique_data', generate_unique_data_method)


# Add the method to DBDataset when this module is imported
try:
    from deepbridge.core.db_data import DBDataset
    _add_method_to_class(DBDataset)
except ImportError:
    pass  # DBDataset not available, skip adding the method