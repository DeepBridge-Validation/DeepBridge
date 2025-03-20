"""
DeepBridge Synthetic Data Generation

This package provides tools for generating synthetic data using various techniques,
including methods for adversarial examples and novel pattern generation.
"""

# Import the main synthesize function which serves as the primary API
from deepbridge.synthetic.synthesizer import synthesize

# Import generator classes and specific generator implementations
from deepbridge.synthetic.generator import SyntheticDataGenerator
from deepbridge.synthetic.generator import _add_method_to_class 
from deepbridge.synthetic.base import BaseSyntheticGenerator, SequentialSyntheticGenerator
from deepbridge.synthetic.methods import (
    GaussianCopulaGenerator,
    CTGANGenerator
)

try:
    from deepbridge.core.db_data import DBDataset
    _add_method_to_class(DBDataset)  # Use the correct name with underscore
except ImportError:
    pass  # DBDataset not available, skip adding the method

__all__ = [
    # Main API
    'synthesize',
    
    # Generator classes
    'SyntheticDataGenerator',
    'BaseSyntheticGenerator',
    'SequentialSyntheticGenerator',
    
    # Generator implementations
    'GaussianCopulaGenerator',
    'CTGANGenerator'
]