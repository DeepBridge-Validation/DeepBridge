"""
DeepBridge Synthetic Data Generation Package
===========================================

This package provides tools for generating high-quality synthetic data
based on real datasets, with a focus on preserving statistical properties
and relationships between variables. It uses Dask for distributed computing
to handle large datasets efficiently.

Main components:
- Methods: Different synthetic data generation techniques
- Metrics: Quality evaluation and privacy assessment
- Reports: Detailed quality reports generation
- Visualization: Tools for comparing real and synthetic data
"""

# Import and apply warning filters
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       message="The iteration is not making good progress")
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       message="invalid value encountered in")
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       message="divide by zero encountered in")
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       module="scipy.optimize")
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       module="scipy.stats._continuous_distns")

# Import Dask and set configuration defaults
import dask
try:
    # Set default Dask configurations for better performance with this package
    dask.config.set({
        'distributed.worker.memory.target': 0.75,  # Target 75% memory utilization
        'distributed.worker.memory.spill': 0.85,   # Spill to disk at 85% usage
        'distributed.worker.memory.pause': 0.95,   # Pause worker at 95% usage
        'array.chunk-size': '100MiB',              # Default chunk size for arrays
        'dataframe.shuffle.compression': 'snappy', # Use snappy compression for shuffling
    })
except:
    # If configuration fails, continue without custom settings
    pass

__version__ = '0.2.0'

from .synthesizer import Synthesize
from .base_generator import BaseGenerator
from .standard_generator import StandardGenerator

__all__ = [
    "Synthesize", 
    "BaseGenerator", 
    "StandardGenerator"
]