"""
DeepBridge Synthetic Data Generation Package
===========================================

This package provides tools for generating high-quality synthetic data
based on real datasets, with a focus on preserving statistical properties
and relationships between variables.

Main components:
- Methods: Different synthetic data generation techniques
- Metrics: Quality evaluation and privacy assessment
- Reports: Detailed quality reports generation
- Visualization: Tools for comparing real and synthetic data
"""

# Importar e aplicar filtros de avisos
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

__version__ = '0.1.0'

from .synthesizer import Synthesize