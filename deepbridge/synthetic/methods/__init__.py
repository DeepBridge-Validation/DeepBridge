"""
Synthetic data generation methods.

This subpackage contains implementations of different synthetic data
generation techniques that can be used with DeepBridge.
"""

from deepbridge.synthetic.methods.gaussian_copula import GaussianCopulaGenerator
from deepbridge.synthetic.methods.ctgan import CTGANGenerator


__all__ = [
    'GaussianCopulaGenerator',
    'CTGANGenerator'
]