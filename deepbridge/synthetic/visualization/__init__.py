"""
Visualization tools for synthetic data analysis.

This module provides various visualization functions for comparing
real and synthetic data distributions and relationships.

Key visualization functions:
- plot_distributions: Compare univariate distributions
- plot_correlation_comparison: Compare correlation matrices
- plot_pairwise_distributions: Create pairwise plots
- plot_joint_distribution: Compare bivariate distributions

Usage:
    from deepbridge.synthetic.visualization.comparison import plot_distributions
    fig = plot_distributions(real_data, synthetic_data)
"""

# Importar e aplicar filtros de avisos
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       message="The iteration is not making good progress")
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       message="invalid value encountered in")
warnings.filterwarnings("ignore", category=RuntimeWarning, 
                       message="divide by zero encountered in")

from .comparison import plot_distributions
from .distribution import (
    plot_correlation_comparison,
    plot_pairwise_distributions,
    plot_joint_distribution
)