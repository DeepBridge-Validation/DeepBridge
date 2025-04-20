"""
Visualization functions for feature perturbation results.

This module provides specialized visualization functions for
understanding the impact of feature perturbations on model performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_feature_perturbation_impact(
    perturbation_results: Dict[str, Any],
    feature_name: str,
    perturbation_type: str = 'noise',
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot impact of perturbing a specific feature on model performance.
    
    Parameters:
    -----------
    perturbation_results : dict
        Results from perturbation tests
    feature_name : str
        Name of the feature to plot
    perturbation_type : str
        Type of perturbation to plot
    metric : str
        Metric to plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.Figure : Plot figure
    """
    # Extract data based on structure
    feature_data = None
    levels = []
    rel_changes = []
    
    # Try different result structures
    if 'per_feature' in perturbation_results:
        # First structure type
        if feature_name in perturbation_results['per_feature']:
            feature_results = perturbation_results['per_feature'][feature_name]
            
            if perturbation_type in feature_results:
                feature_data = feature_results[perturbation_type]
                
    elif 'perturbations' in perturbation_results:
        # Second structure type
        if perturbation_type in perturbation_results['perturbations']:
            type_results = perturbation_results['perturbations'][perturbation_type]
            
            if 'feature_results' in type_results and feature_name in type_results['feature_results']:
                feature_data = type_results['feature_results'][feature_name]
                
    # Extract levels and relative changes
    if feature_data:
        for result in feature_data:
            if 'level' in result and 'relative_change' in result and metric in result['relative_change']:
                levels.append(result['level'])
                rel_changes.append(result['relative_change'][metric])
    
    if not levels or not rel_changes:
        raise ValueError(f"Could not find perturbation data for feature '{feature_name}', type '{perturbation_type}', metric '{metric}'")
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot relative change
    ax.plot(levels, rel_changes, 'o-', linewidth=2, markersize=8)
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel(f'{perturbation_type.title()} Perturbation Level')
    ax.set_ylabel(f'Relative Change in {metric.upper()}')
    ax.set_title(f'Impact of {perturbation_type.title()} Perturbation on {feature_name}')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_perturbation_heatmap(
    perturbation_results: Dict[str, Any],
    perturbation_type: str = 'noise',
    metric: str = 'accuracy',
    level_idx: int = 0,
    top_n: int = 10,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot heatmap of perturbation impact across features.
    
    Parameters:
    -----------
    perturbation_results : dict
        Results from perturbation tests
    perturbation_type : str
        Type of perturbation to plot
    metric : str
        Metric to plot
    level_idx : int
        Index of perturbation level to use
    top_n : int
        Number of top features to include
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.Figure : Plot figure
    """
    # Extract data
    features = []
    rel_changes = []
    
    # Try different result structures
    if 'per_feature' in perturbation_results:
        # First structure type
        for feature, feature_results in perturbation_results['per_feature'].items():
            if perturbation_type in feature_results:
                feature_data = feature_results[perturbation_type]
                
                if len(feature_data) > level_idx:
                    result = feature_data[level_idx]
                    
                    if 'relative_change' in result and metric in result['relative_change']:
                        features.append(feature)
                        rel_changes.append(result['relative_change'][metric])
                        
    elif 'perturbations' in perturbation_results:
        # Second structure type
        if perturbation_type in perturbation_results['perturbations']:
            type_results = perturbation_results['perturbations'][perturbation_type]
            
            if 'feature_results' in type_results:
                for feature, feature_data in type_results['feature_results'].items():
                    if len(feature_data) > level_idx:
                        result = feature_data[level_idx]
                        
                        if 'relative_change' in result and metric in result['relative_change']:
                            features.append(feature)
                            rel_changes.append(result['relative_change'][metric])
    
    if not features or not rel_changes:
        raise ValueError(f"Could not find perturbation data for type '{perturbation_type}', metric '{metric}'")
    
    # Sort features by impact
    if top_n and len(features) > top_n:
        # Sort by absolute impact and take top_n
        sorted_idx = np.argsort(np.abs(rel_changes))[::-1][:top_n]
        features = [features[i] for i in sorted_idx]
        rel_changes = [rel_changes[i] for i in sorted_idx]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create data frame for seaborn
    df = pd.DataFrame({
        'Feature': features,
        'Relative Change': rel_changes
    })
    
    # Create colormap centered at 0
    max_abs = max(abs(min(rel_changes)), abs(max(rel_changes)))
    if max_abs == 0:
        max_abs = 1.0  # Avoid division by zero
        
    # Create heatmap
    sns.barplot(
        data=df,
        y='Feature',
        x='Relative Change',
        ax=ax,
        palette='RdBu_r',
        orient='h'
    )
    
    # Add vertical line at 0
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.7)
    
    # Set labels and title
    level_value = perturbation_results.get('perturbation_levels', [0.1, 0.2, 0.5])[level_idx] \
                 if 'perturbation_levels' in perturbation_results else "?"
                 
    ax.set_title(f'Impact of {perturbation_type.title()} Perturbation (Level {level_value}) on {metric.upper()}')
    ax.set_xlabel(f'Relative Change in {metric.upper()}')
    
    plt.tight_layout()
    return fig


def plot_aggregate_perturbation_impact(
    perturbation_results: Dict[str, Any],
    perturbation_types: Optional[List[str]] = None,
    metric: str = 'accuracy',
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot aggregate impact of perturbations across all features.
    
    Parameters:
    -----------
    perturbation_results : dict
        Results from perturbation tests
    perturbation_types : list of str or None
        Types of perturbation to plot (None for all)
    metric : str
        Metric to plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.Figure : Plot figure
    """
    # Extract available perturbation types
    available_types = []
    
    if 'perturbations' in perturbation_results:
        available_types = list(perturbation_results['perturbations'].keys())
    elif 'perturbation_types' in perturbation_results:
        available_types = perturbation_results['perturbation_types']
    
    if not available_types:
        raise ValueError("Could not find perturbation types in results")
    
    # Filter perturbation types if specified
    if perturbation_types:
        types_to_plot = [t for t in perturbation_types if t in available_types]
        if not types_to_plot:
            raise ValueError(f"None of the specified perturbation types {perturbation_types} found in results")
    else:
        types_to_plot = available_types
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each perturbation type
    for p_type in types_to_plot:
        levels = []
        mean_changes = []
        
        # Extract aggregate results
        if 'perturbations' in perturbation_results and p_type in perturbation_results['perturbations']:
            p_results = perturbation_results['perturbations'][p_type]
            
            if 'aggregate_results' in p_results:
                agg_results = p_results['aggregate_results']
                
                for level, level_results in agg_results.items():
                    if metric in level_results:
                        levels.append(float(level))
                        mean_changes.append(level_results[metric]['mean'])
        
        if levels and mean_changes:
            # Sort by level
            level_indices = np.argsort(levels)
            levels = [levels[i] for i in level_indices]
            mean_changes = [mean_changes[i] for i in level_indices]
            
            # Plot
            ax.plot(levels, mean_changes, 'o-', label=p_type.title(), linewidth=2, markersize=8)
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Set labels and title
    ax.set_xlabel('Perturbation Level')
    ax.set_ylabel(f'Mean Relative Change in {metric.upper()}')
    ax.set_title(f'Aggregate Impact of Perturbations on {metric.upper()}')
    
    # Add legend and grid
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_relative_performance_change(
    perturbation_results: Dict[str, Any],
    perturbation_type: str = 'noise',
    metrics: Optional[List[str]] = None,
    level_idx: int = 0,
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Plot relative performance change across metrics for a perturbation type.
    
    Parameters:
    -----------
    perturbation_results : dict
        Results from perturbation tests
    perturbation_type : str
        Type of perturbation to plot
    metrics : list of str or None
        Metrics to include (None for all)
    level_idx : int
        Index of perturbation level to use
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.Figure : Plot figure
    """
    # Extract metrics if not provided
    available_metrics = set()
    
    # Try different result structures
    if 'perturbations' in perturbation_results and perturbation_type in perturbation_results['perturbations']:
        type_results = perturbation_results['perturbations'][perturbation_type]
        
        if 'aggregate_results' in type_results:
            # Get available levels
            levels = sorted([float(level) for level in type_results['aggregate_results'].keys()])
            
            if len(levels) <= level_idx:
                raise ValueError(f"Level index {level_idx} out of range (max {len(levels)-1})")
                
            level = str(levels[level_idx])
            
            # Get metrics for this level
            if level in type_results['aggregate_results']:
                available_metrics = set(type_results['aggregate_results'][level].keys())
    
    if not available_metrics:
        raise ValueError(f"Could not find metrics for perturbation type '{perturbation_type}'")
    
    # Filter metrics if specified
    if metrics:
        metrics_to_plot = [m for m in metrics if m in available_metrics]
        if not metrics_to_plot:
            raise ValueError(f"None of the specified metrics {metrics} found in results")
    else:
        metrics_to_plot = list(available_metrics)
    
    # Extract data
    metric_data = {}
    
    for metric in metrics_to_plot:
        features = []
        rel_changes = []
        
        if 'perturbations' in perturbation_results and perturbation_type in perturbation_results['perturbations']:
            type_results = perturbation_results['perturbations'][perturbation_type]
            
            if 'feature_results' in type_results:
                for feature, feature_data in type_results['feature_results'].items():
                    if len(feature_data) > level_idx and 'relative_change' in feature_data[level_idx]:
                        if metric in feature_data[level_idx]['relative_change']:
                            features.append(feature)
                            rel_changes.append(feature_data[level_idx]['relative_change'][metric])
        
        if features and rel_changes:
            metric_data[metric] = {
                'features': features,
                'rel_changes': rel_changes
            }
    
    if not metric_data:
        raise ValueError(f"Could not extract relative change data for metrics {metrics_to_plot}")
    
    # Create plot
    fig, axes = plt.subplots(
        len(metric_data), 1, 
        figsize=figsize, 
        sharey=True,
        sharex=True
    )
    
    # Handle single metric case
    if len(metric_data) == 1:
        axes = [axes]
    
    # Get level value
    level_value = perturbation_results.get('perturbation_levels', [0.1, 0.2, 0.5])[level_idx] \
                 if 'perturbation_levels' in perturbation_results else "?"
    
    # Plot each metric
    for i, (metric, data) in enumerate(metric_data.items()):
        ax = axes[i]
        
        # Sort features by impact
        idx = np.argsort(data['rel_changes'])
        features = [data['features'][i] for i in idx]
        rel_changes = [data['rel_changes'][i] for i in idx]
        
        # Plot horizontal bar chart
        bars = ax.barh(range(len(features)), rel_changes, align='center', height=0.7)
        
        # Color bars based on sign
        for j, bar in enumerate(bars):
            if rel_changes[j] < 0:
                bar.set_color('red')
            else:
                bar.set_color('green')
        
        # Add vertical line at 0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Set labels
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        
        if i == len(metric_data) - 1:
            ax.set_xlabel('Relative Change')
            
        ax.set_title(f'{metric.upper()}')
    
    # Set overall title
    fig.suptitle(f'Impact of {perturbation_type.title()} Perturbation (Level {level_value}) Across Metrics', 
                fontsize=14)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle
    return fig