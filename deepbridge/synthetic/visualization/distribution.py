import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import typing as t
from pathlib import Path
import os

def plot_correlation_comparison(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    columns: t.Optional[t.List[str]] = None,
    figsize: t.Tuple[int, int] = (15, 6),
    save_path: t.Optional[t.Union[str, Path]] = None,
    **kwargs
) -> t.Optional[plt.Figure]:
    """
    Plot correlation matrix comparison between real and synthetic data.
    
    Args:
        real_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        columns: List of numerical columns to include (if None, selects all numerical)
        figsize: Figure size (width, height)
        save_path: Path to save the figure (None to display)
        **kwargs: Additional parameters for plotting
            - cmap: Colormap for heatmap
            - vmin: Minimum value for colormap normalization
            - vmax: Maximum value for colormap normalization
    
    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Set the style for better visualization
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Select columns to include
    if columns is None:
        # Select common numerical columns
        columns = []
        for col in set(real_data.columns) & set(synthetic_data.columns):
            if pd.api.types.is_numeric_dtype(real_data[col]) and pd.api.types.is_numeric_dtype(synthetic_data[col]):
                columns.append(col)
    
    if len(columns) < 2:
        print("Not enough numerical columns for correlation analysis")
        return None
    
    # Calculate correlation matrices
    real_corr = real_data[columns].corr()
    synth_corr = synthetic_data[columns].corr()
    
    # Plot correlation matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')  # Set figure background to white
    
    # Set style parameters
    cmap = kwargs.get('cmap', 'coolwarm')
    vmin = kwargs.get('vmin', -1)
    vmax = kwargs.get('vmax', 1)
    
    # Plot real data correlation
    sns.heatmap(real_corr, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax1, 
                fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .8})
    ax1.set_title('Real Data Correlation Matrix', fontsize=14, pad=10)
    
    # Plot synthetic data correlation
    sns.heatmap(synth_corr, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, ax=ax2, 
                fmt='.2f', square=True, linewidths=.5, cbar_kws={"shrink": .8})
    ax2.set_title('Synthetic Data Correlation Matrix', fontsize=14, pad=10)
    
    # Add overall title
    fig.suptitle('Correlation Matrix Comparison', fontsize=16, y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if save_path is not None:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
        plt.close(fig)
        return None
    else:
        return fig

def plot_pairwise_distributions(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    columns: t.Optional[t.List[str]] = None,
    max_cols: int = 5,
    figsize: t.Optional[t.Tuple[int, int]] = None,
    save_path: t.Optional[t.Union[str, Path]] = None,
    **kwargs
) -> t.Optional[plt.Figure]:
    """
    Plot pairwise distributions of real and synthetic data.
    
    Args:
        real_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        columns: List of numerical columns to include (if None, selects numerical)
        max_cols: Maximum number of columns to include in the plot
        figsize: Figure size (width, height)
        save_path: Path to save the figure (None to display)
        **kwargs: Additional parameters for plotting
            - n_samples: Number of samples to use from each dataset
            - plot_type: Type of plot ('scatter', 'kde', or 'hex')
            - alpha: Alpha transparency for scatter points
    
    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Set the style for better visualization
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Select columns to include
    if columns is None:
        # Select common numerical columns
        columns = []
        for col in set(real_data.columns) & set(synthetic_data.columns):
            if pd.api.types.is_numeric_dtype(real_data[col]) and pd.api.types.is_numeric_dtype(synthetic_data[col]):
                columns.append(col)
    
    if len(columns) < 2:
        print("Not enough numerical columns for pairwise analysis")
        return None
    
    # Limit to max_cols
    if len(columns) > max_cols:
        columns = columns[:max_cols]
    
    # Sample data if needed
    n_samples = kwargs.get('n_samples', 1000)
    if len(real_data) > n_samples:
        real_sample = real_data.sample(n_samples, random_state=42)
    else:
        real_sample = real_data
        
    if len(synthetic_data) > n_samples:
        synth_sample = synthetic_data.sample(n_samples, random_state=42)
    else:
        synth_sample = synthetic_data
    
    # Create DataFrames with source indicator
    real_sample = real_sample[columns].copy()
    real_sample['source'] = 'Real'
    
    synth_sample = synth_sample[columns].copy()
    synth_sample['source'] = 'Synthetic'
    
    # Combine data
    combined = pd.concat([real_sample, synth_sample], ignore_index=True)
    
    # Calculate figure size if not provided
    if figsize is None:
        figsize = (3 * len(columns), 3 * len(columns))
    
    # Create pairplot
    plot_type = kwargs.get('plot_type', 'scatter')
    
    if plot_type == 'scatter':
        g = sns.pairplot(
            combined, 
            hue='source', 
            diag_kind='kde',
            plot_kws={'alpha': kwargs.get('alpha', 0.6)},
            diag_kws={'alpha': 0.6},
            vars=columns
        )
        
        # Add title
        g.fig.suptitle('Pairwise Distributions: Real vs Synthetic', y=1.02, fontsize=16)
        g.fig.patch.set_facecolor('white')
        plt.tight_layout()
        
    elif plot_type == 'kde':
        g = sns.pairplot(
            combined, 
            hue='source', 
            diag_kind='kde',
            kind='kde',
            plot_kws={'levels': 5, 'alpha': 0.6},
            diag_kws={'alpha': 0.6},
            vars=columns
        )
        
        # Add title
        g.fig.suptitle('Pairwise Distributions: Real vs Synthetic', y=1.02, fontsize=16)
        g.fig.patch.set_facecolor('white')
        plt.tight_layout()
        
    elif plot_type == 'hex':
        # Create figure
        n_cols = len(columns)
        fig, axs = plt.subplots(n_cols, n_cols, figsize=figsize)
        fig.patch.set_facecolor('white')
        
        # Plot each pair
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                ax = axs[i, j]
                
                if i == j:
                    # Diagonal: plot KDE
                    real_sample[col1].plot.kde(ax=ax, color='blue', label='Real')
                    synth_sample[col1].plot.kde(ax=ax, color='red', label='Synthetic')
                    if i == 0:
                        ax.legend()
                else:
                    # Off-diagonal: plot hex bins
                    # Real data in one panel
                    hb = ax.hexbin(
                        real_sample[col2], 
                        real_sample[col1], 
                        gridsize=20, 
                        cmap='Blues', 
                        alpha=0.7
                    )
                    
                    # Add synthetic data with different color
                    hb2 = ax.hexbin(
                        synth_sample[col2], 
                        synth_sample[col1], 
                        gridsize=20, 
                        cmap='Reds', 
                        alpha=0.7
                    )
                
                # Set labels
                if i == n_cols - 1:
                    ax.set_xlabel(col2)
                else:
                    ax.set_xlabel('')
                
                if j == 0:
                    ax.set_ylabel(col1)
                else:
                    ax.set_ylabel('')
        
        # Add title
        fig.suptitle('Pairwise Distributions: Real vs Synthetic', y=0.98, fontsize=16)
        plt.tight_layout()
        g = fig
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Choose from 'scatter', 'kde', or 'hex'")
    
    # Save or display
    if save_path is not None:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        
        if plot_type == 'hex':
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
            plt.close(fig)
        else:
            g.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
            plt.close(g.fig)
        
        return None
    else:
        return g.fig if plot_type != 'hex' else g

def plot_joint_distribution(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    x_col: str,
    y_col: str,
    figsize: t.Tuple[int, int] = (12, 6),
    save_path: t.Optional[t.Union[str, Path]] = None,
    **kwargs
) -> t.Optional[plt.Figure]:
    """
    Plot joint distribution of two variables for real and synthetic data.
    
    Args:
        real_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        x_col: Column name for x-axis
        y_col: Column name for y-axis
        figsize: Figure size (width, height)
        save_path: Path to save the figure (None to display)
        **kwargs: Additional parameters for plotting
            - kind: Kind of plot ('scatter', 'kde', 'hex')
            - n_samples: Number of samples to use
            - alpha: Alpha transparency for scatter points
    
    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Set the style for better visualization
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Validate columns
    if x_col not in real_data.columns or x_col not in synthetic_data.columns:
        print(f"Column {x_col} not found in both datasets")
        return None
    
    if y_col not in real_data.columns or y_col not in synthetic_data.columns:
        print(f"Column {y_col} not found in both datasets")
        return None
    
    # Sample data if needed
    n_samples = kwargs.get('n_samples', 1000)
    if len(real_data) > n_samples:
        real_sample = real_data.sample(n_samples, random_state=42)
    else:
        real_sample = real_data
        
    if len(synthetic_data) > n_samples:
        synth_sample = synthetic_data.sample(n_samples, random_state=42)
    else:
        synth_sample = synthetic_data
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.patch.set_facecolor('white')
    
    # Determine plot type
    kind = kwargs.get('kind', 'scatter')
    alpha = kwargs.get('alpha', 0.6)
    
    if kind == 'scatter':
        # Scatter plot
        sns.scatterplot(x=x_col, y=y_col, data=real_sample, alpha=alpha, color='blue', ax=ax1)
        sns.scatterplot(x=x_col, y=y_col, data=synth_sample, alpha=alpha, color='red', ax=ax2)
    elif kind == 'kde':
        # KDE plot
        sns.kdeplot(x=x_col, y=y_col, data=real_sample, fill=True, cmap="Blues", ax=ax1)
        sns.kdeplot(x=x_col, y=y_col, data=synth_sample, fill=True, cmap="Reds", ax=ax2)
    elif kind == 'hex':
        # Hexbin plot
        ax1.hexbin(real_sample[x_col], real_sample[y_col], gridsize=20, cmap='Blues', alpha=0.7)
        ax2.hexbin(synth_sample[x_col], synth_sample[y_col], gridsize=20, cmap='Reds', alpha=0.7)
    else:
        raise ValueError(f"Unknown kind: {kind}. Choose from 'scatter', 'kde', or 'hex'")
    
    # Set titles and labels
    ax1.set_title('Real Data', fontsize=14)
    ax2.set_title('Synthetic Data', fontsize=14)
    
    ax1.set_xlabel(x_col)
    ax1.set_ylabel(y_col)
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(y_col)
    
    # Try to sync axis limits for better comparison
    x_min = min(real_sample[x_col].min(), synth_sample[x_col].min())
    x_max = max(real_sample[x_col].max(), synth_sample[x_col].max())
    y_min = min(real_sample[y_col].min(), synth_sample[y_col].min())
    y_max = max(real_sample[y_col].max(), synth_sample[y_col].max())
    
    # Add some margin
    x_margin = (x_max - x_min) * 0.05
    y_margin = (y_max - y_min) * 0.05
    
    ax1.set_xlim(x_min - x_margin, x_max + x_margin)
    ax1.set_ylim(y_min - y_margin, y_max + y_margin)
    ax2.set_xlim(x_min - x_margin, x_max + x_margin)
    ax2.set_ylim(y_min - y_margin, y_max + y_margin)
    
    # Add overall title
    fig.suptitle(f'Joint Distribution of {x_col} and {y_col}', fontsize=16, y=1.05)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if save_path is not None:
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
        plt.close(fig)
        return None
    else:
        return fig