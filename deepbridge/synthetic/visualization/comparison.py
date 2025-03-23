import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import typing as t
from pathlib import Path
import os

def plot_distributions(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    columns: t.Optional[t.List[str]] = None,
    max_cols: int = 10,
    figsize: t.Tuple[int, int] = (15, 12),
    save_path: t.Optional[t.Union[str, Path]] = None,
    **kwargs
) -> t.Optional[plt.Figure]:
    """
    Plot distribution comparisons between real and synthetic data.
    
    Args:
        real_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        columns: List of columns to plot (if None, selects a sample)
        max_cols: Maximum number of columns to plot
        figsize: Figure size (width, height)
        save_path: Path to save the figure (None to display)
        **kwargs: Additional parameters for plotting
            - categorical_columns: List of categorical column names
            - numerical_columns: List of numerical column names
            - n_bins: Number of bins for histograms
    
    Returns:
        Matplotlib figure if save_path is None, otherwise None
    """
    # Set the style for better visualization
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    # Determine which columns to plot
    if columns is None:
        # Select common columns between datasets
        common_cols = list(set(real_data.columns) & set(synthetic_data.columns))
        
        # Limit to max_cols
        if len(common_cols) > max_cols:
            columns = common_cols[:max_cols]
        else:
            columns = common_cols
    else:
        # Verify columns exist in both datasets
        valid_cols = []
        for col in columns:
            if col in real_data.columns and col in synthetic_data.columns:
                valid_cols.append(col)
            else:
                print(f"Warning: Column '{col}' not found in both datasets")
        
        columns = valid_cols[:max_cols]
    
    if not columns:
        print("No valid columns to plot")
        return None
    
    # Determine column types
    categorical_columns = kwargs.get('categorical_columns', [])
    numerical_columns = kwargs.get('numerical_columns', [])
    
    if not categorical_columns and not numerical_columns:
        # Infer types if not provided
        for col in columns:
            if pd.api.types.is_numeric_dtype(real_data[col]) and real_data[col].nunique() > 10:
                numerical_columns.append(col)
            else:
                categorical_columns.append(col)
    
    # Calculate optimal grid dimensions
    n_cols = min(3, len(columns))
    n_rows = (len(columns) + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=100)
    
    # Flatten axs array for easier indexing
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    elif n_rows == 1 or n_cols == 1:
        axs = axs.flatten()
    
    # Set background to white
    fig.patch.set_facecolor('white')
    
    # Plot each column
    for i, col in enumerate(columns):
        ax = axs.flatten()[i] if i < len(axs.flatten()) else axs.flatten()[-1]
        
        if col in numerical_columns:
            # Plot numerical column
            n_bins = kwargs.get('n_bins', 30)
            
            # Use KDE plot if there are many unique values
            if real_data[col].nunique() > 50 and synthetic_data[col].nunique() > 50:
                real_data[col].plot.kde(ax=ax, label="Real", color="blue", alpha=0.7)
                synthetic_data[col].plot.kde(ax=ax, label="Synthetic", color="red", alpha=0.7)
                ax.legend()
            else:
                # Use histograms
                try:
                    # Try to create common bin edges for both datasets
                    all_data = pd.concat([real_data[col], synthetic_data[col]])
                    
                    # Calculate bin edges
                    min_val = all_data.min()
                    max_val = all_data.max()
                    
                    # Add a small margin to avoid edge effects
                    margin = (max_val - min_val) * 0.05
                    bin_edges = np.linspace(min_val - margin, max_val + margin, n_bins + 1)
                    
                    # Plot histograms with common bin edges
                    ax.hist(real_data[col], bins=bin_edges, alpha=0.5, density=True, label="Real", color="blue")
                    ax.hist(synthetic_data[col], bins=bin_edges, alpha=0.5, density=True, label="Synthetic", color="red")
                except Exception as e:
                    # Fallback to separate histograms
                    print(f"Warning: Error creating common bins for {col}: {str(e)}")
                    ax.hist(real_data[col], bins=n_bins, alpha=0.5, density=True, label="Real", color="blue")
                    ax.hist(synthetic_data[col], bins=n_bins, alpha=0.5, density=True, label="Synthetic", color="red")
                
                ax.legend()
        else:
            # Plot categorical column
            # Get value counts for both datasets
            real_vc = real_data[col].value_counts(normalize=True)
            synth_vc = synthetic_data[col].value_counts(normalize=True)
            
            # Combine categories
            all_cats = pd.Series(list(set(real_vc.index) | set(synth_vc.index)))
            
            # Limit to top categories if there are too many
            max_cats = 20
            if len(all_cats) > max_cats:
                top_real = set(real_vc.nlargest(max_cats // 2).index)
                top_synth = set(synth_vc.nlargest(max_cats // 2).index)
                selected_cats = list(top_real | top_synth)
                
                # Add an "Other" category for the rest
                real_other = real_vc[~real_vc.index.isin(selected_cats)].sum()
                synth_other = synth_vc[~synth_vc.index.isin(selected_cats)].sum()
                
                real_vc = real_vc[real_vc.index.isin(selected_cats)]
                synth_vc = synth_vc[synth_vc.index.isin(selected_cats)]
                
                # Add "Other" category if it's significant
                if real_other > 0 or synth_other > 0:
                    real_vc["Other"] = real_other
                    synth_vc["Other"] = synth_other
            
            # Create DataFrame for plotting
            plot_df = pd.DataFrame({
                'Category': list(real_vc.index) + list(synth_vc.index),
                'Value': list(real_vc.values) + list(synth_vc.values),
                'Source': ['Real'] * len(real_vc) + ['Synthetic'] * len(synth_vc)
            })
            
            # Remove duplicates
            plot_df = plot_df.drop_duplicates(subset=['Category', 'Source'])
            
            # Plot
            sns.barplot(x='Category', y='Value', hue='Source', data=plot_df, ax=ax)
            
            # Rotate x-axis labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Set title and labels
        ax.set_title(col)
        ax.set_xlabel('')
        ax.set_ylabel('Density')
    
    # Hide unused subplots
    for i in range(len(columns), len(axs.flatten())):
        axs.flatten()[i].axis('off')
    
    # Add overall title
    fig.suptitle('Distribution Comparison: Real vs Synthetic Data', fontsize=16, y=1.02)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display
    if save_path is not None:
        # Create directory if it doesn't exist
        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        
        # Make sure figure has a white background
        fig.patch.set_facecolor('white')
        
        # Save with high quality and tight layout
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', format='png')
        plt.close(fig)
        return None
    else:
        return fig