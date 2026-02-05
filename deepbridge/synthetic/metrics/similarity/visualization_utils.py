"""
Utility functions for visualizing synthetic data.
"""

import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def visualize_data_comparison(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    method: str = 'tsne',
    n_samples: int = 1000,
    random_state: int = 42,
    figsize: t.Tuple[int, int] = (12, 10),
    save_path: t.Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Visualize comparison between original and synthetic data.

    Args:
        original_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        method: Visualization method ('tsne', 'pca')
        n_samples: Number of samples to visualize
        random_state: Random seed
        figsize: Figure size
        save_path: Path to save the figure
        **kwargs: Additional parameters

    Returns:
        Matplotlib figure
    """
    # Sample data if needed
    if len(original_data) > n_samples:
        orig_sample = original_data.sample(
            n_samples, random_state=random_state
        )
    else:
        orig_sample = original_data

    if len(synthetic_data) > n_samples:
        synth_sample = synthetic_data.sample(
            n_samples, random_state=random_state
        )
    else:
        synth_sample = synthetic_data

    # Get common columns
    common_cols = sorted(set(orig_sample.columns) & set(synth_sample.columns))

    # Filter only numeric columns
    numeric_cols = [
        col
        for col in common_cols
        if pd.api.types.is_numeric_dtype(orig_sample[col])
    ]

    if len(numeric_cols) < 2:
        raise ValueError(
            'At least 2 numeric columns required for visualization'
        )

    # Prepare data
    X_orig = orig_sample[numeric_cols].fillna(0).values
    X_synth = synth_sample[numeric_cols].fillna(0).values

    # Combine data for scaling
    X_combined = np.vstack([X_orig, X_synth])

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)

    # Split back
    X_orig_scaled = X_scaled[: len(X_orig)]
    X_synth_scaled = X_scaled[len(X_orig) :]

    # Reduce dimensionality
    if method == 'tsne':
        # t-SNE for visualization
        reducer = TSNE(n_components=2, random_state=random_state, **kwargs)
        X_reduced = reducer.fit_transform(X_scaled)
    else:
        # PCA as default
        reducer = PCA(n_components=2, random_state=random_state)
        X_reduced = reducer.fit_transform(X_scaled)

    # Split back
    X_orig_reduced = X_reduced[: len(X_orig)]
    X_synth_reduced = X_reduced[len(X_orig) :]

    # Create labels
    orig_labels = ['Original'] * len(X_orig)
    synth_labels = ['Synthetic'] * len(X_synth)

    # Create dataframe for plotting
    plot_df = pd.DataFrame(
        {
            'x': np.concatenate([X_orig_reduced[:, 0], X_synth_reduced[:, 0]]),
            'y': np.concatenate([X_orig_reduced[:, 1], X_synth_reduced[:, 1]]),
            'type': orig_labels + synth_labels,
        }
    )

    # Create plot
    plt.figure(figsize=figsize)
    fig = plt.figure(figsize=figsize)

    # Create scatter plot
    ax = fig.add_subplot(111)
    sns.scatterplot(
        data=plot_df, x='x', y='y', hue='type', alpha=0.7, s=50, ax=ax
    )

    # Set title and labels
    ax.set_title(
        f'Comparison of Original vs Synthetic Data ({method.upper()})',
        fontsize=16,
    )
    ax.set_xlabel('Dimension 1', fontsize=12)
    ax.set_ylabel('Dimension 2', fontsize=12)

    # Improve legend
    ax.legend(title='Data Type', fontsize=12, title_fontsize=14)

    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Set aspect ratio
    ax.set_aspect('equal')

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_distribution_comparison(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    columns: t.Optional[t.List[str]] = None,
    max_cols: int = 6,
    n_rows: int = 2,
    figsize: t.Optional[t.Tuple[int, int]] = None,
    save_path: t.Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot distribution comparison between original and synthetic data.

    Args:
        original_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        columns: Specific columns to plot (if None, selects automatically)
        max_cols: Maximum number of columns to plot
        n_rows: Number of rows in the plot grid
        figsize: Figure size
        save_path: Path to save the figure
        **kwargs: Additional parameters

    Returns:
        Matplotlib figure
    """
    # Select columns to plot
    if columns is None:
        # Get common columns
        common_cols = list(
            set(original_data.columns) & set(synthetic_data.columns)
        )

        # Prioritize numerical columns
        num_cols = [
            col
            for col in common_cols
            if pd.api.types.is_numeric_dtype(original_data[col])
        ]
        cat_cols = [col for col in common_cols if col not in num_cols]

        # Mix numerical and categorical columns
        if max_cols <= len(num_cols) + len(cat_cols):
            # Take half from each type if possible
            num_take = min(max_cols // 2, len(num_cols))
            cat_take = min(max_cols - num_take, len(cat_cols))
            columns = num_cols[:num_take] + cat_cols[:cat_take]
        else:
            columns = num_cols + cat_cols

        # Limit to max_cols
        columns = columns[:max_cols]
    else:
        # Ensure columns exist in both datasets
        columns = [
            col
            for col in columns
            if col in original_data.columns and col in synthetic_data.columns
        ]
        columns = columns[:max_cols]

    # Calculate grid dimensions
    n_cols = min(3, len(columns))
    if n_rows is None:
        n_rows = (len(columns) + n_cols - 1) // n_cols

    # Calculate figure size if not specified
    if figsize is None:
        figsize = (5 * n_cols, 4 * n_rows)

    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # Plot each column
    for i, col in enumerate(columns):
        if i < len(axes):
            ax = axes[i]

            # Check if column is numeric
            if (
                pd.api.types.is_numeric_dtype(original_data[col])
                and original_data[col].nunique() > 10
            ):
                # Numeric column - plot histogram or KDE
                sns.histplot(
                    original_data[col],
                    ax=ax,
                    color='blue',
                    alpha=0.5,
                    label='Original',
                    kde=True,
                )
                sns.histplot(
                    synthetic_data[col],
                    ax=ax,
                    color='red',
                    alpha=0.5,
                    label='Synthetic',
                    kde=True,
                )
            else:
                # Categorical column - plot count plot
                orig_counts = original_data[col].value_counts(normalize=True)
                synth_counts = synthetic_data[col].value_counts(normalize=True)

                # Combine and get top categories
                combined = pd.concat(
                    [
                        orig_counts.rename('Original'),
                        synth_counts.rename('Synthetic'),
                    ],
                    axis=1,
                ).fillna(0)

                # Limit categories if too many
                if len(combined) > 10:
                    combined = combined.iloc[:10]

                # Convert to long format
                plot_data = combined.reset_index().melt(
                    id_vars='index', var_name='Source', value_name='Proportion'
                )

                # Plot
                sns.barplot(
                    data=plot_data,
                    x='index',
                    y='Proportion',
                    hue='Source',
                    ax=ax,
                )
                ax.set_xticklabels(
                    ax.get_xticklabels(), rotation=45, ha='right'
                )

            # Set title and legends
            ax.set_title(col)
            ax.legend()

    # Hide empty subplots
    for i in range(len(columns), len(axes)):
        axes[i].set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_correlation_comparison(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    columns: t.Optional[t.List[str]] = None,
    figsize: t.Tuple[int, int] = (18, 8),
    save_path: t.Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot correlation matrix comparison between original and synthetic data.

    Args:
        original_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        columns: Columns to include in the plot (if None, uses all numerical)
        figsize: Figure size
        save_path: Path to save the figure
        **kwargs: Additional parameters

    Returns:
        Matplotlib figure
    """
    # Select columns to include
    if columns is None:
        # Get common numerical columns
        columns = [
            col
            for col in original_data.columns
            if pd.api.types.is_numeric_dtype(original_data[col])
            and col in synthetic_data.columns
            and pd.api.types.is_numeric_dtype(synthetic_data[col])
        ]

        # Limit to 10 columns for readability
        if len(columns) > 10:
            columns = columns[:10]

    # Ensure there are enough columns
    if len(columns) < 2:
        raise ValueError(
            'At least 2 numerical columns required for correlation comparison'
        )

    # Calculate correlation matrices
    orig_corr = original_data[columns].corr()
    synth_corr = synthetic_data[columns].corr()

    # Calculate absolute difference
    diff_corr = (orig_corr - synth_corr).abs()

    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)

    # Define common parameters
    cmap = kwargs.get('cmap', 'coolwarm')
    vmin = kwargs.get('vmin', -1)
    vmax = kwargs.get('vmax', 1)

    # Plot original data correlation
    sns.heatmap(
        orig_corr,
        ax=ax1,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt='.2f',
        square=True,
        cbar_kws={'shrink': 0.8},
    )
    ax1.set_title('Original Data Correlation', fontsize=14)

    # Plot synthetic data correlation
    sns.heatmap(
        synth_corr,
        ax=ax2,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt='.2f',
        square=True,
        cbar_kws={'shrink': 0.8},
    )
    ax2.set_title('Synthetic Data Correlation', fontsize=14)

    # Plot absolute difference
    sns.heatmap(
        diff_corr,
        ax=ax3,
        cmap='Reds',
        vmin=0,
        vmax=1,
        annot=True,
        fmt='.2f',
        square=True,
        cbar_kws={'shrink': 0.8},
    )
    ax3.set_title('Absolute Difference', fontsize=14)

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_privacy_risk(
    privacy_metrics: dict,
    figsize: t.Tuple[int, int] = (12, 6),
    save_path: t.Optional[str] = None,
) -> plt.Figure:
    """
    Plot privacy risk assessment visualizations.

    Args:
        privacy_metrics: Dictionary with privacy metrics
        figsize: Figure size
        save_path: Path to save the figure

    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Plot 1: Distance histogram
    if 'distance_histogram' in privacy_metrics:
        hist = privacy_metrics['distance_histogram']['counts']
        bin_edges = privacy_metrics['distance_histogram']['bin_edges']
        bin_centers = [
            (bin_edges[i] + bin_edges[i + 1]) / 2
            for i in range(len(bin_edges) - 1)
        ]

        ax1.bar(
            bin_centers, hist, width=(bin_edges[1] - bin_edges[0]), alpha=0.7
        )
        ax1.set_title(
            'Distribution of Nearest Neighbor Distances', fontsize=14
        )
        ax1.set_xlabel('Distance to Nearest Original Record', fontsize=12)
        ax1.set_ylabel('Count', fontsize=12)

        # Add vertical lines for risk thresholds
        ax1.axvline(
            x=0.05, color='red', linestyle='--', alpha=0.7, label='High Risk'
        )
        ax1.axvline(
            x=0.2,
            color='orange',
            linestyle='--',
            alpha=0.7,
            label='Medium Risk',
        )
        ax1.legend()

    # Plot 2: Risk levels pie chart
    if 'risk_levels' in privacy_metrics:
        risk_levels = privacy_metrics['risk_levels']
        labels = ['High Risk', 'Medium Risk', 'Low Risk']
        sizes = [
            risk_levels['high_risk'],
            risk_levels['medium_risk'],
            risk_levels['low_risk'],
        ]
        colors = ['#e74c3c', '#f39c12', '#2ecc71']

        ax2.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            shadow=False,
            startangle=90,
            wedgeprops={'alpha': 0.8},
        )
        ax2.set_title('Privacy Risk Levels', fontsize=14)
        ax2.axis(
            'equal'
        )  # Equal aspect ratio ensures that pie is drawn as a circle

    # Add overall metrics text
    if 'privacy_risk_pct' in privacy_metrics:
        plt.figtext(
            0.5,
            0.01,
            f"Overall Privacy Risk: {privacy_metrics['privacy_risk_pct']:.1f}% of records | "
            f"Mean Distance: {privacy_metrics['mean_distance']:.3f} | "
            f"Min Distance: {privacy_metrics['min_distance']:.3f}",
            ha='center',
            fontsize=12,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
        )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_attribute_distributions(
    original_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    column: str,
    by_column: t.Optional[str] = None,
    figsize: t.Optional[t.Tuple[int, int]] = None,
    save_path: t.Optional[str] = None,
    **kwargs,
) -> plt.Figure:
    """
    Plot detailed distribution comparison for a specific attribute.

    Args:
        original_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        column: Column to analyze
        by_column: Column to group by (for conditional distributions)
        figsize: Figure size
        save_path: Path to save the figure
        **kwargs: Additional parameters

    Returns:
        Matplotlib figure
    """
    # Check if columns exist
    if (
        column not in original_data.columns
        or column not in synthetic_data.columns
    ):
        raise ValueError(f"Column '{column}' not found in both datasets")

    if by_column and (
        by_column not in original_data.columns
        or by_column not in synthetic_data.columns
    ):
        raise ValueError(f"Column '{by_column}' not found in both datasets")

    # Determine if the column is numeric
    is_numeric = (
        pd.api.types.is_numeric_dtype(original_data[column])
        and original_data[column].nunique() > 10
    )

    # Determine plot type and size
    if by_column is None:
        # Simple distribution comparison
        if figsize is None:
            figsize = (12, 6)

        fig, ax = plt.subplots(figsize=figsize)

        if is_numeric:
            # For numeric column, plot KDE or histogram
            sns.histplot(
                original_data[column].dropna(),
                color='blue',
                alpha=0.5,
                label='Original',
                kde=True,
                ax=ax,
            )
            sns.histplot(
                synthetic_data[column].dropna(),
                color='red',
                alpha=0.5,
                label='Synthetic',
                kde=True,
                ax=ax,
            )

            # Add summary statistics as text
            orig_stats = original_data[column].describe()
            synth_stats = synthetic_data[column].describe()

            stats_text = (
                f"Original:   Mean: {orig_stats['mean']:.2f}  Std: {orig_stats['std']:.2f}  "
                f"Min: {orig_stats['min']:.2f}  Max: {orig_stats['max']:.2f}\n"
                f"Synthetic: Mean: {synth_stats['mean']:.2f}  Std: {synth_stats['std']:.2f}  "
                f"Min: {synth_stats['min']:.2f}  Max: {synth_stats['max']:.2f}"
            )

            plt.figtext(
                0.5,
                0.01,
                stats_text,
                ha='center',
                fontsize=10,
                bbox=dict(
                    boxstyle='round,pad=0.5', facecolor='white', alpha=0.8
                ),
            )

        else:
            # For categorical column, plot count or percentage
            orig_counts = (
                original_data[column].value_counts(normalize=True).sort_index()
            )
            synth_counts = (
                synthetic_data[column]
                .value_counts(normalize=True)
                .sort_index()
            )

            # Combine and reindex to include all categories
            all_categories = list(
                set(orig_counts.index) | set(synth_counts.index)
            )
            orig_counts = orig_counts.reindex(all_categories, fill_value=0)
            synth_counts = synth_counts.reindex(all_categories, fill_value=0)

            # Create DataFrame for plotting
            plot_df = pd.DataFrame(
                {'Original': orig_counts, 'Synthetic': synth_counts}
            ).sort_index()

            # For many categories, limit to top N
            if len(plot_df) > 15:
                # Get top categories by combined frequency
                plot_df['combined'] = (
                    plot_df['Original'] + plot_df['Synthetic']
                )
                plot_df = plot_df.nlargest(15, 'combined')
                del plot_df['combined']

            # Plot as bar chart
            plot_df.plot(kind='bar', ax=ax)

            # Rotate x-labels for readability
            plt.xticks(rotation=45, ha='right')

        # Set title and labels
        ax.set_title(f'Distribution Comparison: {column}', fontsize=14)
        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Density' if is_numeric else 'Proportion', fontsize=12)
        ax.legend()

    else:
        # Conditional distribution by another column
        by_column_values = sorted(
            list(
                set(original_data[by_column].unique())
                & set(synthetic_data[by_column].unique())
            )
        )

        # Limit to 4 values for readability
        if len(by_column_values) > 4:
            # Get most frequent values
            orig_by_counts = original_data[by_column].value_counts()
            synth_by_counts = synthetic_data[by_column].value_counts()
            combined_counts = orig_by_counts.add(synth_by_counts, fill_value=0)
            by_column_values = combined_counts.nlargest(4).index.tolist()

        # Calculate grid dimensions
        n_cols = min(2, len(by_column_values))
        n_rows = (len(by_column_values) + n_cols - 1) // n_cols

        if figsize is None:
            figsize = (6 * n_cols, 5 * n_rows)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        # Plot for each value of the by_column
        for i, value in enumerate(by_column_values):
            if i < len(axes):
                ax = axes[i]

                # Filter data
                orig_filtered = original_data[
                    original_data[by_column] == value
                ]
                synth_filtered = synthetic_data[
                    synthetic_data[by_column] == value
                ]

                if is_numeric:
                    # Plot numeric distribution
                    sns.histplot(
                        orig_filtered[column].dropna(),
                        color='blue',
                        alpha=0.5,
                        label='Original',
                        kde=True,
                        ax=ax,
                    )
                    sns.histplot(
                        synth_filtered[column].dropna(),
                        color='red',
                        alpha=0.5,
                        label='Synthetic',
                        kde=True,
                        ax=ax,
                    )
                else:
                    # Plot categorical distribution
                    orig_counts = orig_filtered[column].value_counts(
                        normalize=True
                    )
                    synth_counts = synth_filtered[column].value_counts(
                        normalize=True
                    )

                    all_categories = list(
                        set(orig_counts.index) | set(synth_counts.index)
                    )
                    orig_counts = orig_counts.reindex(
                        all_categories, fill_value=0
                    )
                    synth_counts = synth_counts.reindex(
                        all_categories, fill_value=0
                    )

                    pd.DataFrame(
                        {'Original': orig_counts, 'Synthetic': synth_counts}
                    ).plot(kind='bar', ax=ax)

                    # Rotate x-labels
                    ax.set_xticklabels(
                        ax.get_xticklabels(), rotation=45, ha='right'
                    )

                # Set title and legend
                ax.set_title(f'{by_column} = {value}', fontsize=12)
                ax.legend()

        # Hide empty subplots
        for i in range(len(by_column_values), len(axes)):
            axes[i].set_visible(False)

        # Set overall title
        plt.suptitle(
            f'Distribution of {column} by {by_column}', fontsize=16, y=1.02
        )

    # Adjust layout
    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
