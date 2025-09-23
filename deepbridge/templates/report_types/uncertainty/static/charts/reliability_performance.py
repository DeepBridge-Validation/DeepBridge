"""
Module for generating reliability performance charts.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class ReliabilityPerformanceChart(BaseChartGenerator):
    """
    Generate reliability performance charts showing how model performance varies
    with prediction confidence levels. This helps identify regions where the model
    is more or less reliable.
    """

    def _calculate_performance_by_confidence(self, y_true: np.ndarray, y_prob: np.ndarray,
                                            y_pred: np.ndarray = None,
                                            n_bins: int = 10,
                                            metric: str = 'accuracy') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate performance metrics for different confidence bins.

        Parameters:
        -----------
        y_true : np.ndarray
            True labels
        y_prob : np.ndarray
            Predicted probabilities
        y_pred : np.ndarray, optional
            Predicted classes (if None, will be derived from y_prob > 0.5)
        n_bins : int
            Number of confidence bins
        metric : str
            Performance metric to calculate ('accuracy', 'precision', 'recall', 'f1')

        Returns:
        --------
        Tuple containing:
            - bin centers (confidence levels)
            - performance values per bin
            - sample counts per bin
        """
        # Derive predictions if not provided
        if y_pred is None:
            y_pred = (y_prob > 0.5).astype(int)

        # Get confidence scores (max probability for each prediction)
        # For binary classification, confidence is max(prob, 1-prob)
        confidence = np.maximum(y_prob, 1 - y_prob)

        # Create confidence bins
        bin_edges = np.linspace(0.5, 1.0, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        performance_values = np.zeros(n_bins)
        sample_counts = np.zeros(n_bins)

        # Calculate performance for each bin
        for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            # Find samples in this confidence bin
            if i == n_bins - 1:  # Last bin includes upper edge
                bin_mask = (confidence >= bin_start) & (confidence <= bin_end)
            else:
                bin_mask = (confidence >= bin_start) & (confidence < bin_end)

            sample_counts[i] = np.sum(bin_mask)

            if sample_counts[i] > 0:
                # Calculate metric for this bin
                y_true_bin = y_true[bin_mask]
                y_pred_bin = y_pred[bin_mask]

                if metric == 'accuracy':
                    performance_values[i] = np.mean(y_true_bin == y_pred_bin)
                elif metric == 'precision':
                    if np.sum(y_pred_bin == 1) > 0:
                        performance_values[i] = np.sum((y_true_bin == 1) & (y_pred_bin == 1)) / np.sum(y_pred_bin == 1)
                    else:
                        performance_values[i] = 0
                elif metric == 'recall':
                    if np.sum(y_true_bin == 1) > 0:
                        performance_values[i] = np.sum((y_true_bin == 1) & (y_pred_bin == 1)) / np.sum(y_true_bin == 1)
                    else:
                        performance_values[i] = 0
                elif metric == 'f1':
                    precision = np.sum((y_true_bin == 1) & (y_pred_bin == 1)) / max(np.sum(y_pred_bin == 1), 1)
                    recall = np.sum((y_true_bin == 1) & (y_pred_bin == 1)) / max(np.sum(y_true_bin == 1), 1)
                    if precision + recall > 0:
                        performance_values[i] = 2 * precision * recall / (precision + recall)
                    else:
                        performance_values[i] = 0
            else:
                performance_values[i] = 0

        return bin_centers, performance_values, sample_counts

    def generate(self,
                models_data: Dict[str, Dict[str, Any]],
                n_bins: int = 10,
                metric: str = 'accuracy',
                title: str = "Reliability Performance",
                show_sample_sizes: bool = True,
                show_confidence_bars: bool = True) -> str:
        """
        Generate reliability performance chart.

        Parameters:
        -----------
        models_data : Dict[str, Dict[str, Any]]
            Dictionary with model names as keys and dictionaries containing:
            {
                "model_name": {
                    "y_true": np.ndarray,      # True labels
                    "y_prob": np.ndarray,       # Predicted probabilities
                    "y_pred": np.ndarray,       # Optional: predicted classes
                },
                ...
            }
        n_bins : int
            Number of confidence bins
        metric : str
            Performance metric ('accuracy', 'precision', 'recall', 'f1')
        title : str
            Chart title
        show_sample_sizes : bool
            Whether to show sample sizes in each bin
        show_confidence_bars : bool
            Whether to show confidence intervals

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Validate input data
        if not models_data or not isinstance(models_data, dict):
            logger.warning("Invalid models data for reliability performance chart")
            return ""

        try:
            # Create figure with subplots
            if show_sample_sizes:
                fig, (ax1, ax2) = self.plt.subplots(2, 1, figsize=(12, 10),
                                                    gridspec_kw={'height_ratios': [3, 1]})
            else:
                fig, ax1 = self.plt.subplots(1, 1, figsize=(12, 8))

            # Color palette
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # Track data for comparison
            all_bin_centers = None
            max_sample_count = 0

            # Process each model
            for idx, (model_name, model_data) in enumerate(models_data.items()):
                if 'y_true' not in model_data or 'y_prob' not in model_data:
                    logger.warning(f"Missing required data for model {model_name}")
                    continue

                y_true = np.array(model_data['y_true'])
                y_prob = np.array(model_data['y_prob'])
                y_pred = model_data.get('y_pred', None)

                # Calculate performance by confidence
                bin_centers, performance_values, sample_counts = self._calculate_performance_by_confidence(
                    y_true, y_prob, y_pred, n_bins, metric
                )

                if all_bin_centers is None:
                    all_bin_centers = bin_centers

                max_sample_count = max(max_sample_count, np.max(sample_counts))

                # Get color for this model
                color = colors[idx % len(colors)]

                # Plot performance line
                ax1.plot(bin_centers * 100, performance_values * 100,
                        marker='o', linewidth=2.5, markersize=8,
                        label=f'{model_name}', color=color)

                # Add confidence bars if requested
                if show_confidence_bars:
                    # Calculate standard error for binomial proportion
                    std_errors = np.sqrt(performance_values * (1 - performance_values) / np.maximum(sample_counts, 1))
                    confidence_interval = 1.96 * std_errors * 100  # 95% CI

                    ax1.errorbar(bin_centers * 100, performance_values * 100,
                               yerr=confidence_interval,
                               fmt='none', ecolor=color, alpha=0.3, capsize=5)

                # Plot sample sizes if requested
                if show_sample_sizes:
                    width = (bin_centers[1] - bin_centers[0]) * 100 * 0.8 / len(models_data)
                    offset = (idx - len(models_data) / 2) * width
                    ax2.bar(bin_centers * 100 + offset, sample_counts,
                           width=width, alpha=0.6, color=color, label=model_name)

            # Customize main performance plot
            ax1.set_xlabel('Confidence Level (%)', fontsize=12)
            ax1.set_ylabel(f'{metric.capitalize()} (%)', fontsize=12)
            ax1.set_title(f'{title} - {metric.capitalize()} by Confidence Level',
                         fontsize=14, fontweight='bold')
            ax1.set_xlim([48, 102])
            ax1.set_ylim([0, 105])
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='best', fontsize=10)

            # Add reference lines
            ax1.axhline(y=50, color='gray', linestyle=':', alpha=0.5, label='Random (50%)')
            if all_bin_centers is not None:
                # Add a trend line showing expected performance
                expected_performance = all_bin_centers * 100
                ax1.plot(all_bin_centers * 100, expected_performance,
                        'k--', alpha=0.3, linewidth=1, label='Expected (if perfectly calibrated)')

            # Add interpretation guide
            interpretation_text = (
                f'Interpretation:\n'
                f'• Upward trend: {metric.capitalize()} improves with confidence\n'
                f'• Flat line: {metric.capitalize()} independent of confidence\n'
                f'• Gap from diagonal: Calibration error'
            )
            ax1.text(0.02, 0.98, interpretation_text,
                    transform=ax1.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Customize sample size plot if present
            if show_sample_sizes:
                ax2.set_xlabel('Confidence Level (%)', fontsize=12)
                ax2.set_ylabel('Sample Count', fontsize=12)
                ax2.set_title('Sample Distribution across Confidence Bins', fontsize=12)
                ax2.set_xlim([48, 102])
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.legend(loc='upper left', fontsize=10)

            # Adjust layout
            fig.tight_layout()

            # Save to base64
            return self._save_figure_to_base64(fig)

        except Exception as e:
            logger.error(f"Error generating reliability performance chart: {str(e)}")
            return ""

    def generate_multi_metric(self,
                             models_data: Dict[str, Dict[str, Any]],
                             metrics: List[str] = None,
                             n_bins: int = 10,
                             title: str = "Multi-Metric Reliability Performance") -> str:
        """
        Generate reliability performance chart with multiple metrics in subplots.

        Parameters:
        -----------
        models_data : Dict[str, Dict[str, Any]]
            Dictionary with model data
        metrics : List[str]
            List of metrics to evaluate (default: ['accuracy', 'precision', 'recall', 'f1'])
        n_bins : int
            Number of confidence bins
        title : str
            Chart title

        Returns:
        --------
        str : Base64 encoded image
        """
        self._validate_chart_generator()

        if not models_data:
            logger.warning("No models data provided")
            return ""

        if metrics is None:
            metrics = ['accuracy', 'precision', 'recall', 'f1']

        try:
            # Create subplots for each metric
            n_metrics = len(metrics)
            fig, axes = self.plt.subplots(2, 2, figsize=(14, 12))
            axes = axes.flatten()

            # Color palette
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            # Process each metric
            for metric_idx, metric in enumerate(metrics):
                if metric_idx >= len(axes):
                    break

                ax = axes[metric_idx]

                # Process each model for this metric
                for model_idx, (model_name, model_data) in enumerate(models_data.items()):
                    if 'y_true' not in model_data or 'y_prob' not in model_data:
                        continue

                    y_true = np.array(model_data['y_true'])
                    y_prob = np.array(model_data['y_prob'])
                    y_pred = model_data.get('y_pred', None)

                    # Calculate performance by confidence
                    bin_centers, performance_values, sample_counts = self._calculate_performance_by_confidence(
                        y_true, y_prob, y_pred, n_bins, metric
                    )

                    color = colors[model_idx % len(colors)]

                    # Plot performance
                    ax.plot(bin_centers * 100, performance_values * 100,
                           marker='o', linewidth=2, markersize=6,
                           label=model_name, color=color)

                    # Add confidence intervals
                    std_errors = np.sqrt(performance_values * (1 - performance_values) /
                                        np.maximum(sample_counts, 1))
                    confidence_interval = 1.96 * std_errors * 100

                    ax.fill_between(bin_centers * 100,
                                   (performance_values * 100) - confidence_interval,
                                   (performance_values * 100) + confidence_interval,
                                   alpha=0.2, color=color)

                # Customize subplot
                ax.set_xlabel('Confidence Level (%)', fontsize=10)
                ax.set_ylabel(f'{metric.capitalize()} (%)', fontsize=10)
                ax.set_title(f'{metric.capitalize()}', fontsize=11, fontweight='bold')
                ax.set_xlim([48, 102])
                ax.set_ylim([0, 105])
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', fontsize=9)

            # Overall title
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

            # Remove any unused subplots
            for idx in range(len(metrics), len(axes)):
                fig.delaxes(axes[idx])

            # Adjust layout
            fig.tight_layout()

            # Save to base64
            return self._save_figure_to_base64(fig)

        except Exception as e:
            logger.error(f"Error generating multi-metric reliability performance: {str(e)}")
            return ""

    def generate_from_intervals(self,
                               models_data: Dict[str, Dict[str, Any]],
                               n_bins: int = 10,
                               title: str = "Interval-based Reliability Performance") -> str:
        """
        Generate reliability performance from prediction intervals for regression models.

        Parameters:
        -----------
        models_data : Dict[str, Dict[str, Any]]
            Dictionary with model names as keys and dictionaries containing:
            {
                "model_name": {
                    "predictions": np.ndarray,     # Point predictions
                    "lower_bounds": np.ndarray,    # Lower prediction intervals
                    "upper_bounds": np.ndarray,    # Upper prediction intervals
                    "y_true": np.ndarray,          # True values
                    "interval_widths": np.ndarray  # Optional: pre-computed interval widths
                },
                ...
            }

        Returns:
        --------
        str : Base64 encoded image
        """
        self._validate_chart_generator()

        if not models_data:
            logger.warning("No models data provided")
            return ""

        try:
            fig, (ax1, ax2) = self.plt.subplots(2, 1, figsize=(12, 10),
                                               gridspec_kw={'height_ratios': [1, 1]})

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            for idx, (model_name, model_data) in enumerate(models_data.items()):
                if not all(k in model_data for k in ['predictions', 'lower_bounds', 'upper_bounds', 'y_true']):
                    logger.warning(f"Missing required data for model {model_name}")
                    continue

                predictions = np.array(model_data['predictions'])
                lower = np.array(model_data['lower_bounds'])
                upper = np.array(model_data['upper_bounds'])
                y_true = np.array(model_data['y_true'])

                # Calculate interval widths
                interval_widths = upper - lower

                # Normalize interval widths to use as confidence (narrower = more confident)
                # Inverse relationship: smaller intervals mean higher confidence
                max_width = np.max(interval_widths)
                min_width = np.min(interval_widths)
                if max_width > min_width:
                    confidence = 1 - (interval_widths - min_width) / (max_width - min_width)
                else:
                    confidence = np.ones_like(interval_widths)

                # Create bins based on confidence
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

                coverage_per_bin = []
                mae_per_bin = []
                counts_per_bin = []

                for i in range(n_bins):
                    if i == n_bins - 1:
                        mask = (confidence >= bin_edges[i]) & (confidence <= bin_edges[i + 1])
                    else:
                        mask = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])

                    if np.sum(mask) > 0:
                        # Coverage: fraction of true values within intervals
                        coverage = np.mean((y_true[mask] >= lower[mask]) &
                                         (y_true[mask] <= upper[mask]))
                        coverage_per_bin.append(coverage)

                        # Mean Absolute Error
                        mae = np.mean(np.abs(y_true[mask] - predictions[mask]))
                        mae_per_bin.append(mae)

                        counts_per_bin.append(np.sum(mask))
                    else:
                        coverage_per_bin.append(np.nan)
                        mae_per_bin.append(np.nan)
                        counts_per_bin.append(0)

                color = colors[idx % len(colors)]

                # Plot coverage by confidence
                valid_mask = ~np.isnan(coverage_per_bin)
                ax1.plot(bin_centers[valid_mask] * 100,
                        np.array(coverage_per_bin)[valid_mask] * 100,
                        marker='o', linewidth=2, markersize=8,
                        label=model_name, color=color)

                # Plot MAE by confidence (inverted: lower is better)
                valid_mask_mae = ~np.isnan(mae_per_bin)
                ax2.plot(bin_centers[valid_mask_mae] * 100,
                        np.array(mae_per_bin)[valid_mask_mae],
                        marker='s', linewidth=2, markersize=8,
                        label=model_name, color=color)

            # Customize coverage plot
            ax1.set_xlabel('Confidence Level (based on interval width)', fontsize=12)
            ax1.set_ylabel('Coverage (%)', fontsize=12)
            ax1.set_title('Coverage vs Confidence Level', fontsize=12, fontweight='bold')
            ax1.set_xlim([-5, 105])
            ax1.set_ylim([0, 105])
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='best', fontsize=10)

            # Customize MAE plot
            ax2.set_xlabel('Confidence Level (based on interval width)', fontsize=12)
            ax2.set_ylabel('Mean Absolute Error', fontsize=12)
            ax2.set_title('Prediction Error vs Confidence Level', fontsize=12, fontweight='bold')
            ax2.set_xlim([-5, 105])
            ax2.grid(True, alpha=0.3, linestyle='--')
            ax2.legend(loc='best', fontsize=10)
            ax2.invert_yaxis()  # Lower MAE is better

            # Overall title
            fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)

            fig.tight_layout()
            return self._save_figure_to_base64(fig)

        except Exception as e:
            logger.error(f"Error generating interval-based reliability performance: {str(e)}")
            return ""