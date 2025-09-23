"""
Module for generating reliability bandwidth (calibration) charts.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class ReliabilityBandwidthChart(BaseChartGenerator):
    """
    Generate reliability bandwidth charts showing calibration of predicted probabilities.
    This chart visualizes how well calibrated the model's predicted probabilities are
    by comparing predicted probabilities with actual observed frequencies.
    """

    def _calculate_calibration_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                                    n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate calibration curve data.

        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_prob : np.ndarray
            Predicted probabilities
        n_bins : int
            Number of bins for calibration

        Returns:
        --------
        Tuple containing:
            - mean predicted probabilities per bin
            - fraction of positives per bin (actual frequency)
            - bin edges
        """
        # Create bins
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Initialize arrays
        fraction_of_positives = np.zeros(n_bins)
        mean_predicted_value = np.zeros(n_bins)

        # Calculate calibration data for each bin
        for i, (bin_start, bin_end) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            # Find samples in this bin
            if i == n_bins - 1:  # Last bin includes upper edge
                bin_mask = (y_prob >= bin_start) & (y_prob <= bin_end)
            else:
                bin_mask = (y_prob >= bin_start) & (y_prob < bin_end)

            # Calculate statistics for this bin
            if np.sum(bin_mask) > 0:
                fraction_of_positives[i] = np.mean(y_true[bin_mask])
                mean_predicted_value[i] = np.mean(y_prob[bin_mask])
            else:
                fraction_of_positives[i] = 0
                mean_predicted_value[i] = (bin_start + bin_end) / 2

        return mean_predicted_value, fraction_of_positives, bin_edges

    def _calculate_confidence_bands(self, y_true: np.ndarray, y_prob: np.ndarray,
                                   n_bins: int = 10, confidence: float = 0.95,
                                   n_bootstraps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate confidence bands using bootstrap.

        Parameters:
        -----------
        y_true : np.ndarray
            True binary labels
        y_prob : np.ndarray
            Predicted probabilities
        n_bins : int
            Number of bins for calibration
        confidence : float
            Confidence level for bands (e.g., 0.95 for 95% confidence)
        n_bootstraps : int
            Number of bootstrap samples

        Returns:
        --------
        Tuple containing lower and upper confidence bands
        """
        n_samples = len(y_true)
        bootstrap_curves = []

        # Generate bootstrap samples
        for _ in range(n_bootstraps):
            # Resample with replacement
            idx = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[idx]
            y_prob_boot = y_prob[idx]

            # Calculate calibration curve for bootstrap sample
            mean_pred, frac_pos, _ = self._calculate_calibration_curve(
                y_true_boot, y_prob_boot, n_bins
            )
            bootstrap_curves.append(frac_pos)

        # Calculate confidence bands
        bootstrap_curves = np.array(bootstrap_curves)
        alpha = 1 - confidence
        lower_band = np.percentile(bootstrap_curves, (alpha / 2) * 100, axis=0)
        upper_band = np.percentile(bootstrap_curves, (1 - alpha / 2) * 100, axis=0)

        return lower_band, upper_band

    def generate(self,
                models_data: Dict[str, Dict[str, Any]],
                n_bins: int = 10,
                confidence: float = 0.95,
                title: str = "Reliability Bandwidth (Calibration)",
                show_confidence_bands: bool = True,
                show_histogram: bool = True) -> str:
        """
        Generate reliability bandwidth chart showing calibration with confidence bands.

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
            Number of bins for calibration curve
        confidence : float
            Confidence level for bands (e.g., 0.95 for 95%)
        title : str
            Chart title
        show_confidence_bands : bool
            Whether to show confidence bands
        show_histogram : bool
            Whether to show histogram of predictions

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Validate input data
        if not models_data or not isinstance(models_data, dict):
            logger.warning("Invalid models data for reliability bandwidth chart")
            return ""

        try:
            # Create figure with subplots if histogram is requested
            if show_histogram:
                fig, (ax1, ax2) = self.plt.subplots(2, 1, figsize=(10, 10),
                                                    gridspec_kw={'height_ratios': [3, 1]})
            else:
                fig, ax1 = self.plt.subplots(1, 1, figsize=(10, 8))

            # Color palette for different models
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

            # Plot perfect calibration line (diagonal)
            ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', alpha=0.7, linewidth=2)

            # Process each model
            for idx, (model_name, model_data) in enumerate(models_data.items()):
                if 'y_true' not in model_data or 'y_prob' not in model_data:
                    logger.warning(f"Missing required data for model {model_name}")
                    continue

                y_true = np.array(model_data['y_true'])
                y_prob = np.array(model_data['y_prob'])

                # Ensure binary labels
                if len(np.unique(y_true)) > 2:
                    logger.warning(f"Non-binary labels detected for model {model_name}")
                    continue

                # Calculate calibration curve
                mean_predicted, fraction_positives, bin_edges = self._calculate_calibration_curve(
                    y_true, y_prob, n_bins
                )

                # Get color for this model
                color = colors[idx % len(colors)]

                # Plot calibration curve
                ax1.plot(mean_predicted, fraction_positives,
                        marker='o', linewidth=2, markersize=8,
                        label=f'{model_name}', color=color)

                # Add confidence bands if requested
                if show_confidence_bands and len(y_true) > 30:  # Need enough samples
                    lower_band, upper_band = self._calculate_confidence_bands(
                        y_true, y_prob, n_bins, confidence
                    )

                    # Plot confidence bands
                    ax1.fill_between(mean_predicted, lower_band, upper_band,
                                    alpha=0.2, color=color,
                                    label=f'{model_name} {int(confidence*100)}% CI')

                # Add histogram if requested
                if show_histogram:
                    # Plot histogram of predicted probabilities
                    counts, _ = np.histogram(y_prob, bins=bin_edges)
                    width = bin_edges[1] - bin_edges[0]
                    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    ax2.bar(centers, counts, width=width, alpha=0.5,
                           color=color, edgecolor='black', linewidth=0.5,
                           label=f'{model_name} Distribution')

            # Customize main plot
            ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
            ax1.set_ylabel('Fraction of Positives (Observed Frequency)', fontsize=12)
            ax1.set_title(title, fontsize=14, fontweight='bold')
            ax1.set_xlim([0, 1])
            ax1.set_ylim([0, 1])
            ax1.grid(True, alpha=0.3, linestyle='--')
            ax1.legend(loc='upper left', fontsize=10)

            # Add calibration metrics as text
            ax1.text(0.95, 0.05, 'Lower values → Overconfident\nHigher values → Underconfident',
                    transform=ax1.transAxes, fontsize=9,
                    verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Customize histogram if present
            if show_histogram:
                ax2.set_xlabel('Predicted Probability', fontsize=12)
                ax2.set_ylabel('Count', fontsize=12)
                ax2.set_title('Distribution of Predicted Probabilities', fontsize=12)
                ax2.set_xlim([0, 1])
                ax2.grid(True, alpha=0.3, linestyle='--')
                ax2.legend(loc='upper right', fontsize=10)

            # Adjust layout
            fig.tight_layout()

            # Save to base64
            return self._save_figure_to_base64(fig)

        except Exception as e:
            logger.error(f"Error generating reliability bandwidth chart: {str(e)}")
            return ""

    def generate_from_intervals(self,
                               models_data: Dict[str, Dict[str, Any]],
                               alpha_levels: List[float] = None,
                               title: str = "Reliability Bandwidth from Prediction Intervals") -> str:
        """
        Alternative method to generate reliability bandwidth from prediction intervals.
        Useful when working with regression models with uncertainty quantification.

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
                    "alpha": float                 # Significance level (optional)
                },
                ...
            }
        alpha_levels : List[float]
            List of alpha levels to evaluate (default: [0.1, 0.2, 0.3, 0.4, 0.5])
        title : str
            Chart title

        Returns:
        --------
        str : Base64 encoded image
        """
        self._validate_chart_generator()

        if not models_data:
            logger.warning("No models data provided for reliability bandwidth")
            return ""

        if alpha_levels is None:
            alpha_levels = [0.1, 0.2, 0.3, 0.4, 0.5]

        try:
            fig, ax = self.plt.subplots(figsize=(10, 8))

            # Color palette
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

            # Plot perfect calibration line
            expected_coverage = [1 - alpha for alpha in alpha_levels]
            ax.plot(expected_coverage, expected_coverage, 'k--',
                   label='Perfect Calibration', alpha=0.7, linewidth=2)

            # Process each model
            for idx, (model_name, model_data) in enumerate(models_data.items()):
                actual_coverages = []

                for alpha in alpha_levels:
                    if 'y_true' in model_data and 'lower_bounds' in model_data and 'upper_bounds' in model_data:
                        y_true = np.array(model_data['y_true'])
                        lower = np.array(model_data['lower_bounds'])
                        upper = np.array(model_data['upper_bounds'])

                        # Calculate coverage
                        coverage = np.mean((y_true >= lower) & (y_true <= upper))
                        actual_coverages.append(coverage)
                    else:
                        logger.warning(f"Missing interval data for model {model_name}")
                        break

                if actual_coverages:
                    color = colors[idx % len(colors)]
                    ax.plot(expected_coverage, actual_coverages,
                           marker='o', linewidth=2, markersize=8,
                           label=model_name, color=color)

                    # Add confidence band (simplified)
                    n = len(model_data.get('y_true', []))
                    if n > 0:
                        std_err = np.sqrt(np.array(actual_coverages) * (1 - np.array(actual_coverages)) / n)
                        lower_ci = np.array(actual_coverages) - 1.96 * std_err
                        upper_ci = np.array(actual_coverages) + 1.96 * std_err
                        ax.fill_between(expected_coverage, lower_ci, upper_ci,
                                      alpha=0.2, color=color)

            # Customize plot
            ax.set_xlabel('Expected Coverage (1 - α)', fontsize=12)
            ax.set_ylabel('Actual Coverage', fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlim([0.4, 1.0])
            ax.set_ylim([0.4, 1.0])
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(loc='upper left', fontsize=10)

            # Add interpretation text
            ax.text(0.95, 0.05,
                   'Above diagonal: Conservative (wider intervals)\nBelow diagonal: Overconfident (narrow intervals)',
                   transform=ax.transAxes, fontsize=9,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            fig.tight_layout()
            return self._save_figure_to_base64(fig)

        except Exception as e:
            logger.error(f"Error generating reliability bandwidth from intervals: {str(e)}")
            return ""