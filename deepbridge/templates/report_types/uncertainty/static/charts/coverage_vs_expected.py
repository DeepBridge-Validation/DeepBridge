"""
Module for generating coverage vs expected coverage charts.
"""

import logging
from typing import Dict, List, Any, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class CoverageVsExpectedChart(BaseChartGenerator):
    """
    Generate charts comparing real coverage with expected coverage at different alpha levels.
    """
    
    def _validate_data(self, models_data):
        """Validate input data for the chart."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        
        logger.info(f"Validating coverage vs expected data: {type(models_data)}")
        
        if not isinstance(models_data, dict) or not models_data:
            logger.warning("models_data is not a dictionary or is empty")
            return False
            
        # Log the keys of models_data
        logger.info(f"models_data keys: {list(models_data.keys())}")
        
        # At least one model needs to have valid data
        for model_name, model_data in models_data.items():
            logger.info(f"Checking model data for '{model_name}': {type(model_data)}")
            
            if not isinstance(model_data, dict):
                logger.warning(f"Model data for '{model_name}' is not a dictionary")
                continue
                
            # Log the keys in model_data
            logger.info(f"model_data keys for '{model_name}': {list(model_data.keys())}")
            
            required_keys = ['alphas', 'coverages', 'expected_coverages']
            for key in required_keys:
                if key not in model_data:
                    logger.warning(f"Missing required key '{key}' in model data for '{model_name}'")
                    
            if all(key in model_data for key in required_keys):
                # Check if data is not empty
                is_valid = True
                for key in required_keys:
                    if not isinstance(model_data[key], (list, tuple)):
                        logger.warning(f"'{key}' for model '{model_name}' is not a list or tuple: {type(model_data[key])}")
                        is_valid = False
                    elif len(model_data[key]) == 0:
                        logger.warning(f"'{key}' for model '{model_name}' is empty")
                        is_valid = False
                    else:
                        logger.info(f"'{key}' for model '{model_name}' has {len(model_data[key])} values")
                        
                if is_valid:
                    logger.info(f"Found valid data for model '{model_name}'")
                    return True
                        
        logger.warning("No valid data found in models_data")
        return False
    
    def generate(self,
                models_data: Dict[str, Dict[str, Any]],
                title: str = "Coverage vs Expected Coverage",
                add_annotations: bool = True) -> str:
        """
        Generate a line chart comparing real coverage with expected coverage for different alpha values.

        Parameters:
        -----------
        models_data : Dict[str, Dict[str, Any]]
            Dictionary with model names as keys and dictionaries containing coverage data
            Expected structure:
            {
                "model_name": {
                    "alphas": [alpha1, alpha2, ...],
                    "coverages": [coverage1, coverage2, ...],
                    "expected_coverages": [expected1, expected2, ...] 
                },
                ...
            }
        title : str, optional
            Chart title
        add_annotations : bool, optional
            Whether to add alpha annotations to the chart

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Validate input data
        if not self._validate_data(models_data):
            logger.warning("Invalid models data for coverage vs expected coverage chart")
            return ""

        # If using existing chart generator
        if self.chart_generator and hasattr(self.chart_generator, 'line_chart'):
            try:
                # Prepare data for chart generator
                # This depends on the expected format for the chart generator
                # You'll need to adjust this based on your chart generator's API
                return ""
            except Exception as e:
                logger.error(f"Error using chart generator for coverage vs expected coverage: {str(e)}")

        # Fallback - implement direct charting
        try:
            # Create figure
            fig, ax = self.plt.subplots(figsize=(12, 8))
            
            # Add reference diagonal line for ideal coverage
            ax.plot([0.65, 1.0], [0.65, 1.0], 'k--', alpha=0.5, label='Ideal')

            # Define colors for different models
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
            
            # Plot each model's data
            for i, (model_name, model_data) in enumerate(models_data.items()):
                # Check for required data
                if not all(key in model_data for key in ['alphas', 'coverages', 'expected_coverages']):
                    continue
                
                alphas = model_data['alphas']
                coverages = model_data['coverages']
                expected_coverages = model_data['expected_coverages']
                
                # Ensure all lists have the same length
                min_len = min(len(alphas), len(coverages), len(expected_coverages))
                
                if min_len == 0:
                    continue
                    
                alphas = alphas[:min_len]
                coverages = coverages[:min_len]
                expected_coverages = expected_coverages[:min_len]
                
                # Plot the coverage vs expected coverage
                color = colors[i % len(colors)]
                ax.plot(expected_coverages, coverages, 'o-', linewidth=2, markersize=8, 
                       label=model_name, color=color)
                
                # Add annotations for alpha values if requested
                if add_annotations and i == 0:  # Only annotate the first model to avoid clutter
                    for j, alpha in enumerate(alphas):
                        ax.annotate(f'α={alpha}', 
                                   xy=(expected_coverages[j], coverages[j]), 
                                   xytext=(-10, 10),
                                   textcoords='offset points', 
                                   fontsize=10,
                                   color='black')
            
            # Set labels and title
            ax.set_xlabel('Expected Coverage')
            ax.set_ylabel('Actual Coverage')
            ax.set_title(title)
            
            # Set limits to focus on relevant area
            ax.set_xlim(0.65, 1.0)
            ax.set_ylim(0.65, 1.0)
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='lower right')
            
            # Add explanation text
            textstr = '\n'.join((
                'Closer to diagonal = better calibration',
                'α=0.01 → 99% expected coverage',
                'α=0.05 → 95% expected coverage',
                'α=0.10 → 90% expected coverage',
                'α=0.20 → 80% expected coverage'
            ))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            # Save the figure to base64
            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating coverage vs expected coverage chart: {str(e)}")
            return ""