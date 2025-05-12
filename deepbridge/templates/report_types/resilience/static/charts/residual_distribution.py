"""
Module for generating residual distribution charts.
"""

import logging
import numpy as np
from typing import List, Optional

from .base_chart import BaseChartGenerator

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class ResidualDistributionChart(BaseChartGenerator):
    """
    Generate charts showing the distribution of residuals.
    """
    
    def generate(self,
                worst_residuals: List[float] = None,
                remaining_residuals: List[float] = None,
                all_residuals: List[float] = None,
                title: str = "Model Residual Distribution") -> str:
        """
        Generate a chart showing the distribution of residuals.

        Parameters:
        -----------
        worst_residuals : List[float], optional
            Residuals for worst samples
        remaining_residuals : List[float], optional
            Residuals for remaining samples
        all_residuals : List[float], optional
            All residuals (used if worst and remaining not provided)
        title : str, optional
            Chart title

        Returns:
        --------
        str : Base64 encoded image or empty string if data invalid
        """
        self._validate_chart_generator()

        # Use any available residual data
        has_worst = self._validate_data(worst_residuals)
        has_remaining = self._validate_data(remaining_residuals)
        has_all = self._validate_data(all_residuals)
        
        if not has_worst and not has_remaining and not has_all:
            logger.warning("No valid residuals data for distribution chart")
            return ""
        
        # Clean data by converting to float and handling any invalid values
        clean_worst = []
        clean_remaining = []
        clean_all = []
        
        if has_worst:
            for val in worst_residuals:
                try:
                    clean_worst.append(float(val))
                except (ValueError, TypeError):
                    continue
                    
        if has_remaining:
            for val in remaining_residuals:
                try:
                    clean_remaining.append(float(val))
                except (ValueError, TypeError):
                    continue
                    
        if has_all:
            for val in all_residuals:
                try:
                    clean_all.append(float(val))
                except (ValueError, TypeError):
                    continue
                    
        # Update validation flags based on cleaned data
        has_worst = len(clean_worst) > 0
        has_remaining = len(clean_remaining) > 0
        has_all = len(clean_all) > 0
        
        if not has_worst and not has_remaining and not has_all:
            logger.warning("No valid numeric residuals for distribution chart")
            return ""
        
        # If using existing chart generator
        if self.chart_generator:
            # Create model data structure for boxplot
            residual_data = []
            
            if has_worst and has_remaining:
                # Use separate residuals
                residual_data = [
                    {
                        'name': 'Worst Residuals',
                        'scores': clean_worst,
                        'baseScore': None
                    },
                    {
                        'name': 'Remaining Residuals',
                        'scores': clean_remaining,
                        'baseScore': None
                    }
                ]
            elif has_all:
                # Use combined residuals
                residual_data = [
                    {
                        'name': 'All Residuals',
                        'scores': clean_all,
                        'baseScore': None
                    }
                ]
            elif has_worst:
                # Use only worst residuals
                residual_data = [
                    {
                        'name': 'Worst Residuals',
                        'scores': clean_worst,
                        'baseScore': None
                    }
                ]
            elif has_remaining:
                # Use only remaining residuals
                residual_data = [
                    {
                        'name': 'Remaining Residuals',
                        'scores': clean_remaining,
                        'baseScore': None
                    }
                ]
            
            if residual_data:
                try:
                    # Try boxplot_chart if available
                    if hasattr(self.chart_generator, 'boxplot_chart'):
                        return self.chart_generator.boxplot_chart(
                            models_data=residual_data,
                            title=title,
                            metric_name="Residual Value"
                        )
                except Exception as e:
                    logger.error(f"Error using chart generator for residual distribution: {str(e)}")
        
        # Fallback - implement direct charting if needed
        try:
            # Create figure
            fig, ax = self.plt.subplots(figsize=(12, 6))
            
            if has_worst and has_remaining:
                # Plot separate KDE for worst and remaining
                self.sns.kdeplot(clean_worst, ax=ax, color='red', label='Worst Samples', fill=True, alpha=0.4)
                self.sns.kdeplot(clean_remaining, ax=ax, color='blue', label='Remaining Samples', fill=True, alpha=0.4)
                
                # Add means as vertical lines
                worst_mean = np.mean(clean_worst)
                remaining_mean = np.mean(clean_remaining)
                
                ax.axvline(x=worst_mean, color='red', linestyle='--', alpha=0.7,
                          label=f'Mean (Worst): {worst_mean:.3f}')
                ax.axvline(x=remaining_mean, color='blue', linestyle='--', alpha=0.7,
                          label=f'Mean (Remaining): {remaining_mean:.3f}')
            elif has_all:
                # Plot single KDE for all residuals
                self.sns.kdeplot(clean_all, ax=ax, color='purple', label='All Residuals', fill=True)
                
                # Add mean as vertical line
                mean = np.mean(clean_all)
                ax.axvline(x=mean, color='purple', linestyle='--', alpha=0.7,
                          label=f'Mean: {mean:.3f}')
            elif has_worst:
                # Plot only worst residuals
                self.sns.kdeplot(clean_worst, ax=ax, color='red', label='Worst Samples', fill=True)
                
                # Add mean as vertical line
                mean = np.mean(clean_worst)
                ax.axvline(x=mean, color='red', linestyle='--', alpha=0.7,
                          label=f'Mean: {mean:.3f}')
            elif has_remaining:
                # Plot only remaining residuals
                self.sns.kdeplot(clean_remaining, ax=ax, color='blue', label='Remaining Samples', fill=True)
                
                # Add mean as vertical line
                mean = np.mean(clean_remaining)
                ax.axvline(x=mean, color='blue', linestyle='--', alpha=0.7,
                          label=f'Mean: {mean:.3f}')
            else:
                logger.warning("No residual data for distribution chart")
                return ""
            
            # Labels and title
            ax.set_title(title, fontsize=16)
            ax.set_xlabel("Residual Value", fontsize=14)
            ax.set_ylabel("Density", fontsize=14)
            
            # Add legend
            ax.legend()
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            self.plt.tight_layout()
            
            return self._save_figure_to_base64(fig)
        except Exception as e:
            logger.error(f"Error generating residual distribution chart: {str(e)}")
            return ""