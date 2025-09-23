"""
Uncertainty charts package - provides chart generation for uncertainty reports.
"""

from .base_chart import BaseChartGenerator
from .coverage_vs_expected import CoverageVsExpectedChart
from .width_vs_coverage import WidthVsCoverageChart
from .uncertainty_metrics import UncertaintyMetricsChart
from .feature_importance import FeatureImportanceChart
from .model_comparison import ModelComparisonChart
from .performance_gap_by_alpha import PerformanceGapByAlphaChart
from .interval_boxplot import IntervalBoxplotChart
from .psi_analysis import PSIAnalysisChart
from .top_features_distribution import TopFeaturesDistributionChart
from .reliability_bandwidth import ReliabilityBandwidthChart
from .reliability_performance import ReliabilityPerformanceChart
from .reliability_regions import ReliabilityRegionsChart

# Import and apply fixes
import logging
import sys
import traceback
import importlib
import importlib.util

logger = logging.getLogger("deepbridge.reports")

# Try to import and apply fixed radar chart implementation
try:
    # Check if fixed_radar.py exists
    fixed_radar_spec = importlib.util.find_spec('deepbridge.templates.report_types.uncertainty.static.charts.fixed_radar')
    if fixed_radar_spec is not None:
        from .fixed_radar import generate_radar_chart
        logger.info("Successfully imported radar chart fix")
        HAS_RADAR_FIX = True
    else:
        logger.warning("fixed_radar.py not found")
        HAS_RADAR_FIX = False
except Exception as e:
    logger.error(f"Error importing radar chart fix: {str(e)}")
    logger.error(traceback.format_exc())
    HAS_RADAR_FIX = False

# Try to apply enhanced_charts patch
try:
    # Check if enhanced_charts_fixed.py exists
    fixed_charts_spec = importlib.util.find_spec('deepbridge.templates.report_types.uncertainty.static.charts.enhanced_charts_fixed')
    if fixed_charts_spec is not None:
        from .enhanced_charts_fixed import apply_patch
        if apply_patch():
            logger.info("Successfully applied enhanced_charts patch")
        else:
            logger.warning("Failed to apply enhanced_charts patch")
    else:
        logger.warning("enhanced_charts_fixed.py not found")
except Exception as e:
    logger.error(f"Error applying enhanced charts fix: {str(e)}")
    logger.error(traceback.format_exc())


class UncertaintyChartGenerator:
    """
    Main class that provides access to all uncertainty chart generators.
    """
    
    def __init__(self, seaborn_chart_generator=None):
        """
        Initialize the uncertainty chart generator.
        
        Parameters:
        ----------
        seaborn_chart_generator : SeabornChartGenerator, optional
            Existing chart generator to use for rendering
        """
        self.seaborn_chart_generator = seaborn_chart_generator
        
        # Initialize individual chart generators
        self.coverage_vs_expected = CoverageVsExpectedChart(seaborn_chart_generator)
        self.width_vs_coverage = WidthVsCoverageChart(seaborn_chart_generator)
        self.uncertainty_metrics = UncertaintyMetricsChart(seaborn_chart_generator)
        self.feature_importance = FeatureImportanceChart(seaborn_chart_generator)
        self.model_comparison = ModelComparisonChart(seaborn_chart_generator)
        self.performance_gap_by_alpha = PerformanceGapByAlphaChart(seaborn_chart_generator)
        self.interval_boxplot = IntervalBoxplotChart()
        self.psi_analysis = PSIAnalysisChart()
        self.top_features_distribution = TopFeaturesDistributionChart()
        self.reliability_bandwidth = ReliabilityBandwidthChart(seaborn_chart_generator)
        self.reliability_performance = ReliabilityPerformanceChart(seaborn_chart_generator)
        self.reliability_regions = ReliabilityRegionsChart(seaborn_chart_generator)
    
    # Wrapper methods to maintain backward compatibility
    
    def generate_coverage_vs_expected(self, models_data, title="Coverage vs Expected Coverage", add_annotations=True):
        """Generate a chart comparing real coverage with expected coverage for different alpha values."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        
        # Check for required data structure
        if 'calibration_results' in models_data and isinstance(models_data['calibration_results'], dict):
            # Log data structure for debugging
            if 'alpha_values' in models_data['calibration_results']:
                logger.info(f"Alpha values for generate_coverage_vs_expected: {models_data['calibration_results']['alpha_values']}")
            if 'coverage_values' in models_data['calibration_results']:
                logger.info(f"Coverage values for generate_coverage_vs_expected: {models_data['calibration_results']['coverage_values']}")
            if 'expected_coverages' in models_data['calibration_results']:
                logger.info(f"Expected coverages for generate_coverage_vs_expected: {models_data['calibration_results']['expected_coverages']}")
            
            # Reformat data for the chart generator
            formatted_data = {
                "Primary Model": {
                    "alphas": models_data['calibration_results'].get('alpha_values', []),
                    "coverages": models_data['calibration_results'].get('coverage_values', []),
                    "expected_coverages": models_data['calibration_results'].get('expected_coverages', [])
                }
            }
            
            # Add alternative models if available
            if 'alternative_models' in models_data:
                for model_name, model_data in models_data['alternative_models'].items():
                    if 'calibration_results' in model_data:
                        formatted_data[model_name] = {
                            "alphas": model_data['calibration_results'].get('alpha_values', []),
                            "coverages": model_data['calibration_results'].get('coverage_values', []),
                            "expected_coverages": model_data['calibration_results'].get('expected_coverages', [])
                        }
                        
            # Check if we have valid data
            if formatted_data and any(all(key in model_data for key in ['alphas', 'coverages', 'expected_coverages']) 
                                     for model_data in formatted_data.values()):
                logger.info(f"Formatted data for coverage chart: {formatted_data}")
                return self.coverage_vs_expected.generate(formatted_data, title, add_annotations)
            else:
                logger.warning("No valid data found after formatting for coverage_vs_expected")
                return None
        else:
            logger.warning("Missing calibration_results in models_data")
            return None
    
    def generate_width_vs_coverage(self, models_data, title="Interval Width vs Coverage"):
        """Generate a chart showing the relationship between interval width and coverage."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        
        # Check for required data structure
        if 'calibration_results' in models_data and isinstance(models_data['calibration_results'], dict):
            # Log data structure for debugging
            if 'width_values' in models_data['calibration_results']:
                logger.info(f"Width values for generate_width_vs_coverage: {models_data['calibration_results']['width_values']}")
            if 'coverage_values' in models_data['calibration_results']:
                logger.info(f"Coverage values for generate_width_vs_coverage: {models_data['calibration_results']['coverage_values']}")
            
            # Reformat data for the chart generator
            formatted_data = {
                "Primary Model": {
                    "widths": models_data['calibration_results'].get('width_values', []),
                    "coverages": models_data['calibration_results'].get('coverage_values', [])
                }
            }
            
            # Add alternative models if available
            if 'alternative_models' in models_data:
                for model_name, model_data in models_data['alternative_models'].items():
                    if 'calibration_results' in model_data:
                        formatted_data[model_name] = {
                            "widths": model_data['calibration_results'].get('width_values', []),
                            "coverages": model_data['calibration_results'].get('coverage_values', [])
                        }
            
            # Check if we have valid data
            if formatted_data and any(all(key in model_data for key in ['widths', 'coverages']) 
                                    for model_data in formatted_data.values()):
                logger.info(f"Formatted data for width vs coverage chart: {formatted_data}")
                return self.width_vs_coverage.generate(formatted_data, title)
            else:
                logger.warning("No valid data found after formatting for width_vs_coverage")
                return None
        else:
            logger.warning("Missing calibration_results in models_data")
            return None
    
    def generate_uncertainty_metrics(self, models_data, title="Uncertainty Metrics Comparison"):
        """Generate a chart comparing different uncertainty metrics across models."""
        return self.uncertainty_metrics.generate(models_data, title)
    
    def generate_feature_importance(self, feature_importance_data, title="Feature Importance for Uncertainty"):
        """Generate a chart showing feature importance for uncertainty."""
        return self.feature_importance.generate(feature_importance_data, title)
    
    def generate_model_comparison(self, models_data, title="Model Comparison"):
        """Generate a chart comparing models based on uncertainty metrics."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        
        # Log detailed information about the models_data
        logger.info(f"generate_model_comparison called with: models_data={type(models_data)}")
        
        # Check if we have model data available
        if not models_data:
            logger.warning("No models_data available for model comparison")
            return None
            
        if isinstance(models_data, dict):
            logger.info(f"models_data keys: {list(models_data.keys())}")
        
        # Define all possible metric names to check
        metrics_to_check = [
            # Standard metric names
            'uncertainty_score', 'coverage', 'mean_width',
            
            # Alternate names that might be used
            'uncertainty_quality_score', 'avg_coverage', 'avg_width',
            'avg_normalized_width', 'avg_coverage_error'
        ]
        
        # Function to check if model data has any metrics
        def has_metrics(data):
            if not isinstance(data, dict):
                return False
            
            # Check direct metrics
            for metric in metrics_to_check:
                if metric in data:
                    return True
            
            # Check metrics dictionary if present
            if 'metrics' in data and isinstance(data['metrics'], dict):
                for metric in metrics_to_check:
                    if metric in data['metrics']:
                        return True
            
            return False
        
        # Check primary model metrics
        has_primary = has_metrics(models_data)
        
        # Check alternative models
        has_alternatives = False
        if 'alternative_models' in models_data and isinstance(models_data['alternative_models'], dict):
            for model_name, model_data in models_data['alternative_models'].items():
                if has_metrics(model_data):
                    has_alternatives = True
                    break
        
        if not (has_primary or has_alternatives):
            logger.warning("No model metrics available for comparison")
            return None
        
        # Function to get metric value with fallbacks
        def get_metric_value(data, primary_key, fallback_keys=None, default=0):
            if fallback_keys is None:
                fallback_keys = []
                
            # Try direct access first
            if primary_key in data:
                return data[primary_key]
            
            # Try fallback keys
            for key in fallback_keys:
                if key in data:
                    return data[key]
            
            # Check metrics dictionary
            if 'metrics' in data and isinstance(data['metrics'], dict):
                if primary_key in data['metrics']:
                    return data['metrics'][primary_key]
                
                # Try fallback keys in metrics
                for key in fallback_keys:
                    if key in data['metrics']:
                        return data['metrics'][key]
            
            # Return default if nothing found
            return default
            
        # Format data for chart generator
        formatted_data = {}
        
        # Add primary model if available
        if has_primary:
            primary_name = models_data.get('model_name', 'Primary Model')
            formatted_data[primary_name] = {
                'uncertainty_score': get_metric_value(
                    models_data, 'uncertainty_score', 
                    ['uncertainty_quality_score']),
                'coverage': get_metric_value(
                    models_data, 'coverage', 
                    ['avg_coverage']),
                'mean_width': get_metric_value(
                    models_data, 'mean_width', 
                    ['avg_width', 'avg_normalized_width'])
            }
            
            logger.info(f"Added primary model '{primary_name}' with metrics: {formatted_data[primary_name]}")
            
        # Add alternative models if available
        if has_alternatives:
            for model_name, model_data in models_data['alternative_models'].items():
                if has_metrics(model_data):
                    formatted_data[model_name] = {
                        'uncertainty_score': get_metric_value(
                            model_data, 'uncertainty_score', 
                            ['uncertainty_quality_score']),
                        'coverage': get_metric_value(
                            model_data, 'coverage', 
                            ['avg_coverage']),
                        'mean_width': get_metric_value(
                            model_data, 'mean_width', 
                            ['avg_width', 'avg_normalized_width'])
                    }
                    
                    logger.info(f"Added alternative model '{model_name}' with metrics: {formatted_data[model_name]}")
        
        # Check if we have valid data
        if formatted_data:
            logger.info(f"Formatted data for model comparison: {formatted_data}")
            # Use the standard metric names that the chart generator expects
            metrics = ['uncertainty_score', 'coverage', 'mean_width']
            try:
                return self.model_comparison.generate(formatted_data, metrics, title)
            except Exception as e:
                logger.error(f"Error generating model comparison chart: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        else:
            logger.warning("No valid metrics found for model comparison")
            return None
        
    def generate_performance_gap_by_alpha(self, models_data, title="Performance Gap by Alpha Level", add_annotations=True):
        """Generate a chart showing performance gaps across alpha levels for different models."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        logger.info(f"generate_performance_gap_by_alpha called with: models_data={type(models_data)}, title={title}, add_annotations={add_annotations}")

        # Log the models_data keys
        if isinstance(models_data, dict):
            logger.info(f"models_data keys: {list(models_data.keys())}")

            # Check for calibration_results
            if 'calibration_results' in models_data:
                logger.info(f"calibration_results keys: {list(models_data['calibration_results'].keys())}")

                # Ensure data types are correctly converted
                if isinstance(models_data['calibration_results'], dict):
                    # Convert any numpy arrays to lists for each key
                    for key in ['alpha_values', 'coverage_values', 'expected_coverages', 'width_values']:
                        if key in models_data['calibration_results']:
                            value = models_data['calibration_results'][key]
                            # Convert numpy arrays to lists
                            if hasattr(value, 'tolist') and callable(getattr(value, 'tolist')):
                                models_data['calibration_results'][key] = value.tolist()
                            # Ensure non-list iterables are converted to lists
                            elif not isinstance(value, list) and hasattr(value, '__iter__'):
                                try:
                                    models_data['calibration_results'][key] = list(value)
                                except Exception as e:
                                    logger.warning(f"Failed to convert {key} to list: {str(e)}")

            # Check for alpha_levels
            if 'alpha_levels' in models_data:
                logger.info(f"alpha_levels: {models_data['alpha_levels']}")
                # Convert alpha_levels to list if it's not already
                if hasattr(models_data['alpha_levels'], 'tolist') and callable(getattr(models_data['alpha_levels'], 'tolist')):
                    models_data['alpha_levels'] = models_data['alpha_levels'].tolist()
                elif not isinstance(models_data['alpha_levels'], list) and hasattr(models_data['alpha_levels'], '__iter__'):
                    try:
                        models_data['alpha_levels'] = list(models_data['alpha_levels'])
                    except Exception as e:
                        logger.warning(f"Failed to convert alpha_levels to list: {str(e)}")

        try:
            return self.performance_gap_by_alpha.generate(models_data, title, add_annotations)
        except Exception as e:
            logger.error(f"Error generating performance gap by alpha chart: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_interval_boxplot(self, models_data, title="Interval Width Distribution"):
        """Generate a boxplot chart showing interval width distributions."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        logger.info(f"generate_interval_boxplot called with: models_data={type(models_data)}")

        try:
            return self.interval_boxplot.generate(models_data, title=title)
        except Exception as e:
            logger.error(f"Error generating interval boxplot chart: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_psi_analysis(self, models_data, title="PSI Distribution Analysis", max_features=20):
        """Generate PSI analysis chart showing feature distribution shifts."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        logger.info(f"generate_psi_analysis called with: models_data={type(models_data)}")

        try:
            return self.psi_analysis.generate(models_data, title=title, max_features=max_features)
        except Exception as e:
            logger.error(f"Error generating PSI analysis chart: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_top_features_distribution(self, models_data, top_n=3, show_kde=True, bins=20):
        """Generate distribution comparison for top features by PSI."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        logger.info(f"generate_top_features_distribution called with: models_data={type(models_data)}, top_n={top_n}")

        try:
            return self.top_features_distribution.generate(
                models_data, top_n=top_n, show_kde=show_kde, bins=bins
            )
        except Exception as e:
            logger.error(f"Error generating top features distribution chart: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_reliability_bandwidth(self, models_data, n_bins=10, confidence=0.95,
                                      title="Reliability Bandwidth (Calibration)",
                                      show_confidence_bands=True, show_histogram=True):
        """Generate reliability bandwidth (calibration) chart."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        logger.info(f"generate_reliability_bandwidth called with: models_data={type(models_data)}")

        try:
            return self.reliability_bandwidth.generate(
                models_data, n_bins=n_bins, confidence=confidence,
                title=title, show_confidence_bands=show_confidence_bands,
                show_histogram=show_histogram
            )
        except Exception as e:
            logger.error(f"Error generating reliability bandwidth chart: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_reliability_performance(self, models_data, n_bins=10, metric='accuracy',
                                        title="Reliability Performance",
                                        show_sample_sizes=True, show_confidence_bars=True):
        """Generate reliability performance chart."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        logger.info(f"generate_reliability_performance called with: models_data={type(models_data)}")

        try:
            return self.reliability_performance.generate(
                models_data, n_bins=n_bins, metric=metric,
                title=title, show_sample_sizes=show_sample_sizes,
                show_confidence_bars=show_confidence_bars
            )
        except Exception as e:
            logger.error(f"Error generating reliability performance chart: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_reliability_regions(self, reliability_data, title="Feature Reliability Regions", show_top_n=3):
        """Generate reliability regions chart for top features."""
        import logging
        logger = logging.getLogger("deepbridge.reports")
        logger.info(f"generate_reliability_regions called with: reliability_data={type(reliability_data)}")

        try:
            return self.reliability_regions.generate(
                reliability_data, title=title, show_top_n=show_top_n
            )
        except Exception as e:
            logger.error(f"Error generating reliability regions chart: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None