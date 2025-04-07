"""
Standardized result objects for experiment test results.
These classes implement the interface defined in interfaces.py.
"""

import typing as t
from pathlib import Path
import json
from datetime import datetime

from deepbridge.core.experiment.interfaces import TestResult, ModelResult
from deepbridge.core.experiment.dependencies import check_dependencies

class BaseTestResult(TestResult):
    """Base implementation of the TestResult interface"""
    
    def __init__(self, name: str, results: dict, metadata: t.Optional[dict] = None):
        """
        Initialize with test results
        
        Args:
            name: Name of the test
            results: Raw results dictionary
            metadata: Additional metadata about the test
        """
        self._name = name
        self._results = results
        self._metadata = metadata or {}
        
    @property
    def name(self) -> str:
        """Get the name of the test"""
        return self._name
    
    @property
    def results(self) -> dict:
        """Get the raw results dictionary"""
        return self._results
    
    @property
    def metadata(self) -> dict:
        """Get the test metadata"""
        return self._metadata
    
    def to_html(self) -> str:
        """Convert results to HTML format - generic implementation"""
        # Basic HTML representation
        html = f"<h2>{self.name} Test Results</h2>"
        html += "<table border='1'><tr><th>Metric</th><th>Value</th></tr>"
        
        # Add flattened results as rows in table
        for key, value in self._flatten_dict(self.results).items():
            if not isinstance(value, dict):  # Skip nested dictionaries
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"
                
        html += "</table>"
        return html
    
    def save_report(self, path: t.Union[str, Path], name: t.Optional[str] = None) -> Path:
        """Save results to a report file"""
        path = Path(path)
        report_name = name or f"{self.name}_report.html"
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate full report path
        report_path = path / report_name if path.is_dir() else path
        
        # Generate HTML content
        html_content = self._generate_html_report()
        
        # Write to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return report_path
    
    def _generate_html_report(self) -> str:
        """Generate a complete HTML report"""
        # Basic HTML structure with styling
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>{self.name} Test Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>{self.name} Test Results</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section">
                {self.to_html()}
            </div>
            
            <div class="section">
                <h2>Test Metadata</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    {self._metadata_to_html()}
                </table>
            </div>
        </body>
        </html>
        """
        return html
    
    def _metadata_to_html(self) -> str:
        """Convert metadata to HTML table rows"""
        html = ""
        for key, value in self.metadata.items():
            html += f"<tr><td>{key}</td><td>{value}</td></tr>"
        return html
    
    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '_') -> dict:
        """Flatten a nested dictionary structure"""
        items = {}
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict) and len(v) < 10:  # Only flatten small dictionaries
                items.update(self._flatten_dict(v, new_key, sep))
            else:
                items[new_key] = v
        return items


class RobustnessResult(BaseTestResult):
    """Result object for robustness tests"""
    
    def __init__(self, results: dict, metadata: t.Optional[dict] = None):
        super().__init__("Robustness", results, metadata)
    
    def to_html(self) -> str:
        """Specialized HTML representation for robustness results"""
        html = f"<h2>Robustness Test Results</h2>"
        
        # Extract key robustness metrics
        primary_results = self.results.get('primary_model', self.results)
        
        # Base score and impacts
        base_score = primary_results.get('base_score', 'N/A')
        avg_overall_impact = primary_results.get('avg_overall_impact', 'N/A')
        
        html += f"<p><strong>Base Model Score:</strong> {base_score}</p>"
        html += f"<p><strong>Average Overall Impact:</strong> {avg_overall_impact}</p>"
        
        # Show perturbation methods if available
        if 'raw' in primary_results:
            html += "<h3>Raw Perturbation Results</h3>"
            html += "<table border='1'><tr><th>Level</th><th>Impact</th></tr>"
            
            for level, impact in primary_results.get('raw', {}).get('by_level', {}).items():
                html += f"<tr><td>{level}</td><td>{impact}</td></tr>"
                
            html += "</table>"
            
        if 'quantile' in primary_results:
            html += "<h3>Quantile Perturbation Results</h3>"
            html += "<table border='1'><tr><th>Level</th><th>Impact</th></tr>"
            
            for level, impact in primary_results.get('quantile', {}).get('by_level', {}).items():
                html += f"<tr><td>{level}</td><td>{impact}</td></tr>"
                
            html += "</table>"
        
        # Feature importance if available
        if 'feature_importance' in primary_results or 'feature_impact' in primary_results:
            feature_importance = primary_results.get('feature_importance', primary_results.get('feature_impact', {}))
            
            if feature_importance:
                html += "<h3>Feature Importance</h3>"
                html += "<table border='1'><tr><th>Feature</th><th>Importance</th></tr>"
                
                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                
                for feature, importance in sorted_features:
                    html += f"<tr><td>{feature}</td><td>{importance:.4f}</td></tr>"
                    
                html += "</table>"
        
        return html


class UncertaintyResult(BaseTestResult):
    """Result object for uncertainty tests"""
    
    def __init__(self, results: dict, metadata: t.Optional[dict] = None):
        super().__init__("Uncertainty", results, metadata)
    
    def to_html(self) -> str:
        """Specialized HTML representation for uncertainty results"""
        html = f"<h2>Uncertainty Quantification Results</h2>"
        
        # Extract key uncertainty metrics
        primary_results = self.results.get('primary_model', self.results)
        
        # Coverage stats
        if 'coverage_stats' in primary_results:
            coverage_stats = primary_results['coverage_stats']
            html += "<h3>Coverage Statistics</h3>"
            html += "<table border='1'><tr><th>Alpha</th><th>Expected Coverage</th><th>Actual Coverage</th><th>Avg Interval Width</th></tr>"
            
            for alpha, stats in coverage_stats.items():
                expected = 1 - float(alpha)
                actual = stats.get('coverage', 'N/A')
                width = stats.get('avg_width', 'N/A')
                html += f"<tr><td>{alpha}</td><td>{expected:.2f}</td><td>{actual:.4f}</td><td>{width:.4f}</td></tr>"
                
            html += "</table>"
            
        # Calibration metrics if available  
        if 'calibration_metrics' in primary_results:
            calibration = primary_results['calibration_metrics']
            html += "<h3>Calibration Metrics</h3>"
            html += "<table border='1'><tr><th>Metric</th><th>Value</th></tr>"
            
            for metric, value in calibration.items():
                html += f"<tr><td>{metric}</td><td>{value}</td></tr>"
                
            html += "</table>"
        
        return html


class ResilienceResult(BaseTestResult):
    """Result object for resilience tests"""
    
    def __init__(self, results: dict, metadata: t.Optional[dict] = None):
        super().__init__("Resilience", results, metadata)
    
    def to_html(self) -> str:
        """Specialized HTML representation for resilience results"""
        html = f"<h2>Resilience Test Results</h2>"
        
        # Extract key resilience metrics
        primary_results = self.results.get('primary_model', self.results)
        
        # Base score
        base_score = primary_results.get('base_score', 'N/A')
        html += f"<p><strong>Base Model Score:</strong> {base_score}</p>"
        
        # Drift results by type
        drift_types = ['covariate', 'label', 'concept']
        
        for drift_type in drift_types:
            if drift_type in primary_results:
                html += f"<h3>{drift_type.capitalize()} Drift Results</h3>"
                html += "<table border='1'><tr><th>Intensity</th><th>Score</th><th>Impact</th></tr>"
                
                drift_results = primary_results[drift_type]
                for intensity, result in drift_results.items():
                    score = result.get('score', 'N/A')
                    impact = result.get('impact', 'N/A')
                    html += f"<tr><td>{intensity}</td><td>{score}</td><td>{impact}</td></tr>"
                    
                html += "</table>"
                
        # Overall resilience index if available
        if 'resilience_index' in primary_results:
            html += f"<p><strong>Overall Resilience Index:</strong> {primary_results['resilience_index']}</p>"
            
        return html


class HyperparameterResult(BaseTestResult):
    """Result object for hyperparameter tests"""
    
    def __init__(self, results: dict, metadata: t.Optional[dict] = None):
        super().__init__("Hyperparameter", results, metadata)
    
    def to_html(self) -> str:
        """Specialized HTML representation for hyperparameter results"""
        html = f"<h2>Hyperparameter Importance Results</h2>"
        
        # Extract key hyperparameter metrics
        primary_results = self.results.get('primary_model', self.results)
        
        # Hyperparameter importance
        if 'sorted_importance' in primary_results:
            importance = primary_results['sorted_importance']
            html += "<h3>Hyperparameter Importance</h3>"
            html += "<table border='1'><tr><th>Hyperparameter</th><th>Importance</th></tr>"
            
            # Display sorted by importance
            for param, imp in importance.items():
                html += f"<tr><td>{param}</td><td>{imp:.4f}</td></tr>"
                
            html += "</table>"
            
        # Tuning order if available
        if 'tuning_order' in primary_results:
            tuning_order = primary_results['tuning_order']
            html += "<h3>Recommended Tuning Order</h3>"
            html += "<ol>"
            
            for param in tuning_order:
                html += f"<li>{param}</li>"
                
            html += "</ol>"
            
        # Best parameters if available
        if 'best_params' in primary_results:
            best_params = primary_results['best_params']
            html += "<h3>Best Parameters</h3>"
            html += "<table border='1'><tr><th>Parameter</th><th>Value</th></tr>"
            
            for param, value in best_params.items():
                html += f"<tr><td>{param}</td><td>{value}</td></tr>"
                
            html += "</table>"
            
        return html


class ExperimentResult:
    """
    Container for all test results from an experiment.
    Provides methods to manage results and generate combined reports.
    """
    
    def __init__(self, experiment_type: str, config: dict):
        """
        Initialize with experiment metadata
        
        Args:
            experiment_type: Type of experiment
            config: Experiment configuration
        """
        self.experiment_type = experiment_type
        self.config = config
        self.results = {}
        self.generation_time = datetime.now()
        
    def add_result(self, result: TestResult):
        """Add a test result to the experiment"""
        self.results[result.name.lower()] = result
        
    def get_result(self, name: str) -> t.Optional[TestResult]:
        """Get a specific test result by name"""
        return self.results.get(name.lower())
    
    def to_dict(self) -> dict:
        """Convert all results to a dictionary"""
        result_dict = {
            'experiment_type': self.experiment_type,
            'config': self.config,
            'generation_time': self.generation_time.strftime('%Y-%m-%d %H:%M:%S'),
            'tests_performed': list(self.results.keys())
        }
        
        # Add each test's results to the dictionary
        for name, result in self.results.items():
            result_dict[name] = self._clean_results_dict(result.results)
            
        return result_dict
    
    def _clean_results_dict(self, results_dict: dict) -> dict:
        """Clean the results dictionary by removing redundant information"""
        # Create a deep copy to avoid modifying the original
        import copy
        cleaned = copy.deepcopy(results_dict)
        
        # Handle primary model cleanup
        if 'primary_model' in cleaned:
            primary = cleaned['primary_model']
            
            # Remove redundant metrics entries
            if 'metrics' in primary and 'base_score' in primary['metrics']:
                # If base_score is duplicated in metrics, remove it
                if primary.get('base_score') == primary['metrics'].get('base_score'):
                    del primary['metrics']['base_score']
            
            # Remove metric name if metrics are present (since it's redundant)
            if 'metric' in primary and 'metrics' in primary:
                del primary['metric']
        
        # Handle alternative models cleanup
        if 'alternative_models' in cleaned:
            for model_name, model_data in cleaned['alternative_models'].items():
                # Remove redundant metrics entries
                if 'metrics' in model_data and 'base_score' in model_data['metrics']:
                    # If base_score is duplicated in metrics, remove it
                    if model_data.get('base_score') == model_data['metrics'].get('base_score'):
                        del model_data['metrics']['base_score']
                
                # Remove metric name if metrics are present
                if 'metric' in model_data and 'metrics' in model_data:
                    del model_data['metric']
        
        return cleaned
    
    def to_html(self) -> str:
        """Generate an HTML report with all results"""
        # Basic HTML structure
        html = f"""<!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #eee; border-radius: 5px; }}
                table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #f2f2f2; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <h1>Experiment Results</h1>
            <p>Generated on: {self.generation_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="section summary">
                <h2>Experiment Summary</h2>
                <table>
                    <tr><th>Property</th><th>Value</th></tr>
                    <tr><td>Experiment Type</td><td>{self.experiment_type}</td></tr>
                    <tr><td>Tests Performed</td><td>{', '.join(self.results.keys())}</td></tr>
                </table>
            </div>
        """
        
        # Add each test's HTML representation
        for name, result in self.results.items():
            html += f"<div class='section'>{result.to_html()}</div>"
            
        html += "</body></html>"
        return html
    
    def save_report(self, report_type: t.Optional[str] = None, path: t.Union[str, Path] = None, name: str = "experiment_report.html") -> Path:
        """
        Save a combined HTML report for all test results
        
        Args:
            report_type: Type of report to generate ('robustness', 'uncertainty', etc.)
                If specified, generates a specialized report using the corresponding generator
            path: Directory path or full file path
            name: Filename (used only if path is a directory)
            
        Returns:
            Path to the saved report
        """
        if report_type is not None:
            # Handle specialized report types
            report_type = report_type.lower()
            
            # Default path if none provided
            if path is None:
                path = f"{report_type}_report.html"
                
            # Try to generate specialized report
            try:
                if report_type == 'robustness' and 'robustness' in self.results:
                    from deepbridge.reporting.plots.robustness.robustness_report_generator import generate_robustness_report
                    return generate_robustness_report(
                        self.results['robustness'].results,  # Use the raw results dictionary
                        path,
                        model_name="Primary Model",
                        experiment_info=self.to_dict()  # Use the full experiment dict
                    )
                elif report_type == 'uncertainty' and 'uncertainty' in self.results:
                    from deepbridge.reporting.plots.uncertainty.uncertainty_report_generator import generate_uncertainty_report
                    return generate_uncertainty_report(
                        self.results['uncertainty'].results,  # Use the raw results dictionary
                        path,
                        model_name="Primary Model",
                        experiment_info=self.to_dict()  # Use the full experiment dict
                    )
            except Exception as e:
                import traceback
                print(f"Error generating specialized {report_type} report: {str(e)}")
                print(traceback.format_exc())
                # Continue to default report generation
        
        # Default report generation
        path = Path(path) if path is not None else Path(name)
        
        # Ensure directory exists
        if path.is_dir():
            report_path = path / name
            path.mkdir(parents=True, exist_ok=True)
        else:
            report_path = path
            path.parent.mkdir(parents=True, exist_ok=True)
            
        # Generate HTML content
        html_content = self.to_html()
        
        # Write to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return report_path
    
    @classmethod
    def from_dict(cls, results_dict: dict) -> 'ExperimentResult':
        """
        Create an ExperimentResult instance from a dictionary
        
        Args:
            results_dict: Dictionary containing test results
            
        Returns:
            ExperimentResult instance
        """
        experiment_type = results_dict.get('experiment_type', 'binary_classification')
        config = results_dict.get('config', {})
        
        # Create instance
        instance = cls(experiment_type, config)
        
        # Add test results
        if 'robustness' in results_dict:
            instance.add_result(RobustnessResult(results_dict['robustness']))
            
        if 'uncertainty' in results_dict:
            instance.add_result(UncertaintyResult(results_dict['uncertainty']))
            
        if 'resilience' in results_dict:
            instance.add_result(ResilienceResult(results_dict['resilience']))
            
        if 'hyperparameters' in results_dict:
            instance.add_result(HyperparameterResult(results_dict['hyperparameters']))
            
        return instance


def create_test_result(test_type: str, results: dict, metadata: t.Optional[dict] = None) -> TestResult:
    """
    Factory function to create the appropriate test result object
    
    Args:
        test_type: Type of test ('robustness', 'uncertainty', etc.)
        results: Raw test results
        metadata: Additional test metadata
        
    Returns:
        TestResult instance
    """
    test_type = test_type.lower()
    
    if test_type == 'robustness':
        return RobustnessResult(results, metadata)
    elif test_type == 'uncertainty':
        return UncertaintyResult(results, metadata)
    elif test_type == 'resilience':
        return ResilienceResult(results, metadata)
    elif test_type == 'hyperparameters' or test_type == 'hyperparameter':
        return HyperparameterResult(results, metadata)
    else:
        return BaseTestResult(test_type.capitalize(), results, metadata)


def wrap_results(results_dict: dict) -> ExperimentResult:
    """
    Wrap a dictionary of results in an ExperimentResult object
    
    Args:
        results_dict: Dictionary with test results
        
    Returns:
        ExperimentResult instance
    """
    return ExperimentResult.from_dict(results_dict)

# Import model results
try:
    from deepbridge.core.experiment.model_result import (
        BaseModelResult, ClassificationModelResult, RegressionModelResult, 
        create_model_result
    )
except ImportError:
    # Provide simplified implementations if model_result.py is not available
    class BaseModelResult:
        """Simplified model result implementation"""
        def __init__(self, model_name, model_type, metrics, **kwargs):
            self.model_name = model_name
            self.model_type = model_type
            self.metrics = metrics
            
    def create_model_result(model_name, model_type, metrics, **kwargs):
        """Simplified factory function"""
        return BaseModelResult(model_name, model_type, metrics, **kwargs)