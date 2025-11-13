"""
Standardized result objects for experiment test results.
These classes implement the interface defined in interfaces.py.
Reporting functionality has been removed in this version.
"""

import typing as t
import copy
import os
from pathlib import Path
import json
from datetime import datetime

# Mover importações para o topo
from deepbridge.core.experiment.interfaces import TestResult, ModelResult
from deepbridge.core.experiment.dependencies import check_dependencies

# Importar o gerenciador de relatórios aqui em vez de dentro de um método
from deepbridge.core.experiment.report.report_manager import ReportManager
# Se a importação falhar, isso é um erro crítico, já que estamos migrando para a nova estrutura

# Definir exceções específicas em vez de usar ValueError genérico
class TestResultNotFoundError(Exception):
    """Erro lançado quando um resultado de teste não é encontrado."""
    pass

class ReportGenerationError(Exception):
    """Erro lançado quando a geração de relatório falha."""
    pass


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
    
    def to_dict(self) -> dict:
        """Convert test result to a dictionary format"""
        # Use OrderedDict to maintain key order
        from collections import OrderedDict
        result_dict = OrderedDict()
        
        # Add initial_results first if it exists in the results
        if 'initial_results' in self._results:
            result_dict['initial_results'] = self._results['initial_results']
            
        # Add the rest of the content
        result_dict.update({
            'name': self.name,
            'results': {k: v for k, v in self.results.items() if k != 'initial_results'},
            'metadata': self.metadata
        })
        
        return result_dict
    
    def clean_results_dict(self) -> dict:
        """
        Clean the results dictionary by removing redundant information.
        Cada classe filha pode sobrescrever este método para limpeza específica.
        """
        # Use OrderedDict to maintain key order for consistent serialization
        from collections import OrderedDict
        cleaned = OrderedDict()
        
        # Add initial_results first if it exists
        if 'initial_results' in self._results:
            cleaned['initial_results'] = copy.deepcopy(self._results['initial_results'])
            
        # Add all other results
        for key, value in self._results.items():
            if key != 'initial_results':
                cleaned[key] = copy.deepcopy(value)
                
        return cleaned


class RobustnessResult(BaseTestResult):
    """Result object for robustness tests"""

    def __init__(self, results: dict, metadata: t.Optional[dict] = None):
        super().__init__("Robustness", results, metadata)

    def save_html(self, file_path: str, model_name: str = "Model", report_type: str = "interactive") -> str:
        """
        Generate and save an HTML report for robustness analysis.

        Args:
            file_path: Path where the HTML report will be saved
            model_name: Name of the model for display in the report
            report_type: Type of report to generate ('interactive' with Plotly or 'static' with Matplotlib)

        Returns:
            Path to the generated report file

        Example:
            >>> robustness_result = experiment.run_test('robustness')
            >>> robustness_result.save_html('robustness_interactive.html', 'My Model', report_type='interactive')
            >>> robustness_result.save_html('robustness_static.html', 'My Model', report_type='static')
        """
        from deepbridge.core.experiment.report.report_manager import ReportManager

        # Create report manager
        report_manager = ReportManager()

        # Generate HTML report
        report_path = report_manager.generate_report(
            test_type='robustness',
            results=self._results,
            file_path=file_path,
            model_name=model_name,
            report_type=report_type
        )

        return report_path

    def clean_results_dict(self) -> dict:
        """Implement specific cleaning for robustness results"""
        cleaned = super().clean_results_dict()
        
        # Limpeza específica para resultados de robustez
        if 'primary_model' in cleaned:
            self._clean_model_data(cleaned['primary_model'])
            
        # Limpeza de modelos alternativos
        if 'alternative_models' in cleaned:
            for model_name, model_data in cleaned['alternative_models'].items():
                self._clean_model_data(model_data)
                
        return cleaned
    
    def _clean_model_data(self, model_data: dict) -> None:
        """
        Helper method para limpar dados de um modelo
        
        Args:
            model_data: Dictionary containing model data to clean
        """
        # Remove redundant metrics entries
        if 'metrics' in model_data and 'base_score' in model_data['metrics']:
            # If base_score is duplicated in metrics, remove it
            if model_data.get('base_score') == model_data['metrics'].get('base_score'):
                del model_data['metrics']['base_score']
        
        # Remove metric name if metrics are present
        if 'metric' in model_data and 'metrics' in model_data:
            del model_data['metric']


class UncertaintyResult(BaseTestResult):
    """Result object for uncertainty tests"""

    def __init__(self, results: dict, metadata: t.Optional[dict] = None):
        super().__init__("Uncertainty", results, metadata)

    def save_html(self, file_path: str, model_name: str = "Model", report_type: str = "interactive") -> str:
        """
        Generate and save an HTML report for uncertainty analysis.

        Args:
            file_path: Path where the HTML report will be saved
            model_name: Name of the model for display in the report
            report_type: Type of report to generate ('interactive' with Plotly or 'static' with Matplotlib)

        Returns:
            Path to the generated report file

        Example:
            >>> uncertainty_result = experiment.run_test('uncertainty')
            >>> uncertainty_result.save_html('uncertainty_interactive.html', 'My Model', report_type='interactive')
            >>> uncertainty_result.save_html('uncertainty_static.html', 'My Model', report_type='static')
        """
        from deepbridge.core.experiment.report.report_manager import ReportManager

        # Create report manager
        report_manager = ReportManager()

        # Generate HTML report
        report_path = report_manager.generate_report(
            test_type='uncertainty',
            results=self._results,
            file_path=file_path,
            model_name=model_name,
            report_type=report_type
        )

        return report_path


class ResilienceResult(BaseTestResult):
    """Result object for resilience tests"""

    def __init__(self, results: dict, metadata: t.Optional[dict] = None):
        super().__init__("Resilience", results, metadata)

    def save_html(self, file_path: str, model_name: str = "Model", report_type: str = "interactive") -> str:
        """
        Generate and save an HTML report for resilience analysis.

        Args:
            file_path: Path where the HTML report will be saved
            model_name: Name of the model for display in the report
            report_type: Type of report to generate ('interactive' with Plotly or 'static' with Matplotlib)

        Returns:
            Path to the generated report file

        Example:
            >>> resilience_result = experiment.run_test('resilience')
            >>> resilience_result.save_html('resilience_interactive.html', 'My Model', report_type='interactive')
            >>> resilience_result.save_html('resilience_static.html', 'My Model', report_type='static')
        """
        from deepbridge.core.experiment.report.report_manager import ReportManager

        # Create report manager
        report_manager = ReportManager()

        # Generate HTML report
        report_path = report_manager.generate_report(
            test_type='resilience',
            results=self._results,
            file_path=file_path,
            model_name=model_name,
            report_type=report_type
        )

        return report_path


class HyperparameterResult(BaseTestResult):
    """Result object for hyperparameter tests"""

    def __init__(self, results: dict, metadata: t.Optional[dict] = None):
        super().__init__("Hyperparameter", results, metadata)


class FairnessResult(BaseTestResult):
    """Result object for fairness tests"""

    def __init__(self, results: dict, metadata: t.Optional[dict] = None):
        super().__init__("Fairness", results, metadata)

    @property
    def overall_fairness_score(self) -> float:
        """Get overall fairness score (0-1, higher is better)"""
        return self._results.get('overall_fairness_score', 0.0)

    @property
    def critical_issues(self) -> list:
        """Get list of critical fairness issues"""
        return self._results.get('critical_issues', [])

    @property
    def warnings(self) -> list:
        """Get list of fairness warnings"""
        return self._results.get('warnings', [])

    @property
    def protected_attributes(self) -> list:
        """Get list of protected attributes tested"""
        return self._results.get('protected_attributes', [])

    def save_html(self, file_path: str, model_name: str = "Model", report_type: str = "interactive") -> str:
        """
        Generate and save an HTML report for fairness analysis.

        Args:
            file_path: Path where the HTML report will be saved
            model_name: Name of the model for display in the report
            report_type: Type of report to generate ('interactive' or 'static')

        Returns:
            Path to the generated report file

        Example:
            >>> fairness_result = experiment.run_fairness_tests(config='full')
            >>> fairness_result.save_html('fairness_report.html', model_name='Credit Model')
        """
        from deepbridge.core.experiment.report.report_manager import ReportManager

        # Create report manager
        report_manager = ReportManager()

        # Generate HTML report
        report_path = report_manager.generate_report(
            test_type='fairness',
            results=self._results,
            file_path=file_path,
            model_name=model_name,
            report_type=report_type
        )

        return report_path


class ExperimentResult:
    """
    Container for all test results from an experiment.
    Includes HTML report generation functionality.
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
        self.initial_results = {}  # Storage for initial results
        self.generation_time = datetime.now()
        
    def add_result(self, result: TestResult):
        """Add a test result to the experiment"""
        self.results[result.name.lower()] = result
        
    def get_result(self, name: str) -> t.Optional[TestResult]:
        """Get a specific test result by name"""
        return self.results.get(name.lower())
        
    def save_html(self, test_type: str, file_path: str, model_name: str = "Model", report_type: str = "static", save_chart: bool = False) -> str:
        """
        Generate and save an HTML report for the specified test.

        Args:
            test_type: Type of test ('robustness', 'uncertainty', 'resilience', 'hyperparameter')
            file_path: Path where the HTML report will be saved (relative or absolute)
            model_name: Name of the model for display in the report
            report_type: Type of report to generate ('interactive' or 'static')
            save_chart: Whether to save charts as separate PNG files (default: False)

        Returns:
            Path to the generated report file

        Raises:
            TestResultNotFoundError: If test results not found
            ReportGenerationError: If report generation fails
        """
        # Convert test_type to lowercase for consistency
        test_type = test_type.lower()

        # Check if we have results for this test type
        # Handle the case where hyperparameters is plural but the key is singular
        lookup_key = test_type
        if test_type == "hyperparameters":
            lookup_key = "hyperparameter"

        result = self.results.get(lookup_key)
        if not result:
            raise TestResultNotFoundError(f"No {test_type} test results found. Run the test first.")

        # Usar o gerenciador de relatórios do módulo experiment
        from deepbridge.core.experiment import report_manager

        # Create a complete structure for report generation
        report_data = {}

        # For robustness tests, we need a specific structure with primary_model
        if test_type == 'robustness':
            # Get the results dictionary, maintaining the full structure
            if hasattr(result, 'to_dict'):
                test_result = result.to_dict()['results']
            elif hasattr(result, 'results'):
                test_result = result.results
            else:
                test_result = result  # If result is already a dict

            # Check if we have primary_model directly or nested under 'results'
            if 'primary_model' in test_result:
                # Direct structure - use as is
                report_data = copy.deepcopy(test_result)
                # Ensure advanced tests are copied
                if 'weakspot_analysis' in test_result:
                    report_data['weakspot_analysis'] = test_result['weakspot_analysis']
                if 'overfitting_analysis' in test_result:
                    report_data['overfitting_analysis'] = test_result['overfitting_analysis']
            elif 'results' in test_result and 'primary_model' in test_result['results']:
                # Nested structure - extract and use the primary_model data
                report_data = copy.deepcopy(test_result['results'])
                # Ensure advanced tests are copied from parent level if available
                if 'weakspot_analysis' in test_result:
                    report_data['weakspot_analysis'] = test_result['weakspot_analysis']
                if 'overfitting_analysis' in test_result:
                    report_data['overfitting_analysis'] = test_result['overfitting_analysis']
            else:
                # Create standard structure with minimal data
                report_data = {
                    'primary_model': {
                        'raw': test_result.get('raw', {}),
                        'quantile': test_result.get('quantile', {}),
                        'base_score': test_result.get('base_score', 0),
                        'metrics': test_result.get('metrics', {}),
                        'avg_raw_impact': test_result.get('avg_raw_impact', 0),
                        'avg_quantile_impact': test_result.get('avg_quantile_impact', 0),
                        'avg_overall_impact': test_result.get('avg_overall_impact', 0),
                        'robustness_score': 1.0 - test_result.get('avg_overall_impact', 0),
                        'feature_importance': test_result.get('feature_importance', {}),
                        'model_feature_importance': test_result.get('model_feature_importance', {})
                    }
                }

                # Add feature subset if available
                if 'feature_subset' in test_result:
                    report_data['feature_subset'] = test_result['feature_subset']

                # Add advanced robustness tests if available (WeakSpot and Overfitting)
                if 'weakspot_analysis' in test_result:
                    report_data['weakspot_analysis'] = test_result['weakspot_analysis']
                if 'overfitting_analysis' in test_result:
                    report_data['overfitting_analysis'] = test_result['overfitting_analysis']
        else:
            # For other test types, use the standard approach
            if hasattr(result, 'to_dict'):
                report_data = result.to_dict()['results']
            elif hasattr(result, 'results'):
                report_data = result.results
            else:
                report_data = result  # If result is already a dict

        # Add initial_results if available
        # This is stored in self.results during experiment execution
        if 'initial_results' in self.results:
            initial_results = self.results['initial_results']
            report_data['initial_results'] = initial_results

            # Add initial_model_evaluation from initial_results
            # This is critical for resilience/robustness reports to have feature_importance data
            if isinstance(initial_results, dict):
                # Structure from Experiment: initial_results has 'models', 'config', 'test_configs'
                report_data['initial_model_evaluation'] = initial_results

        # Add experiment config if not present
        if 'config' not in report_data:
            report_data['config'] = self.config

        # Add experiment type
        report_data['experiment_type'] = self.experiment_type

        # Add model_type directly - using the value from the primary model if available
        if 'primary_model' in report_data and 'model_type' in report_data['primary_model']:
            report_data['model_type'] = report_data['primary_model']['model_type']

        # Ensure file_path is absolute
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        # Generate the report
        try:
            report_path = report_manager.generate_report(
                test_type=test_type,
                results=report_data,
                file_path=file_path,
                model_name=model_name,
                report_type=report_type,
                save_chart=save_chart
            )
            return report_path
        except NotImplementedError as e:
            raise ReportGenerationError(f"HTML report generation for {test_type} tests is not implemented: {str(e)}")
        except Exception as e:
            raise ReportGenerationError(f"Failed to generate HTML report: {str(e)}")

    def save_json(self, test_type: str, file_path: str, include_summary: bool = True, compact: bool = False) -> str:
        """
        Save test results to a JSON file for AI analysis.

        Args:
            test_type: Type of test results to save ('robustness', 'uncertainty', etc.)
            file_path: Path where to save the JSON file
            include_summary: Whether to include a summary section with key findings (default: True)
            compact: If True, removes large arrays and keeps only essential metrics for AI validation (default: False)

        Returns:
            Path to the saved JSON file

        Raises:
            TestResultNotFoundError: If test results not found
        """
        import numpy as np

        # Convert test_type to lowercase for consistency
        test_type = test_type.lower()

        # Check if we have results for this test type
        lookup_key = test_type
        if test_type == "hyperparameters":
            lookup_key = "hyperparameter"

        result = self.results.get(lookup_key)
        if not result:
            raise TestResultNotFoundError(f"No {test_type} test results found. Run the test first.")

        # Get the results dictionary
        if hasattr(result, 'clean_results_dict'):
            test_data = result.clean_results_dict()
        elif hasattr(result, 'results'):
            test_data = result.results
        else:
            test_data = result

        # Apply compact mode if requested
        if compact:
            if test_type == 'uncertainty':
                test_data = self._compact_uncertainty_data(test_data)
            elif test_type == 'robustness':
                test_data = self._compact_robustness_data(test_data)
            elif test_type == 'resilience':
                test_data = self._compact_resilience_data(test_data)
            # Add more compact methods for other test types as needed

        # Create the JSON structure
        json_data = {
            "experiment_info": {
                "test_type": test_type,
                "experiment_type": self.experiment_type,
                "generation_time": self.generation_time.strftime('%Y-%m-%d %H:%M:%S'),
                "config": self.config
            },
            "test_results": test_data
        }

        # Add initial model evaluation if available (compact version if requested)
        if 'initial_results' in self.results:
            if compact:
                # Only include essential model metrics
                initial_compact = self._compact_initial_results(self.results['initial_results'])
                json_data["initial_model_evaluation"] = initial_compact
            else:
                json_data["initial_model_evaluation"] = self.results['initial_results']

        # Add summary for robustness tests
        if include_summary and test_type == 'robustness':
            summary = self._generate_robustness_summary(test_data)
            json_data["summary"] = summary

        # Add summary for uncertainty tests
        if include_summary and test_type == 'uncertainty':
            summary = self._generate_uncertainty_summary(test_data)
            json_data["summary"] = summary

        # Function to convert numpy types to Python types
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj

        # Convert all numpy types before serialization
        json_data = convert_numpy_types(json_data)

        # Ensure file_path is absolute
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save to JSON file with appropriate formatting
        with open(file_path, 'w', encoding='utf-8') as f:
            if compact:
                # Compact format: minimal whitespace, reduced indentation
                json.dump(json_data, f, indent=None, separators=(',', ':'), ensure_ascii=False)
            else:
                # Standard format: readable with indentation
                json.dump(json_data, f, indent=2, ensure_ascii=False)

        return file_path

    def _generate_robustness_summary(self, test_data: dict) -> dict:
        """
        Generate a summary of key findings from robustness test results.

        Args:
            test_data: Dictionary containing robustness test results

        Returns:
            Summary dictionary with key findings
        """
        summary = {
            "key_findings": [],
            "model_performance": {},
            "feature_impacts": {},
            "recommendations": []
        }

        # Analyze primary model if available
        if 'primary_model' in test_data:
            primary = test_data['primary_model']

            # Model robustness score
            robustness_score = primary.get('robustness_score', 1.0 - primary.get('avg_overall_impact', 0))
            summary["model_performance"]["robustness_score"] = round(robustness_score, 4)

            # Key metrics
            if 'metrics' in primary:
                summary["model_performance"]["metrics"] = primary['metrics']

            # Average impacts
            summary["model_performance"]["average_impacts"] = {
                "raw_perturbation": round(primary.get('avg_raw_impact', 0), 4),
                "quantile_perturbation": round(primary.get('avg_quantile_impact', 0), 4),
                "overall": round(primary.get('avg_overall_impact', 0), 4)
            }

            # Top impacted features
            if 'feature_importance' in primary:
                features = primary['feature_importance']
                sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)
                summary["feature_impacts"]["most_sensitive_features"] = [
                    {"feature": name, "impact": round(impact, 4)}
                    for name, impact in sorted_features[:10]
                ]
                summary["feature_impacts"]["least_sensitive_features"] = [
                    {"feature": name, "impact": round(impact, 4)}
                    for name, impact in sorted_features[-5:]
                ]

            # Generate key findings
            if robustness_score >= 0.9:
                summary["key_findings"].append("Model shows excellent robustness (score >= 0.9)")
            elif robustness_score >= 0.8:
                summary["key_findings"].append("Model shows good robustness (score >= 0.8)")
            elif robustness_score >= 0.7:
                summary["key_findings"].append("Model shows moderate robustness (score >= 0.7)")
            else:
                summary["key_findings"].append(f"Model shows low robustness (score: {robustness_score:.4f})")

            # Add findings about feature sensitivity
            if 'feature_importance' in primary and len(features) > 0:
                high_impact_features = [k for k, v in features.items() if v > 0.1]
                if high_impact_features:
                    summary["key_findings"].append(f"Found {len(high_impact_features)} highly sensitive features (impact > 0.1)")

            # Recommendations based on results
            if robustness_score < 0.8:
                summary["recommendations"].append("Consider model regularization to improve robustness")

            if len(summary["feature_impacts"].get("most_sensitive_features", [])) > 0:
                top_feature = summary["feature_impacts"]["most_sensitive_features"][0]
                if top_feature["impact"] > 0.2:
                    summary["recommendations"].append(f"Feature '{top_feature['feature']}' is highly sensitive - consider feature engineering or validation")

        # Analyze alternative models if available
        if 'alternative_models' in test_data:
            alt_models = {}
            for model_name, model_data in test_data['alternative_models'].items():
                if 'metrics' in model_data:
                    alt_models[model_name] = {
                        "metrics": model_data['metrics'],
                        "robustness_score": round(1.0 - model_data.get('avg_overall_impact', 0), 4)
                    }

            if alt_models:
                summary["alternative_models_comparison"] = alt_models

                # Find best alternative model
                best_alt = max(alt_models.items(), key=lambda x: x[1].get('robustness_score', 0))
                if best_alt[1]['robustness_score'] > summary["model_performance"]["robustness_score"]:
                    summary["key_findings"].append(f"Alternative model '{best_alt[0]}' shows better robustness")

        return summary

    def _generate_uncertainty_summary(self, test_data: dict) -> dict:
        """
        Generate a summary of key findings from uncertainty test results.

        Args:
            test_data: Dictionary containing uncertainty test results

        Returns:
            Summary dictionary with key findings
        """
        summary = {
            "key_findings": [],
            "model_performance": {},
            "calibration_quality": {},
            "recommendations": []
        }

        # Analyze primary model if available
        if 'primary_model' in test_data:
            primary = test_data['primary_model']

            # Get CRQR data
            crqr = primary.get('crqr', {})
            by_alpha = crqr.get('by_alpha', {})

            # Calculate overall statistics
            if by_alpha:
                coverages = []
                calibration_errors = []
                widths = []

                for alpha_key, alpha_data in by_alpha.items():
                    overall = alpha_data.get('overall_result', {})
                    alpha_val = float(alpha_key)
                    target_coverage = 1 - alpha_val
                    actual_coverage = overall.get('coverage', 0)
                    mean_width = overall.get('mean_width', 0)

                    coverages.append(actual_coverage)
                    calibration_errors.append(abs(target_coverage - actual_coverage))
                    widths.append(mean_width)

                avg_coverage = sum(coverages) / len(coverages) if coverages else 0
                avg_calibration_error = sum(calibration_errors) / len(calibration_errors) if calibration_errors else 0
                avg_width = sum(widths) / len(widths) if widths else 0

                # Model performance metrics
                summary["model_performance"]["average_coverage"] = round(avg_coverage, 4)
                summary["model_performance"]["average_calibration_error"] = round(avg_calibration_error, 4)
                summary["model_performance"]["average_interval_width"] = round(avg_width, 4)
                summary["model_performance"]["uncertainty_score"] = round(primary.get('uncertainty_quality_score', 0), 4)

                # Base model metrics
                if 'metrics' in primary:
                    metrics = primary['metrics']
                    summary["model_performance"]["base_metrics"] = {
                        k: round(float(v), 4) for k, v in metrics.items()
                        if isinstance(v, (int, float))
                    }

                # Calibration quality assessment
                if avg_calibration_error < 0.05:
                    summary["calibration_quality"]["status"] = "EXCELLENT"
                    summary["calibration_quality"]["description"] = "Calibration error < 0.05"
                elif avg_calibration_error < 0.10:
                    summary["calibration_quality"]["status"] = "GOOD"
                    summary["calibration_quality"]["description"] = "Calibration error < 0.10"
                else:
                    summary["calibration_quality"]["status"] = "NEEDS_IMPROVEMENT"
                    summary["calibration_quality"]["description"] = f"Calibration error: {avg_calibration_error:.4f}"

                # Interval width assessment
                if avg_width < 0.5:
                    summary["calibration_quality"]["interval_width"] = "NARROW"
                    summary["calibration_quality"]["width_description"] = "High confidence predictions"
                elif avg_width < 1.0:
                    summary["calibration_quality"]["interval_width"] = "MODERATE"
                    summary["calibration_quality"]["width_description"] = "Moderate uncertainty"
                else:
                    summary["calibration_quality"]["interval_width"] = "WIDE"
                    summary["calibration_quality"]["width_description"] = "High uncertainty detected"

                # Generate key findings
                summary["key_findings"].append(
                    f"Average coverage: {avg_coverage*100:.1f}% (calibration error: {avg_calibration_error:.4f})"
                )

                if avg_calibration_error >= 0.10:
                    summary["key_findings"].append("Calibration needs improvement (error ≥ 0.10)")

                if avg_width > 0.5:
                    summary["key_findings"].append(f"High uncertainty detected (avg width: {avg_width:.4f})")

                if avg_coverage < 0.90:
                    summary["key_findings"].append("Coverage below 90% - model may be overconfident")

                # Recommendations
                if avg_calibration_error >= 0.10:
                    summary["recommendations"].append("Apply calibration methods (Platt scaling, isotonic regression)")

                if avg_width > 0.5:
                    summary["recommendations"].append("Collect more training data to reduce prediction variance")
                    summary["recommendations"].append("Consider ensemble methods")

                if avg_coverage < 0.90:
                    summary["recommendations"].append("Recalibrate or increase interval width")

                # Per-alpha analysis
                summary["per_alpha_analysis"] = []
                for alpha_key in sorted(by_alpha.keys(), key=lambda x: float(x)):
                    alpha_data = by_alpha[alpha_key]
                    overall = alpha_data.get('overall_result', {})
                    alpha_val = float(alpha_key)
                    target_cov = 1 - alpha_val
                    actual_cov = overall.get('coverage', 0)
                    mean_w = overall.get('mean_width', 0)
                    cal_err = abs(target_cov - actual_cov)

                    summary["per_alpha_analysis"].append({
                        "alpha": alpha_val,
                        "target_coverage": round(target_cov, 4),
                        "actual_coverage": round(actual_cov, 4),
                        "calibration_error": round(cal_err, 4),
                        "mean_width": round(mean_w, 4)
                    })

        return summary

    def _compact_uncertainty_data(self, test_data: dict) -> dict:
        """
        Create a compact version of uncertainty test data for AI validation.
        Removes large arrays and keeps only essential metrics.

        Args:
            test_data: Full uncertainty test data

        Returns:
            Compacted dictionary with only essential information
        """
        compact = {}

        if 'primary_model' in test_data:
            primary = test_data['primary_model']
            compact_primary = {}

            # Keep essential metrics
            if 'metrics' in primary:
                compact_primary['metrics'] = primary['metrics']

            # Keep uncertainty quality score
            if 'uncertainty_quality_score' in primary:
                compact_primary['uncertainty_quality_score'] = primary['uncertainty_quality_score']

            # Keep feature importance (top 10 only)
            if 'feature_importance' in primary:
                feat_imp = primary['feature_importance']
                # Sort by absolute importance and keep top 10
                sorted_features = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                compact_primary['feature_importance_top10'] = dict(sorted_features)

            # Keep CRQR results but only summary per alpha (not raw data)
            if 'crqr' in primary:
                crqr = primary['crqr']
                compact_crqr = {}

                if 'by_alpha' in crqr:
                    compact_by_alpha = {}
                    for alpha_key, alpha_data in crqr['by_alpha'].items():
                        # Keep only overall_result, skip sample-level data
                        if 'overall_result' in alpha_data:
                            compact_by_alpha[alpha_key] = {
                                'overall_result': alpha_data['overall_result']
                            }
                    compact_crqr['by_alpha'] = compact_by_alpha

                # Skip by_feature as it's usually very large and buggy
                compact_primary['crqr'] = compact_crqr

            compact['primary_model'] = compact_primary

        # Skip alternative_models if present (usually empty anyway)

        return compact

    def _compact_robustness_data(self, test_data: dict) -> dict:
        """
        Create a compact version of robustness test data for AI validation.

        Args:
            test_data: Full robustness test data

        Returns:
            Compacted dictionary
        """
        compact = {}

        if 'primary_model' in test_data:
            primary = test_data['primary_model']
            compact_primary = {}

            # Keep essential fields
            for key in ['base_score', 'robustness_score', 'avg_raw_impact',
                       'avg_quantile_impact', 'avg_overall_impact', 'metrics']:
                if key in primary:
                    compact_primary[key] = primary[key]

            # Keep feature importance (top 10 only)
            if 'feature_importance' in primary:
                feat_imp = primary['feature_importance']
                sorted_features = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                compact_primary['feature_importance_top10'] = dict(sorted_features)

            # Keep summary of perturbation levels, not all raw data
            if 'raw' in primary and 'levels' in primary['raw']:
                levels_summary = []
                for level_data in primary['raw']['levels']:
                    levels_summary.append({
                        'level': level_data.get('level'),
                        'mean_score': level_data.get('mean_score'),
                        'worst_score': level_data.get('worst_score'),
                        'impact': level_data.get('impact')
                    })
                compact_primary['raw_levels_summary'] = levels_summary

            if 'quantile' in primary and 'levels' in primary['quantile']:
                levels_summary = []
                for level_data in primary['quantile']['levels']:
                    levels_summary.append({
                        'level': level_data.get('level'),
                        'mean_score': level_data.get('mean_score'),
                        'worst_score': level_data.get('worst_score'),
                        'impact': level_data.get('impact')
                    })
                compact_primary['quantile_levels_summary'] = levels_summary

            compact['primary_model'] = compact_primary

        return compact

    def _compact_resilience_data(self, test_data: dict) -> dict:
        """
        Create a compact version of resilience test data for AI validation.
        Removes large arrays of feature distances and keeps only essential metrics.

        Args:
            test_data: Full resilience test data

        Returns:
            Compacted dictionary with only essential information
        """
        compact = {}

        if 'primary_model' in test_data:
            primary = test_data['primary_model']
            compact_primary = {}

            # Keep essential metrics
            if 'metrics' in primary:
                compact_primary['metrics'] = primary['metrics']

            # Keep resilience score
            if 'resilience_score' in primary:
                compact_primary['resilience_score'] = primary['resilience_score']

            # Keep drift detection flag
            if 'drift_detected' in primary:
                compact_primary['drift_detected'] = primary['drift_detected']

            # Keep drift analysis summary (not raw arrays)
            if 'drift_analysis' in primary:
                drift = primary['drift_analysis']
                compact_drift = {}

                # Keep overall drift scores
                if 'data_drift_score' in drift:
                    compact_drift['data_drift_score'] = drift['data_drift_score']
                if 'concept_drift_score' in drift:
                    compact_drift['concept_drift_score'] = drift['concept_drift_score']

                # Keep only top 10 features with highest drift
                if 'per_feature_drift' in drift:
                    feat_drift = drift['per_feature_drift']
                    sorted_features = sorted(feat_drift.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                    compact_drift['per_feature_drift_top10'] = dict(sorted_features)

                compact_primary['drift_analysis'] = compact_drift

            # Keep distribution_shift but only summary per alpha (not all feature distances)
            if 'distribution_shift' in primary:
                dist_shift = primary['distribution_shift']
                compact_dist = {}

                if 'by_alpha' in dist_shift:
                    compact_by_alpha = {}
                    for alpha_key, alpha_data in dist_shift['by_alpha'].items():
                        if 'results' in alpha_data and alpha_data['results']:
                            # Keep only summary from first result (PSI)
                            first_result = alpha_data['results'][0]
                            compact_result = {
                                'method': first_result.get('method'),
                                'alpha': first_result.get('alpha'),
                                'metric': first_result.get('metric'),
                                'distance_metric': first_result.get('distance_metric'),
                                'worst_metric': first_result.get('worst_metric'),
                                'remaining_metric': first_result.get('remaining_metric'),
                                'performance_gap': first_result.get('performance_gap'),
                                'worst_sample_count': first_result.get('worst_sample_count'),
                                'remaining_sample_count': first_result.get('remaining_sample_count')
                            }

                            # Keep only top 10 features with highest distances
                            if 'feature_distances' in first_result and 'top_features' in first_result['feature_distances']:
                                top_features = first_result['feature_distances']['top_features']
                                # Sort and keep top 10
                                sorted_features = sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
                                compact_result['top_features_distances'] = dict(sorted_features)

                            compact_by_alpha[alpha_key] = {'summary': compact_result}

                    compact_dist['by_alpha'] = compact_by_alpha

                compact_primary['distribution_shift'] = compact_dist

            compact['primary_model'] = compact_primary

        return compact

    def _compact_initial_results(self, initial_results: dict) -> dict:
        """
        Create a compact version of initial_results for AI validation.

        Args:
            initial_results: Full initial results

        Returns:
            Compacted dictionary with only model metrics
        """
        compact = {}

        if 'models' in initial_results:
            compact_models = {}
            for model_name, model_data in initial_results['models'].items():
                compact_models[model_name] = {
                    'metrics': model_data.get('metrics', {}),
                    'model_type': model_data.get('model_type', 'Unknown')
                }
                # Keep top 5 feature importance if available
                if 'feature_importance' in model_data:
                    feat_imp = model_data['feature_importance']
                    sorted_features = sorted(feat_imp.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                    compact_models[model_name]['feature_importance_top5'] = dict(sorted_features)

            compact['models'] = compact_models

        return compact

    def to_dict(self) -> dict:
        """
        Convert all results to a dictionary for serialization.
        
        Returns:
            Complete dictionary representation of experiment results, with initial_results as first key
        """
        # Use OrderedDict to maintain key order
        from collections import OrderedDict
        result_dict = OrderedDict()
        
        # Simply add all results in the order they appear in self.results
        # The ExperimentResult.results should already have 'initial_results' as first key
        for name, result in self.results.items():
            if name == 'initial_results':
                # If name is 'initial_results', add it directly
                result_dict['initial_results'] = copy.deepcopy(result)
            else:
                # For other keys, get the complete result
                if hasattr(result, 'clean_results_dict'):
                    result_dict[name] = result.clean_results_dict()
                else:
                    result_dict[name] = copy.deepcopy(result.results)
        
        # Add essential metadata after the test results
        # In a way that doesn't affect the order of the first items
        metadata = {
            'experiment_type': self.experiment_type,
            'config': self.config,
            'generation_time': self.generation_time.strftime('%Y-%m-%d %H:%M:%S'),
            'tests_performed': list(k for k in self.results.keys() if k != 'initial_results')
        }
        
        # Update result_dict with metadata so it appears at the end
        for key, value in metadata.items():
            if key not in result_dict:
                result_dict[key] = value
                
        return result_dict
    
    @classmethod
    def from_dict(cls, results_dict: dict) -> 'ExperimentResult':
        """
        Create an ExperimentResult instance from a dictionary
        
        Args:
            results_dict: Dictionary containing test results
            
        Returns:
            ExperimentResult instance
        """
        # Validar entrada
        required_keys = ['experiment_type', 'config']
        for key in required_keys:
            if key not in results_dict:
                raise ValueError(f"Missing required key in results_dict: {key}")
        
        experiment_type = results_dict.get('experiment_type', 'binary_classification')
        config = results_dict.get('config', {})
        
        # Create instance
        instance = cls(experiment_type, config)
        
        # Create empty OrderedDict for results
        from collections import OrderedDict
        instance.results = OrderedDict()
        
        # Process initial_results first if available at the top level
        if 'initial_results' in results_dict:
            # Add initial_results directly to the results dict
            instance.results['initial_results'] = results_dict['initial_results']
        
        # Add test results
        test_types = {
            'robustness': RobustnessResult,
            'uncertainty': UncertaintyResult,
            'resilience': ResilienceResult,
            'hyperparameter': HyperparameterResult,
            'hyperparameters': HyperparameterResult
        }
        
        # Process test results in the order they appear in results_dict
        for key in results_dict:
            if key in test_types:
                test_result = copy.deepcopy(results_dict[key])
                instance.add_result(test_types[key](test_result))
            
        return instance


# Use dataclass para representação de resultados do modelo
from dataclasses import dataclass

@dataclass
class SimpleModelResult:
    """Simplified model result implementation"""
    model_name: str
    model_type: str
    metrics: dict
    
    # Campos opcionais com valores padrão
    features: list = None
    importance: dict = None
    hyperparameters: dict = None


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
    
    test_classes = {
        'robustness': RobustnessResult,
        'uncertainty': UncertaintyResult,
        'resilience': ResilienceResult,
        'hyperparameter': HyperparameterResult,
        'hyperparameters': HyperparameterResult
    }
    
    # Usar o dicionário para obter a classe correta ou um padrão
    result_class = test_classes.get(test_type, lambda name, results, metadata: 
                                   BaseTestResult(name.capitalize(), results, metadata))
    
    if test_type in test_classes:
        return result_class(results, metadata)
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
    def create_model_result(model_name, model_type, metrics, **kwargs):
        """Simplified factory function"""
        return SimpleModelResult(
            model_name=model_name,
            model_type=model_type,
            metrics=metrics,
            **kwargs
        )