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


class ResilienceResult(BaseTestResult):
    """Result object for resilience tests"""
    
    def __init__(self, results: dict, metadata: t.Optional[dict] = None):
        super().__init__("Resilience", results, metadata)


class HyperparameterResult(BaseTestResult):
    """Result object for hyperparameter tests"""
    
    def __init__(self, results: dict, metadata: t.Optional[dict] = None):
        super().__init__("Hyperparameter", results, metadata)


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
            elif 'results' in test_result and 'primary_model' in test_result['results']:
                # Nested structure - extract and use the primary_model data
                report_data = copy.deepcopy(test_result['results'])
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
        else:
            # For other test types, use the standard approach
            if hasattr(result, 'to_dict'):
                report_data = result.to_dict()['results']
            elif hasattr(result, 'results'):
                report_data = result.results
            else:
                report_data = result  # If result is already a dict

        # Add initial_results if available
        if 'initial_results' in self.results:
            report_data['initial_results'] = self.results['initial_results']

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

    def save_json(self, test_type: str, file_path: str, include_summary: bool = True) -> str:
        """
        Save test results to a JSON file for AI analysis.

        Args:
            test_type: Type of test results to save ('robustness', 'uncertainty', etc.)
            file_path: Path where to save the JSON file
            include_summary: Whether to include a summary section with key findings (default: True)

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

        # Add initial model evaluation if available
        if 'initial_results' in self.results:
            json_data["initial_model_evaluation"] = self.results['initial_results']

        # Add summary for robustness tests
        if include_summary and test_type == 'robustness':
            summary = self._generate_robustness_summary(test_data)
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

        # Save to JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
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