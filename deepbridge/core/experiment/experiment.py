import typing as t
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from deepbridge.metrics.classification import Classification
from deepbridge.utils.model_registry import ModelType

from deepbridge.core.experiment.data_manager import DataManager
from deepbridge.core.experiment.model_evaluation import ModelEvaluation
from deepbridge.core.experiment.managers import ModelManager

# The following imports are done locally to avoid circular imports
# and to allow easier swapping of implementations
# from deepbridge.core.experiment.runner import TestRunner

from deepbridge.core.experiment.interfaces import IExperiment

class Experiment(IExperiment):
    """
    Main Experiment class coordinating different components for modeling tasks.
    This class has been refactored to delegate responsibilities to specialized components.
    Implements the IExperiment interface for standardized interaction.
    """
    
    VALID_TYPES = ["binary_classification", "regression", "forecasting"]
    
    def __init__(
        self,
        dataset: 'DBDataset',
        experiment_type: str,
        test_size: float = 0.2,
        random_state: int = 42,
        config: t.Optional[dict] = None,
        auto_fit: t.Optional[bool] = None,
        tests: t.Optional[t.List[str]] = None,
        feature_subset: t.Optional[t.List[str]] = None
        ):
        """
        Initialize the experiment with configuration and data.

        Args:
            dataset: DBDataset instance with features, target, and optionally model or probabilities
            experiment_type: Type of experiment ("binary_classification", "regression", "forecasting")
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            config: Optional configuration dictionary
            auto_fit: Whether to automatically fit a model. If None, will be set to True only if
                      dataset has probabilities but no model.
            tests: List of tests to prepare for the model. Available tests: ["robustness", "uncertainty", 
                   "resilience", "hyperparameters"]. Tests will only be executed when run_tests() is called.
            feature_subset: List of feature names to specifically test in the experiments.
                           In robustness tests, only these features will be perturbed while
                           all others remain unchanged. For other tests, only these features
                           will be analyzed in detail.
                           
        Note:
            Initialization does NOT run any tests - it only calculates basic metrics for the models.
            To run tests, explicitly call experiment.run_tests("quick") after initialization.
        """
        if experiment_type not in self.VALID_TYPES:
            raise ValueError(f"experiment_type must be one of {self.VALID_TYPES}")
            
        self.experiment_type = experiment_type
        self.dataset = dataset
        self.test_size = test_size
        self.random_state = random_state
        self.config = config or {}
        self.verbose = config.get('verbose', False) if config else False
        self.tests = tests or []
        
        # Feature subset for tests
        self.feature_subset = feature_subset
        
        # Automatically determine auto_fit value based on model presence
        if auto_fit is None:
            # If dataset has a model, auto_fit=False, otherwise auto_fit=True
            auto_fit = not (hasattr(dataset, 'model') and dataset.model is not None)
        
        # Store auto_fit value
        self.auto_fit = auto_fit
        
        # Initialize metrics calculator based on experiment type
        if experiment_type == "binary_classification":
            self.metrics_calculator = Classification()
            
        # Initialize results storage
        self._results_data = {
            'train': {},
            'test': {}
        }
        
        # Initialize distillation model
        self.distillation_model = None
        
        # Initialize helper components
        self.data_manager = DataManager(dataset, test_size, random_state)
        self.model_manager = ModelManager(dataset, self.experiment_type, self.verbose)
        self.model_evaluation = ModelEvaluation(self.experiment_type, self.metrics_calculator)
        
        # Data handling
        self.data_manager.prepare_data()
        self.X_train, self.X_test = self.data_manager.X_train, self.data_manager.X_test
        self.y_train, self.y_test = self.data_manager.y_train, self.data_manager.y_test
        self.prob_train, self.prob_test = self.data_manager.prob_train, self.data_manager.prob_test
        
        # Initialize alternative models
        self.alternative_models = self.model_manager.create_alternative_models(self.X_train, self.y_train)
        
        # Initialize test runner with feature selection
        # Use the new enhanced TestRunner class from runner.py
        from deepbridge.core.experiment.runner import TestRunner
        self.test_runner = TestRunner(
            self.dataset,
            self.alternative_models,
            self.tests,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.verbose,
            self.feature_subset
        )
        self._test_results = {}
        
        # Auto-fit if enabled and dataset has probabilities
        if self.auto_fit and hasattr(dataset, 'original_prob') and dataset.original_prob is not None:
            self._auto_fit_model()
        
        # SEMPRE calcule as métricas iniciais, independentemente dos testes solicitados
        # Apenas obtenha as métricas básicas, sem executar os testes completos
        self.initial_results = self.test_runner.run_initial_tests()
        
        # Process all models to ensure roc_auc is present and remove auc
        if 'models' in self.initial_results:
            # Helper function to calculate roc_auc for a model
            def ensure_roc_auc(model_name, model_data, model_obj):
                if 'metrics' not in model_data:
                    return
                    
                metrics = model_data['metrics']
                
                # If we have auc but not roc_auc, copy it
                if 'auc' in metrics and 'roc_auc' not in metrics:
                    metrics['roc_auc'] = metrics['auc']
                
                # If we still don't have roc_auc, calculate it if possible
                if 'roc_auc' not in metrics and model_obj is not None:
                    try:
                        # Only calculate if model has predict_proba
                        if hasattr(model_obj, 'predict_proba'):
                            from sklearn.metrics import roc_auc_score
                            y_prob = model_obj.predict_proba(self.X_test)
                            if y_prob.shape[1] > 1:  # For binary classification
                                roc_auc = roc_auc_score(self.y_test, y_prob[:, 1])
                                metrics['roc_auc'] = roc_auc
                                if self.verbose:
                                    print(f"Calculated ROC AUC for {model_name}: {roc_auc}")
                    except Exception as e:
                        if self.verbose:
                            print(f"Could not calculate ROC AUC for {model_name}: {str(e)}")
                
                # Ensure both auc and roc_auc are present and consistent
                if 'roc_auc' in metrics and 'auc' not in metrics:
                    metrics['auc'] = metrics['roc_auc']
                elif 'auc' in metrics and 'roc_auc' not in metrics:
                    metrics['roc_auc'] = metrics['auc']
            
            # Process primary model
            if 'primary_model' in self.initial_results['models']:
                ensure_roc_auc('primary_model', 
                              self.initial_results['models']['primary_model'],
                              self.dataset.model if hasattr(self.dataset, 'model') else None)
            
            # Process all alternative models
            for model_name, model_data in self.initial_results['models'].items():
                if model_name != 'primary_model':
                    model_obj = self.alternative_models.get(model_name)
                    ensure_roc_auc(model_name, model_data, model_obj)
        
        if self.verbose:
            print("Métricas iniciais calculadas para todos os modelos.")
            print("Use experiment.run_tests('quick') para executar os testes definidos em tests=[...] (se houver).")
    
    def _auto_fit_model(self):
        """Auto-fit a model when probabilities are available but no model is present"""
        default_model_type = self.model_manager.get_default_model_type()
        
        if default_model_type is not None:
            self.fit(
                student_model_type=default_model_type,
                temperature=1.0,
                alpha=0.5,
                use_probabilities=True,
                verbose=False
            )
        else:
            if self.verbose:
                print("No model types available, skipping auto-fit")
    
    def fit(self, 
             student_model_type=ModelType.LOGISTIC_REGRESSION,
             student_params=None,
             temperature=1.0,
             alpha=0.5,
             use_probabilities=True,
             n_trials=50,
             validation_split=0.2,
             verbose=True,
             distillation_method="surrogate",
             **kwargs):
        """Train a model using either Surrogate Model or Knowledge Distillation approach."""
        if self.experiment_type != "binary_classification":
            raise ValueError("Distillation methods are only supported for binary classification")
        
        # Configure logging
        logging_state = self._configure_logging(verbose)
        
        try:
            # Create and train distillation model
            self.distillation_model = self.model_manager.create_distillation_model(
                distillation_method, 
                student_model_type, 
                student_params,
                temperature, 
                alpha, 
                use_probabilities, 
                n_trials, 
                validation_split
            )
            
            # Train the model
            self.distillation_model.fit(self.X_train, self.y_train, verbose=verbose)
            
            # Evaluate and store results
            train_metrics = self.model_evaluation.evaluate_distillation(
                self.distillation_model, 'train', 
                self.X_train, self.y_train, self.prob_train
            )
            self._results_data['train'] = train_metrics['metrics']
            
            test_metrics = self.model_evaluation.evaluate_distillation(
                self.distillation_model, 'test', 
                self.X_test, self.y_test, self.prob_test
            )
            self._results_data['test'] = test_metrics['metrics']
            
            return self
        finally:
            # Restore logging state
            self._restore_logging(logging_state, verbose)

    def _configure_logging(self, verbose: bool) -> t.Optional[int]:
        """Configure logging for Optuna based on verbose mode"""
        if not verbose:
            optuna_logger = logging.getLogger("optuna")
            optuna_logger_level = optuna_logger.getEffectiveLevel()
            optuna_logger.setLevel(logging.ERROR)
            return optuna_logger_level
        return None
        
    def _restore_logging(self, logging_state: t.Optional[int], verbose: bool) -> None:
        """Restore Optuna logging to original state"""
        if not verbose and logging_state is not None:
            optuna_logger = logging.getLogger("optuna")
            optuna_logger.setLevel(logging_state)

    def run_tests(self, config_name: str = 'quick', **kwargs) -> dict:
        """
        Run all tests specified during initialization with the given configuration.
        
        Args:
            config_name: Name of the configuration to use: 'quick', 'medium', or 'full'
            **kwargs: Additional parameters to pass to the test runner
            
        Returns:
            dict: Dictionary with initial_results and test results that has a save_report method
        """
        from deepbridge.core.experiment.results import wrap_results
        
        # Primeiro, certifique-se de que temos métricas iniciais
        if not hasattr(self, 'initial_results') or not self.initial_results:
            self.initial_results = self.test_runner.run_initial_tests()
            
        # Execute os testes solicitados
        test_results = self.test_runner.run_tests(config_name, **kwargs)
        self._test_results.update(test_results)
        
        # Crie um dicionário combinado com initial_results e test_results
        combined_results = {
            'experiment_type': self.experiment_type,
            'config': {'name': config_name, 'tests': self.tests},
            # Inclua os initial_results para garantir que as métricas dos modelos estejam disponíveis
            'initial_results': self.initial_results,
            # Adicione os resultados dos testes
            **test_results
        }
        
        # Wrap the results in an ExperimentResult object with save_report method
        experiment_result = wrap_results(combined_results)
        
        # Log útil para diagnóstico
        if self.verbose:
            print(f"Testes executados com configuração '{config_name}'")
            print(f"Testes realizados: {list(test_results.keys())}")
            if 'models' in self.initial_results:
                print("Métricas disponíveis para os seguintes modelos:")
                for model_name in self.initial_results['models']:
                    print(f"  - {model_name}")
        
        return experiment_result
        
    def run_test(self, test_type: str, config_name: str = 'quick', **kwargs) -> 'TestResult':
        """
        Run a specific test with the given configuration.
        
        Args:
            test_type: Type of test to run (robustness, uncertainty, etc.)
            config_name: Name of the configuration to use: 'quick', 'medium', or 'full'
            **kwargs: Additional parameters to pass to the test runner
            
        Returns:
            TestResult: Result object for the specific test
        """
        # Delegate to the test runner
        return self.test_runner.run_test(test_type, config_name, **kwargs)

    @property
    def model(self):
        """Return either the distillation model (if trained) or the model from dataset."""
        if hasattr(self, 'distillation_model') and self.distillation_model is not None:
            return self.distillation_model
        elif hasattr(self.dataset, 'model') and self.dataset.model is not None:
            return self.dataset.model
        return None

    def get_student_predictions(self, dataset: str = 'test') -> pd.DataFrame:
        """Get predictions from the trained student model."""
        if not hasattr(self, 'distillation_model') or self.distillation_model is None:
            raise ValueError("No trained distillation model available. Call fit() first")
        
        return self.model_evaluation.get_predictions(
            self.distillation_model,
            self.X_train if dataset == 'train' else self.X_test,
            self.y_train if dataset == 'train' else self.y_test
        )

    def calculate_metrics(self, y_true, y_pred, y_prob=None, teacher_prob=None):
        """Calculate metrics based on experiment type."""
        return self.model_evaluation.calculate_metrics(
            y_true, y_pred, y_prob, teacher_prob
        )

    def compare_all_models(self, dataset='test'):
        """Compare all models including original, alternative, and distilled."""
        X = self.X_train if dataset == 'train' else self.X_test
        y = self.y_train if dataset == 'train' else self.y_test
        
        return self.model_evaluation.compare_all_models(
            dataset,
            self.dataset.model if hasattr(self.dataset, 'model') else None,
            self.alternative_models,
            self.distillation_model if hasattr(self, 'distillation_model') else None,
            X, y
        )

    def get_comprehensive_results(self):
        """Return a comprehensive dictionary with all metrics and information."""
        # Simplified version that returns basic experiment info
        return {
            'experiment_type': self.experiment_type,
            'config': {
                'test_size': self.test_size,
                'random_state': self.random_state,
                'auto_fit': self.auto_fit
            },
            'model_info': {
                'has_primary_model': self.model is not None,
                'has_alternative_models': len(self.alternative_models) > 0 if self.alternative_models else False,
                'has_distillation_model': hasattr(self, 'distillation_model') and self.distillation_model is not None
            }
        }

    def save_report(self, report_type: str = None, report_path: str = None) -> str:
        """
        This method has been deprecated as part of removing visualization and reporting.
        
        Returns:
            str: Message indicating reports are no longer supported
        """
        raise NotImplementedError("Reports and visualizations have been removed from this version. Use model metrics and test results directly.")

    # Direct access to test results
    def get_robustness_results(self):
        """Get the robustness test results."""
        return self.test_results.get('robustness', {})
        
    def get_uncertainty_results(self):
        """Get the uncertainty test results."""
        return self.test_results.get('uncertainty', {})
    
    def get_resilience_results(self):
        """Get the resilience test results."""
        return self.test_results.get('resilience', {})
    
    def get_hyperparameter_results(self):
        """Get the hyperparameter importance test results."""
        return self.test_results.get('hyperparameters', {})

    # Required properties from IExperiment interface
    @property
    def experiment_type(self) -> str:
        """
        Get the experiment type.
        
        Returns:
            String indicating the experiment type (binary_classification, regression, etc.)
        """
        return self._experiment_type
        
    @experiment_type.setter
    def experiment_type(self, value: str):
        """Set the experiment type."""
        self._experiment_type = value
    
    @property
    def test_results(self):
        """
        Get all test results.
        
        Returns:
            Dictionary containing all test results
        """
        return self._test_results
    
    # Proxy properties to maintain backward compatibility
    # @property
    # def results(self):
    #     """Property to get results data"""
    #     return self._results_data

    # @results.setter
    # def results(self, value):
    #     """Property setter for results"""
    #     self._results_data = value

    # @property
    # def metrics(self):
    #     """Get all metrics for both train and test datasets."""
    #     # Forward to model_evaluation's get_metrics
    #     return {
    #         'train': self._results_data.get('train', {}),
    #         'test': self._results_data.get('test', {})
    #     }
        
    @property
    def experiment_info(self):
        """
        Get experiment information including configuration and model metrics.
        This is available immediately after experiment initialization without
        running full tests.
        
        Returns:
        --------
        dict : Dictionary with experiment config and model metrics
        """
        if hasattr(self, 'initial_results'):
            return self.initial_results
        else:
            return {
                'config': {
                    'tests': self.tests,
                    'experiment_type': self.experiment_type,
                    'test_size': self.test_size,
                    'random_state': self.random_state,
                    'auto_fit': self.auto_fit
                }
            }