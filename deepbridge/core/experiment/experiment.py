import typing as t
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from deepbridge.metrics.classification import Classification
from deepbridge.utils.model_registry import ModelType

from deepbridge.core.experiment.data_manager import DataManager
from deepbridge.core.experiment.model_evaluation import ModelEvaluation
from deepbridge.core.experiment.report_generator import ReportGenerator
from deepbridge.core.experiment.managers import ModelManager
from deepbridge.core.experiment.test_runner import TestRunner
from deepbridge.core.experiment.visualization_manager import VisualizationManager

class Experiment:
    """
    Main Experiment class coordinating different components for modeling tasks.
    This class has been refactored to delegate responsibilities to specialized components.
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
        tests: t.Optional[t.List[str]] = None
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
            tests: List of tests to run on the model. Available tests: ["robustness", "uncertainty", 
                   "resilience", "hyperparameters"]
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
        self.report_generator = ReportGenerator()
        
        # Data handling
        self.data_manager.prepare_data()
        self.X_train, self.X_test = self.data_manager.X_train, self.data_manager.X_test
        self.y_train, self.y_test = self.data_manager.y_train, self.data_manager.y_test
        self.prob_train, self.prob_test = self.data_manager.prob_train, self.data_manager.prob_test
        
        # Initialize alternative models
        self.alternative_models = self.model_manager.create_alternative_models(self.X_train, self.y_train)
        
        # Initialize test runner
        self.test_runner = TestRunner(
            self.dataset,
            self.alternative_models,
            self.tests,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.verbose
        )
        self.test_results = {}
        
        # Initialize visualization manager
        self.visualization_manager = VisualizationManager(self.test_runner)
        
        # Auto-fit if enabled and dataset has probabilities
        if self.auto_fit and hasattr(dataset, 'original_prob') and dataset.original_prob is not None:
            self._auto_fit_model()
        
        # Run requested tests if any
        if self.tests:
            self.test_results = self.test_runner.run_initial_tests()
    
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

    def run_tests(self, config_name: str = 'quick') -> dict:
        """
        Run all tests specified during initialization with the given configuration.
        
        Parameters:
        -----------
        config_name : str
            Name of the configuration to use: 'quick', 'medium', or 'full'
            
        Returns:
        --------
        dict : Dictionary with test results
        """
        results = self.test_runner.run_tests(config_name)
        self.test_results.update(results)
        return results

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
        return self.report_generator.generate_comprehensive_results(
            self.experiment_type,
            self.test_size,
            self.random_state,
            self.auto_fit,
            self.dataset,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.model,
            self.alternative_models,
            self.distillation_model if hasattr(self, 'distillation_model') else None,
            self.metrics_calculator
        )

    def save_report(self, report_path: str) -> str:
        """Generate and save an HTML report with all experiment results."""
        return self.report_generator.save_report(
            report_path,
            self.get_comprehensive_results()
        )

    # Delegation methods to VisualizationManager
    def get_robustness_results(self):
        """Get the robustness test results."""
        return self.visualization_manager.get_robustness_results()
        
    def get_robustness_visualizations(self):
        """Get the robustness visualizations generated by the tests."""
        return self.visualization_manager.get_robustness_visualizations()
        
    def plot_robustness_comparison(self):
        """Get the plotly figure showing the comparison of robustness across models."""
        return self.visualization_manager.plot_robustness_comparison()
    
    def plot_robustness_distribution(self):
        """Get the boxplot showing distribution of robustness scores."""
        return self.visualization_manager.plot_robustness_distribution()
    
    def plot_feature_importance_robustness(self):
        """Get the plotly figure showing feature importance for robustness."""
        return self.visualization_manager.plot_feature_importance_robustness()
    
    def plot_perturbation_methods_comparison(self):
        """Get the plotly figure comparing different perturbation methods."""
        return self.visualization_manager.plot_perturbation_methods_comparison()
        
    # Delegation methods to VisualizationManager for uncertainty
    def get_uncertainty_results(self):
        """Get the uncertainty test results."""
        return self.visualization_manager.get_uncertainty_results()
        
    def get_uncertainty_visualizations(self):
        """Get the uncertainty visualizations generated by the tests."""
        return self.visualization_manager.get_uncertainty_visualizations()
        
    def plot_uncertainty_alpha_comparison(self):
        """Get the plotly figure showing the comparison of different alpha levels."""
        return self.visualization_manager.plot_uncertainty_alpha_comparison()
    
    def plot_uncertainty_width_distribution(self):
        """Get the boxplot showing distribution of interval widths."""
        return self.visualization_manager.plot_uncertainty_width_distribution()
    
    def plot_feature_importance_uncertainty(self):
        """Get the plotly figure showing feature importance for uncertainty."""
        return self.visualization_manager.plot_feature_importance_uncertainty()
    
    def plot_coverage_vs_width(self):
        """Get the plotly figure showing trade-off between coverage and width."""
        return self.visualization_manager.plot_coverage_vs_width()
        
    # Delegation methods to visualization manager for resilience and hyperparameter
    def get_resilience_results(self):
        """Get the resilience test results."""
        return self.visualization_manager.get_resilience_results()
    
    def get_hyperparameter_results(self):
        """Get the hyperparameter importance test results."""
        return self.visualization_manager.get_hyperparameter_results()
    
    def get_hyperparameter_importance(self):
        """Get the hyperparameter importance scores for the primary model."""
        return self.visualization_manager.get_hyperparameter_importance()
    
    def get_hyperparameter_tuning_order(self):
        """Get the suggested hyperparameter tuning order for the primary model."""
        return self.visualization_manager.get_hyperparameter_tuning_order()

    # Proxy properties to maintain backward compatibility
    @property
    def results(self):
        """Property to get results data"""
        return self._results_data

    @results.setter
    def results(self, value):
        """Property setter for results"""
        self._results_data = value

    @property
    def metrics(self):
        """Get all metrics for both train and test datasets."""
        # Forward to model_evaluation's get_metrics
        return {
            'train': self._results_data.get('train', {}),
            'test': self._results_data.get('test', {})
        }