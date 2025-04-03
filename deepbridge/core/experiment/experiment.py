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
from deepbridge.core.experiment.managers import ModelManager, RobustnessManager, UncertaintyManager, ResilienceManager, HyperparameterManager

class Experiment:
    """
    Experiment class to handle different types of modeling tasks and their configurations.
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
        
        # Initialize tests results storage
        self.test_results = {}
        
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
        
        # Auto-fit if enabled and dataset has probabilities
        if self.auto_fit and hasattr(dataset, 'original_prob') and dataset.original_prob is not None:
            self._auto_fit_model()
        
        # Run requested tests if any
        if self.tests:
            self._run_tests()
    
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

    def _run_tests(self) -> None:
        """Run the tests specified in self.tests."""
        if self.verbose:
            print(f"Running the following tests: {self.tests}")
            
        # Check if we have a model to test
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            if self.verbose:
                print("No model found in dataset. Skipping tests.")
            return
            
        # Run robustness tests if requested
        if "robustness" in self.tests:
            robustness_manager = RobustnessManager(
                self.dataset, 
                self.alternative_models, 
                self.verbose
            )
            self.test_results['robustness'] = robustness_manager.run_tests()
            
        # Run uncertainty tests if requested
        if "uncertainty" in self.tests:
            uncertainty_manager = UncertaintyManager(
                self.dataset, 
                self.alternative_models, 
                self.verbose
            )
            self.test_results['uncertainty'] = uncertainty_manager.run_tests()
            
        # Run resilience tests if requested
        if "resilience" in self.tests:
            resilience_manager = ResilienceManager(
                self.dataset, 
                self.alternative_models, 
                self.verbose
            )
            self.test_results['resilience'] = resilience_manager.run_tests()
            
        # Run hyperparameter tests if requested
        if "hyperparameters" in self.tests:
            hyperparameter_manager = HyperparameterManager(
                self.dataset, 
                self.alternative_models, 
                self.verbose
            )
            self.test_results['hyperparameters'] = hyperparameter_manager.run_tests()
        
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
        if self.verbose:
            print(f"Running tests with {config_name} configuration...")
            
        # Check if we have a model to test
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            if self.verbose:
                print("No model found in dataset. Skipping tests.")
            return {}
        
        # Initialize results dictionary
        results = {}
        
        # Run robustness tests if requested
        if "robustness" in self.tests:
            from deepbridge.utils.robustness import run_robustness_tests
            from deepbridge.core.db_data import DBDataset
            
            # Initialize robustness results dictionary
            robustness_results = {
                'primary_model': {},
                'alternative_models': {}
            }
            
            # Test primary model
            if self.verbose:
                print(f"Testing robustness of primary model...")
            
            primary_results = run_robustness_tests(
                self.dataset, 
                config_name=config_name,
                metric='AUC', 
                verbose=self.verbose
            )
            robustness_results['primary_model'] = primary_results
            
            # Test alternative models
            if self.alternative_models:
                for model_name, model in self.alternative_models.items():
                    if self.verbose:
                        print(f"Testing robustness of alternative model: {model_name}")
                    
                    # Create a new dataset with the alternative model
                    alt_dataset = DBDataset(
                        train_data=self.dataset.train_data,
                        test_data=self.dataset.test_data,
                        target_column=self.dataset.target_name,
                        model=model
                    )
                    
                    # Run robustness tests on the alternative model
                    alt_results = run_robustness_tests(
                        alt_dataset,
                        config_name=config_name,
                        metric='AUC',
                        verbose=self.verbose
                    )
                    
                    # Store results
                    robustness_results['alternative_models'][model_name] = alt_results
            
            # Store all robustness results
            results['robustness'] = robustness_results
            
        # Run uncertainty tests if requested
        if "uncertainty" in self.tests:
            from deepbridge.utils.uncertainty import run_uncertainty_tests
            from deepbridge.core.db_data import DBDataset
            
            # Initialize uncertainty results dictionary
            uncertainty_results = {
                'primary_model': {},
                'alternative_models': {}
            }
            
            # Test primary model
            if self.verbose:
                print(f"Testing uncertainty quantification of primary model...")
            
            primary_results = run_uncertainty_tests(
                self.dataset, 
                config_name=config_name,
                verbose=self.verbose
            )
            uncertainty_results['primary_model'] = primary_results
            
            # Test alternative models
            if self.alternative_models:
                for model_name, model in self.alternative_models.items():
                    if self.verbose:
                        print(f"Testing uncertainty of alternative model: {model_name}")
                    
                    # Create a new dataset with the alternative model
                    alt_dataset = DBDataset(
                        train_data=self.dataset.train_data,
                        test_data=self.dataset.test_data,
                        target_column=self.dataset.target_name,
                        model=model
                    )
                    
                    # Run uncertainty tests on the alternative model
                    alt_results = run_uncertainty_tests(
                        alt_dataset,
                        config_name=config_name,
                        verbose=self.verbose
                    )
                    
                    # Store results
                    uncertainty_results['alternative_models'][model_name] = alt_results
            
            # Store all uncertainty results
            results['uncertainty'] = uncertainty_results
            
        # Run resilience tests if requested
        if "resilience" in self.tests:
            from deepbridge.utils.resilience import run_resilience_tests
            from deepbridge.core.db_data import DBDataset
            
            # Initialize resilience results dictionary
            resilience_results = {
                'primary_model': {},
                'alternative_models': {}
            }
            
            # Test primary model
            if self.verbose:
                print(f"Testing resilience of primary model...")
            
            primary_results = run_resilience_tests(
                self.dataset, 
                config_name=config_name,
                metric='auc', 
                verbose=self.verbose
            )
            resilience_results['primary_model'] = primary_results
            
            # Test alternative models
            if self.alternative_models:
                for model_name, model in self.alternative_models.items():
                    if self.verbose:
                        print(f"Testing resilience of alternative model: {model_name}")
                    
                    # Create a new dataset with the alternative model
                    alt_dataset = DBDataset(
                        train_data=self.dataset.train_data,
                        test_data=self.dataset.test_data,
                        target_column=self.dataset.target_name,
                        model=model
                    )
                    
                    # Run resilience tests on the alternative model
                    alt_results = run_resilience_tests(
                        alt_dataset,
                        config_name=config_name,
                        metric='auc',
                        verbose=self.verbose
                    )
                    
                    # Store results
                    resilience_results['alternative_models'][model_name] = alt_results
            
            # Store all resilience results
            results['resilience'] = resilience_results
            
        # Run hyperparameter tests if requested
        if "hyperparameters" in self.tests:
            from deepbridge.utils.hyperparameter import run_hyperparameter_tests
            from deepbridge.core.db_data import DBDataset
            
            # Initialize hyperparameter results dictionary
            hyperparameter_results = {
                'primary_model': {},
                'alternative_models': {}
            }
            
            # Test primary model
            if self.verbose:
                print(f"Testing hyperparameter importance of primary model...")
            
            primary_results = run_hyperparameter_tests(
                self.dataset, 
                config_name=config_name,
                metric='accuracy', 
                verbose=self.verbose
            )
            hyperparameter_results['primary_model'] = primary_results
            
            # Test alternative models
            if self.alternative_models:
                for model_name, model in self.alternative_models.items():
                    if self.verbose:
                        print(f"Testing hyperparameter importance of alternative model: {model_name}")
                    
                    # Create a new dataset with the alternative model
                    alt_dataset = DBDataset(
                        train_data=self.dataset.train_data,
                        test_data=self.dataset.test_data,
                        target_column=self.dataset.target_name,
                        model=model
                    )
                    
                    # Run hyperparameter tests on the alternative model
                    alt_results = run_hyperparameter_tests(
                        alt_dataset,
                        config_name=config_name,
                        metric='accuracy',
                        verbose=self.verbose
                    )
                    
                    # Store results
                    hyperparameter_results['alternative_models'][model_name] = alt_results
            
            # Store all hyperparameter results
            results['hyperparameters'] = hyperparameter_results
        
        # Store results in the object for future reference
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

    # Robustness testing methods
    def get_robustness_results(self):
        """
        Get the robustness test results.
        
        Returns:
            dict: Dictionary containing robustness test results for main model and alternatives.
                  Returns None if robustness tests haven't been run.
        """
        if 'robustness' not in self.test_results:
            if "robustness" in self.tests:
                # Run robustness tests if they were requested but not run yet
                robustness_manager = RobustnessManager(
                    self.dataset, 
                    self.alternative_models, 
                    self.verbose
                )
                self.test_results['robustness'] = robustness_manager.run_tests()
                return self.test_results.get('robustness')
            return None
        return self.test_results.get('robustness')
        
    def get_robustness_visualizations(self):
        """
        Get the robustness visualizations generated by the tests.
        
        Returns:
            dict: Dictionary of plotly figures for robustness visualizations.
                  Returns empty dict if no visualizations are available.
        """
        if 'robustness' not in self.test_results:
            if "robustness" in self.tests:
                # Run robustness tests if they were requested but not run yet
                robustness_manager = RobustnessManager(
                    self.dataset, 
                    self.alternative_models, 
                    self.verbose
                )
                self.test_results['robustness'] = robustness_manager.run_tests()
            else:
                return {}
                
        return self.test_results.get('robustness', {}).get('visualizations', {})
        
    def plot_robustness_comparison(self):
        """
        Get the plotly figure showing the comparison of robustness across models.
        
        Returns:
            plotly.graph_objects.Figure: Comparison plot of models robustness.
                                         Returns None if visualization not available.
        """
        visualizations = self.get_robustness_visualizations()
        return visualizations.get('models_comparison')
    
    def plot_robustness_distribution(self):
        """
        Get the boxplot showing distribution of robustness scores.
        
        Returns:
            plotly.graph_objects.Figure: Boxplot of robustness score distributions.
                                         Returns None if visualization not available.
        """
        visualizations = self.get_robustness_visualizations()
        return visualizations.get('score_distribution')
    
    def plot_feature_importance_robustness(self):
        """
        Get the plotly figure showing feature importance for robustness.
        
        Returns:
            plotly.graph_objects.Figure: Feature importance bar chart.
                                         Returns None if visualization not available.
        """
        visualizations = self.get_robustness_visualizations()
        return visualizations.get('feature_importance')
    
    def plot_perturbation_methods_comparison(self):
        """
        Get the plotly figure comparing different perturbation methods.
        
        Returns:
            plotly.graph_objects.Figure: Comparison plot of perturbation methods.
                                         Returns None if visualization not available.
        """
        visualizations = self.get_robustness_visualizations()
        return visualizations.get('perturbation_methods')
        
    # Uncertainty testing methods
    def get_uncertainty_results(self):
        """
        Get the uncertainty test results.
        
        Returns:
            dict: Dictionary containing uncertainty test results for main model and alternatives.
                  Returns None if uncertainty tests haven't been run.
        """
        if 'uncertainty' not in self.test_results:
            if "uncertainty" in self.tests:
                # Run uncertainty tests if they were requested but not run yet
                uncertainty_manager = UncertaintyManager(
                    self.dataset, 
                    self.alternative_models, 
                    self.verbose
                )
                self.test_results['uncertainty'] = uncertainty_manager.run_tests()
                return self.test_results.get('uncertainty')
            return None
        return self.test_results.get('uncertainty')
        
    def get_uncertainty_visualizations(self):
        """
        Get the uncertainty visualizations generated by the tests.
        
        Returns:
            dict: Dictionary of plotly figures for uncertainty visualizations.
                  Returns empty dict if no visualizations are available.
        """
        if 'uncertainty' not in self.test_results:
            if "uncertainty" in self.tests:
                # Run uncertainty tests if they were requested but not run yet
                uncertainty_manager = UncertaintyManager(
                    self.dataset, 
                    self.alternative_models, 
                    self.verbose
                )
                self.test_results['uncertainty'] = uncertainty_manager.run_tests()
            else:
                return {}
                
        return self.test_results.get('uncertainty', {}).get('visualizations', {})
        
    def plot_uncertainty_alpha_comparison(self):
        """
        Get the plotly figure showing the comparison of different alpha levels.
        
        Returns:
            plotly.graph_objects.Figure: Comparison plot of alpha levels.
                                         Returns None if visualization not available.
        """
        visualizations = self.get_uncertainty_visualizations()
        return visualizations.get('alpha_comparison')
    
    def plot_uncertainty_width_distribution(self):
        """
        Get the boxplot showing distribution of interval widths.
        
        Returns:
            plotly.graph_objects.Figure: Boxplot of interval width distributions.
                                         Returns None if visualization not available.
        """
        visualizations = self.get_uncertainty_visualizations()
        return visualizations.get('width_distribution')
    
    def plot_feature_importance_uncertainty(self):
        """
        Get the plotly figure showing feature importance for uncertainty.
        
        Returns:
            plotly.graph_objects.Figure: Feature importance bar chart.
                                         Returns None if visualization not available.
        """
        visualizations = self.get_uncertainty_visualizations()
        return visualizations.get('feature_importance')
    
    def plot_coverage_vs_width(self):
        """
        Get the plotly figure showing trade-off between coverage and width.
        
        Returns:
            plotly.graph_objects.Figure: Coverage vs width plot.
                                         Returns None if visualization not available.
        """
        visualizations = self.get_uncertainty_visualizations()
        return visualizations.get('coverage_vs_width')
        
    # Resilience testing methods
    def get_resilience_results(self):
        """
        Get the resilience test results.
        
        Returns:
            dict: Dictionary containing resilience test results for main model and alternatives.
                  Returns None if resilience tests haven't been run.
        """
        if 'resilience' not in self.test_results:
            if "resilience" in self.tests:
                # Run resilience tests if they were requested but not run yet
                resilience_manager = ResilienceManager(
                    self.dataset, 
                    self.alternative_models, 
                    self.verbose
                )
                self.test_results['resilience'] = resilience_manager.run_tests()
                return self.test_results.get('resilience')
            return None
        return self.test_results.get('resilience')
    
    # Hyperparameter testing methods
    def get_hyperparameter_results(self):
        """
        Get the hyperparameter importance test results.
        
        Returns:
            dict: Dictionary containing hyperparameter importance results for main model and alternatives.
                  Returns None if hyperparameter tests haven't been run.
        """
        if 'hyperparameters' not in self.test_results:
            if "hyperparameters" in self.tests:
                # Run hyperparameter tests if they were requested but not run yet
                hyperparameter_manager = HyperparameterManager(
                    self.dataset, 
                    self.alternative_models, 
                    self.verbose
                )
                self.test_results['hyperparameters'] = hyperparameter_manager.run_tests()
                return self.test_results.get('hyperparameters')
            return None
        return self.test_results.get('hyperparameters')
    
    def get_hyperparameter_importance(self):
        """
        Get the hyperparameter importance scores for the primary model.
        
        Returns:
            dict: Dictionary of parameter names to importance scores.
                  Returns None if hyperparameter tests haven't been run.
        """
        results = self.get_hyperparameter_results()
        if results and 'primary_model' in results:
            return results['primary_model'].get('sorted_importance', {})
        return None
    
    def get_hyperparameter_tuning_order(self):
        """
        Get the suggested hyperparameter tuning order for the primary model.
        
        Returns:
            list: List of parameter names in recommended tuning order.
                  Returns None if hyperparameter tests haven't been run.
        """
        results = self.get_hyperparameter_results()
        if results and 'primary_model' in results:
            return results['primary_model'].get('tuning_order', [])
        return None

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
