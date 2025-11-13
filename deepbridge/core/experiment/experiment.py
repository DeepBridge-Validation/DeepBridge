import typing as t
from pathlib import Path
import pandas as pd
import numpy as np
import logging

from deepbridge.metrics.classification import Classification
from deepbridge.metrics.regression import Regression
from deepbridge.utils.model_registry import ModelType
from deepbridge.utils.logger import get_logger

from deepbridge.core.experiment.data_manager import DataManager
from deepbridge.core.experiment.model_evaluation import ModelEvaluation
from deepbridge.core.experiment.managers import ModelManager

# TestRunner is imported at runtime to avoid circular imports
# This approach is cleaner than local imports in methods

from deepbridge.core.experiment.interfaces import IExperiment

class Experiment(IExperiment):
    """
    Main Experiment class coordinating different components for modeling tasks.
    This class has been refactored to delegate responsibilities to specialized components.
    Implements the IExperiment interface for standardized interaction.
    """
    # Initialize logger for this class
    logger = get_logger("deepbridge.experiment")

    VALID_TYPES = ["binary_classification", "multiclass_classification", "regression", "forecasting"]

    # Common sensitive attribute keywords for auto-detection
    SENSITIVE_ATTRIBUTE_KEYWORDS = {
        'gender', 'sex', 'sexo', 'genero',
        'race', 'raca', 'ethnicity', 'etnia',
        'age', 'idade', 'age_group', 'faixa_etaria',
        'religion', 'religiao',
        'disability', 'deficiencia',
        'marital_status', 'estado_civil',
        'nationality', 'nacionalidade',
        'sexual_orientation', 'orientacao_sexual'
    }

    @staticmethod
    def detect_sensitive_attributes(dataset: 'DBDataset', threshold: float = 0.7) -> t.List[str]:
        """
        Automatically detect potential sensitive/protected attributes in the dataset.

        Uses fuzzy string matching against common sensitive attribute names.

        Args:
            dataset: DBDataset to analyze
            threshold: Similarity threshold for fuzzy matching (0-1)

        Returns:
            List of detected sensitive attribute names

        Example:
            >>> detected = Experiment.detect_sensitive_attributes(dataset)
            >>> print(f"Found sensitive attributes: {detected}")
        """
        from difflib import SequenceMatcher

        detected = []

        # Get all column names from dataset
        if hasattr(dataset, '_data') and isinstance(dataset._data, pd.DataFrame):
            columns = dataset._data.columns.tolist()
        elif hasattr(dataset, 'data') and isinstance(dataset.data, pd.DataFrame):
            columns = dataset.data.columns.tolist()
        elif hasattr(dataset, 'features') and isinstance(dataset.features, pd.DataFrame):
            columns = dataset.features.columns.tolist()
        else:
            return detected

        # Check each column against sensitive keywords
        for col in columns:
            col_lower = col.lower().strip()

            # Exact match
            if col_lower in Experiment.SENSITIVE_ATTRIBUTE_KEYWORDS:
                detected.append(col)
                continue

            # Fuzzy match
            for keyword in Experiment.SENSITIVE_ATTRIBUTE_KEYWORDS:
                similarity = SequenceMatcher(None, col_lower, keyword).ratio()
                if similarity >= threshold:
                    detected.append(col)
                    break

        return detected
    
    def _initialize_components(self, dataset, test_size, random_state):
        """
        Initialize helper components of the experiment.
        
        Args:
            dataset: Source dataset
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        # Initialize helper components
        self.data_manager = DataManager(dataset, test_size, random_state)
        self.model_manager = ModelManager(dataset, self.experiment_type, self.verbose)
        self.model_evaluation = ModelEvaluation(self.experiment_type, self.metrics_calculator)
        
        # Prepare data
        self.data_manager.prepare_data()
        self.X_train, self.X_test = self.data_manager.X_train, self.data_manager.X_test
        self.y_train, self.y_test = self.data_manager.y_train, self.data_manager.y_test
        self.prob_train, self.prob_test = self.data_manager.prob_train, self.data_manager.prob_test
        
        # OTIMIZAÇÃO: Lazy loading de alternative_models
        # Apenas cria se houver testes que realmente precisam (hard_sample resilience)
        # Economiza 30-50s de overhead desnecessário
        self.alternative_models = self.model_manager.create_alternative_models(
            self.X_train, self.y_train,
            lazy=True  # Lazy loading: não treinar até ser necessário
        )
    
    def _initialize_test_runner(self):
        """Initialize the test runner component"""
        # Import here to avoid circular imports
        from deepbridge.core.experiment.test_runner import TestRunner
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
    
    def _process_initial_metrics(self):
        """Calculate initial metrics and standardize format"""
        # Calculate initial metrics - pass self as experiment to allow access to surrogate model
        self.initial_results = self.test_runner.run_initial_tests(experiment=self)
        
        # Process all models to ensure roc_auc is present and properly formatted
        if 'models' in self.initial_results:
            # Process primary model
            if 'primary_model' in self.initial_results['models']:
                self._standardize_metrics('primary_model', 
                              self.initial_results['models']['primary_model'],
                              self.dataset.model if hasattr(self.dataset, 'model') else None)
                
                # Add feature importance for primary model
                if hasattr(self.dataset, 'model') and self.dataset.model is not None:
                    self._calculate_model_feature_importance('primary_model', 
                                      self.initial_results['models']['primary_model'],
                                      self.dataset.model)
            
            # Process all alternative models
            for model_name, model_data in self.initial_results['models'].items():
                if model_name != 'primary_model':
                    model_obj = self.alternative_models.get(model_name)
                    self._standardize_metrics(model_name, model_data, model_obj)
                    
                    # Add feature importance for alternative models
                    if model_obj is not None:
                        self._calculate_model_feature_importance(model_name, model_data, model_obj)
    
    def __init__(
        self,
        dataset: 'DBDataset',
        experiment_type: str,
        test_size: float = 0.2,
        random_state: int = 42,
        config: t.Optional[dict] = None,
        auto_fit: t.Optional[bool] = None,
        tests: t.Optional[t.List[str]] = None,
        feature_subset: t.Optional[t.List[str]] = None,
        protected_attributes: t.Optional[t.List[str]] = None
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
                   "resilience", "hyperparameters", "fairness"]. Tests will only be executed when run_tests() is called.
            feature_subset: List of feature names to specifically test in the experiments.
                           In robustness tests, only these features will be perturbed while
                           all others remain unchanged. For other tests, only these features
                           will be analyzed in detail.
            protected_attributes: List of protected attributes for fairness testing (e.g., ['gender', 'race', 'age']).
                                 Required if 'fairness' is in tests list.

        Note:
            Initialization does NOT run any tests - it only calculates basic metrics for the models.
            To run tests, explicitly call experiment.run_tests("quick") after initialization.
        """
        if experiment_type not in self.VALID_TYPES:
            raise ValueError(f"experiment_type must be one of {self.VALID_TYPES}")
            
        # Set basic properties
        self.experiment_type = experiment_type
        self.dataset = dataset
        self.test_size = test_size
        self.random_state = random_state
        self.config = config or {}
        # Set verbosity from config
        self.verbose = config.get('verbose', False) if config else False
        # Set logger level based on verbose setting
        self.logger.set_verbose(self.verbose)
        self.tests = tests or []
        self.feature_subset = feature_subset

        # Auto-detect protected attributes if fairness testing requested but none provided
        if 'fairness' in self.tests and not protected_attributes:
            self.logger.info("Auto-detecting sensitive attributes for fairness testing...")
            detected = self.detect_sensitive_attributes(dataset)

            if detected:
                self.protected_attributes = detected
                self.logger.info(f"Auto-detected sensitive attributes: {detected}")
                self.logger.warning(
                    "Using auto-detected sensitive attributes. "
                    "For production use, explicitly specify protected_attributes."
                )
            else:
                raise ValueError(
                    "Cannot auto-detect sensitive attributes for fairness testing.\n"
                    "Please explicitly provide protected_attributes=['attr1', 'attr2', ...]\n"
                    "Example: protected_attributes=['gender', 'race', 'age']"
                )
        else:
            self.protected_attributes = protected_attributes

        self.logger.debug(f"Initializing experiment with type: {experiment_type}, tests: {tests}")
        
        # Automatically determine auto_fit value based on model presence
        if auto_fit is None:
            # If dataset has a model, auto_fit=False, otherwise auto_fit=True
            auto_fit = not (hasattr(dataset, 'model') and dataset.model is not None)
        self.auto_fit = auto_fit
        
        # Initialize metrics calculator based on experiment type
        if experiment_type == "binary_classification":
            self.metrics_calculator = Classification()
        elif experiment_type == "regression":
            self.metrics_calculator = Regression()
        else:
            # For forecasting or other types, default to None for now
            self.metrics_calculator = None
            
        # Initialize results storage and models
        self._results_data = {'train': {}, 'test': {}}
        self.distillation_model = None
        
        # Initialize components and prepare data
        self._initialize_components(dataset, test_size, random_state)
        
        # Initialize test runner
        self._initialize_test_runner()
        
        # Auto-fit if enabled and dataset has probabilities
        if self.auto_fit and hasattr(dataset, 'original_prob') and dataset.original_prob is not None:
            self._auto_fit_model()
        
        # Calculate initial metrics
        self._process_initial_metrics()
        
    
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
        # No action needed if no default model type is available
    
    def _create_distillation_model(self, distillation_method, student_model_type, student_params,
                                 temperature, alpha, use_probabilities, n_trials, validation_split):
        """
        Create and configure a distillation model.
        
        Args:
            distillation_method: Which distillation approach to use
            student_model_type: Type of model to use as student
            student_params: Parameters for student model
            temperature: Temperature for distillation
            alpha: Weighting factor for loss combination
            use_probabilities: Whether to use probabilities for distillation
            n_trials: Number of hyperparameter optimization trials
            validation_split: Proportion of data for validation
            
        Returns:
            Configured distillation model instance
        """
        return self.model_manager.create_distillation_model(
            distillation_method, 
            student_model_type, 
            student_params,
            temperature, 
            alpha, 
            use_probabilities, 
            n_trials, 
            validation_split
        )
        
    def _train_and_evaluate_model(self, model, verbose):
        """
        Train the model and evaluate on train and test sets.
        
        Args:
            model: The model to train and evaluate
            verbose: Whether to output training information
            
        Returns:
            Tuple of (train_metrics, test_metrics)
        """
        # Train the model
        model.fit(self.X_train, self.y_train, verbose=verbose)
        
        # Evaluate on train set
        train_metrics = self.model_evaluation.evaluate_distillation(
            model, 'train', 
            self.X_train, self.y_train, self.prob_train
        )
        
        # Evaluate on test set
        test_metrics = self.model_evaluation.evaluate_distillation(
            model, 'test', 
            self.X_test, self.y_test, self.prob_test
        )
        
        return train_metrics, test_metrics
    
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
            # Create distillation model
            self.distillation_model = self._create_distillation_model(
                distillation_method, 
                student_model_type, 
                student_params,
                temperature, 
                alpha, 
                use_probabilities, 
                n_trials, 
                validation_split
            )
            
            # Train and evaluate model
            train_metrics, test_metrics = self._train_and_evaluate_model(self.distillation_model, verbose)
            
            # Store results
            self._results_data['train'] = train_metrics['metrics']
            self._results_data['test'] = test_metrics['metrics']
            
            return self
        finally:
            # Restore logging state
            self._restore_logging(logging_state, verbose)

    def _calculate_model_feature_importance(self, model_name: str, model_data: dict, model_obj: t.Any) -> None:
        """
        Calculate feature importance using the model's built-in feature_importance_ attribute
        
        Args:
            model_name: Name of the model
            model_data: Model data dictionary where results will be stored
            model_obj: The actual model object
        """
        # Check if model has feature_importances_ or coef_ attribute
        if hasattr(model_obj, 'feature_importances_'):
            feature_importances = model_obj.feature_importances_
            feature_names = self.X_train.columns if hasattr(self.X_train, 'columns') else None
            
            # Create feature importance dictionary
            if feature_names is not None and len(feature_names) == len(feature_importances):
                importance_dict = dict(zip(feature_names, feature_importances))
                # Sort by importance value (descending)
                importance_dict = {k: float(v) for k, v in sorted(importance_dict.items(), 
                                                             key=lambda item: item[1], 
                                                             reverse=True)}
                model_data['feature_importance'] = importance_dict
                self.logger.debug(f"Added feature importance from feature_importances_ for {model_name}")
                
        # For linear models that use coef_ instead
        elif hasattr(model_obj, 'coef_'):
            coef = model_obj.coef_
            feature_names = self.X_train.columns if hasattr(self.X_train, 'columns') else None
            
            # Handle different coefficient shapes
            if len(coef.shape) > 1 and coef.shape[0] == 1:
                # For binary classifiers with shape (1, n_features)
                coef = coef[0]
            
            # Create feature importance dictionary using absolute values of coefficients
            if feature_names is not None and len(feature_names) == len(coef):
                # Use absolute values for linear model coefficients
                importance_values = np.abs(coef)
                importance_dict = dict(zip(feature_names, importance_values))
                # Sort by importance value (descending)
                importance_dict = {k: float(v) for k, v in sorted(importance_dict.items(), 
                                                             key=lambda item: item[1], 
                                                             reverse=True)}
                model_data['feature_importance'] = importance_dict
                self.logger.debug(f"Added feature importance from coef_ for {model_name}")

    def _standardize_metrics(self, model_name: str, model_data: dict, model_obj: t.Optional[t.Any] = None) -> None:
        """
        Standardize metrics format and naming conventions.
        Centralized method for metric processing to avoid redundancy.
        
        Args:
            model_name: Name of the model
            model_data: Model data dictionary 
            model_obj: The actual model object (optional)
        """
        # Skip if no metrics available
        if 'metrics' not in model_data:
            return
                
        metrics = model_data['metrics']
        
        # If we still don't have roc_auc, calculate it if possible
        if 'roc_auc' not in metrics and model_obj is not None:
            try:
                # Only calculate if model has predict_proba
                if hasattr(model_obj, 'predict_proba'):
                    from sklearn.metrics import roc_auc_score
                    y_prob = model_obj.predict_proba(self.X_test)

                    # Check if multiclass
                    n_classes = y_prob.shape[1] if len(y_prob.shape) > 1 else 1

                    if n_classes > 2:
                        # Multiclass: use ovr (one-vs-rest) strategy
                        roc_auc = float(roc_auc_score(self.y_test, y_prob, multi_class='ovr', average='macro'))
                        metrics['roc_auc'] = roc_auc
                        self.logger.debug(f"Calculated multiclass ROC AUC for {model_name}: {roc_auc}")
                    elif n_classes == 2:
                        # Binary: use second column (positive class probability)
                        roc_auc = float(roc_auc_score(self.y_test, y_prob[:, 1]))
                        metrics['roc_auc'] = roc_auc
                        self.logger.debug(f"Calculated ROC AUC for {model_name}: {roc_auc}")
            except Exception as e:
                self.logger.warning(f"Could not calculate ROC AUC for {model_name}: {str(e)}")
        
        # Standard metric name is 'roc_auc', convert any 'auc' to 'roc_auc' 
        if 'auc' in metrics:
            # Copy 'auc' value to 'roc_auc' if not already present
            if 'roc_auc' not in metrics:
                metrics['roc_auc'] = float(metrics['auc'])
            # Always remove 'auc' to maintain standardization
            del metrics['auc']
            
        # Ensure all metric values are float type for consistency
        for key, value in metrics.items():
            if value is not None and not isinstance(value, str):
                metrics[key] = float(value)

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
        # Import here to avoid circular imports
        from deepbridge.core.experiment.results import wrap_results
        
        # First, ensure we have initial metrics
        if not hasattr(self, 'initial_results') or not self.initial_results:
            # Pass self as experiment to allow access to surrogate model
            self.initial_results = self.test_runner.run_initial_tests(experiment=self)
            
        # Run the requested tests - pass self as experiment to allow access to surrogate model
        test_kwargs = kwargs.copy()
        test_kwargs['experiment'] = self
        test_results = self.test_runner.run_tests(config_name, **test_kwargs)
        self._test_results.update(test_results)
        
        # Make model feature importance available in test_results for the report
        if 'models' in self.initial_results:
            if 'primary_model' in self.initial_results['models']:
                if 'feature_importance' in self.initial_results['models']['primary_model']:
                    feature_importance = self.initial_results['models']['primary_model']['feature_importance']
                    self._test_results['model_feature_importance'] = feature_importance
        
        # Create a combined dictionary with initial_results as the first key
        # Using ordered dict to ensure initial_results is first
        from collections import OrderedDict
        
        combined_results = OrderedDict()
        combined_results['experiment_type'] = self.experiment_type
        combined_results['config'] = {'name': config_name, 'tests': self.tests}
        
        # Wrap the results in an ExperimentResult object with save_html method
        experiment_result = wrap_results(combined_results)
        
        # Modify the results structure in the ExperimentResult object
        # to have initial_results as the first key followed by test results
        experiment_result.results = OrderedDict()
        experiment_result.results['initial_results'] = self.initial_results
        
        # Add test results to the results dictionary
        for key, value in test_results.items():
            experiment_result.results[key] = value
        
        # Log testing completion
        self.logger.info(f"Tests completed with configuration '{config_name}'")
        self.logger.debug(f"Tests performed: {list(test_results.keys())}")
        
        # Store the experiment result object for later use
        self._experiment_result = experiment_result
        
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
        
    def get_feature_importance(self, model_name='primary_model'):
        """
        Get the feature importance for a specific model.
        
        Args:
            model_name: Name of the model (default: 'primary_model')
            
        Returns:
            dict: Feature importance dictionary or None if not available
            
        Raises:
            ValueError: If model not found or feature importance not calculated
        """
        # Check if we have initial results
        if not hasattr(self, 'initial_results') or not self.initial_results:
            raise ValueError("No model results available. Initialize the experiment first.")
            
        # Check if models dictionary exists in results
        if 'models' not in self.initial_results:
            raise ValueError("No models found in experiment results.")
            
        # Check if requested model exists
        if model_name not in self.initial_results['models']:
            available_models = list(self.initial_results['models'].keys())
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
            
        # Get model data
        model_data = self.initial_results['models'][model_name]
        
        # Check if feature importance is calculated
        if 'feature_importance' not in model_data:
            # If model is available, try to calculate feature importance now
            model_obj = None
            if model_name == 'primary_model' and hasattr(self.dataset, 'model'):
                model_obj = self.dataset.model
            elif model_name in self.alternative_models:
                model_obj = self.alternative_models[model_name]
                
            if model_obj is not None:
                self._calculate_model_feature_importance(model_name, model_data, model_obj)
            
            # Check again if feature importance is now available
            if 'feature_importance' not in model_data:
                raise ValueError(f"Feature importance not available for model '{model_name}'. The model may not support feature importance.")
                
        return model_data['feature_importance']

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

    def save_html(self, test_type: str, file_path: str, model_name: str = None, report_type: str = "interactive") -> str:
        """
        Generate and save an HTML report for the specified test.

        Args:
            test_type: Type of test ('robustness', 'uncertainty', 'resilience', 'hyperparameter')
            file_path: Path where the HTML report will be saved (relative or absolute)
            model_name: Name of the model for display in the report. If None, uses dataset name if available
            report_type: Type of report to generate ('interactive' with Plotly or 'static' with Matplotlib). Default: 'interactive'

        Returns:
            Path to the generated report file

        Raises:
            ValueError: If test results not found or report generation fails
        """
        # Check if we have the test results
        if not hasattr(self, '_test_results') or not self._test_results:
            raise ValueError("No test results available. Run tests first with experiment.run_tests()")
        
        # Get results for the specified test
        test_results = self._test_results.get(test_type.lower())
        if not test_results:
            raise ValueError(f"No {test_type} test results found. Run the test first with experiment.run_tests()")
        
        # Import here to avoid circular imports
        from deepbridge.core.experiment.results import wrap_results, ExperimentResult
        import os
        
        # Ensure file_path is absolute
        if not os.path.isabs(file_path):
            file_path = os.path.abspath(file_path)
        
        # If we already have an ExperimentResult object from run_tests(), use that
        if hasattr(self, '_experiment_result') and isinstance(self._experiment_result, ExperimentResult):
            return self._experiment_result.save_html(test_type, file_path, model_name or self._get_model_name(), report_type=report_type)
        
        # Otherwise, create a new experiment result object
        combined_results = {
            'experiment_type': self.experiment_type,
            'config': {'tests': self.tests},
            # Add the selected test type
            test_type.lower(): test_results
        }
        
        # Create an ExperimentResult and use its save_html method
        experiment_result = wrap_results(combined_results)
        return experiment_result.save_html(test_type, file_path, model_name or self._get_model_name(), report_type=report_type)
    
    def _get_model_name(self) -> str:
        """Get a displayable model name"""
        if hasattr(self.dataset, 'name') and self.dataset.name:
            return f"{self.dataset.name} Model"
        elif hasattr(self.dataset, 'model') and self.dataset.model is not None:
            return type(self.dataset.model).__name__
        else:
            return "Model"

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

    def run_fairness_tests(self, config: str = 'full') -> 'FairnessResult':
        """
        Run fairness tests on the model.

        This method requires protected_attributes to be provided during
        Experiment initialization.

        Parameters:
        -----------
        config : str, default='full'
            Configuration level: 'quick', 'medium', or 'full'

        Returns:
        --------
        FairnessResult : Object with fairness test results

        Raises:
        -------
        ValueError
            If protected_attributes were not provided during initialization

        Example:
        --------
        >>> experiment = Experiment(
        ...     dataset=dataset,
        ...     experiment_type="binary_classification",
        ...     tests=["robustness", "fairness"],
        ...     protected_attributes=['gender', 'race']
        ... )
        >>> fairness_results = experiment.run_fairness_tests(config='full')
        >>> print(f"Fairness Score: {fairness_results.overall_fairness_score:.3f}")
        >>> print(f"Critical Issues: {len(fairness_results.critical_issues)}")
        """
        # Validate that protected_attributes were provided
        if not self.protected_attributes:
            raise ValueError(
                "Cannot run fairness tests: no protected_attributes provided.\n"
                "Initialize Experiment with protected_attributes=['attr1', 'attr2', ...] "
                "to enable fairness testing."
            )

        # Import FairnessSuite
        from deepbridge.validation.wrappers.fairness_suite import FairnessSuite

        # Initialize fairness suite
        fairness_suite = FairnessSuite(
            dataset=self.dataset,
            protected_attributes=self.protected_attributes,
            verbose=self.verbose
        )

        # Run fairness tests
        results = fairness_suite.config(config).run()

        # Check if results is already a FairnessResult
        from deepbridge.core.experiment.results import FairnessResult

        if isinstance(results, FairnessResult):
            fairness_result = results
        else:
            # Wrap dict results in FairnessResult
            fairness_result = FairnessResult(results)

        # Store in test results
        self._test_results['fairness'] = fairness_result

        self.logger.info(f"Fairness tests completed with config '{config}'")
        self.logger.info(f"Fairness Score: {fairness_result.overall_fairness_score:.3f}")
        if fairness_result.critical_issues:
            self.logger.warning(f"Found {len(fairness_result.critical_issues)} critical fairness issues")

        return fairness_result

    # Backward compatibility properties removed in this refactoring

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