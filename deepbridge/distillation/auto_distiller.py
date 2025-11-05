import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any, Tuple, Union
import logging

from deepbridge.utils.model_registry import ModelType
from deepbridge.core.db_data import DBDataset

# Import components from auto submodule
from deepbridge.config.settings import DistillationConfig
from deepbridge.distillation.experiment_runner import ExperimentRunner
from deepbridge.metrics.evaluator import MetricsEvaluator

# Import HPM components
from deepbridge.distillation.techniques.hpm import HPMDistiller, HPMConfig

logger = logging.getLogger(__name__)

class AutoDistiller:
    """
    Automated Knowledge Distillation tool for model compression.
    
    This class automates the process of knowledge distillation by testing
    multiple model types, temperatures, and alpha values to find the optimal 
    configuration for a given dataset.
    
    The implementation is organized to separate concerns:
    - Configuration management
    - Experiment execution
    - Metrics evaluation
    - Visualization
    - Reporting
    """
    
    def __init__(
        self,
        dataset: DBDataset,
        output_dir: str = "distillation_results",
        test_size: float = 0.2,
        random_state: int = 42,
        n_trials: int = 10,
        validation_split: float = 0.2,
        verbose: bool = False,
        method: str = 'auto'
    ):
        """
        Initialize the AutoDistiller.

        Args:
            dataset: DBDataset instance containing features, target, and probabilities
            output_dir: Directory to save results and visualizations
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            n_trials: Number of Optuna trials for hyperparameter optimization
            validation_split: Fraction of data to use for validation during optimization
            verbose: Whether to show progress messages
            method: Distillation method ('auto', 'legacy', 'hpm', 'hybrid')
                - 'auto': Automatically choose best method based on dataset
                - 'legacy': Use traditional grid search approach
                - 'hpm': Use new HPM-KD approach
                - 'hybrid': Run both and compare results
        """
        self.dataset = dataset
        self.method = method

        # Determine actual method if 'auto'
        if method == 'auto':
            self.method = self._choose_best_method(dataset)
            if verbose:
                logger.info(f"Auto-selected method: {self.method}")

        # Initialize based on method
        if self.method == 'hpm':
            # Use HPM distiller
            self._init_hpm(dataset, output_dir, test_size, random_state,
                          n_trials, validation_split, verbose)
        elif self.method == 'hybrid':
            # Initialize both
            self._init_hybrid(dataset, output_dir, test_size, random_state,
                            n_trials, validation_split, verbose)
        else:  # 'legacy'
            # Use traditional approach
            self._init_legacy(dataset, output_dir, test_size, random_state,
                             n_trials, validation_split, verbose)

        # Initialize cache for original metrics
        self._original_metrics_cache = None

    def _init_legacy(self, dataset, output_dir, test_size, random_state,
                    n_trials, validation_split, verbose):
        """Initialize legacy distillation components."""
        # Initialize configuration
        self.config = DistillationConfig(
            output_dir=output_dir,
            test_size=test_size,
            random_state=random_state,
            n_trials=n_trials,
            validation_split=validation_split,
            verbose=verbose
        )

        # Initialize experiment runner
        self.experiment_runner = ExperimentRunner(
            dataset=dataset,
            config=self.config
        )

        # Other components will be initialized after experiments are run
        self.metrics_evaluator = None
        self.visualizer = None
        self.report_generator = None
        self.results_df = None

    def _init_hpm(self, dataset, output_dir, test_size, random_state,
                 n_trials, validation_split, verbose):
        """Initialize HPM distillation components."""
        # Create HPM configuration
        hpm_config = HPMConfig(
            max_configs=16,  # Reduce from 64 to 16
            n_trials=max(3, n_trials // 3),  # Reduce trials with warm start
            validation_split=validation_split,
            random_state=random_state,
            verbose=verbose,
            use_parallel=False,  # Disabled to avoid pickle issues
            use_cache=True,
            use_progressive=True,
            use_multi_teacher=False,  # Disabled until models train successfully
            use_adaptive_temperature=True
        )

        # Initialize HPM distiller
        self.hpm_distiller = HPMDistiller(config=hpm_config)

        # Create compatibility layer
        self.config = DistillationConfig(
            output_dir=output_dir,
            test_size=test_size,
            random_state=random_state,
            n_trials=n_trials,
            validation_split=validation_split,
            verbose=verbose
        )

        # Compatibility attributes
        self.experiment_runner = None  # HPM handles this internally
        self.metrics_evaluator = None
        self.results_df = None

    def _init_hybrid(self, dataset, output_dir, test_size, random_state,
                    n_trials, validation_split, verbose):
        """Initialize both legacy and HPM components."""
        # Initialize legacy
        self._init_legacy(dataset, output_dir, test_size, random_state,
                         n_trials, validation_split, verbose)

        # Also initialize HPM
        hpm_config = HPMConfig(
            max_configs=8,  # Even fewer for hybrid mode
            n_trials=max(2, n_trials // 4),
            validation_split=validation_split,
            random_state=random_state,
            verbose=verbose
        )
        self.hpm_distiller = HPMDistiller(config=hpm_config)

    def _choose_best_method(self, dataset: DBDataset) -> str:
        """
        Choose the best distillation method based on dataset characteristics.

        Args:
            dataset: The dataset to analyze

        Returns:
            Best method name ('legacy' or 'hpm')
        """
        # Get dataset size
        n_samples = len(dataset.X)
        n_features = dataset.X.shape[1] if hasattr(dataset.X, 'shape') else 10

        # Heuristics for method selection
        if n_samples > 10000 or n_features > 50:
            # Large dataset - HPM more efficient
            return 'hpm'
        elif n_samples < 1000:
            # Small dataset - legacy might be sufficient
            return 'legacy'
        else:
            # Medium dataset - use HPM for better quality
            return 'hpm'
    
    def customize_config(
        self,
        model_types: Optional[List[ModelType]] = None,
        temperatures: Optional[List[float]] = None,
        alphas: Optional[List[float]] = None,
        distillation_method: Optional[str] = None
    ):
        """
        Customize the configuration for distillation experiments.
        
        Args:
            model_types: List of ModelType to test (defaults to standard list if None)
            temperatures: List of temperature values to test (defaults to [0.5, 1.0, 2.0] if None)
            alphas: List of alpha values to test (defaults to [0.3, 0.5, 0.7] if None)
            distillation_method: Method to use for distillation ('surrogate' or 'knowledge_distillation')
        """
        self.config.customize(
            model_types=model_types,
            temperatures=temperatures,
            alphas=alphas,
            distillation_method=distillation_method
        )
    
    def original_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate metrics for the original model after train/test split.
        
        Returns:
            Dictionary containing metrics for both train and test sets
        """
        # Check if we've already calculated these metrics
        if self._original_metrics_cache is not None:
            return self._original_metrics_cache

        # For HPM method, calculate metrics from teacher predictions
        if hasattr(self, 'hpm_distiller') and self.experiment_runner is None:
            # Calculate metrics from teacher predictions if available
            if hasattr(self.dataset, 'train_predictions') and hasattr(self.dataset, 'test_predictions'):
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                import numpy as np

                results = {'train': {}, 'test': {}}

                # Get teacher predictions and labels
                train_probs = self.dataset.train_predictions
                test_probs = self.dataset.test_predictions
                y_train = self.dataset.get_target_data('train')
                y_test = self.dataset.get_target_data('test')

                # Check if we have predictions
                if train_probs is None or test_probs is None:
                    return {'train': {}, 'test': {}}

                # Convert probabilities to predictions
                # Assuming binary classification with proba1 being positive class
                if train_probs is not None and 'proba1' in train_probs.columns:
                    train_preds = (train_probs['proba1'].values >= 0.5).astype(int)
                    test_preds = (test_probs['proba1'].values >= 0.5).astype(int)
                    train_proba_pos = train_probs['proba1'].values
                    test_proba_pos = test_probs['proba1'].values
                else:
                    # Fallback to argmax if no proba1
                    train_preds = np.argmax(train_probs.values, axis=1)
                    test_preds = np.argmax(test_probs.values, axis=1)
                    train_proba_pos = train_probs.iloc[:, 1].values if train_probs.shape[1] > 1 else train_probs.iloc[:, 0].values
                    test_proba_pos = test_probs.iloc[:, 1].values if test_probs.shape[1] > 1 else test_probs.iloc[:, 0].values

                # Calculate train metrics
                results['train']['accuracy'] = accuracy_score(y_train, train_preds)
                results['train']['precision'] = precision_score(y_train, train_preds, average='weighted', zero_division=0)
                results['train']['recall'] = recall_score(y_train, train_preds, average='weighted', zero_division=0)
                results['train']['f1_score'] = f1_score(y_train, train_preds, average='weighted', zero_division=0)
                try:
                    results['train']['auc_roc'] = roc_auc_score(y_train, train_proba_pos)
                except:
                    results['train']['auc_roc'] = 0.5

                # Calculate test metrics
                results['test']['accuracy'] = accuracy_score(y_test, test_preds)
                results['test']['precision'] = precision_score(y_test, test_preds, average='weighted', zero_division=0)
                results['test']['recall'] = recall_score(y_test, test_preds, average='weighted', zero_division=0)
                results['test']['f1_score'] = f1_score(y_test, test_preds, average='weighted', zero_division=0)
                try:
                    results['test']['auc_roc'] = roc_auc_score(y_test, test_proba_pos)
                except:
                    results['test']['auc_roc'] = 0.5

                # Cache the results
                self._original_metrics_cache = results
                return results
            else:
                # No teacher predictions available
                return {'train': {}, 'test': {}}

        from deepbridge.metrics.classification import Classification

        metrics_calculator = Classification()
        results = {'train': {}, 'test': {}}

        # Get split data from experiment
        if self.experiment_runner is None:
            return {}
        experiment = self.experiment_runner.experiment
        
        # Train set metrics
        if experiment.prob_train is not None:
            # Get positive class probabilities
            train_probs = self._extract_positive_class_probs(experiment.prob_train)
                
            # Convert probabilities to binary predictions (threshold 0.5)
            train_preds = (train_probs >= 0.5).astype(int)
            
            # Calculate metrics
            train_metrics = metrics_calculator.calculate_metrics(
                y_true=experiment.y_train,
                y_pred=train_preds,
                y_prob=train_probs
            )
            results['train'] = train_metrics
            
        # Test set metrics
        if experiment.prob_test is not None:
            # Get positive class probabilities
            test_probs = self._extract_positive_class_probs(experiment.prob_test)
                
            # Convert probabilities to binary predictions (threshold 0.5)
            test_preds = (test_probs >= 0.5).astype(int)
            
            # Calculate metrics
            test_metrics = metrics_calculator.calculate_metrics(
                y_true=experiment.y_test,
                y_pred=test_preds,
                y_prob=test_probs
            )
            results['test'] = test_metrics
        
        # Print a summary of the original model metrics if verbose is true
        if self.config.verbose:
            print("\n=== Original Model Metrics ===")
            if results['train']:
                print("Train metrics:")
                for metric, value in results['train'].items():
                    if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']:
                        print(f"  {metric}: {value:.4f}")
            
            if results['test']:
                print("\nTest metrics:")
                for metric, value in results['test'].items():
                    if metric in ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']:
                        print(f"  {metric}: {value:.4f}")
            print("=============================\n")
        
        # Cache the results
        self._original_metrics_cache = results
        
        return results
    
    def _extract_positive_class_probs(self, probs):
        """
        Helper method to extract positive class probabilities from different formats.
        
        Args:
            probs: Probabilities in various formats (DataFrame, Series, or numpy array)
            
        Returns:
            numpy array of positive class probabilities
        """
        if isinstance(probs, pd.DataFrame):
            # Try to find the right column
            if 'prob_class_1' in probs.columns:
                return probs['prob_class_1'].values
            elif 'prob1' in probs.columns:
                return probs['prob1'].values
            elif len(probs.columns) == 2:
                # Assume second column is positive class in binary classification
                return probs.iloc[:, 1].values
            else:
                # Fallback to last column
                return probs.iloc[:, -1].values
        elif isinstance(probs, pd.Series):
            return probs.values
        elif isinstance(probs, np.ndarray):
            # For 2D arrays, assume second column (index 1) is positive class
            if len(probs.shape) > 1 and probs.shape[1] > 1:
                return probs[:, 1]
            return probs
        else:
            # Fallback with warning
            if self.config.verbose:
                print(f"Warning: Unrecognized probability format: {type(probs)}")
            return np.array(probs)
    
    def run(self, use_probabilities: bool = True, verbose_output: bool = False) -> pd.DataFrame:
        """
        Run the automated distillation process.
        
        Args:
            use_probabilities: Whether to use pre-calculated probabilities or teacher model
            verbose_output: Whether to display detailed output during training (default: False)
        
        Returns:
            DataFrame containing results for all configurations
        """
        # Store original verbose setting
        original_verbose = self.config.verbose
        
        if not verbose_output:
            # Temporarily redirect stdout to suppress output
            import sys
            import os
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
        
        try:
            # Check if using HPM method
            if hasattr(self, 'hpm_distiller') and self.experiment_runner is None:
                # Run HPM distillation
                # Extract data from dataset
                X_train = self.dataset.get_feature_data('train')
                y_train = self.dataset.get_target_data('train')
                X_test = self.dataset.get_feature_data('test')
                y_test = self.dataset.get_target_data('test')

                # Get teacher probabilities if available
                teacher_probs = None
                if use_probabilities and hasattr(self.dataset, 'train_predictions'):
                    teacher_probs = self.dataset.train_predictions

                # Fit HPM distiller
                self.hpm_distiller.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_test,
                    y_val=y_test,
                    teacher_probs=teacher_probs
                )

                # Get results from HPM distiller
                # For now, create a minimal results DataFrame for compatibility
                self.results_df = self._create_hpm_results_df()
                # HPM handles its own saving
            else:
                # Run legacy experiments
                self.results_df = self.experiment_runner.run_experiments(
                    use_probabilities=use_probabilities,
                    distillation_method=self.config.distillation_method
                )

                # Save results
                self.experiment_runner.save_results()
            
            # Initialize components that depend on results
            self._initialize_analysis_components()
            
            # Visualization and report generation have been removed

            # Add best_model method to the DataFrame
            self.results_df.best_model = lambda metric='test_ks_statistic', minimize=False: self.best_model(metric, minimize)

        finally:
            if not verbose_output:
                # Restore stdout
                sys.stdout.close()
                sys.stdout = original_stdout

                # Print minimal completion message
                if original_verbose:
                    print(f"Completed distillation experiments. Tested {self.config.get_total_configurations()} configurations.")

        return self.results_df
    
    def _create_hpm_results_df(self):
        """Create a results DataFrame from HPM distiller results."""
        import pandas as pd
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, log_loss
        import numpy as np

        # Try to extract real metrics if model is available
        metrics_dict = {
            'model_type': ['HPM_ENSEMBLE'],
            'temperature': [self.hpm_distiller.config.initial_temperature if hasattr(self.hpm_distiller.config, 'initial_temperature') else 3.0],
            'alpha': [0.5],
            'training_time': [self.hpm_distiller.total_time if hasattr(self.hpm_distiller, 'total_time') else 0.0],
        }

        # Try to calculate real metrics if model and data are available
        if hasattr(self.hpm_distiller, 'best_model') and self.hpm_distiller.best_model is not None:
            try:
                # Get test data
                X_test = self.dataset.get_feature_data('test')
                y_test = self.dataset.get_target_data('test')
                X_train = self.dataset.get_feature_data('train')
                y_train = self.dataset.get_target_data('train')

                # Make predictions
                model = self.hpm_distiller.best_model
                if hasattr(model, 'predict'):
                    # Test predictions
                    y_pred_test = model.predict(X_test)
                    y_pred_train = model.predict(X_train)

                    # Calculate test metrics
                    metrics_dict['test_accuracy'] = [accuracy_score(y_test, y_pred_test)]
                    metrics_dict['test_precision'] = [precision_score(y_test, y_pred_test, average='weighted', zero_division=0)]
                    metrics_dict['test_recall'] = [recall_score(y_test, y_pred_test, average='weighted', zero_division=0)]
                    metrics_dict['test_f1_score'] = [f1_score(y_test, y_pred_test, average='weighted', zero_division=0)]

                    # Calculate train metrics
                    metrics_dict['train_accuracy'] = [accuracy_score(y_train, y_pred_train)]
                    metrics_dict['train_precision'] = [precision_score(y_train, y_pred_train, average='weighted', zero_division=0)]
                    metrics_dict['train_recall'] = [recall_score(y_train, y_pred_train, average='weighted', zero_division=0)]
                    metrics_dict['train_f1_score'] = [f1_score(y_train, y_pred_train, average='weighted', zero_division=0)]

                    # Calculate probability-based metrics if available
                    if hasattr(model, 'predict_proba'):
                        y_proba_test = model.predict_proba(X_test)
                        y_proba_train = model.predict_proba(X_train)

                        if y_proba_test.shape[1] == 2:
                            # Binary classification metrics

                            # AUC-ROC
                            try:
                                metrics_dict['test_auc_roc'] = [roc_auc_score(y_test, y_proba_test[:, 1])]
                                metrics_dict['train_auc_roc'] = [roc_auc_score(y_train, y_proba_train[:, 1])]
                            except:
                                metrics_dict['test_auc_roc'] = [0.5]
                                metrics_dict['train_auc_roc'] = [0.5]

                            # Log Loss
                            try:
                                metrics_dict['test_log_loss'] = [log_loss(y_test, y_proba_test)]
                                metrics_dict['train_log_loss'] = [log_loss(y_train, y_proba_train)]
                            except:
                                metrics_dict['test_log_loss'] = [1.0]
                                metrics_dict['train_log_loss'] = [1.0]

                            # KS statistic
                            from scipy.stats import ks_2samp
                            proba_pos = y_proba_test[y_test == 1, 1]
                            proba_neg = y_proba_test[y_test == 0, 1]
                            if len(proba_pos) > 0 and len(proba_neg) > 0:
                                ks_stat, ks_pvalue = ks_2samp(proba_pos, proba_neg)
                                metrics_dict['test_ks_statistic'] = [ks_stat]
                                metrics_dict['test_ks_pvalue'] = [ks_pvalue]
                            else:
                                metrics_dict['test_ks_statistic'] = [0.0]
                                metrics_dict['test_ks_pvalue'] = [1.0]

                            # Gini coefficient
                            try:
                                gini = 2 * metrics_dict['test_auc_roc'][0] - 1
                                metrics_dict['test_gini'] = [gini]
                            except:
                                metrics_dict['test_gini'] = [0.0]
                        else:
                            # Multi-class - use default values for binary-only metrics
                            metrics_dict['test_ks_statistic'] = [0.0]
                            metrics_dict['test_ks_pvalue'] = [1.0]
                            metrics_dict['test_gini'] = [0.0]
                            metrics_dict['test_auc_roc'] = [0.5]
                            metrics_dict['train_auc_roc'] = [0.5]
                    else:
                        # No predict_proba - use default values
                        metrics_dict['test_ks_statistic'] = [0.0]
                        metrics_dict['test_ks_pvalue'] = [1.0]
                        metrics_dict['test_auc_roc'] = [0.5]
                        metrics_dict['train_auc_roc'] = [0.5]
                        metrics_dict['test_log_loss'] = [1.0]
                        metrics_dict['train_log_loss'] = [1.0]
                        metrics_dict['test_gini'] = [0.0]
                else:
                    # Model can't predict - fail instead of using fallback
                    raise ValueError("Model does not support prediction. Cannot generate report without real metrics.")

            except Exception as e:
                # Fail instead of using fallback values
                raise ValueError(f"Failed to calculate real metrics: {e}. Cannot generate report without real metrics.")
        else:
            # No model available - fail instead of using fallback
            raise ValueError("No trained model available. Cannot generate report without real metrics.")

        df = pd.DataFrame(metrics_dict)

        # Add best_model method to DataFrame
        df.best_model = lambda metric='test_ks_statistic', minimize=False: self.best_model(metric, minimize)

        return df

    # Removed _add_placeholder_metrics - no fallback values allowed

    def _initialize_analysis_components(self):
        """Initialize components for analysis after experiments are run."""
        # Initialize metrics evaluator
        self.metrics_evaluator = MetricsEvaluator(
            results_df=self.results_df,
            config=self.config
        )
        
        # Visualizer and report generator have been removed
        self.visualizer = None
        self.report_generator = None
        
    def _ensure_components_initialized(self):
        """Ensure that analysis components are initialized if results are available."""
        if self.results_df is not None and self.metrics_evaluator is None:
            self._initialize_analysis_components()
    
    def find_best_model(self, metric: str = 'test_accuracy', minimize: bool = False) -> Dict:
        """
        Find the best model configuration based on a specific metric.
        
        Args:
            metric: Metric to use for finding the best model (default: 'test_accuracy')
            minimize: Whether the metric should be minimized (default: False)
        
        Returns:
            Dictionary containing the best model configuration
        """
        if self.results_df is None:
            raise ValueError("No results available. Run the distillation process first.")
            
        self._ensure_components_initialized()
        best_config = self.metrics_evaluator.find_best_model(metric=metric, minimize=minimize)
        
        # Convert model_type from string to ModelType enum if needed
        if 'model_type' in best_config and isinstance(best_config['model_type'], str):
            model_type_str = best_config['model_type']
            for model_type in ModelType:
                if model_type.name == model_type_str:
                    best_config['model_type'] = model_type
                    break
        
        return best_config
        
    def get_trained_model(
        self, 
        model_type: Union[ModelType, str], 
        temperature: float, 
        alpha: float,
        distillation_method: Optional[str] = None
    ):
        """
        Get a trained model with specific configuration.
        
        Args:
            model_type: Type of model to train (ModelType enum or string)
            temperature: Temperature parameter
            alpha: Alpha parameter
            distillation_method: Method to use for distillation 
                                (uses config value if None)
        
        Returns:
            Trained distillation model
        """
        # Use o método de distilação da configuração se não for especificado
        method = distillation_method or self.config.distillation_method
        
        # Convert string to ModelType if needed
        if isinstance(model_type, str):
            try:
                model_type = ModelType[model_type]
            except KeyError:
                # Special cases for common model types
                if model_type == 'GBM':
                    model_type = ModelType.GBM
                elif model_type == 'XGB':
                    model_type = ModelType.XGB
                else:
                    raise ValueError(f"Unsupported model type string: {model_type}")
        
        return self.experiment_runner.get_trained_model(
            model_type=model_type,
            temperature=temperature,
            alpha=alpha,
            distillation_method=method
        )
    
    def best_model(self, metric: str = 'test_ks_statistic', minimize: bool = False):
        """
        Get the best trained model based on a specific metric.

        Args:
            metric: Metric to use for finding the best model. Available metrics:
                   - test_accuracy, test_precision, test_recall, test_f1_score
                   - test_auc_roc, test_auc_pr, test_kl_divergence
                   - test_ks_statistic (default), test_ks_pvalue, test_r2_score
                   Also available with 'train_' prefix for training metrics.
            minimize: Whether the metric should be minimized (default: False).
                     Set to True for metrics like kl_divergence, log_loss.

        Returns:
            Trained distillation model ready for predictions

        Example:
            >>> distiller = AutoDistiller(dataset)
            >>> results = distiller.run()
            >>> model = distiller.best_model(metric='test_ks_statistic')
            >>> predictions = model.predict(X_new)
        """
        # For HPM method, return the best model directly
        if hasattr(self, 'hpm_distiller') and self.experiment_runner is None:
            if hasattr(self.hpm_distiller, 'best_model') and self.hpm_distiller.best_model is not None:
                if self.config.verbose:
                    print(f"\n=== HPM Best Model Selected ===")
                    print(f"Model Type: {type(self.hpm_distiller.best_model).__name__}")
                    print("===========================\n")
                return self.hpm_distiller.best_model
            else:
                # If no best model, try to return any successful model
                logger.warning("HPM best model not found")
                return None

        # Legacy path for non-HPM methods
        if self.results_df is None:
            raise ValueError("No results available. Run the distillation process first with distiller.run()")

        # Validate metric name
        valid_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'auc_roc', 'auc_pr', 'kl_divergence',
            'ks_statistic', 'ks_pvalue', 'r2_score', 'log_loss'
        ]

        # Add test_ or train_ prefix if not present
        if not metric.startswith(('test_', 'train_')):
            metric = f'test_{metric}'

        # Check if metric exists in results
        if metric not in self.results_df.columns:
            available_metrics = [col for col in self.results_df.columns
                               if col.startswith(('test_', 'train_')) and
                               any(vm in col for vm in valid_metrics)]
            raise ValueError(f"Metric '{metric}' not found in results. "
                           f"Available metrics: {', '.join(available_metrics)}")

        # Find the best configuration based on the metric
        best_config = self.find_best_model(metric=metric, minimize=minimize)

        # Get and return the trained model with the best configuration
        best_model = self.get_trained_model(
            model_type=best_config['model_type'],
            temperature=best_config['temperature'],
            alpha=best_config['alpha']
        )

        # Print information about the selected model if verbose
        if self.config.verbose:
            print(f"\n=== Best Model Selected ===")
            print(f"Metric: {metric} = {best_config.get(metric, 'N/A'):.4f}")
            print(f"Model Type: {best_config['model_type']}")
            print(f"Temperature: {best_config['temperature']}")
            print(f"Alpha: {best_config['alpha']}")
            print("===========================\n")

        return best_model

    def save_best_model(self, metric: str = 'test_accuracy', minimize: bool = False,
                        file_path: str = 'best_distilled_model.pkl') -> str:
        """
        Find the best model and save it as a pickle file using joblib.
        
        Args:
            metric: Metric to use for finding the best model (default: 'test_accuracy')
            minimize: Whether the metric should be minimized (default: False)
            file_path: Path where to save the model (default: 'best_distilled_model.pkl')
        
        Returns:
            String containing the path where the model was saved
        """
        import joblib
        from pathlib import Path
        
        # Find the best configuration
        best_config = self.find_best_model(metric=metric, minimize=minimize)
        
        # Get the trained model with this configuration
        best_model = self.get_trained_model(
            model_type=best_config['model_type'],
            temperature=best_config['temperature'],
            alpha=best_config['alpha']
        )
        
        # Ensure the directory exists
        output_file = Path(file_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model using joblib
        joblib.dump(best_model, output_file)
        
        if self.config.verbose:
            print(f"Best model saved to: {output_file}")
        
        return str(output_file)
    
    def generate_report(self, output_path: Optional[str] = None, report_type: str = 'interactive') -> str:
        """
        Generate HTML report for distillation results.

        Args:
            output_path: Path to save the report (default: output_dir/distillation_report.html)
            report_type: Type of report to generate ('interactive' or 'static')

        Returns:
            Path to the generated report file

        Raises:
            ValueError: If no results are available
        """
        if self.results_df is None:
            raise ValueError("No results available. Run the distillation process first.")

        # Set default output path if not provided
        if output_path is None:
            output_path = os.path.join(
                self.config.output_dir,
                f'distillation_report_{report_type}.html'
            )

        # Prepare data for the report
        try:
            best_model = self.find_best_model()
            # find_best_model returns a dict, not a DataFrame
            best_model_dict = best_model if best_model else {}
        except Exception:
            best_model_dict = {}

        # Prepare config based on method
        if hasattr(self, 'hpm_distiller') and self.experiment_runner is None:
            # HPM config
            config_data = {
                'method': 'HPM-KD',
                'max_configs': getattr(self.hpm_distiller.config, 'max_configs', 16),
                'use_progressive': getattr(self.hpm_distiller.config, 'use_progressive', True),
                'use_cache': getattr(self.hpm_distiller.config, 'use_cache', True),
                'use_parallel': getattr(self.hpm_distiller.config, 'use_parallel', False),
            }
        else:
            # Legacy config
            model_types_list = []
            if hasattr(self.config, 'model_types'):
                model_types = self.config.model_types
                if model_types:
                    # Check if model_types are strings or objects with name attribute
                    if isinstance(model_types[0], str):
                        model_types_list = model_types
                    else:
                        model_types_list = [mt.name if hasattr(mt, 'name') else str(mt) for mt in model_types]

            config_data = {
                'model_types': model_types_list,
                'temperatures': self.config.temperatures if hasattr(self.config, 'temperatures') else [],
                'alphas': self.config.alphas if hasattr(self.config, 'alphas') else [],
                'n_trials': self.config.n_trials if hasattr(self.config, 'n_trials') else 0,
                'test_size': self.config.test_size if hasattr(self.config, 'test_size') else 0.2,
                'validation_split': self.config.validation_split if hasattr(self.config, 'validation_split') else 0.2
            }

        report_data = {
            'results': self.results_df,
            'original_metrics': self.original_metrics(),
            'best_model': best_model_dict,
            'config': config_data,
            'dataset': self.dataset  # Pass dataset for distribution analysis
        }

        # Import renderers based on report type
        # Get templates directory
        import os
        from pathlib import Path
        deepbridge_dir = Path(__file__).parent.parent
        templates_dir = os.path.join(deepbridge_dir, 'templates')

        if report_type == 'interactive':
            from deepbridge.core.experiment.report.renderers.distillation_renderer import DistillationRenderer
            from deepbridge.core.experiment.report.template_manager import TemplateManager
            from deepbridge.core.experiment.report.asset_manager import AssetManager

            template_manager = TemplateManager(templates_dir)
            asset_manager = AssetManager(templates_dir)
            renderer = DistillationRenderer(template_manager, asset_manager)
        elif report_type == 'static':
            from deepbridge.core.experiment.report.renderers.static.static_distillation_renderer import StaticDistillationRenderer
            from deepbridge.core.experiment.report.template_manager import TemplateManager
            from deepbridge.core.experiment.report.asset_manager import AssetManager

            template_manager = TemplateManager(templates_dir)
            asset_manager = AssetManager(templates_dir)
            renderer = StaticDistillationRenderer(template_manager, asset_manager)
        else:
            raise ValueError(f"Invalid report_type: {report_type}. Must be 'interactive' or 'static'")

        # Generate the report
        report_path = renderer.render(
            results=report_data,
            file_path=output_path,
            model_name="Knowledge Distillation",
            report_type=report_type
        )

        if self.config.verbose:
            print(f"\n✅ Report generated successfully: {report_path}")

        return report_path
    
    def generate_summary(self) -> str:
        """
        This method has been deprecated as reporting functionality has been removed.
        
        Raises:
            NotImplementedError: Always raises this exception
        """
        raise NotImplementedError("Reporting functionality has been removed from this version.")
    
    def create_visualizations(self):
        """
        This method has been deprecated as visualization functionality has been removed.
        
        Raises:
            NotImplementedError: Always raises this exception
        """
        raise NotImplementedError("Visualization functionality has been removed from this version.")
    
    def compare_models(self, best_metric: str = 'test_accuracy', minimize: bool = False) -> pd.DataFrame:
        """
        Compare the original model with the best distilled model.
        
        Args:
            best_metric: Metric to use for finding the best distilled model
            minimize: Whether the metric should be minimized
            
        Returns:
            DataFrame with comparison metrics
        """
        if self.results_df is None:
            raise ValueError("No results available. Run the distillation process first.")
        
        # Get original model metrics
        original_metrics = self.original_metrics()
        
        # Find best distilled model
        best_config = self.find_best_model(metric=best_metric, minimize=minimize)
        
        # Create comparison DataFrame
        comparison = []
        
        # Common metrics to compare
        metrics_to_compare = [
            'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr'
        ]
        
        # Add rows for each dataset and metric
        for dataset in ['train', 'test']:
            for metric in metrics_to_compare:
                original_value = original_metrics[dataset].get(metric, None)
                distilled_value = None
                
                # Get distilled model metric
                distilled_col = f'{dataset}_{metric}'
                if distilled_col in best_config:
                    distilled_value = best_config[distilled_col]
                
                # Calculate difference if both values are available
                diff = None
                if original_value is not None and distilled_value is not None:
                    diff = distilled_value - original_value
                
                # Add to comparison
                comparison.append({
                    'dataset': dataset,
                    'metric': metric,
                    'original': original_value,
                    'distilled': distilled_value,
                    'difference': diff
                })
        
        # Create DataFrame
        df = pd.DataFrame(comparison)
        
        # Print summary if verbose
        if self.config.verbose:
            print("\n=== Model Comparison Summary ===")
            print(f"Best distilled model: {best_config['model_type']}, "
                 f"temp={best_config['temperature']}, alpha={best_config['alpha']}")
            print("\nTest set comparison:")
            test_comparison = df[df['dataset'] == 'test']
            for _, row in test_comparison.iterrows():
                diff_str = f" (Δ: {row['difference']:.4f})" if row['difference'] is not None else ""
                print(f"  {row['metric']}: {row['original']:.4f} → {row['distilled']:.4f}{diff_str}")
            print("================================\n")
        
        return df