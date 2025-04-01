import typing as t
from pathlib import Path
import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split

# Imports absolutos
from deepbridge.metrics.classification import Classification
from deepbridge.utils.model_registry import ModelType

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
        auto_fit: t.Optional[bool] = None
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
        """
        if experiment_type not in self.VALID_TYPES:
            raise ValueError(f"experiment_type must be one of {self.VALID_TYPES}")
            
        self.experiment_type = experiment_type
        self.dataset = dataset
        self.test_size = test_size
        self.random_state = random_state
        self.config = config or {}
        self.verbose = config.get('verbose', False) if config else False
        
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
        
        # Perform train-test split
        self._prepare_data()
        
        # Initialize and create alternative models
        self.alternative_models = {}
        self._create_alternative_models()
        
        # Auto-fit if enabled and dataset has probabilities
        if self.auto_fit and hasattr(dataset, 'original_prob') and dataset.original_prob is not None:
            default_model_type = self._get_default_model_type()
            
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

    def _get_default_model_type(self):
        """Get a default model type for XGBoost or similar"""
        # Try to find XGBoost or fallback to first model type
        for model_type in ModelType:
            if 'XGB' in model_type.name:
                return model_type
        # Fallback to first model type
        return next(iter(ModelType))

    def _create_alternative_models(self):
        """
        Create 3 alternative models different from the original model,
        using ModelRegistry directly without SurrogateModel.
        """
        # Check if dataset has a model
        if not hasattr(self.dataset, 'model') or self.dataset.model is None:
            if self.verbose:
                print("No original model found in dataset. Skipping alternative model creation.")
            return
        
        # Get original model type if possible
        original_model = self.dataset.model
        original_model_name = original_model.__class__.__name__.upper()
        
        # Import necessary classes
        from deepbridge.utils.model_registry import ModelRegistry, ModelType, ModelMode
        
        # Get all available model types from the enum
        all_model_types = []
        for model_type in ModelType:
            all_model_types.append(model_type)
            
        if self.verbose:
            print(f"Available model types: {[mt.name for mt in all_model_types]}")
            print(f"Original model identified as: {original_model.__class__.__name__}")
        
        # Identify original model type by name
        original_model_type = None
        for model_type in all_model_types:
            if model_type.name in original_model_name or original_model_name in model_type.name:
                original_model_type = model_type
                break
                
        if self.verbose:
            print(f"Mapped to model type: {original_model_type}")
        
        # Create a list of models to generate, excluding the original model if identified
        models_to_create = []
        for model_type in all_model_types:
            if model_type != original_model_type:
                models_to_create.append(model_type)
                if len(models_to_create) >= 3:  # Limit to 3 models
                    break
        
        if self.verbose:
            print(f"Creating alternative models: {[m.name for m in models_to_create]}")
        
        # Determine if we're working with a classification problem
        is_classification = self.experiment_type == "binary_classification"
        mode = ModelMode.CLASSIFICATION if is_classification else ModelMode.REGRESSION
        
        # Create and fit each alternative model
        for model_type in models_to_create:
            try:
                # Get model with default parameters directly from ModelRegistry
                model = ModelRegistry.get_model(
                    model_type=model_type,
                    custom_params=None,  # Use default parameters
                    mode=mode  # Use classification or regression mode based on experiment_type
                )
                
                # Fit the model on training data
                if self.verbose:
                    print(f"Fitting {model_type.name} model...")
                
                model.fit(self.X_train, self.y_train)
                
                # Store model with its type name
                self.alternative_models[model_type.name] = model
                
                if self.verbose:
                    print(f"Successfully created and fitted {model_type.name} as {model.__class__.__name__}")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to fit {model_type.name}: {str(e)}")
        
        if self.verbose:
            print(f"Created {len(self.alternative_models)} alternative models")

    def _prepare_data(self) -> None:
        """
        Prepare the data by performing train-test split on features and target.
        """
        X = self.dataset.X
        y = self.dataset.target
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        
        # If we have original probabilities, split them too
        if self.dataset.original_prob is not None:
            prob_train_idx = self.X_train.index
            prob_test_idx = self.X_test.index
            
            self.prob_train = self.dataset.original_prob.loc[prob_train_idx]
            self.prob_test = self.dataset.original_prob.loc[prob_test_idx]
        else:
            self.prob_train = None
            self.prob_test = None

    @property
    def model(self):
        """
        Return either the distillation model (if trained) or the model from dataset (if available).
        """
        if self.distillation_model is not None:
            return self.distillation_model
        elif hasattr(self.dataset, 'model') and self.dataset.model is not None:
            return self.dataset.model
        return None

    def fit(
        self,
        student_model_type: ModelType = ModelType.LOGISTIC_REGRESSION,
        student_params: t.Optional[dict] = None,
        temperature: float = 1.0,
        alpha: float = 0.5,
        use_probabilities: bool = True,
        n_trials: int = 50,
        validation_split: float = 0.2,
        verbose: bool = True,
        distillation_method: str = "surrogate"
    ) -> 'Experiment':
        """
        Train a model using either Surrogate Model or Knowledge Distillation approach.
        
        Args:
            student_model_type: Type of student model to use
            student_params: Custom parameters for student model
            temperature: Temperature parameter (used only for knowledge_distillation method)
            alpha: Weight between teacher's loss and true label loss (used only for knowledge_distillation method)
            use_probabilities: Whether to use pre-calculated probabilities (True) or teacher model (False)
            n_trials: Number of Optuna trials for hyperparameter optimization
            validation_split: Fraction of data to use for validation during optimization
            verbose: Whether to show optimization logs and results
            distillation_method: Method to use for distillation ('surrogate' or 'knowledge_distillation')
            
        Returns:
            self: The experiment instance with trained model
        """
        if self.experiment_type != "binary_classification":
            raise ValueError("Distillation methods are only supported for binary classification")
            
        # Configure logging
        logging_state = self._configure_logging(verbose)
        
        try:
            # Create distillation model
            self.distillation_model = self._create_distillation_model(
                distillation_method, student_model_type, student_params,
                temperature, alpha, use_probabilities, n_trials, validation_split
            )
            
            # Train the model
            self.distillation_model.fit(self.X_train, self.y_train, verbose=verbose)
            
            # Evaluate and store results
            train_metrics = self._evaluate_distillation_model('train')
            self._results_data['train'] = train_metrics['metrics']
            
            test_metrics = self._evaluate_distillation_model('test')
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
            
    def _create_distillation_model(self, 
                                distillation_method: str,
                                student_model_type: ModelType,
                                student_params: t.Optional[dict],
                                temperature: float,
                                alpha: float,
                                use_probabilities: bool,
                                n_trials: int,
                                validation_split: float) -> object:
        """Create appropriate distillation model based on method and available data"""
        if use_probabilities:
            if self.prob_train is None:
                raise ValueError("No teacher probabilities available. Set use_probabilities=False to use teacher model")
            return self._create_model_from_probabilities(
                distillation_method, student_model_type, student_params,
                temperature, alpha, n_trials, validation_split
            )
        else:
            if self.dataset.model is None:
                raise ValueError("No teacher model available. Set use_probabilities=True to use pre-calculated probabilities")
            return self._create_model_from_teacher(
                distillation_method, student_model_type, student_params,
                temperature, alpha, n_trials, validation_split
            )
            
    def _create_model_from_probabilities(self,
                                      distillation_method: str,
                                      student_model_type: ModelType,
                                      student_params: t.Optional[dict],
                                      temperature: float,
                                      alpha: float,
                                      n_trials: int,
                                      validation_split: float) -> object:
        """Create distillation model from pre-calculated probabilities"""
        if distillation_method.lower() == "surrogate":
            # Import at runtime to avoid circular import
            from deepbridge.distillation.techniques.surrogate import SurrogateModel
            
            return SurrogateModel.from_probabilities(
                probabilities=self.prob_train,
                student_model_type=student_model_type,
                student_params=student_params,
                random_state=self.random_state,
                validation_split=validation_split,
                n_trials=n_trials
            )
        elif distillation_method.lower() == "knowledge_distillation":
            # Import at runtime to avoid circular import
            from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
            
            return KnowledgeDistillation.from_probabilities(
                probabilities=self.prob_train,
                student_model_type=student_model_type,
                student_params=student_params,
                temperature=temperature,
                alpha=alpha,
                n_trials=n_trials,
                validation_split=validation_split,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown distillation method: {distillation_method}. Use 'surrogate' or 'knowledge_distillation'")
            
    def _create_model_from_teacher(self,
                                distillation_method: str,
                                student_model_type: ModelType,
                                student_params: t.Optional[dict],
                                temperature: float,
                                alpha: float,
                                n_trials: int,
                                validation_split: float) -> object:
        """Create distillation model from teacher model"""
        if distillation_method.lower() == "surrogate":
            # Surrogate method doesn't support direct use of teacher model
            raise ValueError("The surrogate method does not support direct use of teacher model. "
                           "Please set use_probabilities=True or use method='knowledge_distillation'")
        elif distillation_method.lower() == "knowledge_distillation":
            # Import at runtime to avoid circular import
            from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
            
            return KnowledgeDistillation(
                teacher_model=self.dataset.model,
                student_model_type=student_model_type,
                student_params=student_params,
                temperature=temperature,
                alpha=alpha,
                n_trials=n_trials,
                validation_split=validation_split,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown distillation method: {distillation_method}. Use 'surrogate' or 'knowledge_distillation'")
    
    def _evaluate_distillation_model(self, dataset: str = 'test') -> dict:
        """
        Evaluate the distillation model for the specified dataset.
        
        Args:
            dataset: Which dataset to evaluate ('train' or 'test')
            
        Returns:
            dict: Dictionary containing evaluation metrics and predictions
        """
        print(f"\n=== Evaluating distillation model on {dataset} dataset ===")
        if dataset == 'train':
            X, y, prob = self.X_train, self.y_train, self.prob_train
        else:
            X, y, prob = self.X_test, self.y_test, self.prob_test
        
        # Get probabilities
        student_probs = self.distillation_model.predict(X)
        
        # Convert probabilities to binary predictions
        y_pred = (student_probs > 0.5).astype(int)
        
        # Get full probabilities (for both classes)
        y_prob = self.distillation_model.predict_proba(X)
        
        print(f"Student predictions shape: {y_prob.shape}")
        print(f"First 3 student probabilities: {y_prob[:3]}")
        
        # Extract probability of positive class for student
        student_prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else student_probs
        
        # Prepare teacher probabilities
        if prob is not None:
            print(f"Teacher probabilities type: {type(prob)}")
            if isinstance(prob, pd.DataFrame):
                if 'prob_class_1' in prob.columns:
                    print(f"Using 'prob_class_1' column from teacher probabilities")
                    teacher_prob_pos = prob['prob_class_1'].values
                    teacher_probs = prob[['prob_class_0', 'prob_class_1']].values
                else:
                    # Assume the last column is the probability of the positive class
                    print(f"Using last column as positive class probability")
                    pos_prob = prob.iloc[:, -1].values
                    teacher_prob_pos = pos_prob
                    teacher_probs = np.column_stack([1 - pos_prob, pos_prob])
            else:
                teacher_probs = prob
                teacher_prob_pos = prob[:, 1] if prob.shape[1] > 1 else prob
                    
            print(f"Teacher probabilities shape: {teacher_probs.shape if hasattr(teacher_probs, 'shape') else 'unknown'}")
            print(f"First 3 teacher probabilities (positive class): {teacher_prob_pos[:3]}")
            
            # Manually calculate KS statistic
            try:
                from scipy import stats
                ks_stat, ks_pvalue = stats.ks_2samp(teacher_prob_pos, student_prob_pos)
                print(f"KS Statistic calculation: {ks_stat}, p-value: {ks_pvalue}")
            except Exception as e:
                print(f"Error calculating KS statistic: {str(e)}")
                ks_stat, ks_pvalue = None, None
                
            # Manually calculate R² score
            try:
                from sklearn.metrics import r2_score
                # Sort distributions
                teacher_sorted = np.sort(teacher_prob_pos)
                student_sorted = np.sort(student_prob_pos)
                # Use equal lengths
                min_len = min(len(teacher_sorted), len(student_sorted))
                r2 = r2_score(teacher_sorted[:min_len], student_sorted[:min_len])
                print(f"R² Score calculation: {r2}")
            except Exception as e:
                print(f"Error calculating R² score: {str(e)}")
                r2 = None
        else:
            print(f"No teacher probabilities available for {dataset} dataset")
            ks_stat, ks_pvalue, r2 = None, None, None
        
        # Calculate metrics using the Classification class
        metrics = self.metrics_calculator.calculate_metrics(
            y_true=y,
            y_pred=y_pred,  # Now using binary predictions
            y_prob=student_prob_pos,  # Probability of positive class
            teacher_prob=teacher_prob_pos if prob is not None else None  # Add teacher probability
        )
        
        # Manually add distribution comparison metrics if not present
        if 'ks_statistic' not in metrics or metrics['ks_statistic'] is None:
            metrics['ks_statistic'] = ks_stat
            metrics['ks_pvalue'] = ks_pvalue
            
        if 'r2_score' not in metrics or metrics['r2_score'] is None:
            metrics['r2_score'] = r2
        
        # Add KL divergence if not present and we have teacher probabilities
        if 'kl_divergence' not in metrics and prob is not None:
            try:
                # Calculate KL divergence manually
                # Add epsilon to avoid log(0)
                epsilon = 1e-10
                teacher_prob_pos = np.clip(teacher_prob_pos, epsilon, 1-epsilon)
                student_prob_pos = np.clip(student_prob_pos, epsilon, 1-epsilon)
                
                # For binary classification (calculate for both classes)
                teacher_prob_neg = 1 - teacher_prob_pos
                student_prob_neg = 1 - student_prob_pos
                
                # Calculate KL divergence
                kl_div_pos = np.mean(teacher_prob_pos * np.log(teacher_prob_pos / student_prob_pos))
                kl_div_neg = np.mean(teacher_prob_neg * np.log(teacher_prob_neg / student_prob_neg))
                kl_div = (kl_div_pos + kl_div_neg) / 2
                
                metrics['kl_divergence'] = kl_div
                print(f"Manually calculated KL divergence: {kl_div}")
            except Exception as e:
                print(f"Error calculating KL divergence: {str(e)}")
                metrics['kl_divergence'] = None
        
        # Include best hyperparameters in metrics
        if hasattr(self.distillation_model, 'best_params') and self.distillation_model.best_params:
            metrics['best_params'] = self.distillation_model.best_params
            
        # Include distillation method in metrics
        metrics['distillation_method'] = getattr(self.distillation_model, '__class__', 'unknown').__name__
            
        # Include predictions
        predictions_df = pd.DataFrame({
            'y_true': y,
            'y_pred': y_pred,
            'y_prob': student_prob_pos  # Probability of positive class
        })
        
        if prob is not None:
            # Add teacher probabilities to predictions dataframe
            predictions_df['teacher_prob'] = teacher_prob_pos
        
        print(f"Evaluation metrics: {metrics}")
        print(f"=== Evaluation complete ===\n")
        
        return {'metrics': metrics, 'predictions': predictions_df}
    
    def get_student_predictions(self, dataset: str = 'test') -> pd.DataFrame:
        """
        Get predictions from the trained student model.
        
        Args:
            dataset: Which dataset to get predictions for ('train' or 'test')
            
        Returns:
            DataFrame with predictions and probabilities
        """
        if self.distillation_model is None:
            raise ValueError("No trained distillation model available. Call fit() first")
            
        X = self.X_train if dataset == 'train' else self.X_test
        y_true = self.y_train if dataset == 'train' else self.y_test
        
        # Get probabilities
        probs = self.distillation_model.predict(X)
        
        # Convert to binary predictions
        y_pred = (probs > 0.5).astype(int)
        
        # Get probability distributions
        y_prob = self.distillation_model.predict_proba(X)
        
        # Create DataFrame
        predictions = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'prob_0': y_prob[:, 0],
            'prob_1': y_prob[:, 1]
        })
        
        return predictions
        
    def calculate_student_metrics(self, dataset: str = 'test') -> dict:
        """
        Calculate metrics for the distilled (student) model.
        
        Args:
            dataset: Which dataset to calculate metrics for ('train' or 'test')
            
        Returns:
            dict: Dictionary containing evaluation metrics for the student model
        """
        if self.distillation_model is None:
            raise ValueError("No trained distillation model available. Call fit() first")
            
        # Get predictions from student model
        predictions_df = self.get_student_predictions(dataset)
        
        # Calculate metrics
        metrics = self.calculate_metrics(
            y_true=predictions_df['y_true'],
            y_pred=predictions_df['y_pred'],
            y_prob=predictions_df['prob_1']  # Use probability of positive class
        )
        
        # Store metrics in results
        self.results[dataset] = metrics
        
        return metrics
        
    def compare_teacher_student_metrics(self) -> pd.DataFrame:
        """
        Compare metrics between teacher and student models for both train and test sets.
        
        Returns:
            pd.DataFrame: DataFrame containing metrics comparison
        """
        if self.distillation_model is None:
            raise ValueError("No trained distillation model available. Call fit() first")
            
        results = []
        
        # Calculate metrics for both datasets
        for dataset in ['train', 'test']:
            # Get teacher metrics
            teacher_metrics = None
            if dataset == 'train' and self.prob_train is not None:
                teacher_metrics = self.calculate_metrics(
                    y_true=self.y_train,
                    y_pred=self._get_binary_predictions(self.prob_train),
                    y_prob=self.prob_train['prob_1'] if 'prob_1' in self.prob_train.columns else self.prob_train.iloc[:, -1]
                )
            elif dataset == 'test' and self.prob_test is not None:
                teacher_metrics = self.calculate_metrics(
                    y_true=self.y_test,
                    y_pred=self._get_binary_predictions(self.prob_test),
                    y_prob=self.prob_test['prob_1'] if 'prob_1' in self.prob_test.columns else self.prob_test.iloc[:, -1]
                )
                
            # Get student metrics
            student_metrics = self.calculate_student_metrics(dataset)
            
            # Add to results
            for metric_name in student_metrics.keys():
                result = {
                    'dataset': dataset,
                    'metric': metric_name,
                    'student_value': student_metrics[metric_name]
                }
                if teacher_metrics and metric_name in teacher_metrics:
                    result['teacher_value'] = teacher_metrics[metric_name]
                    result['difference'] = student_metrics[metric_name] - teacher_metrics[metric_name]
                results.append(result)
        
        # Create DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Reorder columns
        if 'teacher_value' in comparison_df.columns:
            column_order = ['dataset', 'metric', 'teacher_value', 'student_value', 'difference']
            comparison_df = comparison_df[column_order]
            
        return comparison_df
        
    def calculate_metrics(self, 
                         y_true: t.Union[np.ndarray, pd.Series],
                         y_pred: t.Union[np.ndarray, pd.Series],
                         y_prob: t.Optional[t.Union[np.ndarray, pd.Series]] = None,
                         teacher_prob: t.Optional[t.Union[np.ndarray, pd.Series]] = None) -> dict:
        """
        Calculate metrics based on experiment type.
        """
        if self.experiment_type == "binary_classification":
            return self.metrics_calculator.calculate_metrics(y_true, y_pred, y_prob, teacher_prob)
        else:
            raise NotImplementedError(f"Metrics calculation not implemented for {self.experiment_type}")
            
    def evaluate_predictions(self, 
                           predictions: pd.DataFrame,
                           dataset: str = 'train',
                           pred_column: t.Optional[str] = None,
                           prob_column: t.Optional[str] = None,
                           threshold: float = 0.5) -> dict:
        """
        Evaluate predictions for the specified dataset.
        """
        if dataset not in ['train', 'test']:
            raise ValueError("dataset must be either 'train' or 'test'")
            
        y_true = self.y_train if dataset == 'train' else self.y_test
        
        # If pred_column is provided, use it directly
        if pred_column is not None:
            y_pred = predictions[pred_column]
        # Otherwise, convert probabilities to binary predictions
        else:
            y_pred = self._get_binary_predictions(predictions, threshold)
        
        # Get probabilities if prob_column is provided
        y_prob = predictions[prob_column] if prob_column else None
        
        metrics = self.calculate_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )
        
        self.results[dataset] = metrics
        return metrics
    
    def get_dataset_split(self, dataset: str = 'train') -> tuple:
        """
        Get the features and target for specified dataset split.
        """
        if dataset == 'train':
            return self.X_train, self.y_train, self.prob_train
        elif dataset == 'test':
            return self.X_test, self.y_test, self.prob_test
        else:
            raise ValueError("dataset must be either 'train' or 'test'")
    
    def _get_binary_predictions(self, probabilities: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
        """
        Convert probability predictions to binary predictions using a threshold.
        """
        # If we have multiple columns, assume the last one is for class 1
        prob_values = probabilities.iloc[:, -1] if len(probabilities.columns) > 1 else probabilities.iloc[:, 0]
        return (prob_values >= threshold).astype(int)

    def get_alternative_models(self):
        """
        Get the alternative models created by the experiment.
        
        Returns:
            dict: Dictionary mapping model names to model instances
        """
        return self.alternative_models
    
    def evaluate_alternative_models(self, dataset='test'):
        """
        Evaluate all alternative models on the specified dataset.
        
        Args:
            dataset: Which dataset to evaluate on ('train' or 'test')
            
        Returns:
            pd.DataFrame: DataFrame with evaluation metrics for each model
        """
        if not self.alternative_models:
            if self.verbose:
                print("No alternative models available.")
            return pd.DataFrame()
            
        # Get the appropriate data
        X = self.X_train if dataset == 'train' else self.X_test
        y = self.y_train if dataset == 'train' else self.y_test
        
        # Evaluate each model
        results = []
        
        # First evaluate original model if available
        if hasattr(self.dataset, 'model') and self.dataset.model is not None:
            original_model = self.dataset.model
            original_name = original_model.__class__.__name__
            metrics = self._evaluate_model(original_model, original_name, 'original', X, y)
            if metrics:
                results.append(metrics)
        
        # Then evaluate alternative models
        for name, model in self.alternative_models.items():
            metrics = self._evaluate_model(model, name, 'alternative', X, y)
            if metrics:
                results.append(metrics)
        
        # Convert results to DataFrame
        return pd.DataFrame(results)
        
    def _evaluate_model(self, model, model_name, model_type, X, y):
        """Helper method to evaluate any model"""
        try:
            # Check if it's a surrogate-created model (regressor)
            is_regressor = "regressor" in model.__class__.__name__.lower()
            
            if is_regressor and self.experiment_type == "binary_classification":
                # For surrogate models in classification problems:
                # 1. Get continuous predictions (logits)
                logits = model.predict(X)
                
                # 2. Convert to probabilities using the sigmoid function
                from scipy.special import expit
                y_prob = expit(logits)
                
                # 3. Convert to binary predictions using threshold
                y_pred = (y_prob > 0.5).astype(int)
                
            else:
                # For regular models
                y_pred = model.predict(X)
                
                # Get probabilities if available
                y_prob = None
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    if probs.shape[1] > 1:  # Binary or multiclass
                        y_prob = probs[:, 1]  # Probability of positive class
            
            # Calculate metrics
            metrics = self.calculate_metrics(
                y_true=y,
                y_pred=y_pred,
                y_prob=y_prob if y_prob is not None else None
            )
            
            # Add model info
            metrics['model_name'] = model_name
            metrics['model_type'] = model_type
            
            return metrics
        except Exception as e:
            if self.verbose:
                print(f"Failed to evaluate model {model_name}: {str(e)}")
            return None

    @property
    def results(self) -> dict:
        """
        Property to get results data
        """
        return self._results_data
    
    @results.setter
    def results(self, value):
        """
        Property setter for results
        """
        self._results_data = value
    
    @property
    def metrics(self) -> dict:
        """
        Get all metrics for both train and test datasets.
        """
        # Calculate metrics if they haven't been calculated yet
        if not self._results_data['train'] and self.prob_train is not None:
            binary_preds = self._get_binary_predictions(self.prob_train)
            prob_values = self.prob_train.iloc[:, -1] if len(self.prob_train.columns) > 1 else self.prob_train.iloc[:, 0]
            
            metrics = self.calculate_metrics(
                y_true=self.y_train,
                y_pred=binary_preds,
                y_prob=prob_values
            )
            self._results_data['train'] = metrics
            
        if not self._results_data['test'] and self.prob_test is not None:
            binary_preds = self._get_binary_predictions(self.prob_test)
            prob_values = self.prob_test.iloc[:, -1] if len(self.prob_test.columns) > 1 else self.prob_test.iloc[:, 0]
            
            metrics = self.calculate_metrics(
                y_true=self.y_test,
                y_pred=binary_preds,
                y_prob=prob_values
            )
            self._results_data['test'] = metrics
            
        return {
            'train': self._results_data['train'],
            'test': self._results_data['test']
        }
    
    def compare_all_models(self, dataset='test'):
        """
        Compare all models including original, alternative, and distilled models.
        
        Args:
            dataset: Which dataset to evaluate on ('train' or 'test')
            
        Returns:
            pd.DataFrame: DataFrame with comparative evaluation metrics for all models
        """
        results = []
        
        # Get the appropriate data
        X = self.X_train if dataset == 'train' else self.X_test
        y = self.y_train if dataset == 'train' else self.y_test
        
        # Add original model if available
        if hasattr(self.dataset, 'model') and self.dataset.model is not None:
            original_model = self.dataset.model
            original_name = original_model.__class__.__name__
            metrics = self._evaluate_model(original_model, original_name, 'original', X, y)
            if metrics:
                results.append(metrics)
        
        # Add alternative models
        for name, model in self.alternative_models.items():
            metrics = self._evaluate_model(model, name, 'alternative', X, y)
            if metrics:
                results.append(metrics)
        
        # Add distilled model if available
        if self.distillation_model is not None:
            try:
                # Get predictions
                student_probs = self.distillation_model.predict(X)
                y_pred = (student_probs > 0.5).astype(int)
                
                # Get probability distributions
                y_prob = self.distillation_model.predict_proba(X)
                student_prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else student_probs
                
                # Calculate metrics
                metrics = self.calculate_metrics(
                    y_true=y,
                    y_pred=y_pred,
                    y_prob=student_prob_pos
                )
                
                # Add model info
                metrics['model_name'] = 'Distilled_' + getattr(self.distillation_model, '__class__', type(self.distillation_model)).__name__
                metrics['model_type'] = 'distilled'
                
                results.append(metrics)
            except Exception as e:
                if self.verbose:
                    print(f"Failed to evaluate distilled model: {str(e)}")
        
        # Convert results to DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Reorder columns to put model info first
        if not comparison_df.empty:
            cols = ['model_name', 'model_type'] + [col for col in comparison_df.columns if col not in ['model_name', 'model_type']]
            comparison_df = comparison_df[cols]
        
        return comparison_df
    
    def get_comprehensive_results(self):
        """
        Return a comprehensive dictionary with all metrics and information
        about the experiment, original model, alternative models, and surrogate model.
        
        Returns:
        --------
        dict : A comprehensive dictionary with all metrics and information
        """
        import pandas as pd
        import numpy as np
        
        # Build the results data structure using the _generate_results_dict helper method
        return self._generate_results_dict()
    
    def _generate_results_dict(self):
        """Helper method to generate the comprehensive results dictionary"""
        import pandas as pd
        import numpy as np
        
        result = {
            'experiment_info': {},
            'dataset_info': {},
            'original_model': {},
            'alternative_models': {},
            'surrogate_model': {},
            'performance_comparison': {},
            'feature_importance': {}
        }
        
        # 1. Experiment Information
        result['experiment_info'] = self._get_experiment_info()
        
        # 2. Dataset Information
        result['dataset_info'] = self._get_dataset_info()
        
        # 3. Original Model Information
        result['original_model'] = self._get_original_model_info()
        
        # 4. Alternative Models Information
        result['alternative_models'] = self._get_alternative_models_info()
        
        # 5. Surrogate Model Information
        result['surrogate_model'] = self._get_surrogate_model_info()
        
        # 6. Performance Comparison
        result['performance_comparison'] = self._get_performance_comparison()
        
        # 7. Feature Importance Analysis
        result['feature_importance'] = self._get_feature_importance()
        
        # 8. Convert numpy types to Python native types for JSON compatibility
        return self._convert_numpy_types(result)
        
    def _get_experiment_info(self):
        """Get basic experiment information"""
        import pandas as pd
        return {
            'experiment_type': self.experiment_type,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'auto_fit': self.auto_fit,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
    def _get_dataset_info(self):
        """Get dataset information"""
        dataset = self.dataset
        return {
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'features_count': len(dataset.features),
            'feature_names': dataset.features,
            'categorical_features': dataset.categorical_features if hasattr(dataset, 'categorical_features') else [],
            'numerical_features': dataset.numerical_features if hasattr(dataset, 'numerical_features') else [],
            'target_name': dataset.target_name,
            'class_distribution': {
                'train': dict(self.y_train.value_counts().items()),
                'test': dict(self.y_test.value_counts().items())
            }
        }
        
    def _get_original_model_info(self):
        """Get information about the original model"""
        dataset = self.dataset
        result = {}
        
        if hasattr(dataset, 'model') and dataset.model is not None:
            original_model = dataset.model
            result = {
                'model_type': original_model.__class__.__name__,
                'model_params': original_model.get_params() if hasattr(original_model, 'get_params') else {},
                'metrics': {}
            }
            
            # Calculate metrics for original model
            try:
                # Get metrics on train and test sets
                result['metrics'] = self._calculate_model_metrics(original_model)
                
                # Get curve data if available
                curves = self._calculate_curve_data(original_model)
                result.update(curves)
                
            except Exception as e:
                print(f"Error calculating metrics for original model: {str(e)}")
                result['metrics_error'] = str(e)
                
        return result
        
    def _calculate_model_metrics(self, model):
        """Calculate metrics for a model on both train and test sets"""
        metrics = {'train': {}, 'test': {}}
        
        try:
            # Train metrics
            X_train, y_train = self.X_train, self.y_train
            y_pred_train = model.predict(X_train)
            
            # Get probabilities if available
            y_prob_train = None
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_train)
                if probs.shape[1] > 1:  # Binary classification
                    y_prob_train = probs[:, 1]
                    
            train_metrics = self.calculate_metrics(
                y_true=y_train,
                y_pred=y_pred_train,
                y_prob=y_prob_train
            )
            
            # Test metrics
            X_test, y_test = self.X_test, self.y_test
            y_pred_test = model.predict(X_test)
            
            # Get probabilities if available
            y_prob_test = None
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)
                if probs.shape[1] > 1:  # Binary classification
                    y_prob_test = probs[:, 1]
                    
            test_metrics = self.calculate_metrics(
                y_true=y_test,
                y_pred=y_pred_test,
                y_prob=y_prob_test
            )
            
            metrics = {
                'train': train_metrics,
                'test': test_metrics
            }
            
        except Exception as e:
            print(f"Error in _calculate_model_metrics: {str(e)}")
            
        return metrics
    
    def _calculate_curve_data(self, model):
        """Calculate ROC and PR curve data for a model"""
        result = {}
        
        if not hasattr(model, 'predict_proba'):
            return result
            
        try:
            from sklearn.metrics import roc_curve, precision_recall_curve
            
            # Get test probabilities
            X_test, y_test = self.X_test, self.y_test
            probs = model.predict_proba(X_test)
            if probs.shape[1] <= 1:
                return result
                
            y_prob_test = probs[:, 1]
            
            # ROC curve
            fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob_test)
            result['roc_curve'] = {
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist(),
                'thresholds': roc_thresholds.tolist()
            }
            
            # PR curve
            precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob_test)
            result['pr_curve'] = {
                'precision': precision.tolist(),
                'recall': recall.tolist(),
                'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
            }
            
        except Exception as e:
            print(f"Error calculating curve data: {str(e)}")
            
        return result
        
    def _get_alternative_models_info(self):
        """Get information about alternative models"""
        dataset = self.dataset
        result = {}
        alternative_models = self.alternative_models
        
        if not alternative_models:
            return result
            
        # Get comparison metrics for all alternative models
        try:
            alt_train_metrics = self.evaluate_alternative_models('train')
            alt_test_metrics = self.evaluate_alternative_models('test')
            
            # Convert DataFrame to dict format
            alt_train_dict = [] if alt_train_metrics.empty else alt_train_metrics.to_dict(orient='records')
            alt_test_dict = [] if alt_test_metrics.empty else alt_test_metrics.to_dict(orient='records')
            
            # Store metrics
            result['comparison'] = {
                'train': alt_train_dict,
                'test': alt_test_dict
            }
            
            # Detailed info for each alternative model
            result['models'] = {}
            
            for name, model in alternative_models.items():
                model_info = {
                    'model_type': model.__class__.__name__,
                    'model_params': model.get_params() if hasattr(model, 'get_params') else {},
                }
                
                # Find this model's metrics in the comparison data
                if alt_test_dict:
                    for model_data in alt_test_dict:
                        if model_data.get('model_name') == name:
                            model_info['test_metrics'] = {k: v for k, v in model_data.items() 
                                                        if k not in ['model_name', 'model_type']}
                            break
                
                if alt_train_dict:
                    for model_data in alt_train_dict:
                        if model_data.get('model_name') == name:
                            model_info['train_metrics'] = {k: v for k, v in model_data.items() 
                                                        if k not in ['model_name', 'model_type']}
                            break
                
                # Get feature importance if available
                model_info['feature_importance'] = self._extract_feature_importance(model, dataset.features)
                result['models'][name] = model_info
                
        except Exception as e:
            print(f"Error processing alternative models: {str(e)}")
            result['error'] = str(e)
            
        return result
        
    def _extract_feature_importance(self, model, feature_names):
        """Extract feature importance from a model if available"""
        importance_dict = {}
        
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                importance_dict = dict(zip(feature_names, importance))
            elif hasattr(model, 'coef_'):
                if len(model.coef_.shape) == 1:
                    importance = model.coef_
                else:
                    importance = model.coef_[0]  # Take first class for multiclass
                importance_dict = dict(zip(feature_names, importance))
        except Exception as e:
            print(f"Error extracting feature importance: {str(e)}")
            
        return importance_dict
        
    def _get_surrogate_model_info(self):
        """Get information about the surrogate (distilled) model"""
        result = {}
        
        if self.distillation_model is None:
            return result
            
        surrogate = self.distillation_model
        
        result = {
            'model_type': surrogate.__class__.__name__,
            'best_params': getattr(surrogate, 'best_params', {}),
            'metrics': {}
        }
        
        # Calculate metrics for surrogate model
        try:
            # Get metrics on train set
            X_train, y_train = self.X_train, self.y_train
            surrogate_probs_train = surrogate.predict(X_train)
            surrogate_pred_train = (surrogate_probs_train > 0.5).astype(int)
            
            # Get full probability distribution
            surrogate_prob_dist_train = surrogate.predict_proba(X_train)
            surrogate_prob_class1_train = surrogate_prob_dist_train[:, 1] if surrogate_prob_dist_train.shape[1] > 1 else surrogate_probs_train
            
            train_metrics = self.calculate_metrics(
                y_true=y_train,
                y_pred=surrogate_pred_train,
                y_prob=surrogate_prob_class1_train
            )
            
            # Get metrics on test set
            X_test, y_test = self.X_test, self.y_test
            surrogate_probs_test = surrogate.predict(X_test)
            surrogate_pred_test = (surrogate_probs_test > 0.5).astype(int)
            
            # Get full probability distribution
            surrogate_prob_dist_test = surrogate.predict_proba(X_test)
            surrogate_prob_class1_test = surrogate_prob_dist_test[:, 1] if surrogate_prob_dist_test.shape[1] > 1 else surrogate_probs_test
            
            test_metrics = self.calculate_metrics(
                y_true=y_test,
                y_pred=surrogate_pred_test,
                y_prob=surrogate_prob_class1_test
            )
            
            result['metrics'] = {
                'train': train_metrics,
                'test': test_metrics
            }
            
            # Get ROC and PR curve data
            try:
                from sklearn.metrics import roc_curve, precision_recall_curve
                
                # ROC curve
                fpr, tpr, roc_thresholds = roc_curve(y_test, surrogate_prob_class1_test)
                result['roc_curve'] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': roc_thresholds.tolist()
                }
                
                # PR curve
                precision, recall, pr_thresholds = precision_recall_curve(y_test, surrogate_prob_class1_test)
                result['pr_curve'] = {
                    'precision': precision.tolist(),
                    'recall': recall.tolist(),
                    'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
                }
            except Exception as e:
                print(f"Error calculating curve data for surrogate model: {str(e)}")
                
            # Calculate distribution comparison metrics between original and surrogate
            dataset = self.dataset
            if hasattr(dataset, 'model') and dataset.model is not None and hasattr(dataset.model, 'predict_proba'):
                try:
                    # Get distribution comparison metrics
                    result['distribution_comparison'] = self._calculate_distribution_comparison(
                        dataset.model, surrogate, self.X_test)
                except Exception as e:
                    print(f"Error calculating distribution comparison: {str(e)}")
            
        except Exception as e:
            print(f"Error calculating metrics for surrogate model: {str(e)}")
            result['metrics_error'] = str(e)
            
        return result
        
    def _calculate_distribution_comparison(self, original_model, surrogate_model, X_test):
        """Calculate distribution comparison metrics between original and surrogate models"""
        import numpy as np
        from scipy import stats
        
        try:
            # Get original model predictions
            original_probs = original_model.predict_proba(X_test)
            original_prob_class1 = original_probs[:, 1] if original_probs.shape[1] > 1 else original_probs
            
            # Get surrogate model predictions
            surrogate_probs = surrogate_model.predict_proba(X_test)
            surrogate_prob_class1 = surrogate_probs[:, 1] if surrogate_probs.shape[1] > 1 else surrogate_probs
            
            # KS test between distributions
            ks_stat, ks_pval = stats.ks_2samp(original_prob_class1, surrogate_prob_class1)
            
            # Jensen-Shannon Divergence
            from scipy.spatial.distance import jensenshannon
            js_dist = jensenshannon(original_prob_class1, surrogate_prob_class1)
            
            # KL Divergence (with smoothing to avoid log(0))
            epsilon = 1e-10
            orig_smooth = np.clip(original_prob_class1, epsilon, 1-epsilon)
            surr_smooth = np.clip(surrogate_prob_class1, epsilon, 1-epsilon)
            
            kl_div_orig_to_surr = np.sum(orig_smooth * np.log(orig_smooth / surr_smooth)) / len(orig_smooth)
            kl_div_surr_to_orig = np.sum(surr_smooth * np.log(surr_smooth / orig_smooth)) / len(surr_smooth)
            
            # Return comparison metrics
            return {
                'ks_statistic': float(ks_stat),
                'ks_pvalue': float(ks_pval),
                'jensen_shannon_distance': float(js_dist),
                'kl_divergence_orig_to_surr': float(kl_div_orig_to_surr),
                'kl_divergence_surr_to_orig': float(kl_div_surr_to_orig)
            }
        except Exception as e:
            print(f"Error in distribution comparison: {str(e)}")
            return {}
    
    def _get_performance_comparison(self):
        """Get performance comparison across all models"""
        result = {}
        
        try:
            comparison_test = self.compare_all_models('test')
            comparison_train = self.compare_all_models('train')
            
            if not comparison_test.empty:
                result['test'] = comparison_test.to_dict(orient='records')
            
            if not comparison_train.empty:
                result['train'] = comparison_train.to_dict(orient='records')
                
            # Calculate the best model for each metric
            if not comparison_test.empty:
                metrics_columns = [col for col in comparison_test.columns 
                                  if col not in ['model_name', 'model_type']]
                
                best_models = {}
                for metric in metrics_columns:
                    # Skip non-numeric columns
                    if comparison_test[metric].dtype == 'object':
                        continue
                        
                    # Determine if higher is better (common knowledge for standard metrics)
                    higher_is_better = metric.lower() not in ['mse', 'rmse', 'error', 'loss', 'kl_divergence']
                    
                    if higher_is_better:
                        best_idx = comparison_test[metric].idxmax()
                    else:
                        best_idx = comparison_test[metric].idxmin()
                        
                    if best_idx is not None:
                        best_model = comparison_test.loc[best_idx, 'model_name']
                        best_value = comparison_test.loc[best_idx, metric]
                        
                        best_models[metric] = {
                            'model': best_model,
                            'value': best_value,
                            'higher_is_better': higher_is_better
                        }
                
                result['best_models'] = best_models
                
                # Find overall best model
                result['overall_best_model'] = self._find_overall_best_model(comparison_test, metrics_columns)
        except Exception as e:
            print(f"Error in performance comparison: {str(e)}")
            result['error'] = str(e)
            
        return result
        
    def _find_overall_best_model(self, comparison_df, metrics_columns):
        """Find the overall best model based on metrics"""
        if 'accuracy' in metrics_columns:
            best_acc_idx = comparison_df['accuracy'].idxmax()
            return {
                'model': comparison_df.loc[best_acc_idx, 'model_name'],
                'model_type': comparison_df.loc[best_acc_idx, 'model_type'],
                'accuracy': comparison_df.loc[best_acc_idx, 'accuracy']
            }
        elif len(metrics_columns) > 0:
            # Use first metric if accuracy not available
            metric = metrics_columns[0]
            higher_is_better = metric.lower() not in ['mse', 'rmse', 'error', 'loss', 'kl_divergence']
            
            if higher_is_better:
                best_idx = comparison_df[metric].idxmax()
            else:
                best_idx = comparison_df[metric].idxmin()
                
            return {
                'model': comparison_df.loc[best_idx, 'model_name'],
                'model_type': comparison_df.loc[best_idx, 'model_type'],
                'primary_metric': {
                    'name': metric,
                    'value': comparison_df.loc[best_idx, metric]
                }
            }
        return {}
    
    def _get_feature_importance(self):
        """Get feature importance analysis across models"""
        import numpy as np
        dataset = self.dataset
        result = {}
        
        try:
            # Collect feature importance from all models
            feature_importance = {}
            
            # From original model
            if (hasattr(dataset, 'model') and dataset.model is not None):
                feature_importance['original'] = self._extract_feature_importance(
                    dataset.model, dataset.features)
            
            # From alternative models
            for name, model in self.alternative_models.items():
                importance = self._extract_feature_importance(model, dataset.features)
                if importance:
                    feature_importance[name] = importance
            
            # From surrogate model if available
            if (self.distillation_model is not None and
                hasattr(self.distillation_model, 'student_model')):
                feature_importance['surrogate'] = self._extract_feature_importance(
                    self.distillation_model.student_model, dataset.features)
            
            # Aggregate feature importance across models
            if feature_importance:
                # Calculate mean importance for each feature
                all_features = dataset.features
                aggregated_importance = {}
                
                for feature in all_features:
                    values = []
                    for model_name, importances in feature_importance.items():
                        if feature in importances:
                            values.append(abs(importances[feature]))  # Use absolute value for consistency
                    
                    if values:
                        aggregated_importance[feature] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'values': dict(zip(feature_importance.keys(), 
                                              [imp.get(feature, 0) for imp in feature_importance.values()]))
                        }
                
                # Sort features by mean importance
                sorted_features = sorted(aggregated_importance.items(), 
                                        key=lambda x: x[1]['mean'], 
                                        reverse=True)
                
                result = {
                    'per_model': feature_importance,
                    'aggregated': aggregated_importance,
                    'top_features': [f[0] for f in sorted_features[:10]]  # Top 10 features
                }
        except Exception as e:
            print(f"Error in feature importance analysis: {str(e)}")
            result['error'] = str(e)
            
        return result
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON compatibility"""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        elif isinstance(obj, pd.DataFrame):
            return self._convert_numpy_types(obj.to_dict(orient='records'))
        elif isinstance(obj, pd.Series):
            return self._convert_numpy_types(obj.to_dict())
        else:
            return obj


    @property
    def results(self):
        """
        Property that returns a comprehensive dictionary with all metrics and information
        about the experiment, original model, alternative models, and surrogate model.
        
        Returns:
        --------
        dict : A comprehensive dictionary with all metrics and information
        """
        import pandas as pd
        import numpy as np
        
        result = {
            'experiment_info': {},
            'dataset_info': {},
            'original_model': {},
            'alternative_models': {},
            'surrogate_model': {},
            'performance_comparison': {},
            'feature_importance': {}
        }
        
        # 1. Experiment Information
        result['experiment_info'] = {
            'experiment_type': self.experiment_type,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'auto_fit': self.auto_fit,
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        # 2. Dataset Information
        dataset = self.dataset
        result['dataset_info'] = {
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'features_count': len(dataset.features),
            'feature_names': dataset.features,
            'categorical_features': dataset.categorical_features if hasattr(dataset, 'categorical_features') else [],
            'numerical_features': dataset.numerical_features if hasattr(dataset, 'numerical_features') else [],
            'target_name': dataset.target_name,
            'class_distribution': {
                'train': dict(self.y_train.value_counts().items()),
                'test': dict(self.y_test.value_counts().items())
            }
        }
        
        # 3. Original Model Information
        if hasattr(dataset, 'model') and dataset.model is not None:
            original_model = dataset.model
            result['original_model'] = {
                'model_type': original_model.__class__.__name__,
                'model_params': original_model.get_params() if hasattr(original_model, 'get_params') else {},
                'metrics': {}
            }
            
            # Calculate metrics for original model
            try:
                # Get metrics on train set
                X_train, y_train = self.X_train, self.y_train
                y_pred_train = original_model.predict(X_train)
                
                # Get probabilities if available
                y_prob_train = None
                if hasattr(original_model, 'predict_proba'):
                    probs = original_model.predict_proba(X_train)
                    if probs.shape[1] > 1:  # Binary classification
                        y_prob_train = probs[:, 1]
                        
                train_metrics = self.calculate_metrics(
                    y_true=y_train,
                    y_pred=y_pred_train,
                    y_prob=y_prob_train
                )
                
                # Get metrics on test set
                X_test, y_test = self.X_test, self.y_test
                y_pred_test = original_model.predict(X_test)
                
                # Get probabilities if available
                y_prob_test = None
                if hasattr(original_model, 'predict_proba'):
                    probs = original_model.predict_proba(X_test)
                    if probs.shape[1] > 1:  # Binary classification
                        y_prob_test = probs[:, 1]
                        
                test_metrics = self.calculate_metrics(
                    y_true=y_test,
                    y_pred=y_pred_test,
                    y_prob=y_prob_test
                )
                
                result['original_model']['metrics'] = {
                    'train': train_metrics,
                    'test': test_metrics
                }
                
                # Get confusion matrix if it's included in metrics
                if 'confusion_matrix' in test_metrics:
                    result['original_model']['confusion_matrix'] = test_metrics['confusion_matrix']
                    
                # Get ROC and PR curve data if available
                if hasattr(self, 'metrics_calculator') and y_prob_test is not None:
                    try:
                        from sklearn.metrics import roc_curve, precision_recall_curve
                        
                        # ROC curve
                        fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob_test)
                        result['original_model']['roc_curve'] = {
                            'fpr': fpr.tolist(),
                            'tpr': tpr.tolist(),
                            'thresholds': roc_thresholds.tolist()
                        }
                        
                        # PR curve
                        precision, recall, pr_thresholds = precision_recall_curve(y_test, y_prob_test)
                        result['original_model']['pr_curve'] = {
                            'precision': precision.tolist(),
                            'recall': recall.tolist(),
                            'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
                        }
                    except Exception as e:
                        print(f"Error calculating curve data for original model: {str(e)}")
                        
            except Exception as e:
                print(f"Error calculating metrics for original model: {str(e)}")
                result['original_model']['metrics_error'] = str(e)
        
        # 4. Alternative Models Information
        alternative_models = self.alternative_models
        if alternative_models:
            # Get comparison metrics for all alternative models
            try:
                alt_train_metrics = self.evaluate_alternative_models('train')
                alt_test_metrics = self.evaluate_alternative_models('test')
                
                # Convert DataFrame to dict format
                if not alt_train_metrics.empty:
                    alt_train_dict = alt_train_metrics.to_dict(orient='records')
                else:
                    alt_train_dict = []
                    
                if not alt_test_metrics.empty:
                    alt_test_dict = alt_test_metrics.to_dict(orient='records')
                else:
                    alt_test_dict = []
                
                # Store metrics
                result['alternative_models']['comparison'] = {
                    'train': alt_train_dict,
                    'test': alt_test_dict
                }
                
                # Detailed info for each alternative model
                result['alternative_models']['models'] = {}
                
                for name, model in alternative_models.items():
                    model_info = {
                        'model_type': model.__class__.__name__,
                        'model_params': model.get_params() if hasattr(model, 'get_params') else {},
                    }
                    
                    # Find this model's metrics in the comparison data
                    if alt_test_dict:
                        for model_data in alt_test_dict:
                            if model_data.get('model_name') == name:
                                model_info['test_metrics'] = {k: v for k, v in model_data.items() 
                                                             if k not in ['model_name', 'model_type']}
                                break
                    
                    if alt_train_dict:
                        for model_data in alt_train_dict:
                            if model_data.get('model_name') == name:
                                model_info['train_metrics'] = {k: v for k, v in model_data.items() 
                                                              if k not in ['model_name', 'model_type']}
                                break
                    
                    # Get feature importance if available
                    if hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                        model_info['feature_importance'] = dict(zip(dataset.features, importance))
                    elif hasattr(model, 'coef_'):
                        if len(model.coef_.shape) == 1:
                            importance = model.coef_
                        else:
                            importance = model.coef_[0]  # Take first class for multiclass
                        model_info['feature_importance'] = dict(zip(dataset.features, importance))
                        
                    result['alternative_models']['models'][name] = model_info
                    
            except Exception as e:
                print(f"Error processing alternative models: {str(e)}")
                result['alternative_models']['error'] = str(e)
        
        # 5. Surrogate Model Information (if available)
        if self.distillation_model is not None:
            surrogate = self.distillation_model
            
            result['surrogate_model'] = {
                'model_type': surrogate.__class__.__name__,
                'best_params': getattr(surrogate, 'best_params', {}),
                'metrics': {}
            }
            
            # Calculate metrics for surrogate model
            try:
                # Get metrics on train set
                X_train, y_train = self.X_train, self.y_train
                surrogate_probs_train = surrogate.predict(X_train)
                surrogate_pred_train = (surrogate_probs_train > 0.5).astype(int)
                
                # Get full probability distribution
                surrogate_prob_dist_train = surrogate.predict_proba(X_train)
                surrogate_prob_class1_train = surrogate_prob_dist_train[:, 1] if surrogate_prob_dist_train.shape[1] > 1 else surrogate_probs_train
                
                train_metrics = self.calculate_metrics(
                    y_true=y_train,
                    y_pred=surrogate_pred_train,
                    y_prob=surrogate_prob_class1_train
                )
                
                # Get metrics on test set
                X_test, y_test = self.X_test, self.y_test
                surrogate_probs_test = surrogate.predict(X_test)
                surrogate_pred_test = (surrogate_probs_test > 0.5).astype(int)
                
                # Get full probability distribution
                surrogate_prob_dist_test = surrogate.predict_proba(X_test)
                surrogate_prob_class1_test = surrogate_prob_dist_test[:, 1] if surrogate_prob_dist_test.shape[1] > 1 else surrogate_probs_test
                
                test_metrics = self.calculate_metrics(
                    y_true=y_test,
                    y_pred=surrogate_pred_test,
                    y_prob=surrogate_prob_class1_test
                )
                
                result['surrogate_model']['metrics'] = {
                    'train': train_metrics,
                    'test': test_metrics
                }
                
                # Get ROC and PR curve data
                try:
                    from sklearn.metrics import roc_curve, precision_recall_curve
                    
                    # ROC curve
                    fpr, tpr, roc_thresholds = roc_curve(y_test, surrogate_prob_class1_test)
                    result['surrogate_model']['roc_curve'] = {
                        'fpr': fpr.tolist(),
                        'tpr': tpr.tolist(),
                        'thresholds': roc_thresholds.tolist()
                    }
                    
                    # PR curve
                    precision, recall, pr_thresholds = precision_recall_curve(y_test, surrogate_prob_class1_test)
                    result['surrogate_model']['pr_curve'] = {
                        'precision': precision.tolist(),
                        'recall': recall.tolist(),
                        'thresholds': pr_thresholds.tolist() if len(pr_thresholds) > 0 else []
                    }
                except Exception as e:
                    print(f"Error calculating curve data for surrogate model: {str(e)}")
                    
                # Calculate distribution comparison metrics between original and surrogate
                if hasattr(dataset, 'model') and dataset.model is not None and hasattr(dataset.model, 'predict_proba'):
                    try:
                        from scipy import stats
                        
                        # Get original model predictions
                        original_probs = dataset.model.predict_proba(X_test)
                        original_prob_class1 = original_probs[:, 1] if original_probs.shape[1] > 1 else original_probs
                        
                        # KS test between distributions
                        ks_stat, ks_pval = stats.ks_2samp(original_prob_class1, surrogate_prob_class1_test)
                        
                        # Jensen-Shannon Divergence
                        from scipy.spatial.distance import jensenshannon
                        js_dist = jensenshannon(original_prob_class1, surrogate_prob_class1_test)
                        
                        # KL Divergence (with smoothing to avoid log(0))
                        epsilon = 1e-10
                        orig_smooth = np.clip(original_prob_class1, epsilon, 1-epsilon)
                        surr_smooth = np.clip(surrogate_prob_class1_test, epsilon, 1-epsilon)
                        
                        kl_div_orig_to_surr = np.sum(orig_smooth * np.log(orig_smooth / surr_smooth)) / len(orig_smooth)
                        kl_div_surr_to_orig = np.sum(surr_smooth * np.log(surr_smooth / orig_smooth)) / len(surr_smooth)
                        
                        # Store comparison metrics
                        result['surrogate_model']['distribution_comparison'] = {
                            'ks_statistic': float(ks_stat),
                            'ks_pvalue': float(ks_pval),
                            'jensen_shannon_distance': float(js_dist),
                            'kl_divergence_orig_to_surr': float(kl_div_orig_to_surr),
                            'kl_divergence_surr_to_orig': float(kl_div_surr_to_orig)
                        }
                    except Exception as e:
                        print(f"Error calculating distribution comparison: {str(e)}")
                
            except Exception as e:
                print(f"Error calculating metrics for surrogate model: {str(e)}")
                result['surrogate_model']['metrics_error'] = str(e)
        
        # 6. Performance Comparison Across All Models
        try:
            comparison_test = self.compare_all_models('test')
            comparison_train = self.compare_all_models('train')
            
            if not comparison_test.empty:
                result['performance_comparison']['test'] = comparison_test.to_dict(orient='records')
            
            if not comparison_train.empty:
                result['performance_comparison']['train'] = comparison_train.to_dict(orient='records')
                
            # Calculate the best model for each metric
            if not comparison_test.empty:
                metrics_columns = [col for col in comparison_test.columns 
                                  if col not in ['model_name', 'model_type']]
                
                best_models = {}
                for metric in metrics_columns:
                    # Skip non-numeric columns
                    if comparison_test[metric].dtype == 'object':
                        continue
                        
                    # Determine if higher is better (common knowledge for standard metrics)
                    higher_is_better = metric.lower() not in ['mse', 'rmse', 'error', 'loss', 'kl_divergence']
                    
                    if higher_is_better:
                        best_idx = comparison_test[metric].idxmax()
                    else:
                        best_idx = comparison_test[metric].idxmin()
                        
                    if best_idx is not None:
                        best_model = comparison_test.loc[best_idx, 'model_name']
                        best_value = comparison_test.loc[best_idx, metric]
                        
                        best_models[metric] = {
                            'model': best_model,
                            'value': best_value,
                            'higher_is_better': higher_is_better
                        }
                
                result['performance_comparison']['best_models'] = best_models
                
                # Find overall best model (based on accuracy for classification)
                if 'accuracy' in metrics_columns:
                    best_acc_idx = comparison_test['accuracy'].idxmax()
                    result['performance_comparison']['overall_best_model'] = {
                        'model': comparison_test.loc[best_acc_idx, 'model_name'],
                        'model_type': comparison_test.loc[best_acc_idx, 'model_type'],
                        'accuracy': comparison_test.loc[best_acc_idx, 'accuracy']
                    }
                elif len(metrics_columns) > 0:
                    # Use first metric if accuracy not available
                    metric = metrics_columns[0]
                    higher_is_better = metric.lower() not in ['mse', 'rmse', 'error', 'loss', 'kl_divergence']
                    
                    if higher_is_better:
                        best_idx = comparison_test[metric].idxmax()
                    else:
                        best_idx = comparison_test[metric].idxmin()
                        
                    result['performance_comparison']['overall_best_model'] = {
                        'model': comparison_test.loc[best_idx, 'model_name'],
                        'model_type': comparison_test.loc[best_idx, 'model_type'],
                        'primary_metric': {
                            'name': metric,
                            'value': comparison_test.loc[best_idx, metric]
                        }
                    }
        except Exception as e:
            print(f"Error in performance comparison: {str(e)}")
            result['performance_comparison']['error'] = str(e)
        
        # 7. Feature Importance Analysis (aggregate across models)
        try:
            # Collect feature importance from all models
            feature_importance = {}
            
            # From original model
            if (hasattr(dataset, 'model') and dataset.model is not None):
                original_model = dataset.model
                if hasattr(original_model, 'feature_importances_'):
                    importance = original_model.feature_importances_
                    feature_importance['original'] = dict(zip(dataset.features, importance))
                elif hasattr(original_model, 'coef_'):
                    if len(original_model.coef_.shape) == 1:
                        importance = original_model.coef_
                    else:
                        importance = original_model.coef_[0]  # Take first class for multiclass
                    feature_importance['original'] = dict(zip(dataset.features, importance))
            
            # From alternative models
            for name, model in alternative_models.items():
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    feature_importance[name] = dict(zip(dataset.features, importance))
                elif hasattr(model, 'coef_'):
                    if len(model.coef_.shape) == 1:
                        importance = model.coef_
                    else:
                        importance = model.coef_[0]  # Take first class for multiclass
                    feature_importance[name] = dict(zip(dataset.features, importance))
            
            # From surrogate model if available
            if (self.distillation_model is not None and
                hasattr(self.distillation_model, 'student_model')):
                student = self.distillation_model.student_model
                if hasattr(student, 'feature_importances_'):
                    importance = student.feature_importances_
                    feature_importance['surrogate'] = dict(zip(dataset.features, importance))
                elif hasattr(student, 'coef_'):
                    if len(student.coef_.shape) == 1:
                        importance = student.coef_
                    else:
                        importance = student.coef_[0]  # Take first class for multiclass
                    feature_importance['surrogate'] = dict(zip(dataset.features, importance))
            
            # Aggregate feature importance across models
            if feature_importance:
                # Calculate mean importance for each feature
                all_features = dataset.features
                aggregated_importance = {}
                
                for feature in all_features:
                    values = []
                    for model_name, importances in feature_importance.items():
                        if feature in importances:
                            values.append(abs(importances[feature]))  # Use absolute value for consistency
                    
                    if values:
                        aggregated_importance[feature] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values),
                            'values': dict(zip(feature_importance.keys(), 
                                              [imp.get(feature, 0) for imp in feature_importance.values()]))
                        }
                
                # Sort features by mean importance
                sorted_features = sorted(aggregated_importance.items(), 
                                        key=lambda x: x[1]['mean'], 
                                        reverse=True)
                
                result['feature_importance'] = {
                    'per_model': feature_importance,
                    'aggregated': aggregated_importance,
                    'top_features': [f[0] for f in sorted_features[:10]]  # Top 10 features
                }
        except Exception as e:
            print(f"Error in feature importance analysis: {str(e)}")
            result['feature_importance']['error'] = str(e)
        
        # 8. Convert numpy types to Python native types for JSON compatibility
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return convert_numpy_types(obj.tolist())
            elif isinstance(obj, pd.DataFrame):
                return convert_numpy_types(obj.to_dict(orient='records'))
            elif isinstance(obj, pd.Series):
                return convert_numpy_types(obj.to_dict())
            else:
                return obj
        
        return convert_numpy_types(result)
    
    @results.setter
    def results(self, value):
        self._results_data = value