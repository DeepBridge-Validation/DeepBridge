import typing as t
from pathlib import Path
import pandas as pd
import numpy as np
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
        auto_fit: bool = True
    ):
        """
        Initialize the experiment with configuration and data.

        Args:
            dataset: DBDataset instance with features, target, and optionally model or probabilities
            experiment_type: Type of experiment ("binary_classification", "regression", "forecasting")
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            config: Optional configuration dictionary
            auto_fit: Whether to automatically fit a model if dataset has probabilities
        """
        if experiment_type not in self.VALID_TYPES:
            raise ValueError(f"experiment_type must be one of {self.VALID_TYPES}")
            
        self.experiment_type = experiment_type
        self.dataset = dataset
        self.test_size = test_size
        self.random_state = random_state
        self.config = config or {}
        
        # Initialize metrics calculator based on experiment type
        if experiment_type == "binary_classification":
            self.metrics_calculator = Classification()
            
        # Initialize results storage
        self.results = {
            'train': {},
            'test': {}
        }
        
        # Initialize distillation model
        self.distillation_model = None
        
        # Perform train-test split
        self._prepare_data()
        
        # Auto-fit if enabled and dataset has probabilities
        if auto_fit and hasattr(dataset, 'original_prob') and dataset.original_prob is not None:
            self.fit(
                student_model_type=ModelType.XGB,
                temperature=1.0,
                alpha=0.5,
                use_probabilities=True,
                verbose=False
            )

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
            
        # Suprimir logs do Optuna se verbose=False
        if not verbose:
            import logging
            optuna_logger = logging.getLogger("optuna")
            optuna_logger_level = optuna_logger.getEffectiveLevel()
            optuna_logger.setLevel(logging.ERROR)  # Mostrar apenas erros

        if use_probabilities:
            if self.prob_train is None:
                raise ValueError("No teacher probabilities available. Set use_probabilities=False to use teacher model")
                
            # Choose between Surrogate Model and Knowledge Distillation
            if distillation_method.lower() == "surrogate":
                # Import em tempo de execução para evitar importação cíclica
                from deepbridge.distillation.techniques.surrogate import SurrogateModel
                
                # Create surrogate model from probabilities
                self.distillation_model = SurrogateModel.from_probabilities(
                    probabilities=self.prob_train,
                    student_model_type=student_model_type,
                    student_params=student_params,
                    random_state=self.random_state,
                    validation_split=validation_split,
                    n_trials=n_trials
                )
            elif distillation_method.lower() == "knowledge_distillation":
                # Import em tempo de execução para evitar importação cíclica
                from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
                
                # Create distillation model from probabilities
                self.distillation_model = KnowledgeDistillation.from_probabilities(
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
        else:
            if self.dataset.model is None:
                raise ValueError("No teacher model available. Set use_probabilities=True to use pre-calculated probabilities")
                
            # Only KnowledgeDistillation supports teacher model directly
            if distillation_method.lower() == "surrogate":
                # Surrogate method doesn't support direct use of teacher model
                raise ValueError("The surrogate method does not support direct use of teacher model. "
                                "Please set use_probabilities=True or use method='knowledge_distillation'")
            elif distillation_method.lower() == "knowledge_distillation":
                # Import em tempo de execução para evitar importação cíclica
                from deepbridge.distillation.techniques.knowledge_distillation import KnowledgeDistillation
                
                # Create distillation model from teacher model
                self.distillation_model = KnowledgeDistillation(
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
        
        # Treinar o modelo com o controle de verbosidade
        self.distillation_model.fit(self.X_train, self.y_train, verbose=verbose)
        
        # Continuar com a avaliação
        train_metrics = self._evaluate_distillation_model('train')
        self.results['train'] = train_metrics['metrics']
        
        test_metrics = self._evaluate_distillation_model('test')
        self.results['test'] = test_metrics['metrics']
        
        # Restaurar nível do logger do Optuna
        if not verbose:
            import logging
            optuna_logger = logging.getLogger("optuna")
            optuna_logger.setLevel(optuna_logger_level)
        
        return self
    
    def _evaluate_distillation_model(self, dataset: str = 'test') -> dict:
        """
        Avalia o modelo de destilação para o conjunto de dados especificado
        
        Args:
            dataset: Qual conjunto de dados avaliar ('train' ou 'test')
            
        Returns:
            dict: Dicionário contendo métricas de avaliação e previsões
        """
        print(f"\n=== Evaluating distillation model on {dataset} dataset ===")
        if dataset == 'train':
            X, y, prob = self.X_train, self.y_train, self.prob_train
        else:
            X, y, prob = self.X_test, self.y_test, self.prob_test
        
        # Obter probabilidades
        student_probs = self.distillation_model.predict(X)
        
        # CORREÇÃO: Convertendo probabilidades para predições binárias
        y_pred = (student_probs > 0.5).astype(int)
        
        # Obter probabilidades completas (para ambas as classes)
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
                    # Assume que a última coluna é a probabilidade da classe positiva
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
        
        # Calcular métricas usando a classe Classification
        metrics = self.metrics_calculator.calculate_metrics(
            y_true=y,
            y_pred=y_pred,  # Agora usando predições binárias
            y_prob=student_prob_pos,  # Probabilidade da classe positiva
            teacher_prob=teacher_prob_pos if prob is not None else None  # Adicionar probabilidade do professor
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
            
        # Include previsões
        predictions_df = pd.DataFrame({
            'y_true': y,
            'y_pred': y_pred,
            'y_prob': student_prob_pos  # Probabilidade da classe positiva
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
        
        # CORREÇÃO: Converter para predições binárias
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

    @property
    def metrics(self) -> dict:
        """
        Get all metrics for both train and test datasets.
        """
        # Calculate metrics if they haven't been calculated yet
        if not self.results['train'] and self.prob_train is not None:
            binary_preds = self._get_binary_predictions(self.prob_train)
            prob_values = self.prob_train.iloc[:, -1] if len(self.prob_train.columns) > 1 else self.prob_train.iloc[:, 0]
            
            metrics = self.calculate_metrics(
                y_true=self.y_train,
                y_pred=binary_preds,
                y_prob=prob_values
            )
            self.results['train'] = metrics
            
        if not self.results['test'] and self.prob_test is not None:
            binary_preds = self._get_binary_predictions(self.prob_test)
            prob_values = self.prob_test.iloc[:, -1] if len(self.prob_test.columns) > 1 else self.prob_test.iloc[:, 0]
            
            metrics = self.calculate_metrics(
                y_true=self.y_test,
                y_pred=binary_preds,
                y_prob=prob_values
            )
            self.results['test'] = metrics
            
        return {
            'train': self.results['train'],
            'test': self.results['test']
        }