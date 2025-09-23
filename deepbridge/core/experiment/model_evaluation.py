import typing as t
import pandas as pd
import numpy as np
from scipy import stats

class ModelEvaluation:
    """
    Handles model evaluation, metric calculation, and model comparison.
    """
    
    def __init__(self, experiment_type, metrics_calculator):
        self.experiment_type = experiment_type
        self.metrics_calculator = metrics_calculator
    
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
    
    def evaluate_distillation(self, model, dataset, X, y, prob=None):
        """
        Evaluate the distillation model for the specified dataset.
        """
        import logging
        logger = logging.getLogger("deepbridge.evaluation")

        logger.debug(f"Evaluating distillation model on {dataset} dataset")

        # Get probabilities
        student_probs = model.predict(X)

        # Convert probabilities to binary predictions
        y_pred = (student_probs > 0.5).astype(int)

        # Get full probabilities (for both classes)
        y_prob = model.predict_proba(X)

        logger.debug(f"Student predictions shape: {y_prob.shape}")
        logger.debug(f"First 3 student probabilities: {y_prob[:3]}")
        
        # Extract probability of positive class for student
        student_prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else student_probs
        
        # Prepare teacher probabilities
        teacher_prob_pos = None
        if prob is not None:
            logger.debug(f"Teacher probabilities type: {type(prob)}")
            if isinstance(prob, pd.DataFrame):
                if 'prob_class_1' in prob.columns:
                    logger.debug(f"Using 'prob_class_1' column from teacher probabilities")
                    teacher_prob_pos = prob['prob_class_1'].values

                    # Verifica se a coluna prob_class_0 existe, caso contrário, calcula-a
                    if 'prob_class_0' in prob.columns:
                        logger.debug(f"Found 'prob_class_0' column in teacher probabilities")
                        teacher_probs = prob[['prob_class_0', 'prob_class_1']].values
                    else:
                        logger.debug(f"Calculating 'prob_class_0' as (1 - prob_class_1)")
                        # Calcula prob_class_0 como 1 - prob_class_1
                        teacher_probs = np.column_stack([1 - teacher_prob_pos, teacher_prob_pos])
                else:
                    # Assume the last column is the probability of the positive class
                    logger.debug(f"Using last column as positive class probability")
                    pos_prob = prob.iloc[:, -1].values
                    teacher_prob_pos = pos_prob
                    teacher_probs = np.column_stack([1 - pos_prob, pos_prob])
            else:
                teacher_probs = prob
                teacher_prob_pos = prob[:, 1] if prob.shape[1] > 1 else prob

            logger.debug(f"Teacher probabilities shape: {teacher_probs.shape if hasattr(teacher_probs, 'shape') else 'unknown'}")
            logger.debug(f"First 3 teacher probabilities (positive class): {teacher_prob_pos[:3]}")

            # Calculate distribution comparison metrics
            ks_stat, ks_pvalue, r2 = self._calculate_distribution_metrics(teacher_prob_pos, student_prob_pos)
        else:
            logger.debug(f"No teacher probabilities available for {dataset} dataset")
            ks_stat, ks_pvalue, r2 = None, None, None
        
        # Calculate metrics using the metrics calculator
        metrics = self.metrics_calculator.calculate_metrics(
            y_true=y,
            y_pred=y_pred,
            y_prob=student_prob_pos,
            teacher_prob=teacher_prob_pos
        )
        
        # Add distribution metrics if not present
        self._add_distribution_metrics(metrics, teacher_prob_pos, student_prob_pos, ks_stat, ks_pvalue, r2)
        
        # Include best hyperparameters and distillation method in metrics
        if hasattr(model, 'best_params') and model.best_params:
            metrics['best_params'] = model.best_params
            
        metrics['distillation_method'] = getattr(model, '__class__', 'unknown').__name__
            
        # Include predictions
        predictions_df = pd.DataFrame({
            'y_true': y,
            'y_pred': y_pred,
            'y_prob': student_prob_pos
        })
        
        if teacher_prob_pos is not None:
            # Add teacher probabilities to predictions dataframe
            predictions_df['teacher_prob'] = teacher_prob_pos
        
        logger.info(f"Evaluation metrics: {metrics}")
        logger.debug(f"Evaluation complete for {dataset} dataset")
        
        return {'metrics': metrics, 'predictions': predictions_df}
    
    def _calculate_distribution_metrics(self, teacher_probs, student_probs):
        """Calculate statistical metrics comparing distributions"""
        try:
            # KS statistic
            ks_stat, ks_pvalue = stats.ks_2samp(teacher_probs, student_probs)
            
            # R² score
            from sklearn.metrics import r2_score
            # Sort distributions
            teacher_sorted = np.sort(teacher_probs)
            student_sorted = np.sort(student_probs)
            # Use equal lengths
            min_len = min(len(teacher_sorted), len(student_sorted))
            r2 = r2_score(teacher_sorted[:min_len], student_sorted[:min_len])
            
            return ks_stat, ks_pvalue, r2
        except Exception as e:
            print(f"Error calculating distribution metrics: {str(e)}")
            return None, None, None

    def _add_distribution_metrics(self, metrics, teacher_probs, student_probs, ks_stat, ks_pvalue, r2):
        """Add distribution comparison metrics to the metrics dictionary"""
        # Add KS statistic if not present
        if 'ks_statistic' not in metrics or metrics['ks_statistic'] is None:
            metrics['ks_statistic'] = ks_stat
            metrics['ks_pvalue'] = ks_pvalue
            
        # Add R² score if not present    
        if 'r2_score' not in metrics or metrics['r2_score'] is None:
            metrics['r2_score'] = r2
        
        # Add KL divergence if not present and we have teacher probabilities
        if 'kl_divergence' not in metrics and teacher_probs is not None:
            try:
                # Calculate KL divergence manually
                # Add epsilon to avoid log(0)
                epsilon = 1e-10
                teacher_prob_pos = np.clip(teacher_probs, epsilon, 1-epsilon)
                student_prob_pos = np.clip(student_probs, epsilon, 1-epsilon)
                
                # For binary classification (calculate for both classes)
                teacher_prob_neg = 1 - teacher_prob_pos
                student_prob_neg = 1 - student_prob_pos
                
                # Calculate KL divergence
                kl_div_pos = np.mean(teacher_prob_pos * np.log(teacher_prob_pos / student_prob_pos))
                kl_div_neg = np.mean(teacher_prob_neg * np.log(teacher_prob_neg / student_prob_neg))
                kl_div = (kl_div_pos + kl_div_neg) / 2
                
                metrics['kl_divergence'] = kl_div
            except Exception as e:
                print(f"Error calculating KL divergence: {str(e)}")
                metrics['kl_divergence'] = None
                
    def get_predictions(self, model, X, y_true):
        """Get predictions from a model"""
        # Get probabilities
        probs = model.predict(X)
        
        # Convert to binary predictions
        y_pred = (probs > 0.5).astype(int)
        
        # Get probability distributions
        y_prob = model.predict_proba(X)
        
        # Create DataFrame
        predictions = pd.DataFrame({
            'y_true': y_true,
            'y_pred': y_pred,
            'prob_0': y_prob[:, 0],
            'prob_1': y_prob[:, 1]
        })
        
        return predictions
    
    def evaluate_model(self, model, model_name, model_type, X, y):
        """Evaluate a single model"""
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
            print(f"Failed to evaluate model {model_name}: {str(e)}")
            return None
    
    def compare_all_models(self, dataset, original_model, alternative_models, distilled_model, X, y):
        """Compare all models on the specified dataset"""
        results = []
        
        # Add original model if available
        if original_model is not None:
            original_name = original_model.__class__.__name__
            metrics = self.evaluate_model(original_model, original_name, 'original', X, y)
            if metrics:
                results.append(metrics)
        
        # Add alternative models
        for name, model in alternative_models.items():
            metrics = self.evaluate_model(model, name, 'alternative', X, y)
            if metrics:
                results.append(metrics)
        
        # Add distilled model if available
        if distilled_model is not None:
            try:
                # Get predictions
                student_probs = distilled_model.predict(X)
                y_pred = (student_probs > 0.5).astype(int)
                
                # Get probability distributions
                y_prob = distilled_model.predict_proba(X)
                student_prob_pos = y_prob[:, 1] if y_prob.shape[1] > 1 else student_probs
                
                # Calculate metrics
                metrics = self.calculate_metrics(
                    y_true=y,
                    y_pred=y_pred,
                    y_prob=student_prob_pos
                )
                
                # Add model info
                metrics['model_name'] = 'Distilled_' + getattr(distilled_model, '__class__', type(distilled_model)).__name__
                metrics['model_type'] = 'distilled'
                
                results.append(metrics)
            except Exception as e:
                print(f"Failed to evaluate distilled model: {str(e)}")
        
        # Convert results to DataFrame
        comparison_df = pd.DataFrame(results)
        
        # Reorder columns to put model info first
        if not comparison_df.empty:
            cols = ['model_name', 'model_type'] + [col for col in comparison_df.columns if col not in ['model_name', 'model_type']]
            comparison_df = comparison_df[cols]
        
        return comparison_df
