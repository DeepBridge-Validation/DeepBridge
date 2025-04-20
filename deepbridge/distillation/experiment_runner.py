import os
import time
import pandas as pd
from typing import List, Dict, Any, Optional

from deepbridge.core.experiment import Experiment
from deepbridge.core.db_data import DBDataset
from deepbridge.utils.model_registry import ModelType
from deepbridge.metrics.classification import Classification

from deepbridge.config.settings import DistillationConfig

class ExperimentRunner:
    """
    Handles the execution of distillation experiments.
    
    Manages experiment setup, execution, and result collection for
    multiple model configurations.
    """
    
    def __init__(
        self,
        dataset: DBDataset,
        config: DistillationConfig
    ):
        """
        Initialize the experiment runner.
        
        Args:
            dataset: DBDataset instance containing features, target, and probabilities
            config: Configuration for the experiments
        """
        self.dataset = dataset
        self.config = config
        self.results = []
        
        # Create output directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize experiment
        self.experiment = Experiment(
            dataset=dataset,
            experiment_type="binary_classification",
            test_size=config.test_size,
            random_state=config.random_state
        )
        
        # Initialize metrics calculator
        self.metrics_calculator = Classification()
    
    def run_experiments(self, use_probabilities: bool = True, distillation_method: str = "knowledge_distillation") -> pd.DataFrame:
        """
        Run distillation experiments for all configurations.
        
        Args:
            use_probabilities: Whether to use pre-calculated probabilities or teacher model
            distillation_method: Method to use for distillation ('surrogate' or 'knowledge_distillation')
        
        Returns:
            DataFrame containing results for all configurations
        """
        start_time = time.time()
        self.results = []
        
        self.config.log_info(f"Starting experiments with {len(self.config.model_types)} models, "
                             f"{len(self.config.temperatures)} temperatures, and {len(self.config.alphas)} alpha values")
        self.config.log_info(f"Total configurations to test: {self.config.get_total_configurations()}")
        self.config.log_info(f"Using distillation method: {distillation_method}")
        
        # Test all combinations
        for model_type in self.config.model_types:
            for temperature in self.config.temperatures:
                for alpha in self.config.alphas:
                    self.config.log_info(f"Testing: {model_type.name}, temp={temperature}, alpha={alpha}")
                    
                    result = self._run_single_experiment(
                        model_type=model_type,
                        temperature=temperature,
                        alpha=alpha,
                        use_probabilities=use_probabilities,
                        distillation_method=distillation_method
                    )
                    self.results.append(result)
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        end_time = time.time()
        self.config.log_info(f"Experiments completed in {end_time - start_time:.2f} seconds")
        
        return self.results_df
    
    def _run_single_experiment(
        self,
        model_type: ModelType,
        temperature: float,
        alpha: float,
        use_probabilities: bool,
        distillation_method: str = "knowledge_distillation"
    ) -> Dict[str, Any]:
        """
        Run a single distillation experiment with specific parameters.
        
        Args:
            model_type: Type of model to use
            temperature: Temperature value for distillation
            alpha: Alpha value for distillation
            use_probabilities: Whether to use pre-calculated probabilities
            distillation_method: Method to use for distillation ('surrogate' or 'knowledge_distillation')
        
        Returns:
            Dictionary containing experiment results
        """
        try:
            self.experiment.fit(
                student_model_type=model_type,
                temperature=temperature,
                alpha=alpha,
                use_probabilities=use_probabilities,
                n_trials=self.config.n_trials,
                validation_split=self.config.validation_split,
                verbose=False,
                distillation_method=distillation_method
            )
            
            # Get metrics from experiment results
            train_metrics = self.experiment.results['train']
            test_metrics = self.experiment.results['test']
            
            # Store results with all available metrics
            result = {
                'model_type': model_type.name,
                'temperature': temperature,
                'alpha': alpha,
                'distillation_method': distillation_method,
                'train_accuracy': train_metrics.get('accuracy', None),
                'test_accuracy': test_metrics.get('accuracy', None),
                'train_precision': train_metrics.get('precision', None),
                'test_precision': test_metrics.get('precision', None),
                'train_recall': train_metrics.get('recall', None),
                'test_recall': test_metrics.get('recall', None),
                'train_f1': train_metrics.get('f1_score', None),
                'test_f1': test_metrics.get('f1_score', None),
                'train_auc_roc': train_metrics.get('auc_roc', None),
                'test_auc_roc': test_metrics.get('auc_roc', None),
                'train_auc_pr': train_metrics.get('auc_pr', None),
                'test_auc_pr': test_metrics.get('auc_pr', None),
                'train_kl_divergence': train_metrics.get('kl_divergence', None),
                'test_kl_divergence': test_metrics.get('kl_divergence', None),
                # Add new metrics
                'train_ks_statistic': train_metrics.get('ks_statistic', None),
                'test_ks_statistic': test_metrics.get('ks_statistic', None),
                'train_ks_pvalue': train_metrics.get('ks_pvalue', None),
                'test_ks_pvalue': test_metrics.get('ks_pvalue', None),
                'train_r2_score': train_metrics.get('r2_score', None),
                'test_r2_score': test_metrics.get('r2_score', None),
                'best_params': str(test_metrics.get('best_params', {}))
            }
            
            # Incluir informação sobre o método de distilação usado nos resultados
            if 'distillation_method' in test_metrics:
                result['distillation_class'] = test_metrics['distillation_method']
            
            return result
            
        except Exception as e:
            self.config.log_info(f"Error running experiment: {e}")
            return {
                'model_type': model_type.name,
                'temperature': temperature,
                'alpha': alpha,
                'distillation_method': distillation_method,
                'error': str(e)
            }
    
    def get_trained_model(
        self, 
        model_type: ModelType, 
        temperature: float, 
        alpha: float, 
        distillation_method: str = "knowledge_distillation"
    ):
        """
        Get a trained model with specific configuration.
        
        Args:
            model_type: Type of model to train
            temperature: Temperature parameter
            alpha: Alpha parameter
            distillation_method: Method to use for distillation ('surrogate' or 'knowledge_distillation')
        
        Returns:
            Trained distillation model
        """
        self.experiment.fit(
            student_model_type=model_type,
            temperature=temperature,
            alpha=alpha,
            use_probabilities=True,
            n_trials=self.config.n_trials,
            validation_split=self.config.validation_split,
            verbose=self.config.verbose,
            distillation_method=distillation_method
        )
        
        return self.experiment.distillation_model
    
    def save_results(self):
        """Save results to CSV file."""
        if hasattr(self, 'results_df'):
            results_path = os.path.join(self.config.output_dir, "distillation_results.csv")
            self.results_df.to_csv(results_path, index=False, sep=';')
            self.config.log_info(f"Results saved to {results_path}")
        else:
            raise ValueError("No results available. Run experiments first.")