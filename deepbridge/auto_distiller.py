import os
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Union

from deepbridge.distillation.classification.model_registry import ModelType
from deepbridge.db_data import DBDataset

# Import components from auto submodule
from deepbridge.auto.config import DistillationConfig
from deepbridge.auto.experiment_runner import ExperimentRunner
from deepbridge.auto.metrics import MetricsEvaluator
from deepbridge.auto.visualization import Visualizer
from deepbridge.auto.reporting import ReportGenerator

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
        verbose: bool = True
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
        """
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
    
    def customize_config(
        self,
        model_types: Optional[List[ModelType]] = None,
        temperatures: Optional[List[float]] = None,
        alphas: Optional[List[float]] = None
    ):
        """
        Customize the configuration for distillation experiments.
        
        Args:
            model_types: List of ModelType to test (defaults to standard list if None)
            temperatures: List of temperature values to test (defaults to [0.5, 1.0, 2.0] if None)
            alphas: List of alpha values to test (defaults to [0.3, 0.5, 0.7] if None)
        """
        self.config.customize(
            model_types=model_types,
            temperatures=temperatures,
            alphas=alphas
        )
    
    def run(self, use_probabilities: bool = True) -> pd.DataFrame:
        """
        Run the automated distillation process.
        
        Args:
            use_probabilities: Whether to use pre-calculated probabilities or teacher model
        
        Returns:
            DataFrame containing results for all configurations
        """
        # Run experiments
        self.results_df = self.experiment_runner.run_experiments(
            use_probabilities=use_probabilities
        )
        
        # Save results
        self.experiment_runner.save_results()
        
        # Initialize components that depend on results
        self._initialize_analysis_components()
        
        # Create visualizations
        self.visualizer.create_all_visualizations()
        
        # Generate and save report
        self.report_generator.save_report()
        
        return self.results_df
    
    def _initialize_analysis_components(self):
        """Initialize components for analysis after experiments are run."""
        # Initialize metrics evaluator
        self.metrics_evaluator = MetricsEvaluator(
            results_df=self.results_df,
            config=self.config
        )
        
        # Initialize visualizer
        self.visualizer = Visualizer(
            results_df=self.results_df,
            config=self.config,
            metrics_evaluator=self.metrics_evaluator
        )
        
        # Initialize report generator
        self.report_generator = ReportGenerator(
            results_df=self.results_df,
            config=self.config,
            metrics_evaluator=self.metrics_evaluator
        )
        
    def _ensure_components_initialized(self):
        """Ensure that analysis components are initialized if results are available."""
        if self.results_df is not None and (
            self.metrics_evaluator is None or
            self.visualizer is None or
            self.report_generator is None
        ):
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
        return self.metrics_evaluator.find_best_model(metric=metric, minimize=minimize)
    
    def get_trained_model(self, model_type: ModelType, temperature: float, alpha: float):
        """
        Get a trained model with specific configuration.
        
        Args:
            model_type: Type of model to train
            temperature: Temperature parameter
            alpha: Alpha parameter
        
        Returns:
            Trained distillation model
        """
        return self.experiment_runner.get_trained_model(
            model_type=model_type,
            temperature=temperature,
            alpha=alpha
        )
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive report of the distillation results.
        
        Returns:
            String containing the report
        """
        if self.results_df is None:
            raise ValueError("No results available. Run the distillation process first.")
            
        self._ensure_components_initialized()
        return self.report_generator.generate_report()
    
    def generate_summary(self) -> str:
        """
        Generate a brief summary of the distillation results.
        
        Returns:
            String containing a summary
        """
        if self.results_df is None:
            raise ValueError("No results available. Run the distillation process first.")
            
        self._ensure_components_initialized()
        return self.report_generator.generate_summary()
    
    def create_visualizations(self):
        """Create and save all visualizations."""
        if self.results_df is None:
            raise ValueError("No results available. Run the distillation process first.")
            
        self._ensure_components_initialized()
        self.visualizer.create_all_visualizations()