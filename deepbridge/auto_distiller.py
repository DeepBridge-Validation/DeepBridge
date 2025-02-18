import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
import time
from pathlib import Path

from deepbridge.distillation.classification.model_registry import ModelType
from deepbridge.experiment import Experiment
from deepbridge.db_data import DBDataset
from deepbridge.metrics.classification import Classification

class AutoDistiller:
    """
    Automated Knowledge Distillation tool for model compression.
    
    This class automates the process of knowledge distillation by testing
    multiple model types, temperatures, and alpha values to find the optimal 
    configuration for a given dataset.
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
        self.dataset = dataset
        self.output_dir = output_dir
        self.test_size = test_size
        self.random_state = random_state
        self.n_trials = n_trials
        self.validation_split = validation_split
        self.verbose = verbose
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize experiment
        self.experiment = Experiment(
            dataset=dataset,
            experiment_type="binary_classification",
            test_size=test_size,
            random_state=random_state
        )
        
        # Initialize metrics calculator
        self.metrics_calculator = Classification()
        
        # Set default configuration
        self._set_default_config()
        
        # Suppress Optuna logs if not verbose
        if not verbose:
            optuna_logger = logging.getLogger("optuna")
            optuna_logger.setLevel(logging.ERROR)
    
    def _set_default_config(self):
        """Set default configuration for model types, temperatures, and alphas."""
        self.model_types = [
            ModelType.LOGISTIC_REGRESSION,
            ModelType.DECISION_TREE,
            ModelType.GBM,
            ModelType.XGB
        ]
        
        self.temperatures = [0.5, 1.0, 2.0]
        self.alphas = [0.3, 0.5, 0.7]
        self.results = []
    
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
        if model_types is not None:
            self.model_types = model_types
        if temperatures is not None:
            self.temperatures = temperatures
        if alphas is not None:
            self.alphas = alphas
    
    def run(self, use_probabilities: bool = True) -> pd.DataFrame:
        """
        Run the automated distillation process.
        
        Args:
            use_probabilities: Whether to use pre-calculated probabilities or teacher model
        
        Returns:
            DataFrame containing results for all configurations
        """
        start_time = time.time()
        self.results = []
        
        if self.verbose:
            print(f"Starting AutoDistiller with {len(self.model_types)} models, "
                  f"{len(self.temperatures)} temperatures, and {len(self.alphas)} alpha values")
            print(f"Total configurations to test: {len(self.model_types) * len(self.temperatures) * len(self.alphas)}")
        
        # Testar todas as combinações
        for model_type in self.model_types:
            for temperature in self.temperatures:
                for alpha in self.alphas:
                    if self.verbose:
                        print(f"Testing: {model_type.name}, temp={temperature}, alpha={alpha}")
                    
                    # Executar fit com diferentes parâmetros
                    try:
                        self.experiment.fit(
                            student_model_type=model_type,
                            temperature=temperature,
                            alpha=alpha,
                            use_probabilities=use_probabilities,
                            n_trials=self.n_trials,
                            validation_split=self.validation_split,
                            verbose=False
                        )
                        
                        # Obter métricas
                        train_metrics = self.experiment.results['train']
                        test_metrics = self.experiment.results['test']
                        
                        # Armazenar resultados com todas as métricas disponíveis
                        result = {
                            'model_type': model_type.name,
                            'temperature': temperature,
                            'alpha': alpha,
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
                            'best_params': str(test_metrics.get('best_params', {}))
                        }
                        
                        self.results.append(result)
                        
                    except Exception as e:
                        if self.verbose:
                            print(f"Error running experiment: {e}")
                        self.results.append({
                            'model_type': model_type.name,
                            'temperature': temperature,
                            'alpha': alpha,
                            'error': str(e)
                        })
        
        # Transformar resultados em DataFrame
        self.results_df = pd.DataFrame(self.results)
        
        # Save results
        self._save_results()
        
        # Create visualizations
        self._create_visualizations()
        
        end_time = time.time()
        if self.verbose:
            print(f"AutoDistiller completed in {end_time - start_time:.2f} seconds")
            print(f"Results saved to {self.output_dir}")
        
        return self.results_df
    
    def _save_results(self):
        """Save results to CSV file."""
        results_path = os.path.join(self.output_dir, "distillation_results.csv")
        self.results_df.to_csv(results_path, index=False)
    
    def _create_visualizations(self):
        """Create and save visualizations of the results."""
        # Remove rows with error
        valid_results = self.results_df.dropna(subset=['test_accuracy'])
        
        if valid_results.empty:
            if self.verbose:
                print("No valid results to visualize")
            return
        
        self._plot_kl_divergence_by_temperature(valid_results)
        self._plot_accuracy_by_alpha(valid_results)
        self._plot_model_comparison(valid_results)
        
        # Plot additional metrics
        self._plot_metric_comparison(valid_results, 'f1')
        self._plot_metric_comparison(valid_results, 'auc_roc')
        self._plot_metric_comparison(valid_results, 'auc_pr')
    
    def _plot_kl_divergence_by_temperature(self, results_df: pd.DataFrame):
        """
        Plot KL divergence by temperature for each model type.
        
        Args:
            results_df: DataFrame containing valid results
        """
        plt.figure(figsize=(15, 10))
        model_types = results_df['model_type'].unique()
        
        for i, temp in enumerate(self.temperatures):
            plt.subplot(2, 2, i+1)
            
            temp_data = results_df[results_df['temperature'] == temp]
            models = []
            kl_means = []
            kl_stds = []
            
            for model in model_types:
                model_data = temp_data[temp_data['model_type'] == model]['test_kl_divergence']
                if not model_data.empty and not model_data.isna().all():
                    models.append(model)
                    kl_means.append(model_data.mean())
                    kl_stds.append(model_data.std())
            
            if models:  # Só plota se houver dados válidos
                x = range(len(models))
                plt.bar(x, kl_means, yerr=kl_stds, capsize=10)
                plt.xlabel('Model')
                plt.ylabel('KL Divergence (Test)')
                plt.title(f'KL Divergence with Temperature = {temp}')
                plt.xticks(x, models, rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
            else:
                plt.text(0.5, 0.5, 'No valid data for this temperature',
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'kl_divergence_by_temperature.png'))
        plt.close()
    
    def _plot_accuracy_by_alpha(self, results_df: pd.DataFrame):
        """
        Plot accuracy by alpha for each model type.
        
        Args:
            results_df: DataFrame containing valid results
        """
        plt.figure(figsize=(15, 10))
        model_types = results_df['model_type'].unique()
        
        for i, a in enumerate(self.alphas):
            plt.subplot(2, 2, i+1)
            
            alpha_data = results_df[results_df['alpha'] == a]
            models = []
            acc_means = []
            acc_stds = []
            
            for model in model_types:
                model_data = alpha_data[alpha_data['model_type'] == model]['test_accuracy']
                if not model_data.empty and not model_data.isna().all():
                    models.append(model)
                    acc_means.append(model_data.mean())
                    acc_stds.append(model_data.std())
            
            if models:  # Só plota se houver dados válidos
                x = range(len(models))
                plt.bar(x, acc_means, yerr=acc_stds, capsize=10)
                plt.xlabel('Model')
                plt.ylabel('Accuracy (Test)')
                plt.title(f'Accuracy with Alpha = {a}')
                plt.xticks(x, models, rotation=45)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
            else:
                plt.text(0.5, 0.5, 'No valid data for this alpha',
                       horizontalalignment='center', verticalalignment='center',
                       transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_by_alpha.png'))
        plt.close()
    
    def _plot_metric_comparison(self, results_df: pd.DataFrame, metric_name: str):
        """
        Plot comparison of a specific metric across models.
        
        Args:
            results_df: DataFrame containing valid results
            metric_name: Name of the metric to plot (without 'train_' or 'test_' prefix)
        """
        plt.figure(figsize=(15, 10))
        model_types = results_df['model_type'].unique()
        
        for i, temp in enumerate(self.temperatures):
            plt.subplot(len(self.temperatures), 1, i+1)
            
            temp_data = results_df[results_df['temperature'] == temp]
            models = []
            metric_values = []
            
            for model in model_types:
                model_data = temp_data[temp_data['model_type'] == model][f'test_{metric_name}']
                if not model_data.empty and not model_data.isna().all():
                    models.append(model)
                    alpha_values = []
                    for alpha in self.alphas:
                        alpha_data = temp_data[(temp_data['model_type'] == model) & 
                                               (temp_data['alpha'] == alpha)][f'test_{metric_name}']
                        if not alpha_data.empty and not alpha_data.isna().all():
                            alpha_values.append(alpha_data.mean())
                    metric_values.append(alpha_values)
            
            if models and metric_values:
                x = range(len(models))
                width = 0.2
                
                for j, alpha in enumerate(self.alphas):
                    alpha_vals = [vals[j] if j < len(vals) else None for vals in metric_values]
                    alpha_vals = [v for v in alpha_vals if v is not None]
                    if alpha_vals:
                        plt.bar([pos + j*width for pos in x[:len(alpha_vals)]], 
                                alpha_vals, 
                                width=width, 
                                label=f'Alpha={alpha}')
                
                plt.xlabel('Model')
                plt.ylabel(f'{metric_name.upper()} (Test)')
                plt.title(f'{metric_name.upper()} with Temperature = {temp}')
                plt.xticks([pos + width for pos in x[:len(models)]], models, rotation=45)
                plt.legend()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
            else:
                plt.text(0.5, 0.5, f'No valid {metric_name} data for this temperature',
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{metric_name}_comparison.png'))
        plt.close()
    
    def _plot_model_comparison(self, results_df: pd.DataFrame):
        """
        Plot overall model comparison.
        
        Args:
            results_df: DataFrame containing valid results
        """
        plt.figure(figsize=(12, 8))
        
        model_types = results_df['model_type'].unique()
        model_metrics = []
        
        for model in model_types:
            model_data = results_df[results_df['model_type'] == model]
            model_metrics.append({
                'model': model,
                'avg_accuracy': model_data['test_accuracy'].mean(),
                'max_accuracy': model_data['test_accuracy'].max(),
                'avg_f1': model_data['test_f1'].mean() if 'test_f1' in model_data else None,
                'max_f1': model_data['test_f1'].max() if 'test_f1' in model_data else None,
                'avg_auc_roc': model_data['test_auc_roc'].mean() if 'test_auc_roc' in model_data else None,
                'max_auc_roc': model_data['test_auc_roc'].max() if 'test_auc_roc' in model_data else None,
                'avg_auc_pr': model_data['test_auc_pr'].mean() if 'test_auc_pr' in model_data else None,
                'max_auc_pr': model_data['test_auc_pr'].max() if 'test_auc_pr' in model_data else None,
                'avg_kl_div': model_data['test_kl_divergence'].mean(),
                'min_kl_div': model_data['test_kl_divergence'].min()
            })
        
        model_metrics_df = pd.DataFrame(model_metrics)
        
        # Plot max accuracy and min KL divergence
        x = range(len(model_types))
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot accuracy bars
        bars = ax1.bar(x, model_metrics_df['max_accuracy'], color='royalblue', alpha=0.7)
        ax1.set_xlabel('Model Type')
        ax1.set_ylabel('Max Accuracy', color='royalblue')
        ax1.tick_params(axis='y', labelcolor='royalblue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_metrics_df['model'], rotation=45)
        
        # Add accuracy values on top of bars
        for bar, value in zip(bars, model_metrics_df['max_accuracy']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', color='royalblue', fontweight='bold')
        
        # Create second y-axis for KL divergence
        ax2 = ax1.twinx()
        line = ax2.plot(x, model_metrics_df['min_kl_div'], 'ro-', linewidth=2, markersize=8)
        ax2.set_ylabel('Min KL Divergence', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add KL divergence values
        for i, value in enumerate(model_metrics_df['min_kl_div']):
            ax2.text(i, value + 0.02, f'{value:.3f}', ha='center', va='bottom', color='red')
        
        plt.title('Model Comparison: Maximum Accuracy and Minimum KL Divergence')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'))
        plt.close()
        
        # Plot additional metrics comparison 
        metrics = ['accuracy', 'f1', 'auc_roc', 'auc_pr', 'kl_div']
        metric_labels = ['Accuracy', 'F1 Score', 'AUC-ROC', 'AUC-PR', 'KL Divergence']
        
        plt.figure(figsize=(18, 12))
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            plt.subplot(3, 2, i+1)
            
            if metric == 'kl_div':
                # For KL divergence, we want the minimum value
                values = [m[f'min_{metric}'] for m in model_metrics]
                title = f'Minimum {label} by Model'
            else:
                # For other metrics, we want the maximum value
                values = [m[f'max_{metric}'] for m in model_metrics]
                title = f'Maximum {label} by Model'
            
            bars = plt.bar(range(len(model_types)), values, color='steelblue')
            plt.xlabel('Model Type')
            plt.ylabel(label)
            plt.title(title)
            plt.xticks(range(len(model_types)), [m['model'] for m in model_metrics], rotation=45)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add values on top of bars
            for bar, value in zip(bars, values):
                if pd.notnull(value):  # Check if the value is not NaN
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                            f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'all_metrics_comparison.png'))
        plt.close()
    
    def find_best_model(self, metric: str = 'test_accuracy', minimize: bool = False) -> Dict:
        """
        Find the best model configuration based on a specific metric.
        
        Args:
            metric: Metric to use for finding the best model (default: 'test_accuracy')
            minimize: Whether the metric should be minimized (default: False)
        
        Returns:
            Dictionary containing the best model configuration
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("No results available. Run the distillation process first.")
        
        valid_results = self.results_df.dropna(subset=[metric])
        
        if valid_results.empty:
            raise ValueError(f"No valid results for metric: {metric}")
        
        if minimize:
            best_idx = valid_results[metric].idxmin()
        else:
            best_idx = valid_results[metric].idxmax()
        
        best_config = valid_results.loc[best_idx].to_dict()
        
        if self.verbose:
            print(f"Best model configuration based on {metric}:")
            for key, value in best_config.items():
                print(f"  {key}: {value}")
        
        return best_config
    
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
        self.experiment.fit(
            student_model_type=model_type,
            temperature=temperature,
            alpha=alpha,
            use_probabilities=True,
            n_trials=self.n_trials,
            validation_split=self.validation_split,
            verbose=self.verbose
        )
        
        return self.experiment.distillation_model
    
    def generate_report(self):
        """
        Generate a comprehensive report of the distillation results.
        
        Returns:
            String containing the report
        """
        if not hasattr(self, 'results_df'):
            raise ValueError("No results available. Run the distillation process first.")
        
        valid_results = self.results_df.dropna(subset=['test_accuracy'])
        
        if valid_results.empty:
            return "No valid results to generate report"
        
        report = ["# Knowledge Distillation Report\n"]
        
        # Add general information
        report.append("## General Information")
        report.append(f"- Number of models tested: {len(self.model_types)}")
        report.append(f"- Temperatures tested: {self.temperatures}")
        report.append(f"- Alpha values tested: {self.alphas}")
        report.append(f"- Total configurations: {len(self.model_types) * len(self.temperatures) * len(self.alphas)}")
        report.append(f"- Valid results: {len(valid_results)}")
        report.append("")
        
        # Add best configurations for each metric
        report.append("## Best Configurations")
        
        # Define all metrics to find best models for
        metrics = [
            ('test_accuracy', False, 'Test Accuracy'),
            ('test_f1', False, 'F1 Score'),
            ('test_auc_roc', False, 'AUC-ROC'),
            ('test_auc_pr', False, 'AUC-PR'),
            ('test_kl_divergence', True, 'KL Divergence')
        ]
        
        for metric, minimize, metric_name in metrics:
            try:
                best = self.find_best_model(metric=metric, minimize=minimize)
                report.append(f"### Best Model by {metric_name}")
                report.append(f"- Model Type: {best['model_type']}")
                report.append(f"- Temperature: {best['temperature']}")
                report.append(f"- Alpha: {best['alpha']}")
                report.append(f"- Test Accuracy: {best.get('test_accuracy', 'N/A')}")
                report.append(f"- Test F1: {best.get('test_f1', 'N/A')}")
                report.append(f"- Test AUC-ROC: {best.get('test_auc_roc', 'N/A')}")
                report.append(f"- Test AUC-PR: {best.get('test_auc_pr', 'N/A')}")
                report.append(f"- KL Divergence (Test): {best.get('test_kl_divergence', 'N/A')}")
                report.append(f"- Parameters: {best.get('best_params', 'N/A')}")
                report.append("")
            except (ValueError, KeyError) as e:
                report.append(f"### Best Model by {metric_name}")
                report.append(f"Unable to find best model: {str(e)}")
                report.append("")
        
        # Add model comparison
        report.append("## Model Comparison")
        model_comparison = valid_results.groupby('model_type').agg({
            'test_accuracy': ['mean', 'max', 'std'],
            'train_accuracy': ['mean', 'max', 'std'],
            'test_f1': ['mean', 'max', 'std'] if 'test_f1' in valid_results.columns else None,
            'test_auc_roc': ['mean', 'max', 'std'] if 'test_auc_roc' in valid_results.columns else None,
            'test_auc_pr': ['mean', 'max', 'std'] if 'test_auc_pr' in valid_results.columns else None,
            'test_kl_divergence': ['mean', 'min', 'std']
        }).reset_index()
        
        # Filter out None values from the aggregation
        model_comparison = model_comparison.dropna(axis=1, how='all')
        
        model_comparison_str = model_comparison.to_string()
        report.append("```")
        report.append(model_comparison_str)
        report.append("```")
        report.append("")
        
        # Add temperature impact
        report.append("## Impact of Temperature")
        metrics_columns = ['test_accuracy', 'test_f1', 'test_auc_roc', 'test_auc_pr', 'test_kl_divergence']
        available_metrics = [col for col in metrics_columns if col in valid_results.columns]
        
        temp_impact = valid_results.groupby(['model_type', 'temperature'])[available_metrics].agg('mean').reset_index()
        
        temp_impact_str = temp_impact.to_string()
        report.append("```")
        report.append(temp_impact_str)
        report.append("```")
        report.append("")
        
        # Add alpha impact
        report.append("## Impact of Alpha")
        alpha_impact = valid_results.groupby(['model_type', 'alpha'])[available_metrics].agg('mean').reset_index()
        
        alpha_impact_str = alpha_impact.to_string()
        report.append("```")
        report.append(alpha_impact_str)
        report.append("```")
        
        # Save report
        report_path = os.path.join(self.output_dir, "distillation_report.md")
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
        
        if self.verbose:
            print(f"Report saved to {report_path}")
        
        return '\n'.join(report)