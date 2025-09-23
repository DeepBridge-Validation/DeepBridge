"""
Data transformer for distillation reports.
Transforms raw distillation results into a format suitable for report generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("deepbridge.reports")


class DistillationDataTransformer:
    """
    Transforms distillation experiment results for report generation.
    """

    def transform(self, results: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform raw distillation results into report-ready format.

        Args:
            results: Dictionary containing:
                - 'results': DataFrame with distillation results
                - 'original_metrics': Metrics of the original model
                - 'best_model': Best model configuration
                - 'config': Experiment configuration
            config: Optional additional configuration

        Returns:
            Dictionary with transformed data ready for report rendering
        """
        logger.info("Transforming distillation data for report")

        # Extract components
        results_df = results.get('results', pd.DataFrame())
        original_metrics = results.get('original_metrics', {})
        best_model = results.get('best_model', {})
        experiment_config = results.get('config', config or {})

        # Transform the data
        transformed = {
            'experiment_summary': self._create_experiment_summary(results_df, experiment_config),
            'original_model': self._transform_original_metrics(original_metrics),
            'best_model': self._transform_best_model(best_model, results_df),
            'all_models': self._transform_all_models(results_df),
            'hyperparameter_analysis': self._analyze_hyperparameters(results_df),
            'performance_comparison': self._create_performance_comparison(results_df, original_metrics),
            'tradeoff_analysis': self._analyze_tradeoffs(results_df, original_metrics),
            'recommendations': self._generate_recommendations(results_df, original_metrics),
            'config': experiment_config,
            'metadata': self._create_metadata(results_df)
        }

        logger.info(f"Transformation complete. Processed {len(results_df)} model configurations")
        return transformed

    def _create_experiment_summary(self, results_df: pd.DataFrame, config: Dict) -> Dict[str, Any]:
        """Create summary statistics for the experiment."""
        if results_df.empty:
            return self._empty_summary()

        return {
            'total_models_tested': len(results_df),
            'model_types': list(results_df['model_type'].unique()) if 'model_type' in results_df else [],
            'temperature_range': {
                'min': float(results_df['temperature'].min()) if 'temperature' in results_df else 0,
                'max': float(results_df['temperature'].max()) if 'temperature' in results_df else 0,
                'values': sorted(results_df['temperature'].unique().tolist()) if 'temperature' in results_df else []
            },
            'alpha_range': {
                'min': float(results_df['alpha'].min()) if 'alpha' in results_df else 0,
                'max': float(results_df['alpha'].max()) if 'alpha' in results_df else 0,
                'values': sorted(results_df['alpha'].unique().tolist()) if 'alpha' in results_df else []
            },
            'total_training_time': float(results_df['training_time'].sum()) if 'training_time' in results_df else 0,
            'average_training_time': float(results_df['training_time'].mean()) if 'training_time' in results_df else 0,
            'n_trials': config.get('n_trials', 0)
        }

    def _empty_summary(self) -> Dict[str, Any]:
        """Return empty summary structure."""
        return {
            'total_models_tested': 0,
            'model_types': [],
            'temperature_range': {'min': 0, 'max': 0, 'values': []},
            'alpha_range': {'min': 0, 'max': 0, 'values': []},
            'total_training_time': 0,
            'average_training_time': 0,
            'n_trials': 0
        }

    def _transform_original_metrics(self, original_metrics: Dict) -> Dict[str, Any]:
        """Transform original model metrics."""
        transformed = {
            'train_metrics': {},
            'test_metrics': {},
            'has_metrics': False
        }

        if original_metrics:
            if 'train' in original_metrics:
                transformed['train_metrics'] = self._clean_metrics(original_metrics['train'])
            if 'test' in original_metrics:
                transformed['test_metrics'] = self._clean_metrics(original_metrics['test'])
            transformed['has_metrics'] = bool(transformed['train_metrics'] or transformed['test_metrics'])

        return transformed

    def _transform_best_model(self, best_model: Dict, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Transform best model information."""
        if not best_model and not results_df.empty:
            # Find best model by test accuracy if not provided
            best_idx = results_df['test_accuracy'].idxmax() if 'test_accuracy' in results_df else 0
            best_model = results_df.iloc[best_idx].to_dict()

        if best_model:
            return {
                'model_type': best_model.get('model_type', 'Unknown'),
                'temperature': float(best_model.get('temperature', 1.0)),
                'alpha': float(best_model.get('alpha', 0.5)),
                'metrics': self._extract_model_metrics(best_model),
                'training_time': float(best_model.get('training_time', 0)),
                'model_complexity': float(best_model.get('model_complexity', 0)),
                'config_string': f"{best_model.get('model_type', 'Unknown')}_T{best_model.get('temperature', 0)}_A{best_model.get('alpha', 0)}"
            }

        return {
            'model_type': 'None',
            'temperature': 0,
            'alpha': 0,
            'metrics': {},
            'training_time': 0,
            'model_complexity': 0,
            'config_string': 'No model found'
        }

    def _transform_all_models(self, results_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Transform all model results."""
        models = []

        for idx, row in results_df.iterrows():
            model_data = {
                'id': idx,
                'model_type': row.get('model_type', 'Unknown'),
                'temperature': float(row.get('temperature', 0)),
                'alpha': float(row.get('alpha', 0)),
                'metrics': self._extract_model_metrics(row),
                'training_time': float(row.get('training_time', 0)),
                'model_complexity': float(row.get('model_complexity', 0)),
                'config_string': f"{row.get('model_type', 'Unknown')}_T{row.get('temperature', 0)}_A{row.get('alpha', 0)}"
            }
            models.append(model_data)

        return models

    def _extract_model_metrics(self, model_data: Any) -> Dict[str, float]:
        """Extract and clean metrics from model data."""
        metrics = {}
        metric_keys = [
            'train_accuracy', 'test_accuracy',
            'train_precision', 'test_precision',
            'train_recall', 'test_recall',
            'train_f1_score', 'test_f1_score', 'train_f1', 'test_f1',  # Support both f1_score and f1 naming
            'train_auc_roc', 'test_auc_roc',
            'train_auc_pr', 'test_auc_pr',
            'train_ks_statistic', 'test_ks_statistic'
        ]

        if isinstance(model_data, pd.Series):
            model_dict = model_data.to_dict()
        elif isinstance(model_data, dict):
            model_dict = model_data
        else:
            return metrics

        for key in metric_keys:
            if key in model_dict and pd.notna(model_dict[key]):
                metrics[key] = float(model_dict[key])

        # Handle naming compatibility: map test_f1 to test_f1_score for template compatibility
        if 'test_f1' in metrics and 'test_f1_score' not in metrics:
            metrics['test_f1_score'] = metrics['test_f1']
        if 'train_f1' in metrics and 'train_f1_score' not in metrics:
            metrics['train_f1_score'] = metrics['train_f1']

        return metrics

    def _analyze_hyperparameters(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze hyperparameter impact."""
        analysis = {
            'temperature_impact': {},
            'alpha_impact': {},
            'interaction_matrix': {},
            'optimal_ranges': {}
        }

        if results_df.empty or 'test_accuracy' not in results_df:
            return analysis

        # Temperature impact
        if 'temperature' in results_df:
            temp_groups = results_df.groupby('temperature')['test_accuracy']
            analysis['temperature_impact'] = {
                'values': temp_groups.mean().to_dict(),
                'std': temp_groups.std().to_dict(),
                'best': float(results_df.loc[results_df['test_accuracy'].idxmax(), 'temperature'])
            }

        # Alpha impact
        if 'alpha' in results_df:
            alpha_groups = results_df.groupby('alpha')['test_accuracy']
            analysis['alpha_impact'] = {
                'values': alpha_groups.mean().to_dict(),
                'std': alpha_groups.std().to_dict(),
                'best': float(results_df.loc[results_df['test_accuracy'].idxmax(), 'alpha'])
            }

        # Interaction matrix
        if 'temperature' in results_df and 'alpha' in results_df:
            pivot = results_df.pivot_table(
                values='test_accuracy',
                index='temperature',
                columns='alpha',
                aggfunc='mean'
            )
            analysis['interaction_matrix'] = {
                'data': pivot.values.tolist(),
                'temperatures': pivot.index.tolist(),
                'alphas': pivot.columns.tolist()
            }

        # Optimal ranges (top 20% performers)
        threshold = results_df['test_accuracy'].quantile(0.8)
        top_performers = results_df[results_df['test_accuracy'] >= threshold]

        if not top_performers.empty:
            analysis['optimal_ranges'] = {
                'temperature': {
                    'min': float(top_performers['temperature'].min()),
                    'max': float(top_performers['temperature'].max()),
                    'mean': float(top_performers['temperature'].mean())
                },
                'alpha': {
                    'min': float(top_performers['alpha'].min()),
                    'max': float(top_performers['alpha'].max()),
                    'mean': float(top_performers['alpha'].mean())
                }
            }

        return analysis

    def _create_performance_comparison(self, results_df: pd.DataFrame, original_metrics: Dict) -> Dict[str, Any]:
        """Create performance comparison data."""
        comparison = {
            'metrics_comparison': [],
            'model_rankings': {},
            'improvement_stats': {}
        }

        if results_df.empty:
            return comparison

        # Get original test metrics
        original_test = original_metrics.get('test', {})

        # Key metrics for comparison
        key_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']

        for metric in key_metrics:
            test_metric = f'test_{metric}'
            if test_metric in results_df:
                original_value = original_test.get(metric, 0)

                metric_data = {
                    'metric': metric,
                    'original': original_value,
                    'best_distilled': float(results_df[test_metric].max()),
                    'worst_distilled': float(results_df[test_metric].min()),
                    'mean_distilled': float(results_df[test_metric].mean()),
                    'improvement': float(results_df[test_metric].max() - original_value) if original_value else 0
                }
                comparison['metrics_comparison'].append(metric_data)

        # Model rankings (support both f1_score and f1 naming)
        ranking_metrics = ['test_accuracy', 'test_f1_score', 'test_f1', 'training_time', 'model_complexity']
        for metric in ranking_metrics:
            if metric in results_df:
                # For time and complexity, lower is better
                ascending = metric in ['training_time', 'model_complexity']
                ranked = results_df.nsmallest(5, metric) if ascending else results_df.nlargest(5, metric)

                comparison['model_rankings'][metric] = [
                    {
                        'rank': i + 1,
                        'model': row.get('model_type', 'Unknown'),
                        'config': f"T{row.get('temperature', 0)}_A{row.get('alpha', 0)}",
                        'value': float(row[metric])
                    }
                    for i, (_, row) in enumerate(ranked.iterrows())
                ]

        # Improvement statistics
        if 'test_accuracy' in results_df and original_test.get('accuracy'):
            improved_models = results_df[results_df['test_accuracy'] > original_test['accuracy']]
            comparison['improvement_stats'] = {
                'models_improved': len(improved_models),
                'models_total': len(results_df),
                'improvement_rate': len(improved_models) / len(results_df) if len(results_df) > 0 else 0,
                'max_improvement': float((results_df['test_accuracy'].max() - original_test['accuracy']) * 100),
                'avg_improvement': float((improved_models['test_accuracy'].mean() - original_test['accuracy']) * 100) if len(improved_models) > 0 else 0
            }

        return comparison

    def _analyze_tradeoffs(self, results_df: pd.DataFrame, original_metrics: Dict) -> Dict[str, Any]:
        """Analyze tradeoffs between performance and complexity."""
        tradeoffs = {
            'accuracy_vs_complexity': [],
            'accuracy_vs_time': [],
            'pareto_frontier': [],
            'efficiency_scores': []
        }

        if results_df.empty or 'test_accuracy' not in results_df:
            return tradeoffs

        # Accuracy vs Complexity
        if 'model_complexity' in results_df:
            for _, row in results_df.iterrows():
                tradeoffs['accuracy_vs_complexity'].append({
                    'model': row.get('model_type', 'Unknown'),
                    'accuracy': float(row['test_accuracy']),
                    'complexity': float(row['model_complexity']),
                    'config': f"T{row.get('temperature', 0)}_A{row.get('alpha', 0)}"
                })

        # Accuracy vs Training Time
        if 'training_time' in results_df:
            for _, row in results_df.iterrows():
                tradeoffs['accuracy_vs_time'].append({
                    'model': row.get('model_type', 'Unknown'),
                    'accuracy': float(row['test_accuracy']),
                    'time': float(row['training_time']),
                    'config': f"T{row.get('temperature', 0)}_A{row.get('alpha', 0)}"
                })

        # Calculate Pareto frontier
        if 'model_complexity' in results_df:
            pareto_points = self._calculate_pareto_frontier(
                results_df['model_complexity'].values,
                results_df['test_accuracy'].values
            )

            for idx in pareto_points:
                row = results_df.iloc[idx]
                tradeoffs['pareto_frontier'].append({
                    'model': row.get('model_type', 'Unknown'),
                    'accuracy': float(row['test_accuracy']),
                    'complexity': float(row['model_complexity']),
                    'config': f"T{row.get('temperature', 0)}_A{row.get('alpha', 0)}"
                })

        # Efficiency scores
        if 'model_complexity' in results_df and 'training_time' in results_df:
            # Normalize complexity and time
            max_complexity = results_df['model_complexity'].max()
            max_time = results_df['training_time'].max()

            for _, row in results_df.iterrows():
                # Efficiency = accuracy / (normalized_complexity * normalized_time)
                norm_complexity = row['model_complexity'] / max_complexity if max_complexity > 0 else 1
                norm_time = row['training_time'] / max_time if max_time > 0 else 1

                efficiency = row['test_accuracy'] / (norm_complexity * norm_time + 0.001)  # Add small value to avoid division by zero

                tradeoffs['efficiency_scores'].append({
                    'model': row.get('model_type', 'Unknown'),
                    'efficiency': float(efficiency),
                    'accuracy': float(row['test_accuracy']),
                    'complexity': float(row['model_complexity']),
                    'time': float(row['training_time']),
                    'config': f"T{row.get('temperature', 0)}_A{row.get('alpha', 0)}"
                })

        # Sort efficiency scores
        tradeoffs['efficiency_scores'] = sorted(
            tradeoffs['efficiency_scores'],
            key=lambda x: x['efficiency'],
            reverse=True
        )

        return tradeoffs

    def _calculate_pareto_frontier(self, x_values: np.ndarray, y_values: np.ndarray) -> List[int]:
        """
        Calculate Pareto frontier indices.
        For accuracy (y) we want to maximize, for complexity (x) we want to minimize.
        """
        pareto_indices = []

        for i in range(len(x_values)):
            is_pareto = True
            for j in range(len(x_values)):
                if i != j:
                    # Check if point j dominates point i
                    # j dominates i if j has lower complexity AND higher accuracy
                    if x_values[j] < x_values[i] and y_values[j] >= y_values[i]:
                        is_pareto = False
                        break
                    elif x_values[j] <= x_values[i] and y_values[j] > y_values[i]:
                        is_pareto = False
                        break

            if is_pareto:
                pareto_indices.append(i)

        return pareto_indices

    def _generate_recommendations(self, results_df: pd.DataFrame, original_metrics: Dict) -> List[Dict[str, Any]]:
        """Generate model recommendations based on different criteria."""
        recommendations = []

        if results_df.empty:
            return recommendations

        # Best overall performance
        if 'test_accuracy' in results_df:
            best_accuracy = results_df.nlargest(1, 'test_accuracy').iloc[0]
            recommendations.append({
                'scenario': 'Best Overall Performance',
                'model': best_accuracy.get('model_type', 'Unknown'),
                'config': f"Temperature={best_accuracy.get('temperature', 0)}, Alpha={best_accuracy.get('alpha', 0)}",
                'reason': f"Highest test accuracy: {best_accuracy['test_accuracy']:.4f}",
                'metrics': {
                    'accuracy': float(best_accuracy['test_accuracy']),
                    'f1_score': float(best_accuracy.get('test_f1_score', 0)),
                    'training_time': float(best_accuracy.get('training_time', 0)),
                    'complexity': float(best_accuracy.get('model_complexity', 0))
                }
            })

        # Most efficient (best accuracy per unit complexity)
        if 'model_complexity' in results_df and 'test_accuracy' in results_df:
            results_df['efficiency'] = results_df['test_accuracy'] / (results_df['model_complexity'] + 0.001)
            most_efficient = results_df.nlargest(1, 'efficiency').iloc[0]
            recommendations.append({
                'scenario': 'Best Efficiency (Accuracy/Complexity)',
                'model': most_efficient.get('model_type', 'Unknown'),
                'config': f"Temperature={most_efficient.get('temperature', 0)}, Alpha={most_efficient.get('alpha', 0)}",
                'reason': f"Best balance of accuracy ({most_efficient['test_accuracy']:.4f}) and complexity ({most_efficient['model_complexity']:.2f})",
                'metrics': {
                    'accuracy': float(most_efficient['test_accuracy']),
                    'f1_score': float(most_efficient.get('test_f1_score', 0)),
                    'training_time': float(most_efficient.get('training_time', 0)),
                    'complexity': float(most_efficient.get('model_complexity', 0)),
                    'efficiency': float(most_efficient['efficiency'])
                }
            })

        # Fastest training with good performance
        if 'training_time' in results_df and 'test_accuracy' in results_df:
            # Filter models with accuracy above median
            median_accuracy = results_df['test_accuracy'].median()
            good_models = results_df[results_df['test_accuracy'] >= median_accuracy]

            if not good_models.empty:
                fastest = good_models.nsmallest(1, 'training_time').iloc[0]
                recommendations.append({
                    'scenario': 'Fast Training with Good Performance',
                    'model': fastest.get('model_type', 'Unknown'),
                    'config': f"Temperature={fastest.get('temperature', 0)}, Alpha={fastest.get('alpha', 0)}",
                    'reason': f"Training time: {fastest['training_time']:.2f}s with accuracy: {fastest['test_accuracy']:.4f}",
                    'metrics': {
                        'accuracy': float(fastest['test_accuracy']),
                        'f1_score': float(fastest.get('test_f1_score', 0)),
                        'training_time': float(fastest['training_time']),
                        'complexity': float(fastest.get('model_complexity', 0))
                    }
                })

        # Best for production (balanced score)
        if all(col in results_df for col in ['test_accuracy', 'test_f1_score', 'model_complexity']):
            # Normalize metrics
            max_complexity = results_df['model_complexity'].max()

            results_df['production_score'] = (
                0.4 * results_df['test_accuracy'] +
                0.3 * results_df['test_f1_score'] +
                0.3 * (1 - results_df['model_complexity'] / max_complexity if max_complexity > 0 else 0)
            )

            best_production = results_df.nlargest(1, 'production_score').iloc[0]
            recommendations.append({
                'scenario': 'Best for Production',
                'model': best_production.get('model_type', 'Unknown'),
                'config': f"Temperature={best_production.get('temperature', 0)}, Alpha={best_production.get('alpha', 0)}",
                'reason': 'Optimal balance of performance, robustness, and complexity',
                'metrics': {
                    'accuracy': float(best_production['test_accuracy']),
                    'f1_score': float(best_production.get('test_f1_score', 0)),
                    'training_time': float(best_production.get('training_time', 0)),
                    'complexity': float(best_production.get('model_complexity', 0)),
                    'production_score': float(best_production['production_score'])
                }
            })

        # Model that beats original
        if original_metrics.get('test', {}).get('accuracy'):
            original_acc = original_metrics['test']['accuracy']
            better_models = results_df[results_df['test_accuracy'] > original_acc]

            if not better_models.empty:
                # Get the one with lowest complexity among those that beat original
                if 'model_complexity' in better_models:
                    simplest_better = better_models.nsmallest(1, 'model_complexity').iloc[0]
                    recommendations.append({
                        'scenario': 'Simplest Model Better Than Original',
                        'model': simplest_better.get('model_type', 'Unknown'),
                        'config': f"Temperature={simplest_better.get('temperature', 0)}, Alpha={simplest_better.get('alpha', 0)}",
                        'reason': f"Beats original ({original_acc:.4f}) with lower complexity",
                        'metrics': {
                            'accuracy': float(simplest_better['test_accuracy']),
                            'f1_score': float(simplest_better.get('test_f1_score', 0)),
                            'training_time': float(simplest_better.get('training_time', 0)),
                            'complexity': float(simplest_better.get('model_complexity', 0)),
                            'improvement': float(simplest_better['test_accuracy'] - original_acc)
                        }
                    })

        return recommendations

    def _clean_metrics(self, metrics: Dict) -> Dict[str, float]:
        """Clean and format metrics dictionary."""
        cleaned = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and pd.notna(value):
                cleaned[key] = float(value)
        return cleaned

    def _create_metadata(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Create metadata for the report."""
        return {
            'report_type': 'distillation',
            'total_models': len(results_df),
            'has_results': not results_df.empty,
            'columns': list(results_df.columns) if not results_df.empty else [],
            'metrics_available': self._identify_available_metrics(results_df)
        }

    def _identify_available_metrics(self, results_df: pd.DataFrame) -> List[str]:
        """Identify which metrics are available in the results."""
        metrics = []
        possible_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'auc_roc', 'auc_pr', 'ks_statistic'
        ]

        for metric in possible_metrics:
            if f'test_{metric}' in results_df or f'train_{metric}' in results_df:
                metrics.append(metric)

        return metrics