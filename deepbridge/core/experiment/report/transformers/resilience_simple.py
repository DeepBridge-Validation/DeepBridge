"""
Simple data transformer for resilience reports - Following distillation pattern.
Transforms raw resilience results into a format suitable for report generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger("deepbridge.reports")


class ResilienceDataTransformerSimple:
    """
    Transforms resilience experiment results for report generation.
    Simple, clean approach following the distillation pattern.
    """

    def transform(self, results: Dict[str, Any], model_name: str = "Model") -> Dict[str, Any]:
        """
        Transform raw resilience results into report-ready format.

        Args:
            results: Dictionary containing:
                - 'test_results': Test results with primary_model data
                - 'initial_model_evaluation': Initial evaluation with feature_importance
            model_name: Name of the model

        Returns:
            Dictionary with transformed data ready for report rendering
        """
        logger.info("=" * 80)
        logger.info("TRANSFORMING RESILIENCE DATA FOR REPORT (SIMPLE)")
        logger.info("=" * 80)

        # Log top-level keys
        logger.info(f"Top-level keys in results: {list(results.keys())}")

        # Extract main components
        # Handle both formats: with and without test_results wrapper
        if 'test_results' in results:
            # Format from JSON: results['test_results']['primary_model']
            test_results = results.get('test_results', {})
            primary_model = test_results.get('primary_model', {})
            logger.info("Using nested format: results['test_results']['primary_model']")
            logger.info(f"test_results keys: {list(test_results.keys())}")
        else:
            # Format from save_html: results['primary_model'] directly
            primary_model = results.get('primary_model', {})
            logger.info("Using flat format: results['primary_model']")

        logger.info(f"primary_model keys: {list(primary_model.keys())}")

        initial_eval = results.get('initial_model_evaluation', {})
        logger.info(f"initial_eval keys: {list(initial_eval.keys())}")

        # Transform features
        features_data = self._transform_features(initial_eval)

        # Transform the data
        transformed = {
            'model_name': model_name,
            'model_type': primary_model.get('model_type', 'Unknown'),

            # Summary metrics (across all test types)
            'summary': self._create_summary(primary_model),

            # Test-specific scores
            'test_scores': primary_model.get('test_scores', {}),

            # Distribution shift scenarios (original test type)
            'distribution_shift': self._transform_scenarios(primary_model),

            # New test types
            'worst_sample': self._transform_worst_sample(primary_model),
            'worst_cluster': self._transform_worst_cluster(primary_model),
            'outer_sample': self._transform_outer_sample(primary_model),
            'hard_sample': self._transform_hard_sample(primary_model),

            # Feature importance (original structure for backward compatibility)
            'features': features_data,

            # Feature importance (array format for interactive template table)
            'features_table': self._prepare_features_table(features_data),

            # Charts data (ready for Plotly)
            'charts': self._prepare_charts(primary_model, initial_eval),

            # Metadata
            'metadata': {
                'total_scenarios': self._count_all_scenarios(primary_model),
                'total_features': self._count_features(initial_eval),
                'distance_metrics': primary_model.get('distance_metrics', []),
                'alphas': primary_model.get('alphas', []),
                'test_types': self._get_available_test_types(primary_model)
            }
        }

        logger.info(f"Transformation complete. {transformed['metadata']['total_scenarios']} scenarios, "
                   f"{transformed['metadata']['total_features']} features")
        return transformed

    def _create_summary(self, primary_model: Dict) -> Dict[str, Any]:
        """Create summary statistics across all test types."""
        # Collect all performance gaps from all test types
        all_gaps = []

        # Distribution shift
        dist_shift = primary_model.get('distribution_shift', {}).get('all_results', [])
        for s in dist_shift:
            gap = s.get('performance_gap')
            if gap is not None and not (isinstance(gap, float) and np.isnan(gap)):
                all_gaps.append(gap)

        # Worst sample
        worst_sample = primary_model.get('worst_sample', {}).get('all_results', [])
        for s in worst_sample:
            gap = s.get('performance_gap')
            if gap is not None and not (isinstance(gap, float) and np.isnan(gap)):
                all_gaps.append(gap)

        # Worst cluster
        worst_cluster = primary_model.get('worst_cluster', {}).get('all_results', [])
        for s in worst_cluster:
            gap = s.get('performance_gap')
            if gap is not None and not (isinstance(gap, float) and np.isnan(gap)):
                all_gaps.append(gap)

        # Outer sample
        outer_sample = primary_model.get('outer_sample', {}).get('all_results', [])
        for s in outer_sample:
            gap = s.get('performance_gap')
            if gap is not None and not (isinstance(gap, float) and np.isnan(gap)):
                all_gaps.append(gap)

        # Hard sample
        hard_sample = primary_model.get('hard_sample', {}).get('all_results', [])
        for s in hard_sample:
            gap = s.get('performance_gap')
            if gap is not None and not (isinstance(gap, float) and np.isnan(gap)):
                all_gaps.append(gap)

        # Calculate statistics
        if all_gaps:
            avg_gap = np.mean(all_gaps)
            max_gap = np.max(all_gaps)
            min_gap = np.min(all_gaps)
        else:
            avg_gap = max_gap = min_gap = 0.0

        # Count scenarios across all test types
        total_scenarios = (len(dist_shift) + len(worst_sample) +
                          len(worst_cluster) + len(outer_sample) + len(hard_sample))

        # Get base performance - handle case where metrics might be a list or dict
        base_performance = 0.0
        metrics_data = primary_model.get('metrics', {})
        if isinstance(metrics_data, dict):
            base_performance = float(metrics_data.get('accuracy', 0))
        # If metrics is a list (metric names), we don't have base performance values

        return {
            'resilience_score': float(primary_model.get('resilience_score', 1.0)),
            'total_scenarios': total_scenarios,
            'valid_scenarios': len(all_gaps),
            'avg_performance_gap': float(avg_gap),
            'max_performance_gap': float(max_gap),
            'min_performance_gap': float(min_gap),
            'base_performance': base_performance,

            # Per-test-type counts
            'test_counts': {
                'distribution_shift': len(dist_shift),
                'worst_sample': len(worst_sample),
                'worst_cluster': len(worst_cluster),
                'outer_sample': len(outer_sample),
                'hard_sample': len(hard_sample)
            }
        }

    def _transform_scenarios(self, primary_model: Dict) -> List[Dict[str, Any]]:
        """Transform shift scenarios for display."""
        scenarios = primary_model.get('distribution_shift', {}).get('all_results', [])

        logger.info("=" * 70)
        logger.info("TRANSFORMING SCENARIOS FOR OVERVIEW CHART")
        logger.info(f"Raw scenarios from distribution_shift.all_results: {len(scenarios)}")

        transformed_scenarios = []
        for i, scenario in enumerate(scenarios):
            logger.info(f"Raw scenario {i+1}: {scenario}")
            # Handle NaN values
            perf_gap = scenario.get('performance_gap')
            worst_metric = scenario.get('worst_metric')
            remaining_metric = scenario.get('remaining_metric')

            # Convert NaN to None for JSON serialization
            if isinstance(perf_gap, float) and np.isnan(perf_gap):
                perf_gap = None
            if isinstance(worst_metric, float) and np.isnan(worst_metric):
                worst_metric = None
            if isinstance(remaining_metric, float) and np.isnan(remaining_metric):
                remaining_metric = None

            transformed = {
                'id': i + 1,
                'name': scenario.get('name', f"Scenario {i + 1}"),
                'alpha': float(scenario.get('alpha', 0)),
                'distance_metric': scenario.get('distance_metric', 'unknown'),
                'metric': scenario.get('metric', 'unknown'),
                'performance_gap': float(perf_gap) if perf_gap is not None else None,
                'baseline_performance': float(worst_metric) if worst_metric is not None else None,
                'target_performance': float(remaining_metric) if remaining_metric is not None else None,
                'is_valid': perf_gap is not None
            }
            transformed_scenarios.append(transformed)
            logger.info(f"Transformed scenario {i+1}: is_valid={transformed['is_valid']}, "
                       f"gap={transformed['performance_gap']}, "
                       f"metric={transformed['distance_metric']}, "
                       f"alpha={transformed['alpha']}")

        logger.info(f"Total valid scenarios: {sum(1 for s in transformed_scenarios if s['is_valid'])}")
        logger.info("=" * 70)

        return transformed_scenarios

    def _transform_worst_sample(self, primary_model: Dict) -> Dict[str, Any]:
        """Transform worst-sample test results."""
        worst_sample_data = primary_model.get('worst_sample', {})
        all_results = worst_sample_data.get('all_results', [])

        transformed_results = []
        for i, result in enumerate(all_results):
            # Handle NaN values
            perf_gap = result.get('performance_gap')
            worst_metric = result.get('worst_metric')
            remaining_metric = result.get('remaining_metric')

            if isinstance(perf_gap, float) and np.isnan(perf_gap):
                perf_gap = None
            if isinstance(worst_metric, float) and np.isnan(worst_metric):
                worst_metric = None
            if isinstance(remaining_metric, float) and np.isnan(remaining_metric):
                remaining_metric = None

            transformed_results.append({
                'id': i + 1,
                'alpha': float(result.get('alpha', 0)),
                'ranking_method': result.get('ranking_method', 'unknown'),
                'metric': result.get('metric', 'unknown'),
                'performance_gap': float(perf_gap) if perf_gap is not None else None,
                'worst_metric': float(worst_metric) if worst_metric is not None else None,
                'remaining_metric': float(remaining_metric) if remaining_metric is not None else None,
                'n_worst_samples': int(result.get('n_worst_samples', 0)),
                'n_remaining_samples': int(result.get('n_remaining_samples', 0)),
                'is_valid': perf_gap is not None
            })

        # Get by_alpha aggregations
        by_alpha = {}
        for alpha, data in worst_sample_data.get('by_alpha', {}).items():
            by_alpha[str(alpha)] = {
                'avg_performance_gap': float(data.get('avg_performance_gap', 0)),
                'results_count': int(data.get('results_count', 0))
            }

        return {
            'all_results': transformed_results,
            'by_alpha': by_alpha,
            'total_tests': len(transformed_results)
        }

    def _transform_worst_cluster(self, primary_model: Dict) -> Dict[str, Any]:
        """Transform worst-cluster test results."""
        worst_cluster_data = primary_model.get('worst_cluster', {})
        all_results = worst_cluster_data.get('all_results', [])

        transformed_results = []
        for i, result in enumerate(all_results):
            # Handle NaN values
            perf_gap = result.get('performance_gap')
            worst_metric = result.get('worst_cluster_metric')
            remaining_metric = result.get('remaining_metric')

            if isinstance(perf_gap, float) and np.isnan(perf_gap):
                perf_gap = None
            if isinstance(worst_metric, float) and np.isnan(worst_metric):
                worst_metric = None
            if isinstance(remaining_metric, float) and np.isnan(remaining_metric):
                remaining_metric = None

            # Extract top contributing features
            feature_contributions = result.get('feature_contributions', {})
            top_features = sorted(
                [{'name': k, 'contribution': float(v)} for k, v in feature_contributions.items()],
                key=lambda x: abs(x['contribution']),
                reverse=True
            )[:10]

            # Handle worst_cluster_id which might be None
            worst_cluster_id_raw = result.get('worst_cluster_id', -1)
            worst_cluster_id = int(worst_cluster_id_raw) if worst_cluster_id_raw is not None else -1

            worst_cluster_size_raw = result.get('worst_cluster_size', 0)
            worst_cluster_size = int(worst_cluster_size_raw) if worst_cluster_size_raw is not None else 0

            remaining_size_raw = result.get('remaining_size', 0)
            remaining_size = int(remaining_size_raw) if remaining_size_raw is not None else 0

            transformed_results.append({
                'id': i + 1,
                'n_clusters': int(result.get('n_clusters', 0)),
                'worst_cluster_id': worst_cluster_id,
                'metric': result.get('metric', 'unknown'),
                'performance_gap': float(perf_gap) if perf_gap is not None else None,
                'worst_cluster_metric': float(worst_metric) if worst_metric is not None else None,
                'remaining_metric': float(remaining_metric) if remaining_metric is not None else None,
                'worst_cluster_size': worst_cluster_size,
                'remaining_size': remaining_size,
                'top_features': top_features,
                'is_valid': perf_gap is not None
            })

        # Get by_n_clusters aggregations
        by_n_clusters = {}
        for n_clusters, data in worst_cluster_data.get('by_n_clusters', {}).items():
            by_n_clusters[str(n_clusters)] = {
                'avg_performance_gap': float(data.get('avg_performance_gap', 0)),
                'results_count': int(data.get('results_count', 0))
            }

        return {
            'all_results': transformed_results,
            'by_n_clusters': by_n_clusters,
            'total_tests': len(transformed_results)
        }

    def _transform_outer_sample(self, primary_model: Dict) -> Dict[str, Any]:
        """Transform outer-sample test results."""
        outer_sample_data = primary_model.get('outer_sample', {})
        all_results = outer_sample_data.get('all_results', [])

        transformed_results = []
        for i, result in enumerate(all_results):
            # Handle NaN values
            perf_gap = result.get('performance_gap')
            outer_metric = result.get('outer_metric')
            inner_metric = result.get('inner_metric')

            if isinstance(perf_gap, float) and np.isnan(perf_gap):
                perf_gap = None
            if isinstance(outer_metric, float) and np.isnan(outer_metric):
                outer_metric = None
            if isinstance(inner_metric, float) and np.isnan(inner_metric):
                inner_metric = None

            transformed_results.append({
                'id': i + 1,
                'alpha': float(result.get('alpha', 0)),
                'outlier_method': result.get('outlier_method', 'unknown'),
                'metric': result.get('metric', 'unknown'),
                'performance_gap': float(perf_gap) if perf_gap is not None else None,
                'outer_metric': float(outer_metric) if outer_metric is not None else None,
                'inner_metric': float(inner_metric) if inner_metric is not None else None,
                'n_outer_samples': int(result.get('n_outer_samples', 0)),
                'n_inner_samples': int(result.get('n_inner_samples', 0)),
                'is_valid': perf_gap is not None
            })

        # Get by_alpha aggregations
        by_alpha = {}
        for alpha, data in outer_sample_data.get('by_alpha', {}).items():
            by_alpha[str(alpha)] = {
                'avg_performance_gap': float(data.get('avg_performance_gap', 0)),
                'results_count': int(data.get('results_count', 0))
            }

        return {
            'all_results': transformed_results,
            'by_alpha': by_alpha,
            'total_tests': len(transformed_results)
        }

    def _transform_hard_sample(self, primary_model: Dict) -> Dict[str, Any]:
        """Transform hard-sample test results."""
        hard_sample_data = primary_model.get('hard_sample', {})
        all_results = hard_sample_data.get('all_results', [])

        transformed_results = []
        for i, result in enumerate(all_results):
            # Handle NaN values and check if test was actually run
            perf_gap = result.get('performance_gap')
            hard_metric = result.get('hard_metric')
            easy_metric = result.get('easy_metric')

            # Check if test was skipped (no alternative models available)
            if hard_metric is None and easy_metric is None:
                # Test was skipped
                transformed_results.append({
                    'id': i + 1,
                    'skipped': True,
                    'reason': 'No alternative models available',
                    'is_valid': False
                })
                continue

            if isinstance(perf_gap, float) and np.isnan(perf_gap):
                perf_gap = None
            if isinstance(hard_metric, float) and np.isnan(hard_metric):
                hard_metric = None
            if isinstance(easy_metric, float) and np.isnan(easy_metric):
                easy_metric = None

            # Extract model disagreements
            model_disagreements = result.get('model_disagreements', {})
            disagreement_list = [
                {'model_pair': k, 'disagreement': float(v)}
                for k, v in model_disagreements.items()
            ]

            transformed_results.append({
                'id': i + 1,
                'skipped': False,
                'disagreement_threshold': float(result.get('disagreement_threshold', 0)),
                'metric': result.get('metric', 'unknown'),
                'performance_gap': float(perf_gap) if perf_gap is not None else None,
                'hard_metric': float(hard_metric) if hard_metric is not None else None,
                'easy_metric': float(easy_metric) if easy_metric is not None else None,
                'n_hard_samples': int(result.get('n_hard_samples', 0)),
                'n_easy_samples': int(result.get('n_easy_samples', 0)),
                'model_disagreements': disagreement_list,
                'is_valid': perf_gap is not None
            })

        # Get by_threshold aggregations
        by_threshold = {}
        for threshold, data in hard_sample_data.get('by_threshold', {}).items():
            by_threshold[str(threshold)] = {
                'avg_performance_gap': float(data.get('avg_performance_gap', 0)),
                'results_count': int(data.get('results_count', 0))
            }

        return {
            'all_results': transformed_results,
            'by_threshold': by_threshold,
            'total_tests': len(transformed_results)
        }

    def _transform_features(self, initial_eval: Dict) -> Dict[str, Any]:
        """Transform feature importance data."""
        # Get feature importance from initial evaluation
        models = initial_eval.get('models', {})
        primary_model = models.get('primary_model', {})
        feature_importance = primary_model.get('feature_importance', {})

        if not feature_importance:
            logger.warning("No feature importance data found")
            return {
                'total': 0,
                'importance': {},
                'top_10': [],
                'feature_list': []
            }

        # Convert to list and sort by absolute importance
        feature_list = [
            {
                'name': name,
                'importance': float(value),
                'importance_abs': abs(float(value))
            }
            for name, value in feature_importance.items()
        ]

        # Sort by absolute importance
        feature_list_sorted = sorted(feature_list, key=lambda x: x['importance_abs'], reverse=True)

        return {
            'total': len(feature_importance),
            'importance': feature_importance,  # Original dict
            'top_10': feature_list_sorted[:10],  # Top 10 features
            'top_20': feature_list_sorted[:20],  # Top 20 features
            'feature_list': feature_list_sorted  # All features sorted
        }

    def _prepare_features_table(self, features_data: Dict) -> List[Dict[str, Any]]:
        """
        Prepare feature data in array format for interactive template table.

        Args:
            features_data: Features dictionary from _transform_features

        Returns:
            List of feature dictionaries with name, missing_impact, and importance
        """
        logger.info("=" * 80)
        logger.info("PREPARING FEATURES TABLE DATA")
        logger.info("=" * 80)

        feature_list = features_data.get('feature_list', [])

        if not feature_list:
            logger.warning("No features available for table")
            return []

        # Convert to table format
        # Note: missing_impact is set to 0.0 for now since resilience tests
        # don't currently measure impact of missing data per feature
        table_data = []
        for feature in feature_list:
            table_data.append({
                'name': feature['name'],
                'missing_impact': 0.0,  # Placeholder - not measured in current resilience tests
                'importance': feature['importance_abs']
            })

        logger.info(f"Prepared {len(table_data)} features for table")
        logger.info(f"Sample: {table_data[0] if table_data else 'N/A'}")
        logger.info("=" * 80)

        return table_data

    def _prepare_charts(self, primary_model: Dict, initial_eval: Dict) -> Dict[str, Any]:
        """Prepare data for Plotly charts."""
        logger.info("=" * 80)
        logger.info("PREPARING CHARTS FOR RESILIENCE REPORT")
        logger.info("=" * 80)

        logger.info(f"Checking for missing data information in primary_model...")
        logger.info(f"Keys in primary_model: {list(primary_model.keys())}")

        # Check for missing data fields
        if 'missing_data' in primary_model:
            logger.info("✓ Found 'missing_data' in primary_model!")
            logger.info(f"  missing_data keys: {list(primary_model['missing_data'].keys())}")
        else:
            logger.warning("✗ 'missing_data' NOT found in primary_model")

        if 'by_level' in primary_model:
            logger.info("✓ Found 'by_level' in primary_model!")
        else:
            logger.warning("✗ 'by_level' NOT found in primary_model")

        scenarios = self._transform_scenarios(primary_model)
        features = self._transform_features(initial_eval)

        # Get transformed data for new test types
        worst_sample = self._transform_worst_sample(primary_model)
        worst_cluster = self._transform_worst_cluster(primary_model)
        outer_sample = self._transform_outer_sample(primary_model)
        hard_sample = self._transform_hard_sample(primary_model)

        # Generate charts
        scenarios_by_alpha_chart = self._chart_scenarios_by_alpha(scenarios)

        # Generate feature importance chart
        feature_importance_chart = self._chart_feature_importance(features)

        charts = {
            # Original distribution shift charts
            'overview': self._chart_overview(primary_model, scenarios),
            'scenarios_by_alpha': scenarios_by_alpha_chart,
            'score_distribution': scenarios_by_alpha_chart,  # Alias for template compatibility
            'scenarios_by_metric': self._chart_scenarios_by_metric(scenarios),
            'feature_importance': feature_importance_chart,
            'features': feature_importance_chart,  # Alias for interactive template
            'boxplot': self._chart_boxplot(scenarios),

            # New test type charts
            'worst_sample': self._chart_worst_sample(worst_sample),
            'worst_cluster': self._chart_worst_cluster(worst_cluster),
            'outer_sample': self._chart_outer_sample(outer_sample),
            'hard_sample': self._chart_hard_sample(hard_sample),

            # Comparative analysis
            'test_type_comparison': self._chart_test_type_comparison(primary_model),

            # Missing data analysis charts (for interactive template compatibility)
            'by_level': self._chart_by_level(primary_model),
            'patterns': self._chart_missing_patterns(primary_model),
            'strategies': self._chart_imputation_strategies(primary_model)
        }

        logger.info(f"Generated charts: {list(charts.keys())}")

        # Log score_distribution chart details for debugging
        if 'score_distribution' in charts:
            score_dist = charts['score_distribution']
            logger.info(f"score_distribution chart type: {type(score_dist)}")
            if isinstance(score_dist, dict):
                logger.info(f"score_distribution has 'data': {'data' in score_dist}")
                logger.info(f"score_distribution has 'layout': {'layout' in score_dist}")
                if 'data' in score_dist:
                    logger.info(f"score_distribution data length: {len(score_dist.get('data', []))}")
                    logger.info(f"score_distribution data sample: {score_dist.get('data', [])[:1]}")

        return charts

    def _chart_overview(self, primary_model: Dict, scenarios: List[Dict]) -> Dict[str, Any]:
        """Create overview chart data (performance gap by alpha)."""
        logger.info("=" * 70)
        logger.info("CREATING OVERVIEW CHART (Resilience Overview)")
        logger.info(f"Total scenarios received: {len(scenarios)}")

        # Log all scenarios for debugging
        for i, scenario in enumerate(scenarios):
            logger.info(f"Scenario {i+1}: name={scenario.get('name')}, "
                       f"metric={scenario.get('distance_metric')}, "
                       f"alpha={scenario.get('alpha')}, "
                       f"gap={scenario.get('performance_gap')}, "
                       f"is_valid={scenario.get('is_valid')}")

        # Group scenarios by distance metric
        metrics = {}
        for scenario in scenarios:
            metric = scenario['distance_metric']
            if metric not in metrics:
                metrics[metric] = {'alphas': [], 'gaps': [], 'names': []}

            if scenario['is_valid']:
                metrics[metric]['alphas'].append(scenario['alpha'])
                metrics[metric]['gaps'].append(scenario['performance_gap'])
                metrics[metric]['names'].append(scenario['name'])
                logger.info(f"Added to chart: metric={metric}, alpha={scenario['alpha']}, "
                           f"gap={scenario['performance_gap']}, name={scenario['name']}")
            else:
                logger.warning(f"Skipped invalid scenario: {scenario['name']}")

        # Create Plotly traces
        traces = []
        for metric, data in metrics.items():
            if data['alphas']:
                logger.info(f"Creating trace for metric '{metric}': "
                           f"{len(data['alphas'])} points, "
                           f"alphas={data['alphas']}, gaps={data['gaps']}")
                traces.append({
                    'x': data['alphas'],
                    'y': data['gaps'],
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': metric,
                    'text': data['names'],
                    'hovertemplate': '<b>%{text}</b><br>Alpha: %{x}<br>Gap: %{y:.4f}<extra></extra>'
                })
            else:
                logger.warning(f"Metric '{metric}' has no valid data points")

        logger.info(f"Total traces created: {len(traces)}")
        logger.info("=" * 70)

        layout = {
            'title': 'Performance Gap by Alpha',
            'xaxis': {'title': 'Alpha (Distribution Shift Strength)'},
            'yaxis': {'title': 'Performance Gap'},
            'hovermode': 'closest',
            'showlegend': True
        }

        return {'data': traces, 'layout': layout}

    def _chart_scenarios_by_alpha(self, scenarios: List[Dict]) -> Dict[str, Any]:
        """Create chart showing performance gap and absolute performance by alpha."""
        # Group by alpha - get unique values (no repetition)
        alpha_data = {}
        for scenario in scenarios:
            alpha = scenario['alpha']
            if alpha not in alpha_data and scenario['is_valid']:
                alpha_data[alpha] = {
                    'gap': scenario['performance_gap'],
                    'worst': scenario.get('baseline_performance', 0),
                    'remaining': scenario.get('target_performance', 1)
                }

        # Sort by alpha
        sorted_alphas = sorted(alpha_data.keys())
        gaps = [alpha_data[a]['gap'] for a in sorted_alphas]
        worst_perfs = [alpha_data[a]['worst'] for a in sorted_alphas]
        remaining_perfs = [alpha_data[a]['remaining'] for a in sorted_alphas]

        # Create traces
        traces = [
            {
                'x': sorted_alphas,
                'y': gaps,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Performance Gap',
                'line': {'color': 'rgb(255, 65, 54)', 'width': 3},
                'marker': {'size': 10},
                'yaxis': 'y2',
                'hovertemplate': '<b>Performance Gap</b><br>Alpha: %{x}<br>Gap: %{y:.4f}<extra></extra>'
            },
            {
                'x': sorted_alphas,
                'y': worst_perfs,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Worst Samples',
                'line': {'color': 'rgb(255, 127, 80)', 'width': 2, 'dash': 'dash'},
                'marker': {'size': 8},
                'hovertemplate': '<b>Worst Samples</b><br>Alpha: %{x}<br>Accuracy: %{y:.4f}<extra></extra>'
            },
            {
                'x': sorted_alphas,
                'y': remaining_perfs,
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Remaining Samples',
                'line': {'color': 'rgb(46, 204, 113)', 'width': 2, 'dash': 'dash'},
                'marker': {'size': 8},
                'hovertemplate': '<b>Remaining Samples</b><br>Alpha: %{x}<br>Accuracy: %{y:.4f}<extra></extra>'
            }
        ]

        layout = {
            'title': 'Performance Gap and Absolute Performance by Alpha',
            'xaxis': {'title': 'Alpha (Shift Strength)', 'tickmode': 'array', 'tickvals': sorted_alphas},
            'yaxis': {
                'title': 'Accuracy',
                'side': 'left',
                'range': [min(worst_perfs) - 0.05 if worst_perfs else 0, 1.05]
            },
            'yaxis2': {
                'title': 'Performance Gap',
                'side': 'right',
                'overlaying': 'y',
                'range': [0, max(gaps) * 1.2 if gaps else 0.2]
            },
            'hovermode': 'x unified',
            'showlegend': True,
            'legend': {'x': 0.02, 'y': 0.98}
        }

        return {'data': traces, 'layout': layout}

    def _chart_scenarios_by_metric(self, scenarios: List[Dict]) -> Dict[str, Any]:
        """Create chart grouped by distance metric."""
        # Group by distance metric
        metrics = {}
        for scenario in scenarios:
            metric = scenario['distance_metric']
            if metric not in metrics:
                metrics[metric] = []
            if scenario['is_valid']:
                metrics[metric].append(scenario['performance_gap'])

        # Create box plot
        traces = []
        for metric, gaps in metrics.items():
            if gaps:
                traces.append({
                    'y': gaps,
                    'type': 'box',
                    'name': metric,
                    'boxmean': 'sd'
                })

        layout = {
            'title': 'Performance Distribution by Distance Metric',
            'yaxis': {'title': 'Performance Gap'},
            'showlegend': True
        }

        return {'data': traces, 'layout': layout}

    def _chart_feature_importance(self, features: Dict) -> Dict[str, Any]:
        """Create feature importance chart."""
        logger.info("=" * 80)
        logger.info("CREATING FEATURE IMPORTANCE CHART")
        logger.info("=" * 80)

        top_features = features['top_10']

        logger.info(f"Total features available: {features.get('total', 0)}")
        logger.info(f"Top 10 features count: {len(top_features)}")

        if not top_features:
            logger.warning("No top features available for chart")
            return {'data': [], 'layout': {}}

        # Log top features
        for i, f in enumerate(top_features, 1):
            logger.info(f"Feature {i}: {f['name']} = {f['importance_abs']:.4f}")

        # Reverse order for horizontal bar chart
        top_features_reversed = list(reversed(top_features))

        trace = {
            'x': [f['importance_abs'] for f in top_features_reversed],
            'y': [f['name'] for f in top_features_reversed],
            'type': 'bar',
            'orientation': 'h',
            'marker': {'color': 'rgb(55, 83, 109)'},
            'hovertemplate': '<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        }

        layout = {
            'title': 'Top 10 Most Important Features',
            'xaxis': {'title': 'Importance'},
            'yaxis': {'title': 'Feature'},
            'margin': {'l': 150}
        }

        logger.info(f"Feature importance chart created with {len(top_features)} features")
        logger.info("=" * 80)

        return {'data': [trace], 'layout': layout}

    def _chart_boxplot(self, scenarios: List[Dict]) -> Dict[str, Any]:
        """Create overall boxplot of performance gaps."""
        valid_gaps = [s['performance_gap'] for s in scenarios if s['is_valid']]

        if not valid_gaps:
            return {'data': [], 'layout': {}}

        trace = {
            'y': valid_gaps,
            'type': 'box',
            'name': 'All Scenarios',
            'boxmean': 'sd'
        }

        layout = {
            'title': 'Overall Performance Gap Distribution',
            'yaxis': {'title': 'Performance Gap'}
        }

        return {'data': [trace], 'layout': layout}

    def _chart_worst_sample(self, worst_sample: Dict) -> Dict[str, Any]:
        """Create chart for worst-sample test results."""
        results = worst_sample.get('all_results', [])
        valid_results = [r for r in results if r['is_valid']]

        if not valid_results:
            return {'data': [], 'layout': {}}

        # Group by ranking method
        ranking_methods = {}
        for result in valid_results:
            method = result['ranking_method']
            if method not in ranking_methods:
                ranking_methods[method] = {'alphas': [], 'gaps': []}

            ranking_methods[method]['alphas'].append(result['alpha'])
            ranking_methods[method]['gaps'].append(result['performance_gap'])

        # Create traces
        traces = []
        for method, data in ranking_methods.items():
            traces.append({
                'x': data['alphas'],
                'y': data['gaps'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': method,
                'hovertemplate': f'<b>{method}</b><br>Alpha: %{{x}}<br>Gap: %{{y:.4f}}<extra></extra>'
            })

        layout = {
            'title': 'Worst-Sample Test: Performance Gap by Alpha',
            'xaxis': {'title': 'Alpha (Worst Sample Ratio)'},
            'yaxis': {'title': 'Performance Gap'},
            'hovermode': 'closest',
            'showlegend': True
        }

        return {'data': traces, 'layout': layout}

    def _chart_worst_cluster(self, worst_cluster: Dict) -> Dict[str, Any]:
        """Create chart for worst-cluster test results."""
        results = worst_cluster.get('all_results', [])
        valid_results = [r for r in results if r['is_valid']]

        if not valid_results:
            return {'data': [], 'layout': {}}

        # Create bar chart showing performance gap by n_clusters
        n_clusters_list = sorted(list(set([r['n_clusters'] for r in valid_results])))
        gaps_by_n_clusters = []

        for n in n_clusters_list:
            cluster_results = [r for r in valid_results if r['n_clusters'] == n]
            avg_gap = np.mean([r['performance_gap'] for r in cluster_results])
            gaps_by_n_clusters.append(avg_gap)

        trace = {
            'x': n_clusters_list,
            'y': gaps_by_n_clusters,
            'type': 'bar',
            'marker': {'color': 'rgb(26, 118, 255)'},
            'hovertemplate': '<b>K=%{x}</b><br>Avg Gap: %{y:.4f}<extra></extra>'
        }

        layout = {
            'title': 'Worst-Cluster Test: Performance Gap by Number of Clusters',
            'xaxis': {'title': 'Number of Clusters (K)'},
            'yaxis': {'title': 'Average Performance Gap'},
            'hovermode': 'closest'
        }

        return {'data': [trace], 'layout': layout}

    def _chart_outer_sample(self, outer_sample: Dict) -> Dict[str, Any]:
        """Create chart for outer-sample test results."""
        results = outer_sample.get('all_results', [])
        valid_results = [r for r in results if r['is_valid']]

        if not valid_results:
            return {'data': [], 'layout': {}}

        # Group by outlier method
        outlier_methods = {}
        for result in valid_results:
            method = result['outlier_method']
            if method not in outlier_methods:
                outlier_methods[method] = {'alphas': [], 'gaps': []}

            outlier_methods[method]['alphas'].append(result['alpha'])
            outlier_methods[method]['gaps'].append(result['performance_gap'])

        # Create traces
        traces = []
        for method, data in outlier_methods.items():
            traces.append({
                'x': data['alphas'],
                'y': data['gaps'],
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': method,
                'hovertemplate': f'<b>{method}</b><br>Alpha: %{{x}}<br>Gap: %{{y:.4f}}<extra></extra>'
            })

        layout = {
            'title': 'Outer-Sample Test: Performance Gap by Alpha',
            'xaxis': {'title': 'Alpha (Outlier Ratio)'},
            'yaxis': {'title': 'Performance Gap'},
            'hovermode': 'closest',
            'showlegend': True
        }

        return {'data': traces, 'layout': layout}

    def _chart_hard_sample(self, hard_sample: Dict) -> Dict[str, Any]:
        """Create chart for hard-sample test results."""
        results = hard_sample.get('all_results', [])
        valid_results = [r for r in results if r['is_valid'] and not r.get('skipped', False)]

        if not valid_results:
            # Check if tests were skipped
            skipped_results = [r for r in results if r.get('skipped', False)]
            if skipped_results:
                return {
                    'data': [],
                    'layout': {
                        'title': 'Hard-Sample Test: Not Available',
                        'annotations': [{
                            'text': 'No alternative models available for hard-sample test',
                            'xref': 'paper',
                            'yref': 'paper',
                            'x': 0.5,
                            'y': 0.5,
                            'showarrow': False
                        }]
                    }
                }
            return {'data': [], 'layout': {}}

        # Create bar chart showing performance gap by disagreement threshold
        thresholds = sorted(list(set([r['disagreement_threshold'] for r in valid_results])))
        gaps_by_threshold = []

        for threshold in thresholds:
            threshold_results = [r for r in valid_results if r['disagreement_threshold'] == threshold]
            avg_gap = np.mean([r['performance_gap'] for r in threshold_results])
            gaps_by_threshold.append(avg_gap)

        trace = {
            'x': thresholds,
            'y': gaps_by_threshold,
            'type': 'bar',
            'marker': {'color': 'rgb(255, 65, 54)'},
            'hovertemplate': '<b>Threshold=%{x}</b><br>Avg Gap: %{y:.4f}<extra></extra>'
        }

        layout = {
            'title': 'Hard-Sample Test: Performance Gap by Disagreement Threshold',
            'xaxis': {'title': 'Disagreement Threshold'},
            'yaxis': {'title': 'Average Performance Gap'},
            'hovermode': 'closest'
        }

        return {'data': [trace], 'layout': layout}

    def _chart_test_type_comparison(self, primary_model: Dict) -> Dict[str, Any]:
        """Create comparison chart across all test types."""
        # Get test scores
        test_scores = primary_model.get('test_scores', {})

        if not test_scores:
            return {'data': [], 'layout': {}}

        # Create bar chart
        test_types = []
        scores = []

        for test_type, score in test_scores.items():
            if score is not None and not (isinstance(score, float) and np.isnan(score)):
                test_types.append(test_type.replace('_', ' ').title())
                scores.append(float(score))

        if not test_types:
            return {'data': [], 'layout': {}}

        trace = {
            'x': test_types,
            'y': scores,
            'type': 'bar',
            'marker': {'color': 'rgb(55, 83, 109)'},
            'hovertemplate': '<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>'
        }

        layout = {
            'title': 'Resilience Score by Test Type',
            'xaxis': {'title': 'Test Type'},
            'yaxis': {'title': 'Resilience Score', 'range': [0, 1]},
            'hovermode': 'closest'
        }

        return {'data': [trace], 'layout': layout}

    def _chart_by_level(self, primary_model: Dict) -> Dict[str, Any]:
        """
        Create chart for missing data by level.
        This chart is expected by the interactive resilience template.
        """
        logger.info("=" * 80)
        logger.info("CREATING BY_LEVEL CHART (Missing Data Impact)")
        logger.info("=" * 80)

        # Check if we have missing data information
        missing_data = primary_model.get('missing_data', {})
        by_level = primary_model.get('by_level', {})

        logger.info(f"missing_data available: {bool(missing_data)}")
        logger.info(f"by_level available: {bool(by_level)}")

        # If we have missing data results
        if missing_data or by_level:
            # Use by_level if available, otherwise use missing_data
            level_data = by_level if by_level else missing_data

            logger.info(f"Found level data: {list(level_data.keys())}")

            # Extract levels and impacts
            levels = []
            mean_scores = []
            worst_scores = []

            for level_str, data in sorted(level_data.items(), key=lambda x: float(x[0])):
                level = float(level_str)
                levels.append(level)

                # Extract metrics
                mean_score = data.get('mean_score', data.get('avg_performance', 0))
                worst_score = data.get('worst_score', data.get('min_performance', 0))

                mean_scores.append(float(mean_score))
                worst_scores.append(float(worst_score))

                logger.info(f"Level {level}%: mean={mean_score}, worst={worst_score}")

            # Create traces
            traces = [
                {
                    'x': levels,
                    'y': mean_scores,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Mean Score',
                    'line': {'color': 'rgb(55, 83, 109)', 'width': 3},
                    'marker': {'size': 10},
                    'hovertemplate': '<b>Mean Score</b><br>Missing %: %{x}<br>Score: %{y:.4f}<extra></extra>'
                },
                {
                    'x': levels,
                    'y': worst_scores,
                    'type': 'scatter',
                    'mode': 'lines+markers',
                    'name': 'Worst Score',
                    'line': {'color': 'rgb(255, 65, 54)', 'width': 2, 'dash': 'dash'},
                    'marker': {'size': 8},
                    'hovertemplate': '<b>Worst Score</b><br>Missing %: %{x}<br>Score: %{y:.4f}<extra></extra>'
                }
            ]

            layout = {
                'title': 'Performance Degradation by Missing Data Level',
                'xaxis': {'title': 'Missing Data %'},
                'yaxis': {'title': 'Performance Score'},
                'hovermode': 'x unified',
                'showlegend': True
            }

            logger.info(f"Created by_level chart with {len(levels)} data points")
            logger.info("=" * 80)

            return {'data': traces, 'layout': layout}

        else:
            # No missing data available - return empty chart
            logger.warning("No missing data found in primary_model")
            logger.warning("The 'By Missing %' tab in the report will be empty")
            logger.warning("This is expected if the resilience test focused on distribution shift")
            logger.info("=" * 80)

            return {
                'data': [],
                'layout': {
                    'title': 'Missing Data Analysis Not Available',
                    'annotations': [{
                        'text': 'This resilience test did not include missing data analysis.<br>' +
                                'The test focused on distribution shift and sample-based resilience.',
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5,
                        'showarrow': False,
                        'font': {'size': 14},
                        'xanchor': 'center',
                        'yanchor': 'middle'
                    }],
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False}
                }
            }

    def _chart_missing_patterns(self, primary_model: Dict) -> Dict[str, Any]:
        """
        Create chart for missing data patterns (MCAR, MAR, MNAR).
        This chart is expected by the interactive resilience template.
        """
        logger.info("=" * 80)
        logger.info("CREATING MISSING PATTERNS CHART")
        logger.info("=" * 80)

        # Check if we have missing data patterns information
        missing_patterns = primary_model.get('missing_patterns', {})

        logger.info(f"missing_patterns available: {bool(missing_patterns)}")

        if missing_patterns:
            logger.info(f"Found patterns data: {list(missing_patterns.keys())}")

            # Extract patterns (MCAR, MAR, MNAR)
            pattern_names = []
            performance_scores = []

            for pattern_name, pattern_data in missing_patterns.items():
                pattern_names.append(pattern_name)
                score = pattern_data.get('performance', pattern_data.get('score', 0))
                performance_scores.append(float(score))
                logger.info(f"Pattern {pattern_name}: performance={score}")

            # Create bar chart
            trace = {
                'x': pattern_names,
                'y': performance_scores,
                'type': 'bar',
                'marker': {'color': ['rgb(55, 126, 184)', 'rgb(77, 175, 74)', 'rgb(228, 26, 28)']},
                'hovertemplate': '<b>%{x}</b><br>Performance: %{y:.4f}<extra></extra>'
            }

            layout = {
                'title': 'Performance Under Different Missing Data Patterns',
                'xaxis': {'title': 'Missing Data Pattern'},
                'yaxis': {'title': 'Performance Score'},
                'hovermode': 'closest'
            }

            logger.info(f"Created missing patterns chart with {len(pattern_names)} patterns")
            logger.info("=" * 80)

            return {'data': [trace], 'layout': layout}

        else:
            # No missing patterns data available
            logger.warning("No missing patterns data found in primary_model")
            logger.warning("This is expected if the resilience test did not include missing data pattern analysis")
            logger.info("=" * 80)

            return {
                'data': [],
                'layout': {
                    'title': 'Missing Data Pattern Analysis Not Available',
                    'annotations': [{
                        'text': 'This resilience test did not include missing data pattern analysis (MCAR, MAR, MNAR).<br>' +
                                'The test focused on distribution shift and sample-based resilience.',
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5,
                        'showarrow': False,
                        'font': {'size': 14},
                        'xanchor': 'center',
                        'yanchor': 'middle'
                    }],
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False}
                }
            }

    def _chart_imputation_strategies(self, primary_model: Dict) -> Dict[str, Any]:
        """
        Create chart for imputation strategy comparison.
        This chart is expected by the interactive resilience template.
        """
        logger.info("=" * 80)
        logger.info("CREATING IMPUTATION STRATEGIES CHART")
        logger.info("=" * 80)

        # Check if we have imputation strategies information
        imputation_strategies = primary_model.get('imputation_strategies', {})

        logger.info(f"imputation_strategies available: {bool(imputation_strategies)}")

        if imputation_strategies:
            logger.info(f"Found strategies data: {list(imputation_strategies.keys())}")

            # Extract strategies
            strategy_names = []
            performance_scores = []

            for strategy_name, strategy_data in imputation_strategies.items():
                strategy_names.append(strategy_name)
                score = strategy_data.get('performance', strategy_data.get('score', 0))
                performance_scores.append(float(score))
                logger.info(f"Strategy {strategy_name}: performance={score}")

            # Create bar chart
            trace = {
                'x': strategy_names,
                'y': performance_scores,
                'type': 'bar',
                'marker': {'color': 'rgb(31, 119, 180)'},
                'hovertemplate': '<b>%{x}</b><br>Performance: %{y:.4f}<extra></extra>'
            }

            layout = {
                'title': 'Performance Comparison Across Imputation Strategies',
                'xaxis': {'title': 'Imputation Strategy'},
                'yaxis': {'title': 'Performance Score'},
                'hovermode': 'closest'
            }

            logger.info(f"Created imputation strategies chart with {len(strategy_names)} strategies")
            logger.info("=" * 80)

            return {'data': [trace], 'layout': layout}

        else:
            # No imputation strategies data available
            logger.warning("No imputation strategies data found in primary_model")
            logger.warning("This is expected if the resilience test did not include imputation analysis")
            logger.info("=" * 80)

            return {
                'data': [],
                'layout': {
                    'title': 'Imputation Strategy Analysis Not Available',
                    'annotations': [{
                        'text': 'This resilience test did not include imputation strategy comparison.<br>' +
                                'The test focused on distribution shift and sample-based resilience.',
                        'xref': 'paper',
                        'yref': 'paper',
                        'x': 0.5,
                        'y': 0.5,
                        'showarrow': False,
                        'font': {'size': 14},
                        'xanchor': 'center',
                        'yanchor': 'middle'
                    }],
                    'xaxis': {'visible': False},
                    'yaxis': {'visible': False}
                }
            }

    def _count_all_scenarios(self, primary_model: Dict) -> int:
        """Count total scenarios across all test types."""
        total = 0
        total += len(primary_model.get('distribution_shift', {}).get('all_results', []))
        total += len(primary_model.get('worst_sample', {}).get('all_results', []))
        total += len(primary_model.get('worst_cluster', {}).get('all_results', []))
        total += len(primary_model.get('outer_sample', {}).get('all_results', []))
        total += len(primary_model.get('hard_sample', {}).get('all_results', []))
        return total

    def _count_features(self, initial_eval: Dict) -> int:
        """Count total features."""
        models = initial_eval.get('models', {})
        primary_model = models.get('primary_model', {})
        feature_importance = primary_model.get('feature_importance', {})
        return len(feature_importance)

    def _get_available_test_types(self, primary_model: Dict) -> List[str]:
        """Get list of test types that have results."""
        available = []
        if primary_model.get('distribution_shift', {}).get('all_results', []):
            available.append('distribution_shift')
        if primary_model.get('worst_sample', {}).get('all_results', []):
            available.append('worst_sample')
        if primary_model.get('worst_cluster', {}).get('all_results', []):
            available.append('worst_cluster')
        if primary_model.get('outer_sample', {}).get('all_results', []):
            available.append('outer_sample')
        if primary_model.get('hard_sample', {}).get('all_results', []):
            available.append('hard_sample')
        return available
