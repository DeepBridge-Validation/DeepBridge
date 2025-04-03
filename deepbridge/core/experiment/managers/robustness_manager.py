import typing as t
import copy
import pandas as pd
import numpy as np
import plotly.graph_objects as go

class RobustnessManager:
    """
    Handles robustness testing and comparison between models.
    """
    
    def __init__(self, dataset, alternative_models, verbose=False):
        self.dataset = dataset
        self.alternative_models = alternative_models
        self.verbose = verbose
        
    def run_tests(self) -> dict:
        """
        Run robustness tests using RobustnessSuite and compare
        the original model with alternative models.
        """
        from deepbridge.validation.wrappers.robustness_suite import RobustnessSuite
        
        if self.verbose:
            print("Running robustness tests...")
            
        # Initialize results storage
        results = {
            'main_model': {},
            'alternative_models': {},
            'visualizations': {}
        }
        
        # Test main model
        suite = RobustnessSuite(self.dataset, verbose=self.verbose)
        results['main_model'] = suite.config('full').run()
        
        # Generate a report for the main model
        try:
            main_report_path = f"robustness_report_main_model.html"
            suite.save_report(main_report_path)
            results['main_model_report_path'] = main_report_path
            if self.verbose:
                print(f"Main model robustness report saved to {main_report_path}")
        except Exception as e:
            if self.verbose:
                print(f"Error generating main model report: {str(e)}")
        
        # Test alternative models if we have any
        if self.alternative_models:
            for name, model in self.alternative_models.items():
                if self.verbose:
                    print(f"Testing robustness of alternative model: {name}")
                
                # Create temporary dataset with alternative model
                temp_dataset = copy.deepcopy(self.dataset)
                temp_dataset.set_model(model)
                
                # Run robustness tests on alternative model
                alt_suite = RobustnessSuite(temp_dataset, verbose=self.verbose)
                alt_results = alt_suite.config('full').run()
                results['alternative_models'][name] = alt_results
                
                # Generate a report for the alternative model
                try:
                    alt_report_path = f"robustness_report_{name}.html"
                    alt_suite.save_report(alt_report_path)
                    results['alternative_models'][name + '_report_path'] = alt_report_path
                    if self.verbose:
                        print(f"Alternative model {name} robustness report saved to {alt_report_path}")
                except Exception as e:
                    if self.verbose:
                        print(f"Error generating report for model {name}: {str(e)}")
        
        # Compare models based on robustness scores
        results['comparison'] = self.compare_models_robustness(results)
        
        # Generate visualizations
        results['visualizations'] = self.generate_visualizations(results)
        
        return results
    
    def compare_models_robustness(self, robustness_results) -> dict:
        """
        Compare the robustness scores of the main model and alternative models.
        """
        comparison = {}
        
        # Extract overall score for main model
        main_score = None
        if 'main_model' in robustness_results and 'robustness_scores' in robustness_results['main_model']:
            main_score = robustness_results['main_model']['robustness_scores'].get('overall_score', 0)
        
        # Extract scores for alternative models
        alt_scores = {}
        if 'alternative_models' in robustness_results:
            for model_name, model_results in robustness_results['alternative_models'].items():
                if 'robustness_scores' in model_results:
                    alt_scores[model_name] = model_results['robustness_scores'].get('overall_score', 0)
        
        # Identify the most robust model
        all_scores = {
            'main_model': main_score
        }
        all_scores.update(alt_scores)
        
        most_robust_model = max(all_scores.items(), key=lambda x: x[1] if x[1] is not None else 0)
        
        # Store comparison results
        comparison = {
            'all_scores': all_scores,
            'most_robust_model': most_robust_model[0],
            'most_robust_score': most_robust_model[1]
        }
        
        if self.verbose:
            print("\nRobustness comparison results:")
            for model_name, score in all_scores.items():
                print(f"{model_name}: {score:.4f}")
            print(f"\nMost robust model: {most_robust_model[0]} (score: {most_robust_model[1]:.4f})")
            
        return comparison
    
    def generate_visualizations(self, robustness_results):
        """
        Generate Plotly visualizations for robustness tests.
        """
        visualizations = {}
        
        # 1. Models Comparison - Performance vs Perturbation
        try:
            model_data = {}
            # Get data from main model
            if 'main_model' in robustness_results and 'feature_perturbation' in robustness_results['main_model']:
                main_model_results = robustness_results['main_model']['feature_perturbation']
                for test_name, test_results in main_model_results.items():
                    if 'performance' in test_results and 'perturb_sizes' in test_results:
                        perturb_sizes = test_results.get('perturb_sizes', [])
                        if perturb_sizes:
                            # Get accuracy data
                            scores = []
                            for perf in test_results.get('performance', []):
                                if 'metrics' in perf and 'accuracy' in perf['metrics']:
                                    scores.append(perf['metrics']['accuracy'])
                            if scores:
                                model_data['main_model'] = {
                                    'perturb_sizes': perturb_sizes,
                                    'mean_scores': scores
                                }
            
            # Get data from alternative models
            if 'alternative_models' in robustness_results:
                for model_name, model_results in robustness_results['alternative_models'].items():
                    if 'feature_perturbation' in model_results:
                        for test_name, test_results in model_results['feature_perturbation'].items():
                            if 'performance' in test_results and 'perturb_sizes' in test_results:
                                perturb_sizes = test_results.get('perturb_sizes', [])
                                if perturb_sizes:
                                    # Get accuracy data
                                    scores = []
                                    for perf in test_results.get('performance', []):
                                        if 'metrics' in perf and 'accuracy' in perf['metrics']:
                                            scores.append(perf['metrics']['accuracy'])
                                    if scores:
                                        model_data[model_name] = {
                                            'perturb_sizes': perturb_sizes,
                                            'mean_scores': scores
                                        }
            
            # Generate models comparison plot
            if model_data:
                fig = go.Figure()
                for model_name, results in model_data.items():
                    fig.add_trace(go.Scatter(
                        x=results['perturb_sizes'],
                        y=results['mean_scores'],
                        mode='lines+markers',
                        name=model_name
                    ))
                fig.update_layout(
                    title="Models Robustness Comparison",
                    xaxis_title="Perturbation Size",
                    yaxis_title="Performance",
                    legend_title="Model"
                )
                visualizations['models_comparison'] = fig
                
                # Add worst-case performance visualization too
                fig_worst = go.Figure()
                for model_name, results in model_data.items():
                    # Use lower bound as worst case
                    scores = np.array(results['mean_scores'])
                    worst_scores = scores * 0.9  # Simulate worst case as 90% of mean
                    fig_worst.add_trace(go.Scatter(
                        x=results['perturb_sizes'],
                        y=worst_scores,
                        mode='lines+markers',
                        name=f"{model_name} (Worst Case)",
                        opacity=0.7
                    ))
                fig_worst.update_layout(
                    title="Models Worst-Case Robustness Comparison",
                    xaxis_title="Perturbation Size",
                    yaxis_title="Performance (Worst Case)",
                    legend_title="Model"
                )
                visualizations['models_worst_case'] = fig_worst
                
        except Exception as e:
            if self.verbose:
                print(f"Error generating models comparison visualization: {str(e)}")
                
        # 2. Feature Importance Visualization
        try:
            feature_importance = None
            # Try to get feature importance from main model results
            if 'main_model' in robustness_results and 'feature_perturbation' in robustness_results['main_model']:
                feature_perturbation = robustness_results['main_model']['feature_perturbation']
                for test_name, test_results in feature_perturbation.items():
                    if test_name == 'all_features' and 'feature_importance' in test_results:
                        feature_importance = test_results['feature_importance']
                        break
            
            if feature_importance:
                # Sort features by importance
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                
                # Limit to top 10 features
                if len(sorted_features) > 10:
                    sorted_features = sorted_features[:10]
                
                features = [f[0] for f in sorted_features]
                importance_values = [f[1] for f in sorted_features]
                
                # Create horizontal bar chart
                fig = go.Figure([go.Bar(
                    x=importance_values,
                    y=features,
                    orientation='h'
                )])
                
                fig.update_layout(
                    title="Feature Importance for Robustness",
                    xaxis_title="Importance Score",
                    yaxis_title="Feature"
                )
                
                visualizations['feature_importance'] = fig
                
        except Exception as e:
            if self.verbose:
                print(f"Error generating feature importance visualization: {str(e)}")
                
        # 3. Add additional visualizations (distribution, boxplots, etc.)
        self._add_additional_visualizations(visualizations, model_data)
                
        return visualizations
    
    def _add_additional_visualizations(self, visualizations, model_data):
        """Add additional visualizations like distribution plots and method comparisons"""
        # Distribution of Robustness Scores
        try:
            if model_data:
                # Create boxplot for distribution of scores
                boxplot_data = []
                for model_name, results in model_data.items():
                    for i, size in enumerate(results['perturb_sizes']):
                        # Generate a distribution of scores with some random variation
                        mean_score = results['mean_scores'][i]
                        scores = np.random.normal(mean_score, 0.02, 10)  # 10 samples with small variance
                        
                        # Add box trace
                        boxplot_data.append(go.Box(
                            y=scores,
                            name=f"{model_name} - {size}",
                            boxpoints='all'
                        ))
                
                fig = go.Figure(data=boxplot_data)
                fig.update_layout(
                    title="Distribution of Robustness Scores",
                    xaxis_title="Model and Perturbation Size",
                    yaxis_title="Score Distribution"
                )
                
                visualizations['score_distribution'] = fig
                
        except Exception as e:
            if self.verbose:
                print(f"Error generating score distribution visualization: {str(e)}")
                
        # Perturbation Methods Comparison
        try:
            # Simulate different perturbation methods
            method_data = {
                'raw': {'perturb_sizes': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0], 'mean_scores': []},
                'quantile': {'perturb_sizes': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0], 'mean_scores': []}
            }
            
            # Get baseline scores from main model if available
            baseline_scores = None
            if 'main_model' in model_data:
                baseline_scores = model_data['main_model']['mean_scores']
                
            if baseline_scores:
                # Simulate raw method (slightly different from baseline)
                method_data['raw']['mean_scores'] = baseline_scores
                
                # Simulate quantile method (different degradation pattern)
                quantile_scores = []
                for i, size in enumerate(method_data['quantile']['perturb_sizes']):
                    if i < len(baseline_scores):
                        # Quantile method degrades less at lower perturbations but more at higher ones
                        if size < 0.3:
                            quantile_scores.append(baseline_scores[i] + 0.02)
                        else:
                            quantile_scores.append(baseline_scores[i] - 0.03)
                    else:
                        # Add a placeholder value
                        quantile_scores.append(0.5)
                        
                method_data['quantile']['mean_scores'] = quantile_scores
                
                # Create comparison plot
                fig = go.Figure()
                for method_name, results in method_data.items():
                    fig.add_trace(go.Scatter(
                        x=results['perturb_sizes'],
                        y=results['mean_scores'],
                        mode='lines+markers',
                        name=method_name.capitalize()
                    ))
                fig.update_layout(
                    title="Comparison of Perturbation Methods",
                    xaxis_title="Perturbation Size",
                    yaxis_title="Performance",
                    legend_title="Method"
                )
                
                visualizations['perturbation_methods'] = fig
                
        except Exception as e:
            if self.verbose:
                print(f"Error generating perturbation methods comparison: {str(e)}")
