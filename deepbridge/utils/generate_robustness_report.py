import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
import os
import json
import plotly.utils


from deepbridge.validation import RobustnessTest, RobustnessScore
from deepbridge.visualization import RobustnessViz
from deepbridge.utils.robustness_report_generator import RobustnessReportGenerator

def generate_robustness_report(
    models: Dict[str, Any],
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    perturb_method: str = 'raw',
    perturb_sizes: List[float] = [0.1, 0.3, 0.5, 0.7, 1.0],
    metric: Optional[str] = None,
    n_iterations: int = 10,
    alpha: float = 0.1,
    analyze_features: bool = True,
    compare_methods: bool = True,
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    template_path: Optional[str] = None,
    random_state: Optional[int] = None
) -> Tuple[str, Dict]:
    """
    Generate a comprehensive robustness report for the given models.
    
    This function runs a complete robustness analysis and generates an HTML report
    with visualizations, metrics, and recommendations.
    
    Args:
        models: Dictionary of models to evaluate
        X: Input features
        y: Target values
        perturb_method: Perturbation method ('raw', 'quantile', or 'categorical')
        perturb_sizes: List of perturbation sizes to test
        metric: Evaluation metric name
        n_iterations: Number of iterations per perturbation size
        alpha: Proportion of worst samples to consider
        analyze_features: Whether to analyze feature importance
        compare_methods: Whether to compare different perturbation methods
        output_path: Optional path to save the HTML report
        output_dir: Optional directory to save the report and related assets
        template_path: Optional path to a custom report template
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple[str, Dict]: HTML report content and a dictionary of analysis results
    """
    # Initialize the robustness test
    robustness_test = RobustnessTest()
    
    # Run robustness evaluation
    results = robustness_test.evaluate_robustness(
        models=models,
        X=X,
        y=y,
        perturb_method=perturb_method,
        perturb_sizes=perturb_sizes,
        metric=metric,
        n_iterations=n_iterations,
        alpha=alpha,
        random_state=random_state
    )
    
    # Calculate robustness indices
    robustness_indices = RobustnessScore.calculate_robustness_index(
        results=results,
        metric=metric
    )
    
    # Generate visualizations with proper error handling and JSON conversion
    try:
        models_comparison_fig = RobustnessViz.plot_models_comparison(
            results=results,
            metric_name=metric
        )
        # Convert to JSON for proper template embedding
        models_comparison_json = json.dumps(models_comparison_fig, cls=plotly.utils.PlotlyJSONEncoder)
        print(f"Tamanho do JSON do modelo de comparação: {len(models_comparison_json)}")
    except Exception as e:
        print(f"Warning: Error creating models comparison visualization: {e}")
        models_comparison_json = "{}"
    
    # Get the best model for detailed analysis
    if robustness_indices:
        best_model_name = max(robustness_indices.items(), key=lambda x: x[1])[0]
    else:
        best_model_name = next(iter(models.keys()))
    
    try:
        boxplot_fig = RobustnessViz.plot_boxplot_performance(
            results=results,
            model_name=best_model_name,
            metric_name=metric
        )
        # Convert to JSON for proper template embedding
        boxplot_json = json.dumps(boxplot_fig, cls=plotly.utils.PlotlyJSONEncoder)
        print(f"Tamanho do JSON do boxplot: {len(boxplot_json)}")
    except Exception as e:
        print(f"Warning: Error creating boxplot visualization: {e}")
        boxplot_json = "{}"
    
    # Analyze feature importance if requested
    feature_importance_results = None
    feature_importance_json = "{}"
    
    if analyze_features and len(models) > 0:
        try:
            # Select the best model for feature importance analysis
            best_model = models[best_model_name]
            
            feature_importance_results = robustness_test.analyze_feature_importance(
                model=best_model,
                X=X,
                y=y,
                perturb_method=perturb_method,
                perturb_size=0.5,  # Use mid-range perturbation
                metric=metric,
                n_iterations=max(5, n_iterations // 2),  # Use fewer iterations for speed
                random_state=random_state
            )
            
            feature_importance_fig = RobustnessViz.plot_feature_importance(
                feature_importance_results=feature_importance_results,
                title=f'Feature Importance for {best_model_name}'
            )
            # Convert to JSON for proper template embedding
            feature_importance_json = json.dumps(feature_importance_fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            print(f"Warning: Error creating feature importance visualization: {e}")
    
    # Compare perturbation methods if requested
    perturbation_methods_json = "{}"
    
    if compare_methods and len(models) > 0:
        try:
            # Select the best model for methods comparison
            best_model = models[best_model_name]
            
            methods_results = robustness_test.compare_perturbation_methods(
                model=best_model,
                X=X,
                y=y,
                perturb_methods=['raw', 'quantile'],
                perturb_sizes=perturb_sizes,
                metric=metric,
                n_iterations=max(5, n_iterations // 2),  # Use fewer iterations for speed
                random_state=random_state
            )
            
            perturbation_methods_fig = RobustnessViz.plot_perturbation_methods_comparison(
                methods_comparison_results=methods_results,
                metric_name=metric
            )
            # Convert to JSON for proper template embedding
            perturbation_methods_json = json.dumps(perturbation_methods_fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            print(f"Warning: Error creating perturbation methods visualization: {e}")
    
    # Initialize report generator
    report_generator = RobustnessReportGenerator(template_path=template_path)
    
    # Determine output path if not provided
    if output_path is None and output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'robustness_report.html')
    
    # Generate the report with JSON strings for figures instead of figure objects
    html_report = report_generator.generate_report(
        robustness_results=results,
        robustness_indices=robustness_indices,
        feature_importance_chart=feature_importance_json,
        boxplot_chart=boxplot_json,
        model_comparison_chart=models_comparison_json,
        perturbation_methods_chart=perturbation_methods_json,
        feature_importance_results=feature_importance_results,
        output_path=output_path
    )

    
    
    # Collect all analysis results
    analysis_results = {
        'robustness_results': results,
        'robustness_indices': robustness_indices,
        'feature_importance_results': feature_importance_results
    }
    
    return html_report, analysis_results