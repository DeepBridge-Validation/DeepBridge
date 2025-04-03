import pandas as pd
import numpy as np
import typing as t
from pathlib import Path
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from ..visualization import comparison, distribution

def generate_quality_report(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    quality_metrics: dict,
    report_path: t.Optional[t.Union[str, Path]] = None,
    generator_info: str = "",
    include_data_samples: bool = True,
    report_format: str = 'html',
    include_visualizations: bool = True,
    **kwargs
) -> str:
    """
    Generate a detailed quality report for synthetic data.
    
    Args:
        real_data: Original real dataset
        synthetic_data: Generated synthetic dataset
        quality_metrics: Quality metrics dictionary
        report_path: Path to save the report (None for default location)
        generator_info: Information about the generator used
        include_data_samples: Whether to include samples of real/synthetic data
        report_format: Format of the report ('html', 'md', or 'json')
        include_visualizations: Whether to include data visualizations in the report
        **kwargs: Additional parameters
    
    Returns:
        Path to the generated report
    """
    # Create report filename if not provided
    if report_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_data_report_{timestamp}"
        
        if report_format == 'html':
            filename += ".html"
        elif report_format == 'md':
            filename += ".md"
        else:  # default to json
            filename += ".json"
        
        report_path = Path("reports") / filename
    else:
        report_path = Path(report_path)
    
    # Create directory if it doesn't exist
    os.makedirs(report_path.parent, exist_ok=True)
    
    # Create directory for visualizations
    viz_dir = report_path.parent / "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Generate visualizations if needed
    visualization_paths = {}
    if include_visualizations and report_format == 'html':
        visualization_paths = _generate_visualizations(
            real_data, synthetic_data, viz_dir, report_path.parent, **kwargs
        )
    
    # Generate the report based on the format
    if report_format == 'html':
        report_content = _generate_html_report(
            real_data, 
            synthetic_data, 
            quality_metrics, 
            generator_info, 
            include_data_samples,
            include_visualizations,
            report_path=report_path,
            visualization_paths=visualization_paths,
            **kwargs
        )
    elif report_format == 'md':
        report_content = _generate_markdown_report(
            real_data, 
            synthetic_data, 
            quality_metrics, 
            generator_info, 
            include_data_samples,
            **kwargs
        )
    else:  # default to json
        report_content = _generate_json_report(
            real_data, 
            synthetic_data, 
            quality_metrics, 
            generator_info, 
            include_data_samples,
            **kwargs
        )
    
    # Write report to file
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"Report saved to: {report_path}")
    return str(report_path)

def _generate_visualizations(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    viz_dir: Path,
    report_dir: Path,
    **kwargs
) -> dict:
    """Generate visualizations for the report and return paths."""
    viz_paths = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # Determine columns to visualize (limit to 5)
        common_cols = list(set(real_data.columns) & set(synthetic_data.columns))
        viz_cols = common_cols[:5] if len(common_cols) > 5 else common_cols
        
        # Get numerical and categorical columns
        numerical_cols = [col for col in viz_cols if pd.api.types.is_numeric_dtype(real_data[col])]
        categorical_cols = [col for col in viz_cols if not pd.api.types.is_numeric_dtype(real_data[col])]
        
        # Create distribution plot
        if viz_cols:
            dist_plot_path = viz_dir / f"distributions_{timestamp}.png"
            
            # Generate the plot
            comparison.plot_distributions(
                real_data=real_data, 
                synthetic_data=synthetic_data,
                columns=viz_cols,
                numerical_columns=numerical_cols,
                categorical_columns=categorical_cols,
                save_path=dist_plot_path,
                figsize=(15, 12),
                dpi=100
            )
            
            # Store relative path
            viz_paths['distributions'] = os.path.relpath(dist_plot_path, report_dir)
        
        # Create correlation matrix comparison if we have at least 2 numerical columns
        if len(numerical_cols) >= 2:
            corr_plot_path = viz_dir / f"correlations_{timestamp}.png"
            
            # Generate the plot
            distribution.plot_correlation_comparison(
                real_data=real_data, 
                synthetic_data=synthetic_data,
                columns=numerical_cols[:5],  # Limit to 5
                save_path=corr_plot_path,
                figsize=(15, 6),
                dpi=100
            )
            
            # Store relative path
            viz_paths['correlations'] = os.path.relpath(corr_plot_path, report_dir)
        
    except Exception as e:
        print(f"Error generating visualizations: {str(e)}")
    
    return viz_paths

def _generate_html_report(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    quality_metrics: dict,
    generator_info: str,
    include_data_samples: bool,
    include_visualizations: bool = True,
    report_path: t.Optional[Path] = None,
    visualization_paths: t.Optional[dict] = None,
    **kwargs
) -> str:
    """Generate an HTML report on synthetic data quality."""
    
    # Use Jinja2 to render the template with data
    import jinja2
    import os
    
    # Path to the central templates directory
    template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'reports', 'templates')
    
    # Initialize Jinja2 environment
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        autoescape=jinja2.select_autoescape(['html', 'xml'])
    )
    
    try:
        # Load the template
        template = env.get_template('synthetic_report.html')
        
        # Prepare timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare data for the template
        template_data = {
            'generation_date': timestamp,
            'generator_info': {
                'method': generator_info,
                'num_samples': len(synthetic_data),
                'random_state': "N/A"  # This could be extracted from kwargs if available
            },
            'quality_score': quality_metrics.get('overall', {}).get('statistical_similarity', 0.75),
            'metrics': {
                'overall': quality_metrics.get('overall', {})
            },
            'numerical_cols': [col for col in real_data.columns if col in synthetic_data.columns 
                             and pd.api.types.is_numeric_dtype(real_data[col])],
            'categorical_cols': [col for col in real_data.columns if col in synthetic_data.columns 
                               and not pd.api.types.is_numeric_dtype(real_data[col])],
            'detailed_metrics': {
                'statistical': quality_metrics.get('numerical', {}),
                'privacy': quality_metrics.get('privacy', {}),
                'utility': quality_metrics.get('categorical', {})
            },
            'real_sample_columns': real_data.columns[:5].tolist(),
            'real_sample_data': real_data.head(5).values.tolist(),
            'synthetic_sample_columns': synthetic_data.columns[:5].tolist(),
            'synthetic_sample_data': synthetic_data.head(5).values.tolist(),
            'numerical_data': [],  # Would need to be prepared
            'categorical_data': [], # Would need to be prepared
            'correlation_data': {},  # Would need to be prepared
            'visualization_paths': visualization_paths or {}
        }
        
        # Render the template
        return template.render(**template_data)
        
    except Exception as e:
        print(f"Error using template: {str(e)}. Falling back to basic HTML.")
        
        # Basic HTML structure with embedded CSS
        html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Synthetic Data Quality Report</title>
        <style>
            body { 
                font-family: Arial, sans-serif; 
                line-height: 1.6; 
                margin: 0; 
                padding: 20px; 
                color: #333; 
                background-color: #f8f9fa;
            }
            h1, h2, h3 { 
                color: #2c3e50; 
                margin-top: 1.5em;
                margin-bottom: 0.8em;
            }
            .container { 
                max-width: 1200px; 
                margin: 0 auto; 
                background-color: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .header { 
                background: linear-gradient(135deg, #0062cc 0%, #1e88e5 100%);
                color: white; 
                padding: 30px 20px; 
                margin-bottom: 30px; 
                border-radius: 8px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .section { 
                margin-bottom: 40px; 
                padding: 20px; 
                background-color: #fff; 
                border-radius: 8px; 
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border: 1px solid #eee;
            }
            table { 
                width: 100%; 
                border-collapse: collapse; 
                margin-bottom: 20px; 
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            th, td { 
                padding: 12px 15px; 
                text-align: left; 
                border-bottom: 1px solid #ddd; 
            }
            th { 
                background-color: #f2f2f2; 
                font-weight: bold;
                border-top: 1px solid #ddd;
                border-bottom: 2px solid #ddd;
            }
            tr:hover { 
                background-color: #f5f5f5; 
            }
            .metric-good { 
                color: #28a745; 
                font-weight: bold;
            }
            .metric-medium { 
                color: #ffc107; 
                font-weight: bold;
            }
            .metric-poor { 
                color: #dc3545; 
                font-weight: bold;
            }
            .summary { 
                font-weight: bold; 
                margin-top: 10px; 
            }
            .timestamp { 
                font-size: 0.9em; 
                color: rgba(255,255,255,0.8); 
                margin-top: 10px; 
            }
            img { 
                max-width: 100%; 
                height: auto; 
                margin: 15px 0; 
                border: 1px solid #ddd; 
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            pre {
                background-color: #f6f8fa;
                border: 1px solid #e1e4e8;
                border-radius: 6px;
                padding: 15px;
                font-family: monospace;
                overflow-x: auto;
            }
            .footer {
                text-align: center;
                padding: 20px;
                margin-top: 30px;
                color: #6c757d;
                font-size: 0.9em;
                border-top: 1px solid #eee;
            }
        </style>
    </head>
    <body>
        <div class="container">
    """
    
    # Header section
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html += f"""
            <div class="header">
                <h1>Synthetic Data Quality Report</h1>
                <p>This report evaluates the quality of synthetic data compared to real data.</p>
                <p class="timestamp">Generated on: {timestamp}</p>
            </div>
    """
    
    # Dataset overview
    html += f"""
            <div class="section">
                <h2>Dataset Overview</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Real Data</th>
                        <th>Synthetic Data</th>
                    </tr>
                    <tr>
                        <td>Number of rows</td>
                        <td>{len(real_data)}</td>
                        <td>{len(synthetic_data)}</td>
                    </tr>
                    <tr>
                        <td>Number of columns</td>
                        <td>{len(real_data.columns)}</td>
                        <td>{len(synthetic_data.columns)}</td>
                    </tr>
                    <tr>
                        <td>Memory usage</td>
                        <td>{real_data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB</td>
                        <td>{synthetic_data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB</td>
                    </tr>
                </table>
                
                <h3>Generator Information</h3>
                <pre>{generator_info}</pre>
            </div>
    """
    
    # Overall metrics
    html += """
            <div class="section">
                <h2>Overall Quality Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                        <th>Interpretation</th>
                    </tr>
    """
    
    # Check if overall metrics exist
    if 'overall' in quality_metrics:
        for key, value in quality_metrics['overall'].items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                # Determine quality level
                quality_class = "metric-medium"
                interpretation = "Acceptable"
                
                if "error" in key.lower() or "difference" in key.lower():
                    if value < 0.1:
                        quality_class = "metric-good"
                        interpretation = "Excellent"
                    elif value < 0.2:
                        quality_class = "metric-good"
                        interpretation = "Good"
                    elif value > 0.3:
                        quality_class = "metric-poor"
                        interpretation = "Poor"
                
                elif "ks_statistic" in key.lower():
                    if value < 0.1:
                        quality_class = "metric-good"
                        interpretation = "Distributions are very similar"
                    elif value < 0.2:
                        quality_class = "metric-good"
                        interpretation = "Distributions are similar"
                    elif value > 0.3:
                        quality_class = "metric-poor"
                        interpretation = "Distributions differ significantly"
                
                elif "score" in key.lower():
                    if value > 0.9:
                        quality_class = "metric-good"
                        interpretation = "Excellent"
                    elif value > 0.8:
                        quality_class = "metric-good"
                        interpretation = "Good"
                    elif value > 0.7:
                        quality_class = "metric-medium"
                        interpretation = "Fair"
                    elif value > 0.6:
                        quality_class = "metric-medium"
                        interpretation = "Acceptable"
                    else:
                        quality_class = "metric-poor"
                        interpretation = "Poor"
                
                html += f"""
                        <tr>
                            <td>{key}</td>
                            <td class="{quality_class}">{value:.4f}</td>
                            <td>{interpretation}</td>
                        </tr>
                """
            elif value is not None:
                html += f"""
                        <tr>
                            <td>{key}</td>
                            <td>{value}</td>
                            <td>-</td>
                        </tr>
                """
    
    html += """
                </table>
            </div>
    """
    
    # Numerical metrics
    if 'numerical' in quality_metrics and quality_metrics['numerical']:
        html += """
            <div class="section">
                <h2>Numerical Column Metrics</h2>
        """
        
        for col, col_metrics in quality_metrics['numerical'].items():
            html += f"""
                <h3>Column: {col}</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """
            
            if 'error' in col_metrics:
                html += f"""
                    <tr>
                        <td>Error</td>
                        <td class="metric-poor">{col_metrics['error']}</td>
                    </tr>
                """
            else:
                # Basic statistics
                for metric in ['mean_real', 'mean_synthetic', 'mean_relative_error',
                               'std_real', 'std_synthetic', 'std_relative_error',
                               'min_real', 'min_synthetic', 'max_real', 'max_synthetic']:
                    if metric in col_metrics and not pd.isna(col_metrics[metric]):
                        html += f"""
                            <tr>
                                <td>{metric}</td>
                                <td>{col_metrics[metric]:.4f}</td>
                            </tr>
                        """
                
                # Statistical tests
                if 'ks_statistic' in col_metrics and not pd.isna(col_metrics['ks_statistic']):
                    quality_class = "metric-good" if col_metrics['ks_statistic'] < 0.1 else \
                                   "metric-medium" if col_metrics['ks_statistic'] < 0.2 else \
                                   "metric-poor"
                    
                    html += f"""
                        <tr>
                            <td>Kolmogorov-Smirnov statistic</td>
                            <td class="{quality_class}">{col_metrics['ks_statistic']:.4f}</td>
                        </tr>
                        <tr>
                            <td>Kolmogorov-Smirnov p-value</td>
                            <td>{col_metrics['ks_pvalue']:.4f}</td>
                        </tr>
                    """
                
                if 'jensen_shannon_dist' in col_metrics and not pd.isna(col_metrics['jensen_shannon_dist']):
                    quality_class = "metric-good" if col_metrics['jensen_shannon_dist'] < 0.1 else \
                                   "metric-medium" if col_metrics['jensen_shannon_dist'] < 0.2 else \
                                   "metric-poor"
                    
                    html += f"""
                        <tr>
                            <td>Jensen-Shannon distance</td>
                            <td class="{quality_class}">{col_metrics['jensen_shannon_dist']:.4f}</td>
                        </tr>
                    """
            
            html += """
                </table>
            """
        
        html += """
            </div>
        """
    
    # Categorical metrics
    if 'categorical' in quality_metrics and quality_metrics['categorical']:
        html += """
            <div class="section">
                <h2>Categorical Column Metrics</h2>
        """
        
        for col, col_metrics in quality_metrics['categorical'].items():
            html += f"""
                <h3>Column: {col}</h3>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """
            
            if 'error' in col_metrics:
                html += f"""
                    <tr>
                        <td>Error</td>
                        <td class="metric-poor">{col_metrics['error']}</td>
                    </tr>
                """
            else:
                # Basic statistics
                for metric in ['category_count_real', 'category_count_synthetic', 
                               'category_coverage', 'missing_categories', 'extra_categories']:
                    if metric in col_metrics:
                        html += f"""
                            <tr>
                                <td>{metric}</td>
                                <td>{col_metrics[metric]}</td>
                            </tr>
                        """
                
                # Distribution difference
                if 'distribution_difference' in col_metrics and not pd.isna(col_metrics['distribution_difference']):
                    quality_class = "metric-good" if col_metrics['distribution_difference'] < 0.1 else \
                                   "metric-medium" if col_metrics['distribution_difference'] < 0.2 else \
                                   "metric-poor"
                    
                    html += f"""
                        <tr>
                            <td>Distribution difference</td>
                            <td class="{quality_class}">{col_metrics['distribution_difference']:.4f}</td>
                        </tr>
                    """
                
                # Chi-square test
                if 'chi2_pvalue' in col_metrics and not pd.isna(col_metrics['chi2_pvalue']):
                    quality_class = "metric-good" if col_metrics['chi2_pvalue'] > 0.05 else \
                                   "metric-poor"
                    
                    html += f"""
                        <tr>
                            <td>Chi-square statistic</td>
                            <td>{col_metrics['chi2_statistic']:.4f}</td>
                        </tr>
                        <tr>
                            <td>Chi-square p-value</td>
                            <td class="{quality_class}">{col_metrics['chi2_pvalue']:.4f}</td>
                        </tr>
                    """
            
            html += """
                </table>
            """
        
        html += """
            </div>
        """
    
    # Add visualizations if requested
    if include_visualizations and visualization_paths:
        html += """
            <div class="section">
                <h2>Data Visualizations</h2>
        """
        
        # Distributions visualization
        if 'distributions' in visualization_paths:
            dist_path = visualization_paths['distributions']
            html += f"""
                <h3>Distributions Comparison</h3>
                <img src="{dist_path}" alt="Distribution comparison">
            """
        
        # Correlations visualization
        if 'correlations' in visualization_paths:
            corr_path = visualization_paths['correlations']
            html += f"""
                <h3>Correlation Matrix Comparison</h3>
                <img src="{corr_path}" alt="Correlation comparison">
            """
        
        html += """
            </div>
        """
    
    # Data samples
    if include_data_samples:
        html += """
            <div class="section">
                <h2>Data Samples</h2>
                
                <h3>Real Data Sample</h3>
        """
        
        real_sample = real_data.head(5).to_html(index=False)
        html += real_sample
        
        html += """
                <h3>Synthetic Data Sample</h3>
        """
        
        synth_sample = synthetic_data.head(5).to_html(index=False)
        html += synth_sample
        
        html += """
            </div>
        """
    
    # Footer
    html += """
            <div class="footer">
                <p>This report was generated using DeepBridge Synthetic Data Generator.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def _generate_markdown_report(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    quality_metrics: dict,
    generator_info: str,
    include_data_samples: bool,
    **kwargs
) -> str:
    """Generate a Markdown report on synthetic data quality."""
    # Header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md = f"""# Synthetic Data Quality Report

Generated on: {timestamp}

## Dataset Overview

| Metric | Real Data | Synthetic Data |
|--------|-----------|---------------|
| Number of rows | {len(real_data)} | {len(synthetic_data)} |
| Number of columns | {len(real_data.columns)} | {len(synthetic_data.columns)} |
| Memory usage | {real_data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB | {synthetic_data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB |

### Generator Information

```
{generator_info}
```

## Overall Quality Metrics

| Metric | Value | Interpretation |
|--------|-------|---------------|
"""
    
    if 'overall' in quality_metrics:
        for key, value in quality_metrics['overall'].items():
            if isinstance(value, (int, float)) and not pd.isna(value):
                # Determine quality level
                interpretation = "Acceptable"
                
                if "error" in key.lower() or "difference" in key.lower():
                    if value < 0.1:
                        interpretation = "Excellent"
                    elif value < 0.2:
                        interpretation = "Good"
                    elif value > 0.3:
                        interpretation = "Poor"
                
                elif "ks_statistic" in key.lower():
                    if value < 0.1:
                        interpretation = "Distributions are very similar"
                    elif value < 0.2:
                        interpretation = "Distributions are similar"
                    elif value > 0.3:
                        interpretation = "Distributions differ significantly"
                
                elif "score" in key.lower():
                    if value > 0.9:
                        interpretation = "Excellent"
                    elif value > 0.8:
                        interpretation = "Good"
                    elif value > 0.7:
                        interpretation = "Fair"
                    else:
                        interpretation = "Poor"
                
                md += f"| {key} | {value:.4f} | {interpretation} |\n"
            elif value is not None:
                md += f"| {key} | {value} | - |\n"
    
    # Numerical metrics
    if 'numerical' in quality_metrics and quality_metrics['numerical']:
        md += "\n## Numerical Column Metrics\n"
        
        for col, col_metrics in quality_metrics['numerical'].items():
            md += f"\n### Column: {col}\n\n"
            
            if 'error' in col_metrics:
                md += f"**Error**: {col_metrics['error']}\n"
            else:
                md += "| Metric | Value |\n|--------|-------|\n"
                
                # Basic statistics
                for metric in ['mean_real', 'mean_synthetic', 'mean_relative_error',
                               'std_real', 'std_synthetic', 'std_relative_error',
                               'min_real', 'min_synthetic', 'max_real', 'max_synthetic']:
                    if metric in col_metrics and not pd.isna(col_metrics[metric]):
                        md += f"| {metric} | {col_metrics[metric]:.4f} |\n"
                
                # Statistical tests
                if 'ks_statistic' in col_metrics and not pd.isna(col_metrics['ks_statistic']):
                    md += f"| Kolmogorov-Smirnov statistic | {col_metrics['ks_statistic']:.4f} |\n"
                    md += f"| Kolmogorov-Smirnov p-value | {col_metrics['ks_pvalue']:.4f} |\n"
                
                if 'jensen_shannon_dist' in col_metrics and not pd.isna(col_metrics['jensen_shannon_dist']):
                    md += f"| Jensen-Shannon distance | {col_metrics['jensen_shannon_dist']:.4f} |\n"
    
    # Categorical metrics
    if 'categorical' in quality_metrics and quality_metrics['categorical']:
        md += "\n## Categorical Column Metrics\n"
        
        for col, col_metrics in quality_metrics['categorical'].items():
            md += f"\n### Column: {col}\n\n"
            
            if 'error' in col_metrics:
                md += f"**Error**: {col_metrics['error']}\n"
            else:
                md += "| Metric | Value |\n|--------|-------|\n"
                
                # Basic statistics
                for metric in ['category_count_real', 'category_count_synthetic', 
                               'category_coverage', 'missing_categories', 'extra_categories']:
                    if metric in col_metrics:
                        md += f"| {metric} | {col_metrics[metric]} |\n"
                
                # Distribution difference
                if 'distribution_difference' in col_metrics and not pd.isna(col_metrics['distribution_difference']):
                    md += f"| Distribution difference | {col_metrics['distribution_difference']:.4f} |\n"
                
                # Chi-square test
                if 'chi2_pvalue' in col_metrics and not pd.isna(col_metrics['chi2_pvalue']):
                    md += f"| Chi-square statistic | {col_metrics['chi2_statistic']:.4f} |\n"
                    md += f"| Chi-square p-value | {col_metrics['chi2_pvalue']:.4f} |\n"
    
    # Data samples
    if include_data_samples:
        md += "\n## Data Samples\n"
        
        md += "\n### Real Data Sample\n\n"
        real_sample = real_data.head(5).to_markdown(index=False)
        md += real_sample
        
        md += "\n\n### Synthetic Data Sample\n\n"
        synth_sample = synthetic_data.head(5).to_markdown(index=False)
        md += synth_sample
    
    # Footer
    md += "\n\n---\nThis report was generated using DeepBridge Synthetic Data Generator.\n"
    
    return md

def _generate_json_report(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    quality_metrics: dict,
    generator_info: str,
    include_data_samples: bool,
    **kwargs
) -> str:
    """Generate a JSON report on synthetic data quality."""
    # Create report structure
    report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_overview": {
            "real_data_rows": len(real_data),
            "synthetic_data_rows": len(synthetic_data),
            "real_data_columns": len(real_data.columns),
            "synthetic_data_columns": len(synthetic_data.columns),
            "real_data_memory_mb": real_data.memory_usage(deep=True).sum() / (1024 * 1024),
            "synthetic_data_memory_mb": synthetic_data.memory_usage(deep=True).sum() / (1024 * 1024)
        },
        "generator_info": generator_info,
        "quality_metrics": quality_metrics
    }
    
    # Add data samples if requested
    if include_data_samples:
        report["data_samples"] = {
            "real_data": real_data.head(5).to_dict(orient='records'),
            "synthetic_data": synthetic_data.head(5).to_dict(orient='records')
        }
    
    # Helper function to convert numpy types to Python native types
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif pd.isna(obj):
            return None
        else:
            return obj
    
    # Convert numpy types to Python native types
    report_native = convert_numpy(report)
    
    # Convert to JSON string
    return json.dumps(report_native, indent=2, default=str)