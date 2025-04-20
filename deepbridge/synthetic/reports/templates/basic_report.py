"""
Basic report template for synthetic data quality reports.

This module provides templates for generating synthetic data quality reports
in various formats (HTML, Markdown, JSON).
"""

def get_html_template() -> str:
    """Get basic HTML template for quality reports."""
    return """
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
            <!-- Header -->
            <div class="header">
                <h1>Synthetic Data Quality Report</h1>
                <p>This report evaluates the quality of synthetic data compared to real data.</p>
                <p class="timestamp">Generated on: {timestamp}</p>
            </div>
            
            <!-- Dataset Overview -->
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
                        <td>{real_rows}</td>
                        <td>{synthetic_rows}</td>
                    </tr>
                    <tr>
                        <td>Number of columns</td>
                        <td>{real_cols}</td>
                        <td>{synthetic_cols}</td>
                    </tr>
                    <tr>
                        <td>Memory usage</td>
                        <td>{real_memory:.2f} MB</td>
                        <td>{synthetic_memory:.2f} MB</td>
                    </tr>
                </table>
                
                <h3>Generator Information</h3>
                <pre>{generator_info}</pre>
            </div>
            
            <!-- Overall Metrics -->
            <div class="section">
                <h2>Overall Quality Metrics</h2>
                {overall_metrics_html}
            </div>
            
            <!-- Numerical Metrics -->
            <div class="section" id="numerical-metrics">
                <h2>Numerical Column Metrics</h2>
                {numerical_metrics_html}
            </div>
            
            <!-- Categorical Metrics -->
            <div class="section" id="categorical-metrics">
                <h2>Categorical Column Metrics</h2>
                {categorical_metrics_html}
            </div>
            
            <!-- Visualizations -->
            <div class="section" id="visualizations">
                <h2>Data Visualizations</h2>
                {visualizations_html}
            </div>
            
            <!-- Data Samples -->
            <div class="section" id="data-samples">
                <h2>Data Samples</h2>
                {data_samples_html}
            </div>
            
            <!-- Footer -->
            <div class="footer">
                <p>This report was generated using DeepBridge Synthetic Data Generator.</p>
            </div>
        </div>
    </body>
    </html>
    """

def get_markdown_template() -> str:
    """Get basic Markdown template for quality reports."""
    return """
    # Synthetic Data Quality Report
    
    Generated on: {timestamp}
    
    ## Dataset Overview
    
    | Metric | Real Data | Synthetic Data |
    |--------|-----------|---------------|
    | Number of rows | {real_rows} | {synthetic_rows} |
    | Number of columns | {real_cols} | {synthetic_cols} |
    | Memory usage | {real_memory:.2f} MB | {synthetic_memory:.2f} MB |
    
    ### Generator Information
    
    ```
    {generator_info}
    ```
    
    ## Overall Quality Metrics
    
    {overall_metrics_md}
    
    ## Numerical Column Metrics
    
    {numerical_metrics_md}
    
    ## Categorical Column Metrics
    
    {categorical_metrics_md}
    
    ## Data Samples
    
    {data_samples_md}
    
    ---
    Generated using DeepBridge Synthetic Data Generator
    """

def get_json_template() -> str:
    """Get basic JSON template for quality reports."""
    return """{
    "timestamp": "{timestamp}",
    "dataset_overview": {
        "real_data_rows": {real_rows},
        "synthetic_data_rows": {synthetic_rows},
        "real_data_columns": {real_cols},
        "synthetic_data_columns": {synthetic_cols},
        "real_data_memory_mb": {real_memory},
        "synthetic_data_memory_mb": {synthetic_memory}
    },
    "generator_info": "{generator_info_escaped}",
    "quality_metrics": {
        "overall": {overall_metrics_json},
        "numerical": {numerical_metrics_json},
        "categorical": {categorical_metrics_json}
    },
    "data_samples": {
        "real_data": {real_data_sample_json},
        "synthetic_data": {synthetic_data_sample_json}
    }
}
"""