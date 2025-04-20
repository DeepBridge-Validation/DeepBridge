"""
Templates for synthetic data quality reports.

This module provides HTML templates for generating detailed reports
about synthetic data quality metrics and visualizations.
"""

import os
from pathlib import Path

def initialize_templates():
    """Initialize the template directory with default templates."""
    # Get the current directory
    current_dir = Path(__file__).parent
    
    # Create the synthetic_report_template.html file
    template_path = current_dir / "synthetic_report_template.html"
    
    # If the template already exists, skip creation
    if template_path.exists():
        return
    
    # Get the default template content
    default_template_content = _get_default_template_content()
    
    # Create the template file
    with open(template_path, 'w', encoding='utf-8') as f:
        f.write(default_template_content)
    
    print(f"Template initialized at: {template_path}")

def _get_default_template_content():
    """Get the default template content."""
    return """<!DOCTYPE html>
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
        <!-- Template content will be inserted here -->
    </div>
</body>
</html>"""

# Initialize templates when the module is imported
initialize_templates()