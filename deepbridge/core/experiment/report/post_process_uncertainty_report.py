"""
Post-processing script for uncertainty reports to fix missing charts.
This script can be used to extract base64 encoded images from logs and insert them into the HTML report.
"""

import os
import re
import base64
import logging
import argparse
from typing import Dict, Any, Optional

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("deepbridge.post_process")

def extract_base64_from_log(log_file: str, chart_name: str) -> Optional[str]:
    """
    Extract base64 encoded image data for a specific chart from a log file.
    
    Args:
        log_file: Path to the log file
        chart_name: Name of the chart to extract
        
    Returns:
        Base64 encoded image data or None if not found
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            log_content = f.read()
            
        # Look for patterns like "Chart {chart_name} has data of length 12345"
        chart_pattern = re.compile(rf"Chart {re.escape(chart_name)} has data of length (\d+)")
        matches = chart_pattern.findall(log_content)
        
        if matches:
            logger.info(f"Found chart {chart_name} with data length {matches[0]}")
            
            # Now try to find the actual base64 data
            # Assuming the base64 data starts with "data:image/png;base64,"
            base64_pattern = re.compile(r'data:image/png;base64,([A-Za-z0-9+/=]+)')
            base64_matches = base64_pattern.findall(log_content)
            
            if base64_matches:
                # Return the most recent base64 data (assuming it's for our chart)
                return f"data:image/png;base64,{base64_matches[-1]}"
        
        logger.warning(f"Could not find base64 data for chart {chart_name} in the log file")
        return None
    except Exception as e:
        logger.error(f"Error extracting base64 data from log: {str(e)}")
        return None

def fix_html_report(html_file: str, chart_name_mapping: Dict[str, str], log_file: Optional[str] = None) -> bool:
    """
    Fix an HTML report by adding missing charts.
    
    Args:
        html_file: Path to the HTML report
        chart_name_mapping: Mapping of chart names to their expected names in the template
        log_file: Optional path to a log file to extract base64 data from
        
    Returns:
        True if the report was fixed, False otherwise
    """
    try:
        # Read the HTML file
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Create a charts directory if saving images
        charts_dir = os.path.join(os.path.dirname(html_file), 'uncertainty_charts')
        os.makedirs(charts_dir, exist_ok=True)
        
        # Check for missing charts in the report
        fixed = False
        
        # Try to find what chart names are used in the template
        chart_refs = set()
        chart_patterns = [
            r'{%\s*if\s*charts\.(\w+)\s*%}',  # {% if charts.chart_name %}
            r'{{\s*charts\.(\w+)\s*}}'        # {{ charts.chart_name }}
        ]
        
        for pattern in chart_patterns:
            matches = re.findall(pattern, html_content)
            chart_refs.update(matches)
        
        logger.info(f"Found {len(chart_refs)} chart references in the HTML: {chart_refs}")
        
        # Check each chart in our mapping
        for chart_name, template_name in chart_name_mapping.items():
            # Search for the template pattern
            pattern = f"{{{{ charts.{template_name} }}}}"
            
            if pattern in html_content:
                logger.info(f"Found chart placeholder: {pattern}")
                
                # Check if it's already been replaced
                if f'src="data:image/png;base64,' in html_content:
                    logger.info(f"Chart {template_name} already has image data")
                    continue
                
                # Try to get the base64 data from the log file
                base64_data = None
                if log_file:
                    base64_data = extract_base64_from_log(log_file, chart_name)
                    
                if base64_data:
                    # Replace the placeholder with an image tag
                    img_tag = f'<img src="{base64_data}" alt="{template_name}" />'
                    html_content = html_content.replace(pattern, img_tag)
                    fixed = True
                    logger.info(f"Added image for {template_name} from log data")
        
        if fixed:
            # Write the updated HTML
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Successfully fixed HTML report: {html_file}")
            return True
        else:
            logger.info("No changes needed or no chart data found")
            return False
    except Exception as e:
        logger.error(f"Error fixing HTML report: {str(e)}")
        return False

def create_placeholder_charts(html_file: str, missing_charts: list) -> bool:
    """
    Create placeholder charts for missing charts in the HTML report.
    
    Args:
        html_file: Path to the HTML report
        missing_charts: List of chart names that are missing
        
    Returns:
        True if placeholder charts were added, False otherwise
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        
        # Read the HTML file
        with open(html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        fixed = False
        
        # Create a placeholder chart for each missing chart
        for chart_name in missing_charts:
            # Check if the chart placeholder exists in the HTML
            pattern = f"{{{{ charts.{chart_name} }}}}"
            
            if pattern in html_content:
                logger.info(f"Creating placeholder chart for: {chart_name}")
                
                # Create a placeholder figure
                plt.figure(figsize=(10, 6))
                
                # Create some example data based on the chart type
                if "comparison" in chart_name:
                    # Bar chart for comparisons
                    labels = ['Model A', 'Model B', 'Model C']
                    values = np.random.rand(3) * 10
                    plt.bar(labels, values)
                    plt.ylabel('Score')
                    plt.title(f'Placeholder for {chart_name}')
                elif "distribution" in chart_name or "bandwidth" in chart_name or "widths" in chart_name:
                    # Histogram for distributions
                    data = np.random.normal(0, 1, 100)
                    plt.hist(data, bins=20, alpha=0.7)
                    plt.axvline(np.mean(data), color='red', linestyle='--')
                    plt.xlabel('Value')
                    plt.ylabel('Frequency')
                    plt.title(f'Placeholder for {chart_name}')
                else:
                    # Generic scatter plot
                    x = np.linspace(0, 10, 30)
                    y = np.sin(x) + np.random.normal(0, 0.2, size=30)
                    plt.scatter(x, y)
                    plt.xlabel('X')
                    plt.ylabel('Y')
                    plt.title(f'Placeholder for {chart_name}')
                
                # Add watermark
                plt.figtext(0.5, 0.01, "Placeholder - Not actual data",
                           ha="center", fontsize=8, color="gray")
                
                # Save to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                buffer.seek(0)
                plt.close()
                
                base64_data = f"data:image/png;base64,{base64.b64encode(buffer.read()).decode('utf-8')}"
                
                # Replace the placeholder with an image tag
                img_tag = f'<img src="{base64_data}" alt="{chart_name}" />'
                html_content = html_content.replace(pattern, img_tag)
                fixed = True
                logger.info(f"Added placeholder image for {chart_name}")
        
        if fixed:
            # Write the updated HTML
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Successfully added placeholder charts to HTML report: {html_file}")
            return True
        else:
            logger.info("No placeholder charts were added")
            return False
    except Exception as e:
        logger.error(f"Error creating placeholder charts: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Post-process uncertainty reports to fix missing charts')
    parser.add_argument('html_file', help='Path to the HTML report to fix')
    parser.add_argument('--log', help='Optional path to a log file to extract chart data from')
    parser.add_argument('--placeholder', action='store_true', help='Create placeholder charts for missing charts')
    args = parser.parse_args()
    
    # Define the chart name mapping
    chart_name_mapping = {
        # Bidirectional mappings for reliability charts
        'reliability_distribution': 'feature_reliability',
        'feature_reliability': 'reliability_distribution',
        'reliability_analysis': 'feature_reliability',
        
        # Bidirectional mappings for bandwidth charts
        'marginal_bandwidth': 'interval_widths_comparison',
        'interval_widths_comparison': 'marginal_bandwidth',
        'width_distribution': 'interval_widths_comparison',
        'interval_widths_boxplot': 'interval_widths_comparison',
        
        # Model comparison charts - keep only one reference to performance_gap_by_alpha
        'model_comparison': 'model_comparison',
        'model_metrics_comparison': 'model_comparison',
        'model_metrics': 'model_comparison',
        'model_comparison_chart': 'model_comparison',
        
        # Other charts - performance_gap_by_alpha only in Overview section
        'performance_gap_by_alpha': 'performance_gap_by_alpha',
        'coverage_vs_expected': 'coverage_vs_expected',
        'width_vs_coverage': 'width_vs_coverage',
        'uncertainty_metrics': 'uncertainty_metrics',
        'feature_importance': 'feature_importance'
    }
    
    # First try to fix the report using data from logs
    fixed = fix_html_report(args.html_file, chart_name_mapping, args.log)
    
    # If requested, create placeholder charts for any missing charts
    if args.placeholder and not fixed:
        # Determine which charts are missing by looking for the placeholders in the HTML
        with open(args.html_file, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        missing_charts = []
        for chart_name in chart_name_mapping.values():
            pattern = f"{{{{ charts.{chart_name} }}}}"
            if pattern in html_content:
                missing_charts.append(chart_name)
        
        if missing_charts:
            logger.info(f"Found {len(missing_charts)} missing charts: {missing_charts}")
            create_placeholder_charts(args.html_file, missing_charts)
        else:
            logger.info("No missing chart placeholders found in the HTML")

if __name__ == "__main__":
    main()