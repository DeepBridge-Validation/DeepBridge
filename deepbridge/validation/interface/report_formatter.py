"""
Report formatting utilities.

This module provides tools for formatting and exporting validation
reports in various formats.
"""

import json
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import matplotlib.pyplot as plt


class ReportFormatter:
    """
    Format validation reports in various output formats.
    
    This class provides methods for converting validation results
    into structured reports in various formats.
    """
    
    def __init__(
        self,
        report_data: Dict[str, Any],
        title: str = "Validation Report",
        include_plots: bool = True
    ):
        """
        Initialize the report formatter.
        
        Parameters:
        -----------
        report_data : dict
            Validation results data
        title : str
            Report title
        include_plots : bool
            Whether to include plots in the report
        """
        self.report_data = report_data
        self.title = title
        self.include_plots = include_plots
        self.plots = report_data.get('plots', {})
    
    def to_markdown(self) -> str:
        """
        Format the report as Markdown.
        
        Returns:
        --------
        str : Markdown formatted report
        """
        md = f"# {self.title}\n\n"
        
        # Add summary section
        if 'summary' in self.report_data:
            md += "## Summary\n\n"
            summary = self.report_data['summary']
            
            for key, value in summary.items():
                # Format key
                key_formatted = key.replace('_', ' ').title()
                
                # Format value based on type
                if isinstance(value, float):
                    value_formatted = f"{value:.4f}"
                elif isinstance(value, dict):
                    value_formatted = json.dumps(value, indent=2)
                else:
                    value_formatted = str(value)
                    
                md += f"**{key_formatted}**: {value_formatted}\n\n"
        
        # Add metrics section
        if 'metrics' in self.report_data:
            md += "## Metrics\n\n"
            metrics = self.report_data['metrics']
            
            # Format as table
            md += "| Metric | Value |\n"
            md += "| ------ | ----- |\n"
            
            for key, value in metrics.items():
                # Format key and value
                key_formatted = key.replace('_', ' ').title()
                
                if isinstance(value, float):
                    value_formatted = f"{value:.4f}"
                else:
                    value_formatted = str(value)
                    
                md += f"| {key_formatted} | {value_formatted} |\n"
                
            md += "\n"
        
        # Add details sections
        for section_name, section_data in self.report_data.items():
            if section_name in ['summary', 'metrics', 'plots']:
                continue
                
            # Format section name
            section_formatted = section_name.replace('_', ' ').title()
            md += f"## {section_formatted}\n\n"
            
            if isinstance(section_data, dict):
                # Nested dictionary
                for key, value in section_data.items():
                    # Format key
                    key_formatted = key.replace('_', ' ').title()
                    
                    md += f"### {key_formatted}\n\n"
                    
                    if isinstance(value, dict):
                        # Format as nested list
                        for sub_key, sub_value in value.items():
                            sub_key_formatted = sub_key.replace('_', ' ').title()
                            
                            if isinstance(sub_value, float):
                                sub_value_formatted = f"{sub_value:.4f}"
                            else:
                                sub_value_formatted = str(sub_value)
                                
                            md += f"- **{sub_key_formatted}**: {sub_value_formatted}\n"
                    elif isinstance(value, list):
                        # Format as list
                        for item in value:
                            if isinstance(item, dict):
                                md += "- " + ", ".join([f"{k}: {v}" for k, v in item.items()]) + "\n"
                            else:
                                md += f"- {item}\n"
                    else:
                        # Format as single value
                        if isinstance(value, float):
                            value_formatted = f"{value:.4f}"
                        else:
                            value_formatted = str(value)
                            
                        md += f"{value_formatted}\n"
                        
                    md += "\n"
            elif isinstance(section_data, list):
                # List data
                for item in section_data:
                    if isinstance(item, dict):
                        md += "- " + ", ".join([f"{k}: {v}" for k, v in item.items()]) + "\n"
                    else:
                        md += f"- {item}\n"
                        
                md += "\n"
            else:
                # Simple value
                md += f"{str(section_data)}\n\n"
        
        # Add plots section note
        if self.include_plots and self.plots:
            md += "## Plots\n\n"
            md += "*Note: Plots are available in HTML and other supported formats.*\n\n"
            
        return md
    
    def to_html(self) -> str:
        """
        Format the report as HTML.
        
        Returns:
        --------
        str : HTML formatted report
        """
        # Convert markdown to HTML
        try:
            import markdown
            md_content = self.to_markdown()
            html_content = markdown.markdown(md_content)
        except ImportError:
            # Fallback if markdown package is not available
            html_content = self.to_markdown().replace('\n', '<br>').replace('# ', '<h1>').replace('## ', '<h2>')
        
        # Add plots if available and requested
        if self.include_plots and self.plots:
            plot_html = "<h2>Plots</h2>"
            
            for plot_name, plot_fig in self.plots.items():
                if isinstance(plot_fig, plt.Figure):
                    # Convert matplotlib figure to HTML
                    from io import BytesIO
                    import base64
                    
                    buf = BytesIO()
                    plot_fig.savefig(buf, format='png')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    
                    plot_html += f"<h3>{plot_name.replace('_', ' ').title()}</h3>"
                    plot_html += f'<img src="data:image/png;base64,{img_str}" alt="{plot_name}" />'
            
            html_content += plot_html
        
        # Wrap in HTML structure
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{self.title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                h3 {{ color: #2980b9; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        return html
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Format the report as a dictionary.
        
        Returns:
        --------
        dict : Report data as a dictionary
        """
        # Return a copy to avoid modifying the original
        # Remove plots from the dictionary as they are not serializable
        report_dict = {k: v for k, v in self.report_data.items() if k != 'plots'}
        return {
            'title': self.title,
            'data': report_dict
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Format the report as a pandas DataFrame.
        
        Returns:
        --------
        pandas.DataFrame : Report data as a DataFrame
        """
        # Flatten nested dictionary
        flat_data = {}
        
        def flatten_dict(d, prefix=''):
            for k, v in d.items():
                if isinstance(v, dict):
                    flatten_dict(v, prefix + k + '_')
                elif not isinstance(v, (list, plt.Figure)):
                    flat_data[prefix + k] = v
        
        # Flatten data excluding plots
        report_data_no_plots = {k: v for k, v in self.report_data.items() if k != 'plots'}
        flatten_dict(report_data_no_plots)
        
        # Convert to DataFrame
        return pd.DataFrame([flat_data])


def format_report(
    report_data: Dict[str, Any],
    output_format: str = 'markdown',
    title: str = "Validation Report",
    include_plots: bool = True
) -> Union[str, Dict[str, Any], pd.DataFrame]:
    """
    Format a validation report in the specified format.
    
    Parameters:
    -----------
    report_data : dict
        Validation results data
    output_format : str
        Output format: 'markdown', 'html', 'dict', or 'dataframe'
    title : str
        Report title
    include_plots : bool
        Whether to include plots in the report
        
    Returns:
    --------
    str, dict, or DataFrame : Formatted report
    """
    formatter = ReportFormatter(report_data, title, include_plots)
    
    if output_format == 'markdown':
        return formatter.to_markdown()
    elif output_format == 'html':
        return formatter.to_html()
    elif output_format == 'dict':
        return formatter.to_dict()
    elif output_format == 'dataframe':
        return formatter.to_dataframe()
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def export_report(
    report_data: Dict[str, Any],
    output_file: str,
    title: str = "Validation Report",
    include_plots: bool = True
) -> None:
    """
    Export a validation report to a file.
    
    Parameters:
    -----------
    report_data : dict
        Validation results data
    output_file : str
        Output file path (format determined by extension)
    title : str
        Report title
    include_plots : bool
        Whether to include plots in the report
    """
    # Determine format from file extension
    if output_file.endswith('.md'):
        output_format = 'markdown'
    elif output_file.endswith('.html'):
        output_format = 'html'
    elif output_file.endswith('.json'):
        output_format = 'dict'
    elif output_file.endswith('.csv'):
        output_format = 'dataframe'
    else:
        raise ValueError(f"Unsupported file extension: {output_file}")
    
    # Format report
    formatted_report = format_report(report_data, output_format, title, include_plots)
    
    # Write to file
    if output_format == 'markdown' or output_format == 'html':
        with open(output_file, 'w') as f:
            f.write(formatted_report)
    elif output_format == 'dict':
        with open(output_file, 'w') as f:
            json.dump(formatted_report, f, indent=2)
    elif output_format == 'dataframe':
        formatted_report.to_csv(output_file, index=False)