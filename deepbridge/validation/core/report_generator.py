"""
Report generation utility for validation results.

This module provides a flexible report generator that can create
formatted reports from validation results in various formats.
"""

from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
import json
import datetime


class ReportGenerator:
    """
    Generate formatted reports from validation results.
    
    This class provides methods to create reports in various formats
    (markdown, JSON, DataFrame, etc.) from validation results.
    """
    
    def __init__(self, validation_results: Dict, title: str = "Validation Report"):
        """
        Initialize the report generator.
        
        Parameters:
        -----------
        validation_results : dict
            Results from validation
        title : str
            Title for the report
        """
        self.results = validation_results
        self.title = title
        self.timestamp = datetime.datetime.now()
    
    def to_markdown(self, include_timestamp: bool = True) -> str:
        """
        Generate a markdown-formatted report.
        
        Parameters:
        -----------
        include_timestamp : bool
            Whether to include a timestamp in the report
            
        Returns:
        --------
        str : Markdown-formatted report
        """
        md = f"# {self.title}\n\n"
        
        if include_timestamp:
            md += f"Generated on: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Add summary section if available
        if 'summary' in self.results:
            md += "## Summary\n\n"
            
            summary = self.results['summary']
            if isinstance(summary, dict):
                for key, value in summary.items():
                    # Format the key for better readability
                    formatted_key = key.replace('_', ' ').title()
                    
                    # Format the value based on type
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    elif isinstance(value, dict):
                        formatted_value = json.dumps(value, indent=2)
                    else:
                        formatted_value = str(value)
                        
                    md += f"**{formatted_key}**: {formatted_value}\n\n"
            else:
                md += f"{summary}\n\n"
        
        # Add detailed results sections
        for section_name, section_content in self.results.items():
            if section_name == 'summary':
                continue  # Already handled above
                
            # Format the section name for better readability
            formatted_section = section_name.replace('_', ' ').title()
            md += f"## {formatted_section}\n\n"
            
            if isinstance(section_content, dict):
                md += self._dict_to_markdown(section_content)
            elif isinstance(section_content, list):
                md += self._list_to_markdown(section_content)
            else:
                md += f"{section_content}\n\n"
        
        return md
    
    def _dict_to_markdown(self, d: Dict, level: int = 0) -> str:
        """
        Convert a dictionary to markdown format.
        
        Parameters:
        -----------
        d : dict
            Dictionary to convert
        level : int
            Current nesting level
            
        Returns:
        --------
        str : Markdown representation
        """
        md = ""
        indent = "  " * level
        
        for key, value in d.items():
            # Format the key for better readability
            formatted_key = key.replace('_', ' ').title()
            
            if isinstance(value, dict):
                # Recursive case for nested dict
                md += f"{indent}### {formatted_key}\n\n"
                md += self._dict_to_markdown(value, level + 1)
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # Handle list of dicts as a table
                md += f"{indent}### {formatted_key}\n\n"
                md += self._list_to_markdown(value, level + 1)
            elif isinstance(value, list):
                # Format as bullet list
                md += f"{indent}### {formatted_key}\n\n"
                for item in value:
                    md += f"{indent}- {item}\n"
                md += "\n"
            elif isinstance(value, float):
                # Format floats with precision
                md += f"{indent}**{formatted_key}**: {value:.4f}\n\n"
            else:
                md += f"{indent}**{formatted_key}**: {value}\n\n"
        
        return md
    
    def _list_to_markdown(self, lst: List, level: int = 0) -> str:
        """
        Convert a list to markdown format.
        
        Parameters:
        -----------
        lst : list
            List to convert
        level : int
            Current nesting level
            
        Returns:
        --------
        str : Markdown representation
        """
        md = ""
        indent = "  " * level
        
        if not lst:
            return f"{indent}*No items*\n\n"
            
        if isinstance(lst[0], dict):
            # Try to create a table
            keys = lst[0].keys()
            
            # Create table header
            md += f"{indent}| " + " | ".join(k.replace('_', ' ').title() for k in keys) + " |\n"
            md += f"{indent}| " + " | ".join(['---'] * len(keys)) + " |\n"
            
            # Add rows
            for item in lst:
                row = []
                for key in keys:
                    value = item.get(key, "")
                    if isinstance(value, float):
                        value = f"{value:.4f}"
                    row.append(str(value))
                md += f"{indent}| " + " | ".join(row) + " |\n"
            
            md += "\n"
        else:
            # Simple bullet list
            for item in lst:
                md += f"{indent}- {item}\n"
            md += "\n"
        
        return md
    
    def to_dict(self) -> Dict:
        """
        Generate a dictionary report.
        
        Returns:
        --------
        dict : Report as a dictionary
        """
        # Create a copy to avoid modifying the original
        return {
            'title': self.title,
            'timestamp': self.timestamp.isoformat(),
            'results': self.results
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Generate a DataFrame report.
        
        This flattens the results into a single row DataFrame.
        
        Returns:
        --------
        pd.DataFrame : Report as a DataFrame
        """
        # Flatten the nested dictionary
        flat_dict = self._flatten_dict(self.results)
        
        # Convert to DataFrame
        return pd.DataFrame([flat_dict])
    
    def _flatten_dict(self, d: Dict, parent_key: str = '') -> Dict:
        """
        Flatten a nested dictionary.
        
        Parameters:
        -----------
        d : dict
            Dictionary to flatten
        parent_key : str
            Parent key for prefixing
            
        Returns:
        --------
        dict : Flattened dictionary
        """
        flat_dict = {}
        
        for key, value in d.items():
            new_key = f"{parent_key}_{key}" if parent_key else key
            
            if isinstance(value, dict):
                # Recursively flatten nested dictionaries
                flat_dict.update(self._flatten_dict(value, new_key))
            elif isinstance(value, list) and value and isinstance(value[0], dict):
                # For list of dicts, try to summarize
                flat_dict[f"{new_key}_count"] = len(value)
            elif isinstance(value, list):
                # For simple lists, join as string
                if all(isinstance(item, (str, int, float, bool)) for item in value):
                    flat_dict[new_key] = ', '.join(str(item) for item in value)
                else:
                    flat_dict[f"{new_key}_count"] = len(value)
            else:
                # Simple value
                flat_dict[new_key] = value
        
        return flat_dict
    
    def to_json(self, indent: int = 2) -> str:
        """
        Generate a JSON report.
        
        Parameters:
        -----------
        indent : int
            Indentation level for JSON formatting
            
        Returns:
        --------
        str : JSON-formatted report
        """
        # Create a serializable dictionary
        report_dict = self.to_dict()
        
        # Convert to JSON
        return json.dumps(report_dict, indent=indent, default=str)
    
    def to_html(self) -> str:
        """
        Generate an HTML report.
        
        Returns:
        --------
        str : HTML-formatted report
        """
        # Convert markdown to HTML
        markdown_report = self.to_markdown()
        
        # Simple conversion - in a real implementation, you'd want to use
        # a proper markdown-to-HTML converter like markdown2 or mistune
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
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
            </style>
        </head>
        <body>
            <pre>{markdown_report}</pre>
        </body>
        </html>
        """
        
        return html