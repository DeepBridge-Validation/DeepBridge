"""
Standard implementation of the BaseReporter interface.
"""

import typing as t
import os
import json
import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import jinja2

from .base_reporter import BaseReporter

class StandardReporter(BaseReporter):
    """
    Standard implementation of the BaseReporter interface.
    Provides HTML report generation capabilities using templates.
    """
    
    def __init__(self, template_dir: t.Optional[str] = None, 
                 template_name: t.Optional[str] = None):
        """
        Initialize the standard reporter.
        
        Args:
            template_dir: Directory containing report templates
            template_name: Default template name to use
        """
        super().__init__(template_dir)
        self.template_name = template_name
        
        # Initialize Jinja environment
        self.jinja_env = None
        if self.template_dir and os.path.exists(self.template_dir):
            self.jinja_env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.template_dir),
                autoescape=jinja2.select_autoescape(['html', 'xml'])
            )
    
    def generate_report(self, data: t.Dict[str, t.Any], **kwargs) -> str:
        """
        Generate report content from data.
        
        Args:
            data: Data to include in the report
            **kwargs: Additional report parameters
                - template_name: Override default template
                - report_title: Title for the report
                - timestamp_format: Format for timestamp
                
        Returns:
            str: Report content (HTML)
        """
        # Get template name from kwargs or use default
        template_name = kwargs.get('template_name', self.template_name)
        if not template_name:
            raise ValueError("Template name must be provided")
        
        # If we have a Jinja environment, use it
        if self.jinja_env:
            try:
                # Load the template
                template = self.jinja_env.get_template(template_name)
                
                # Add report generation timestamp
                timestamp_format = kwargs.get('timestamp_format', '%Y-%m-%d %H:%M:%S')
                timestamp = datetime.datetime.now().strftime(timestamp_format)
                
                # Prepare template data (combine data with kwargs)
                template_data = {
                    'generation_date': timestamp,
                    'report_title': kwargs.get('report_title', 'DeepBridge Report'),
                    **data
                }
                
                # Convert any numpy types to native Python types
                template_data = self._convert_numpy_types(template_data)
                
                # Render the template
                return template.render(**template_data)
                
            except Exception as e:
                raise ValueError(f"Error rendering template: {str(e)}")
        
        # Otherwise, use basic template loading
        return self._generate_basic_report(data, **kwargs)
    
    def _generate_basic_report(self, data: t.Dict[str, t.Any], **kwargs) -> str:
        """
        Generate a basic report without Jinja2.
        
        Args:
            data: Data to include in the report
            **kwargs: Additional report parameters
        
        Returns:
            str: HTML report content
        """
        # Load template content
        template_name = kwargs.get('template_name', self.template_name)
        template_content = self._load_template(template_name)
        
        if not template_content:
            raise ValueError(f"Template {template_name} not found")
        
        # Get report title and timestamp
        report_title = kwargs.get('report_title', 'DeepBridge Report')
        timestamp_format = kwargs.get('timestamp_format', '%Y-%m-%d %H:%M:%S')
        timestamp = datetime.datetime.now().strftime(timestamp_format)
        
        # Replace placeholders in template
        content = template_content
        content = content.replace("{{report_title}}", report_title)
        content = content.replace("{{generation_date}}", timestamp)
        
        # Convert data to JSON for embedding in the report
        data_json = json.dumps(self._convert_numpy_types(data), indent=2)
        content = content.replace("{{data_json}}", data_json)
        
        # Insert data blocks as needed
        if "{{data_blocks}}" in content:
            data_blocks_html = self._generate_data_blocks(data)
            content = content.replace("{{data_blocks}}", data_blocks_html)
        
        return content
    
    def _generate_data_blocks(self, data: t.Dict[str, t.Any]) -> str:
        """
        Generate HTML for data blocks.
        
        Args:
            data: Data dictionary
            
        Returns:
            str: HTML content for data blocks
        """
        html = ""
        
        for section_name, section_data in data.items():
            html += f'<div class="section" id="{section_name}">\n'
            html += f'<h2>{section_name.replace("_", " ").title()}</h2>\n'
            
            if isinstance(section_data, dict):
                html += self._dict_to_html(section_data)
            elif isinstance(section_data, list):
                html += self._list_to_html(section_data)
            elif isinstance(section_data, pd.DataFrame):
                html += section_data.to_html(classes="table table-striped", index=False)
            else:
                html += f'<p>{str(section_data)}</p>\n'
            
            html += '</div>\n'
        
        return html
    
    def _dict_to_html(self, data: dict) -> str:
        """Convert dictionary to HTML table."""
        html = '<table class="table table-striped">\n'
        html += '<tr><th>Key</th><th>Value</th></tr>\n'
        
        for key, value in data.items():
            formatted_key = str(key).replace("_", " ").title()
            
            if isinstance(value, dict):
                formatted_value = f'<details><summary>View Details</summary>{self._dict_to_html(value)}</details>'
            elif isinstance(value, list):
                formatted_value = f'<details><summary>View Details</summary>{self._list_to_html(value)}</details>'
            elif isinstance(value, pd.DataFrame):
                formatted_value = f'<details><summary>View Details</summary>{value.to_html(index=False)}</details>'
            else:
                formatted_value = self._format_value(value)
            
            html += f'<tr><td>{formatted_key}</td><td>{formatted_value}</td></tr>\n'
            
        html += '</table>\n'
        return html
    
    def _list_to_html(self, data: list) -> str:
        """Convert list to HTML."""
        if not data:
            return "<p>Empty list</p>"
            
        if isinstance(data[0], dict):
            # Convert list of dicts to a table
            if len(data) > 0:
                keys = data[0].keys()
                html = '<table class="table table-striped">\n'
                html += '<tr>'
                for key in keys:
                    html += f'<th>{str(key).replace("_", " ").title()}</th>'
                html += '</tr>\n'
                
                for item in data:
                    html += '<tr>'
                    for key in keys:
                        value = item.get(key, "")
                        html += f'<td>{self._format_value(value)}</td>'
                    html += '</tr>\n'
                    
                html += '</table>\n'
                return html
        
        # Simple list
        html = '<ul>\n'
        for item in data:
            html += f'<li>{self._format_value(item)}</li>\n'
        html += '</ul>\n'
        return html
    
    def save_report(self, 
                   output_path: t.Union[str, Path], 
                   data: t.Dict[str, t.Any], 
                   **kwargs) -> str:
        """
        Generate and save a report to a file.
        
        Args:
            output_path: Path to save the report file
            data: Data to include in the report
            **kwargs: Additional report parameters
                - template_name: Override default template
                - report_title: Title for the report
                - create_basic_html: If True, create a basic HTML wrapper if template fails
                
        Returns:
            str: Path to the saved report
        """
        # Ensure directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Generate report content
            content = self.generate_report(data, **kwargs)
        except ValueError as e:
            # If template is missing or invalid, fallback to basic HTML if requested
            if kwargs.get('create_basic_html', False):
                report_title = kwargs.get('report_title', 'DeepBridge Report')
                content = self._create_basic_html_report(data, report_title)
            else:
                raise e
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return str(output_path)
        
    def _create_basic_html_report(self, data: t.Dict[str, t.Any], title: str) -> str:
        """
        Create a basic HTML report without a template.
        
        Args:
            data: Report data
            title: Report title
            
        Returns:
            str: HTML report content
        """
        data_blocks = self._generate_data_blocks(data)
        
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .section {{ margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                h1, h2 {{ color: #2c3e50; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            {data_blocks}
        </body>
        </html>
        """
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON compatibility."""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._convert_numpy_types(obj.tolist())
        elif isinstance(obj, pd.DataFrame):
            return self._convert_numpy_types(obj.to_dict(orient='records'))
        elif isinstance(obj, pd.Series):
            return self._convert_numpy_types(obj.to_dict())
        else:
            return obj