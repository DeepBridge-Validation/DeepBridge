"""
Base interface for report generation components.
"""

import typing as t
import os
from abc import ABC, abstractmethod
from pathlib import Path

class BaseReporter(ABC):
    """
    Abstract base class for all reporting components.
    Defines the common interface that all reporters should implement.
    """
    
    def __init__(self, template_dir: t.Optional[str] = None):
        """
        Initialize the base reporter.
        
        Args:
            template_dir: Directory containing report templates
        """
        self.template_dir = template_dir
        if template_dir is None:
            # Try to find default template directory
            module_dir = os.path.dirname(os.path.abspath(__file__))
            default_template_dir = os.path.join(module_dir, '..', 'reports', 'templates')
            if os.path.exists(default_template_dir):
                self.template_dir = default_template_dir
    
    @abstractmethod
    def generate_report(self, data: t.Dict[str, t.Any], **kwargs) -> str:
        """
        Generate report content from data.
        
        Args:
            data: Data to include in the report
            **kwargs: Additional report parameters
            
        Returns:
            str: Report content (HTML, Markdown, etc.)
        """
        pass
    
    @abstractmethod
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
            
        Returns:
            str: Path to the saved report
        """
        # Ensure directory exists
        directory = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(directory, exist_ok=True)
        
        # Generate report content
        content = self.generate_report(data, **kwargs)
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        return str(output_path)
    
    def _load_template(self, template_name: str) -> t.Optional[str]:
        """
        Load a template file from the template directory.
        
        Args:
            template_name: Name of the template file
            
        Returns:
            str: Template content or None if not found
        """
        if not self.template_dir:
            return None
            
        template_path = os.path.join(self.template_dir, template_name)
        if not os.path.exists(template_path):
            return None
            
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _format_value(self, value: t.Any) -> str:
        """
        Format a value for display in a report.
        
        Args:
            value: Value to format
            
        Returns:
            str: Formatted value
        """
        if isinstance(value, (int, float)):
            # Format numbers with appropriate precision
            if isinstance(value, int):
                return str(value)
            elif abs(value) < 0.001 or abs(value) >= 1000:
                return f"{value:.3e}"
            else:
                return f"{value:.4f}"
        elif isinstance(value, bool):
            return "Yes" if value else "No"
        elif value is None:
            return "N/A"
        else:
            return str(value)