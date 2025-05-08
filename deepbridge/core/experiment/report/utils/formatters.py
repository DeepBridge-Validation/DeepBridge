"""
Formatters for report data display.
"""

import logging
from typing import Any, Dict, List, Union, Optional

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class DataFormatter:
    """
    Formats data for display in reports.
    """
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 2) -> str:
        """
        Format a value as a percentage.
        
        Parameters:
        -----------
        value : float
            Value to format (0.0 to 1.0)
        decimal_places : int, optional
            Number of decimal places to display
            
        Returns:
        --------
        str : Formatted percentage
        """
        if value is None:
            return "N/A"
        
        try:
            formatted = f"{value * 100:.{decimal_places}f}%"
            return formatted
        except (ValueError, TypeError):
            logger.warning(f"Error formatting percentage value: {value}")
            return "N/A"
    
    @staticmethod
    def format_number(value: float, decimal_places: int = 3) -> str:
        """
        Format a numeric value.
        
        Parameters:
        -----------
        value : float
            Value to format
        decimal_places : int, optional
            Number of decimal places to display
            
        Returns:
        --------
        str : Formatted number
        """
        if value is None:
            return "N/A"
        
        try:
            formatted = f"{value:.{decimal_places}f}"
            return formatted
        except (ValueError, TypeError):
            logger.warning(f"Error formatting numeric value: {value}")
            return "N/A"
    
    @staticmethod
    def format_list(values: List[Any], separator: str = ", ") -> str:
        """
        Format a list of values as a string.
        
        Parameters:
        -----------
        values : List[Any]
            List of values to format
        separator : str, optional
            Separator between values
            
        Returns:
        --------
        str : Formatted list
        """
        if not values:
            return ""
        
        try:
            return separator.join(str(v) for v in values)
        except Exception as e:
            logger.warning(f"Error formatting list: {str(e)}")
            return str(values)
    
    @staticmethod
    def format_object_list(objects: List[Dict[str, Any]], key: str, separator: str = ", ") -> str:
        """
        Format a list of objects by extracting a specific key.
        
        Parameters:
        -----------
        objects : List[Dict[str, Any]]
            List of objects to format
        key : str
            Key to extract from each object
        separator : str, optional
            Separator between values
            
        Returns:
        --------
        str : Formatted list of values
        """
        if not objects:
            return ""
        
        try:
            values = [str(obj.get(key, "")) for obj in objects]
            return separator.join(v for v in values if v)
        except Exception as e:
            logger.warning(f"Error formatting object list: {str(e)}")
            return ""