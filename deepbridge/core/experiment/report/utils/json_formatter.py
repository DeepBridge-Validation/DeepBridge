"""
JSON formatter utility for report data.
Ensures proper JSON formatting for JavaScript consumption.
"""

import re
import json
import logging
from typing import Any, Dict, Union, List

try:
    import numpy as np
except ImportError:
    # Create a minimal np substitute if numpy is not available
    class NumpySubstitute:
        @staticmethod
        def isnan(x):
            return x != x
            
        @staticmethod
        def isinf(x):
            return x == float('inf') or x == float('-inf')
    
    np = NumpySubstitute()

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class JsonFormatter:
    """
    Utility class for formatting JSON data to ensure it's valid
    for JavaScript consumption in reports.
    """
    
    @staticmethod
    def sanitize_json_string(json_str: str) -> str:
        """
        Sanitize a JSON string to remove trailing commas and fix common syntax issues.
        
        Parameters:
        -----------
        json_str : str
            JSON string to sanitize
            
        Returns:
        --------
        str : Sanitized JSON string
        """
        if not json_str:
            return "{}"
        
        # Safety check to prevent errors with None
        if json_str is None:
            return "{}"
            
        # First, direct string replacements for quick fixes
        json_str = json_str.replace(',}', '}')
        json_str = json_str.replace(',]', ']')
        json_str = json_str.replace(',)', ')')
        
        # Fix trailing commas in objects - more robust with regex
        # Pattern: { ..., } - matches an object that ends with a comma before the closing brace
        json_str = re.sub(r',(\s*})', r'\1', json_str)
        
        # Fix trailing commas in arrays
        # Pattern: [ ..., ] - matches an array that ends with a comma before the closing bracket
        json_str = re.sub(r',(\s*\])', r'\1', json_str)
        
        # Fix trailing commas in function parameters
        # Pattern: ( ..., ) - matches parameters that end with a comma before the closing parenthesis
        json_str = re.sub(r',(\s*\))', r'\1', json_str)
        
        # Replace NaN, Infinity and -Infinity with null (which is valid JSON)
        json_str = re.sub(r'\bNaN\b', 'null', json_str)
        json_str = re.sub(r'\bInfinity\b', 'null', json_str)
        json_str = re.sub(r'\b-Infinity\b', 'null', json_str)
        json_str = re.sub(r'\bundefined\b', 'null', json_str)
        
        # Fix unquoted property names (JavaScript allows this but JSON doesn't)
        json_str = re.sub(r'([{,]\s*)([a-zA-Z0-9_$]+)(\s*:)', r'\1"\2"\3', json_str)
        
        # Ensure we don't have trailing commas in nested structures
        # This is a more complex fix for nested objects with trailing commas
        while True:
            # Keep applying the fix until no changes are made
            new_str = re.sub(r',(\s*[}\])])', r'\1', json_str)
            if new_str == json_str:
                break
            json_str = new_str
            
        return json_str
    
    @staticmethod
    def format_for_javascript(data: Any) -> str:
        """
        Format data as JSON that's valid for JavaScript.
        Converts Python data structure to JSON string, then sanitizes it.
        
        Parameters:
        -----------
        data : Any
            Data to format as JSON
            
        Returns:
        --------
        str : Sanitized JSON string
        """
        try:
            # First, cleanup numeric values that might cause JSON issues
            if isinstance(data, dict):
                # Process dictionary values recursively
                for key in list(data.keys()):
                    # Skip null/None values
                    if data[key] is None:
                        continue
                        
                    # Handle special numeric values in nested dictionaries
                    if isinstance(data[key], dict):
                        data[key] = JsonFormatter._clean_dict_values(data[key])
                    
                    # Handle special numeric values in nested lists
                    elif isinstance(data[key], list):
                        data[key] = JsonFormatter._clean_list_values(data[key])
                    
                    # Handle special numeric values directly
                    elif isinstance(data[key], float):
                        if np.isnan(data[key]) or np.isinf(data[key]):
                            data[key] = None
            
            # Convert to regular JSON
            json_str = json.dumps(data, default=str)
            
            # Then sanitize the string
            return JsonFormatter.sanitize_json_string(json_str)
        except Exception as e:
            logger.error(f"Error formatting JSON for JavaScript: {str(e)}")
            # Return a safe fallback
            return "{}"
    
    @staticmethod
    def _clean_dict_values(data_dict: Dict) -> Dict:
        """
        Recursively clean dictionary values to handle special numeric values.
        """
        if not isinstance(data_dict, dict):
            return data_dict
            
        for key in list(data_dict.keys()):
            # Skip null/None values
            if data_dict[key] is None:
                continue
                
            # Handle special numeric values in nested dictionaries
            if isinstance(data_dict[key], dict):
                data_dict[key] = JsonFormatter._clean_dict_values(data_dict[key])
            
            # Handle special numeric values in nested lists
            elif isinstance(data_dict[key], list):
                data_dict[key] = JsonFormatter._clean_list_values(data_dict[key])
            
            # Handle special numeric values directly
            elif isinstance(data_dict[key], float):
                try:
                    if np.isnan(data_dict[key]) or np.isinf(data_dict[key]):
                        data_dict[key] = None
                except:
                    # If numpy is not available or other error, use standard checks
                    if data_dict[key] != data_dict[key] or data_dict[key] == float('inf') or data_dict[key] == float('-inf'):
                        data_dict[key] = None
        
        return data_dict
    
    @staticmethod
    def _clean_list_values(data_list: List) -> List:
        """
        Recursively clean list values to handle special numeric values.
        """
        if not isinstance(data_list, list):
            return data_list
            
        for i in range(len(data_list)):
            # Skip null/None values
            if data_list[i] is None:
                continue
                
            # Handle special numeric values in nested dictionaries
            if isinstance(data_list[i], dict):
                data_list[i] = JsonFormatter._clean_dict_values(data_list[i])
            
            # Handle special numeric values in nested lists
            elif isinstance(data_list[i], list):
                data_list[i] = JsonFormatter._clean_list_values(data_list[i])
            
            # Handle special numeric values directly
            elif isinstance(data_list[i], float):
                try:
                    if np.isnan(data_list[i]) or np.isinf(data_list[i]):
                        data_list[i] = None
                except:
                    # If numpy is not available or other error, use standard checks
                    if data_list[i] != data_list[i] or data_list[i] == float('inf') or data_list[i] == float('-inf'):
                        data_list[i] = None
        
        return data_list
    
    @staticmethod
    def validate_and_format(data: Union[Dict, List, str]) -> str:
        """
        Validate and format data for JavaScript.
        
        Parameters:
        -----------
        data : Union[Dict, List, str]
            Data to format as JSON. Can be a dictionary, list, or JSON string
            
        Returns:
        --------
        str : Validated and sanitized JSON string
        """
        try:
            # If it's already a string, parse it first to validate
            if isinstance(data, str):
                try:
                    # Try to parse as JSON first
                    parsed_data = json.loads(data)
                    # Then re-format properly
                    return JsonFormatter.format_for_javascript(parsed_data)
                except json.JSONDecodeError:
                    # If it's not valid JSON, try to sanitize directly
                    sanitized = JsonFormatter.sanitize_json_string(data)
                    # Validate the sanitized version
                    try:
                        json.loads(sanitized)
                        return sanitized
                    except json.JSONDecodeError:
                        logger.error("Could not sanitize invalid JSON string")
                        return "{}"
            else:
                # For dicts and lists, convert to JSON string
                return JsonFormatter.format_for_javascript(data)
        except Exception as e:
            logger.error(f"Error validating JSON: {str(e)}")
            return "{}"
    
    @staticmethod
    def embed_in_html(data: Any, variable_name: str = "reportData") -> str:
        """
        Format data as JSON and embed in HTML script tag.
        
        Parameters:
        -----------
        data : Any
            Data to format as JSON
        variable_name : str, optional
            JavaScript variable name to assign the data to
            
        Returns:
        --------
        str : HTML script tag containing the JSON data
        """
        json_str = JsonFormatter.format_for_javascript(data)
        
        # Create script tag with the formatted JSON
        script = f"""
<script type="text/javascript">
    // JSON data for report, sanitized to prevent syntax errors
    const {variable_name} = {json_str};
</script>
"""
        return script