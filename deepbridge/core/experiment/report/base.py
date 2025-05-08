"""
Base data transformation module for report generation.
Transforms experiment results data into format suitable for templates.
"""

import logging
import datetime
import copy
from typing import Dict, Any, Optional

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class DataTransformer:
    """
    Base class for transforming experiment results data for templates.
    """
    
    def __init__(self):
        """Initialize the data transformer."""
        # Import numpy if available for handling numpy types
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.np = None
            logger.warning("NumPy not available. NumPy type conversion disabled.")
    
    def transform(self, results: Dict[str, Any], model_name: str, timestamp: Optional[str] = None) -> Dict[str, Any]:
        """
        Transform results data for template rendering.
        
        Parameters:
        -----------
        results : Dict[str, Any]
            Raw experiment results
        model_name : str
            Name of the model
        timestamp : str, optional
            Timestamp for the report
            
        Returns:
        --------
        Dict[str, Any] : Transformed data for templates
        """
        # Base implementation should be overridden by subclasses
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Create a deep copy to avoid modifying the original
        report_data = self._deep_copy(results)
        
        # Add metadata
        report_data['model_name'] = report_data.get('model_name', model_name)
        report_data['timestamp'] = report_data.get('timestamp', timestamp)
        
        # Convert numpy types
        report_data = self.convert_numpy_types(report_data)
        
        return report_data
    
    def _deep_copy(self, obj: Any) -> Any:
        """
        Create a deep copy of an object, handling special types.
        
        Parameters:
        -----------
        obj : Any
            Object to copy
            
        Returns:
        --------
        Any : Deep copy of the object
        """
        try:
            return copy.deepcopy(obj)
        except Exception as e:
            logger.warning(f"Error in deep copy: {str(e)}, falling back to manual copy")
            
            # Manual copy for problematic objects
            if isinstance(obj, dict):
                return {k: self._deep_copy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._deep_copy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(self._deep_copy(item) for item in obj)
            elif isinstance(obj, set):
                return {self._deep_copy(item) for item in obj}
            elif self.np is not None and isinstance(obj, self.np.ndarray):
                return obj.tolist()  # Convert numpy arrays to lists
            else:
                return obj  # Return the object as is if it can't be copied
    
    def convert_numpy_types(self, data: Any) -> Any:
        """
        Convert numpy types to Python native types for JSON serialization.
        
        Parameters:
        -----------
        data : Any
            Data that may contain numpy types
            
        Returns:
        --------
        Any : Data with numpy types converted to Python native types
        """
        np = self.np
        if np is None:
            return data
            
        if isinstance(data, dict):
            return {k: self.convert_numpy_types(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.convert_numpy_types(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.convert_numpy_types(item) for item in data)
        elif isinstance(data, (datetime.datetime, datetime.date)):
            return data.isoformat()
        elif hasattr(np, 'integer') and isinstance(data, np.integer):
            return int(data)
        elif hasattr(np, 'floating') and isinstance(data, np.floating):
            # Handle NaN and Inf values
            if np.isnan(data) or np.isinf(data):
                return None
            return float(data)
        elif isinstance(data, np.ndarray):
            # Convert array to list, handling NaN/Inf values
            result = data.tolist()
            if np.issubdtype(data.dtype, np.floating):
                if isinstance(result, list):
                    return [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) 
                            else self.convert_numpy_types(x) for x in result]
            return result
        elif isinstance(data, float) and (np is not None) and (np.isnan(data) or np.isinf(data)):
            return None
        else:
            return data