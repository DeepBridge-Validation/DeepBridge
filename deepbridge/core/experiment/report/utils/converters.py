"""
Converters for report data types.
"""

import datetime
import logging
import math
from typing import Any, Dict, List, Union, Optional

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class DataTypeConverter:
    """
    Converts various data types for report generation.
    """
    
    def __init__(self):
        """Initialize the converter."""
        # Import numpy if available
        try:
            import numpy as np
            self.np = np
        except ImportError:
            self.np = None
            logger.warning("NumPy not available. NumPy type conversion disabled.")
    
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
        elif np is not None:
            # Integer types
            if hasattr(np, 'integer') and isinstance(data, np.integer):
                return int(data)
            elif any(hasattr(np, t) and isinstance(data, getattr(np, t)) 
                    for t in ['int8', 'int16', 'int32', 'int64', 'intc', 'intp']):
                return int(data)
            
            # Float types
            if hasattr(np, 'floating') and isinstance(data, np.floating):
                # Handle NaN and Inf values
                if np.isnan(data) or np.isinf(data):
                    return None
                return float(data)
            elif any(hasattr(np, t) and isinstance(data, getattr(np, t)) 
                    for t in ['float16', 'float32', 'float64']):
                if np.isnan(data) or np.isinf(data):
                    return None
                return float(data)
            
            # Array types
            elif isinstance(data, np.ndarray):
                # Convert array to list, handling NaN/Inf values
                result = data.tolist()
                if np.issubdtype(data.dtype, np.floating):
                    if isinstance(result, list):
                        return [None if (isinstance(x, float) and (np.isnan(x) or np.isinf(x))) 
                                else self.convert_numpy_types(x) for x in result]
                return result
        
        # Handle other types
        if isinstance(data, float) and (math.isnan(data) or math.isinf(data)):
            return None
        
        return data

    def json_serializer(self, obj: Any) -> Any:
        """
        JSON serializer for objects not serializable by default json code.
        
        Parameters:
        -----------
        obj : Any
            Object to serialize
            
        Returns:
        --------
        Any : Serialized object
            
        Raises:
        -------
        TypeError: If object cannot be serialized
        """
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        raise TypeError(f"Type {type(obj)} not serializable")