"""
Data integration module.
Handles integration with data transformers for report generation.
"""

import json
import logging
from typing import Dict, Any

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class DataIntegrationManager:
    """
    Handles integration with data transformers.
    """
    
    def __init__(self, asset_manager):
        """
        Initialize with reference to parent asset manager.
        
        Parameters:
        -----------
        asset_manager : AssetManager
            Parent asset manager that owns this integration manager
        """
        self.asset_manager = asset_manager
    
    def serialize_data_for_template(self, data: Dict[str, Any]) -> str:
        """
        Serialize data for use in the template as JavaScript object.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Data to serialize
            
        Returns:
        --------
        str : JSON-serialized data as JavaScript object 
        """
        try:
            # Convert to JSON, ensuring proper escaping
            json_str = json.dumps(data, ensure_ascii=False, indent=2)
            
            # Wrap in JavaScript variable declaration
            return f"const reportData = {json_str};"
        except Exception as e:
            logger.error(f"Error serializing data: {str(e)}")
            # Return empty object on error
            return "const reportData = {};"
    
    def prepare_template_context(self, test_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare context for template rendering with all assets.
        
        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
        data : Dict[str, Any]
            Test data to include in the report
            
        Returns:
        --------
        Dict[str, Any] : Context for template rendering
        """
        try:
            # Get all assets
            assets = self.asset_manager.create_full_report_assets(test_type)
            
            # Serialize data for JavaScript
            serialized_data = self.serialize_data_for_template(data)
            
            # Build context
            context = {
                # Report data
                'report_data': data,
                'serialized_data': serialized_data,
                'test_type': test_type,
                
                # Assets
                'css': assets.get('css', '/* No CSS loaded */'),
                'js': assets.get('js', '// No JavaScript loaded'),
                'logo': assets.get('logo', ''),
                'favicon': assets.get('favicon', ''),
                'icons': assets.get('icons', {}),
                
                # Common HTML fragments
                'footer': assets.get('common', {}).get('footer.html', ''),
                'header': assets.get('common', {}).get('header.html', ''),
                'meta': assets.get('common', {}).get('meta.html', ''),
                'navigation': assets.get('common', {}).get('navigation.html', ''),
                
                # Test-specific partials
                'partials': assets.get('test_partials', {})
            }
            
            return context
        except Exception as e:
            logger.error(f"Error preparing template context: {str(e)}")
            raise
    
    def get_transformer_for_test_type(self, test_type: str):
        """
        Get appropriate DataTransformer class for the test type.
        
        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
            
        Returns:
        --------
        DataTransformer : Instance of the appropriate DataTransformer class
        
        Raises:
        -------
        ImportError: If the DataTransformer module cannot be imported
        ValueError: If the test type is not supported
        """
        try:
            # Import the data transformer module
            from .data_transformer import (
                DataTransformer,
                RobustnessDataTransformer,
                UncertaintyDataTransformer,
                ResilienceDataTransformer,
                HyperparameterDataTransformer
            )
            
            # Map test types to transformer classes
            transformers = {
                'robustness': RobustnessDataTransformer,
                'uncertainty': UncertaintyDataTransformer,
                'resilience': ResilienceDataTransformer,
                'hyperparameter': HyperparameterDataTransformer,
                'hyperparameters': HyperparameterDataTransformer
            }
            
            # Get appropriate transformer
            transformer_class = transformers.get(test_type.lower())
            if transformer_class is None:
                raise ValueError(f"Unsupported test type: {test_type}")
                
            # Return instance of the transformer
            return transformer_class()
            
        except ImportError as e:
            logger.error(f"Error importing DataTransformer module: {str(e)}")
            raise ImportError(f"Failed to import DataTransformer module: {str(e)}")
            
    def transform_data(self, test_type: str, data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Transform raw test data using the appropriate transformer.
        
        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
        data : Dict[str, Any]
            Raw test data
        model_name : str
            Name of the model for display
            
        Returns:
        --------
        Dict[str, Any] : Transformed data for template
        
        Raises:
        -------
        ValueError: If the test type is not supported or data transformation fails
        """
        try:
            # Get appropriate transformer
            transformer = self.get_transformer_for_test_type(test_type)
            
            # Transform data
            transformed_data = transformer.transform(data, model_name)
            logger.info(f"Successfully transformed {test_type} data for model: {model_name}")
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming {test_type} data: {str(e)}")
            raise ValueError(f"Failed to transform {test_type} data: {str(e)}")