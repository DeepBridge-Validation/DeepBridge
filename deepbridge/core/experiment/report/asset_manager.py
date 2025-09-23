"""
Asset management module for report generation.
Core functionality for handling assets.
"""

import os
import logging
from typing import Dict, Any, Optional

# Import submodules
from .file_discovery import FileDiscoveryManager
from .asset_processor import AssetProcessor
from .data_integration import DataIntegrationManager

# Configure logger
logger = logging.getLogger("deepbridge.reports")

class AssetManager:
    """
    Manages static assets (CSS, JS, images) for report generation.
    """
    
    def __init__(self, templates_dir: str):
        """
        Initialize the asset manager.
        
        Parameters:
        -----------
        templates_dir : str
            Base directory containing templates and assets
            
        Raises:
        -------
        FileNotFoundError: If templates directory doesn't exist
        """
        if not os.path.exists(templates_dir):
            raise FileNotFoundError(f"Templates directory not found: {templates_dir}")
            
        self.templates_dir = templates_dir
        
        # Set up paths for assets directories
        self.assets_dir = os.path.join(self.templates_dir, 'assets')
        self.common_dir = os.path.join(self.templates_dir, 'common')

        # Use templates/assets/images folder for images (logo and favicon)
        self.images_dir = os.path.join(self.templates_dir, 'assets', 'images')

        # Set up paths for favicon and logo
        self.favicon_path = os.path.join(self.images_dir, 'favicon.ico')
        self.logo_path = os.path.join(self.images_dir, 'logo.png')
        
        # Initialize sub-managers
        self.file_manager = FileDiscoveryManager(self)
        self.asset_processor = AssetProcessor(self)
        self.data_manager = DataIntegrationManager(self)
        
        # Validate core directories exist
        self._validate_core_directories()
    
    def _validate_core_directories(self):
        """
        Validate that core directories exist.
        
        Raises:
        -------
        FileNotFoundError: If any core directory doesn't exist
        """
        core_dirs = {
            'assets': self.assets_dir,
            'common': self.common_dir
        }
        
        for name, path in core_dirs.items():
            if not os.path.exists(path):
                logger.warning(f"{name} directory not found: {path}")
    
    # File discovery methods
    def find_css_path(self, test_type: str) -> str:
        """Find CSS directory for the specified test type."""
        return self.file_manager.find_css_path(test_type)
    
    def find_js_path(self, test_type: str) -> str:
        """Find JavaScript directory for the specified test type."""
        return self.file_manager.find_js_path(test_type)
    
    def get_asset_path(self, report_type: str, asset_path: str) -> str:
        """
        Get the full path to an asset file.

        Parameters:
        -----------
        report_type : str
            The type of report (e.g., 'distillation', 'uncertainty', etc.)
        asset_path : str
            The relative path to the asset within the report type directory

        Returns:
        --------
        str
            The full path to the asset file
        """
        # Try to find in report-specific assets first
        report_dir = os.path.join(self.templates_dir, 'report_types', report_type)
        specific_path = os.path.join(report_dir, asset_path)

        if os.path.exists(specific_path):
            return specific_path

        # Fall back to common assets
        common_path = os.path.join(self.assets_dir, asset_path)
        if os.path.exists(common_path):
            return common_path

        # If not found, return the expected path anyway
        return specific_path

    def get_generic_css_path(self) -> str:
        """Get path to generic CSS assets."""
        return self.file_manager.get_generic_css_path()
    
    def get_generic_js_path(self) -> str:
        """Get path to generic JavaScript assets."""
        return self.file_manager.get_generic_js_path()
    
    def get_common_html_files(self) -> Dict[str, str]:
        """Get contents of common HTML template files."""
        return self.file_manager.get_common_html_files()
    
    def get_test_partials(self, test_type: str) -> Dict[str, str]:
        """Get HTML partial templates for a specific test type."""
        return self.file_manager.get_test_partials(test_type)
    
    # Asset processing methods
    def get_css_content(self, css_dir: str, test_type: str = None, files: Optional[Dict[str, str]] = None) -> str:
        """Combine CSS files into a single string."""
        return self.asset_processor.get_css_content(css_dir, test_type, files)
    
    def get_generic_css_content(self) -> str:
        """Combine generic CSS files from assets/css directory."""
        return self.asset_processor.get_generic_css_content()
    
    def get_combined_css_content(self, test_type: str) -> str:
        """Combine generic CSS files with test-specific CSS files."""
        return self.asset_processor.get_combined_css_content(test_type)
    
    def get_js_content(self, js_dir: str, files: Optional[Dict[str, str]] = None) -> str:
        """Combine JavaScript files into a single string."""
        return self.asset_processor.get_js_content(js_dir, files)
    
    def get_generic_js_content(self) -> str:
        """Combine generic JavaScript files from assets/js directory."""
        return self.asset_processor.get_generic_js_content()
    
    def get_combined_js_content(self, test_type: str) -> str:
        """Combine generic JavaScript files with test-specific JavaScript files."""
        return self.asset_processor.get_combined_js_content(test_type)
    
    def get_base64_image(self, image_path: str) -> str:
        """Convert image to base64 string."""
        return self.asset_processor.get_base64_image(image_path)
    
    def get_logo_base64(self) -> str:
        """Get base64 encoded logo."""
        return self.asset_processor.get_logo_base64()
    
    def get_favicon_base64(self) -> str:
        """Get base64 encoded favicon."""
        return self.asset_processor.get_favicon_base64()
    
    def get_icons(self) -> Dict[str, str]:
        """Get base64 encoded icons from the icons directory."""
        return self.asset_processor.get_icons()
    
    def create_full_report_assets(self, test_type: str) -> Dict[str, Any]:
        """Create a complete set of assets for a report."""
        return self.asset_processor.create_full_report_assets(test_type)
    
    # Data integration methods
    def serialize_data_for_template(self, data: Dict[str, Any]) -> str:
        """Serialize data for use in the template as JavaScript object."""
        return self.data_manager.serialize_data_for_template(data)
    
    def prepare_template_context(self, test_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for template rendering with all assets."""
        return self.data_manager.prepare_template_context(test_type, data)
    
    def get_transformer_for_test_type(self, test_type: str):
        """Get appropriate DataTransformer class for the test type."""
        return self.data_manager.get_transformer_for_test_type(test_type)
    
    def transform_data(self, test_type: str, data: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """Transform raw test data using the appropriate transformer."""
        return self.data_manager.transform_data(test_type, data, model_name)
    
    # File operations
    def write_json_data_file(self, data: Dict[str, Any], output_dir: str, filename: str = "dados_transformados.json") -> str:
        """Write data to a JSON file in the output directory."""
        return self.file_manager.write_json_data_file(data, output_dir, filename)
    
    def copy_assets_to_output(self, test_type: str, output_dir: str) -> Dict[str, str]:
        """Copy required assets to the output directory."""
        return self.file_manager.copy_assets_to_output(test_type, output_dir)
    
    def _copy_file(self, src_path: str, dest_path: str) -> None:
        """Copy a file from source to destination."""
        return self.file_manager._copy_file(src_path, dest_path)