"""
File discovery module for asset management.
Discovers and loads files from the templates directory.

**DEPRECATED (Phase 2):** This module is deprecated and will be removed in a future version.
Use `deepbridge.core.experiment.report.utils.file_utils` instead.

Migration:
    Old: asset_manager.file_manager._discover_css_files(css_dir)
    New: file_utils.find_css_files(css_dir)
"""

import os
import json
import logging
import warnings
from typing import Dict, Any, Optional, List

# Configure logger
logger = logging.getLogger("deepbridge.reports")

# Deprecation warning
warnings.warn(
    "FileDiscoveryManager is deprecated and will be removed in a future version. "
    "Use deepbridge.core.experiment.report.utils.file_utils instead.",
    DeprecationWarning,
    stacklevel=2
)

class FileDiscoveryManager:
    """
    Manages discovery and loading of files for asset management.
    """
    
    def __init__(self, asset_manager):
        """
        Initialize with reference to parent asset manager.
        
        Parameters:
        -----------
        asset_manager : AssetManager
            Parent asset manager that owns this discovery manager
        """
        self.asset_manager = asset_manager
    
    def find_css_path(self, test_type: str, report_type: str = None) -> str:
        """
        Find CSS directory for the specified test type.

        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
        report_type : str, optional
            Report type ('static' or 'interactive')

        Returns:
        --------
        str : Path to CSS directory

        Raises:
        -------
        FileNotFoundError: If CSS directory is not found
        """
        css_paths = [
            # Direct CSS path (e.g., robustness/css)
            os.path.join(self.asset_manager.templates_dir, 'report_types', test_type, 'css'),
            # Interactive report CSS path (e.g., uncertainty/interactive/css)
            os.path.join(self.asset_manager.templates_dir, 'report_types', test_type, 'interactive', 'css'),
            # Static report CSS path (e.g., uncertainty/static/css)
            os.path.join(self.asset_manager.templates_dir, 'report_types', test_type, 'static', 'css'),
        ]

        # If report_type is specified, prioritize that path
        if report_type:
            report_type_lower = report_type.lower()
            if report_type_lower in ['interactive', 'static']:
                priority_path = os.path.join(self.asset_manager.templates_dir, 'report_types', test_type, report_type_lower, 'css')
                if os.path.exists(priority_path):
                    logger.info(f"Using CSS path for {report_type_lower} report: {priority_path}")
                    return priority_path

        for path in css_paths:
            if os.path.exists(path):
                logger.info(f"Using CSS path: {path}")
                return path

        error_msg = f"CSS directory not found for {test_type}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    def find_js_path(self, test_type: str, report_type: str = None) -> str:
        """
        Find JavaScript directory for the specified test type.

        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
        report_type : str, optional
            Report type ('static' or 'interactive')

        Returns:
        --------
        str : Path to JavaScript directory

        Raises:
        -------
        FileNotFoundError: If JavaScript directory is not found
        """
        js_paths = [
            # Direct JS path (e.g., robustness/js)
            os.path.join(self.asset_manager.templates_dir, 'report_types', test_type, 'js'),
            # Interactive report JS path (e.g., uncertainty/interactive/js)
            os.path.join(self.asset_manager.templates_dir, 'report_types', test_type, 'interactive', 'js'),
            # Static report JS path (e.g., uncertainty/static/js)
            os.path.join(self.asset_manager.templates_dir, 'report_types', test_type, 'static', 'js'),
        ]

        # If report_type is specified, prioritize that path
        if report_type:
            report_type_lower = report_type.lower()
            if report_type_lower in ['interactive', 'static']:
                priority_path = os.path.join(self.asset_manager.templates_dir, 'report_types', test_type, report_type_lower, 'js')
                if os.path.exists(priority_path):
                    logger.info(f"Using JS path for {report_type_lower} report: {priority_path}")
                    return priority_path

        for path in js_paths:
            if os.path.exists(path):
                logger.info(f"Using JS path: {path}")
                return path

        error_msg = f"JavaScript directory not found for {test_type}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    def get_generic_css_path(self) -> str:
        """
        Get path to generic CSS assets.
        
        Returns:
        --------
        str : Path to generic CSS directory
        
        Raises:
        -------
        FileNotFoundError: If CSS directory is not found
        """
        css_path = os.path.join(self.asset_manager.assets_dir, 'css')
        if not os.path.exists(css_path):
            error_msg = f"Generic CSS directory not found: {css_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        return css_path
    
    def get_generic_js_path(self) -> str:
        """
        Get path to generic JavaScript assets.
        
        Returns:
        --------
        str : Path to generic JavaScript directory
        
        Raises:
        -------
        FileNotFoundError: If JavaScript directory is not found
        """
        js_path = os.path.join(self.asset_manager.assets_dir, 'js')
        if not os.path.exists(js_path):
            error_msg = f"Generic JavaScript directory not found: {js_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        return js_path
    
    def get_common_html_files(self) -> Dict[str, str]:
        """
        Get contents of common HTML template files.
        
        Returns:
        --------
        Dict[str, str] : Dictionary mapping file names to their contents
        """
        if not os.path.exists(self.asset_manager.common_dir):
            logger.warning(f"Common directory not found: {self.asset_manager.common_dir}")
            return {}
        
        html_files = {}
        for file_name in os.listdir(self.asset_manager.common_dir):
            if file_name.endswith('.html'):
                file_path = os.path.join(self.asset_manager.common_dir, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        html_files[file_name] = f.read()
                    logger.info(f"Loaded common HTML file: {file_name}")
                except Exception as e:
                    logger.warning(f"Error reading HTML file {file_path}: {str(e)}")
        
        return html_files
    
    def get_test_partials(self, test_type: str) -> Dict[str, str]:
        """
        Get HTML partial templates for a specific test type.
        
        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
            
        Returns:
        --------
        Dict[str, str] : Dictionary mapping partial names to their contents
        """
        partials_dir = os.path.join(self.asset_manager.templates_dir, 'report_types', test_type, 'partials')
        if not os.path.exists(partials_dir):
            logger.warning(f"Partials directory not found for {test_type}: {partials_dir}")
            return {}
        
        partials = {}
        for file_name in os.listdir(partials_dir):
            if file_name.endswith('.html'):
                partial_name = os.path.splitext(file_name)[0]
                file_path = os.path.join(partials_dir, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        partials[partial_name] = f.read()
                    logger.info(f"Loaded {test_type} partial: {partial_name}")
                except Exception as e:
                    logger.warning(f"Error reading partial file {file_path}: {str(e)}")
        
        return partials
    
    def _discover_css_files(self, css_dir: str) -> Dict[str, str]:
        """
        Discover all CSS files in a directory and its subdirectories.
        
        Parameters:
        -----------
        css_dir : str
            Directory containing CSS files
            
        Returns:
        --------
        Dict[str, str] : Dictionary mapping logical names to relative file paths
        """
        discovered_files = {}
        
        # Check for main.css in the root directory
        main_css_candidates = [
            "main.css",
            "styles.css",
            "style.css",
            "report.css"
        ]
        
        # Find first existing main CSS file
        for candidate in main_css_candidates:
            candidate_path = os.path.join(css_dir, candidate)
            if os.path.exists(candidate_path):
                discovered_files["main"] = candidate
                logger.info(f"Found main CSS file: {candidate}")
                break
        
        # If no main CSS found, look for any CSS file with 'main' or 'style' in the name
        if "main" not in discovered_files:
            for file_name in os.listdir(css_dir):
                if file_name.endswith('.css') and ('main' in file_name.lower() or 'style' in file_name.lower()):
                    discovered_files["main"] = file_name
                    logger.info(f"Found alternative main CSS file: {file_name}")
                    break
        
        # Check for components directory
        components_dir = os.path.join(css_dir, "components")
        if os.path.exists(components_dir) and os.path.isdir(components_dir):
            logger.info(f"Found components subdirectory: {components_dir}")
            
            # Process files in components directory
            for file_name in os.listdir(components_dir):
                if file_name.endswith('.css'):
                    component_name = os.path.splitext(file_name)[0]
                    rel_path = os.path.join("components", file_name)
                    discovered_files[component_name] = rel_path
                    logger.info(f"Found component CSS: {component_name} -> {rel_path}")
        
        # Process any additional CSS files in the root directory
        for file_name in os.listdir(css_dir):
            if file_name.endswith('.css') and file_name not in main_css_candidates:
                file_base = os.path.splitext(file_name)[0]
                if file_base not in discovered_files:
                    discovered_files[file_base] = file_name
                    logger.info(f"Found additional CSS file in root: {file_base} -> {file_name}")
        
        return discovered_files
    
    def _discover_js_files(self, js_dir: str) -> Dict[str, str]:
        """
        Discover all JavaScript files in a directory and its subdirectories.
        
        Parameters:
        -----------
        js_dir : str
            Directory containing JavaScript files
            
        Returns:
        --------
        Dict[str, str] : Dictionary mapping logical names to relative file paths
        
        Note:
        -----
        For robustness reports, fix_boxplot.js is handled separately in asset_processor.get_combined_js_content()
        to ensure it's included in the final combined JS content.
        """
        discovered_files = {}
        
        # Check if main.js exists in the root directory
        main_js_path = os.path.join(js_dir, "main.js")
        if os.path.exists(main_js_path):
            discovered_files["main"] = "main.js"
            logger.info(f"Found main.js in root directory: {main_js_path}")
        
        # Find utils.js in root directory
        utils_js_path = os.path.join(js_dir, "utils.js")
        if os.path.exists(utils_js_path):
            discovered_files["utils"] = "utils.js"
            logger.info(f"Found utils.js in root directory: {utils_js_path}")
        
        # Dictionary to track subdirectories we've found
        subdirs = {
            "charts": False,
            "controllers": False,
            "components": False
        }
        
        # Check for common subdirectories
        for subdir in subdirs.keys():
            subdir_path = os.path.join(js_dir, subdir)
            if os.path.exists(subdir_path) and os.path.isdir(subdir_path):
                subdirs[subdir] = True
                logger.info(f"Found {subdir} subdirectory: {subdir_path}")
                
                # Process files in this subdirectory
                for file_name in os.listdir(subdir_path):
                    if file_name.endswith('.js'):
                        file_base = os.path.splitext(file_name)[0]
                        logical_name = f"{subdir}_{file_base}"
                        rel_path = os.path.join(subdir, file_name)
                        discovered_files[logical_name] = rel_path
                        logger.info(f"Found JS file: {logical_name} -> {rel_path}")
        
        # Process any additional JS files in the root directory
        for file_name in os.listdir(js_dir):
            if file_name.endswith('.js') and file_name != "main.js" and file_name != "utils.js":
                file_base = os.path.splitext(file_name)[0]
                if file_base not in discovered_files:
                    discovered_files[file_base] = file_name
                    logger.info(f"Found additional JS file in root: {file_base} -> {file_name}")
        
        return discovered_files
    
    def write_json_data_file(self, data: Dict[str, Any], output_dir: str, filename: str = "dados_transformados.json") -> str:
        """
        Write data to a JSON file in the output directory.
        
        Parameters:
        -----------
        data : Dict[str, Any]
            Data to write to file
        output_dir : str
            Directory where to save the file
        filename : str, optional
            Name of the JSON file
            
        Returns:
        --------
        str : Path to the created JSON file
        
        Raises:
        -------
        IOError: If writing the file fails
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate file path
        file_path = os.path.join(output_dir, filename)
        
        try:
            # Write data to file with proper encoding and formatting
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Data written to JSON file: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error writing data to JSON file: {str(e)}")
            raise IOError(f"Failed to write data to JSON file: {str(e)}")
    
    def copy_assets_to_output(self, test_type: str, output_dir: str) -> Dict[str, str]:
        """
        Copy required assets to the output directory.
        
        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
        output_dir : str
            Directory where the report and assets will be saved
            
        Returns:
        --------
        Dict[str, str] : Paths to copied asset files
        """
        asset_paths = {}
        
        # Create output directories if they don't exist
        css_output_dir = os.path.join(output_dir, 'css')
        js_output_dir = os.path.join(output_dir, 'js')
        images_output_dir = os.path.join(output_dir, 'images')
        
        for directory in [css_output_dir, js_output_dir, images_output_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
        
        # Write combined CSS to file
        try:
            css_content = self.asset_manager.get_combined_css_content(test_type)
            css_path = os.path.join(css_output_dir, 'styles.css')
            with open(css_path, 'w', encoding='utf-8') as f:
                f.write(css_content)
            asset_paths['css'] = css_path
            logger.info(f"Combined CSS written to: {css_path}")
        except Exception as e:
            logger.error(f"Error copying CSS assets: {str(e)}")
        
        # Write combined JS to file
        try:
            js_content = self.asset_manager.get_combined_js_content(test_type)
            js_path = os.path.join(js_output_dir, 'main.js')
            with open(js_path, 'w', encoding='utf-8') as f:
                f.write(js_content)
            asset_paths['js'] = js_path
            logger.info(f"Combined JavaScript written to: {js_path}")
        except Exception as e:
            logger.error(f"Error copying JavaScript assets: {str(e)}")
        
        # Copy logo if exists
        try:
            if os.path.exists(self.asset_manager.logo_path):
                logo_output_path = os.path.join(images_output_dir, 'logo.png')
                self._copy_file(self.asset_manager.logo_path, logo_output_path)
                asset_paths['logo'] = logo_output_path
                logger.info(f"Logo copied to: {logo_output_path}")
        except Exception as e:
            logger.warning(f"Error copying logo: {str(e)}")
        
        # Copy favicon if exists
        try:
            if os.path.exists(self.asset_manager.favicon_path):
                favicon_output_path = os.path.join(images_output_dir, 'favicon.ico')
                self._copy_file(self.asset_manager.favicon_path, favicon_output_path)
                asset_paths['favicon'] = favicon_output_path
                logger.info(f"Favicon copied to: {favicon_output_path}")
        except Exception as e:
            logger.warning(f"Error copying favicon: {str(e)}")
            
        # Copy icons if they exist
        try:
            icons_dir = os.path.join(self.asset_manager.images_dir, 'icons')
            if os.path.exists(icons_dir):
                icons_output_dir = os.path.join(images_output_dir, 'icons')
                if not os.path.exists(icons_output_dir):
                    os.makedirs(icons_output_dir)
                
                for file_name in os.listdir(icons_dir):
                    if file_name.endswith(('.svg', '.png', '.jpg', '.jpeg', '.ico')):
                        icon_path = os.path.join(icons_dir, file_name)
                        icon_output_path = os.path.join(icons_output_dir, file_name)
                        self._copy_file(icon_path, icon_output_path)
                        logger.info(f"Icon {file_name} copied to: {icon_output_path}")
                
                asset_paths['icons_dir'] = icons_output_dir
        except Exception as e:
            logger.warning(f"Error copying icons: {str(e)}")
        
        return asset_paths
        
    def _copy_file(self, src_path: str, dest_path: str) -> None:
        """
        Copy a file from source to destination.
        
        Parameters:
        -----------
        src_path : str
            Source file path
        dest_path : str
            Destination file path
            
        Raises:
        -------
        IOError: If copying the file fails
        """
        try:
            # Read source file
            with open(src_path, 'rb') as src_file:
                content = src_file.read()
            
            # Write to destination file
            with open(dest_path, 'wb') as dest_file:
                dest_file.write(content)
                
        except Exception as e:
            logger.error(f"Error copying file from {src_path} to {dest_path}: {str(e)}")
            raise IOError(f"Failed to copy file: {str(e)}")