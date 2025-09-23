"""
Asset processor module.
Processes CSS and JS files for report generation.
"""

import os
import base64
import logging
import re
from typing import Dict, Any, Optional

# Configure logger
logger = logging.getLogger("deepbridge.reports")

# Import our JavaScript syntax fixer
try:
    from .js_syntax_fixer import JavaScriptSyntaxFixer
    SYNTAX_FIXER_AVAILABLE = True
    logger.info("JavaScript syntax fixer loaded")
except ImportError:
    logger.warning("JavaScript syntax fixer not available, syntax errors may occur")
    SYNTAX_FIXER_AVAILABLE = False
    
    # Simple fallback if the module is not available
    class JavaScriptSyntaxFixer:
        @staticmethod
        def apply_all_fixes(js_content):
            # Simple trailing comma fix
            js_content = re.sub(r',(\s*\})', r'\1', js_content)
            return js_content

class AssetProcessor:
    """
    Processes CSS and JS files for report generation.
    """
    
    def __init__(self, asset_manager):
        """
        Initialize with reference to parent asset manager.
        
        Parameters:
        -----------
        asset_manager : AssetManager
            Parent asset manager that owns this processor
        """
        self.asset_manager = asset_manager
    
    def get_css_content(self, css_dir: str, test_type: str = None, files: Optional[Dict[str, str]] = None) -> str:
        """
        Combine CSS files into a single string.
        
        Parameters:
        -----------
        css_dir : str
            Directory containing CSS files
        test_type : str
            Type of test for specific CSS files
        files : Dict[str, str], optional
            Dictionary mapping names to relative file paths
            
        Returns:
        --------
        str : Combined CSS content
        
        Raises:
        -------
        FileNotFoundError: If CSS directory or files don't exist
        """
        if not os.path.exists(css_dir):
            raise FileNotFoundError(f"CSS directory not found: {css_dir}")
        
        # If files not provided, discover all available CSS files
        if files is None:
            files = self.asset_manager.file_manager._discover_css_files(css_dir)
        
        # If no main CSS file found, raise an error
        if 'main' not in files:
            raise FileNotFoundError(f"No main CSS file found in {css_dir}")
        
        # Read and combine CSS files
        css_content = "/* ----- Combined CSS Styles ----- */\n\n"
        
        # Process main CSS file first
        main_path = os.path.join(css_dir, files['main'])
        try:
            with open(main_path, 'r', encoding='utf-8') as f:
                css_content += f.read() + "\n\n"
            logger.info(f"Added main CSS from: {main_path}")
        except Exception as e:
            error_msg = f"Error reading main CSS file {main_path}: {str(e)}"
            logger.error(error_msg)
            raise IOError(error_msg)
        
        # Remove main to avoid processing it again
        files_to_process = files.copy()
        files_to_process.pop('main', None)
        
        # Process remaining CSS files
        for name, rel_path in files_to_process.items():
            component_path = os.path.join(css_dir, rel_path)
            if os.path.exists(component_path):
                try:
                    css_content += f"/* ----- {name} ----- */\n"
                    with open(component_path, 'r', encoding='utf-8') as f:
                        css_content += f.read() + "\n\n"
                    logger.info(f"Added {name} CSS from: {component_path}")
                except Exception as e:
                    logger.warning(f"Error reading CSS file {component_path}: {str(e)}")
            else:
                logger.warning(f"CSS file not found: {component_path}")
        
        return css_content
    
    def get_generic_css_content(self) -> str:
        """
        Combine generic CSS files from assets/css directory.
        
        Returns:
        --------
        str : Combined generic CSS content
        
        Raises:
        -------
        FileNotFoundError: If CSS directory doesn't exist
        """
        css_dir = self.asset_manager.get_generic_css_path()
        
        # Discover all CSS files in the generic directory
        files = self.asset_manager.file_manager._discover_css_files(css_dir)
        
        if not files:
            raise FileNotFoundError(f"No CSS files found in {css_dir}")
        
        # Get the combined CSS content using the discovered files
        return self.get_css_content(css_dir, files=files)
    
    def get_combined_css_content(self, test_type: str) -> str:
        """
        Combine generic CSS files with test-specific CSS files.
        
        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
            
        Returns:
        --------
        str : Combined CSS content (generic + test-specific)
        """
        # Get generic CSS
        try:
            generic_css = self.get_generic_css_content()
            logger.info("Successfully loaded generic CSS")
        except Exception as e:
            logger.warning(f"Error loading generic CSS: {str(e)}")
            generic_css = "/* No generic CSS loaded */\n\n"
        
        # Get test-specific CSS
        try:
            css_dir = self.asset_manager.find_css_path(test_type)
            test_css = self.get_css_content(css_dir, test_type)
            logger.info(f"Successfully loaded {test_type} CSS")
        except Exception as e:
            logger.warning(f"Error loading {test_type} CSS: {str(e)}")
            test_css = f"/* No {test_type} CSS loaded */\n\n"
        
        # Combine CSS
        combined_css = "/* ===== Combined CSS (Generic + Test-specific) ===== */\n\n"
        combined_css += generic_css + "\n\n" + test_css
        
        return combined_css
    
    def get_js_content(self, js_dir: str, files: Optional[Dict[str, str]] = None) -> str:
        """
        Combine JavaScript files into a single string.
        
        Parameters:
        -----------
        js_dir : str
            Directory containing JavaScript files
        files : Dict[str, str], optional
            Dictionary mapping names to relative file paths
            
        Returns:
        --------
        str : Combined JavaScript content
        
        Raises:
        -------
        FileNotFoundError: If JavaScript directory or files don't exist
        """
        if not os.path.exists(js_dir):
            raise FileNotFoundError(f"JavaScript directory not found: {js_dir}")
        
        # If files not provided, discover all available JavaScript files
        if files is None:
            files = self.asset_manager.file_manager._discover_js_files(js_dir)
            
            if not files:
                raise FileNotFoundError(f"No JavaScript files found in {js_dir}")
        
        # Process each JavaScript file
        processed_js = {}
        for name, rel_path in files.items():
            full_path = os.path.join(js_dir, rel_path)
            if os.path.exists(full_path):
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Remove imports/exports for browser compatibility
                    lines = content.split('\n')
                    processed_lines = []
                    
                    for line in lines:
                        # Skip import and export lines
                        if line.strip().startswith('import ') and 'from' in line:
                            continue
                        elif line.strip().startswith('export default'):
                            continue
                        else:
                            processed_lines.append(line)
                    
                    processed_js[name] = '\n'.join(processed_lines)
                    logger.info(f"Processed JavaScript file: {full_path}")
                except Exception as e:
                    logger.warning(f"Error processing JavaScript file {full_path}: {str(e)}")
            else:
                logger.warning(f"JavaScript file not found: {full_path}")
        
        # If no JS files were processed, raise an error
        if not processed_js:
            raise FileNotFoundError(f"No valid JavaScript files found in {js_dir}")
        
        # Combine all JS files
        js_modules = "\n\n// ----- Módulos JS Combinados ----- //\n\n"
        
        # First, check if there's a main.js file to process first
        if 'main' in processed_js:
            js_modules += "// ----- main.js ----- //\n"
            js_modules += processed_js['main']
            js_modules += "\n\n"
            del processed_js['main']  # Remove from processed_js so it's not included again
            
        # First process utility files
        utils_files = [name for name in processed_js.keys() if 'util' in name.lower()]
        for name in utils_files:
            js_modules += f"// ----- {name} ----- //\n"
            js_modules += processed_js[name]
            js_modules += "\n\n"
            processed_js.pop(name, None)  # Remove to avoid duplication
            
        # Then process chart modules
        chart_files = [name for name in processed_js.keys() if 'chart' in name.lower()]
        for name in chart_files:
            js_modules += f"// ----- {name} ----- //\n"
            js_modules += processed_js[name]
            js_modules += "\n\n"
            processed_js.pop(name, None)  # Remove to avoid duplication
            
        # Then process controller modules
        controller_files = [name for name in processed_js.keys() if 'controller' in name.lower()]
        for name in controller_files:
            js_modules += f"// ----- {name} ----- //\n"
            js_modules += processed_js[name]
            js_modules += "\n\n"
            processed_js.pop(name, None)  # Remove to avoid duplication
            
        # Process any remaining files
        for name, content in processed_js.items():
            js_modules += f"// ----- {name} ----- //\n"
            js_modules += content
            js_modules += "\n\n"
        
        # Add initialization script without fallback functionality
        js_modules += """
        // Script principal de inicialização
        document.addEventListener('DOMContentLoaded', function() {
            console.log("Report initialized");
            
            // Initialize tabs
            initTabs();
            
            // Initialize charts
            initCharts();
        });
        
        function initTabs() {
            const tabButtons = document.querySelectorAll('.tab-btn');
            if (tabButtons.length > 0) {
                tabButtons.forEach(button => {
                    button.addEventListener('click', function() {
                        const targetTab = this.getAttribute('data-tab');
                        showTab(targetTab, this);
                    });
                });
                
                // Show first tab by default
                tabButtons[0].click();
            }
        }
        
        function showTab(tabId, buttonElement) {
            // Hide all tabs and remove active class from buttons
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            
            // Show selected tab and mark button as active
            document.getElementById(tabId).classList.add('active');
            buttonElement.classList.add('active');
        }
        
        function initCharts() {
            if (typeof initializeCharts === 'function') {
                initializeCharts();
            } else if (typeof Plotly !== 'undefined') {
                console.log("Plotly detected");
                // No fallback charts - rely on application code to initialize
            }
        }
        """
        
        return js_modules
    
    def get_generic_js_content(self) -> str:
        """
        Combine generic JavaScript files from assets/js directory.
        
        Returns:
        --------
        str : Combined generic JavaScript content
        
        Raises:
        -------
        FileNotFoundError: If JavaScript directory doesn't exist
        """
        js_dir = self.asset_manager.get_generic_js_path()
        
        # Discover all JavaScript files in the generic directory
        files = self.asset_manager.file_manager._discover_js_files(js_dir)
        
        # If no JS files found, raise an error
        if not files:
            raise FileNotFoundError(f"No generic JavaScript files found in {js_dir}")
        
        # Process each JavaScript file
        processed_js = {}
        for name, rel_path in files.items():
            full_path = os.path.join(js_dir, rel_path)
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Remove imports/exports for browser compatibility
                lines = content.split('\n')
                processed_lines = []
                
                for line in lines:
                    # Skip import and export lines
                    if line.strip().startswith('import ') and 'from' in line:
                        continue
                    elif line.strip().startswith('export default'):
                        continue
                    else:
                        processed_lines.append(line)
                
                processed_js[name] = '\n'.join(processed_lines)
                logger.info(f"Processed generic JavaScript file: {full_path}")
            except Exception as e:
                logger.warning(f"Error processing generic JavaScript file {full_path}: {str(e)}")
        
        # Combine in a logical order
        js_modules = "\n\n// ----- Generic JS Modules ----- //\n\n"
        
        # First process utils.js
        if 'utils' in processed_js:
            js_modules += "// ----- utils.js ----- //\n"
            js_modules += processed_js['utils']
            js_modules += "\n\n"
            processed_js.pop('utils', None)
        
        # Then process component files
        component_files = [name for name in processed_js.keys() if 'component' in name.lower()]
        for name in component_files:
            js_modules += f"// ----- {name} ----- //\n"
            js_modules += processed_js[name]
            js_modules += "\n\n"
            processed_js.pop(name, None)
        
        # Process any remaining files
        for name, content in processed_js.items():
            js_modules += f"// ----- {name} ----- //\n"
            js_modules += content
            js_modules += "\n\n"
        
        return js_modules
    
    def get_combined_js_content(self, test_type: str) -> str:
        """
        Combine generic JavaScript files with test-specific JavaScript files,
        applying syntax fixes if needed.
        
        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
            
        Returns:
        --------
        str : Combined JavaScript content (generic + test-specific)
        """
        # Get generic JavaScript
        try:
            generic_js = self.get_generic_js_content()
            logger.info("Successfully loaded generic JavaScript")
        except Exception as e:
            logger.warning(f"Error loading generic JavaScript: {str(e)}")
            generic_js = "// No generic JavaScript loaded\n\n"
        
        # Get test-specific JavaScript path
        js_dir = self.asset_manager.find_js_path(test_type)
        if not js_dir or not os.path.exists(js_dir):
            logger.warning(f"No JS directory found for test type {test_type}")
            test_js = f"// No {test_type} JavaScript loaded\n\n"
        else:
            # Always prepare critical error handling scripts first
            critical_scripts = []
            
            # Check for global error handler
            global_error_handler_path = os.path.join(js_dir, 'global_error_handler.js')
            if os.path.exists(global_error_handler_path):
                try:
                    with open(global_error_handler_path, 'r', encoding='utf-8') as f:
                        critical_scripts.append(f"// Global Error Handler\n{f.read()}")
                        logger.info("Added global_error_handler.js to critical scripts")
                except Exception as e:
                    logger.warning(f"Error reading global_error_handler.js: {str(e)}")
            
            # Check if syntax_fixer.js content should be included
            syntax_fixer_path = os.path.join(js_dir, 'syntax_fixer.js')
            if os.path.exists(syntax_fixer_path):
                try:
                    with open(syntax_fixer_path, 'r', encoding='utf-8') as f:
                        critical_scripts.append(f"// Syntax Fixer\n{f.read()}")
                        logger.info("Added syntax_fixer.js to critical scripts")
                except Exception as e:
                    logger.warning(f"Error reading syntax_fixer.js: {str(e)}")
            
            # Check for fixed_syntax.js
            fixed_syntax_path = os.path.join(js_dir, 'fixed_syntax.js')
            if os.path.exists(fixed_syntax_path):
                try:
                    with open(fixed_syntax_path, 'r', encoding='utf-8') as f:
                        critical_scripts.append(f"// Fixed Syntax\n{f.read()}")
                        logger.info("Added fixed_syntax.js to critical scripts")
                except Exception as e:
                    logger.warning(f"Error reading fixed_syntax.js: {str(e)}")
            
            # Check for safe_chart_manager.js
            safe_chart_manager_path = os.path.join(js_dir, 'safe_chart_manager.js')
            if os.path.exists(safe_chart_manager_path):
                try:
                    with open(safe_chart_manager_path, 'r', encoding='utf-8') as f:
                        critical_scripts.append(f"// Safe Chart Manager\n{f.read()}")
                        logger.info("Added safe_chart_manager.js to critical scripts")
                except Exception as e:
                    logger.warning(f"Error reading safe_chart_manager.js: {str(e)}")
            
            # Check for model_chart_fix.js
            model_chart_fix_path = os.path.join(js_dir, 'model_chart_fix.js')
            if os.path.exists(model_chart_fix_path):
                try:
                    with open(model_chart_fix_path, 'r', encoding='utf-8') as f:
                        critical_scripts.append(f"// Model Chart Fix\n{f.read()}")
                        logger.info("Added model_chart_fix.js to critical scripts")
                except Exception as e:
                    logger.warning(f"Error reading model_chart_fix.js: {str(e)}")
            
            # Join all critical scripts
            critical_js = "\n\n".join(critical_scripts) if critical_scripts else ""
            
            try:
                # Get test-specific JavaScript
                test_js = self.get_js_content(js_dir)
                logger.info(f"Successfully loaded {test_type} JavaScript")
            except Exception as e:
                logger.warning(f"Error loading {test_type} JavaScript: {str(e)}")
                test_js = f"// No {test_type} JavaScript loaded\n\n"
        
        # For robustness reports, directly include fix_boxplot.js if it exists
        additional_js = ""
        if test_type == "robustness":
            try:
                fix_boxplot_path = os.path.join(js_dir, "fix_boxplot.js")
                if os.path.exists(fix_boxplot_path):
                    # Get file modification time for cache-busting
                    mod_time = os.path.getmtime(fix_boxplot_path)
                    
                    # Force reload the content (no caching)
                    with open(fix_boxplot_path, 'r', encoding='utf-8') as f:
                        fix_boxplot_content = f.read()
                    
                    # Add clear markers and version/timestamp for debugging
                    additional_js = f"\n\n// ===== Boxplot Fix Script (Direct Inclusion - Modified: {mod_time}) ===== //\n\n"
                    additional_js += fix_boxplot_content
                    logger.info(f"Successfully included fix_boxplot.js directly (Modified: {mod_time})")
                else:
                    logger.warning(f"fix_boxplot.js not found at {fix_boxplot_path}")
            except Exception as e:
                logger.error(f"Error including fix_boxplot.js: {str(e)}")
        
        # Combine JavaScript based on whether we have critical scripts
        if critical_js:
            combined_js = "// ===== Critical Fixes (Load First) ===== //\n\n"
            combined_js += critical_js
            combined_js += "\n\n// ===== Generic JavaScript ===== //\n\n"
            combined_js += generic_js
            combined_js += "\n\n// ===== Test-specific JavaScript ===== //\n\n"
            combined_js += test_js
            combined_js += additional_js
            logger.info("Combined JS with critical fixes first, then generic and test-specific")
        else:
            combined_js = "// ===== Combined JavaScript (Generic + Test-specific) ===== //\n\n"
            combined_js += generic_js + "\n\n" + test_js + additional_js
            logger.info("Combined JS with standard approach (no critical fixes found)")
        
        # Apply syntax fixes to avoid JavaScript errors
        try:
            logger.info(f"Applying syntax fixes to {test_type} JavaScript")
            fixed_js = JavaScriptSyntaxFixer.apply_all_fixes(combined_js)
            logger.info("JavaScript syntax fixes applied successfully")
            return fixed_js
        except Exception as e:
            logger.error(f"Error applying JavaScript syntax fixes: {str(e)}")
            # Return original even if fixing fails
            return combined_js
    
    def get_base64_image(self, image_path: str) -> str:
        """
        Convert image to base64 string with data URL format.

        Parameters:
        -----------
        image_path : str
            Path to the image file

        Returns:
        --------
        str : Base64 encoded image string as data URL

        Raises:
        -------
        FileNotFoundError: If the image file doesn't exist
        Exception: If encoding the image fails
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            # Determine MIME type based on file extension
            ext = os.path.splitext(image_path)[1].lower()
            mime_types = {
                '.png': 'image/png',
                '.jpg': 'image/jpeg',
                '.jpeg': 'image/jpeg',
                '.gif': 'image/gif',
                '.svg': 'image/svg+xml',
                '.ico': 'image/x-icon'
            }
            mime_type = mime_types.get(ext, 'image/png')

            with open(image_path, "rb") as img_file:
                base64_str = base64.b64encode(img_file.read()).decode('utf-8')
                return f"data:{mime_type};base64,{base64_str}"
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {str(e)}")
            raise
    
    def get_logo_base64(self) -> str:
        """
        Get base64 encoded logo.
        
        Returns:
        --------
        str : Base64 encoded logo image
        
        Raises:
        -------
        FileNotFoundError: If logo file doesn't exist
        """
        if not os.path.exists(self.asset_manager.logo_path):
            raise FileNotFoundError(f"Logo file not found: {self.asset_manager.logo_path}")
        return self.get_base64_image(self.asset_manager.logo_path)
    
    def get_favicon_base64(self) -> str:
        """
        Get base64 encoded favicon.
        
        Returns:
        --------
        str : Base64 encoded favicon image
        
        Raises:
        -------
        FileNotFoundError: If favicon file doesn't exist
        """
        if not os.path.exists(self.asset_manager.favicon_path):
            raise FileNotFoundError(f"Favicon file not found: {self.asset_manager.favicon_path}")
        return self.get_base64_image(self.asset_manager.favicon_path)
    
    def get_icons(self) -> Dict[str, str]:
        """
        Get base64 encoded icons from the icons directory.
        
        Returns:
        --------
        Dict[str, str] : Dictionary mapping icon names to base64 encoded strings
        """
        icons_dir = os.path.join(self.asset_manager.images_dir, 'icons')
        if not os.path.exists(icons_dir):
            logger.warning(f"Icons directory not found: {icons_dir}")
            return {}
        
        icons = {}
        for file_name in os.listdir(icons_dir):
            if file_name.endswith(('.svg', '.png', '.jpg', '.jpeg', '.ico')):
                icon_name = os.path.splitext(file_name)[0]
                icon_path = os.path.join(icons_dir, file_name)
                try:
                    icons[icon_name] = self.get_base64_image(icon_path)
                    logger.info(f"Loaded icon: {icon_name}")
                except Exception as e:
                    logger.warning(f"Error loading icon {icon_name}: {str(e)}")
        
        return icons
    
    def create_full_report_assets(self, test_type: str) -> Dict[str, Any]:
        """
        Create a complete set of assets for a report.
        
        Parameters:
        -----------
        test_type : str
            Type of test ('robustness', 'uncertainty', etc.)
            
        Returns:
        --------
        Dict[str, Any] : Dictionary with all assets for the report
        """
        report_assets = {}
        
        # Get combined CSS and JavaScript
        try:
            report_assets['css'] = self.get_combined_css_content(test_type)
        except Exception as e:
            logger.error(f"Error getting combined CSS: {str(e)}")
            report_assets['css'] = "/* Error loading CSS */"
        
        try:
            report_assets['js'] = self.get_combined_js_content(test_type)
        except Exception as e:
            logger.error(f"Error getting combined JavaScript: {str(e)}")
            report_assets['js'] = "// Error loading JavaScript"
        
        # Get images
        try:
            report_assets['logo'] = self.get_logo_base64()
        except Exception as e:
            logger.warning(f"Error getting logo: {str(e)}")
            report_assets['logo'] = ""
        
        try:
            report_assets['favicon'] = self.get_favicon_base64()
        except Exception as e:
            logger.warning(f"Error getting favicon: {str(e)}")
            report_assets['favicon'] = ""
        
        # Get icons
        try:
            report_assets['icons'] = self.get_icons()
        except Exception as e:
            logger.warning(f"Error getting icons: {str(e)}")
            report_assets['icons'] = {}
        
        # Get common HTML templates
        try:
            report_assets['common_html'] = self.asset_manager.get_common_html_files()
        except Exception as e:
            logger.warning(f"Error getting common HTML files: {str(e)}")
            report_assets['common_html'] = {}
        
        # Get test-specific partials
        try:
            report_assets['test_partials'] = self.asset_manager.get_test_partials(test_type)
        except Exception as e:
            logger.warning(f"Error getting test partials: {str(e)}")
            report_assets['test_partials'] = {}
        
        return report_assets