"""
Test fix_boxplot.js inclusion in robustness reports.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# Import modules to test
from ..asset_manager import AssetManager
from ..asset_processor import AssetProcessor


class TestFixBoxplotInclusion(unittest.TestCase):
    """
    Tests to ensure fix_boxplot.js is properly included in the rendered reports.
    """
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for templates
        self.temp_dir = tempfile.mkdtemp()
        
        # Create directory structure for robustness report
        self.report_types_dir = os.path.join(self.temp_dir, 'report_types')
        self.robustness_dir = os.path.join(self.report_types_dir, 'robustness')
        self.robustness_js_dir = os.path.join(self.robustness_dir, 'js')
        
        # Create assets directory
        self.assets_dir = os.path.join(self.temp_dir, 'assets')
        self.assets_js_dir = os.path.join(self.assets_dir, 'js')
        self.common_dir = os.path.join(self.temp_dir, 'common')
        
        # Create necessary directories
        os.makedirs(self.robustness_js_dir, exist_ok=True)
        os.makedirs(self.assets_js_dir, exist_ok=True)
        os.makedirs(self.common_dir, exist_ok=True)
        
        # Create sample fix_boxplot.js file
        self.fix_boxplot_content = """/**
 * Custom boxplot initialization to ensure real data is used.
 * Version 1.2.0
 */
function initializeBoxplotChart() {
    console.log("Fixed boxplot initialized");
}
"""
        with open(os.path.join(self.robustness_js_dir, 'fix_boxplot.js'), 'w') as f:
            f.write(self.fix_boxplot_content)
        
        # Create a sample main.js file for robustness
        self.main_js_content = "// Main JS file\nconsole.log('Robustness main.js');"
        with open(os.path.join(self.robustness_js_dir, 'main.js'), 'w') as f:
            f.write(self.main_js_content)
        
        # Create a sample utils.js file for generic assets
        self.utils_js_content = "// Utils JS file\nfunction utilFunction() {}"
        with open(os.path.join(self.assets_js_dir, 'utils.js'), 'w') as f:
            f.write(self.utils_js_content)
    
    def tearDown(self):
        """Clean up the test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_fix_boxplot_inclusion(self):
        """Test that fix_boxplot.js is included in the combined JS content."""
        # Initialize asset manager with our temp directory
        asset_manager = AssetManager(self.temp_dir)
        
        # Get combined JS content for robustness
        combined_js = asset_manager.get_combined_js_content('robustness')
        
        # Check if fix_boxplot content is included
        self.assertIn("Custom boxplot initialization", combined_js, 
                      "fix_boxplot.js content should be included in combined JS")
        self.assertIn("Fixed boxplot initialized", combined_js,
                     "fix_boxplot.js function should be included")
        
        # Make sure main.js content is also there
        self.assertIn("Robustness main.js", combined_js,
                     "Main JS content should be included")
                     
        # Make sure utils.js content is also there
        self.assertIn("utilFunction", combined_js,
                     "Utils JS content should be included")
    
    def test_fix_boxplot_not_included_for_other_reports(self):
        """Test that fix_boxplot.js is not included in other report types."""
        # Create a directory for uncertainty reports
        uncertainty_dir = os.path.join(self.report_types_dir, 'uncertainty')
        uncertainty_js_dir = os.path.join(uncertainty_dir, 'js')
        os.makedirs(uncertainty_js_dir, exist_ok=True)
        
        # Create a sample main.js file for uncertainty
        uncertainty_js_content = "// Uncertainty main.js\nconsole.log('Uncertainty main.js');"
        with open(os.path.join(uncertainty_js_dir, 'main.js'), 'w') as f:
            f.write(uncertainty_js_content)
        
        # Initialize asset manager with our temp directory
        asset_manager = AssetManager(self.temp_dir)
        
        # Get combined JS content for uncertainty
        combined_js = asset_manager.get_combined_js_content('uncertainty')
        
        # Check that fix_boxplot content is NOT included
        self.assertNotIn("Custom boxplot initialization", combined_js,
                         "fix_boxplot.js content should not be included in uncertainty reports")
        
        # Make sure uncertainty main.js content is there
        self.assertIn("Uncertainty main.js", combined_js,
                     "Uncertainty main JS content should be included")


if __name__ == '__main__':
    unittest.main()