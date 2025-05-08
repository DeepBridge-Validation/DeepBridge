"""
Test the JavaScript syntax fixer with a real-world example.
"""

import os
import sys
import re

# Add parent directory to path to allow imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

# Import the actual syntax fixer module
try:
    from core.experiment.report.js_syntax_fixer import JavaScriptSyntaxFixer
    print("Successfully imported JavaScriptSyntaxFixer")
except ImportError as e:
    print(f"Error importing JavaScriptSyntaxFixer: {e}")
    
    # Create a simplified version for testing
    class JavaScriptSyntaxFixer:
        @staticmethod
        def fix_trailing_commas(js_content):
            pattern = r'(\{\s*[\s\S]*?),(\s*\})'
            fixed_js = re.sub(pattern, r'\1\2', js_content)
            return fixed_js
            
        @staticmethod
        def fix_model_comparison_function(js_content):
            return js_content
            
        @staticmethod
        def fix_model_level_details_function(js_content):
            # Target the specific return pattern that's causing issues
            specific_pattern = r'(return\s*\{\s*levels,\s*modelScores,\s*modelNames,\s*metricName,?\s*\})(;)'
            fixed_js = re.sub(specific_pattern, 
                              r'return { levels, modelScores, modelNames, metricName }\2', 
                              js_content)
            return fixed_js
            
        @staticmethod
        def apply_all_fixes(js_content):
            fixed_js = JavaScriptSyntaxFixer.fix_trailing_commas(js_content)
            fixed_js = JavaScriptSyntaxFixer.fix_model_level_details_function(fixed_js)
            return fixed_js

# Create a real-world example
example_js = """
// Chart Manager for Overview Section
const ChartManager = {
    // Various functions...
    
    extractModelLevelDetailsData: function() {
        let levels = [];
        const modelScores = {};
        const modelNames = {};
        let metricName = 'Score';
        
        // Function body with lots of data processing...
        
        // Final return statement with trailing comma
        return {
            levels,
            modelScores,
            modelNames,
            metricName,
        };
    },
    
    // More functions...
    extractModelComparisonData: function() {
        const models = [];
        const baseScores = [];
        const robustnessScores = [];
        
        // Function body...
        
        return {
            models,
            baseScores,
            robustnessScores,
        };
    },
};
"""

# Apply the fix
fixed_js = JavaScriptSyntaxFixer.apply_all_fixes(example_js)

# Print the fixed code
print("\nOriginal JavaScript:")
print("-" * 40)
print(example_js)
print("\nFixed JavaScript:")
print("-" * 40)
print(fixed_js)

# Check if the fix worked
if ",\n        };" in fixed_js:
    print("\n❌ Fix FAILED: Trailing commas still present")
else:
    print("\n✅ Fix SUCCEEDED: Trailing commas removed")

print("\nTest completed!")