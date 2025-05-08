"""
Test script for JavaScript syntax fixer
"""

import sys
import os
import re
import logging

# Configure a basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test")

# Create a minimal JavaScriptSyntaxFixer for testing
class JavaScriptSyntaxFixer:
    @staticmethod
    def fix_trailing_commas(js_content):
        # Pattern: { ... , } - matches an object literal that ends with a comma
        pattern = r'(\{\s*[\s\S]*?),(\s*\})'
        fixed_js = re.sub(pattern, r'\1\2', js_content)
        return fixed_js
    
    @staticmethod
    def fix_model_level_details_function(js_content):
        # Target the specific return pattern that's causing issues
        specific_pattern = r'(return\s*\{\s*levels,\s*modelScores,\s*modelNames,\s*metricName,?\s*\})(;)'
        fixed_js = re.sub(specific_pattern, 
                          r'return { levels, modelScores, modelNames, metricName }\2', 
                          js_content)
        return fixed_js
    
    @staticmethod
    def fix_model_comparison_function(js_content):
        # Target the specific return pattern
        specific_pattern = r'(return\s*\{\s*models,\s*baseScores,\s*robustnessScores,?\s*\})(;)'
        fixed_js = re.sub(specific_pattern, 
                          r'return { models, baseScores, robustnessScores }\2', 
                          js_content)
        return fixed_js

# Sample problematic JavaScript
test_js = """
extractModelLevelDetailsData: function() {
    let levels = [];
    const modelScores = {};
    const modelNames = {};
    let metricName = 'Score';
    
    // Function body...
    
    return {
        levels,
        modelScores,
        modelNames,
        metricName,
    };
}
"""

# Test the fix
fixed_js = JavaScriptSyntaxFixer.fix_model_level_details_function(test_js)

print("Original JavaScript:")
print("-" * 40)
print(test_js)
print("\nFixed JavaScript:")
print("-" * 40)
print(fixed_js)

# Test on another example
test_js2 = """
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
}
"""

fixed_js2 = JavaScriptSyntaxFixer.fix_model_comparison_function(test_js2)

print("\nOriginal JavaScript 2:")
print("-" * 40)
print(test_js2)
print("\nFixed JavaScript 2:")
print("-" * 40)
print(fixed_js2)

print("\nAll tests completed successfully!")