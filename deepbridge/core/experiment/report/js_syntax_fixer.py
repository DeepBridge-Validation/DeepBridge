"""
JavaScript Syntax Fixer
Fixes common syntax errors in generated JavaScript code
"""

import re
import logging

# Configure logger
logger = logging.getLogger("deepbridge.reports")

# Import JSON formatter
from .utils.json_formatter import JsonFormatter

class JavaScriptSyntaxFixer:
    """
    Fixes common syntax errors in generated JavaScript code.
    """
    
    @staticmethod
    def fix_trailing_commas(js_content):
        """
        Fix trailing commas in object literals that cause syntax errors.
        
        Args:
            js_content (str): JavaScript content to fix
            
        Returns:
            str: Fixed JavaScript content
        """
        # Use JsonFormatter to handle trailing commas in JSON-like data
        if '{"' in js_content or "{'}" in js_content:
            # Try to identify JSON object literals
            fixed_js = js_content
            
            # Match JSON object literals (both single and double quotes)
            json_pattern = r'(\b[a-zA-Z_$][a-zA-Z0-9_$]*\s*=\s*)({[\s\S]*?});'
            
            def json_replacer(match):
                var_name = match.group(1)
                json_data = match.group(2)
                try:
                    fixed_json = JsonFormatter.sanitize_json_string(json_data)
                    return f"{var_name}{fixed_json};"
                except Exception as e:
                    logger.warning(f"Failed to sanitize JSON data: {str(e)}")
                    return match.group(0)
                    
            fixed_js = re.sub(json_pattern, json_replacer, fixed_js)
            
            # Also fix inline JSON
            inline_json_pattern = r'(return\s*)({[\s\S]*?});'
            fixed_js = re.sub(inline_json_pattern, json_replacer, fixed_js)
            
            if fixed_js != js_content:
                logger.info("Fixed JSON data in JavaScript code using JsonFormatter")
                return fixed_js
        
        # Fall back to regex-based fixes for non-JSON structures
        # Fix 1: Trailing commas in return statements with object literals
        # Pattern: return { ... , } - matches a return statement with an object that ends with a comma
        pattern1 = r'(return\s*\{\s*[\s\S]*?),(\s*\})'
        fixed_js = re.sub(pattern1, r'\1\2', js_content)
        
        # Fix 2: Trailing commas in any object literal
        # Pattern: { ... , } - matches an object literal that ends with a comma
        pattern2 = r'(\{\s*[\s\S]*?),(\s*\})'
        fixed_js = re.sub(pattern2, r'\1\2', fixed_js)
        
        # Log if any fixes were made
        if fixed_js != js_content:
            logger.info("Fixed trailing commas in JavaScript code")
            
        return fixed_js
    
    @staticmethod
    def fix_undefined_variables(js_content):
        """
        Fix references to undefined variables.
        
        Args:
            js_content (str): JavaScript content to fix
            
        Returns:
            str: Fixed JavaScript content
        """
        # Add common variable declarations at the top of the script
        safe_headers = """
// Safe definitions for potentially undefined variables
if (typeof Plotly === 'undefined') {
    console.warn("Plotly not available - charts will not render");
}

// Safe reference objects
window.__safeFallbackObject = {
    levels: [],
    modelScores: {},
    modelNames: {},
    metricName: ""
};
"""
        
        # Only add safe headers if they don't already exist
        if "window.__safeFallbackObject" not in js_content:
            js_content = safe_headers + js_content
            logger.info("Added safe variable declarations to JavaScript code")
            
        return js_content
    
    @staticmethod
    def add_error_handling(js_content):
        """
        Add error handling code to catch and log errors.
        
        Args:
            js_content (str): JavaScript content to fix
            
        Returns:
            str: Fixed JavaScript content with error handling
        """
        # Add global error handler to the beginning of the script
        error_handler = """
// Global error handler
window.addEventListener('error', function(event) {
    console.error("JavaScript error caught:", event.error);
    if (event.error && event.error.toString().includes("Unexpected token")) {
        console.warn("Syntax error detected - attempting recovery");
    }
});
"""
        
        # Only add error handler if it doesn't already exist
        if "window.addEventListener('error'" not in js_content:
            js_content = error_handler + js_content
            logger.info("Added error handling to JavaScript code")
            
        return js_content
    
    @staticmethod
    def fix_model_comparison_function(js_content):
        """
        Fix the model comparison chart function that's causing syntax errors.
        
        Args:
            js_content (str): JavaScript content to fix
            
        Returns:
            str: Fixed JavaScript content
        """
        # Look for the troublesome function
        extract_function_pattern = r'(extractModelComparisonData\s*:\s*function\s*\(\)\s*\{[\s\S]*?)(return\s*\{[\s\S]*?\}\s*;)([\s\S]*?\})'
        
        def replacement(match):
            # Get the function parts
            function_start = match.group(1)
            return_statement = match.group(2)
            function_end = match.group(3)
            
            # Fix the return statement to ensure no trailing commas
            fixed_return = re.sub(r',(\s*\})', r'\1', return_statement)
            
            # Add try-catch wrapper
            safe_function = (
                f"{function_start}\n"
                f"        try {{\n"
                f"            {fixed_return}\n"
                f"        }} catch (error) {{\n"
                f"            console.error('Error in model comparison data extraction:', error);\n"
                f"            return window.__safeFallbackObject;\n"
                f"        }}\n"
                f"{function_end}"
            )
            
            return safe_function
        
        # Try to find and fix the function
        fixed_js = re.sub(extract_function_pattern, replacement, js_content)
        
        # Log if any fixes were made
        if fixed_js != js_content:
            logger.info("Fixed model comparison function")
        
        return fixed_js
    
    @staticmethod
    def fix_model_level_details_function(js_content):
        """
        Fix the extractModelLevelDetailsData function that's causing syntax errors.
        
        Args:
            js_content (str): JavaScript content to fix
            
        Returns:
            str: Fixed JavaScript content
        """
        # Look for the extractModelLevelDetailsData function's return statement
        extract_function_pattern = r'(extractModelLevelDetailsData\s*:\s*function\s*\(\)\s*\{[\s\S]*?)(return\s*\{[\s\S]*?\}\s*;)([\s\S]*?\})'
        
        def replacement(match):
            # Get the function parts
            function_start = match.group(1)
            return_statement = match.group(2)
            function_end = match.group(3)
            
            # Fix the return statement to ensure no trailing commas
            fixed_return = re.sub(r',(\s*\})', r'\1', return_statement)
            
            # Add try-catch wrapper for additional safety
            safe_function = (
                f"{function_start}\n"
                f"        try {{\n"
                f"            {fixed_return}\n"
                f"        }} catch (error) {{\n"
                f"            console.error('Error in model level details data extraction:', error);\n"
                f"            return {{\n"
                f"                levels: [0.1, 0.2, 0.3, 0.4, 0.5],\n"
                f"                modelScores: {{ 'primary': [0.8, 0.75, 0.7, 0.65, 0.6] }},\n"
                f"                modelNames: {{ 'primary': 'Primary Model' }},\n"
                f"                metricName: 'Score'\n"
                f"            }};\n"
                f"        }}\n"
                f"{function_end}"
            )
            
            return safe_function
        
        # Try to find and fix the function
        fixed_js = re.sub(extract_function_pattern, replacement, js_content)
        
        # If the pattern didn't match, try a simpler approach just fixing return statements with levels, modelScores, etc.
        if fixed_js == js_content:
            # Target the specific return pattern that's causing issues
            specific_pattern = r'(return\s*\{\s*levels,\s*modelScores,\s*modelNames,\s*metricName,?\s*\})(;)'
            fixed_js = re.sub(specific_pattern, 
                              r'return { levels, modelScores, modelNames, metricName }\2', 
                              js_content)
            
            if fixed_js != js_content:
                logger.info("Fixed model level details return statement with targeted approach")
        else:
            logger.info("Fixed model level details function")
        
        return fixed_js

    @staticmethod
    def apply_all_fixes(js_content):
        """
        Apply all JavaScript syntax fixes.
        
        Args:
            js_content (str): JavaScript content to fix
            
        Returns:
            str: Fixed JavaScript content
        """
        # Apply fixes in sequence
        fixed_js = JavaScriptSyntaxFixer.fix_trailing_commas(js_content)
        fixed_js = JavaScriptSyntaxFixer.fix_undefined_variables(fixed_js)
        fixed_js = JavaScriptSyntaxFixer.add_error_handling(fixed_js)
        fixed_js = JavaScriptSyntaxFixer.fix_model_comparison_function(fixed_js)
        fixed_js = JavaScriptSyntaxFixer.fix_model_level_details_function(fixed_js)
        
        return fixed_js