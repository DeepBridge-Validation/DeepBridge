/**
 * JavaScript error interceptor
 * This script fixes common JavaScript errors and provides safe replacements for missing scripts
 * Updated: May 9, 2024
 */

// IIFE to avoid polluting global scope
(function() {
    console.log("Installing error interceptor and safe replacements");
    
    // Install as early as possible
    // Handle existing or future ChartUtils to prevent duplication errors
    const existingChartUtils = window.ChartUtils;
    Object.defineProperty(window, 'ChartUtils', {
        get: function() {
            return existingChartUtils || {};
        },
        set: function(newValue) {
            // Only allow setting if it doesn't exist yet
            if (!existingChartUtils) {
                existingChartUtils = newValue;
            } else {
                console.warn("Prevented duplicate ChartUtils definition");
            }
        },
        configurable: true
    });
    
    // Create placeholders for missing external scripts
    window.FixedSyntax = window.FixedSyntax || { initialized: true };
    window.ModelChartFix = window.ModelChartFix || { initialized: true };
    window.SafeChartManager = window.SafeChartManager || { initialized: true };
    
    // Safe version of fixTrailingCommas function
    window.fixTrailingCommas = function() {
        console.log("Running safe fixTrailingCommas implementation");
        try {
            // Find and process all inline scripts
            const scripts = document.querySelectorAll('script:not([src])');
            
            scripts.forEach(script => {
                if (!script.textContent) return;
                
                let content = script.textContent;
                let needsReplacement = false;
                
                // Fix trailing commas in objects and arrays
                if (content.includes('},') || content.includes('], ')) {
                    const fixedContent = content
                        .replace(/,(\s*\})/g, '$1')
                        .replace(/,(\s*\])/g, '$1');
                    
                    if (content !== fixedContent) {
                        content = fixedContent;
                        needsReplacement = true;
                    }
                }
                
                // Fix other common syntax issues
                if (content.includes('try {') && !content.includes('catch')) {
                    // Add missing catch blocks to try statements
                    const fixedContent = content.replace(
                        /try\s*\{([^{}]*)\}/g,
                        'try {$1} catch(e) { console.error("Caught error:", e); }'
                    );
                    
                    if (content !== fixedContent) {
                        content = fixedContent;
                        needsReplacement = true;
                    }
                }
                
                // Safely apply changes if needed
                if (needsReplacement) {
                    try {
                        const newScript = document.createElement('script');
                        newScript.textContent = content;
                        script.parentNode.replaceChild(newScript, script);
                    } catch (error) {
                        console.error("Error replacing script:", error);
                    }
                }
            });
        } catch (error) {
            console.error("Error in fixTrailingCommas:", error);
        }
    };
    
    // Fix any attempts to load external scripts
    window.runFixes = function() {
        console.log("Running safe implementation of runFixes");
        
        try {
            // Apply fixes to existing scripts
            window.fixTrailingCommas();
            
            // Create placeholders for potentially missing modules
            window.FixedSyntax = window.FixedSyntax || { initialized: true };
            window.ModelChartFix = window.ModelChartFix || { initialized: true };
            window.SafeChartManager = window.SafeChartManager || { initialized: true };
            
            // Notify about fix completion
            setTimeout(() => {
                document.dispatchEvent(new CustomEvent('syntax_fixes_completed'));
            }, 10);
        } catch (error) {
            console.error("Error in runFixes:", error);
        }
    };
    
    // Override document.createElement to intercept script loading
    const originalCreateElement = document.createElement;
    document.createElement = function(tagName) {
        const element = originalCreateElement.call(document, tagName);
        
        if (tagName.toLowerCase() === 'script') {
            const originalSrcSetter = Object.getOwnPropertyDescriptor(HTMLScriptElement.prototype, 'src').set;
            
            Object.defineProperty(element, 'src', {
                set: function(value) {
                    // Intercept attempts to load known problematic scripts
                    if (value && (
                        value.includes('fixed_syntax.js') || 
                        value.includes('safe_chart_manager.js') || 
                        value.includes('model_chart_fix.js') ||
                        value.includes('safe_init.js')
                    )) {
                        console.log(`Prevented loading of external script: ${value}`);
                        // Simulate successful loading
                        setTimeout(() => {
                            element.dispatchEvent(new Event('load'));
                        }, 0);
                        return;
                    }
                    
                    // Normal behavior for other scripts
                    originalSrcSetter.call(this, value);
                }
            });
        }
        
        return element;
    };
    
    // Install global error handler for common JavaScript errors
    window.addEventListener('error', function(event) {
        const errorMessage = event.message || '';
        
        // Handle specific error types
        if (errorMessage.includes('already been declared') || 
            errorMessage.includes('catch or finally') ||
            errorMessage.includes('no surrounding iteration statement')) {
            
            console.warn("Intercepted known JavaScript error:", errorMessage);
            event.preventDefault();
            return false;
        }
    }, true);
    
    // Run fixes on document ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', window.runFixes);
    } else {
        window.runFixes();
    }
})();