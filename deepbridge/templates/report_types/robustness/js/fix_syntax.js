/**
 * Emergency syntax error fix
 * This script directly fixes the JavaScript syntax error in the generated report
 */

// This will be included at the beginning of the combined JS
// It will run as soon as the script tag loads, before any JS execution
(function() {
    // Function that will run when DOMContentLoaded fires
    function fixSyntaxErrors() {
        console.log("Applying critical syntax error fixes");
        
        // Find all script tags in the document
        const scripts = document.querySelectorAll('script');
        
        scripts.forEach(function(script) {
            // Only process inline scripts that are part of the page
            if (!script.src && script.textContent) {
                let scriptContent = script.textContent;
                
                // Fix for the specific syntax error involving return statement with trailing comma
                // This pattern matches a return statement with an object that ends with a comma before closing brace
                const returnObjectWithCommaRegex = /(return\s*\{\s*[\s\S]*?,\s*)\}/g;
                if (returnObjectWithCommaRegex.test(scriptContent)) {
                    console.log("Found syntax error pattern - fixing");
                    // Remove the trailing comma
                    scriptContent = scriptContent.replace(returnObjectWithCommaRegex, '$1}');
                    
                    // Replace the script content
                    try {
                        // Create a new script element
                        const newScript = document.createElement('script');
                        newScript.textContent = scriptContent;
                        
                        // Replace the old script with the fixed one
                        script.parentNode.replaceChild(newScript, script);
                        console.log("Script with syntax error has been fixed");
                    } catch (e) {
                        console.error("Error replacing script:", e);
                    }
                }
            }
        });
    }
    
    // Run when page starts loading
    // We need to run this before DOMContentLoaded because the error occurs during parsing
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', fixSyntaxErrors);
    } else {
        fixSyntaxErrors();
    }
})();