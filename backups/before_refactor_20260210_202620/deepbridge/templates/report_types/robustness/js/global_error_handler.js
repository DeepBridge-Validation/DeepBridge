/**
 * Global JavaScript error handler
 * Intercepts and handles JavaScript syntax errors, particularly "Illegal continue" errors
 * Version 1.0 - May 7, 2024
 */

// Execute immediately to capture errors as early as possible
(function() {
    // Store original error handler
    const originalOnError = window.onerror;
    
    // Install global error handler
    window.onerror = function(message, source, lineno, colno, error) {
        // Check for illegal continue errors
        if (message && (
            message.includes("Illegal continue") || 
            message.includes("no surrounding iteration statement") ||
            message.includes("Unexpected token 'continue'")
        )) {
            console.error("Caught illegal continue statement:", {
                message,
                source,
                lineno,
                colno
            });
            
            // Log to console for debugging
            console.warn("%cIllegal continue statement detected and intercepted", 
                         "background: #f8d7da; color: #721c24; padding: 5px; border-radius: 3px;");
            console.info(`Source: ${source}, Line: ${lineno}, Column: ${colno}`);
            
            // Attempt to add visual indicator in the UI
            setTimeout(function() {
                const errorBanner = document.createElement('div');
                errorBanner.style.cssText = "position: fixed; bottom: 10px; right: 10px; background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); z-index: 9999; max-width: 400px; font-size: 14px;";
                errorBanner.innerHTML = `
                    <div style="font-weight: bold; margin-bottom: 5px;">JavaScript Syntax Error Intercepted</div>
                    <div>A syntax error was caught by the error handler. The error was in a callback function using 'continue' outside of a loop.</div>
                    <div style="margin-top: 8px; font-size: 12px;">File: ${source.split('/').pop()}, Line: ${lineno}</div>
                    <button style="margin-top: 10px; background: #842029; color: white; border: none; padding: 5px 10px; border-radius: 3px; cursor: pointer;">Dismiss</button>
                `;
                
                document.body.appendChild(errorBanner);
                
                // Add dismiss functionality
                const dismissButton = errorBanner.querySelector('button');
                if (dismissButton) {
                    dismissButton.addEventListener('click', function() {
                        errorBanner.remove();
                    });
                }
                
                // Auto-dismiss after 10 seconds
                setTimeout(function() {
                    if (document.body.contains(errorBanner)) {
                        errorBanner.remove();
                    }
                }, 10000);
            }, 1000);
            
            // Return true to indicate we've handled the error
            return true;
        }
        
        // For other errors, call the original handler if it exists
        if (typeof originalOnError === 'function') {
            return originalOnError(message, source, lineno, colno, error);
        }
        
        // Return false to let the error propagate
        return false;
    };
    
    // Also intercept unhandled promise rejections
    window.addEventListener('unhandledrejection', function(event) {
        const error = event.reason;
        
        // Check for illegal continue errors
        if (error && error.toString && (
            error.toString().includes("Illegal continue") || 
            error.toString().includes("no surrounding iteration statement") ||
            error.toString().includes("Unexpected token 'continue'")
        )) {
            console.error("Caught illegal continue statement in promise:", error);
            
            // Prevent the error from propagating
            event.preventDefault();
        }
    });
    
    console.log("Global JavaScript error handler installed to catch illegal continue statements");
})();

// Also run when document is ready to ensure proper initialization
document.addEventListener('DOMContentLoaded', function() {
    console.log("Global error handler initialized and ready");
});