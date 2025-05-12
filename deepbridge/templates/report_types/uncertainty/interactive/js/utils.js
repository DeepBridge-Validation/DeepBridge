// Utility functions for robustness report
/**
 * Format a numeric value for display
 * 
 * @param {number} value - The value to format
 * @param {number} decimals - Number of decimal places (default: 3)
 * @param {boolean} percentage - Whether to format as percentage
 * @returns {string} Formatted value
 */
function formatValue(value, decimals = 3, percentage = false) {
    if (value === null || value === undefined || isNaN(value)) {
        return "N/A";
    }
    
    if (percentage) {
        return (value * 100).toFixed(decimals) + "%";
    }
    
    return value.toFixed(decimals);
}

/**
 * Get color based on value (red-to-green gradient)
 * 
 * @param {number} value - Value between 0 and 1
 * @param {boolean} invertScale - If true, 0 is green and 1 is red (default: false)
 * @returns {string} RGB color string
 */
function getScoreColor(value, invertScale = false) {
    if (value === null || value === undefined || isNaN(value)) {
        return "rgb(150, 150, 150)";
    }
    
    // Ensure value is between 0 and 1
    value = Math.max(0, Math.min(1, value));
    
    // Invert scale if needed
    if (invertScale) {
        value = 1 - value;
    }
    
    // Red to green gradient
    const red = Math.round(255 * (1 - value));
    const green = Math.round(255 * value);
    
    return `rgb(${red}, ${green}, 50)`;
}

/**
 * Toggle element visibility
 * 
 * @param {string} elementId - The ID of the element to toggle
 */
function toggleElement(elementId) {
    const element = document.getElementById(elementId);
    if (element) {
        if (element.classList.contains('hidden')) {
            element.classList.remove('hidden');
        } else {
            element.classList.add('hidden');
        }
    }
}

/**
 * Show message in an element
 * 
 * @param {HTMLElement} element - Element to show message in
 * @param {string} message - Message text
 * @param {string} type - Message type (info, warning, error)
 */
function showMessage(element, message, type = 'info') {
    if (!element) return;
    
    let iconClass = 'info-icon';
    let bgColor = '#e3f2fd';
    let textColor = '#0d47a1';
    
    if (type === 'warning') {
        iconClass = 'warning-icon';
        bgColor = '#fff3e0';
        textColor = '#e65100';
    } else if (type === 'error') {
        iconClass = 'error-icon';
        bgColor = '#ffebee';
        textColor = '#c62828';
    }
    
    element.innerHTML = `
        <div class="message ${type}" style="padding: 15px; background: ${bgColor}; color: ${textColor}; border-radius: 4px; margin: 10px 0;">
            <span class="${iconClass}"></span>
            ${message}
        </div>
    `;
}

/**
 * Show loader in an element
 * 
 * @param {HTMLElement} element - Element to show loader in
 * @param {string} message - Loading message
 */
function showLoader(element, message = 'Loading...') {
    if (!element) return;
    
    element.innerHTML = `
        <div class="loader-container" style="text-align: center; padding: 20px;">
            <div class="loader" style="display: inline-block; width: 40px; height: 40px; border: 3px solid #f3f3f3; border-radius: 50%; border-top: 3px solid #3498db; animation: spin 1s linear infinite;"></div>
            <p style="margin-top: 10px;">${message}</p>
        </div>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    `;
}