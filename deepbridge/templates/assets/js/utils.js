/**
 * Utility functions for DeepBridge reports
 */

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
        <div class="message ${type}">
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
        <div class="loader-container">
            <div class="loader"></div>
            <p>${message}</p>
        </div>
    `;
}

/**
 * Generate a random number from normal distribution
 * @param {number} mean - Mean of the distribution
 * @param {number} stdDev - Standard deviation
 * @returns {number} Random number
 */
function normalRandom(mean, stdDev) {
    // Box-Muller transform
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    return mean + z0 * stdDev;
}

/**
 * Show no data message in container element
 * @param {HTMLElement} element - Container element
 * @param {string} message - Message to display
 */
function showNoDataMessage(element, message) {
    if (!element) return;
    
    element.innerHTML = `
        <div class="data-unavailable">
            <div class="data-message">
                <span class="message-icon">📊</span>
                <h3>No Data Available</h3>
                <p>${message}</p>
            </div>
        </div>`;
}

/**
 * Show error message in container element
 * @param {HTMLElement} element - Container element
 * @param {string} errorMessage - Error message to display
 */
function showErrorMessage(element, errorMessage) {
    if (!element) return;
    
    element.innerHTML = `
        <div class="error-message">
            <div class="message-container">
                <span class="message-icon">⚠️</span>
                <h3>Error</h3>
                <p>${errorMessage}</p>
            </div>
        </div>`;
}