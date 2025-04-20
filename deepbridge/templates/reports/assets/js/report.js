/**
 * DeepBridge Report JavaScript
 * This file contains shared functionality for all report templates
 */

// Global function for tab switching
function showTab(tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Deactivate all tabs
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Show the selected tab content
    document.getElementById(tabId).classList.add('active');
    
    // Activate the selected tab
    document.querySelectorAll('.tab').forEach(tab => {
        if (tab.textContent.toLowerCase().includes(tabId)) {
            tab.classList.add('active');
        }
    });
    
    // Trigger resize event to ensure plots render correctly
    window.dispatchEvent(new Event('resize'));
}

// Format numbers with appropriate precision
function formatNumber(value, precision = 3) {
    if (typeof value !== 'number') return value;
    return value.toFixed(precision);
}

// Create color-coded metric display based on value
function createMetricDisplay(value, thresholds = {high: 0.8, medium: 0.6, low: 0}) {
    if (typeof value !== 'number') return value;
    
    let className = '';
    if (value >= thresholds.high) {
        className = 'metric-good';
    } else if (value >= thresholds.medium) {
        className = 'metric-average';
    } else {
        className = 'metric-poor';
    }
    
    return `<span class="${className}">${formatNumber(value)}</span>`;
}

// Format date for report generation timestamps
function formatDate(date) {
    if (!date) return new Date().toLocaleString();
    if (typeof date === 'string') {
        // Try to parse the string as a date
        try {
            return new Date(date).toLocaleString();
        } catch (e) {
            return date;
        }
    }
    return date.toLocaleString();
}

// Print report
function printReport() {
    window.print();
}

// Export report as PDF (requires browser print to PDF capability)
function exportPDF() {
    const originalTitle = document.title;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    document.title = `DeepBridge_Report_${timestamp}`;
    window.print();
    document.title = originalTitle;
}

// Add a fail-safe for tab clicking
function addTabHandlers() {
    // Ensure all tabs have proper handlers
    document.querySelectorAll('.tab').forEach(tab => {
        tab.onclick = function() {
            // Extract the tab id from data attribute or from text content
            const tabId = this.dataset.tab || 
                          this.textContent.trim().toLowerCase().replace(/\s+/g, '-');
            showTab(tabId);
            return false;
        };
    });
}