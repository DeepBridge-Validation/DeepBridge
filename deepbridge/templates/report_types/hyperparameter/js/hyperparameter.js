/**
 * Main JavaScript for hyperparameter analysis report
 */

// Initialize with report data
document.addEventListener('DOMContentLoaded', function() {
    // Get the report data from the window object or fetch it
    initializeReportData();
    
    // Set up tab navigation
    setupTabs();
});

/**
 * Initialize the report data
 */
function initializeReportData() {
    // If report data is not already available, fetch it
    if (!window.reportData) {
        console.log('Report data not found, attempting to fetch from API...');
        fetchReportData();
    } else {
        console.log('Report data loaded:', window.reportData);
        // Trigger rendering of all charts
        triggerChartRendering();
    }
}

/**
 * Fetch report data from the API
 */
function fetchReportData() {
    // Get the report ID from the URL or a data attribute
    const reportId = document.getElementById('report-container').dataset.reportId;
    
    fetch(`/api/reports/${reportId}`)
        .then(response => response.json())
        .then(data => {
            window.reportData = data;
            console.log('Report data fetched:', data);
            triggerChartRendering();
        })
        .catch(error => {
            console.error('Error fetching report data:', error);
            showErrorMessage('Failed to load report data. Please try refreshing the page.');
        });
}

/**
 * Trigger the rendering of all charts
 */
function triggerChartRendering() {
    // Ensure Plotly is loaded
    if (typeof Plotly === 'undefined') {
        console.error('Plotly is not loaded. Loading it dynamically...');
        const script = document.createElement('script');
        script.src = 'https://cdn.plot.ly/plotly-2.29.1.min.js';
        script.onload = function() {
            console.log('Plotly loaded successfully');
            // Dispatch event after Plotly is loaded
            dispatchRenderEvent();
        };
        document.head.appendChild(script);
    } else {
        // Plotly already loaded, dispatch event immediately
        dispatchRenderEvent();
    }
}

/**
 * Dispatch custom event to trigger chart rendering
 */
function dispatchRenderEvent() {
    // Small delay to ensure the DOM is fully ready
    setTimeout(() => {
        document.dispatchEvent(new CustomEvent('renderCharts'));
    }, 100);
}

/**
 * Set up tab navigation
 */
function setupTabs() {
    const tabLinks = document.querySelectorAll('.tab-link');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            // Remove active class from all tabs
            tabLinks.forEach(l => l.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab
            link.classList.add('active');
            
            // Show corresponding content
            const tabId = link.getAttribute('href').substring(1);
            document.getElementById(tabId).classList.add('active');
            
            // Trigger resize event for Plotly charts
            window.dispatchEvent(new Event('resize'));
        });
    });
}

/**
 * Display an error message to the user
 */
function showErrorMessage(message) {
    // Create an alert element
    const alertDiv = document.createElement('div');
    alertDiv.className = 'alert alert-danger';
    alertDiv.textContent = message;
    
    // Insert at the top of the report container
    const container = document.getElementById('report-container');
    container.insertBefore(alertDiv, container.firstChild);
}