/**
 * Main controller for the details section of the resilience report
 * 
 * Coordinates all interactions and data flow for the details section,
 * delegating specific rendering tasks to specialized components
 */
class DetailsController {
    /**
     * Initialize the details controller
     * @param {Object} reportData - The full resilience report data
     * @param {DataExtractor} dataExtractor - Utility to extract data from report structure
     * @param {ChartFactory} chartFactory - Factory for creating charts
     */
    constructor(reportData, dataExtractor, chartFactory) {
        this.reportData = reportData;
        this.dataExtractor = dataExtractor;
        this.chartFactory = chartFactory;
        this.shiftData = null;
        
        // Initialize component renderers
        this.metricsTable = new MetricsTable();
        this.shiftResults = new ShiftResults(chartFactory);
        
        // Container references
        this.metricsTableContainer = document.getElementById('metrics-table-container');
        this.shiftResultsContainer = document.getElementById('shift-results-container');
        
        // Bind event handlers
        this.handleTabChange = this.handleTabChange.bind(this);
        
        // Initialize event listeners
        this.initEventListeners();
    }
    
    /**
     * Extract and prepare all data needed for the details section
     */
    prepareData() {
        // Extract distribution shift data
        this.shiftData = this.dataExtractor.getDistributionShiftData(this.reportData);
        return this;
    }
    
    /**
     * Initialize all event listeners for the details section
     */
    initEventListeners() {
        // Tab navigation event for lazy loading
        document.addEventListener('tabChanged', this.handleTabChange);
        
        // Render when charts should be rendered
        document.addEventListener('renderCharts', () => {
            if (document.getElementById('details').classList.contains('active')) {
                this.render();
            }
        });
    }
    
    /**
     * Handle tab change event
     * @param {Event} event - Tab change event
     */
    handleTabChange(event) {
        if (event.detail && event.detail.tabId === 'details') {
            this.render();
        }
    }
    
    /**
     * Render all components for the details section
     */
    render() {
        // Ensure we have data before rendering
        if (!this.shiftData) {
            this.prepareData();
        }
        
        // If still no data, show error messages
        if (!this.shiftData) {
            this.showNoDataMessages();
            return;
        }
        
        // Render metrics table
        this.metricsTable.render(this.metricsTableContainer, this.shiftData);
        
        // Render shift results
        this.shiftResults.render(this.shiftResultsContainer, this.shiftData);
    }
    
    /**
     * Show no data messages in all containers
     */
    showNoDataMessages() {
        const noDataHtml = this.getNoDataHtml('Details data not available');
        
        if (this.metricsTableContainer) {
            this.metricsTableContainer.innerHTML = this.getNoDataHtml('Model metrics data not available');
        }
        
        if (this.shiftResultsContainer) {
            this.shiftResultsContainer.innerHTML = this.getNoDataHtml('Distribution shift results not available');
        }
    }
    
    /**
     * Get HTML for a no data message
     * @param {string} message - The message to display
     * @return {string} HTML for the message
     */
    getNoDataHtml(message) {
        return `
            <div class="alert alert-info">
                <strong>No data available</strong><br>
                ${message}
            </div>
        `;
    }
}

export default DetailsController;