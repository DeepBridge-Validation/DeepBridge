// DetailsController.js - placeholder
/**
 * Main controller for the details section of the uncertainty report
 * 
 * Coordinates all interactions and data flow for the details section,
 * delegating specific rendering tasks to specialized components
 */
import MetricsTable from './MetricsTable.js';
import AlphaLevelDetails from './AlphaLevelDetails.js';

class DetailsController {
  /**
   * Initialize the details controller
   * @param {Object} reportData - The full uncertainty report data
   * @param {DataExtractor} dataExtractor - Utility to extract data from report structure
   * @param {ChartFactory} chartFactory - Factory for creating charts
   */
  constructor(reportData, dataExtractor, chartFactory) {
    this.reportData = reportData;
    this.dataExtractor = dataExtractor;
    this.chartFactory = chartFactory;
    this.uncertaintyData = null;
    
    // Initialize component renderers
    this.metricsTable = new MetricsTable();
    this.alphaLevelDetails = new AlphaLevelDetails(chartFactory);
    
    // Container references
    this.metricsTableContainer = document.getElementById('metrics-table-container');
    this.alphaDetailsContainer = document.getElementById('alpha-details-container');
    
    // Initialize event listeners
    this.initEventListeners();
  }
  
  /**
   * Extract and prepare all data needed for the details section
   */
  prepareData() {
    // Extract uncertainty data using the data extractor
    this.uncertaintyData = this.dataExtractor.getUncertaintyData(this.reportData);
    return this;
  }
  
  /**
   * Initialize all event listeners for the details section
   */
  initEventListeners() {
    // Tab navigation event for lazy loading
    document.addEventListener('tab-changed', (e) => {
      if (e.detail.tabId === 'details') {
        this.render();
      }
    });
    
    // Add other event listeners for interactive elements
    document.addEventListener('renderCharts', () => {
      this.render();
    });
  }
  
  /**
   * Render all tables and details for the details section
   */
  render() {
    // Ensure we have data before rendering
    if (!this.uncertaintyData) {
      this.prepareData();
    }
    
    // If still no data, show error messages
    if (!this.uncertaintyData) {
      this.showNoDataMessages();
      return;
    }
    
    // Render metrics table
    this.metricsTable.render(
      this.metricsTableContainer, 
      this.uncertaintyData
    );
    
    // Render alpha level details
    this.alphaLevelDetails.render(
      this.alphaDetailsContainer, 
      this.uncertaintyData
    );
  }
  
  /**
   * Show no data messages in all containers
   */
  showNoDataMessages() {
    const noDataHtml = this.getNoDataHtml('Uncertainty details not available');
    
    if (this.metricsTableContainer) {
      this.metricsTableContainer.innerHTML = noDataHtml;
    }
    
    if (this.alphaDetailsContainer) {
      this.alphaDetailsContainer.innerHTML = noDataHtml;
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