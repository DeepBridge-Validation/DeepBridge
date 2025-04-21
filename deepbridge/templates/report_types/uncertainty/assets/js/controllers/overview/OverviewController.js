// OverviewController.js - placeholder
/**
 * Main controller for the overview section of the uncertainty report
 * 
 * Coordinates all interactions and data flow for the overview section,
 * delegating specific rendering tasks to specialized components
 */
import CoverageCharts from './CoverageCharts.js';
import SummaryMetrics from './SummaryMetrics.js';

class OverviewController {
  /**
   * Initialize the overview controller
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
    this.coverageCharts = new CoverageCharts(chartFactory);
    this.summaryMetrics = new SummaryMetrics();
    
    // Container references
    this.alphaCoverageChartContainer = document.getElementById('alpha-coverage-chart-container');
    this.summaryContainer = document.getElementById('uncertainty-summary-container');
    
    // Initialize event listeners
    this.initEventListeners();
  }
  
  /**
   * Extract and prepare all data needed for the overview section
   */
  prepareData() {
    // Extract uncertainty data using the data extractor
    this.uncertaintyData = this.dataExtractor.getUncertaintyData(this.reportData);
    return this;
  }
  
  /**
   * Initialize all event listeners for the overview section
   */
  initEventListeners() {
    // Tab navigation event for lazy loading
    document.addEventListener('tab-changed', (e) => {
      if (e.detail.tabId === 'overview') {
        this.render();
      }
    });
    
    // Add any other event listeners for interactive elements
    document.addEventListener('renderCharts', () => {
      this.render();
    });
  }
  
  /**
   * Render all charts and metrics for the overview section
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
    
    // Render alpha coverage chart
    this.coverageCharts.renderAlphaCoverageChart(
      this.alphaCoverageChartContainer, 
      this.uncertaintyData
    );
    
    // Render summary metrics if the container exists
    if (this.summaryContainer) {
      this.summaryMetrics.render(
        this.summaryContainer, 
        this.uncertaintyData
      );
    }
  }
  
  /**
   * Show no data messages in all containers
   */
  showNoDataMessages() {
    const noDataHtml = this.getNoDataHtml('Uncertainty data not available');
    
    if (this.alphaCoverageChartContainer) {
      this.alphaCoverageChartContainer.innerHTML = noDataHtml;
    }
    
    if (this.summaryContainer) {
      this.summaryContainer.innerHTML = noDataHtml;
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

export default OverviewController;