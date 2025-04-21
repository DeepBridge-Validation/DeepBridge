// CalibrationController.js - placeholder
/**
 * Main controller for the calibration section of the uncertainty report
 * 
 * Coordinates all interactions and data flow for the calibration section,
 * delegating specific rendering tasks to specialized components
 */
import CalibrationCharts from './CalibrationCharts.js';
import CalibrationTables from './CalibrationTables.js';

class CalibrationController {
  /**
   * Initialize the calibration controller
   * @param {Object} reportData - The full uncertainty report data
   * @param {DataExtractor} dataExtractor - Utility to extract data from report structure
   * @param {ChartFactory} chartFactory - Factory for creating charts
   */
  constructor(reportData, dataExtractor, chartFactory) {
    this.reportData = reportData;
    this.dataExtractor = dataExtractor;
    this.chartFactory = chartFactory;
    this.calibrationData = null;
    
    // Initialize component renderers
    this.calibrationCharts = new CalibrationCharts(chartFactory);
    this.calibrationTables = new CalibrationTables();
    
    // Chart container references
    this.calibrationChartContainer = document.getElementById('calibration-chart-container');
    this.widthChartContainer = document.getElementById('width-chart-container');
    
    // Initialize event listeners
    this.initEventListeners();
  }
  
  /**
   * Extract and prepare all data needed for the calibration section
   */
  prepareData() {
    // Extract calibration data using the data extractor
    this.calibrationData = this.dataExtractor.getCalibrationData(this.reportData);
    return this;
  }
  
  /**
   * Initialize all event listeners for the calibration section
   */
  initEventListeners() {
    // Tab navigation event for lazy loading
    document.addEventListener('tab-changed', (e) => {
      if (e.detail.tabId === 'calibration') {
        this.render();
      }
    });
    
    // Add any other event listeners for interactive elements
    document.addEventListener('renderCharts', () => {
      this.render();
    });
  }
  
  /**
   * Render all charts and tables for the calibration section
   */
  render() {
    // Ensure we have data before rendering
    if (!this.calibrationData) {
      this.prepareData();
    }
    
    // If still no data, show error messages
    if (!this.calibrationData) {
      this.showNoDataMessages();
      return;
    }
    
    // Render calibration chart
    this.calibrationCharts.renderCalibrationChart(
      this.calibrationChartContainer, 
      this.calibrationData
    );
    
    // Render interval width chart
    this.calibrationCharts.renderWidthChart(
      this.widthChartContainer, 
      this.calibrationData
    );
  }
  
  /**
   * Show no data messages in all containers
   */
  showNoDataMessages() {
    const noDataHtml = this.getNoDataHtml('Calibration data not available');
    
    if (this.calibrationChartContainer) {
      this.calibrationChartContainer.innerHTML = noDataHtml;
    }
    
    if (this.widthChartContainer) {
      this.widthChartContainer.innerHTML = noDataHtml;
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

export default CalibrationController;