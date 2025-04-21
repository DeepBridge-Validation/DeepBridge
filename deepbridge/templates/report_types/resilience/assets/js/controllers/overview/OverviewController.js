// OverviewController.js - placeholder
/**
 * Main controller for the overview section of the resilience report
 * 
 * Coordinates all interactions and visualizations for the overview section,
 * including performance gap charts, summary tables, and resilience metrics.
 */
import PerformanceGapCharts from './PerformanceGapCharts.js';
import SummaryTables from './SummaryTables.js';

class OverviewController {
  /**
   * Initialize the overview controller
   * @param {Object} reportData - The full resilience report data
   * @param {DataExtractor} dataExtractor - Utility to extract data from report structure
   * @param {ChartFactory} chartFactory - Factory for creating charts
   */
  constructor(reportData, dataExtractor, chartFactory) {
    this.reportData = reportData;
    this.dataExtractor = dataExtractor;
    this.chartFactory = chartFactory;
    this.resilienceData = null;
    this.distributionData = null;
    
    // Initialize component renderers
    this.performanceGapCharts = new PerformanceGapCharts(chartFactory);
    this.summaryTables = new SummaryTables();
    
    // Chart container references
    this.performanceGapContainer = document.getElementById('performance-gap-container');
    this.intensityChartContainer = document.getElementById('intensity-chart-container');
    this.featureImpactContainer = document.getElementById('feature-impact-container');
    
    // Table container references
    this.shiftResultsContainer = document.getElementById('shift-results-container');
    this.metricsDetailsContainer = document.getElementById('metrics-details-container');
    
    // Tab navigation elements
    this.tabButtons = document.querySelectorAll('[data-tab]');
    
    // Chart selector buttons
    this.chartButtons = document.querySelectorAll('[data-chart-type]');
    
    // Initialize event listeners
    this.initEventListeners();
  }
  
  /**
   * Extract and prepare all data needed for the overview section
   */
  prepareData() {
    // Extract resilience data
    this.resilienceData = this.dataExtractor.getResilienceData(this.reportData);
    
    // Extract distribution data if available
    this.distributionData = this.dataExtractor.getDistributionData(this.reportData);
    
    return this;
  }
  
  /**
   * Initialize all event listeners for the overview section
   */
  initEventListeners() {
    // Tab navigation
    this.tabButtons.forEach(tab => {
      tab.addEventListener('click', (e) => {
        const tabId = e.target.getAttribute('data-tab');
        this.showTab(tabId);
      });
    });
    
    // Chart selector buttons
    this.chartButtons.forEach(button => {
      button.addEventListener('click', (e) => {
        const chartType = e.target.getAttribute('data-chart-type');
        this.showChart(chartType);
      });
    });
    
    // Tab navigation event for lazy loading
    document.addEventListener('tab-changed', (e) => {
      if (e.detail.tabId === 'overview') {
        this.render();
      }
    });
  }
  
  /**
   * Render all charts and tables for the overview section
   */
  render() {
    // Ensure we have data before rendering
    if (!this.resilienceData) {
      this.prepareData();
    }
    
    // If still no data, show error messages
    if (!this.resilienceData) {
      this.showNoDataMessages();
      return;
    }
    
    // Render performance gap charts
    this.performanceGapCharts.renderPerformanceGapChart(
      this.performanceGapContainer, 
      this.resilienceData
    );
    
    this.performanceGapCharts.renderIntensityChart(
      this.intensityChartContainer, 
      this.resilienceData
    );
    
    this.performanceGapCharts.renderFeatureImpactChart(
      this.featureImpactContainer, 
      this.resilienceData
    );
    
    // Render summary tables
    this.summaryTables.renderShiftResultsTable(
      this.shiftResultsContainer, 
      this.resilienceData
    );
    
    this.summaryTables.renderMetricsDetailsTable(
      this.metricsDetailsContainer, 
      this.resilienceData
    );
  }
  
  /**
   * Show the specified tab
   * @param {string} tabId - The ID of the tab to show
   */
  showTab(tabId) {
    // Update active tab button
    this.tabButtons.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });
    
    // Update content visibility
    document.querySelectorAll('.tab-content').forEach(content => {
      content.classList.toggle('active', content.id === tabId);
    });
    
    // Trigger resize event to ensure proper chart rendering
    window.dispatchEvent(new Event('resize'));
  }
  
  /**
   * Show the specified chart type
   * @param {string} chartType - The type of chart to show
   */
  showChart(chartType) {
    // Update active chart button
    this.chartButtons.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-chart-type') === chartType);
    });
    
    // Update chart visibility
    document.querySelectorAll('.chart-container').forEach(container => {
      container.classList.toggle('active', container.getAttribute('data-chart-type') === chartType);
    });
    
    // Trigger resize event to ensure proper chart rendering
    window.dispatchEvent(new Event('resize'));
  }
  
  /**
   * Show no data messages in all containers
   */
  showNoDataMessages() {
    const noDataHtml = this.getNoDataHtml('No resilience data available');
    
    if (this.performanceGapContainer) {
      this.performanceGapContainer.innerHTML = noDataHtml;
    }
    
    if (this.intensityChartContainer) {
      this.intensityChartContainer.innerHTML = noDataHtml;
    }
    
    if (this.featureImpactContainer) {
      this.featureImpactContainer.innerHTML = noDataHtml;
    }
    
    if (this.shiftResultsContainer) {
      this.shiftResultsContainer.innerHTML = noDataHtml;
    }
    
    if (this.metricsDetailsContainer) {
      this.metricsDetailsContainer.innerHTML = noDataHtml;
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