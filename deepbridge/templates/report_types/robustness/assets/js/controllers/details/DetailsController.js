/**
 * Main controller for the details section of the robustness report
 * 
 * Coordinates all interactions and data flow for the details section,
 * delegating specific rendering tasks to specialized components
 */
import RawDetails from './RawDetails.js';
import QuantileDetails from './QuantileDetails.js';
import MetricsTable from './MetricsTable.js';
import { formatNumber, formatPercent, formatChangeFromBase } from '../../utils/Formatters.js';

class DetailsController {
  /**
   * Initialize the details controller
   * @param {Object} reportData - The full robustness report data
   * @param {DataExtractor} dataExtractor - Utility to extract data from report structure
   * @param {ChartFactory} chartFactory - Factory for creating charts
   */
  constructor(reportData, dataExtractor, chartFactory) {
    this.reportData = reportData;
    this.dataExtractor = dataExtractor;
    this.chartFactory = chartFactory;
    this.perturbationData = null;
    
    // Initialize detail renderers
    this.rawDetailsRenderer = new RawDetails(chartFactory);
    this.quantileDetailsRenderer = new QuantileDetails(chartFactory);
    this.metricsTableRenderer = new MetricsTable();
    
    // Container references
    this.rawDetailsContainer = document.getElementById('raw-details-container');
    this.quantileDetailsContainer = document.getElementById('quantile-details-container');
    this.metricsTableContainer = document.getElementById('metrics-table-container');
    
    // Subtab navigation elements
    this.subtabs = document.querySelectorAll('[data-subtab][data-parent="details"]');
    
    // Initialize event listeners
    this.initEventListeners();
  }
  
  /**
   * Extract and prepare all data needed for the details section
   */
  prepareData() {
    // Extract perturbation data
    this.perturbationData = this.dataExtractor.getPerturbationData(this.reportData);
    return this;
  }
  
  /**
   * Initialize all event listeners for the details section
   */
  initEventListeners() {
    // Subtab navigation
    this.subtabs.forEach(tab => {
      tab.addEventListener('click', (e) => {
        const subtabId = e.target.getAttribute('data-subtab');
        this.showSubtab(subtabId);
      });
    });
    
    // Tab navigation event for lazy loading
    document.addEventListener('tab-changed', (e) => {
      if (e.detail.tabId === 'details') {
        this.render();
      }
    });
  }
  
  /**
   * Render all components for the details section
   */
  render() {
    // Ensure we have data before rendering
    if (!this.perturbationData) {
      this.prepareData();
    }
    
    // If still no data, show error messages
    if (!this.perturbationData) {
      this.showNoDataMessages();
      return;
    }
    
    // Render detail sections
    this.rawDetailsRenderer.render(this.rawDetailsContainer, this.perturbationData);
    this.quantileDetailsRenderer.render(this.quantileDetailsContainer, this.perturbationData);
    this.metricsTableRenderer.render(this.metricsTableContainer, this.perturbationData);
  }
  
  /**
   * Show the specified subtab
   * @param {string} subtabId - The ID of the subtab to show
   */
  showSubtab(subtabId) {
    // Update subtab active states
    this.subtabs.forEach(tab => {
      tab.classList.toggle('active', tab.getAttribute('data-subtab') === subtabId);
    });
    
    // Update content visibility
    document.querySelectorAll('.subtab-content[data-parent="details"]').forEach(content => {
      content.classList.toggle('active', content.id === subtabId);
    });
    
    // Trigger resize event to ensure proper chart rendering
    window.dispatchEvent(new Event('resize'));
  }
  
  /**
   * Show no data messages in all containers
   */
  showNoDataMessages() {
    const noDataHtml = this.getNoDataHtml('Details data not available');
    
    if (this.rawDetailsContainer) {
      this.rawDetailsContainer.innerHTML = this.getNoDataHtml('Raw perturbation details not available');
    }
    
    if (this.quantileDetailsContainer) {
      this.quantileDetailsContainer.innerHTML = this.getNoDataHtml('Quantile perturbation data not available');
    }
    
    if (this.metricsTableContainer) {
      this.metricsTableContainer.innerHTML = this.getNoDataHtml('Model metrics data not available');
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