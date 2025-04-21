/**
 * Main controller for the feature importance section of the robustness report
 * 
 * Coordinates all interactions and visualizations for feature importance analysis,
 * including robustness importance, model importance, and comparison visualizations.
 */
import ImportanceCharts from './ImportanceCharts.js';
import ComparisonCharts from './ComparisonCharts.js';
import TableRenderers from './TableRenderers.js';

class FeatureController {
  /**
   * Initialize the feature controller
   * @param {Object} reportData - The full robustness report data
   * @param {DataExtractor} dataExtractor - Utility to extract data from report structure
   * @param {ChartFactory} chartFactory - Factory for creating charts
   */
  constructor(reportData, dataExtractor, chartFactory) {
    this.reportData = reportData;
    this.dataExtractor = dataExtractor;
    this.chartFactory = chartFactory;
    this.featureData = null;
    
    // Initialize chart and table renderers
    this.importanceCharts = new ImportanceCharts(chartFactory);
    this.comparisonCharts = new ComparisonCharts(chartFactory);
    this.tableRenderers = new TableRenderers();
    
    // Chart container references
    this.featureImportanceContainer = document.getElementById('feature-importance-container');
    this.modelFeaturesContainer = document.getElementById('model-features-container');
    
    // Comparison chart containers
    this.importanceBarChartContainer = document.getElementById('importance-bar-chart');
    this.importanceScatterChartContainer = document.getElementById('importance-scatter-chart');
    this.importanceRadarChartContainer = document.getElementById('importance-radar-chart');
    this.importanceHeatmapContainer = document.getElementById('importance-heatmap');
    this.importanceRankingContainer = document.getElementById('importance-ranking');
    
    // Table container references
    this.featureImportanceTableContainer = document.getElementById('feature-importance-table');
    this.modelFeaturesTableContainer = document.getElementById('model-features-table');
    this.importanceComparisonTableContainer = document.getElementById('importance-comparison-table');
    
    // Subtab navigation elements
    this.subtabs = document.querySelectorAll('[data-subtab][data-parent="features"]');
    
    // Chart selector buttons
    this.chartButtons = document.querySelectorAll('.chart-btn');
    
    // Initialize event listeners
    this.initEventListeners();
  }
  
  /**
   * Extract and prepare all data needed for the feature section
   */
  prepareData() {
    // Extract feature importance data
    this.featureData = this.dataExtractor.getFeatureImportanceData(this.reportData);
    return this;
  }
  
  /**
   * Initialize all event listeners for the feature section
   */
  initEventListeners() {
    // Subtab navigation
    this.subtabs.forEach(tab => {
      tab.addEventListener('click', (e) => {
        const subtabId = e.target.getAttribute('data-subtab');
        this.showSubtab(subtabId);
      });
    });
    
    // Chart selector buttons
    this.chartButtons.forEach(button => {
      button.addEventListener('click', (e) => {
        const chartId = e.target.id;
        this.showComparisonChart(chartId);
      });
    });
    
    // Tab navigation event for lazy loading
    document.addEventListener('tab-changed', (e) => {
      if (e.detail.tabId === 'feature_impact') {
        this.render();
      }
    });
  }
  
  /**
   * Render all charts and tables for the feature section
   */
  render() {
    // Ensure we have data before rendering
    if (!this.featureData) {
      this.prepareData();
    }
    
    // If still no data, show error messages
    if (!this.featureData) {
      this.showNoDataMessages();
      return;
    }
    
    // Render robustness importance visualizations
    this.importanceCharts.renderRobustnessImportance(
      this.featureImportanceContainer, 
      this.featureData
    );
    
    this.tableRenderers.renderRobustnessImportanceTable(
      this.featureImportanceTableContainer, 
      this.featureData
    );
    
    // Render model importance visualizations
    this.importanceCharts.renderModelImportance(
      this.modelFeaturesContainer, 
      this.featureData
    );
    
    this.tableRenderers.renderModelImportanceTable(
      this.modelFeaturesTableContainer, 
      this.featureData
    );
    
    // Render comparison visualizations
    this.renderComparisonCharts();
    
    this.tableRenderers.renderComparisonTable(
      this.importanceComparisonTableContainer, 
      this.featureData
    );
  }
  
  /**
   * Render comparison charts (bar, scatter, radar, heatmap, ranking)
   */
  renderComparisonCharts() {
    if (!this.featureData || 
        !this.featureData.featureImportance || Object.keys(this.featureData.featureImportance).length === 0 ||
        !this.featureData.modelFeatureImportance || Object.keys(this.featureData.modelFeatureImportance).length === 0) {
      this.showComparisonNoData();
      return;
    }
    
    // Get combined data for common features
    const combinedData = this.getCombinedFeatureData();
    
    // Render each chart type
    this.comparisonCharts.renderBarChart(this.importanceBarChartContainer, combinedData);
    this.comparisonCharts.renderScatterChart(this.importanceScatterChartContainer, combinedData);
    this.comparisonCharts.renderRadarChart(this.importanceRadarChartContainer, combinedData);
    this.comparisonCharts.renderHeatmapChart(this.importanceHeatmapContainer, combinedData);
    this.comparisonCharts.renderRankingChart(this.importanceRankingContainer, combinedData);
    
    // Show the bar chart by default (it's already visible)
    this.showComparisonChart('barChartBtn');
  }
  
  /**
   * Get combined feature data for features present in both importance sets
   * @return {Array} Array of combined feature data objects
   */
  getCombinedFeatureData() {
    const featureImportance = this.featureData.featureImportance || {};
    const modelFeatureImportance = this.featureData.modelFeatureImportance || {};
    const featureSubset = this.featureData.featureSubset || [];
    
    // Find common features between both sets
    const allFeatures = new Set([
      ...Object.keys(featureImportance),
      ...Object.keys(modelFeatureImportance)
    ]);
    
    // Combine the data
    return Array.from(allFeatures).map(feature => ({
      feature,
      robustnessImportance: featureImportance[feature] || 0,
      modelImportance: modelFeatureImportance[feature] || 0,
      inSubset: featureSubset.includes(feature)
    }));
  }
  
  /**
   * Show the specified comparison chart and hide others
   * @param {string} chartButtonId - The ID of the button for the chart to show
   */
  showComparisonChart(chartButtonId) {
    // Update button active states
    this.chartButtons.forEach(button => {
      button.classList.toggle('active', button.id === chartButtonId);
    });
    
    // Hide all charts
    if (this.importanceBarChartContainer) this.importanceBarChartContainer.style.display = 'none';
    if (this.importanceScatterChartContainer) this.importanceScatterChartContainer.style.display = 'none';
    if (this.importanceRadarChartContainer) this.importanceRadarChartContainer.style.display = 'none';
    if (this.importanceHeatmapContainer) this.importanceHeatmapContainer.style.display = 'none';
    if (this.importanceRankingContainer) this.importanceRankingContainer.style.display = 'none';
    
    // Show the selected chart
    let container;
    switch (chartButtonId) {
      case 'barChartBtn':
        container = this.importanceBarChartContainer;
        break;
      case 'scatterChartBtn':
        container = this.importanceScatterChartContainer;
        break;
      case 'radarChartBtn':
        container = this.importanceRadarChartContainer;
        break;
      case 'heatmapBtn':
        container = this.importanceHeatmapContainer;
        break;
      case 'rankingBtn':
        container = this.importanceRankingContainer;
        break;
      default:
        container = this.importanceBarChartContainer;
    }
    
    if (container) {
      container.style.display = 'block';
    }
    
    // Trigger resize event for proper rendering
    window.dispatchEvent(new Event('resize'));
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
    document.querySelectorAll('.subtab-content[data-parent="features"]').forEach(content => {
      content.classList.toggle('active', content.id === subtabId);
    });
    
    // Trigger resize event to ensure proper chart rendering
    window.dispatchEvent(new Event('resize'));
  }
  
  /**
   * Show no data messages in all containers
   */
  showNoDataMessages() {
    this.showNoDataMessage(this.featureImportanceContainer, 'Robustness feature importance data not available');
    this.showNoDataMessage(this.featureImportanceTableContainer, 'Robustness feature importance data not available');
    this.showNoDataMessage(this.modelFeaturesContainer, 'Model feature importance data not available');
    this.showNoDataMessage(this.modelFeaturesTableContainer, 'Model feature importance data not available');
    this.showComparisonNoData();
  }
  
  /**
   * Show no data messages for comparison charts and table
   */
  showComparisonNoData() {
    const noDataHtml = `
      <div class="alert alert-info">
        <strong>Comparison data not available</strong><br>
        Both model and robustness feature importance data are required for comparison.
      </div>
    `;
    
    if (this.importanceBarChartContainer) {
      this.importanceBarChartContainer.innerHTML = noDataHtml;
    }
    
    if (this.importanceScatterChartContainer) {
      this.importanceScatterChartContainer.innerHTML = noDataHtml;
    }
    
    if (this.importanceRadarChartContainer) {
      this.importanceRadarChartContainer.innerHTML = noDataHtml;
    }
    
    if (this.importanceHeatmapContainer) {
      this.importanceHeatmapContainer.innerHTML = noDataHtml;
    }
    
    if (this.importanceRankingContainer) {
      this.importanceRankingContainer.innerHTML = noDataHtml;
    }
    
    if (this.importanceComparisonTableContainer) {
      this.importanceComparisonTableContainer.innerHTML = noDataHtml;
    }
  }
  
  /**
   * Show a no data message in a specific container
   * @param {HTMLElement} container - The container element
   * @param {string} message - The message to display
   */
  showNoDataMessage(container, message) {
    if (!container) return;
    
    container.innerHTML = `
      <div class="alert alert-info">
        <strong>No data available</strong><br>
        ${message}
      </div>
    `;
  }
}

export default FeatureController;