/**
 * Main controller for the overview section of the robustness report
 * 
 * Coordinates all interactions and visualizations for the overview section,
 * including performance charts, results tables, and model comparison visualizations.
 */
import PerformanceCharts from './PerformanceCharts.js';
import ResultTables from './ResultTables.js';
import ModelComparison from './ModelComparison.js';

class OverviewController {
  /**
   * Initialize the overview controller
   * @param {Object} reportData - The full robustness report data
   * @param {DataExtractor} dataExtractor - Utility to extract data from report structure
   * @param {ChartFactory} chartFactory - Factory for creating charts
   */
  constructor(reportData, dataExtractor, chartFactory) {
    this.reportData = reportData;
    this.dataExtractor = dataExtractor;
    this.chartFactory = chartFactory;
    this.perturbationData = null;
    this.modelComparisonData = null;
    
    // Initialize component renderers
    this.performanceCharts = new PerformanceCharts(chartFactory);
    this.resultTables = new ResultTables();
    this.modelComparison = new ModelComparison(chartFactory);
    
    // Chart container references
    this.perturbationChartContainer = document.getElementById('perturbation-chart-container');
    this.worstScoreChartContainer = document.getElementById('worst-score-chart-container');
    this.meanScoreChartContainer = document.getElementById('mean-score-chart-container');
    
    // Model comparison container references
    this.barChartContainer = document.getElementById('model-comparison-chart-container');
    this.lineChartContainer = document.getElementById('model-level-details-container');
    
    // Table container references
    this.rawPerturbationTableContainer = document.getElementById('raw-perturbation-table');
    this.quantilePerturbationTableContainer = document.getElementById('quantile-perturbation-table');
    this.modelComparisonTableContainer = document.getElementById('model-comparison-analysis');
    
    // Chart navigation elements
    this.performanceChartButtons = document.querySelectorAll('[data-chart-type]');
    this.modelComparisonButtons = document.querySelectorAll('[data-comparison-view]');
    
    // Initialize event listeners
    this.initEventListeners();
  }
  
  /**
   * Extract and prepare all data needed for the overview section
   */
  prepareData() {
    // Extract perturbation data
    this.perturbationData = this.dataExtractor.getPerturbationData(this.reportData);
    
    // Extract model comparison data if alternative models are available
    if (this.perturbationData && this.perturbationData.alternativeModels) {
      this.modelComparisonData = {
        primaryModel: {
          name: this.perturbationData.modelName,
          baseScore: this.perturbationData.baseScore,
          levels: this.perturbationData.levels,
          scores: this.perturbationData.scores,
          worstScores: this.perturbationData.worstScores
        },
        alternativeModels: Object.entries(this.perturbationData.alternativeModels).map(([name, model]) => ({
          name: name,
          baseScore: model.baseScore,
          levels: this.perturbationData.levels,
          scores: model.scores,
          worstScores: model.worstScores
        }))
      };
    }
    
    return this;
  }
  
  /**
   * Initialize all event listeners for the overview section
   */
  initEventListeners() {
    // Performance chart type toggle
    this.performanceChartButtons.forEach(button => {
      button.addEventListener('click', (e) => {
        const chartType = e.target.getAttribute('data-chart-type');
        this.showPerformanceChart(chartType);
      });
    });
    
    // Model comparison view toggle
    this.modelComparisonButtons.forEach(button => {
      button.addEventListener('click', (e) => {
        const viewType = e.target.getAttribute('data-comparison-view');
        this.showModelComparisonView(viewType);
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
    if (!this.perturbationData) {
      this.prepareData();
    }
    
    // If still no data, show error messages
    if (!this.perturbationData) {
      this.showNoDataMessages();
      return;
    }
    
    // Render perturbation charts
    this.performanceCharts.renderPerturbationChart(
      this.perturbationChartContainer, 
      this.perturbationData
    );
    
    this.performanceCharts.renderWorstScoreChart(
      this.worstScoreChartContainer, 
      this.perturbationData
    );
    
    this.performanceCharts.renderMeanScoreChart(
      this.meanScoreChartContainer, 
      this.perturbationData
    );
    
    // Render model comparison if data is available
    if (this.modelComparisonData) {
      this.modelComparison.renderBarChart(
        this.barChartContainer, 
        this.modelComparisonData
      );
      
      this.modelComparison.renderLineChart(
        this.lineChartContainer, 
        this.modelComparisonData
      );
      
      this.modelComparison.renderComparisonTable(
        this.modelComparisonTableContainer,
        this.modelComparisonData
      );
    } else {
      this.showModelComparisonNoData();
    }
    
    // Render result tables
    this.resultTables.renderRawPerturbationTable(
      this.rawPerturbationTableContainer, 
      this.perturbationData
    );
    
    this.resultTables.renderQuantilePerturbationTable(
      this.quantilePerturbationTableContainer, 
      this.perturbationData
    );
    
    // Show the default charts after everything is rendered
    this.showPerformanceChart('regular');
    this.showModelComparisonView('overview');
  }
  
  /**
   * Show the performance chart of the specified type
   * @param {string} chartType - The type of chart to show ('regular', 'worst', or 'mean')
   */
  showPerformanceChart(chartType) {
    // Update button active states
    this.performanceChartButtons.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-chart-type') === chartType);
    });
    
    // Hide all charts
    this.perturbationChartContainer.style.display = 'none';
    this.worstScoreChartContainer.style.display = 'none';
    this.meanScoreChartContainer.style.display = 'none';
    
    // Show the selected chart
    switch (chartType) {
      case 'regular':
        this.perturbationChartContainer.style.display = 'block';
        break;
      case 'worst':
        this.worstScoreChartContainer.style.display = 'block';
        break;
      case 'mean':
        this.meanScoreChartContainer.style.display = 'block';
        break;
      default:
        this.perturbationChartContainer.style.display = 'block';
    }
    
    // Trigger resize event to ensure proper chart rendering
    window.dispatchEvent(new Event('resize'));
  }
  
  /**
   * Show the model comparison view of the specified type
   * @param {string} viewType - The type of view to show ('overview' or 'level-details')
   */
  showModelComparisonView(viewType) {
    // Update button active states
    this.modelComparisonButtons.forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-comparison-view') === viewType);
    });
    
    // Hide all views
    this.barChartContainer.style.display = 'none';
    this.lineChartContainer.style.display = 'none';
    
    // Show the selected view
    switch (viewType) {
      case 'overview':
        this.barChartContainer.style.display = 'block';
        break;
      case 'level-details':
        this.lineChartContainer.style.display = 'block';
        break;
      default:
        this.barChartContainer.style.display = 'block';
    }
    
    // Trigger resize event to ensure proper chart rendering
    window.dispatchEvent(new Event('resize'));
  }
  
  /**
   * Show no data messages in all containers
   */
  showNoDataMessages() {
    const noDataHtml = this.getNoDataHtml('No data available');
    
    if (this.perturbationChartContainer) {
      this.perturbationChartContainer.innerHTML = noDataHtml;
    }
    
    if (this.worstScoreChartContainer) {
      this.worstScoreChartContainer.innerHTML = noDataHtml;
    }
    
    if (this.meanScoreChartContainer) {
      this.meanScoreChartContainer.innerHTML = noDataHtml;
    }
    
    if (this.rawPerturbationTableContainer) {
      this.rawPerturbationTableContainer.innerHTML = noDataHtml;
    }
    
    if (this.quantilePerturbationTableContainer) {
      this.quantilePerturbationTableContainer.innerHTML = noDataHtml;
    }
    
    this.showModelComparisonNoData();
  }
  
  /**
   * Show no data message for model comparison
   */
  showModelComparisonNoData() {
    const noDataHtml = this.getNoDataHtml('No comparison data available. This report does not contain alternative model data for comparison.');
    
    if (this.barChartContainer) {
      this.barChartContainer.innerHTML = noDataHtml;
    }
    
    if (this.lineChartContainer) {
      this.lineChartContainer.innerHTML = noDataHtml;
    }
    
    if (this.modelComparisonTableContainer) {
      this.modelComparisonTableContainer.innerHTML = noDataHtml;
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