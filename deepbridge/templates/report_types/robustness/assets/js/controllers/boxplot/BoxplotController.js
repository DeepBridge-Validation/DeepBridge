import ChartRenderer from './ChartRenderer.js';
import TableRenderer from './TableRenderer.js';

class BoxplotController {
  constructor(reportData, dataExtractor, chartFactory) {
    this.reportData = reportData;
    this.dataExtractor = dataExtractor;
    this.chartFactory = chartFactory;
    this.perturbationData = null;
    
    // Initialize renderers
    this.chartRenderer = new ChartRenderer(chartFactory);
    this.tableRenderer = new TableRenderer();
    
    // Container references
    this.primaryBoxplotContainer = document.getElementById('primary-model-boxplot-container');
    this.modelComparisonBoxplotContainer = document.getElementById('models-comparison-boxplot-container');
    this.distributionStatsContainer = document.getElementById('distribution-stats-container');
    this.statsTableBody = document.getElementById('stats-table-body');
    this.degradationTableBody = document.getElementById('degradation-table-body');
    
    // Initialize event listeners
    this.initEventListeners();
  }
  
  prepareData() {
    // Extract perturbation data
    this.perturbationData = this.dataExtractor.getPerturbationData(this.reportData);
    
    // Generate synthetic distribution data based on the perturbation results
    if (this.perturbationData) {
      this.perturbationData.distribution = this.generateDistributionData(this.perturbationData);
    }
    
    return this;
  }
  
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
    
    // Render charts and tables using the specialized renderers
    this.chartRenderer.renderPrimaryModelBoxplot(
      this.primaryBoxplotContainer, 
      this.perturbationData
    );
    
    this.chartRenderer.renderModelComparisonBoxplot(
      this.modelComparisonBoxplotContainer, 
      this.perturbationData
    );
    
    this.chartRenderer.renderDistributionAnalysis(
      this.distributionStatsContainer, 
      this.perturbationData
    );
    
    this.tableRenderer.renderStatisticalSummary(
      this.statsTableBody, 
      this.perturbationData
    );
    
    this.tableRenderer.renderDegradationAnalysis(
      this.degradationTableBody, 
      this.perturbationData
    );
  }
  
  // Other methods that coordinate activities but delegate rendering and data manipulation
  // to specialized classes
}

export default BoxplotController;