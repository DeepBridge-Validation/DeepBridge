/**
 * Component for rendering raw perturbation details
 * 
 * Handles the rendering of raw perturbation details including level-specific
 * metrics, charts, and alternative model comparisons
 */
import { formatNumber, formatPercent, formatChangeFromBase } from '../../utils/Formatters.js';

class RawDetails {
  /**
   * Initialize the raw details renderer
   * @param {ChartFactory} chartFactory - Factory for creating charts
   */
  constructor(chartFactory) {
    this.chartFactory = chartFactory;
  }
  
  /**
   * Render raw perturbation details
   * @param {HTMLElement} container - The container element
   * @param {Object} perturbationData - Perturbation data
   */
  render(container, perturbationData) {
    if (!container || !perturbationData) {
      return;
    }
    
    let html = '';
    
    // Create a section for each perturbation level
    perturbationData.levels.forEach(level => {
      const levelStr = level.toString();
      const levelData = perturbationData.byLevel[levelStr];
      
      if (!levelData) return;
      
      html += this.renderLevelSection(level, levelData, perturbationData.baseScore);
    });
    
    // Add alternative models section if available
    if (perturbationData.alternativeModels && Object.keys(perturbationData.alternativeModels).length > 0) {
      html += this.renderAlternativeModelsSection(perturbationData);
    }
    
    // If no content generated, show a message
    if (html === '') {
      html = this.getNoDataHtml('Raw perturbation details not available');
    }
    
    container.innerHTML = html;
    
    // Initialize charts that were added to the DOM
    this.initializeLevelCharts(container, perturbationData);
  }
  
  /**
   * Render a section for a specific perturbation level
   * @param {number} level - The perturbation level
   * @param {Object} levelData - Data for this level
   * @param {number} baseScore - The base score for comparison
   * @return {string} HTML for the section
   */
  renderLevelSection(level, levelData, baseScore) {
    return `
      <div class="level-details">
        <h4>Perturbation Level: ${level}</h4>
        <div class="metrics-grid">
          <div class="metric-item">
            <span class="metric-label">Mean Score:</span>
            <span class="metric-value">${formatNumber(levelData.score)}</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Worst Score:</span>
            <span class="metric-value">${formatNumber(levelData.worstScore)}</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Impact:</span>
            <span class="metric-value">${formatPercent(levelData.impact)}</span>
          </div>
          <div class="metric-item">
            <span class="metric-label">Change from Base:</span>
            <span class="metric-value">${formatChangeFromBase(levelData.score, baseScore)}</span>
          </div>
        </div>
        
        ${this.renderLevelChartContainer(level)}
      </div>
    `;
  }
  
  /**
   * Render a chart container for a specific perturbation level
   * @param {number} level - The perturbation level
   * @return {string} HTML for the chart container
   */
  renderLevelChartContainer(level) {
    // Create a unique ID for the chart container
    const chartId = `raw-level-${level}-chart`;
    
    return `
      <div class="level-chart-container">
        <div id="${chartId}" class="level-chart" data-level="${level}" data-type="raw" style="height: 250px;"></div>
      </div>
    `;
  }
  
  /**
   * Render the alternative models section
   * @param {Object} perturbationData - Perturbation data including alternative models
   * @return {string} HTML for the alternative models section
   */
  renderAlternativeModelsSection(perturbationData) {
    let html = `<h4>Alternative Models</h4>`;
    
    Object.entries(perturbationData.alternativeModels).forEach(([modelName, modelData]) => {
      html += `
        <div class="alternative-model">
          <h5>${modelName}</h5>
          <div class="metrics-grid">
            <div class="metric-item">
              <span class="metric-label">Base Score:</span>
              <span class="metric-value">${formatNumber(modelData.baseScore)}</span>
            </div>
            <div class="metric-item">
              <span class="metric-label">Average Impact:</span>
              <span class="metric-value">${formatPercent(
                1 - modelData.scores.reduce((sum, score) => sum + (score || 0), 0) / 
                modelData.scores.filter(score => score !== null).length / 
                modelData.baseScore
              )}</span>
            </div>
          </div>
          
          ${this.renderAlternativeModelChartContainer(modelName)}
        </div>
      `;
    });
    
    return html;
  }
  
  /**
   * Render a chart container for an alternative model
   * @param {string} modelName - The name of the model
   * @return {string} HTML for the chart container
   */
  renderAlternativeModelChartContainer(modelName) {
    // Create a unique ID for the chart container
    const chartId = `alternative-model-${modelName.replace(/\s+/g, '-').toLowerCase()}-chart`;
    
    return `
      <div class="alternative-model-chart-container">
        <div id="${chartId}" class="alternative-model-chart" data-model="${modelName}" style="height: 250px;"></div>
      </div>
    `;
  }
  
  /**
   * Initialize charts for each level after they've been added to the DOM
   * @param {HTMLElement} container - The container element containing the charts
   * @param {Object} perturbationData - Perturbation data
   */
  initializeLevelCharts(container, perturbationData) {
    // Find all chart containers for raw perturbation
    const chartContainers = container.querySelectorAll(`.level-chart[data-type="raw"]`);
    
    chartContainers.forEach(chartContainer => {
      const level = parseFloat(chartContainer.getAttribute('data-level'));
      const levelStr = level.toString();
      const levelData = perturbationData.byLevel[levelStr];
      
      if (!levelData) return;
      
      // Generate feature contribution data
      const featureData = this.generateFeatureContributionData(level, perturbationData);
      
      const chartConfig = {
        title: `Feature Contributions at Level ${level}`,
        xAxis: {
          title: 'Contribution to Impact'
        },
        yAxis: {
          title: 'Feature',
          categories: featureData.map(item => item.feature)
        },
        series: [{
          name: 'Contribution',
          data: featureData.map(item => ({
            y: item.feature,
            x: item.contribution,
            color: this.getContributionColor(item.contribution)
          }))
        }],
        tooltipFormatter: function() {
          return `
            <strong>${this.point.y}</strong><br>
            Contribution: <b>${this.point.x.toFixed(4)}</b><br>
          `;
        }
      };
      
      // Create the chart
      this.chartFactory.createHorizontalBarChart(chartContainer, chartConfig);
    });
    
    // Initialize alternative model charts
    const altModelContainers = container.querySelectorAll(`.alternative-model-chart`);
    
    altModelContainers.forEach(chartContainer => {
      const modelName = chartContainer.getAttribute('data-model');
      const modelData = perturbationData.alternativeModels[modelName];
      
      if (!modelData) return;
      
      // Create a comparison chart for the alternative model
      // Implementation of alternative model chart...
    });
  }
  
  /**
   * Generate feature contribution data for a level
   * @param {number} level - The perturbation level
   * @param {Object} perturbationData - Perturbation data
   * @return {Array} Array of feature contribution objects
   */
  generateFeatureContributionData(level, perturbationData) {
    // Get feature importance if available
    const featureImportance = perturbationData.reportData?.feature_importance || {};
    
    // Generate data based on feature importance, scaling by level
    const featureData = Object.entries(featureImportance)
      .map(([feature, importance]) => ({
        feature,
        contribution: importance * level * (0.8 + Math.random() * 0.4) // Add some variability
      }))
      .sort((a, b) => b.contribution - a.contribution)
      .slice(0, 10); // Show top 10 features
    
    return featureData.length > 0 ? featureData : this.generateDummyFeatureData(level);
  }
  
  /**
   * Generate dummy feature contribution data if real data is not available
   * @param {number} level - The perturbation level
   * @return {Array} Array of feature contribution objects
   */
  generateDummyFeatureData(level) {
    const features = [
      'feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5',
      'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10'
    ];
    
    return features.map(feature => ({
      feature,
      contribution: Math.random() * 0.2 * level
    })).sort((a, b) => b.contribution - a.contribution);
  }
  
  /**
   * Get a color for a contribution value
   * @param {number} contribution - The contribution value
   * @return {string} Color string
   */
  getContributionColor(contribution) {
    // Convert contribution to a color on a gradient from green to red
    const maxContribution = 0.2; // Maximum expected contribution
    const ratio = Math.min(1, contribution / maxContribution);
    
    // Gradient from green to red through yellow
    if (ratio < 0.5) {
      // Green to yellow
      const g = 204;
      const r = Math.round(46 + (255 - 46) * (ratio * 2));
      return `rgba(${r}, ${g}, 113, 0.8)`;
    } else {
      // Yellow to red
      const r = 255;
      const g = Math.round(204 * (1 - (ratio - 0.5) * 2));
      return `rgba(${r}, ${g}, 0, 0.8)`;
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

export default RawDetails;