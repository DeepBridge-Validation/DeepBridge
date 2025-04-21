/**
 * Feature importance chart renderer
 * 
 * Handles rendering of feature importance charts for both
 * robustness and model importance
 */
import { formatNumber } from '../../utils/Formatters.js';

class ImportanceCharts {
  /**
   * Initialize the importance charts renderer
   * @param {ChartFactory} chartFactory - Factory for creating charts
   */
  constructor(chartFactory) {
    this.chartFactory = chartFactory;
  }
  
  /**
   * Render the robustness importance chart
   * @param {HTMLElement} container - The container element
   * @param {Object} featureData - Feature importance data
   */
  renderRobustnessImportance(container, featureData) {
    if (!container || !featureData || 
        !featureData.featureImportance || Object.keys(featureData.featureImportance).length === 0) {
      this.showNoDataMessage(container, 'Robustness feature importance data not available');
      return;
    }
    
    // Prepare data for chart - convert object to sorted array
    const featureItems = Object.entries(featureData.featureImportance)
      .map(([feature, importance]) => ({
        feature,
        importance,
        inSubset: featureData.featureSubset.includes(feature)
      }))
      .sort((a, b) => b.importance - a.importance);
    
    // Take top 15 features for visualization
    const topFeatures = featureItems.slice(0, 15);
    
    const chartConfig = {
      title: 'Robustness Feature Importance',
      xAxis: {
        title: 'Importance Score'
      },
      yAxis: {
        title: 'Feature',
        categories: topFeatures.map(item => item.feature)
      },
      series: [{
        name: 'Importance',
        data: topFeatures.map(item => ({
          y: item.feature,
          x: item.importance,
          color: item.inSubset ? 'rgba(46, 204, 113, 0.8)' : 'rgba(46, 204, 113, 0.4)'
        }))
      }],
      tooltipFormatter: function() {
        const feature = this.point.y;
        const importance = this.point.x;
        const inSubset = topFeatures.find(f => f.feature === feature).inSubset;
        
        return `
          <strong>${feature}</strong><br>
          Importance: <b>${importance.toFixed(4)}</b><br>
          ${inSubset ? '<b>In optimal feature subset</b>' : 'Not in optimal subset'}
        `;
      }
    };
    
    // Create the chart
    this.chartFactory.createHorizontalBarChart(container, chartConfig);
  }
  
  /**
   * Render the model importance chart
   * @param {HTMLElement} container - The container element
   * @param {Object} featureData - Feature importance data
   */
  renderModelImportance(container, featureData) {
    if (!container || !featureData || 
        !featureData.modelFeatureImportance || Object.keys(featureData.modelFeatureImportance).length === 0) {
      this.showNoDataMessage(container, 'Model feature importance data not available');
      return;
    }
    
    // Prepare data for chart - convert object to sorted array
    const featureItems = Object.entries(featureData.modelFeatureImportance)
      .map(([feature, importance]) => ({
        feature,
        importance,
        inSubset: featureData.featureSubset.includes(feature)
      }))
      .sort((a, b) => b.importance - a.importance);
    
    // Take top 15 features for visualization
    const topFeatures = featureItems.slice(0, 15);
    
    const chartConfig = {
      title: 'Model Feature Importance',
      xAxis: {
        title: 'Importance Score'
      },
      yAxis: {
        title: 'Feature',
        categories: topFeatures.map(item => item.feature)
      },
      series: [{
        name: 'Importance',
        data: topFeatures.map(item => ({
          y: item.feature,
          x: item.importance,
          color: item.inSubset ? 'rgba(27, 120, 222, 0.8)' : 'rgba(27, 120, 222, 0.4)'
        }))
      }],
      tooltipFormatter: function() {
        const feature = this.point.y;
        const importance = this.point.x;
        const inSubset = topFeatures.find(f => f.feature === feature).inSubset;
        
        return `
          <strong>${feature}</strong><br>
          Importance: <b>${importance.toFixed(4)}</b><br>
          ${inSubset ? '<b>In optimal feature subset</b>' : 'Not in optimal subset'}
        `;
      }
    };
    
    // Create the chart
    this.chartFactory.createHorizontalBarChart(container, chartConfig);
  }
  
  /**
   * Show a no data message in a container
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

export default ImportanceCharts;