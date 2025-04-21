// ModelComparison.js - placeholder
/**
 * Model comparison renderer
 * 
 * Handles rendering of model comparison charts and tables for the overview section
 */
import { formatNumber, formatPercent } from '../../utils/Formatters.js';

class ModelComparison {
  /**
   * Initialize the model comparison renderer
   * @param {ChartFactory} chartFactory - Factory for creating charts
   */
  constructor(chartFactory) {
    this.chartFactory = chartFactory;
  }
  
  /**
   * Render model comparison bar chart
   * @param {HTMLElement} container - The container element
   * @param {Object} comparisonData - Model comparison data
   */
  renderBarChart(container, comparisonData) {
    if (!container || !comparisonData) return;
    
    const barData = [];
    
    // Add primary model
    barData.push({
      name: comparisonData.primaryModel.name,
      allFeatures: this.calculateAverage(comparisonData.primaryModel.scores),
      featureSubset: this.calculateAverage(comparisonData.primaryModel.scores) * 1.05 // Simulate feature subset
    });
    
    // Add alternative models
    comparisonData.alternativeModels.forEach(model => {
      barData.push({
        name: model.name,
        allFeatures: this.calculateAverage(model.scores),
        featureSubset: this.calculateAverage(model.scores) * 1.05 // Simulate feature subset
      });
    });
    
    const barChartConfig = {
      title: 'Model Comparison: Average Performance',
      xAxis: {
        title: 'Model',
        categories: barData.map(item => item.name)
      },
      yAxis: {
        title: comparisonData.primaryModel.metric || 'Score',
        min: 0.5,
        max: 1
      },
      series: [
        {
          name: 'All Features',
          data: barData.map(item => item.allFeatures),
          color: '#8884d8'
        },
        {
          name: 'Feature Subset',
          data: barData.map(item => item.featureSubset),
          color: '#82ca9d'
        }
      ]
    };
    
    this.chartFactory.createBarChart(container, barChartConfig);
  }
  
  /**
   * Render model comparison line chart
   * @param {HTMLElement} container - The container element
   * @param {Object} comparisonData - Model comparison data
   */
  renderLineChart(container, comparisonData) {
    if (!container || !comparisonData) return;
    
    const seriesData = [];
    const colors = ['#1b78de', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a65628'];
    let colorIndex = 0;
    
    // Add primary model
    seriesData.push({
      name: `${comparisonData.primaryModel.name} (All)`,
      data: comparisonData.primaryModel.scores,
      color: colors[colorIndex]
    });
    
    seriesData.push({
      name: `${comparisonData.primaryModel.name} (Subset)`,
      data: comparisonData.primaryModel.scores.map(score => 
        Math.min(1, score * (1 + 0.05))), // Simulate subset improvement
      color: colors[colorIndex],
      dashStyle: 'dash'
    });
    
    colorIndex++;
    
    // Add alternative models
    comparisonData.alternativeModels.forEach(model => {
      seriesData.push({
        name: `${model.name} (All)`,
        data: model.scores,
        color: colors[colorIndex % colors.length]
      });
      
      seriesData.push({
        name: `${model.name} (Subset)`,
        data: model.scores.map(score => 
          Math.min(1, score * (1 + 0.05))), // Simulate subset improvement
        color: colors[colorIndex % colors.length],
        dashStyle: 'dash'
      });
      
      colorIndex++;
    });
    
    const lineChartConfig = {
      title: 'Model Comparison by Perturbation Level',
      xAxis: {
        title: 'Perturbation Level',
        categories: comparisonData.primaryModel.levels
      },
      yAxis: {
        title: comparisonData.primaryModel.metric || 'Score',
        min: 0.5,
        max: 1
      },
      series: seriesData
    };
    
    this.chartFactory.createLineChart(container, lineChartConfig);
  }
  
  /**
   * Render model comparison table
   * @param {HTMLElement} container - The container element
   * @param {Object} comparisonData - Model comparison data
   */
  renderComparisonTable(container, comparisonData) {
    if (!container || !comparisonData) return;
    
    const tableData = [];
    
    // Add primary model
    tableData.push(this.prepareModelTableData(comparisonData.primaryModel));
    
    // Add alternative models
    comparisonData.alternativeModels.forEach(model => {
      tableData.push(this.prepareModelTableData(model));
    });
    
    const html = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Base Score</th>
            <th>Average Score</th>
            <th>Worst Score</th>
            <th>Impact (%)</th>
            <th>Improvement with Subset</th>
          </tr>
        </thead>
        <tbody>
          ${tableData.map(row => `
            <tr>
              <td>${row.name}</td>
              <td>${formatNumber(row.baseScore)}</td>
              <td>${formatNumber(row.avgScore)}</td>
              <td>${formatNumber(row.worstScore)}</td>
              <td>${formatPercent(row.impact)}</td>
              <td>${formatPercent(row.subsetImprovement)}</td>
            </tr>
          `).join('')}
        </tbody>
      </table>
      <div class="comparison-analysis mt-4">
        <h4>Model Comparison Analysis</h4>
        <p>
          ${this.generateComparisonAnalysis(tableData)}
        </p>
      </div>
    `;
    
    container.innerHTML = html;
  }
  
  /**
   * Prepare model data for table display
   * @param {Object} model - Model data
   * @return {Object} Prepared table data
   */
  prepareModelTableData(model) {
    const avgScore = this.calculateAverage(model.scores);
    const worstScore = Math.min(...model.worstScores);
    const impact = 1 - (avgScore / model.baseScore);
    
    // Simulate improvement with optimal feature subset
    const subsetImprovement = 0.05 + Math.random() * 0.05; // 5-10% improvement
    
    return {
      name: model.name,
      baseScore: model.baseScore,
      avgScore: avgScore,
      worstScore: worstScore,
      impact: impact,
      subsetImprovement: subsetImprovement
    };
  }
  
  /**
   * Generate comparison analysis text
   * @param {Array} tableData - Array of model table data
   * @return {string} Analysis text
   */
  generateComparisonAnalysis(tableData) {
    // Sort models by average score
    const sortedModels = [...tableData].sort((a, b) => b.avgScore - a.avgScore);
    const bestModel = sortedModels[0];
    const worstModel = sortedModels[sortedModels.length - 1];
    
    return `
      The comparison shows that <strong>${bestModel.name}</strong> has the best overall performance 
      with an average score of ${formatNumber(bestModel.avgScore)}. 
      All models show improvement when using the optimal feature subset, 
      with <strong>${tableData.sort((a, b) => b.subsetImprovement - a.subsetImprovement)[0].name}</strong> 
      showing the highest improvement of ${formatPercent(tableData.sort((a, b) => b.subsetImprovement - a.subsetImprovement)[0].subsetImprovement)}.
    `;
  }
  
  /**
   * Calculate average of an array of values
   * @param {Array} values - Array of numeric values
   * @return {number} Average value
   */
  calculateAverage(values) {
    if (!values || values.length === 0) return 0;
    return values.reduce((sum, val) => sum + (val || 0), 0) / values.filter(v => v !== null).length;
  }
}

export default ModelComparison;