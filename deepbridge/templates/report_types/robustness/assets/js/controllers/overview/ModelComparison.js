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
    if (!container || !comparisonData) {
      container.innerHTML = "<div class='data-message'>No comparison data available for visualization.</div>";
      return;
    }
    
    // Check if we have feature subset data
    const hasFeatureSubset = comparisonData.primaryModel.featureSubsetScores || 
                            (comparisonData.primaryModel.subsetScores && comparisonData.primaryModel.subsetScores.length > 0);
    
    if (!hasFeatureSubset && 
        !comparisonData.alternativeModels.some(model => model.featureSubsetScores || 
                                              (model.subsetScores && model.subsetScores.length > 0))) {
      // If no feature subset data is available, just show all features chart
      const barData = [];
      
      // Add primary model
      barData.push({
        name: comparisonData.primaryModel.name,
        allFeatures: this.calculateAverage(comparisonData.primaryModel.scores)
      });
      
      // Add alternative models
      comparisonData.alternativeModels.forEach(model => {
        barData.push({
          name: model.name,
          allFeatures: this.calculateAverage(model.scores)
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
          min: Math.max(0, Math.min(...barData.map(item => item.allFeatures)) * 0.9),
          max: Math.min(1, Math.max(...barData.map(item => item.allFeatures)) * 1.1)
        },
        series: [
          {
            name: 'All Features',
            data: barData.map(item => item.allFeatures),
            color: '#8884d8'
          }
        ]
      };
      
      this.chartFactory.createBarChart(container, barChartConfig);
      return;
    }
    
    // If we have feature subset data, show both
    const barData = [];
    
    // Add primary model
    barData.push({
      name: comparisonData.primaryModel.name,
      allFeatures: this.calculateAverage(comparisonData.primaryModel.scores),
      featureSubset: comparisonData.primaryModel.featureSubsetScores ? 
                    this.calculateAverage(comparisonData.primaryModel.featureSubsetScores) :
                    comparisonData.primaryModel.subsetScores ? 
                    this.calculateAverage(comparisonData.primaryModel.subsetScores) : null
    });
    
    // Add alternative models
    comparisonData.alternativeModels.forEach(model => {
      barData.push({
        name: model.name,
        allFeatures: this.calculateAverage(model.scores),
        featureSubset: model.featureSubsetScores ? 
                      this.calculateAverage(model.featureSubsetScores) :
                      model.subsetScores ? 
                      this.calculateAverage(model.subsetScores) : null
      });
    });
    
    // Filter out null subset values
    const hasValidSubset = barData.some(item => item.featureSubset !== null);
    
    const barChartConfig = {
      title: 'Model Comparison: Average Performance',
      xAxis: {
        title: 'Model',
        categories: barData.map(item => item.name)
      },
      yAxis: {
        title: comparisonData.primaryModel.metric || 'Score',
        min: Math.max(0, Math.min(...barData.map(item => 
          Math.min(item.allFeatures, item.featureSubset || item.allFeatures))) * 0.9),
        max: Math.min(1, Math.max(...barData.map(item => 
          Math.max(item.allFeatures, item.featureSubset || 0))) * 1.1)
      },
      series: [
        {
          name: 'All Features',
          data: barData.map(item => item.allFeatures),
          color: '#8884d8'
        }
      ]
    };
    
    // Only add feature subset series if we have valid data
    if (hasValidSubset) {
      barChartConfig.series.push({
        name: 'Feature Subset',
        data: barData.map(item => item.featureSubset),
        color: '#82ca9d'
      });
    } else {
      console.warn("No feature subset data available for bar chart visualization");
    }
    
    this.chartFactory.createBarChart(container, barChartConfig);
  }
  
  /**
   * Render model comparison line chart
   * @param {HTMLElement} container - The container element
   * @param {Object} comparisonData - Model comparison data
   */
  renderLineChart(container, comparisonData) {
    if (!container || !comparisonData) {
      container.innerHTML = "<div class='data-message'>No comparison data available for visualization.</div>";
      return;
    }
    
    // Check if we have feature subset data
    const hasFeatureSubset = comparisonData.primaryModel.featureSubsetScores || 
                            (comparisonData.primaryModel.subsetScores && comparisonData.primaryModel.subsetScores.length > 0);
    
    const seriesData = [];
    const colors = ['#1b78de', '#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a65628'];
    let colorIndex = 0;
    
    // Add primary model
    seriesData.push({
      name: `${comparisonData.primaryModel.name} (All)`,
      data: comparisonData.primaryModel.scores,
      color: colors[colorIndex]
    });
    
    // Add feature subset data for primary model if available
    if (hasFeatureSubset) {
      const subsetScores = comparisonData.primaryModel.featureSubsetScores || 
                         comparisonData.primaryModel.subsetScores;
                         
      if (subsetScores && subsetScores.length > 0) {
        seriesData.push({
          name: `${comparisonData.primaryModel.name} (Subset)`,
          data: subsetScores,
          color: colors[colorIndex],
          dashStyle: 'dash'
        });
      }
    }
    
    colorIndex++;
    
    // Add alternative models
    comparisonData.alternativeModels.forEach(model => {
      seriesData.push({
        name: `${model.name} (All)`,
        data: model.scores,
        color: colors[colorIndex % colors.length]
      });
      
      // Add feature subset data for alternative model if available
      const modelHasSubset = model.featureSubsetScores || 
                           (model.subsetScores && model.subsetScores.length > 0);
      
      if (modelHasSubset) {
        const subsetScores = model.featureSubsetScores || model.subsetScores;
        
        if (subsetScores && subsetScores.length > 0) {
          seriesData.push({
            name: `${model.name} (Subset)`,
            data: subsetScores,
            color: colors[colorIndex % colors.length],
            dashStyle: 'dash'
          });
        }
      }
      
      colorIndex++;
    });
    
    // Find min and max for y-axis
    const allScores = seriesData.flatMap(series => series.data.filter(Boolean));
    const minScore = Math.min(...allScores);
    const maxScore = Math.max(...allScores);
    
    const lineChartConfig = {
      title: 'Model Comparison by Perturbation Level',
      xAxis: {
        title: 'Perturbation Level',
        categories: comparisonData.primaryModel.levels
      },
      yAxis: {
        title: comparisonData.primaryModel.metric || 'Score',
        min: Math.max(0, minScore * 0.9),
        max: Math.min(1, maxScore * 1.1)
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
    if (!container || !comparisonData) {
      container.innerHTML = "<div class='data-message'>No comparison data available for visualization.</div>";
      return;
    }
    
    const tableData = [];
    
    // Add primary model
    tableData.push(this.prepareModelTableData(comparisonData.primaryModel));
    
    // Add alternative models
    comparisonData.alternativeModels.forEach(model => {
      tableData.push(this.prepareModelTableData(model));
    });
    
    // Check if we have any subset improvement data
    const hasSubsetData = tableData.some(row => row.subsetImprovement !== null);
    
    const html = `
      <table class="data-table">
        <thead>
          <tr>
            <th>Model</th>
            <th>Base Score</th>
            <th>Average Score</th>
            <th>Worst Score</th>
            <th>Impact (%)</th>
            ${hasSubsetData ? '<th>Improvement with Subset</th>' : ''}
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
              ${hasSubsetData ? 
                `<td>${row.subsetImprovement !== null ? formatPercent(row.subsetImprovement) : 'N/A'}</td>` : ''}
            </tr>
          `).join('')}
        </tbody>
      </table>
      <div class="comparison-analysis mt-4">
        <h4>Model Comparison Analysis</h4>
        <p>
          ${this.generateComparisonAnalysis(tableData, hasSubsetData)}
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
    const worstScore = Math.min(...model.worstScores.filter(Boolean));
    const impact = 1 - (avgScore / model.baseScore);
    
    // Check if we have actual subset data
    let subsetImprovement = null;
    
    if (model.featureSubsetScores && model.featureSubsetScores.length > 0) {
      // Calculate actual improvement from feature subset scores
      const avgSubsetScore = this.calculateAverage(model.featureSubsetScores);
      subsetImprovement = (avgSubsetScore - avgScore) / avgScore;
    } else if (model.subsetScores && model.subsetScores.length > 0) {
      // Calculate actual improvement from subset scores
      const avgSubsetScore = this.calculateAverage(model.subsetScores);
      subsetImprovement = (avgSubsetScore - avgScore) / avgScore;
    } else if (model.subsetImprovement !== undefined) {
      // Use explicit improvement value if provided
      subsetImprovement = model.subsetImprovement;
    }
    
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
   * @param {boolean} hasSubsetData - Whether we have feature subset data
   * @return {string} Analysis text
   */
  generateComparisonAnalysis(tableData, hasSubsetData = false) {
    if (!tableData || tableData.length === 0) {
      return "No model comparison data available for analysis.";
    }
    
    // Sort models by average score
    const sortedModels = [...tableData].sort((a, b) => b.avgScore - a.avgScore);
    const bestModel = sortedModels[0];
    const worstModel = sortedModels[sortedModels.length - 1];
    
    if (!hasSubsetData) {
      // No feature subset data available
      return `
        The comparison shows that <strong>${bestModel.name}</strong> has the best overall performance 
        with an average score of ${formatNumber(bestModel.avgScore)}. 
        The model with the lowest performance is <strong>${worstModel.name}</strong> with an 
        average score of ${formatNumber(worstModel.avgScore)}.
      `;
    } else {
      // We have feature subset data
      const modelsWithSubset = tableData.filter(model => model.subsetImprovement !== null);
      
      if (modelsWithSubset.length === 0) {
        // No models have subset data even though the flag was set
        return `
          The comparison shows that <strong>${bestModel.name}</strong> has the best overall performance 
          with an average score of ${formatNumber(bestModel.avgScore)}. 
          No feature subset data is available for improvement analysis.
        `;
      }
      
      // Sort models by subset improvement
      const sortedByImprovement = [...modelsWithSubset].sort((a, b) => b.subsetImprovement - a.subsetImprovement);
      const bestImprovement = sortedByImprovement[0];
      
      return `
        The comparison shows that <strong>${bestModel.name}</strong> has the best overall performance 
        with an average score of ${formatNumber(bestModel.avgScore)}. 
        Models that have feature subset data show varying levels of improvement, 
        with <strong>${bestImprovement.name}</strong> 
        showing the highest improvement of ${formatPercent(bestImprovement.subsetImprovement)}.
      `;
    }
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