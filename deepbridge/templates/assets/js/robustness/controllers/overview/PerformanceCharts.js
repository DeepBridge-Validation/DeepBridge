/**
 * Performance charts renderer
 * 
 * Handles rendering of performance charts for the overview section,
 * including regular perturbation chart, worst score chart, and mean score chart
 */
class PerformanceCharts {
    /**
     * Initialize the performance charts renderer
     * @param {ChartFactory} chartFactory - Factory for creating charts
     */
    constructor(chartFactory) {
      this.chartFactory = chartFactory;
    }
    
    /**
     * Render the main perturbation chart showing performance across perturbation levels
     * @param {HTMLElement} container - The container element
     * @param {Object} perturbationData - Perturbation data with scores and levels
     */
    renderPerturbationChart(container, perturbationData) {
      if (!container || !perturbationData) return;
      
      const chartConfig = {
        title: 'Performance Under Perturbation',
        xAxis: {
          title: 'Perturbation Level',
          categories: perturbationData.levels
        },
        yAxis: {
          title: perturbationData.metric || 'Score',
          min: Math.max(0, Math.min(...perturbationData.scores) - 0.1),
          max: Math.min(1, Math.max(perturbationData.baseScore, ...perturbationData.scores) + 0.05)
        },
        series: [
          {
            name: perturbationData.modelName,
            data: perturbationData.scores,
            color: '#1b78de'
          }
        ],
        referenceLine: {
          value: perturbationData.baseScore,
          name: 'Base Score',
          color: '#2ecc71'
        }
      };
      
      // Add alternative models if available
      if (perturbationData.alternativeModels) {
        this.addAlternativeModels(chartConfig, perturbationData);
      }
      
      // Create the chart
      this.chartFactory.createLineChart(container, chartConfig);
    }
    
    /**
     * Render the worst score chart showing worst-case performance
     * @param {HTMLElement} container - The container element
     * @param {Object} perturbationData - Perturbation data with worst scores and levels
     */
    renderWorstScoreChart(container, perturbationData) {
      if (!container || !perturbationData) return;
      
      const chartConfig = {
        title: 'Worst-Case Performance Under Perturbation',
        xAxis: {
          title: 'Perturbation Level',
          categories: perturbationData.levels
        },
        yAxis: {
          title: perturbationData.metric || 'Score',
          min: Math.max(0, Math.min(...perturbationData.worstScores) - 0.1),
          max: Math.min(1, Math.max(perturbationData.baseScore, ...perturbationData.worstScores) + 0.05)
        },
        series: [
          {
            name: perturbationData.modelName,
            data: perturbationData.worstScores,
            color: '#1b78de'
          }
        ],
        referenceLine: {
          value: perturbationData.baseScore,
          name: 'Base Score',
          color: '#2ecc71'
        },
        thresholdLine: {
          value: 0.75,
          name: 'Acceptable Threshold',
          color: '#e74c3c'
        }
      };
      
      // Add alternative models if available
      if (perturbationData.alternativeModels) {
        this.addAlternativeModelsWorstScores(chartConfig, perturbationData);
      }
      
      // Create the chart
      this.chartFactory.createLineChart(container, chartConfig);
    }
    
    /**
     * Render the mean score chart showing average performance with confidence intervals
     * @param {HTMLElement} container - The container element
     * @param {Object} perturbationData - Perturbation data with scores and levels
     */
    renderMeanScoreChart(container, perturbationData) {
      if (!container || !perturbationData) return;
      
      // Create a modified dataset with confidence intervals
      const seriesData = [];
      
      // Add primary model
      seriesData.push({
        name: perturbationData.modelName,
        data: perturbationData.scores,
        color: '#1b78de',
        // Add simulated confidence interval of +/- 0.05 (or extract from actual data if available)
        confidenceInterval: perturbationData.scores.map(score => [
          Math.max(0, score - 0.05),
          Math.min(1, score + 0.05)
        ])
      });
      
      // Add alternative models if available
      if (perturbationData.alternativeModels) {
        this.addAlternativeModelsWithConfidence(seriesData, perturbationData);
      }
      
      const chartConfig = {
        title: 'Mean Performance with Confidence Intervals',
        xAxis: {
          title: 'Perturbation Level',
          categories: perturbationData.levels
        },
        yAxis: {
          title: perturbationData.metric || 'Score',
          min: Math.max(0, Math.min(...perturbationData.scores) - 0.1),
          max: Math.min(1, Math.max(perturbationData.baseScore, ...perturbationData.scores) + 0.05)
        },
        series: seriesData,
        showConfidenceIntervals: true,
        referenceLine: {
          value: perturbationData.baseScore,
          name: 'Base Score',
          color: '#2ecc71'
        }
      };
      
      // Create the chart
      this.chartFactory.createAreaChart(container, chartConfig);
    }
    
    /**
     * Add alternative models to the chart configuration
     * @param {Object} chartConfig - Chart configuration object
     * @param {Object} perturbationData - Perturbation data with alternative models
     */
    addAlternativeModels(chartConfig, perturbationData) {
      const colors = ['#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf'];
      let colorIndex = 0;
      
      Object.entries(perturbationData.alternativeModels).forEach(([name, model]) => {
        chartConfig.series.push({
          name: name,
          data: model.scores,
          color: colors[colorIndex % colors.length]
        });
        
        colorIndex++;
      });
    }
    
    /**
     * Add alternative models' worst scores to the chart configuration
     * @param {Object} chartConfig - Chart configuration object
     * @param {Object} perturbationData - Perturbation data with alternative models
     */
    addAlternativeModelsWorstScores(chartConfig, perturbationData) {
      const colors = ['#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf'];
      let colorIndex = 0;
      
      Object.entries(perturbationData.alternativeModels).forEach(([name, model]) => {
        if (model.worstScores) {
          chartConfig.series.push({
            name: name,
            data: model.worstScores,
            color: colors[colorIndex % colors.length]
          });
          
          colorIndex++;
        }
      });
    }
    
    /**
     * Add alternative models with confidence intervals to series data
     * @param {Array} seriesData - Series data array
     * @param {Object} perturbationData - Perturbation data with alternative models
     */
    addAlternativeModelsWithConfidence(seriesData, perturbationData) {
      const colors = ['#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf'];
      let colorIndex = 0;
      
      Object.entries(perturbationData.alternativeModels).forEach(([name, model]) => {
        seriesData.push({
          name: name,
          data: model.scores,
          color: colors[colorIndex % colors.length],
          confidenceInterval: model.scores.map(score => [
            Math.max(0, score - 0.05),
            Math.min(1, score + 0.05)
          ])
        });
        
        colorIndex++;
      });
    }
  }
  
  export default PerformanceCharts;