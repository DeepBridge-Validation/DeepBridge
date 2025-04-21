/**
 * Feature comparison chart renderer
 * 
 * Handles rendering of feature comparison charts including bar, scatter,
 * radar, heatmap, and ranking visualizations
 */
import { formatNumber } from '../../utils/Formatters.js';

class ComparisonCharts {
  /**
   * Initialize the comparison charts renderer
   * @param {ChartFactory} chartFactory - Factory for creating charts
   */
  constructor(chartFactory) {
    this.chartFactory = chartFactory;
  }
  
  /**
   * Render bar comparison chart
   * @param {HTMLElement} container - The container element
   * @param {Array} combinedData - Combined feature importance data
   */
  renderBarChart(container, combinedData) {
    if (!container) return;
    
    // Get top 10 features by model importance
    const topFeatures = combinedData
      .sort((a, b) => b.modelImportance - a.modelImportance)
      .slice(0, 10);
    
    const chartConfig = {
      title: 'Feature Importance Comparison (Top 10 Features)',
      xAxis: {
        title: 'Feature',
        categories: topFeatures.map(item => item.feature)
      },
      yAxis: {
        title: 'Importance Score',
        min: 0,
        max: 1
      },
      series: [
        {
          name: 'Model Importance',
          data: topFeatures.map(item => ({
            y: item.modelImportance,
            color: item.inSubset ? 'rgba(27, 120, 222, 0.9)' : 'rgba(27, 120, 222, 0.5)'
          }))
        },
        {
          name: 'Robustness Importance',
          data: topFeatures.map(item => ({
            y: item.robustnessImportance,
            color: item.inSubset ? 'rgba(46, 204, 113, 0.9)' : 'rgba(46, 204, 113, 0.5)'
          }))
        }
      ],
      tooltipFormatter: function() {
        const feature = this.series.chart.xAxis[0].categories[this.point.x];
        const importance = this.point.y;
        const type = this.series.name;
        const inSubset = topFeatures.find(f => f.feature === feature).inSubset;
        
        return `
          <strong>${feature}</strong><br>
          ${type}: <b>${importance.toFixed(4)}</b><br>
          ${inSubset ? '<b>In optimal feature subset</b>' : 'Not in optimal subset'}
        `;
      }
    };
    
    // Create the chart
    this.chartFactory.createBarChart(container, chartConfig);
  }
  
  /**
   * Render scatter comparison chart
   * @param {HTMLElement} container - The container element
   * @param {Array} combinedData - Combined feature importance data
   */
  renderScatterChart(container, combinedData) {
    if (!container) return;
    
    // Calculate means for quadrant lines
    const xValues = combinedData.map(item => item.modelImportance);
    const yValues = combinedData.map(item => item.robustnessImportance);
    
    const xMean = xValues.reduce((sum, val) => sum + val, 0) / xValues.length;
    const yMean = yValues.reduce((sum, val) => sum + val, 0) / yValues.length;
    
    const chartConfig = {
      title: 'Model vs. Robustness Feature Importance',
      xAxis: {
        title: 'Model Feature Importance',
        min: 0,
        max: Math.max(...xValues) * 1.1
      },
      yAxis: {
        title: 'Robustness Feature Importance',
        min: 0,
        max: Math.max(...yValues) * 1.1
      },
      series: [{
        name: 'Features',
        data: combinedData.map(item => ({
          x: item.modelImportance,
          y: item.robustnessImportance,
          name: item.feature,
          color: item.inSubset ? 'rgba(255, 115, 0, 1.0)' : 'rgba(135, 155, 188, 0.7)',
          marker: {
            radius: item.inSubset ? 7 : 5
          }
        }))
      }],
      quadrantLines: {
        xValue: xMean,
        yValue: yMean
      },
      quadrantLabels: [
        { text: 'Low Importance,<br>Low Robustness', position: { x: xMean / 2, y: yMean / 2 } },
        { text: 'High Importance,<br>Low Robustness', position: { x: (Math.max(...xValues) + xMean) / 2, y: yMean / 2 } },
        { text: 'Low Importance,<br>High Robustness', position: { x: xMean / 2, y: (Math.max(...yValues) + yMean) / 2 } },
        { text: 'High Importance,<br>High Robustness', position: { x: (Math.max(...xValues) + xMean) / 2, y: (Math.max(...yValues) + yMean) / 2 } }
      ],
      tooltipFormatter: function() {
        return `
          <strong>${this.point.name}</strong><br>
          Model Importance: <b>${this.point.x.toFixed(4)}</b><br>
          Robustness Importance: <b>${this.point.y.toFixed(4)}</b><br>
          ${this.point.inSubset ? '<b>In optimal feature subset</b>' : 'Not in optimal subset'}
        `;
      }
    };
    
    // Create the chart
    this.chartFactory.createScatterChart(container, chartConfig);
  }
  
  /**
   * Render radar comparison chart
   * @param {HTMLElement} container - The container element
   * @param {Array} combinedData - Combined feature importance data
   */
  renderRadarChart(container, combinedData) {
    if (!container) return;
    
    // Get top features by combined importance
    const topFeatures = combinedData
      .map(item => ({
        ...item,
        combinedImportance: item.modelImportance + item.robustnessImportance
      }))
      .sort((a, b) => b.combinedImportance - a.combinedImportance)
      .slice(0, 8);
    
    // Extract data for chart
    const features = topFeatures.map(item => item.feature);
    const modelImportance = topFeatures.map(item => item.modelImportance);
    const robustnessImportance = topFeatures.map(item => item.robustnessImportance);
    
    const chartConfig = {
      title: 'Feature Importance Radar Chart (Top 8 Features)',
      categories: features,
      series: [
        {
          name: 'Model Importance',
          data: modelImportance,
          color: 'rgba(27, 120, 222, 0.7)'
        },
        {
          name: 'Robustness Importance',
          data: robustnessImportance,
          color: 'rgba(46, 204, 113, 0.7)'
        }
      ]
    };
    
    // Create the chart
    this.chartFactory.createRadarChart(container, chartConfig);
  }
  
  /**
   * Render heatmap comparison chart
   * @param {HTMLElement} container - The container element
   * @param {Array} combinedData - Combined feature importance data
   */
  renderHeatmapChart(container, combinedData) {
    if (!container) return;
    
    // Get top features by combined importance
    const topFeatures = combinedData
      .map(item => ({
        ...item,
        combinedImportance: item.modelImportance + item.robustnessImportance
      }))
      .sort((a, b) => b.combinedImportance - a.combinedImportance)
      .slice(0, 15);
    
    // Create 2D data array for heatmap (transpose for correct visual display)
    const data = [
      topFeatures.map(item => item.modelImportance),
      topFeatures.map(item => item.robustnessImportance)
    ];
    
    const chartConfig = {
      title: 'Feature Importance Heatmap',
      xAxis: {
        title: 'Feature',
        categories: topFeatures.map(item => item.feature)
      },
      yAxis: {
        title: 'Importance Type',
        categories: ['Model Importance', 'Robustness Importance']
      },
      colorAxis: {
        min: 0,
        max: Math.max(
          ...topFeatures.map(item => item.modelImportance),
          ...topFeatures.map(item => item.robustnessImportance)
        )
      },
      data: data,
      dataLabels: {
        enabled: true,
        formatter: function() {
          return this.point.value.toFixed(2);
        }
      },
      annotations: topFeatures.filter(item => item.inSubset).map(item => ({
        x: topFeatures.map(f => f.feature).indexOf(item.feature),
        y: 0.5,
        marker: {
          symbol: 'star',
          fillColor: 'rgba(255, 255, 255, 0.8)',
          radius: 8
        }
      }))
    };
    
    // Create the chart
    this.chartFactory.createHeatmapChart(container, chartConfig);
  }
  
  /**
   * Render ranking comparison chart
   * @param {HTMLElement} container - The container element
   * @param {Array} combinedData - Combined feature importance data
   */
  renderRankingChart(container, combinedData) {
    if (!container) return;
    
    // Calculate rankings for model and robustness importance
    const withRankings = combinedData.map(item => ({
      ...item,
      modelRank: 0,
      robustnessRank: 0
    }));
    
    // Assign model ranks
    withRankings
      .sort((a, b) => b.modelImportance - a.modelImportance)
      .forEach((item, index) => {
        item.modelRank = index + 1;
      });
    
    // Assign robustness ranks
    withRankings
      .sort((a, b) => b.robustnessImportance - a.robustnessImportance)
      .forEach((item, index) => {
        item.robustnessRank = index + 1;
      });
    
    // Calculate rank differences and sort by difference
    const withDifferences = withRankings.map(item => ({
      ...item,
      rankDiff: Math.abs(item.modelRank - item.robustnessRank)
    }));
    
    // Get top features by rank difference
    const topByDifference = withDifferences
      .sort((a, b) => b.rankDiff - a.rankDiff)
      .slice(0, 15);
    
    // Prepare data for chart
    const features = topByDifference.map(item => item.feature);
    const modelRanks = topByDifference.map(item => item.modelRank);
    const robustnessRanks = topByDifference.map(item => item.robustnessRank);
    
    const chartConfig = {
      title: 'Feature Importance Ranking Comparison',
      xAxis: {
        title: 'Feature',
        categories: features
      },
      yAxis: {
        title: 'Rank (1 = highest importance)',
        reversed: true,
        min: 1,
        max: Math.max(
          ...topByDifference.map(item => item.modelRank),
          ...topByDifference.map(item => item.robustnessRank)
        ) + 1
      },
      series: [
        {
          name: 'Model Importance Rank',
          data: modelRanks.map((rank, index) => ({
            y: rank,
            marker: {
              radius: topByDifference[index].inSubset ? 7 : 5
            }
          })),
          color: 'rgba(27, 120, 222, 0.8)'
        },
        {
          name: 'Robustness Importance Rank',
          data: robustnessRanks.map((rank, index) => ({
            y: rank,
            marker: {
              radius: topByDifference[index].inSubset ? 7 : 5
            }
          })),
          color: 'rgba(46, 204, 113, 0.8)'
        }
      ],
      connectingLines: topByDifference.map((item, index) => ({
        x: index,
        y1: item.modelRank,
        y2: item.robustnessRank,
        color: this.getRankDiffColor(item.rankDiff)
      }))
    };
    
    // Create the chart
    this.chartFactory.createRankingChart(container, chartConfig);
  }
  
  /**
   * Get a color for a rank difference
   * @param {number} diff - The rank difference
   * @return {string} Color string
   */
  getRankDiffColor(diff) {
    if (diff > 10) {
      return 'rgba(231, 76, 60, 0.4)';  // Red for large differences
    } else if (diff > 5) {
      return 'rgba(241, 196, 15, 0.4)'; // Yellow for medium differences
    } else {
      return 'rgba(46, 204, 113, 0.2)'; // Green for small differences
    }
  }
}

export default ComparisonCharts;