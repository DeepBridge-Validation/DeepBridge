/**
 * Performance gap charts renderer
 * 
 * Handles rendering of performance gap charts for the overview section,
 * including performance gaps, shift intensity, and feature impact visualizations
 */

class PerformanceGapCharts {
    /**
     * Initialize the performance gap charts renderer
     * @param {ChartFactory} chartFactory - Factory for creating charts
     */
    constructor(chartFactory) {
      this.chartFactory = chartFactory;
    }
    
    /**
     * Render the main performance gap chart
     * @param {HTMLElement} container - The container element
     * @param {Object} resilienceData - Resilience data with shift results
     */
    renderPerformanceGapChart(container, resilienceData) {
      if (!container || !resilienceData) return;
      
      const chartConfig = {
        title: 'Performance Gap Across Shifts',
        xAxis: {
          title: 'Shift Type',
          categories: resilienceData.shifts.map(shift => shift.type)
        },
        yAxis: {
          title: resilienceData.metric || 'Score',
          min: Math.max(0, Math.min(...resilienceData.shifts.map(shift => shift.score)) - 0.1),
          max: Math.min(1, Math.max(resilienceData.baseScore, ...resilienceData.shifts.map(shift => shift.score)) + 0.05)
        },
        series: [
          {
            name: resilienceData.modelName || 'Primary Model',
            data: resilienceData.shifts.map(shift => shift.score),
            color: '#1b78de'
          }
        ],
        referenceLine: {
          value: resilienceData.baseScore,
          name: 'Base Score',
          color: '#2ecc71'
        }
      };
      
      // Add alternative models if available
      if (resilienceData.alternativeModels) {
        this.addAlternativeModels(chartConfig, resilienceData);
      }
      
      // Create the chart
      this.chartFactory.createBarChart(container, chartConfig);
    }
    
    /**
     * Render the shift intensity chart
     * @param {HTMLElement} container - The container element
     * @param {Object} resilienceData - Resilience data with shift results
     */
    renderIntensityChart(container, resilienceData) {
      if (!container || !resilienceData) return;
      
      // Prepare data for intensity chart
      const intensityData = resilienceData.shifts.map(shift => ({
        name: shift.type,
        y: shift.intensity || 0.5, // Default to 0.5 if not available
        impact: (resilienceData.baseScore - shift.score) / resilienceData.baseScore,
        color: this.getIntensityColor(shift.intensity || 0.5)
      }));
      
      const chartConfig = {
        title: 'Shift Intensity vs Performance Impact',
        xAxis: {
          title: 'Shift Type',
          categories: resilienceData.shifts.map(shift => shift.type)
        },
        yAxis: {
          title: 'Intensity / Impact',
          min: 0,
          max: 1
        },
        series: [
          {
            name: 'Shift Intensity',
            data: intensityData.map(item => ({
              y: item.y,
              color: item.color
            })),
            type: 'column'
          },
          {
            name: 'Performance Impact',
            data: intensityData.map(item => item.impact),
            type: 'line',
            color: '#e74c3c'
          }
        ],
        tooltipFormatter: function() {
          const shift = this.x;
          const isIntensity = this.series.name === 'Shift Intensity';
          const value = isIntensity ? intensityData.find(item => item.name === shift).y : 
                                    intensityData.find(item => item.name === shift).impact;
          
          return `
            <strong>${shift}</strong><br>
            ${this.series.name}: <b>${(value * 100).toFixed(2)}%</b>
          `;
        }
      };
      
      // Create the chart
      this.chartFactory.createCombinationChart(container, chartConfig);
    }
    
    /**
     * Render the feature impact chart
     * @param {HTMLElement} container - The container element
     * @param {Object} resilienceData - Resilience data with feature impact information
     */
    renderFeatureImpactChart(container, resilienceData) {
      if (!container || !resilienceData || !resilienceData.featureImpact) return;
      
      // Get top features by impact
      const featureImpact = resilienceData.featureImpact;
      const topFeatures = Object.entries(featureImpact)
        .map(([feature, impact]) => ({ feature, impact }))
        .sort((a, b) => b.impact - a.impact)
        .slice(0, 10); // Show top 10 features
      
      const chartConfig = {
        title: 'Feature Impact on Resilience',
        xAxis: {
          title: 'Impact Score'
        },
        yAxis: {
          title: 'Feature',
          categories: topFeatures.map(item => item.feature)
        },
        series: [{
          name: 'Impact',
          data: topFeatures.map(item => ({
            y: item.feature,
            x: item.impact,
            color: this.getImpactColor(item.impact)
          }))
        }],
        tooltipFormatter: function() {
          const feature = this.point.y;
          const impact = this.point.x;
          
          return `
            <strong>${feature}</strong><br>
            Impact: <b>${impact.toFixed(4)}</b><br>
            Higher values indicate greater sensitivity to distribution shifts
          `;
        }
      };
      
      // Create the chart
      this.chartFactory.createHorizontalBarChart(container, chartConfig);
    }
    
    /**
     * Add alternative models to the chart configuration
     * @param {Object} chartConfig - Chart configuration object
     * @param {Object} resilienceData - Resilience data with alternative models
     */
    addAlternativeModels(chartConfig, resilienceData) {
      const colors = ['#e41a1c', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf'];
      let colorIndex = 0;
      
      Object.entries(resilienceData.alternativeModels).forEach(([name, model]) => {
        chartConfig.series.push({
          name: name,
          data: model.shifts.map(shift => shift.score),
          color: colors[colorIndex % colors.length]
        });
        
        colorIndex++;
      });
    }
    
    /**
     * Get a color for an intensity value
     * @param {number} intensity - The intensity value (0-1)
     * @return {string} Color string
     */
    getIntensityColor(intensity) {
      // Generate color from green to red based on intensity
      const red = Math.round(255 * intensity);
      const green = Math.round(255 * (1 - intensity));
      return `rgb(${red}, ${green}, 50)`;
    }
    
    /**
     * Get a color for an impact value
     * @param {number} impact - The impact value (0-1)
     * @return {string} Color string
     */
    getImpactColor(impact) {
      // Generate color from blue to orange based on impact
      if (impact < 0.33) {
        return 'rgba(46, 204, 113, 0.8)'; // Green for low impact
      } else if (impact < 0.66) {
        return 'rgba(241, 196, 15, 0.8)'; // Yellow for medium impact
      } else {
        return 'rgba(231, 76, 60, 0.8)'; // Red for high impact
      }
    }
  }
  
  export default PerformanceGapCharts;