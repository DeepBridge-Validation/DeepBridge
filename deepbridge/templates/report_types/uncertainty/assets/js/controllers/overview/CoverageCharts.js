// CoverageCharts.js - placeholder
/**
 * Coverage charts renderer
 * 
 * Handles rendering of coverage-related charts for the overview section,
 * including alpha coverage bar charts and comparative visualizations
 */
class CoverageCharts {
    /**
     * Initialize the coverage charts renderer
     * @param {ChartFactory} chartFactory - Factory for creating charts
     */
    constructor(chartFactory) {
      this.chartFactory = chartFactory;
    }
    
    /**
     * Render alpha coverage chart showing coverage by confidence level
     * @param {HTMLElement} container - The container element
     * @param {Object} uncertaintyData - The uncertainty data
     */
    renderAlphaCoverageChart(container, uncertaintyData) {
      if (!container || !uncertaintyData) return;
      
      // Sort alpha levels to ensure proper ordering
      const alphaLevels = [...uncertaintyData.alphaLevels].sort((a, b) => a - b);
      
      // Prepare data arrays for the chart
      const expectedCoverage = alphaLevels.map(alpha => 1 - alpha);
      const observedCoverage = alphaLevels.map(alpha => 
        uncertaintyData.byAlpha[alpha] ? uncertaintyData.byAlpha[alpha].coverage : null
      );
      
      // Format categories for display (confidence levels)
      const categories = alphaLevels.map(alpha => {
        const confidence = 1 - alpha;
        return `${(confidence * 100).toFixed(0)}%`;
      });
      
      // Calculate coverage difference to determine colors
      const coverageDiff = observedCoverage.map((observed, i) => {
        if (observed === null) return null;
        return Math.abs(observed - expectedCoverage[i]);
      });
      
      // Get colors based on difference
      const barColors = coverageDiff.map(diff => {
        if (diff === null) return '#cccccc';
        if (diff <= 0.01) return '#2ecc71'; // Excellent
        if (diff <= 0.05) return '#3498db'; // Good
        if (diff <= 0.1) return '#f39c12';  // Moderate
        if (diff <= 0.15) return '#e67e22'; // Fair
        return '#e74c3c';                   // Poor
      });
      
      const chartConfig = {
        title: 'Coverage by Confidence Level',
        xAxis: {
          title: 'Confidence Level',
          categories: categories
        },
        yAxis: {
          title: 'Coverage',
          min: 0,
          max: 1
        },
        series: [
          {
            name: 'Observed Coverage',
            data: observedCoverage.map((value, i) => ({
              y: value,
              color: barColors[i]
            }))
          },
          {
            name: 'Expected Coverage',
            type: 'line',
            data: expectedCoverage,
            color: '#e74c3c',
            marker: {
              enabled: true,
              symbol: 'diamond',
              radius: 4
            }
          }
        ],
        tooltipFormatter: function() {
          const alpha = alphaLevels[this.point.x];
          const confidence = 1 - alpha;
          const expected = expectedCoverage[this.point.x];
          const observed = this.point.y;
          
          if (this.series.name === 'Expected Coverage') {
            return `
              <strong>Confidence Level: ${(confidence * 100).toFixed(0)}%</strong><br>
              (α = ${alpha})<br>
              Expected Coverage: <b>${expected.toFixed(4)}</b>
            `;
          } else {
            const diff = observed - expected;
            const diffText = diff >= 0 ? `+${diff.toFixed(4)}` : diff.toFixed(4);
            return `
              <strong>Confidence Level: ${(confidence * 100).toFixed(0)}%</strong><br>
              (α = ${alpha})<br>
              Observed Coverage: <b>${observed.toFixed(4)}</b><br>
              Expected Coverage: <b>${expected.toFixed(4)}</b><br>
              Difference: <b>${diffText}</b>
            `;
          }
        }
      };
      
      // Create the chart
      this.chartFactory.createBarChart(container, chartConfig);
    }
    
    /**
     * Render a method comparison chart showing coverage across different methods
     * @param {HTMLElement} container - The container element
     * @param {Object} uncertaintyData - The uncertainty data
     */
    renderMethodComparisonChart(container, uncertaintyData) {
      if (!container || !uncertaintyData || !uncertaintyData.methodComparison) return;
      
      const comparison = uncertaintyData.methodComparison;
      
      // Extract methods and alpha levels
      const methods = Object.keys(comparison);
      const alphaLevels = [...uncertaintyData.alphaLevels].sort((a, b) => a - b);
      
      // Format categories for display (confidence levels)
      const categories = alphaLevels.map(alpha => {
        const confidence = 1 - alpha;
        return `${(confidence * 100).toFixed(0)}%`;
      });
      
      // Prepare series data
      const series = methods.map(method => {
        return {
          name: method,
          data: alphaLevels.map(alpha => {
            if (!comparison[method] || !comparison[method][alpha]) return null;
            return comparison[method][alpha].coverage;
          })
        };
      });
      
      // Add expected coverage as reference line
      series.push({
        name: 'Expected',
        type: 'line',
        data: alphaLevels.map(alpha => 1 - alpha),
        dashStyle: 'dash',
        color: '#95a5a6',
        marker: {
          enabled: false
        }
      });
      
      const chartConfig = {
        title: 'Coverage Comparison Across Methods',
        xAxis: {
          title: 'Confidence Level',
          categories: categories
        },
        yAxis: {
          title: 'Coverage',
          min: 0,
          max: 1
        },
        series: series,
        tooltipFormatter: function() {
          const alpha = alphaLevels[this.point.x];
          const confidence = 1 - alpha;
          const expected = 1 - alpha;
          
          return `
            <strong>${this.series.name}</strong><br>
            Confidence Level: ${(confidence * 100).toFixed(0)}%<br>
            (α = ${alpha})<br>
            Coverage: <b>${this.point.y.toFixed(4)}</b><br>
            ${this.series.name !== 'Expected' ? `Expected: <b>${expected.toFixed(4)}</b>` : ''}
          `;
        }
      };
      
      // Create the chart
      this.chartFactory.createLineChart(container, chartConfig);
    }
    
    /**
     * Render prediction interval width distribution
     * @param {HTMLElement} container - The container element
     * @param {Object} uncertaintyData - The uncertainty data
     * @param {number} alpha - Specific alpha level to show (optional)
     */
    renderWidthDistributionChart(container, uncertaintyData, alpha) {
      if (!container || !uncertaintyData) return;
      
      // If no specific alpha provided, use the first available
      if (alpha === undefined) {
        alpha = uncertaintyData.alphaLevels[0];
      }
      
      const levelData = uncertaintyData.byAlpha[alpha];
      if (!levelData || !levelData.widthDistribution) {
        container.innerHTML = `
          <div class="alert alert-info">
            <strong>No width distribution data available</strong><br>
            Width distribution data is not available for this alpha level.
          </div>
        `;
        return;
      }
      
      const distribution = levelData.widthDistribution;
      
      const chartConfig = {
        title: `Prediction Interval Width Distribution (α=${alpha})`,
        xAxis: {
          title: 'Interval Width',
          categories: distribution.bins || []
        },
        yAxis: {
          title: 'Frequency'
        },
        series: [
          {
            name: 'Frequency',
            data: distribution.values || [],
            color: '#9b59b6'
          }
        ]
      };
      
      // Create the chart
      this.chartFactory.createColumnChart(container, chartConfig);
    }
  }
  
  export default CoverageCharts;