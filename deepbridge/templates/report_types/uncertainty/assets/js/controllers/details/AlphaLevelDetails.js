// AlphaLevelDetails.js - placeholder
/**
 * Alpha level details renderer for uncertainty report
 * 
 * Handles rendering of detailed analysis for each alpha level,
 * including metrics, small visualizations, and breakdown tables
 */

class AlphaLevelDetails {
    /**
     * Initialize the alpha level details renderer
     * @param {ChartFactory} chartFactory - Factory for creating charts
     */
    constructor(chartFactory) {
      this.chartFactory = chartFactory;
    }
    
    /**
     * Render all alpha level detail sections
     * @param {HTMLElement} container - The container element
     * @param {Object} uncertaintyData - The uncertainty data
     */
    render(container, uncertaintyData) {
      if (!container || !uncertaintyData) {
        return;
      }
      
      // Sort alpha levels for consistent display
      const alphaLevels = [...uncertaintyData.alphaLevels].sort((a, b) => a - b);
      
      // Build HTML for all alpha level details
      let html = '';
      
      alphaLevels.forEach(alpha => {
        const levelData = uncertaintyData.byAlpha[alpha] || {};
        html += this.renderAlphaLevelSection(alpha, levelData);
      });
      
      // Display message if no alpha levels available
      if (html === '') {
        html = `
          <div class="alert alert-info">
            <strong>No alpha level details available</strong><br>
            The uncertainty data does not contain details for specific alpha levels.
          </div>
        `;
      }
      
      container.innerHTML = html;
      
      // After HTML is added to DOM, create the charts
      this.initializeCharts(container, uncertaintyData);
    }
    
    /**
     * Render a single alpha level detail section
     * @param {number} alpha - Alpha level
     * @param {Object} levelData - Data for this alpha level
     * @return {string} HTML for the alpha level section
     */
    renderAlphaLevelSection(alpha, levelData) {
      const confidence = 1 - alpha;
      const expectedCoverage = confidence;
      const observedCoverage = levelData.coverage || 0;
      const coverageDiff = observedCoverage - expectedCoverage;
      const averageWidth = levelData.averageWidth || 0;
      
      // Determine coverage quality class
      const coverageClass = this.getCoverageClass(observedCoverage, expectedCoverage);
      
      return `
        <div class="alpha-detail-card" data-alpha="${alpha}">
          <h4>Alpha Level: ${alpha.toFixed(2)} (Confidence: ${confidence.toFixed(2)})</h4>
          
          <div class="alpha-metrics">
            <div class="metric-grid">
              <div class="metric-row">
                <div class="metric-label">Expected Coverage:</div>
                <div class="metric-value">${expectedCoverage.toFixed(4)}</div>
              </div>
              <div class="metric-row">
                <div class="metric-label">Observed Coverage:</div>
                <div class="metric-value ${coverageClass}">${observedCoverage.toFixed(4)}</div>
              </div>
              <div class="metric-row">
                <div class="metric-label">Coverage Difference:</div>
                <div class="metric-value ${coverageDiff > 0 ? 'positive' : 'negative'}">${coverageDiff > 0 ? '+' : ''}${coverageDiff.toFixed(4)}</div>
              </div>
              <div class="metric-row">
                <div class="metric-label">Average Interval Width:</div>
                <div class="metric-value">${averageWidth.toFixed(4)}</div>
              </div>
              
              ${this.renderAdditionalMetrics(levelData)}
            </div>
          </div>
          
          <div class="alpha-chart-container">
            <div id="alpha-chart-${alpha.toString().replace('.', '-')}" class="alpha-small-chart" data-alpha="${alpha}"></div>
          </div>
          
          ${this.renderAlphaLevelTable(alpha, levelData)}
        </div>
      `;
    }
    
    /**
     * Render additional metrics if available
     * @param {Object} levelData - Data for this alpha level
     * @return {string} HTML for additional metrics
     */
    renderAdditionalMetrics(levelData) {
      let html = '';
      
      // Add miscoverage metrics if available
      if (levelData.miscoverage !== undefined) {
        html += `
          <div class="metric-row">
            <div class="metric-label">Miscoverage Rate:</div>
            <div class="metric-value">${levelData.miscoverage.toFixed(4)}</div>
          </div>
        `;
      }
      
      // Add sharpness metrics if available
      if (levelData.sharpness !== undefined) {
        html += `
          <div class="metric-row">
            <div class="metric-label">Sharpness:</div>
            <div class="metric-value">${levelData.sharpness.toFixed(4)}</div>
          </div>
        `;
      }
      
      // Add quantile loss metrics if available
      if (levelData.quantileLoss !== undefined) {
        html += `
          <div class="metric-row">
            <div class="metric-label">Quantile Loss:</div>
            <div class="metric-value">${levelData.quantileLoss.toFixed(4)}</div>
          </div>
        `;
      }
      
      return html;
    }
    
    /**
     * Render a detailed breakdown table for an alpha level
     * @param {number} alpha - Alpha level
     * @param {Object} levelData - Data for this alpha level
     * @return {string} HTML for the breakdown table
     */
    renderAlphaLevelTable(alpha, levelData) {
      // If no detailed breakdown available, return empty string
      if (!levelData.details || !levelData.details.categories) {
        return '';
      }
      
      const categories = levelData.details.categories;
      const coverages = levelData.details.coverages || {};
      const widths = levelData.details.widths || {};
      
      let html = `
        <div class="breakdown-table-container">
          <h5>Performance Breakdown</h5>
          <div class="table-wrapper">
            <table class="data-table breakdown-table">
              <thead>
                <tr>
                  <th>Category</th>
                  <th>Coverage</th>
                  <th>Average Width</th>
                  <th>Sample Count</th>
                </tr>
              </thead>
              <tbody>
      `;
      
      // Add rows for each category
      categories.forEach(category => {
        const categoryCoverage = coverages[category] || 0;
        const categoryWidth = widths[category] || 0;
        const sampleCount = levelData.details.counts ? (levelData.details.counts[category] || 0) : 'N/A';
        
        // Determine coverage quality class
        const coverageClass = this.getCoverageClass(categoryCoverage, 1 - alpha);
        
        html += `
          <tr>
            <td>${category}</td>
            <td class="${coverageClass}">${categoryCoverage.toFixed(4)}</td>
            <td>${categoryWidth.toFixed(4)}</td>
            <td>${typeof sampleCount === 'number' ? sampleCount : sampleCount}</td>
          </tr>
        `;
      });
      
      html += `
              </tbody>
            </table>
          </div>
        </div>
      `;
      
      return html;
    }
    
    /**
     * Get a CSS class for coverage quality
     * @param {number} observed - Observed coverage
     * @param {number} expected - Expected coverage
     * @return {string} CSS class name
     */
    getCoverageClass(observed, expected) {
      const diff = Math.abs(observed - expected);
      
      if (diff <= 0.01) {
        return 'excellent';
      } else if (diff <= 0.05) {
        return 'good';
      } else if (diff <= 0.1) {
        return 'moderate';
      } else if (diff <= 0.15) {
        return 'fair';
      } else {
        return 'poor';
      }
    }
    
    /**
     * Initialize charts for each alpha level
     * @param {HTMLElement} container - Parent container element
     * @param {Object} uncertaintyData - The uncertainty data
     */
    initializeCharts(container, uncertaintyData) {
      // Find all alpha chart containers
      const chartContainers = container.querySelectorAll('.alpha-small-chart');
      
      chartContainers.forEach(chartContainer => {
        const alpha = parseFloat(chartContainer.getAttribute('data-alpha'));
        const levelData = uncertaintyData.byAlpha[alpha] || {};
        
        // Create a small visualization for this alpha level
        this.createAlphaLevelChart(chartContainer, alpha, levelData);
      });
    }
    
    /**
     * Create a small chart for an alpha level
     * @param {HTMLElement} container - Chart container element
     * @param {number} alpha - Alpha level
     * @param {Object} levelData - Data for this alpha level
     */
    createAlphaLevelChart(container, alpha, levelData) {
      // If no distribution data available, create a simple coverage comparison
      if (!levelData.distribution) {
        this.createCoverageComparisonChart(container, alpha, levelData);
        return;
      }
      
      // If distribution data is available, create a distribution plot
      this.createDistributionChart(container, alpha, levelData);
    }
    
    /**
     * Create a simple coverage comparison chart
     * @param {HTMLElement} container - Chart container element
     * @param {number} alpha - Alpha level
     * @param {Object} levelData - Data for this alpha level
     */
    createCoverageComparisonChart(container, alpha, levelData) {
      const expectedCoverage = 1 - alpha;
      const observedCoverage = levelData.coverage || 0;
      
      const chartConfig = {
        chart: {
          type: 'bar',
          height: 150
        },
        title: {
          text: 'Coverage Comparison',
          style: { fontSize: '12px' }
        },
        xAxis: {
          categories: ['Expected', 'Observed'],
          title: { text: null }
        },
        yAxis: {
          title: { text: null },
          min: 0,
          max: 1
        },
        legend: { enabled: false },
        series: [{
          name: 'Coverage',
          data: [
            { y: expectedCoverage, color: '#95a5a6' },
            { 
              y: observedCoverage, 
              color: this.getCoverageColorByClass(
                this.getCoverageClass(observedCoverage, expectedCoverage)
              )
            }
          ]
        }],
        plotOptions: {
          bar: {
            dataLabels: {
              enabled: true,
              format: '{y:.2f}'
            }
          }
        }
      };
      
      this.chartFactory.createBarChart(container, chartConfig);
    }
    
    /**
     * Create a distribution chart for an alpha level
     * @param {HTMLElement} container - Chart container element
     * @param {number} alpha - Alpha level
     * @param {Object} levelData - Data for this alpha level
     */
    createDistributionChart(container, alpha, levelData) {
      const distribution = levelData.distribution;
      
      // Create histogram-like data
      const chartConfig = {
        chart: {
          type: 'column',
          height: 150
        },
        title: {
          text: 'Prediction Interval Distribution',
          style: { fontSize: '12px' }
        },
        xAxis: {
          categories: distribution.bins || [],
          title: { text: null }
        },
        yAxis: {
          title: { text: null }
        },
        legend: { enabled: false },
        series: [{
          name: 'Frequency',
          data: distribution.values || []
        }],
        plotOptions: {
          column: {
            pointPadding: 0,
            borderWidth: 0
          }
        }
      };
      
      this.chartFactory.createColumnChart(container, chartConfig);
    }
    
    /**
     * Get a color for a coverage class
     * @param {string} coverageClass - CSS coverage class
     * @return {string} Color hex code
     */
    getCoverageColorByClass(coverageClass) {
      switch (coverageClass) {
        case 'excellent': return '#2ecc71';
        case 'good': return '#3498db';
        case 'moderate': return '#f39c12';
        case 'fair': return '#e67e22';
        case 'poor': return '#e74c3c';
        default: return '#95a5a6';
      }
    }
  }
  
  export default AlphaLevelDetails;