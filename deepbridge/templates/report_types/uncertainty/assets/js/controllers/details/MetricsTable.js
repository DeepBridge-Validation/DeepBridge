// MetricsTable.js - placeholder
/**
 * Metrics table renderer for uncertainty report details
 * 
 * Handles rendering of detailed metric tables showing various
 * uncertainty quantification performance metrics
 */

class MetricsTable {
    /**
     * Initialize the metrics table renderer
     */
    constructor() {
      // Initialize any needed properties
    }
    
    /**
     * Render the uncertainty metrics table
     * @param {HTMLElement} container - The container element
     * @param {Object} uncertaintyData - The uncertainty data
     */
    render(container, uncertaintyData) {
      if (!container || !uncertaintyData) {
        return;
      }
      
      const summary = uncertaintyData.summary || {};
      
      let html = `
        <div class="table-wrapper">
          <table class="data-table metrics-table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Rating</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody>
      `;
      
      // Add overall uncertainty score
      html += this.createMetricRow(
        'Uncertainty Score',
        summary.uncertaintyScore,
        'Overall quality score for uncertainty quantification',
        this.getRatingClass(summary.uncertaintyScore)
      );
      
      // Add coverage metrics
      html += this.createMetricRow(
        'Average Coverage',
        summary.averageCoverage,
        'Average actual coverage across all confidence levels',
        this.getCoverageRatingClass(summary.averageCoverage, summary.expectedCoverage)
      );
      
      html += this.createMetricRow(
        'Expected Coverage',
        summary.expectedCoverage,
        'Target coverage based on confidence levels',
        'neutral'
      );
      
      // Add calibration metrics
      html += this.createMetricRow(
        'Maximum Calibration Error',
        summary.maximumCalibrationError,
        'Largest discrepancy between expected and observed coverage',
        this.getErrorRatingClass(summary.maximumCalibrationError, true)
      );
      
      html += this.createMetricRow(
        'Expected Calibration Error',
        summary.expectedCalibrationError,
        'Average absolute difference between expected and observed coverage',
        this.getErrorRatingClass(summary.expectedCalibrationError)
      );
      
      // Add width metrics
      html += this.createMetricRow(
        'Average Width',
        summary.averageWidth,
        'Average prediction interval width across all confidence levels',
        'neutral'
      );
      
      html += this.createMetricRow(
        'Average Normalized Width',
        summary.averageNormalizedWidth,
        'Average width normalized by the data range',
        this.getWidthRatingClass(summary.averageNormalizedWidth)
      );
      
      // Add sharpness metrics if available
      if (summary.sharpness !== undefined) {
        html += this.createMetricRow(
          'Sharpness',
          summary.sharpness,
          'Measure of prediction interval tightness',
          this.getSharpnessRatingClass(summary.sharpness)
        );
      }
      
      // Add miscoverage metrics
      html += this.createMetricRow(
        'Average Miscoverage',
        summary.averageMiscoverage,
        'Fraction of test samples outside the prediction intervals',
        this.getErrorRatingClass(summary.averageMiscoverage)
      );
      
      // Add any other available metrics
      if (summary.consistencyScore !== undefined) {
        html += this.createMetricRow(
          'Consistency Score',
          summary.consistencyScore,
          'Consistency of coverage across different confidence levels',
          this.getRatingClass(summary.consistencyScore)
        );
      }
      
      html += `
            </tbody>
          </table>
        </div>
        <div class="table-notes">
          <div class="note-item">
            <span class="note-icon">ℹ️</span>
            <span class="note-text">
              These metrics provide a comprehensive assessment of the uncertainty model's performance.
              Good uncertainty quantification should have high coverage, low calibration error, and appropriate interval widths.
            </span>
          </div>
        </div>
      `;
      
      container.innerHTML = html;
    }
    
    /**
     * Create a metric table row
     * @param {string} name - Metric name
     * @param {number} value - Metric value
     * @param {string} description - Metric description
     * @param {string} ratingClass - CSS class for the rating
     * @return {string} HTML for the table row
     */
    createMetricRow(name, value, description, ratingClass) {
      const formattedValue = value !== undefined ? this.formatNumber(value) : 'N/A';
      const ratingText = this.getRatingText(ratingClass);
      
      return `
        <tr>
          <td>${name}</td>
          <td>${formattedValue}</td>
          <td class="${ratingClass}">${ratingText}</td>
          <td>${description}</td>
        </tr>
      `;
    }
    
    /**
     * Format a number for display
     * @param {number} value - The number to format
     * @param {number} precision - Decimal places to display
     * @return {string} Formatted number
     */
    formatNumber(value, precision = 4) {
      if (value === null || value === undefined) {
        return 'N/A';
      }
      return value.toFixed(precision);
    }
    
    /**
     * Get a CSS class for a rating based on score
     * @param {number} score - The score value (0-1)
     * @return {string} CSS rating class
     */
    getRatingClass(score) {
      if (score >= 0.9) {
        return 'excellent';
      } else if (score >= 0.8) {
        return 'good';
      } else if (score >= 0.7) {
        return 'moderate';
      } else if (score >= 0.6) {
        return 'fair';
      } else {
        return 'poor';
      }
    }
    
    /**
     * Get a CSS class for a coverage rating
     * @param {number} observed - Observed coverage
     * @param {number} expected - Expected coverage
     * @return {string} CSS rating class
     */
    getCoverageRatingClass(observed, expected) {
      if (!expected) expected = 0.9; // Default expected coverage
      
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
     * Get a CSS class for an error rating (lower is better)
     * @param {number} error - Error value
     * @param {boolean} isMaxError - Whether this is a maximum error metric
     * @return {string} CSS rating class
     */
    getErrorRatingClass(error, isMaxError = false) {
      const thresholds = isMaxError ? 
        [0.03, 0.07, 0.12, 0.18] : // Stricter thresholds for max error
        [0.02, 0.05, 0.08, 0.12];  // Standard thresholds for average error
      
      if (error <= thresholds[0]) {
        return 'excellent';
      } else if (error <= thresholds[1]) {
        return 'good';
      } else if (error <= thresholds[2]) {
        return 'moderate';
      } else if (error <= thresholds[3]) {
        return 'fair';
      } else {
        return 'poor';
      }
    }
    
    /**
     * Get a CSS class for a width rating (context-dependent)
     * @param {number} width - Width value
     * @return {string} CSS rating class
     */
    getWidthRatingClass(width) {
      // For width metrics, the interpretation is context-dependent
      // A narrow width is good if coverage is maintained
      // This is a simplified approach
      if (width <= 0.1) {
        return 'excellent';
      } else if (width <= 0.2) {
        return 'good';
      } else if (width <= 0.3) {
        return 'moderate';
      } else if (width <= 0.4) {
        return 'fair';
      } else {
        return 'poor';
      }
    }
    
    /**
     * Get a CSS class for a sharpness rating
     * @param {number} sharpness - Sharpness value
     * @return {string} CSS rating class
     */
    getSharpnessRatingClass(sharpness) {
      // For sharpness, higher values are generally better
      if (sharpness >= 0.9) {
        return 'excellent';
      } else if (sharpness >= 0.8) {
        return 'good';
      } else if (sharpness >= 0.7) {
        return 'moderate';
      } else if (sharpness >= 0.6) {
        return 'fair';
      } else {
        return 'poor';
      }
    }
    
    /**
     * Get rating text based on rating class
     * @param {string} ratingClass - CSS rating class
     * @return {string} Rating text
     */
    getRatingText(ratingClass) {
      switch (ratingClass) {
        case 'excellent': return 'Excellent';
        case 'good': return 'Good';
        case 'moderate': return 'Moderate';
        case 'fair': return 'Fair';
        case 'poor': return 'Poor';
        default: return 'N/A';
      }
    }
  }
  
  export default MetricsTable;