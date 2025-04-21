/**
 * Formatters
 * 
 * Text and number formatters for displaying uncertainty metrics
 * and statistics in a human-readable format.
 */

class Formatters {
    /**
     * Initialize the formatters
     */
    constructor() {
      // Default formatting options
      this.defaultOptions = {
        precision: 4,           // Default decimal places for numbers
        percentPrecision: 2,    // Default decimal places for percentages
        confidencePrecision: 2, // Default decimal places for confidence intervals
        useCurrency: false      // Whether to use currency formatting by default
      };
    }
    
    /**
     * Format a number with specified precision
     * @param {number} value - Number to format
     * @param {number} precision - Decimal places (default: 4)
     * @param {boolean} useGrouping - Use thousand separators (default: true)
     * @return {string} Formatted number
     */
    formatNumber(value, precision = this.defaultOptions.precision, useGrouping = true) {
      if (value === null || value === undefined || isNaN(value)) {
        return '-';
      }
      
      return value.toLocaleString('en-US', {
        minimumFractionDigits: precision,
        maximumFractionDigits: precision,
        useGrouping: useGrouping
      });
    }
    
    /**
     * Format a number as percentage
     * @param {number} value - Number to format (0-1)
     * @param {number} precision - Decimal places (default: 2)
     * @return {string} Formatted percentage
     */
    formatPercent(value, precision = this.defaultOptions.percentPrecision) {
      if (value === null || value === undefined || isNaN(value)) {
        return '-';
      }
      
      return `${(value * 100).toLocaleString('en-US', {
        minimumFractionDigits: precision,
        maximumFractionDigits: precision
      })}%`;
    }
    
    /**
     * Format a confidence interval
     * @param {number} lower - Lower bound
     * @param {number} upper - Upper bound
     * @param {number} precision - Decimal places (default: 2)
     * @param {boolean} usePercent - Format as percentages (default: false)
     * @return {string} Formatted confidence interval
     */
    formatConfidenceInterval(lower, upper, precision = this.defaultOptions.confidencePrecision, usePercent = false) {
      if (lower === null || upper === null || 
          lower === undefined || upper === undefined || 
          isNaN(lower) || isNaN(upper)) {
        return '-';
      }
      
      if (usePercent) {
        const formattedLower = this.formatPercent(lower, precision);
        const formattedUpper = this.formatPercent(upper, precision);
        return `[${formattedLower}, ${formattedUpper}]`;
      } else {
        const formattedLower = this.formatNumber(lower, precision);
        const formattedUpper = this.formatNumber(upper, precision);
        return `[${formattedLower}, ${formattedUpper}]`;
      }
    }
    
    /**
     * Format a number as a currency
     * @param {number} value - Number to format
     * @param {string} currency - Currency code (default: 'USD')
     * @param {number} precision - Decimal places (default: 2)
     * @return {string} Formatted currency
     */
    formatCurrency(value, currency = 'USD', precision = 2) {
      if (value === null || value === undefined || isNaN(value)) {
        return '-';
      }
      
      return value.toLocaleString('en-US', {
        style: 'currency',
        currency: currency,
        minimumFractionDigits: precision,
        maximumFractionDigits: precision
      });
    }
    
    /**
     * Format a value with a unit
     * @param {number} value - Number to format
     * @param {string} unit - Unit to append
     * @param {number} precision - Decimal places (default: 2)
     * @return {string} Formatted value with unit
     */
    formatWithUnit(value, unit, precision = 2) {
      if (value === null || value === undefined || isNaN(value)) {
        return '-';
      }
      
      return `${this.formatNumber(value, precision)} ${unit}`;
    }
    
    /**
     * Format a p-value
     * @param {number} pValue - P-value to format
     * @param {number} threshold - Significance threshold (default: 0.001)
     * @return {string} Formatted p-value
     */
    formatPValue(pValue, threshold = 0.001) {
      if (pValue === null || pValue === undefined || isNaN(pValue)) {
        return '-';
      }
      
      if (pValue < threshold) {
        return `< ${threshold}`;
      }
      
      // Use more decimal places for small p-values
      const precision = pValue < 0.01 ? 4 : 3;
      return this.formatNumber(pValue, precision);
    }
    
    /**
     * Format a coverage score
     * @param {number} actual - Actual coverage (0-1)
     * @param {number} target - Target coverage (0-1)
     * @param {number} precision - Decimal places (default: 2)
     * @return {string} Formatted coverage
     */
    formatCoverage(actual, target, precision = this.defaultOptions.percentPrecision) {
      if (actual === null || target === null || 
          actual === undefined || target === undefined || 
          isNaN(actual) || isNaN(target)) {
        return '-';
      }
      
      const actualStr = this.formatPercent(actual, precision);
      const targetStr = this.formatPercent(target, precision);
      const error = actual - target;
      const errorStr = error >= 0 ? `+${this.formatPercent(error, precision)}` : this.formatPercent(error, precision);
      
      return `${actualStr} (Target: ${targetStr}, Error: ${errorStr})`;
    }
    
    /**
     * Format a calibration score
     * @param {number} score - Calibration score (0-1)
     * @param {boolean} invert - Whether to invert (1 - score) for error to accuracy conversion
     * @param {number} precision - Decimal places (default: 2)
     * @return {string} Formatted calibration score
     */
    formatCalibration(score, invert = false, precision = this.defaultOptions.percentPrecision) {
      if (score === null || score === undefined || isNaN(score)) {
        return '-';
      }
      
      // Invert for accuracy vs error
      const displayValue = invert ? 1 - score : score;
      
      return this.formatPercent(displayValue, precision);
    }
    
    /**
     * Format an uncertainty score
     * @param {number} score - Uncertainty score (0-1)
     * @param {boolean} includeLabel - Include rating label (default: false)
     * @param {number} precision - Decimal places (default: 1)
     * @return {string} Formatted uncertainty score
     */
    formatUncertaintyScore(score, includeLabel = false, precision = 1) {
      if (score === null || score === undefined || isNaN(score)) {
        return '-';
      }
      
      const formattedScore = this.formatPercent(score, precision);
      
      if (!includeLabel) {
        return formattedScore;
      }
      
      let label;
      if (score >= 0.9) label = 'Excellent';
      else if (score >= 0.75) label = 'Good';
      else if (score >= 0.6) label = 'Moderate';
      else if (score >= 0.4) label = 'Fair';
      else if (score >= 0.2) label = 'Poor';
      else label = 'Critical';
      
      return `${formattedScore} (${label})`;
    }
    
    /**
     * Format an alpha level
     * @param {number} alpha - Alpha level (0-1)
     * @param {boolean} asConfidence - Format as confidence level (default: false)
     * @param {number} precision - Decimal places (default: 2)
     * @return {string} Formatted alpha level
     */
    formatAlphaLevel(alpha, asConfidence = false, precision = this.defaultOptions.percentPrecision) {
      if (alpha === null || alpha === undefined || isNaN(alpha)) {
        return '-';
      }
      
      // Convert alpha to confidence level if requested
      const value = asConfidence ? 1 - alpha : alpha;
      
      return this.formatPercent(value, precision);
    }
    
    /**
     * Format a date
     * @param {Date|string|number} date - Date to format
     * @param {string} format - Format style ('short', 'medium', 'long', 'full')
     * @return {string} Formatted date
     */
    formatDate(date, format = 'medium') {
      if (!date) return '-';
      
      const dateObj = date instanceof Date ? date : new Date(date);
      
      if (isNaN(dateObj.getTime())) {
        return '-';
      }
      
      let options;
      
      switch (format) {
        case 'short':
          options = { year: 'numeric', month: '2-digit', day: '2-digit' };
          break;
        case 'medium':
          options = { year: 'numeric', month: 'short', day: 'numeric' };
          break;
        case 'long':
          options = { year: 'numeric', month: 'long', day: 'numeric', weekday: 'long' };
          break;
        case 'full':
          options = {
            year: 'numeric', month: 'long', day: 'numeric', weekday: 'long',
            hour: '2-digit', minute: '2-digit', second: '2-digit'
          };
          break;
        default:
          options = { year: 'numeric', month: 'short', day: 'numeric' };
      }
      
      return dateObj.toLocaleDateString('en-US', options);
    }
    
    /**
     * Generate a badge HTML for a status or category
     * @param {string} text - Badge text
     * @param {string} type - Badge type (success, warning, danger, info, etc.)
     * @param {Object} options - Badge options
     * @return {string} Badge HTML
     */
    createBadgeHtml(text, type = 'info', options = {}) {
      // Badge colors
      const colors = {
        success: { bg: '#d4edda', text: '#155724', border: '#c3e6cb' },
        warning: { bg: '#fff3cd', text: '#856404', border: '#ffeeba' },
        danger: { bg: '#f8d7da', text: '#721c24', border: '#f5c6cb' },
        info: { bg: '#d1ecf1', text: '#0c5460', border: '#bee5eb' },
        primary: { bg: '#cfe2ff', text: '#084298', border: '#b6d4fe' },
        secondary: { bg: '#e2e3e5', text: '#41464b', border: '#d3d6d8' }
      };
      
      const color = colors[type] || colors.info;
      
      return `
        <span style="
          display: inline-block;
          padding: 0.25em 0.6em;
          font-size: 0.75em;
          font-weight: 700;
          line-height: 1;
          text-align: center;
          white-space: nowrap;
          vertical-align: baseline;
          border-radius: 0.25rem;
          background-color: ${color.bg};
          color: ${color.text};
          border: 1px solid ${color.border};
        ">${text}</span>
      `;
    }
    
    /**
     * Format interval width rating
     * @param {number} width - Interval width
     * @param {number} maxWidth - Maximum expected width
     * @param {boolean} includeLabel - Include rating label (default: false)
     * @return {string} Formatted interval width
     */
    formatIntervalWidth(width, maxWidth, includeLabel = false) {
      if (width === null || width === undefined || isNaN(width)) {
        return '-';
      }
      
      const formattedWidth = this.formatNumber(width, 2);
      
      if (!includeLabel) {
        return formattedWidth;
      }
      
      // Normalize width to 0-1 range
      const normalizedWidth = Math.min(1, width / maxWidth);
      
      // Invert so that smaller widths get better ratings
      const score = 1 - normalizedWidth;
      
      let label;
      if (score >= 0.8) label = 'Excellent (Narrow)';
      else if (score >= 0.6) label = 'Good';
      else if (score >= 0.4) label = 'Moderate';
      else if (score >= 0.2) label = 'Fair';
      else label = 'Poor (Wide)';
      
      return `${formattedWidth} (${label})`;
    }
    
    /**
     * Format a sharpness metric
     * @param {number} sharpness - Sharpness value (higher is better)
     * @param {boolean} includeLabel - Include rating label (default: false)
     * @return {string} Formatted sharpness
     */
    formatSharpness(sharpness, includeLabel = false) {
      if (sharpness === null || sharpness === undefined || isNaN(sharpness)) {
        return '-';
      }
      
      const formattedSharpness = this.formatNumber(sharpness, 2);
      
      if (!includeLabel) {
        return formattedSharpness;
      }
      
      // Map sharpness to a 0-1 scale
      // This is a simplified approach as sharpness scale depends on domain
      const normalizedSharpness = Math.min(1, sharpness / 10);
      
      let label;
      if (normalizedSharpness >= 0.8) label = 'Excellent (Very Sharp)';
      else if (normalizedSharpness >= 0.6) label = 'Good';
      else if (normalizedSharpness >= 0.4) label = 'Moderate';
      else if (normalizedSharpness >= 0.2) label = 'Fair';
      else label = 'Poor (Diffuse)';
      
      return `${formattedSharpness} (${label})`;
    }
  }
  
  export default Formatters;