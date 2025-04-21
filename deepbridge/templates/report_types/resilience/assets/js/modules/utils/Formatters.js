// Formatters.js - placeholder
/**
 * Formatters
 * 
 * Utility functions for formatting numbers, dates, text, and other
 * data for display in the resilience report.
 */

class Formatters {
    /**
     * Format a number with specified precision
     * @param {number} value - Number to format
     * @param {number} precision - Decimal places (default: 2)
     * @param {boolean} useGrouping - Use thousand separators (default: true)
     * @return {string} Formatted number
     */
    static formatNumber(value, precision = 2, useGrouping = true) {
      if (value === null || value === undefined || isNaN(value)) {
        return '-';
      }
      
      return new Intl.NumberFormat('en-US', {
        minimumFractionDigits: precision,
        maximumFractionDigits: precision,
        useGrouping: useGrouping
      }).format(value);
    }
    
    /**
     * Format a number as percentage
     * @param {number} value - Number to format (0-1)
     * @param {number} precision - Decimal places (default: 1)
     * @return {string} Formatted percentage
     */
    static formatPercent(value, precision = 1) {
      if (value === null || value === undefined || isNaN(value)) {
        return '-';
      }
      
      return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: precision,
        maximumFractionDigits: precision
      }).format(value);
    }
    
    /**
     * Format a change in value with sign
     * @param {number} value - Current value
     * @param {number} baseline - Baseline value
     * @param {boolean} asPercent - Format as percentage (default: true)
     * @return {string} Formatted change
     */
    static formatChange(value, baseline, asPercent = true) {
      if (value === null || baseline === null || 
          value === undefined || baseline === undefined || 
          isNaN(value) || isNaN(baseline) || baseline === 0) {
        return '-';
      }
      
      const change = value - baseline;
      const percentChange = change / Math.abs(baseline);
      
      const formattedValue = asPercent ? 
        Formatters.formatPercent(percentChange) : 
        Formatters.formatNumber(change);
      
      const sign = change >= 0 ? '+' : '';
      
      return sign + formattedValue;
    }
    
    /**
     * Format a date
     * @param {Date|string|number} date - Date to format
     * @param {string} format - Format style ('short', 'medium', 'long', 'full')
     * @return {string} Formatted date
     */
    static formatDate(date, format = 'medium') {
      if (!date) return '-';
      
      const dateObj = date instanceof Date ? date : new Date(date);
      
      if (isNaN(dateObj.getTime())) {
        return '-';
      }
      
      let options;
      
      switch (format) {
        case 'short':
          options = { 
            year: 'numeric', 
            month: '2-digit', 
            day: '2-digit' 
          };
          break;
        case 'medium':
          options = { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric' 
          };
          break;
        case 'long':
          options = { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric', 
            weekday: 'long' 
          };
          break;
        case 'full':
          options = { 
            year: 'numeric', 
            month: 'long', 
            day: 'numeric', 
            weekday: 'long',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            timeZoneName: 'short'
          };
          break;
        default:
          options = { 
            year: 'numeric', 
            month: 'short', 
            day: 'numeric' 
          };
      }
      
      return new Intl.DateTimeFormat('en-US', options).format(dateObj);
    }
    
    /**
     * Format a time duration
     * @param {number} milliseconds - Duration in milliseconds
     * @param {boolean} compact - Use compact format (default: false)
     * @return {string} Formatted duration
     */
    static formatDuration(milliseconds, compact = false) {
      if (milliseconds === null || milliseconds === undefined || isNaN(milliseconds)) {
        return '-';
      }
      
      const seconds = Math.floor(milliseconds / 1000);
      const minutes = Math.floor(seconds / 60);
      const hours = Math.floor(minutes / 60);
      const days = Math.floor(hours / 24);
      
      if (compact) {
        if (days > 0) return `${days}d`;
        if (hours > 0) return `${hours}h`;
        if (minutes > 0) return `${minutes}m`;
        if (seconds > 0) return `${seconds}s`;
        return `${milliseconds}ms`;
      } else {
        const parts = [];
        
        if (days > 0) parts.push(`${days} day${days !== 1 ? 's' : ''}`);
        if (hours % 24 > 0) parts.push(`${hours % 24} hour${hours % 24 !== 1 ? 's' : ''}`);
        if (minutes % 60 > 0) parts.push(`${minutes % 60} minute${minutes % 60 !== 1 ? 's' : ''}`);
        if (seconds % 60 > 0) parts.push(`${seconds % 60} second${seconds % 60 !== 1 ? 's' : ''}`);
        
        return parts.length > 0 ? parts.join(', ') : '0 seconds';
      }
    }
    
    /**
     * Format a file size
     * @param {number} bytes - Size in bytes
     * @param {number} precision - Decimal places (default: 1)
     * @return {string} Formatted file size
     */
    static formatFileSize(bytes, precision = 1) {
      if (bytes === null || bytes === undefined || isNaN(bytes)) {
        return '-';
      }
      
      const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
      let size = bytes;
      let unitIndex = 0;
      
      while (size >= 1024 && unitIndex < units.length - 1) {
        size /= 1024;
        unitIndex++;
      }
      
      return `${Formatters.formatNumber(size, precision)} ${units[unitIndex]}`;
    }
    
    /**
     * Format a value with metric prefix (k, M, G, etc.)
     * @param {number} value - Number to format
     * @param {number} precision - Decimal places (default: 1)
     * @return {string} Formatted value with prefix
     */
    static formatMetric(value, precision = 1) {
      if (value === null || value === undefined || isNaN(value)) {
        return '-';
      }
      
      const absValue = Math.abs(value);
      
      if (absValue >= 1e9) {
        return `${Formatters.formatNumber(value / 1e9, precision)}G`;
      } else if (absValue >= 1e6) {
        return `${Formatters.formatNumber(value / 1e6, precision)}M`;
      } else if (absValue >= 1e3) {
        return `${Formatters.formatNumber(value / 1e3, precision)}k`;
      } else if (absValue < 1e-3) {
        return `${Formatters.formatNumber(value * 1e6, precision)}µ`;
      } else {
        return Formatters.formatNumber(value, precision);
      }
    }
    
    /**
     * Format a value with appropriate unit
     * @param {number} value - Number to format
     * @param {string} unit - Unit of measurement
     * @param {number} precision - Decimal places (default: 2)
     * @return {string} Formatted value with unit
     */
    static formatWithUnit(value, unit, precision = 2) {
      if (value === null || value === undefined || isNaN(value)) {
        return '-';
      }
      
      return `${Formatters.formatNumber(value, precision)} ${unit}`;
    }
    
    /**
     * Format a resilience score
     * @param {number} score - Resilience score (0-1)
     * @param {boolean} includeLabel - Include rating label (default: false)
     * @return {string} Formatted resilience score
     */
    static formatResilienceScore(score, includeLabel = false) {
      if (score === null || score === undefined || isNaN(score)) {
        return '-';
      }
      
      const formattedScore = Formatters.formatPercent(score, 1);
      
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
     * Format impact level
     * @param {number} impact - Impact level (0-1)
     * @param {boolean} includeLabel - Include impact label (default: false)
     * @return {string} Formatted impact level
     */
    static formatImpact(impact, includeLabel = false) {
      if (impact === null || impact === undefined || isNaN(impact)) {
        return '-';
      }
      
      const formattedImpact = Formatters.formatPercent(impact, 1);
      
      if (!includeLabel) {
        return formattedImpact;
      }
      
      let label;
      
      if (impact <= 0.1) label = 'Minimal';
      else if (impact <= 0.2) label = 'Low';
      else if (impact <= 0.3) label = 'Moderate';
      else if (impact <= 0.5) label = 'Significant';
      else if (impact <= 0.7) label = 'High';
      else label = 'Severe';
      
      return `${formattedImpact} (${label})`;
    }
    
    /**
     * Format shift type
     * @param {string} shiftType - Shift type identifier
     * @return {string} Formatted shift type
     */
    static formatShiftType(shiftType) {
      switch (shiftType) {
        case 'covariate_shift':
          return 'Covariate Shift';
        case 'concept_drift':
          return 'Concept Drift';
        case 'dataset_shift':
          return 'Dataset Shift';
        case 'minor_shift':
          return 'Minor Shift';
        default:
          return shiftType ? shiftType.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : 'Unknown';
      }
    }
    
    /**
     * Format feature name for display
     * @param {string} name - Feature name
     * @param {boolean} shorten - Shorten long names (default: true)
     * @return {string} Formatted feature name
     */
    static formatFeatureName(name, shorten = true) {
      if (!name) return '';
      
      // Remove common prefixes
      let formatted = name.replace(/^(feature_|attr_|f_)/i, '');
      
      // Convert snake_case to space separated
      formatted = formatted.replace(/_/g, ' ');
      
      // Convert camelCase to space separated
      formatted = formatted.replace(/([a-z])([A-Z])/g, '$1 $2');
      
      // Capitalize first letter of each word
      formatted = formatted.replace(/\b\w/g, c => c.toUpperCase());
      
      // Shorten if needed
      if (shorten && formatted.length > 30) {
        return formatted.substring(0, 27) + '...';
      }
      
      return formatted;
    }
    
    /**
     * Truncate text to a specific length
     * @param {string} text - Text to truncate
     * @param {number} maxLength - Maximum length
     * @param {string} suffix - Truncation suffix (default: '...')
     * @return {string} Truncated text
     */
    static truncateText(text, maxLength, suffix = '...') {
      if (!text || text.length <= maxLength) {
        return text || '';
      }
      
      // Try to truncate at a space to avoid cutting words
      const lastSpace = text.lastIndexOf(' ', maxLength - suffix.length);
      
      if (lastSpace > maxLength * 0.7) {
        return text.substring(0, lastSpace) + suffix;
      }
      
      return text.substring(0, maxLength - suffix.length) + suffix;
    }
    
    /**
     * Generate pluralized text based on count
     * @param {number} count - Count value
     * @param {string} singular - Singular form
     * @param {string} plural - Plural form
     * @return {string} Pluralized text
     */
    static pluralize(count, singular, plural) {
      return count === 1 ? singular : plural;
    }
    
    /**
     * Format a JSON string for display
     * @param {Object} obj - Object to format
     * @param {number} indent - Indentation spaces (default: 2)
     * @return {string} Formatted JSON string
     */
    static formatJson(obj, indent = 2) {
      try {
        return JSON.stringify(obj, null, indent);
      } catch (error) {
        return String(obj);
      }
    }
    
    /**
     * Format a distance metric value
     * @param {number} value - Distance metric value
     * @param {string} metricType - Type of metric (js_divergence, wasserstein, etc.)
     * @return {string} Formatted distance metric
     */
    static formatDistanceMetric(value, metricType) {
      if (value === null || value === undefined || isNaN(value)) {
        return '-';
      }
      
      // Different formatting based on metric type
      switch (metricType) {
        case 'js_divergence':
        case 'hellinger':
          // These are already normalized to [0,1]
          return Formatters.formatNumber(value, 3);
        case 'kl_divergence':
          // KL divergence can be larger
          return Formatters.formatNumber(value, 2);
        case 'wasserstein':
          // Wasserstein depends on the scale of the data
          return Formatters.formatNumber(value, 3);
        default:
          return Formatters.formatNumber(value, 3);
      }
    }
    
    /**
     * Get HTML for a colored badge
     * @param {string} text - Badge text
     * @param {string} color - Badge color
     * @param {Object} options - Badge options
     * @return {string} HTML for badge
     */
    static getBadgeHtml(text, color, options = {}) {
      const defaults = {
        textColor: '#fff',
        borderRadius: '3px',
        padding: '2px 6px',
        fontWeight: 'bold',
        fontSize: '0.85em'
      };
      
      const settings = { ...defaults, ...options };
      
      return `<span style="
        background-color: ${color};
        color: ${settings.textColor};
        border-radius: ${settings.borderRadius};
        padding: ${settings.padding};
        font-weight: ${settings.fontWeight};
        font-size: ${settings.fontSize};
        display: inline-block;
      ">${text}</span>`;
    }
    
    /**
     * Get HTML for a colored resilience badge
     * @param {number} score - Resilience score (0-1)
     * @return {string} HTML for badge
     */
    static getResilienceBadgeHtml(score) {
      if (score === null || score === undefined || isNaN(score)) {
        return Formatters.getBadgeHtml('Unknown', '#6c757d');
      }
      
      let color, label;
      
      if (score >= 0.9) {
        color = '#28a745';
        label = 'Excellent';
      } else if (score >= 0.75) {
        color = '#5cb85c';
        label = 'Good';
      } else if (score >= 0.6) {
        color = '#ffc107';
        label = 'Moderate';
      } else if (score >= 0.4) {
        color = '#fd7e14';
        label = 'Fair';
      } else if (score >= 0.2) {
        color = '#dc3545';
        label = 'Poor';
      } else {
        color = '#8b0000';
        label = 'Critical';
      }
      
      return Formatters.getBadgeHtml(
        `${label} (${Formatters.formatPercent(score, 0)})`, 
        color
      );
    }
    
    /**
     * Get HTML for a colored impact badge
     * @param {number} impact - Impact value (0-1)
     * @return {string} HTML for badge
     */
    static getImpactBadgeHtml(impact) {
      if (impact === null || impact === undefined || isNaN(impact)) {
        return Formatters.getBadgeHtml('Unknown', '#6c757d');
      }
      
      let color, label;
      
      if (impact <= 0.1) {
        color = '#28a745';
        label = 'Minimal';
      } else if (impact <= 0.2) {
        color = '#5cb85c';
        label = 'Low';
      } else if (impact <= 0.3) {
        color = '#ffc107';
        label = 'Moderate';
      } else if (impact <= 0.5) {
        color = '#fd7e14';
        label = 'Significant';
      } else if (impact <= 0.7) {
        color = '#dc3545';
        label = 'High';
      } else {
        color = '#8b0000';
        label = 'Severe';
      }
      
      return Formatters.getBadgeHtml(
        `${label} (${Formatters.formatPercent(impact, 0)})`, 
        color
      );
    }
    
    /**
     * Get HTML for a shift type badge
     * @param {string} shiftType - Shift type identifier
     * @return {string} HTML for badge
     */
    static getShiftTypeBadgeHtml(shiftType) {
      let color, label;
      
      switch (shiftType) {
        case 'covariate_shift':
          color = '#fd7e14';
          label = 'Covariate Shift';
          break;
        case 'concept_drift':
          color = '#6f42c1';
          label = 'Concept Drift';
          break;
        case 'dataset_shift':
          color = '#dc3545';
          label = 'Dataset Shift';
          break;
        case 'minor_shift':
          color = '#5cb85c';
          label = 'Minor Shift';
          break;
        default:
          color = '#6c757d';
          label = shiftType ? 
            shiftType.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()) : 
            'Unknown';
      }
      
      return Formatters.getBadgeHtml(label, color);
    }
  }
  
  export default Formatters;