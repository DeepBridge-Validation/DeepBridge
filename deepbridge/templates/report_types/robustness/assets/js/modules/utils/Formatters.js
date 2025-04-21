/**
 * Formatters.js - Data formatting utilities for robustness visualization
 * Provides helper functions for formatting and displaying data
 */

/**
 * Formats a number with specified precision
 * @param {number} value - Number to format
 * @param {number} precision - Decimal precision (default: 2)
 * @param {boolean} useGrouping - Whether to use thousand separators (default: true)
 * @returns {string} - Formatted number
 */
function formatNumber(value, precision = 2, useGrouping = true) {
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
 * Formats a number as percentage
 * @param {number} value - Number to format (0-1)
 * @param {number} precision - Decimal precision (default: 1)
 * @returns {string} - Formatted percentage
 */
function formatPercent(value, precision = 1) {
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
 * Formats a date using specified format
 * @param {Date|string|number} date - Date to format
 * @param {string} format - Format type ('short', 'medium', 'long', 'full', 'iso', 'timestamp')
 * @returns {string} - Formatted date
 */
function formatDate(date, format = 'medium') {
  if (!date) return '-';
  
  const dateObj = date instanceof Date ? date : new Date(date);
  
  if (isNaN(dateObj.getTime())) {
    return '-';
  }
  
  switch (format) {
    case 'short':
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
      }).format(dateObj);
      
    case 'medium':
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
      }).format(dateObj);
      
    case 'long':
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
      }).format(dateObj);
      
    case 'full':
      return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        weekday: 'long',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        timeZoneName: 'short'
      }).format(dateObj);
      
    case 'iso':
      return dateObj.toISOString();
      
    case 'timestamp':
      return dateObj.getTime().toString();
      
    default:
      return dateObj.toLocaleDateString();
  }
}

/**
 * Formats a significance level with stars
 * @param {number} pValue - P-value
 * @returns {string} - Significance indicator
 */
function formatSignificance(pValue) {
  if (pValue === null || pValue === undefined || isNaN(pValue)) {
    return '-';
  }
  
  if (pValue < 0.001) return '***';
  if (pValue < 0.01) return '**';
  if (pValue < 0.05) return '*';
  return 'ns';
}

/**
 * Formats a value with an appropriate metric prefix (k, M, G, etc.)
 * @param {number} value - Number to format
 * @param {number} precision - Decimal precision (default: 1)
 * @returns {string} - Formatted value with prefix
 */
function formatWithMetricPrefix(value, precision = 1) {
  if (value === null || value === undefined || isNaN(value)) {
    return '-';
  }
  
  const abs = Math.abs(value);
  
  if (abs >= 1e9) {
    return formatNumber(value / 1e9, precision) + 'G';
  } else if (abs >= 1e6) {
    return formatNumber(value / 1e6, precision) + 'M';
  } else if (abs >= 1e3) {
    return formatNumber(value / 1e3, precision) + 'k';
  } else if (abs < 1e-3) {
    return formatNumber(value / 1e-6, precision) + 'μ';
  } else {
    return formatNumber(value, precision);
  }
}

/**
 * Formats a duration in milliseconds
 * @param {number} milliseconds - Duration in milliseconds
 * @param {string} format - Format type ('compact', 'full', 'seconds', 'hh:mm:ss')
 * @returns {string} - Formatted duration
 */
function formatDuration(milliseconds, format = 'compact') {
  if (milliseconds === null || milliseconds === undefined || isNaN(milliseconds)) {
    return '-';
  }
  
  const seconds = milliseconds / 1000;
  const minutes = seconds / 60;
  const hours = minutes / 60;
  const days = hours / 24;
  
  switch (format) {
    case 'compact':
      if (days >= 1) return formatNumber(days, 1) + 'd';
      if (hours >= 1) return formatNumber(hours, 1) + 'h';
      if (minutes >= 1) return formatNumber(minutes, 1) + 'm';
      if (seconds >= 1) return formatNumber(seconds, 1) + 's';
      return formatNumber(milliseconds, 0) + 'ms';
      
    case 'full':
      if (days >= 1) {
        return `${Math.floor(days)}d ${Math.floor(hours % 24)}h ${Math.floor(minutes % 60)}m ${Math.floor(seconds % 60)}s`;
      }
      if (hours >= 1) {
        return `${Math.floor(hours)}h ${Math.floor(minutes % 60)}m ${Math.floor(seconds % 60)}s`;
      }
      if (minutes >= 1) {
        return `${Math.floor(minutes)}m ${Math.floor(seconds % 60)}s`;
      }
      if (seconds >= 1) {
        return `${formatNumber(seconds, 1)}s`;
      }
      return `${formatNumber(milliseconds, 0)}ms`;
      
    case 'seconds':
      return formatNumber(seconds, 2) + 's';
      
    case 'hh:mm:ss':
      const h = Math.floor(hours).toString().padStart(2, '0');
      const m = Math.floor(minutes % 60).toString().padStart(2, '0');
      const s = Math.floor(seconds % 60).toString().padStart(2, '0');
      return `${h}:${m}:${s}`;
      
    default:
      return formatNumber(milliseconds, 0) + 'ms';
  }
}

/**
 * Formats a value based on its type
 * @param {*} value - Value to format
 * @param {string} type - Value type ('number', 'percent', 'date', 'duration', etc.)
 * @param {Object} options - Formatting options
 * @returns {string} - Formatted value
 */
function formatValue(value, type = 'auto', options = {}) {
  if (value === null || value === undefined) {
    return options.nullValue || '-';
  }
  
  // Auto-detect type if needed
  if (type === 'auto') {
    if (value instanceof Date) {
      type = 'date';
    } else if (typeof value === 'number') {
      if (value >= 0 && value <= 1 && options.preferPercent) {
        type = 'percent';
      } else {
        type = 'number';
      }
    } else if (typeof value === 'boolean') {
      type = 'boolean';
    } else {
      type = 'string';
    }
  }
  
  // Format based on type
  switch (type) {
    case 'number':
      return formatNumber(value, options.precision || 2, options.useGrouping);
      
    case 'percent':
      return formatPercent(value, options.precision || 1);
      
    case 'date':
      return formatDate(value, options.dateFormat || 'medium');
      
    case 'duration':
      return formatDuration(value, options.durationFormat || 'compact');
      
    case 'metric':
      return formatWithMetricPrefix(value, options.precision || 1);
      
    case 'significance':
      return formatSignificance(value);
      
    case 'boolean':
      return value ? (options.trueLabel || 'Yes') : (options.falseLabel || 'No');
      
    case 'string':
      if (options.maxLength && value.length > options.maxLength) {
        return value.substring(0, options.maxLength) + (options.ellipsis ? '...' : '');
      }
      return value;
      
    default:
      return String(value);
  }
}

/**
 * Formats a feature name for display
 * @param {string} name - Original feature name
 * @param {Object} options - Formatting options
 * @returns {string} - Formatted feature name
 */
function formatFeatureName(name, options = {}) {
  const defaults = {
    maxLength: 30,
    camelToSpaces: true,
    removePrefix: true,
    removeSuffix: true,
    capitalizeWords: true,
    prefixes: ['feature_', 'attr_', 'var_', 'f_'],
    suffixes: ['_value', '_attr', '_var', '_feature']
  };
  
  const settings = { ...defaults, ...options };
  let formatted = name;
  
  // Remove prefixes
  if (settings.removePrefix) {
    for (const prefix of settings.prefixes) {
      if (formatted.startsWith(prefix)) {
        formatted = formatted.substring(prefix.length);
        break;
      }
    }
  }
  
  // Remove suffixes
  if (settings.removeSuffix) {
    for (const suffix of settings.suffixes) {
      if (formatted.endsWith(suffix)) {
        formatted = formatted.substring(0, formatted.length - suffix.length);
        break;
      }
    }
  }
  
  // Convert camelCase to spaces
  if (settings.camelToSpaces) {
    formatted = formatted.replace(/([a-z])([A-Z])/g, '$1 $2');
  }
  
  // Replace underscores with spaces
  formatted = formatted.replace(/_/g, ' ');
  
  // Capitalize words
  if (settings.capitalizeWords) {
    formatted = formatted.replace(/\b\w/g, c => c.toUpperCase());
  }
  
  // Trim and handle maximum length
  formatted = formatted.trim();
  if (settings.maxLength && formatted.length > settings.maxLength) {
    formatted = formatted.substring(0, settings.maxLength) + '...';
  }
  
  return formatted;
}

/**
 * Generates a text description of perturbation levels
 * @param {number} level - Perturbation level (0-1)
 * @param {string} type - Perturbation type ('raw', 'quantile', 'custom')
 * @returns {string} - Human-readable description
 */
function describePerturbation(level, type = 'raw') {
  if (level === 0) {
    return 'No perturbation';
  }
  
  if (type === 'raw') {
    if (level <= 0.1) return 'Minimal noise';
    if (level <= 0.25) return 'Low noise';
    if (level <= 0.5) return 'Medium noise';
    if (level <= 0.75) return 'High noise';
    return 'Extreme noise';
  } else if (type === 'quantile') {
    if (level <= 0.1) return 'Minimal distribution shift';
    if (level <= 0.25) return 'Low distribution shift';
    if (level <= 0.5) return 'Medium distribution shift';
    if (level <= 0.75) return 'High distribution shift';
    return 'Extreme distribution shift';
  } else {
    if (level <= 0.2) return 'Very low perturbation';
    if (level <= 0.4) return 'Low perturbation';
    if (level <= 0.6) return 'Medium perturbation';
    if (level <= 0.8) return 'High perturbation';
    return 'Very high perturbation';
  }
}

/**
 * Generates a qualitative description of robustness score
 * @param {number} score - Robustness score (0-1)
 * @returns {string} - Qualitative description
 */
function describeRobustness(score) {
  if (score >= 0.9) return 'Excellent robustness';
  if (score >= 0.8) return 'Good robustness';
  if (score >= 0.7) return 'Moderate robustness';
  if (score >= 0.6) return 'Fair robustness';
  if (score >= 0.5) return 'Weak robustness';
  return 'Poor robustness';
}

/**
 * Generates a qualitative description of impact
 * @param {number} impact - Impact value (0-1)
 * @returns {string} - Qualitative description
 */
function describeImpact(impact) {
  if (impact <= 0.1) return 'Very low impact';
  if (impact <= 0.2) return 'Low impact';
  if (impact <= 0.3) return 'Moderate impact';
  if (impact <= 0.4) return 'Significant impact';
  if (impact <= 0.5) return 'High impact';
  return 'Severe impact';
}

/**
 * Formats a model name for display
 * @param {string} name - Original model name
 * @returns {string} - Formatted model name
 */
function formatModelName(name) {
  // Remove common prefixes and suffixes
  let formatted = name
    .replace(/^model_/i, '')
    .replace(/_model$/i, '')
    .replace(/^ml_/i, '')
    .replace(/_ml$/i, '')
    .replace(/^algorithm_/i, '')
    .replace(/_algorithm$/i, '');
  
  // Convert snake_case to spaces
  formatted = formatted.replace(/_/g, ' ');
  
  // Handle common model acronyms
  const acronyms = {
    'rf': 'Random Forest',
    'gbm': 'Gradient Boosting',
    'xgb': 'XGBoost',
    'lgb': 'LightGBM',
    'cb': 'CatBoost',
    'dt': 'Decision Tree',
    'lr': 'Linear Regression',
    'logreg': 'Logistic Regression',
    'svm': 'Support Vector Machine',
    'knn': 'K-Nearest Neighbors',
    'mlp': 'Multi-layer Perceptron',
    'nb': 'Naive Bayes',
    'nn': 'Neural Network',
    'cnn': 'Convolutional Neural Network',
    'rnn': 'Recurrent Neural Network',
    'lstm': 'LSTM Network',
    'gru': 'GRU Network',
    'dnn': 'Deep Neural Network'
  };
  
  // Replace acronyms
  for (const [acronym, full] of Object.entries(acronyms)) {
    const regex = new RegExp(`\\b${acronym}\\b`, 'i');
    formatted = formatted.replace(regex, full);
  }
  
  // Capitalize words
  formatted = formatted.replace(/\b\w/g, c => c.toUpperCase());
  
  return formatted.trim();
}

/**
 * Truncates text to a specified length
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @param {string} ellipsis - Ellipsis character(s)
 * @returns {string} - Truncated text
 */
function truncateText(text, maxLength = 100, ellipsis = '...') {
  if (!text) return '';
  
  if (text.length <= maxLength) {
    return text;
  }
  
  // Try to truncate at a space
  const truncated = text.substr(0, maxLength);
  const lastSpace = truncated.lastIndexOf(' ');
  
  if (lastSpace > maxLength * 0.7) {
    return truncated.substr(0, lastSpace) + ellipsis;
  }
  
  return truncated + ellipsis;
}

/**
 * Creates a short hash from a string
 * @param {string} str - Input string
 * @param {number} length - Hash length (default: 8)
 * @returns {string} - Hash string
 */
function createHash(str, length = 8) {
  let hash = 0;
  
  if (!str || str.length === 0) {
    return '0'.repeat(length);
  }
  
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash; // Convert to 32bit integer
  }
  
  // Convert to string and ensure positive
  const hashStr = Math.abs(hash).toString(16);
  
  // Pad or truncate to desired length
  if (hashStr.length < length) {
    return hashStr.padStart(length, '0');
  }
  
  return hashStr.slice(-length);
}

// Export the utilities
export {
  formatNumber,
  formatPercent,
  formatDate,
  formatSignificance,
  formatWithMetricPrefix,
  formatDuration,
  formatValue,
  formatFeatureName,
  describePerturbation,
  describeRobustness,
  describeImpact,
  formatModelName,
  truncateText,
  createHash
};