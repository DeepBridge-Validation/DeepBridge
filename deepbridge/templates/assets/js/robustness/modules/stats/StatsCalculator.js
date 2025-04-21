// StatsCalculator.js - placeholder
/**
 * Statistics Calculator
 * 
 * Core utility for calculating statistical measures from data values
 * including mean, median, variance, standard deviation, and percentiles.
 */

class StatsCalculator {
    /**
     * Initialize the stats calculator
     */
    constructor() {
      // Default configuration for statistical calculations
      this.defaultConfig = {
        precision: 4,        // Decimal places for numeric results
        minValues: 2,        // Minimum number of values required for most calculations
        outlierThreshold: 1.5 // Multiplier for IQR in outlier detection
      };
    }
    
    /**
     * Calculate basic statistical measures for an array of values
     * @param {Array} values - Array of numeric values
     * @return {Object} Statistical measures
     */
    calculateBasicStats(values) {
      if (!values || values.length === 0) {
        return {
          count: 0,
          mean: null,
          sum: null,
          min: null,
          max: null,
          range: null
        };
      }
      
      // Filter out non-numeric values
      const numericValues = values.filter(val => !isNaN(val) && val !== null);
      
      if (numericValues.length === 0) {
        return {
          count: 0,
          mean: null,
          sum: null,
          min: null,
          max: null,
          range: null
        };
      }
      
      // Calculate basic statistics
      const count = numericValues.length;
      const sum = numericValues.reduce((acc, val) => acc + val, 0);
      const mean = sum / count;
      const min = Math.min(...numericValues);
      const max = Math.max(...numericValues);
      const range = max - min;
      
      return {
        count,
        mean,
        sum,
        min,
        max,
        range
      };
    }
    
    /**
     * Calculate variance and standard deviation
     * @param {Array} values - Array of numeric values
     * @param {boolean} isSample - Whether the values represent a sample (vs population)
     * @return {Object} Variance and standard deviation
     */
    calculateVariance(values, isSample = true) {
      if (!values || values.length < this.defaultConfig.minValues) {
        return {
          variance: null,
          stdDev: null
        };
      }
      
      // Filter out non-numeric values
      const numericValues = values.filter(val => !isNaN(val) && val !== null);
      
      if (numericValues.length < this.defaultConfig.minValues) {
        return {
          variance: null,
          stdDev: null
        };
      }
      
      // Calculate mean
      const mean = numericValues.reduce((acc, val) => acc + val, 0) / numericValues.length;
      
      // Calculate sum of squared deviations
      const sumSquaredDeviations = numericValues.reduce(
        (acc, val) => acc + Math.pow(val - mean, 2), 0
      );
      
      // Calculate variance
      // For a sample, divide by n-1
      // For a population, divide by n
      const divisor = isSample ? numericValues.length - 1 : numericValues.length;
      const variance = sumSquaredDeviations / divisor;
      
      // Calculate standard deviation
      const stdDev = Math.sqrt(variance);
      
      return {
        variance,
        stdDev
      };
    }
    
    /**
     * Calculate percentiles and quantiles
     * @param {Array} values - Array of numeric values
     * @return {Object} Percentiles and related measures
     */
    calculatePercentiles(values) {
      if (!values || values.length < this.defaultConfig.minValues) {
        return {
          median: null,
          q1: null,
          q3: null,
          iqr: null,
          p10: null,
          p90: null
        };
      }
      
      // Filter and sort values
      const numericValues = values
        .filter(val => !isNaN(val) && val !== null)
        .sort((a, b) => a - b);
      
      if (numericValues.length < this.defaultConfig.minValues) {
        return {
          median: null,
          q1: null,
          q3: null,
          iqr: null,
          p10: null,
          p90: null
        };
      }
      
      // Calculate percentiles
      const median = this.getPercentile(numericValues, 0.5);
      const q1 = this.getPercentile(numericValues, 0.25);
      const q3 = this.getPercentile(numericValues, 0.75);
      const p10 = this.getPercentile(numericValues, 0.1);
      const p90 = this.getPercentile(numericValues, 0.9);
      
      // Calculate interquartile range
      const iqr = q3 - q1;
      
      return {
        median,
        q1,
        q3,
        iqr,
        p10,
        p90
      };
    }
    
    /**
     * Get a specific percentile from sorted values
     * @param {Array} sortedValues - Sorted array of numeric values
     * @param {number} percentile - Percentile to calculate (0-1)
     * @return {number} Percentile value
     */
    getPercentile(sortedValues, percentile) {
      if (!sortedValues || sortedValues.length === 0 || percentile < 0 || percentile > 1) {
        return null;
      }
      
      // Calculate index
      const index = percentile * (sortedValues.length - 1);
      const lowerIndex = Math.floor(index);
      const upperIndex = Math.ceil(index);
      
      // Handle exact match
      if (lowerIndex === upperIndex) {
        return sortedValues[lowerIndex];
      }
      
      // Interpolate between values
      const weight = index - lowerIndex;
      return sortedValues[lowerIndex] * (1 - weight) + sortedValues[upperIndex] * weight;
    }
    
    /**
     * Detect outliers using the IQR method
     * @param {Array} values - Array of numeric values
     * @param {number} threshold - Multiplier for IQR
     * @return {Object} Outlier information
     */
    detectOutliers(values, threshold = this.defaultConfig.outlierThreshold) {
      if (!values || values.length < 4) { // Need at least 4 values for meaningful outlier detection
        return {
          lowerBound: null,
          upperBound: null,
          outliers: []
        };
      }
      
      // Calculate quartiles and IQR
      const percentiles = this.calculatePercentiles(values);
      
      if (percentiles.q1 === null || percentiles.q3 === null) {
        return {
          lowerBound: null,
          upperBound: null,
          outliers: []
        };
      }
      
      // Calculate bounds
      const lowerBound = percentiles.q1 - threshold * percentiles.iqr;
      const upperBound = percentiles.q3 + threshold * percentiles.iqr;
      
      // Identify outliers
      const outliers = values.filter(val => val < lowerBound || val > upperBound);
      
      return {
        lowerBound,
        upperBound,
        outliers
      };
    }
    
    /**
     * Calculate a complete statistical summary
     * @param {Array} values - Array of numeric values
     * @return {Object} Complete statistical summary
     */
    calculateSummary(values) {
      if (!values || values.length === 0) {
        return {
          count: 0,
          mean: null,
          median: null,
          min: null,
          max: null,
          stdDev: null,
          q1: null,
          q3: null,
          iqr: null
        };
      }
      
      // Calculate all statistics
      const basicStats = this.calculateBasicStats(values);
      const varianceStats = this.calculateVariance(values);
      const percentileStats = this.calculatePercentiles(values);
      
      // Combine results
      return {
        count: basicStats.count,
        mean: basicStats.mean,
        median: percentileStats.median,
        min: basicStats.min,
        max: basicStats.max,
        range: basicStats.range,
        variance: varianceStats.variance,
        stdDev: varianceStats.stdDev,
        q1: percentileStats.q1,
        q3: percentileStats.q3,
        iqr: percentileStats.iqr,
        p10: percentileStats.p10,
        p90: percentileStats.p90
      };
    }
    
    /**
     * Calculate z-scores for values
     * @param {Array} values - Array of numeric values
     * @return {Array} Z-scores for each value
     */
    calculateZScores(values) {
      if (!values || values.length < this.defaultConfig.minValues) {
        return [];
      }
      
      // Calculate mean and standard deviation
      const basicStats = this.calculateBasicStats(values);
      const varianceStats = this.calculateVariance(values);
      
      if (basicStats.mean === null || varianceStats.stdDev === null || varianceStats.stdDev === 0) {
        return values.map(() => null);
      }
      
      // Calculate z-scores
      return values.map(val => (val - basicStats.mean) / varianceStats.stdDev);
    }
    
    /**
     * Calculate the sum of squared differences
     * @param {Array} values1 - First array of values
     * @param {Array} values2 - Second array of values
     * @return {number} Sum of squared differences
     */
    calculateSumSquaredDiff(values1, values2) {
      if (!values1 || !values2 || values1.length !== values2.length || values1.length === 0) {
        return null;
      }
      
      return values1.reduce((sum, val, i) => sum + Math.pow(val - values2[i], 2), 0);
    }
    
    /**
     * Format a number to specified precision
     * @param {number} value - Number to format
     * @param {number} precision - Decimal places
     * @return {number} Formatted number
     */
    formatNumber(value, precision = this.defaultConfig.precision) {
      if (value === null || value === undefined || isNaN(value)) {
        return null;
      }
      
      return Number(value.toFixed(precision));
    }
  }
  
  export default StatsCalculator;