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
     * Calculate confidence interval for a mean
     * @param {Array} values - Array of values
     * @param {number} confidenceLevel - Confidence level (0-1)
     * @return {Object} Lower and upper bounds of the confidence interval
     */
    calculateConfidenceInterval(values, confidenceLevel = 0.95) {
      if (!values || values.length < this.defaultConfig.minValues) {
        return { lower: null, upper: null };
      }
      
      const stats = this.calculateBasicStats(values);
      const varianceStats = this.calculateVariance(values);
      
      if (stats.mean === null || varianceStats.stdDev === null) {
        return { lower: null, upper: null };
      }
      
      // Z-score for the confidence level (approx. normal distribution)
      // 95% confidence level has z-score of 1.96
      const alpha = 1 - confidenceLevel;
      const z = this.getZScore(alpha);
      
      // Standard error of the mean
      const standardError = varianceStats.stdDev / Math.sqrt(stats.count);
      
      // Confidence interval
      const marginOfError = z * standardError;
      const lower = stats.mean - marginOfError;
      const upper = stats.mean + marginOfError;
      
      return { lower, upper };
    }
    
    /**
     * Get z-score for a given alpha level
     * @param {number} alpha - Alpha level (e.g., 0.05 for 95% confidence)
     * @return {number} Z-score
     */
    getZScore(alpha) {
      // Common z-scores for standard alpha levels
      const zScores = {
        0.01: 2.576,
        0.05: 1.96,
        0.1: 1.645,
        0.2: 1.282,
        0.5: 0.674
      };
      
      // Return cached z-score if available
      if (zScores[alpha] !== undefined) {
        return zScores[alpha];
      }
      
      // Approximation of the inverse normal CDF for custom alpha levels
      // Using Abramowitz and Stegun approximation (algorithm 26.2.23)
      const p = 1 - alpha / 2;
      let z;
      
      if (p <= 0 || p >= 1) {
        z = 0;
      } else if (p < 0.5) {
        // Handle p < 0.5
        z = -this.approximateInverseNormalCDF(1 - p);
      } else {
        // Handle p >= 0.5
        z = this.approximateInverseNormalCDF(p);
      }
      
      return z;
    }
    
    /**
     * Approximate the inverse normal CDF for p >= 0.5
     * @param {number} p - Probability (0.5 <= p < 1)
     * @return {number} Z-score
     */
    approximateInverseNormalCDF(p) {
      // Coefficients for the approximation
      const c0 = 2.515517;
      const c1 = 0.802853;
      const c2 = 0.010328;
      const d1 = 1.432788;
      const d2 = 0.189269;
      const d3 = 0.001308;
      
      // Auxiliary variable
      const t = Math.sqrt(-2 * Math.log(1 - p));
      
      // Approximation formula
      return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t);
    }
    
    /**
     * Generate bootstrap samples for uncertainty estimation
     * @param {Array} values - Original sample values
     * @param {number} bootstrapCount - Number of bootstrap samples to generate
     * @param {number} sampleSize - Size of each bootstrap sample (default: same as original)
     * @return {Array} Array of bootstrap samples
     */
    generateBootstrapSamples(values, bootstrapCount = 1000, sampleSize = null) {
      if (!values || values.length === 0 || bootstrapCount <= 0) {
        return [];
      }
      
      const n = values.length;
      const size = sampleSize || n;
      const bootstrapSamples = [];
      
      // Generate bootstrap samples
      for (let i = 0; i < bootstrapCount; i++) {
        const bootstrapSample = [];
        
        // Sample with replacement
        for (let j = 0; j < size; j++) {
          const randomIndex = Math.floor(Math.random() * n);
          bootstrapSample.push(values[randomIndex]);
        }
        
        bootstrapSamples.push(bootstrapSample);
      }
      
      return bootstrapSamples;
    }
    
    /**
     * Calculate bootstrap confidence interval for a statistic
     * @param {Array} values - Original sample values
     * @param {Function} statFunc - Function to calculate the statistic
     * @param {number} confidenceLevel - Confidence level (0-1)
     * @param {number} bootstrapCount - Number of bootstrap samples to generate
     * @return {Object} Bootstrap confidence interval
     */
    bootstrapConfidenceInterval(values, statFunc, confidenceLevel = 0.95, bootstrapCount = 1000) {
      if (!values || values.length === 0 || !statFunc) {
        return { lower: null, upper: null };
      }
      
      // Generate bootstrap samples
      const bootstrapSamples = this.generateBootstrapSamples(values, bootstrapCount);
      
      // Calculate the statistic for each bootstrap sample
      const bootstrapStats = bootstrapSamples.map(sample => statFunc(sample));
      
      // Calculate percentiles for confidence interval
      const sortedStats = bootstrapStats.sort((a, b) => a - b);
      const alpha = 1 - confidenceLevel;
      const lowerPercentile = alpha / 2;
      const upperPercentile = 1 - (alpha / 2);
      
      const lower = this.getPercentile(sortedStats, lowerPercentile);
      const upper = this.getPercentile(sortedStats, upperPercentile);
      
      return { lower, upper };
    }
  }
  
  export default StatsCalculator;