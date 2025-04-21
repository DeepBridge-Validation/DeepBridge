// StatsCalculator.js - placeholder
/**
 * Statistical Calculator
 * 
 * Core utilities for calculating statistical measures and properties
 * of data distributions, essential for shift analysis.
 */

class StatsCalculator {
    /**
     * Calculate the mean of an array of values
     * @param {Array} values - Array of numeric values
     * @return {number} Mean value or null if invalid input
     */
    calculateMean(values) {
      if (!values || values.length === 0) {
        return null;
      }
      
      const sum = values.reduce((acc, val) => acc + val, 0);
      return sum / values.length;
    }
    
    /**
     * Calculate the variance of an array of values
     * @param {Array} values - Array of numeric values
     * @param {boolean} isSample - Whether values represent a sample (vs population)
     * @return {number} Variance value or null if invalid input
     */
    calculateVariance(values, isSample = true) {
      if (!values || values.length <= 1) {
        return null;
      }
      
      const mean = this.calculateMean(values);
      const sumSquaredDiffs = values.reduce((acc, val) => {
        return acc + Math.pow(val - mean, 2);
      }, 0);
      
      // For a sample, divide by n-1 (Bessel's correction)
      // For a population, divide by n
      const divisor = isSample ? values.length - 1 : values.length;
      return sumSquaredDiffs / divisor;
    }
    
    /**
     * Calculate the standard deviation of an array of values
     * @param {Array} values - Array of numeric values
     * @param {boolean} isSample - Whether values represent a sample (vs population)
     * @return {number} Standard deviation or null if invalid input
     */
    calculateStdDev(values, isSample = true) {
      const variance = this.calculateVariance(values, isSample);
      return variance !== null ? Math.sqrt(variance) : null;
    }
    
    /**
     * Calculate the median of an array of values
     * @param {Array} values - Array of numeric values
     * @return {number} Median value or null if invalid input
     */
    calculateMedian(values) {
      if (!values || values.length === 0) {
        return null;
      }
      
      // Clone and sort the array
      const sorted = [...values].sort((a, b) => a - b);
      const middle = Math.floor(sorted.length / 2);
      
      // If length is odd, return the middle element
      // If length is even, return average of the two middle elements
      if (sorted.length % 2 === 1) {
        return sorted[middle];
      } else {
        return (sorted[middle - 1] + sorted[middle]) / 2;
      }
    }
    
    /**
     * Calculate a specific percentile from an array of values
     * @param {Array} values - Array of numeric values
     * @param {number} percentile - Percentile to calculate (0-1)
     * @return {number} Percentile value or null if invalid input
     */
    calculatePercentile(values, percentile) {
      if (!values || values.length === 0 || percentile < 0 || percentile > 1) {
        return null;
      }
      
      // Clone and sort the array
      const sorted = [...values].sort((a, b) => a - b);
      
      // Calculate the index
      const index = percentile * (sorted.length - 1);
      const lowerIndex = Math.floor(index);
      const upperIndex = Math.ceil(index);
      
      // Handle exact match
      if (lowerIndex === upperIndex) {
        return sorted[lowerIndex];
      }
      
      // Interpolate between values
      const weight = index - lowerIndex;
      return sorted[lowerIndex] * (1 - weight) + sorted[upperIndex] * weight;
    }
    
    /**
     * Calculate quartiles (Q1, median, Q3) from an array of values
     * @param {Array} values - Array of numeric values
     * @return {Object} Quartile values or null if invalid input
     */
    calculateQuartiles(values) {
      if (!values || values.length < 4) {
        return {
          q1: null,
          median: null,
          q3: null
        };
      }
      
      return {
        q1: this.calculatePercentile(values, 0.25),
        median: this.calculatePercentile(values, 0.5),
        q3: this.calculatePercentile(values, 0.75)
      };
    }
    
    /**
     * Calculate the interquartile range (IQR) from an array of values
     * @param {Array} values - Array of numeric values
     * @return {number} IQR value or null if invalid input
     */
    calculateIQR(values) {
      const quartiles = this.calculateQuartiles(values);
      
      if (quartiles.q1 === null || quartiles.q3 === null) {
        return null;
      }
      
      return quartiles.q3 - quartiles.q1;
    }
    
    /**
     * Calculate the skewness of an array of values
     * @param {Array} values - Array of numeric values
     * @return {number} Skewness value or null if invalid input
     */
    calculateSkewness(values) {
      if (!values || values.length < 3) {
        return null;
      }
      
      const mean = this.calculateMean(values);
      const stdDev = this.calculateStdDev(values);
      
      if (stdDev === 0) {
        return 0; // No skewness if all values are the same
      }
      
      // Calculate the sum of cubed deviations
      const sumCubedDeviations = values.reduce((sum, val) => {
        return sum + Math.pow((val - mean) / stdDev, 3);
      }, 0);
      
      // Skewness = (Σ((x_i - μ)/σ)^3) / n
      return sumCubedDeviations / values.length;
    }
    
    /**
     * Calculate the kurtosis of an array of values
     * @param {Array} values - Array of numeric values
     * @return {number} Kurtosis value or null if invalid input
     */
    calculateKurtosis(values) {
      if (!values || values.length < 4) {
        return null;
      }
      
      const mean = this.calculateMean(values);
      const stdDev = this.calculateStdDev(values);
      
      if (stdDev === 0) {
        return 0; // Undefined kurtosis if all values are the same
      }
      
      // Calculate the sum of fourth power of deviations
      const sumQuarticDeviations = values.reduce((sum, val) => {
        return sum + Math.pow((val - mean) / stdDev, 4);
      }, 0);
      
      // Kurtosis = (Σ((x_i - μ)/σ)^4) / n
      return sumQuarticDeviations / values.length - 3; // Excess kurtosis (normal = 0)
    }
    
    /**
     * Calculate the min and max values of an array
     * @param {Array} values - Array of numeric values
     * @return {Object} Min and max values or null if invalid input
     */
    calculateRange(values) {
      if (!values || values.length === 0) {
        return {
          min: null,
          max: null,
          range: null
        };
      }
      
      const min = Math.min(...values);
      const max = Math.max(...values);
      
      return {
        min,
        max,
        range: max - min
      };
    }
    
    /**
     * Calculate outliers using the IQR method
     * @param {Array} values - Array of numeric values
     * @param {number} multiplier - IQR multiplier for outlier detection (default: 1.5)
     * @return {Object} Outlier information
     */
    findOutliers(values, multiplier = 1.5) {
      if (!values || values.length < 4) {
        return {
          outliers: [],
          lowerBound: null,
          upperBound: null
        };
      }
      
      const quartiles = this.calculateQuartiles(values);
      const iqr = quartiles.q3 - quartiles.q1;
      
      const lowerBound = quartiles.q1 - (multiplier * iqr);
      const upperBound = quartiles.q3 + (multiplier * iqr);
      
      const outliers = values.filter(val => val < lowerBound || val > upperBound);
      
      return {
        outliers,
        lowerBound,
        upperBound
      };
    }
    
    /**
     * Calculate a comprehensive statistical summary
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
          range: null,
          variance: null,
          stdDev: null,
          q1: null,
          q3: null,
          iqr: null,
          skewness: null,
          kurtosis: null
        };
      }
      
      const count = values.length;
      const mean = this.calculateMean(values);
      const range = this.calculateRange(values);
      const quartiles = this.calculateQuartiles(values);
      const variance = this.calculateVariance(values);
      const stdDev = Math.sqrt(variance);
      const iqr = quartiles.q3 - quartiles.q1;
      const skewness = this.calculateSkewness(values);
      const kurtosis = this.calculateKurtosis(values);
      
      return {
        count,
        mean,
        median: quartiles.median,
        ...range,
        variance,
        stdDev,
        q1: quartiles.q1,
        q3: quartiles.q3,
        iqr,
        skewness,
        kurtosis
      };
    }
    
    /**
     * Check if two distributions are significantly different
     * @param {Array} a - First distribution
     * @param {Array} b - Second distribution
     * @param {number} significance - Significance level (0-1, default: 0.05)
     * @return {Object} Test result
     */
    testDistributionDifference(a, b, significance = 0.05) {
      if (!a || !b || a.length < 5 || b.length < 5) {
        return {
          isDifferent: null,
          pValue: null,
          testName: null
        };
      }
      
      // Implementation note: In a real environment, this would use a statistical
      // test like Kolmogorov-Smirnov or Mann-Whitney U. For this example, we'll
      // use a simplified approach based on means and standard deviations.
      
      const statsA = this.calculateSummary(a);
      const statsB = this.calculateSummary(b);
      
      // Calculate t-statistic for difference in means
      const pooledStdDev = Math.sqrt(
        ((a.length - 1) * Math.pow(statsA.stdDev, 2) + 
         (b.length - 1) * Math.pow(statsB.stdDev, 2)) / 
        (a.length + b.length - 2)
      );
      
      const tStat = Math.abs(statsA.mean - statsB.mean) / 
                   (pooledStdDev * Math.sqrt(1/a.length + 1/b.length));
      
      // Simplified p-value calculation (approximate)
      // In a real implementation, we would use a proper t-distribution
      const degreesOfFreedom = a.length + b.length - 2;
      let pValue;
      
      if (degreesOfFreedom >= 30) {
        // Use normal approximation for large samples
        pValue = 2 * (1 - this.normalCDF(tStat));
      } else {
        // Simplified approximation for smaller samples
        pValue = 2 * (1 - this.normalCDF(tStat * (1 - 1/(4 * degreesOfFreedom))));
      }
      
      return {
        isDifferent: pValue < significance,
        pValue,
        testName: 'Approximate t-test'
      };
    }
    
    /**
     * Calculate the CDF of a standard normal distribution
     * @param {number} x - Value to evaluate
     * @return {number} Cumulative probability
     */
    normalCDF(x) {
      // Approximation of the normal CDF
      // Source: Abramowitz and Stegun approximation
      if (x < 0) {
        return 1 - this.normalCDF(-x);
      }
      
      const b1 = 0.319381530;
      const b2 = -0.356563782;
      const b3 = 1.781477937;
      const b4 = -1.821255978;
      const b5 = 1.330274429;
      const p = 0.2316419;
      const c = 0.39894228;
      
      const t = 1.0 / (1.0 + p * x);
      const poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))));
      
      return 1.0 - c * Math.exp(-x * x / 2.0) * poly;
    }
  }
  
  export default StatsCalculator;