/**
 * Statistical Analysis
 * 
 * Provides statistical analysis functions for robustness data,
 * including calculations for distributions, correlations, and metrics.
 */

class StatsAnalyzer {
    /**
     * Initialize the statistical analyzer
     */
    constructor() {
      // Configuration for statistical calculations
      this.config = {
        confidenceLevel: 0.95, // 95% confidence level for intervals
        outlierThreshold: 1.5  // IQR multiplier for outlier detection
      };
    }
    
    /**
     * Calculate statistical measures for a distribution
     * @param {Array} values - Array of numeric values
     * @return {Object} Statistical measures (mean, median, std, min, max, quantiles)
     */
    calculateStats(values) {
      if (!values || values.length === 0) {
        return {
          mean: null,
          median: null,
          stdDev: null,
          min: null,
          max: null,
          q1: null,
          q3: null,
          iqr: null
        };
      }
      
      // Sort values for easier calculations
      const sortedValues = [...values].sort((a, b) => a - b);
      
      // Basic statistics
      const min = sortedValues[0];
      const max = sortedValues[sortedValues.length - 1];
      const sum = sortedValues.reduce((acc, val) => acc + val, 0);
      const mean = sum / sortedValues.length;
      
      // Variance and standard deviation
      const variance = sortedValues.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / sortedValues.length;
      const stdDev = Math.sqrt(variance);
      
      // Median and quartiles
      const median = this.getPercentile(sortedValues, 0.5);
      const q1 = this.getPercentile(sortedValues, 0.25);
      const q3 = this.getPercentile(sortedValues, 0.75);
      const iqr = q3 - q1;
      
      return {
        mean,
        median,
        stdDev,
        min,
        max,
        q1,
        q3,
        iqr
      };
    }
    
    /**
     * Get a percentile value from a sorted array
     * @param {Array} sortedValues - Sorted array of values
     * @param {number} percentile - Percentile (0-1)
     * @return {number} Percentile value
     */
    getPercentile(sortedValues, percentile) {
      if (!sortedValues || sortedValues.length === 0) {
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
     * Calculate confidence interval for a mean
     * @param {Array} values - Array of values
     * @param {number} confidenceLevel - Confidence level (0-1)
     * @return {Object} Lower and upper bounds of the confidence interval
     */
    calculateConfidenceInterval(values, confidenceLevel = this.config.confidenceLevel) {
      if (!values || values.length < 2) {
        return { lower: null, upper: null };
      }
      
      const stats = this.calculateStats(values);
      
      // Z-score for the confidence level (approx. normal distribution)
      // 95% confidence level has z-score of 1.96
      const z = 1.96;
      
      // Standard error of the mean
      const standardError = stats.stdDev / Math.sqrt(values.length);
      
      // Confidence interval
      const marginOfError = z * standardError;
      const lower = Math.max(0, stats.mean - marginOfError);
      const upper = Math.min(1, stats.mean + marginOfError);
      
      return { lower, upper };
    }
    
    /**
     * Detect outliers in a dataset using IQR method
     * @param {Array} values - Array of values
     * @param {number} threshold - IQR multiplier for outlier detection
     * @return {Array} Outlier values
     */
    detectOutliers(values, threshold = this.config.outlierThreshold) {
      if (!values || values.length < 4) {
        return [];
      }
      
      const stats = this.calculateStats(values);
      
      // IQR method for outlier detection
      const lowerBound = stats.q1 - threshold * stats.iqr;
      const upperBound = stats.q3 + threshold * stats.iqr;
      
      return values.filter(value => value < lowerBound || value > upperBound);
    }
    
    /**
     * Calculate metrics for a level-wise performance comparison
     * @param {Object} perturbationData - Perturbation data
     * @return {Object} Level metrics
     */
    calculateLevelMetrics(perturbationData) {
      if (!perturbationData || !perturbationData.levels || perturbationData.levels.length === 0) {
        return [];
      }
      
      const levelMetrics = [];
      
      // Calculate metrics for each level
      perturbationData.levels.forEach((level, index) => {
        const score = perturbationData.scores[index];
        const worstScore = perturbationData.worstScores[index];
        const baseScore = perturbationData.baseScore;
        
        // Calculate impact as percentage decrease from base score
        const impact = (baseScore - score) / baseScore;
        
        // Calculate robustness as 1 - impact
        const robustness = 1 - impact;
        
        // Calculate variability as the difference between mean and worst score
        const variability = score - worstScore;
        
        // Calculate stability as 1 - normalized variability
        const normalizedVariability = variability / score;
        const stability = 1 - normalizedVariability;
        
        levelMetrics.push({
          level,
          score,
          worstScore,
          impact,
          robustness,
          variability,
          stability
        });
      });
      
      return levelMetrics;
    }
    
    /**
     * Calculate correlation between two sets of values
     * @param {Array} values1 - First array of values
     * @param {Array} values2 - Second array of values
     * @return {number} Correlation coefficient (-1 to 1)
     */
    calculateCorrelation(values1, values2) {
      if (!values1 || !values2 || values1.length !== values2.length || values1.length < 2) {
        return null;
      }
      
      // Calculate means
      const mean1 = values1.reduce((sum, val) => sum + val, 0) / values1.length;
      const mean2 = values2.reduce((sum, val) => sum + val, 0) / values2.length;
      
      // Calculate deviations
      const dev1 = values1.map(val => val - mean1);
      const dev2 = values2.map(val => val - mean2);
      
      // Calculate sum of squared deviations
      const sqDev1 = dev1.reduce((sum, val) => sum + val * val, 0);
      const sqDev2 = dev2.reduce((sum, val) => sum + val * val, 0);
      
      // Calculate sum of products of deviations
      const sumProdDev = dev1.reduce((sum, val, i) => sum + val * dev2[i], 0);
      
      // Calculate correlation coefficient
      const correlation = sumProdDev / Math.sqrt(sqDev1 * sqDev2);
      
      return correlation;
    }
    
    /**
     * Generate synthetic distribution data for visualizations
     * @param {Object} stats - Statistical measures for the distribution
     * @param {number} sampleCount - Number of samples to generate
     * @return {Array} Array of synthetic samples
     */
    generateDistributionSamples(stats, sampleCount = 100) {
      if (!stats || !stats.mean || !stats.stdDev) {
        return [];
      }
      
      const samples = [];
      
      // Generate samples using Box-Muller transform for normal distribution
      for (let i = 0; i < sampleCount; i++) {
        // Box-Muller transform
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        
        // Transform to desired distribution
        let sample = stats.mean + stats.stdDev * z;
        
        // Bound within min and max
        sample = Math.max(stats.min, Math.min(stats.max, sample));
        
        samples.push(sample);
      }
      
      return samples;
    }
    
    /**
     * Calculate robustness metrics for model comparison
     * @param {Array} models - Array of model data objects
     * @return {Object} Comparative metrics
     */
    calculateComparativeMetrics(models) {
      if (!models || models.length === 0) {
        return {
          bestModel: null,
          avgRobustness: null,
          bestRobustness: null,
          worstRobustness: null,
          robustnessSpread: null
        };
      }
      
      // Calculate robustness scores
      const robustnessScores = models.map(model => {
        const baseScore = model.baseScore || 1;
        const avgScore = model.scores.reduce((sum, score) => sum + (score || 0), 0) / 
                        model.scores.filter(score => score !== null).length;
        
        return (avgScore / baseScore);
      });
      
      // Find best and worst models
      const bestIndex = robustnessScores.indexOf(Math.max(...robustnessScores));
      const worstIndex = robustnessScores.indexOf(Math.min(...robustnessScores));
      
      return {
        bestModel: models[bestIndex].name,
        avgRobustness: robustnessScores.reduce((sum, score) => sum + score, 0) / robustnessScores.length,
        bestRobustness: robustnessScores[bestIndex],
        worstRobustness: robustnessScores[worstIndex],
        robustnessSpread: robustnessScores[bestIndex] - robustnessScores[worstIndex]
      };
    }
  }
  
  export default StatsAnalyzer;