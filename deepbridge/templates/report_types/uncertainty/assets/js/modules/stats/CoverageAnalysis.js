// CoverageAnalysis.js - placeholder
/**
 * Coverage Analysis
 * 
 * Utilities for assessing the coverage of prediction intervals
 * and confidence intervals for uncertainty quantification.
 */
import StatsCalculator from './StatsCalculator.js';

class CoverageAnalysis {
  /**
   * Initialize the coverage analysis utilities
   */
  constructor() {
    this.statsCalculator = new StatsCalculator();
    
    // Default alpha levels for confidence intervals
    this.defaultAlphaLevels = [0.01, 0.05, 0.1, 0.2, 0.5];
  }
  
  /**
   * Calculate empirical coverage for a single alpha level
   * @param {Array} lowerBounds - Lower bounds of prediction intervals
   * @param {Array} upperBounds - Upper bounds of prediction intervals
   * @param {Array} actualValues - Actual observed values
   * @return {number} Empirical coverage (0-1)
   */
  calculateCoverage(lowerBounds, upperBounds, actualValues) {
    if (!lowerBounds || !upperBounds || !actualValues || 
        lowerBounds.length !== upperBounds.length || 
        lowerBounds.length !== actualValues.length ||
        lowerBounds.length === 0) {
      return null;
    }
    
    // Count how many actual values fall within the prediction intervals
    let coveredCount = 0;
    
    for (let i = 0; i < actualValues.length; i++) {
      if (actualValues[i] >= lowerBounds[i] && actualValues[i] <= upperBounds[i]) {
        coveredCount++;
      }
    }
    
    // Calculate empirical coverage
    return coveredCount / actualValues.length;
  }
  
  /**
   * Calculate coverage for multiple alpha levels
   * @param {Object} predictionIntervals - Prediction intervals for different alpha levels
   * @param {Array} actualValues - Actual observed values
   * @return {Object} Coverage for each alpha level
   */
  calculateCoverageByAlpha(predictionIntervals, actualValues) {
    if (!predictionIntervals || !actualValues || actualValues.length === 0) {
      return {};
    }
    
    const coverageByAlpha = {};
    
    // Calculate coverage for each alpha level
    Object.entries(predictionIntervals).forEach(([alpha, intervals]) => {
      if (!intervals.lowerBounds || !intervals.upperBounds) return;
      
      const coverage = this.calculateCoverage(
        intervals.lowerBounds,
        intervals.upperBounds,
        actualValues
      );
      
      coverageByAlpha[alpha] = coverage;
    });
    
    return coverageByAlpha;
  }
  
  /**
   * Calculate average coverage across alpha levels
   * @param {Object} coverageByAlpha - Coverage for each alpha level
   * @return {number} Average coverage (0-1)
   */
  calculateAverageCoverage(coverageByAlpha) {
    if (!coverageByAlpha || Object.keys(coverageByAlpha).length === 0) {
      return null;
    }
    
    const coverages = Object.values(coverageByAlpha).filter(c => c !== null);
    
    if (coverages.length === 0) return null;
    
    return coverages.reduce((sum, c) => sum + c, 0) / coverages.length;
  }
  
  /**
   * Calculate coverage error (difference between empirical and expected coverage)
   * @param {Object} coverageByAlpha - Coverage for each alpha level
   * @return {Object} Coverage error for each alpha level
   */
  calculateCoverageError(coverageByAlpha) {
    if (!coverageByAlpha || Object.keys(coverageByAlpha).length === 0) {
      return {};
    }
    
    const coverageError = {};
    
    Object.entries(coverageByAlpha).forEach(([alpha, coverage]) => {
      if (coverage === null) return;
      
      // Expected coverage = 1 - alpha
      const expectedCoverage = 1 - parseFloat(alpha);
      coverageError[alpha] = coverage - expectedCoverage;
    });
    
    return coverageError;
  }
  
  /**
   * Calculate average absolute coverage error
   * @param {Object} coverageError - Coverage error for each alpha level
   * @return {number} Average absolute coverage error
   */
  calculateAverageAbsoluteCoverageError(coverageError) {
    if (!coverageError || Object.keys(coverageError).length === 0) {
      return null;
    }
    
    const errors = Object.values(coverageError).filter(e => e !== null);
    
    if (errors.length === 0) return null;
    
    // Calculate average absolute error
    return errors.reduce((sum, e) => sum + Math.abs(e), 0) / errors.length;
  }
  
  /**
   * Calculate interval widths for different alpha levels
   * @param {Object} predictionIntervals - Prediction intervals for different alpha levels
   * @return {Object} Average interval width for each alpha level
   */
  calculateIntervalWidths(predictionIntervals) {
    if (!predictionIntervals) {
      return {};
    }
    
    const intervalWidths = {};
    
    Object.entries(predictionIntervals).forEach(([alpha, intervals]) => {
      if (!intervals.lowerBounds || !intervals.upperBounds ||
          intervals.lowerBounds.length !== intervals.upperBounds.length ||
          intervals.lowerBounds.length === 0) {
        intervalWidths[alpha] = null;
        return;
      }
      
      // Calculate width for each interval
      const widths = [];
      for (let i = 0; i < intervals.lowerBounds.length; i++) {
        widths.push(intervals.upperBounds[i] - intervals.lowerBounds[i]);
      }
      
      // Calculate average width
      intervalWidths[alpha] = widths.reduce((sum, w) => sum + w, 0) / widths.length;
    });
    
    return intervalWidths;
  }
  
  /**
   * Calculate average interval width across alpha levels
   * @param {Object} intervalWidths - Interval widths for each alpha level
   * @return {number} Average interval width
   */
  calculateAverageIntervalWidth(intervalWidths) {
    if (!intervalWidths || Object.keys(intervalWidths).length === 0) {
      return null;
    }
    
    const widths = Object.values(intervalWidths).filter(w => w !== null);
    
    if (widths.length === 0) return null;
    
    return widths.reduce((sum, w) => sum + w, 0) / widths.length;
  }
  
  /**
   * Generate prediction intervals from standard deviations
   * @param {Array} predictions - Point predictions
   * @param {Array} stdDeviations - Standard deviations for each prediction
   * @param {Array} alphaLevels - Alpha levels for intervals (default: [0.01, 0.05, 0.1, 0.2, 0.5])
   * @return {Object} Prediction intervals for each alpha level
   */
  generatePredictionIntervals(predictions, stdDeviations, alphaLevels = this.defaultAlphaLevels) {
    if (!predictions || !stdDeviations || 
        predictions.length !== stdDeviations.length ||
        predictions.length === 0) {
      return {};
    }
    
    const predictionIntervals = {};
    
    // Calculate z-scores for each alpha level
    alphaLevels.forEach(alpha => {
      // z-score for (1 - alpha/2) quantile of standard normal distribution
      // For 95% confidence interval (alpha = 0.05), z = 1.96
      const z = this.getZScore(alpha);
      
      const lowerBounds = [];
      const upperBounds = [];
      
      // Calculate prediction intervals
      for (let i = 0; i < predictions.length; i++) {
        const margin = z * stdDeviations[i];
        lowerBounds.push(predictions[i] - margin);
        upperBounds.push(predictions[i] + margin);
      }
      
      predictionIntervals[alpha] = {
        lowerBounds,
        upperBounds
      };
    });
    
    return predictionIntervals;
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
   * Calculate the Continuous Ranked Probability Score (CRPS)
   * @param {Array} lowerBounds - Lower bounds of prediction intervals
   * @param {Array} upperBounds - Upper bounds of prediction intervals
   * @param {Array} actualValues - Actual observed values
   * @return {number} CRPS score
   */
  calculateCRPS(lowerBounds, upperBounds, actualValues) {
    if (!lowerBounds || !upperBounds || !actualValues || 
        lowerBounds.length !== upperBounds.length || 
        lowerBounds.length !== actualValues.length ||
        lowerBounds.length === 0) {
      return null;
    }
    
    // Simplified CRPS calculation based on prediction intervals
    let crpsSum = 0;
    
    for (let i = 0; i < actualValues.length; i++) {
      const actual = actualValues[i];
      const lower = lowerBounds[i];
      const upper = upperBounds[i];
      const width = upper - lower;
      
      if (actual < lower) {
        // Actual value below the interval
        crpsSum += lower - actual;
      } else if (actual > upper) {
        // Actual value above the interval
        crpsSum += actual - upper;
      } else {
        // Actual value within the interval
        // Simplified formula for uniform distribution within the interval
        const normalizedPos = (actual - lower) / width;
        crpsSum += width * (normalizedPos * normalizedPos + (1 - normalizedPos) * (1 - normalizedPos)) / 2;
      }
    }
    
    return crpsSum / actualValues.length;
  }
}

export default CoverageAnalysis;