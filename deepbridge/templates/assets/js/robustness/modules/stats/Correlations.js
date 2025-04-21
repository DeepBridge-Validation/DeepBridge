/**
 * Correlation Analysis
 * 
 * Utilities for calculating correlations and relationships
 * between variables, including Pearson, Spearman, and regression.
 */
import StatsCalculator from './StatsCalculator.js';

class Correlation {
  /**
   * Initialize the correlation analyzer
   */
  constructor() {
    // Initialize stats calculator
    this.statsCalculator = new StatsCalculator();
  }
  
  /**
   * Calculate Pearson correlation coefficient
   * @param {Array} x - First array of values
   * @param {Array} y - Second array of values
   * @return {number} Correlation coefficient (-1 to 1)
   */
  calculatePearson(x, y) {
    if (!x || !y || x.length !== y.length || x.length < 2) {
      return null;
    }
    
    // Calculate means
    const meanX = x.reduce((sum, val) => sum + val, 0) / x.length;
    const meanY = y.reduce((sum, val) => sum + val, 0) / y.length;
    
    // Calculate covariance and variances
    let covariance = 0;
    let varianceX = 0;
    let varianceY = 0;
    
    for (let i = 0; i < x.length; i++) {
      const xDiff = x[i] - meanX;
      const yDiff = y[i] - meanY;
      
      covariance += xDiff * yDiff;
      varianceX += xDiff * xDiff;
      varianceY += yDiff * yDiff;
    }
    
    // Check for zero variance
    if (varianceX === 0 || varianceY === 0) {
      return 0; // No correlation if one variable doesn't vary
    }
    
    // Calculate correlation coefficient
    return covariance / (Math.sqrt(varianceX) * Math.sqrt(varianceY));
  }
  
  /**
   * Calculate Spearman's rank correlation coefficient
   * @param {Array} x - First array of values
   * @param {Array} y - Second array of values
   * @return {number} Rank correlation coefficient (-1 to 1)
   */
  calculateSpearman(x, y) {
    if (!x || !y || x.length !== y.length || x.length < 2) {
      return null;
    }
    
    // Rank the values
    const xRanks = this.rankValues(x);
    const yRanks = this.rankValues(y);
    
    // Calculate Pearson correlation on the ranks
    return this.calculatePearson(xRanks, yRanks);
  }
  
  /**
   * Rank values (convert to 1-based ranks)
   * @param {Array} values - Array of values to rank
   * @return {Array} Array of ranks
   */
  rankValues(values) {
    if (!values || values.length === 0) {
      return [];
    }
    
    // Create array of [value, index] pairs
    const indexed = values.map((value, index) => [value, index]);
    
    // Sort by value
    indexed.sort((a, b) => a[0] - b[0]);
    
    // Assign ranks (handling ties with average rank)
    const ranks = new Array(values.length);
    let i = 0;
    
    while (i < indexed.length) {
      const value = indexed[i][0];
      let j = i + 1;
      
      // Find all elements with the same value
      while (j < indexed.length && indexed[j][0] === value) {
        j++;
      }
      
      // Assign average rank to ties
      const rank = (i + j - 1) / 2 + 1; // Average of positions (+1 for 1-based rank)
      
      for (let k = i; k < j; k++) {
        ranks[indexed[k][1]] = rank;
      }
      
      i = j;
    }
    
    return ranks;
  }
  
  /**
   * Calculate linear regression coefficients
   * @param {Array} x - Independent variable values
   * @param {Array} y - Dependent variable values
   * @return {Object} Regression coefficients and statistics
   */
  calculateLinearRegression(x, y) {
    if (!x || !y || x.length !== y.length || x.length < 2) {
      return {
        slope: null,
        intercept: null,
        rSquared: null
      };
    }
    
    // Calculate means
    const meanX = x.reduce((sum, val) => sum + val, 0) / x.length;
    const meanY = y.reduce((sum, val) => sum + val, 0) / y.length;
    
    // Calculate sums for regression formula
    let sumXY = 0;
    let sumXX = 0;
    let sumYY = 0;
    
    for (let i = 0; i < x.length; i++) {
      const xDiff = x[i] - meanX;
      const yDiff = y[i] - meanY;
      
      sumXY += xDiff * yDiff;
      sumXX += xDiff * xDiff;
      sumYY += yDiff * yDiff;
    }
    
    // Check for zero variance in x
    if (sumXX === 0) {
      return {
        slope: 0,
        intercept: meanY,
        rSquared: 0
      };
    }
    
    // Calculate slope and intercept
    const slope = sumXY / sumXX;
    const intercept = meanY - slope * meanX;
    
    // Calculate R-squared
    const rSquared = Math.pow(sumXY / (Math.sqrt(sumXX) * Math.sqrt(sumYY)), 2);
    
    // Calculate predicted values and residuals
    const predicted = x.map(x_i => slope * x_i + intercept);
    const residuals = y.map((y_i, i) => y_i - predicted[i]);
    
    // Calculate standard error of the slope
    const residualSumSquares = residuals.reduce((sum, res) => sum + res * res, 0);
    const standardError = Math.sqrt(residualSumSquares / (x.length - 2) / sumXX);
    
    return {
      slope,
      intercept,
      rSquared,
      standardError,
      predicted,
      residuals
    };
  }
  
  /**
   * Calculate correlation matrix for multiple variables
   * @param {Array<Array>} data - Array of variable arrays
   * @param {string} method - Correlation method ('pearson' or 'spearman')
   * @return {Array<Array>} Correlation matrix
   */
  calculateCorrelationMatrix(data, method = 'pearson') {
    if (!data || !Array.isArray(data) || data.length === 0) {
      return [];
    }
    
    const variables = data.length;
    const matrix = Array(variables).fill(0).map(() => Array(variables).fill(0));
    
    // Fill the correlation matrix
    for (let i = 0; i < variables; i++) {
      // Diagonal is always 1 (correlation with self)
      matrix[i][i] = 1;
      
      for (let j = i + 1; j < variables; j++) {
        // Calculate correlation
        let correlation;
        if (method.toLowerCase() === 'spearman') {
          correlation = this.calculateSpearman(data[i], data[j]);
        } else {
          correlation = this.calculatePearson(data[i], data[j]);
        }
        
        // Matrix is symmetric
        matrix[i][j] = correlation;
        matrix[j][i] = correlation;
      }
    }
    
    return matrix;
  }
  
  /**
   * Calculate Kendall's Tau (rank correlation)
   * @param {Array} x - First array of values
   * @param {Array} y - Second array of values
   * @return {number} Kendall's Tau coefficient
   */
  calculateKendallTau(x, y) {
    if (!x || !y || x.length !== y.length || x.length < 2) {
      return null;
    }
    
    const n = x.length;
    let concordant = 0;
    let discordant = 0;
    
    // Count concordant and discordant pairs
    for (let i = 0; i < n - 1; i++) {
      for (let j = i + 1; j < n; j++) {
        const xDiff = x[i] - x[j];
        const yDiff = y[i] - y[j];
        
        if (xDiff * yDiff > 0) {
          concordant++;
        } else if (xDiff * yDiff < 0) {
          discordant++;
        }
        // Tied pairs are ignored
      }
    }
    
    // Calculate Kendall's Tau
    return (concordant - discordant) / (0.5 * n * (n - 1));
  }
  
  /**
   * Calculate the coefficient of determination (R-squared)
   * @param {Array} actual - Actual values
   * @param {Array} predicted - Predicted values
   * @return {number} R-squared value
   */
  calculateRSquared(actual, predicted) {
    if (!actual || !predicted || actual.length !== predicted.length || actual.length < 2) {
      return null;
    }
    
    // Calculate mean of actual values
    const mean = actual.reduce((sum, val) => sum + val, 0) / actual.length;
    
    // Calculate sum of squares
    let totalSumSquares = 0;
    let residualSumSquares = 0;
    
    for (let i = 0; i < actual.length; i++) {
      totalSumSquares += Math.pow(actual[i] - mean, 2);
      residualSumSquares += Math.pow(actual[i] - predicted[i], 2);
    }
    
    // Check for zero variance
    if (totalSumSquares === 0) {
      return 0;
    }
    
    // Calculate R-squared
    return 1 - (residualSumSquares / totalSumSquares);
  }
  
  /**
   * Generate points for a regression line
   * @param {Object} regression - Regression results
   * @param {Array} x - X-axis values
   * @return {Object} Points for the regression line
   */
  generateRegressionLine(regression, x = null) {
    if (!regression || regression.slope === null) {
      return {
        x: [],
        y: []
      };
    }
    
    // If x values not provided, create range based on original data
    let xValues = x;
    if (!xValues && regression.predicted && regression.predicted.length > 0) {
      const originalX = regression.predicted.map((_, i) => i);
      const min = Math.min(...originalX);
      const max = Math.max(...originalX);
      const range = max - min;
      
      // Create evenly spaced points
      xValues = Array(100).fill(0).map((_, i) => min + (i / 99) * range);
    } else if (!xValues) {
      xValues = [0, 1]; // Fallback
    }
    
    // Calculate y values from regression formula
    const yValues = xValues.map(x_i => regression.slope * x_i + regression.intercept);
    
    return {
      x: xValues,
      y: yValues
    };
  }
}

export default Correlation;