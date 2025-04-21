/**
 * Time Series Analysis
 * 
 * Utilities for analyzing temporal patterns in robustness metrics
 * to identify trends, seasonality, and changes over time.
 */
import StatsCalculator from './StatsCalculator.js';
import Correlation from './Correlation.js';

class TimeSeriesAnalysis {
  /**
   * Initialize the time series analyzer
   */
  constructor() {
    // Initialize dependencies
    this.statsCalculator = new StatsCalculator();
    this.correlation = new Correlation();
  }
  
  /**
   * Calculate moving average
   * @param {Array} values - Time series values
   * @param {number} window - Window size for moving average
   * @return {Array} Moving average values
   */
  calculateMovingAverage(values, window = 3) {
    if (!values || values.length < window) {
      return [];
    }
    
    const result = [];
    
    // Calculate moving average
    for (let i = 0; i <= values.length - window; i++) {
      const windowValues = values.slice(i, i + window);
      const average = windowValues.reduce((sum, val) => sum + val, 0) / window;
      result.push(average);
    }
    
    return result;
  }
  
  /**
   * Apply exponential smoothing
   * @param {Array} values - Time series values
   * @param {number} alpha - Smoothing factor (0-1)
   * @return {Array} Smoothed values
   */
  calculateExponentialSmoothing(values, alpha = 0.3) {
    if (!values || values.length === 0) {
      return [];
    }
    
    const result = [values[0]]; // First value is unchanged
    
    // Apply exponential smoothing
    for (let i = 1; i < values.length; i++) {
      const smoothed = alpha * values[i] + (1 - alpha) * result[i - 1];
      result.push(smoothed);
    }
    
    return result;
  }
  
  /**
   * Calculate linear trend
   * @param {Array} values - Time series values
   * @param {Array} times - Time points (optional, defaults to indices)
   * @return {Object} Trend information
   */
  calculateTrend(values, times = null) {
    if (!values || values.length < 2) {
      return {
        slope: null,
        intercept: null,
        trendValues: [],
        detrended: []
      };
    }
    
    // Create time points if not provided
    const timePoints = times || Array.from({ length: values.length }, (_, i) => i);
    
    // Calculate linear regression
    const regression = this.correlation.calculateLinearRegression(timePoints, values);
    
    // Calculate trend values
    const trendValues = timePoints.map(t => regression.slope * t + regression.intercept);
    
    // Calculate detrended values
    const detrended = values.map((val, i) => val - trendValues[i]);
    
    return {
      slope: regression.slope,
      intercept: regression.intercept,
      trendValues,
      detrended
    };
  }
  
  /**
   * Detect seasonality in time series
   * @param {Array} values - Time series values
   * @param {number} maxLag - Maximum lag to test
   * @return {Object} Seasonality information
   */
  detectSeasonality(values, maxLag = 12) {
    if (!values || values.length <= maxLag) {
      return {
        seasonal: false,
        period: null,
        strength: null
      };
    }
    
    // Detrend the series
    const detrended = this.calculateTrend(values).detrended;
    
    // Calculate autocorrelations for different lags
    const autocorrelations = [];
    const maxLagToUse = Math.min(maxLag, Math.floor(values.length / 2));
    
    for (let lag = 1; lag <= maxLagToUse; lag++) {
      const lagged = detrended.slice(0, detrended.length - lag);
      const current = detrended.slice(lag);
      
      const correlation = this.correlation.calculatePearson(lagged, current);
      autocorrelations.push({
        lag,
        correlation
      });
    }
    
    // Sort by correlation strength (descending)
    autocorrelations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
    
    // Check if highest correlation is significant
    const highestCorrelation = autocorrelations[0];
    const isSignificant = Math.abs(highestCorrelation.correlation) > 1.96 / Math.sqrt(values.length);
    
    return {
      seasonal: isSignificant,
      period: isSignificant ? highestCorrelation.lag : null,
      strength: isSignificant ? highestCorrelation.correlation : null,
      autocorrelations
    };
  }
  
  /**
   * Decompose time series into trend, seasonal, and residual components
   * @param {Array} values - Time series values
   * @param {number} period - Seasonal period
   * @return {Object} Decomposed components
   */
  decomposeTimeSeries(values, period = 12) {
    if (!values || values.length < period * 2) {
      return {
        trend: [],
        seasonal: [],
        residual: []
      };
    }
    
    // Step 1: Calculate trend using moving average
    const movingAverageWindow = period % 2 === 0 ? period + 1 : period;
    const trend = this.calculateMovingAverage(values, movingAverageWindow);
    
    // Extend trend to match original series length
    const paddedTrend = Array(Math.floor(movingAverageWindow / 2))
      .fill(trend[0])
      .concat(trend)
      .concat(Array(Math.floor(movingAverageWindow / 2)).fill(trend[trend.length - 1]));
    
    // Cut to original length
    const finalTrend = paddedTrend.slice(0, values.length);
    
    // Step 2: Calculate detrended series
    const detrended = values.map((val, i) => val - finalTrend[i]);
    
    // Step 3: Calculate seasonal component
    const seasonal = Array(values.length).fill(0);
    
    // Calculate average for each position in the seasonal cycle
    for (let i = 0; i < period; i++) {
      const seasonalValues = [];
      
      for (let j = i; j < detrended.length; j += period) {
        seasonalValues.push(detrended[j]);
      }
      
      const seasonalAvg = seasonalValues.reduce((sum, val) => sum + val, 0) / seasonalValues.length;
      
      for (let j = i; j < seasonal.length; j += period) {
        seasonal[j] = seasonalAvg;
      }
    }
    
    // Ensure seasonal component sums to zero
    const seasonalMean = seasonal.reduce((sum, val) => sum + val, 0) / seasonal.length;
    const adjustedSeasonal = seasonal.map(val => val - seasonalMean);
    
    // Step 4: Calculate residual component
    const residual = values.map((val, i) => val - finalTrend[i] - adjustedSeasonal[i]);
    
    return {
      trend: finalTrend,
      seasonal: adjustedSeasonal,
      residual
    };
  }
  
  /**
   * Forecast future values using simple exponential smoothing
   * @param {Array} values - Time series values
   * @param {number} horizon - Forecast horizon
   * @param {number} alpha - Smoothing factor (0-1)
   * @return {Array} Forecasted values
   */
  forecastSimpleExponential(values, horizon = 5, alpha = 0.3) {
    if (!values || values.length === 0) {
      return [];
    }
    
    // Apply exponential smoothing
    const smoothed = this.calculateExponentialSmoothing(values, alpha);
    
    // Use last smoothed value for forecast
    const lastValue = smoothed[smoothed.length - 1];
    const forecast = Array(horizon).fill(lastValue);
    
    return forecast;
  }
  
  /**
   * Forecast future values using Holt-Winters method
   * @param {Array} values - Time series values
   * @param {number} horizon - Forecast horizon
   * @param {number} alpha - Level smoothing factor (0-1)
   * @param {number} beta - Trend smoothing factor (0-1)
   * @param {number} gamma - Seasonal smoothing factor (0-1)
   * @param {number} period - Seasonal period
   * @return {Array} Forecasted values
   */
  forecastHoltWinters(values, horizon = 5, alpha = 0.3, beta = 0.1, gamma = 0.1, period = 12) {
    if (!values || values.length < period * 2) {
      return [];
    }
    
    // Initialize level, trend, and seasonal components
    let level = values.slice(0, period).reduce((sum, val) => sum + val, 0) / period;
    let trend = (values[period] - values[0]) / period;
    
    // Initialize seasonal components
    const seasons = Array(period).fill(0);
    
    for (let i = 0; i < period; i++) {
      seasons[i % period] = values[i] / level;
    }
    
    // Apply Holt-Winters method
    for (let i = period; i < values.length; i++) {
      const oldLevel = level;
      const seasonalIndex = i % period;
      
      // Update level
      level = alpha * (values[i] / seasons[seasonalIndex]) + (1 - alpha) * (oldLevel + trend);
      
      // Update trend
      trend = beta * (level - oldLevel) + (1 - beta) * trend;
      
      // Update seasonal component
      seasons[seasonalIndex] = gamma * (values[i] / level) + (1 - gamma) * seasons[seasonalIndex];
    }
    
    // Generate forecast
    const forecast = [];
    
    for (let i = 0; i < horizon; i++) {
      const forecastIndex = (values.length + i) % period;
      forecast.push((level + (i + 1) * trend) * seasons[forecastIndex]);
    }
    
    return forecast;
  }
  
  /**
   * Calculate anomaly scores for a time series
   * @param {Array} values - Time series values
   * @param {number} window - Window size for moving average
   * @return {Array} Anomaly scores
   */
  calculateAnomalyScores(values, window = 5) {
    if (!values || values.length < window) {
      return [];
    }
    
    const anomalyScores = [];
    
    // Calculate moving average and standard deviation
    for (let i = 0; i < values.length; i++) {
      // Define window bounds
      const start = Math.max(0, i - window);
      const end = Math.min(values.length, i + window + 1);
      
      // Get window values (excluding current point)
      const windowValues = [
        ...values.slice(start, i),
        ...values.slice(i + 1, end)
      ];
      
      // Calculate statistics
      const stats = this.statsCalculator.calculateSummary(windowValues);
      
      // Calculate z-score
      const zScore = stats.stdDev > 0 ? 
        Math.abs((values[i] - stats.mean) / stats.stdDev) : 
        0;
      
      anomalyScores.push(zScore);
    }
    
    return anomalyScores;
  }
}

export default TimeSeriesAnalysis;