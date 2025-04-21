/**
 * Distribution Analysis
 * 
 * Utilities for analyzing and generating probability distributions,
 * including normal, uniform, and empirical distributions.
 */
import StatsCalculator from './StatsCalculator.js';

class Distribution {
  /**
   * Initialize the distribution analyzer
   */
  constructor() {
    // Initialize stats calculator
    this.statsCalculator = new StatsCalculator();
    
    // Default distribution parameters
    this.defaultParams = {
      sampleSize: 100,   // Number of samples to generate
      seed: null,        // Random seed for reproducibility
      bounds: {          // Bounds for bounded distributions
        min: 0,
        max: 1
      }
    };
  }
  
  /**
   * Generate samples from a normal distribution
   * @param {number} mean - Mean of the distribution
   * @param {number} stdDev - Standard deviation
   * @param {Object} options - Generation options
   * @return {Array} Generated samples
   */
  generateNormal(mean, stdDev, options = {}) {
    const params = { ...this.defaultParams, ...options };
    const samples = [];
    
    for (let i = 0; i < params.sampleSize; i++) {
      // Box-Muller transform to generate normally distributed random numbers
      const u1 = Math.random();
      const u2 = Math.random();
      
      // Generate standard normal value (mean=0, stdDev=1)
      const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      
      // Transform to desired distribution
      let sample = mean + stdDev * z;
      
      // Apply bounds if specified
      if (params.bounded) {
        sample = Math.max(params.bounds.min, Math.min(params.bounds.max, sample));
      }
      
      samples.push(sample);
    }
    
    return samples;
  }
  
  /**
   * Generate samples from a uniform distribution
   * @param {number} min - Minimum value
   * @param {number} max - Maximum value
   * @param {Object} options - Generation options
   * @return {Array} Generated samples
   */
  generateUniform(min, max, options = {}) {
    const params = { ...this.defaultParams, ...options };
    const samples = [];
    
    for (let i = 0; i < params.sampleSize; i++) {
      // Generate uniform value between min and max
      const sample = min + Math.random() * (max - min);
      samples.push(sample);
    }
    
    return samples;
  }
  
  /**
   * Generate samples from a skewed distribution
   * @param {number} mean - Mean of the distribution
   * @param {number} stdDev - Standard deviation
   * @param {number} skew - Skewness parameter (-1 to 1)
   * @param {Object} options - Generation options
   * @return {Array} Generated samples
   */
  generateSkewed(mean, stdDev, skew = 0, options = {}) {
    const params = { ...this.defaultParams, ...options };
    const samples = [];
    
    for (let i = 0; i < params.sampleSize; i++) {
      // Generate base normal sample
      const u1 = Math.random();
      const u2 = Math.random();
      let z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      
      // Apply skew
      if (skew !== 0) {
        // Generate skewed sample using the method of powers
        const sign = z >= 0 ? 1 : -1;
        const absZ = Math.abs(z);
        
        // Skew the distribution by raising to a power
        // For positive skew, compress large negative values and expand large positive values
        // For negative skew, do the opposite
        const power = skew >= 0 ? 1 - skew : 1 / (1 + skew);
        z = sign * Math.pow(absZ, power);
      }
      
      // Transform to desired distribution
      let sample = mean + stdDev * z;
      
      // Apply bounds if specified
      if (params.bounded) {
        sample = Math.max(params.bounds.min, Math.min(params.bounds.max, sample));
      }
      
      samples.push(sample);
    }
    
    return samples;
  }
  
  /**
   * Generate samples from an empirical distribution
   * @param {Array} sourceData - Source data to sample from
   * @param {Object} options - Generation options
   * @return {Array} Generated samples
   */
  generateEmpirical(sourceData, options = {}) {
    if (!sourceData || sourceData.length === 0) {
      return [];
    }
    
    const params = { ...this.defaultParams, ...options };
    const samples = [];
    
    // Calculate statistics if needed for resampling with noise
    let stats = null;
    if (params.addNoise) {
      stats = this.statsCalculator.calculateSummary(sourceData);
    }
    
    for (let i = 0; i < params.sampleSize; i++) {
      // Sample from source data
      const randomIndex = Math.floor(Math.random() * sourceData.length);
      let sample = sourceData[randomIndex];
      
      // Add noise if requested
      if (params.addNoise && stats && stats.stdDev) {
        // Add small random noise based on the standard deviation
        const noise = (Math.random() - 0.5) * stats.stdDev * (params.noiseLevel || 0.1);
        sample += noise;
        
        // Apply bounds if specified
        if (params.bounded) {
          sample = Math.max(params.bounds.min, Math.min(params.bounds.max, sample));
        }
      }
      
      samples.push(sample);
    }
    
    return samples;
  }
  
  /**
   * Generate synthetic distribution data based on summary statistics
   * @param {Object} stats - Statistical summary (mean, stdDev, min, max, etc.)
   * @param {string} type - Distribution type ('normal', 'uniform', 'skewed', etc.)
   * @param {Object} options - Generation options
   * @return {Array} Generated samples
   */
  generateFromStats(stats, type = 'normal', options = {}) {
    if (!stats || !stats.mean || !stats.stdDev) {
      return [];
    }
    
    // Set bounds from stats if not provided
    const boundedOptions = {
      ...options,
      bounded: true,
      bounds: options.bounds || {
        min: stats.min,
        max: stats.max
      }
    };
    
    // Generate based on distribution type
    switch (type.toLowerCase()) {
      case 'normal':
        return this.generateNormal(stats.mean, stats.stdDev, boundedOptions);
      case 'uniform':
        return this.generateUniform(stats.min, stats.max, boundedOptions);
      case 'skewed':
        return this.generateSkewed(stats.mean, stats.stdDev, options.skew || 0, boundedOptions);
      default:
        return this.generateNormal(stats.mean, stats.stdDev, boundedOptions);
    }
  }
  
  /**
   * Calculate probability density function (PDF) values for a distribution
   * @param {Array} values - Array of x values to calculate density for
   * @param {Object} stats - Distribution statistics (mean, stdDev)
   * @param {string} type - Distribution type
   * @return {Array} PDF values
   */
  calculatePDF(values, stats, type = 'normal') {
    if (!values || values.length === 0 || !stats) {
      return [];
    }
    
    switch (type.toLowerCase()) {
      case 'normal':
        return this.calculateNormalPDF(values, stats.mean, stats.stdDev);
      case 'uniform':
        return this.calculateUniformPDF(values, stats.min, stats.max);
      default:
        return this.calculateNormalPDF(values, stats.mean, stats.stdDev);
    }
  }
  
  /**
   * Calculate normal PDF values
   * @param {Array} values - X values
   * @param {number} mean - Mean of the distribution
   * @param {number} stdDev - Standard deviation
   * @return {Array} PDF values
   */
  calculateNormalPDF(values, mean, stdDev) {
    if (!values || values.length === 0 || !stdDev) {
      return [];
    }
    
    const factor = 1 / (stdDev * Math.sqrt(2 * Math.PI));
    
    return values.map(x => {
      const exponent = -0.5 * Math.pow((x - mean) / stdDev, 2);
      return factor * Math.exp(exponent);
    });
  }
  
  /**
   * Calculate uniform PDF values
   * @param {Array} values - X values
   * @param {number} min - Minimum value
   * @param {number} max - Maximum value
   * @return {Array} PDF values
   */
  calculateUniformPDF(values, min, max) {
    if (!values || values.length === 0 || min === max) {
      return [];
    }
    
    const density = 1 / (max - min);
    
    return values.map(x => {
      return (x >= min && x <= max) ? density : 0;
    });
  }
  
  /**
   * Calculate histogram bins from data
   * @param {Array} data - Input data array
   * @param {number} binCount - Number of bins (or 'auto' for automatic binning)
   * @return {Object} Histogram data with bins and counts
   */
  calculateHistogram(data, binCount = 'auto') {
    if (!data || data.length === 0) {
      return {
        bins: [],
        counts: [],
        binWidth: null
      };
    }
    
    // Calculate range
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min;
    
    // Determine number of bins
    let bins;
    if (binCount === 'auto') {
      // Sturges' formula for bin count
      bins = Math.ceil(Math.log2(data.length) + 1);
    } else {
      bins = Math.max(1, Math.floor(binCount));
    }
    
    // Calculate bin width
    const binWidth = range / bins;
    
    // Create bin edges
    const binEdges = Array(bins + 1).fill(0).map((_, i) => min + i * binWidth);
    
    // Count values in each bin
    const counts = Array(bins).fill(0);
    
    data.forEach(value => {
      if (value === max) {
        // Edge case: maximum value goes in the last bin
        counts[bins - 1]++;
      } else {
        // Find appropriate bin
        const binIndex = Math.floor((value - min) / binWidth);
        if (binIndex >= 0 && binIndex < bins) {
          counts[binIndex]++;
        }
      }
    });
    
    // Create bin objects
    const binData = Array(bins).fill(0).map((_, i) => ({
      start: binEdges[i],
      end: binEdges[i + 1],
      count: counts[i],
      density: counts[i] / (data.length * binWidth)
    }));
    
    return {
      binData,
      binEdges,
      counts,
      binWidth
    };
  }
  
  /**
   * Calculate the kernel density estimate (KDE) for a distribution
   * @param {Array} data - Input data array
   * @param {Array} x - Points at which to evaluate the KDE
   * @param {number} bandwidth - Bandwidth parameter (or 'auto')
   * @return {Array} KDE values
   */
  calculateKDE(data, x, bandwidth = 'auto') {
    if (!data || data.length === 0 || !x || x.length === 0) {
      return [];
    }
    
    // Calculate standard deviation
    const std = this.statsCalculator.calculateVariance(data).stdDev;
    
    if (std === null || std === 0) {
      return x.map(() => 0);
    }
    
    // Determine bandwidth
    let h;
    if (bandwidth === 'auto') {
      // Silverman's rule of thumb
      h = 1.06 * std * Math.pow(data.length, -0.2);
    } else {
      h = bandwidth;
    }
    
    // Calculate KDE for each point
    return x.map(xi => {
      // Sum of kernels
      const kernelSum = data.reduce((sum, dataPoint) => {
        // Gaussian kernel
        const z = (xi - dataPoint) / h;
        const kernel = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
        return sum + kernel;
      }, 0);
      
      // Normalize by bandwidth and data length
      return kernelSum / (h * data.length);
    });
  }
  
  /**
   * Generate a violin plot dataset from distribution data
   * @param {Array} data - Input data array
   * @param {Object} options - Options for violin plot generation
   * @return {Object} Violin plot dataset
   */
  generateViolinPlotData(data, options = {}) {
    if (!data || data.length === 0) {
      return {
        x: [],
        y: [],
        stats: null
      };
    }
    
    const params = {
      points: 50,      // Number of points for the violin curve
      bandwidth: 'auto', // Bandwidth for KDE
      ...options
    };
    
    // Calculate statistics
    const stats = this.statsCalculator.calculateSummary(data);
    
    // Define the points at which to evaluate the KDE
    const padding = stats.iqr * 1.5;
    const min = Math.max(0, stats.min - padding);
    const max = stats.max + padding;
    
    const y = Array(params.points).fill(0).map((_, i) => 
      min + (i / (params.points - 1)) * (max - min)
    );
    
    // Calculate KDE
    const densities = this.calculateKDE(data, y, params.bandwidth);
    
    // Scale densities for better visualization
    const maxDensity = Math.max(...densities);
    const scaleFactor = options.width || 1;
    
    // Create the violin plot data (mirrored KDE)
    const violinData = {
      y: [],   // Y-coordinates (the actual values)
      x: [],   // X-coordinates (the densities, mirrored)
      stats    // Statistical summary
    };
    
    // Build mirrored KDE curve
    y.forEach((yVal, i) => {
      const density = densities[i] / maxDensity * scaleFactor;
      
      // Add right side of violin
      violinData.y.push(yVal);
      violinData.x.push(density);
      
      // Add left side of violin (mirrored)
      violinData.y.push(yVal);
      violinData.x.push(-density);
    });
    
    return violinData;
  }
}

export default Distribution;