/**
 * Chart Utilities
 * 
 * Utility functions for chart rendering and data processing
 * in the resilience report visualizations.
 */

class ChartUtils {
    /**
     * Initialize the chart utilities
     */
    constructor() {
      // Color palettes for different chart types
      this.colorPalettes = {
        categorical: [
          '#4e79a7', '#f28e2c', '#e15759', '#76b7b2', '#59a14f',
          '#edc949', '#af7aa1', '#ff9da7', '#9c755f', '#bab0ab'
        ],
        sequential: {
          blues: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'],
          greens: ['#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b'],
          reds: ['#fff5f0', '#fee0d2', '#fcbba1', '#fc9272', '#fb6a4a', '#ef3b2c', '#cb181d', '#a50f15', '#67000d']
        },
        diverging: {
          redBlue: ['#053061', '#2166ac', '#4393c3', '#92c5de', '#f7f7f7', '#f4a582', '#d6604d', '#b2182b', '#67001f'],
          redGreen: ['#006d2c', '#2ca25f', '#66c2a4', '#b2e2e2', '#f7f7f7', '#fddbc7', '#ef8a62', '#d7301f', '#b2182b'],
          purpleGreen: ['#40004b', '#762a83', '#9970ab', '#c2a5cf', '#f7f7f7', '#a6dba0', '#5aae61', '#1b7837', '#00441b']
        }
      };
    }
    
    /**
     * Generate a color for a category
     * @param {number} index - Category index
     * @param {number} count - Total number of categories
     * @return {string} Color string
     */
    getCategoryColor(index, count = 10) {
      // Use categorical palette for small number of categories
      if (count <= this.colorPalettes.categorical.length) {
        return this.colorPalettes.categorical[index % this.colorPalettes.categorical.length];
      }
      
      // Generate colors based on hue for larger sets
      const hue = (index * 360 / count) % 360;
      return `hsl(${hue}, 70%, 60%)`;
    }
    
    /**
     * Generate a color scale based on values
     * @param {number} value - Value between 0 and 1
     * @param {string} scale - Scale name ('reds', 'blues', 'greens', etc.)
     * @return {string} Color string
     */
    getColorFromScale(value, scale = 'blues') {
      // Ensure value is between 0 and 1
      const safeValue = Math.max(0, Math.min(1, value));
      
      // Get the color scale
      let palette;
      if (scale.includes('-')) {
        // Use diverging palette
        const [from, to] = scale.split('-');
        palette = this.colorPalettes.diverging[`${from}${to.charAt(0).toUpperCase() + to.slice(1)}`];
      } else {
        // Use sequential palette
        palette = this.colorPalettes.sequential[scale.toLowerCase()];
      }
      
      if (!palette) {
        palette = this.colorPalettes.sequential.blues;
      }
      
      // Calculate the index in the palette
      const index = Math.floor(safeValue * (palette.length - 1));
      return palette[index];
    }
    
    /**
     * Get a color for a performance score
     * @param {number} score - Score value (0-1)
     * @return {string} Color string
     */
    getPerformanceColor(score) {
      if (score >= 0.9) {
        return '#2ecc71'; // Excellent - Green
      } else if (score >= 0.8) {
        return '#27ae60'; // Good - Dark Green
      } else if (score >= 0.7) {
        return '#f1c40f'; // Moderate - Yellow
      } else if (score >= 0.6) {
        return '#f39c12'; // Fair - Orange
      } else {
        return '#e74c3c'; // Poor - Red
      }
    }
    
    /**
     * Calculate percentage decrease from base
     * @param {number} value - Current value
     * @param {number} base - Base value
     * @return {string} Formatted percentage
     */
    formatChangeFromBase(value, base) {
      if (!base) return '0%';
      
      const change = (value - base) / base * 100;
      const sign = change >= 0 ? '+' : '';
      return `${sign}${change.toFixed(2)}%`;
    }
    
    /**
     * Format a number with specified precision
     * @param {number} value - Value to format
     * @param {number} precision - Decimal precision
     * @return {string} Formatted number
     */
    formatNumber(value, precision = 4) {
      if (value === undefined || value === null) return 'N/A';
      return value.toFixed(precision);
    }
    
    /**
     * Generate synthetic data for distribution visualization
     * @param {Object} stats - Statistical properties
     * @param {number} count - Number of points to generate
     * @return {Array} Generated data points
     */
    generateDistributionData(stats, count = 100) {
      const { mean = 0, stdDev = 1 } = stats;
      const data = [];
      
      // Generate normal distribution using Box-Muller transform
      for (let i = 0; i < count; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        const value = mean + z * stdDev;
        data.push(value);
      }
      
      return data;
    }
    
    /**
     * Calculate kernel density estimation
     * @param {Array} data - Input data
     * @param {Array} grid - Grid points to evaluate density
     * @param {number} bandwidth - Kernel bandwidth
     * @return {Array} Density values at grid points
     */
    calculateKDE(data, grid, bandwidth = null) {
      if (!data || data.length === 0 || !grid || grid.length === 0) {
        return Array(grid.length).fill(0);
      }
      
      // Calculate standard deviation if bandwidth not provided
      if (!bandwidth) {
        const mean = data.reduce((a, b) => a + b, 0) / data.length;
        const variance = data.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / data.length;
        const stdDev = Math.sqrt(variance);
        
        // Silverman's rule of thumb
        bandwidth = 1.06 * stdDev * Math.pow(data.length, -0.2);
      }
      
      // Calculate density at each grid point
      return grid.map(x => {
        // Sum of kernel functions at x
        const sum = data.reduce((acc, xi) => {
          // Gaussian kernel
          const z = (x - xi) / bandwidth;
          return acc + Math.exp(-0.5 * z * z);
        }, 0);
        
        // Normalize by bandwidth and data length
        return sum / (bandwidth * Math.sqrt(2 * Math.PI) * data.length);
      });
    }
    
    /**
/**
   * Calculate distance between two distributions
   * @param {Array} dist1 - First distribution values
   * @param {Array} dist2 - Second distribution values
   * @param {string} method - Distance method ('kl', 'js', 'wasserstein')
   * @return {number} Distance value
   */
calculateDistributionDistance(dist1, dist2, method = 'kl') {
    if (!dist1 || !dist2 || dist1.length === 0 || dist2.length === 0) {
      return null;
    }
    
    // Create histogram representations with equal bins
    const min = Math.min(Math.min(...dist1), Math.min(...dist2));
    const max = Math.max(Math.max(...dist1), Math.max(...dist2));
    const binCount = 20;
    const binWidth = (max - min) / binCount;
    
    // Count samples in each bin
    const hist1 = Array(binCount).fill(0);
    const hist2 = Array(binCount).fill(0);
    
    dist1.forEach(val => {
      const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
      hist1[binIndex]++;
    });
    
    dist2.forEach(val => {
      const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
      hist2[binIndex]++;
    });
    
    // Normalize histograms to get probability distributions
    const sum1 = hist1.reduce((a, b) => a + b, 0);
    const sum2 = hist2.reduce((a, b) => a + b, 0);
    
    const p1 = hist1.map(count => count / sum1);
    const p2 = hist2.map(count => count / sum2);
    
    // Calculate distance based on specified method
    switch (method.toLowerCase()) {
      case 'kl': // Kullback-Leibler divergence
        return this.calculateKLDivergence(p1, p2);
      case 'js': // Jensen-Shannon divergence
        return this.calculateJSDivergence(p1, p2);
      case 'wasserstein': // Wasserstein distance (Earth Mover's Distance)
        return this.calculateWassersteinDistance(p1, p2);
      default:
        return this.calculateKLDivergence(p1, p2);
    }
  }
  
  /**
   * Calculate Kullback-Leibler divergence
   * @param {Array} p - First probability distribution
   * @param {Array} q - Second probability distribution
   * @return {number} KL divergence value
   */
  calculateKLDivergence(p, q) {
    // KL divergence: sum(p(i) * log(p(i) / q(i)))
    let divergence = 0;
    
    for (let i = 0; i < p.length; i++) {
      // Handle zero probabilities
      if (p[i] === 0) continue;
      
      // Add small epsilon to avoid division by zero
      const qVal = q[i] === 0 ? 1e-10 : q[i];
      divergence += p[i] * Math.log(p[i] / qVal);
    }
    
    return divergence;
  }
  
  /**
   * Calculate Jensen-Shannon divergence
   * @param {Array} p - First probability distribution
   * @param {Array} q - Second probability distribution
   * @return {number} JS divergence value
   */
  calculateJSDivergence(p, q) {
    // JS divergence: 0.5 * (KL(p||m) + KL(q||m)) where m = (p+q)/2
    const m = p.map((val, i) => (val + q[i]) / 2);
    
    const klPM = this.calculateKLDivergence(p, m);
    const klQM = this.calculateKLDivergence(q, m);
    
    return 0.5 * (klPM + klQM);
  }
  
  /**
   * Calculate Wasserstein distance (Earth Mover's Distance)
   * @param {Array} p - First probability distribution
   * @param {Array} q - Second probability distribution
   * @return {number} Wasserstein distance value
   */
  calculateWassersteinDistance(p, q) {
    // Cumulative distributions
    const cdf1 = [];
    const cdf2 = [];
    
    let sum1 = 0;
    let sum2 = 0;
    
    for (let i = 0; i < p.length; i++) {
      sum1 += p[i];
      sum2 += q[i];
      cdf1.push(sum1);
      cdf2.push(sum2);
    }
    
    // Calculate Wasserstein distance as area between CDFs
    let distance = 0;
    for (let i = 0; i < cdf1.length; i++) {
      distance += Math.abs(cdf1[i] - cdf2[i]);
    }
    
    return distance;
  }
  
  /**
   * Calculate statistics for a distribution
   * @param {Array} data - Distribution values
   * @return {Object} Statistics object
   */
  calculateDistributionStats(data) {
    if (!data || data.length === 0) {
      return {
        count: 0,
        min: null,
        max: null,
        mean: null,
        median: null,
        stdDev: null
      };
    }
    
    // Sort data for percentile calculations
    const sortedData = [...data].sort((a, b) => a - b);
    
    // Basic statistics
    const count = data.length;
    const min = sortedData[0];
    const max = sortedData[count - 1];
    const sum = data.reduce((acc, val) => acc + val, 0);
    const mean = sum / count;
    
    // Calculate median
    const midPoint = Math.floor(count / 2);
    const median = count % 2 === 0 
      ? (sortedData[midPoint - 1] + sortedData[midPoint]) / 2
      : sortedData[midPoint];
    
    // Calculate standard deviation
    const variance = data.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / count;
    const stdDev = Math.sqrt(variance);
    
    return {
      count,
      min,
      max,
      mean,
      median,
      stdDev
    };
  }
  
  /**
   * Generate a grid of points for density estimation
   * @param {Array} data - Distribution values
   * @param {number} pointCount - Number of grid points
   * @return {Array} Grid points
   */
  generateDensityGrid(data, pointCount = 100) {
    if (!data || data.length === 0) {
      return [];
    }
    
    // Get min and max with padding
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min;
    const padding = range * 0.1;
    
    // Create evenly spaced grid
    const grid = [];
    const start = min - padding;
    const end = max + padding;
    const step = (end - start) / (pointCount - 1);
    
    for (let i = 0; i < pointCount; i++) {
      grid.push(start + i * step);
    }
    
    return grid;
  }
  
  /**
   * Format a value as a percentage
   * @param {number} value - Value to format
   * @param {number} precision - Decimal precision
   * @return {string} Formatted percentage
   */
  formatPercent(value, precision = 2) {
    if (value === undefined || value === null) return 'N/A';
    return `${(value * 100).toFixed(precision)}%`;
  }
  
  /**
   * Generate a gradient color between two colors
   * @param {string} color1 - Start color in hex format
   * @param {string} color2 - End color in hex format
   * @param {number} ratio - Ratio between colors (0-1)
   * @return {string} Interpolated color in hex format
   */
  interpolateColor(color1, color2, ratio) {
    // Convert hex to RGB
    const parseColor = (hexStr) => {
      const hex = hexStr.replace('#', '');
      const r = parseInt(hex.substring(0, 2), 16);
      const g = parseInt(hex.substring(2, 4), 16);
      const b = parseInt(hex.substring(4, 6), 16);
      return [r, g, b];
    };
    
    // Convert RGB to hex
    const toHex = (r, g, b) => {
      const componentToHex = (c) => {
        const hex = c.toString(16);
        return hex.length === 1 ? '0' + hex : hex;
      };
      
      return `#${componentToHex(r)}${componentToHex(g)}${componentToHex(b)}`;
    };
    
    // Interpolate colors
    const [r1, g1, b1] = parseColor(color1);
    const [r2, g2, b2] = parseColor(color2);
    
    const r = Math.round(r1 + (r2 - r1) * ratio);
    const g = Math.round(g1 + (g2 - g1) * ratio);
    const b = Math.round(b1 + (b2 - b1) * ratio);
    
    return toHex(r, g, b);
  }
}

export default ChartUtils;