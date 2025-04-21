// DistributionData.js - placeholder
/**
 * Distribution Data Handler
 * 
 * Specialized handler for extracting and processing distribution data
 * from the resilience report structure.
 */

class DistributionData {
    /**
     * Initialize the distribution data handler
     */
    constructor() {
      // Default shift types for fallback
      this.defaultShiftTypes = ["covariate", "concept", "prediction"];
    }
    
    /**
     * Extract distribution data from report data
     * @param {Object} reportData - The full report data object
     * @return {Object|null} Extracted distribution data or null if not available
     */
    extract(reportData) {
      if (!reportData) return null;
      
      let distributionData = null;
      
      // Check if distribution data is directly in reportData
      if (reportData.distribution_data) {
        distributionData = this.extractDistributionData(reportData.distribution_data);
      } 
      // Check if distribution data is in shifts
      else if (reportData.shifts && reportData.shifts.length > 0 && reportData.shifts[0].distributions) {
        distributionData = this.extractDistributionDataFromShifts(reportData.shifts);
      }
      // Check if in primary_model
      else if (reportData.primary_model && reportData.primary_model.shifts) {
        distributionData = this.extractDistributionDataFromShifts(reportData.primary_model.shifts);
      }
      
      return distributionData;
    }
    
    /**
     * Extract distribution data from dedicated distribution_data object
     * @param {Object} distributionData - Distribution data object
     * @return {Object} Normalized distribution data
     */
    extractDistributionData(distributionData) {
      if (!distributionData || !distributionData.shifts) {
        return null;
      }
      
      const shifts = [];
      
      // Process each shift type
      Object.entries(distributionData.shifts).forEach(([shiftType, shiftData]) => {
        // Skip if no distributions
        if (!shiftData.distributions) return;
        
        // Extract distributions data
        const distributions = [];
        
        if (shiftData.distributions.original) {
          distributions.push({
            name: 'original',
            values: shiftData.distributions.original.values || [],
            statistics: shiftData.distributions.original.statistics || null
          });
        }
        
        if (shiftData.distributions.shifted) {
          distributions.push({
            name: 'shifted',
            values: shiftData.distributions.shifted.values || [],
            statistics: shiftData.distributions.shifted.statistics || null
          });
        }
        
        // Add metrics if available
        const metrics = shiftData.metrics || {};
        
        shifts.push({
          type: shiftType,
          distributions,
          metrics
        });
      });
      
      return {
        shifts
      };
    }
    
    /**
     * Extract distribution data from shifts array
     * @param {Array} shifts - Array of shift objects
     * @return {Object} Normalized distribution data
     */
    extractDistributionDataFromShifts(shifts) {
      if (!shifts || shifts.length === 0) {
        return null;
      }
      
      const extractedShifts = shifts.map(shift => {
        // Skip if no distributions
        if (!shift.distributions) return null;
        
        // Extract distributions data
        const distributions = [];
        
        if (shift.distributions.original) {
          distributions.push({
            name: 'original',
            values: shift.distributions.original.values || [],
            statistics: shift.distributions.original.statistics || null
          });
        }
        
        if (shift.distributions.shifted) {
          distributions.push({
            name: 'shifted',
            values: shift.distributions.shifted.values || [],
            statistics: shift.distributions.shifted.statistics || null
          });
        }
        
        // Add metrics if available
        const metrics = shift.distance_metrics || shift.metrics || {};
        
        return {
          type: shift.type,
          distributions,
          metrics
        };
      }).filter(shift => shift !== null);
      
      if (extractedShifts.length === 0) {
        return null;
      }
      
      return {
        shifts: extractedShifts
      };
    }
    
    /**
     * Generate synthetic distribution data if real data not available
     * @param {string} shiftType - Shift type
     * @param {Object} stats - Statistics for original distribution
     * @return {Object} Synthetic distribution data
     */
    generateSyntheticDistribution(shiftType, stats = null) {
      // Default statistics if not provided
      const defaultStats = {
        mean: 0.7,
        stdDev: 0.15,
        min: 0.3,
        max: 1.0
      };
      
      const distributionStats = stats || defaultStats;
      
      // Generate original distribution (closer to normal)
      const originalValues = this.generateNormalDistribution(
        distributionStats.mean,
        distributionStats.stdDev,
        100,
        distributionStats.min,
        distributionStats.max
      );
      
      // Generate shifted distribution based on shift type
      let shiftedValues = [];
      let shiftIntensity = 0.5; // Default shift intensity
      
      switch (shiftType) {
        case 'covariate':
          // Covariate shift: change in input distribution (shift mean)
          shiftedValues = this.generateNormalDistribution(
            distributionStats.mean * 0.8,
            distributionStats.stdDev * 1.2,
            100,
            distributionStats.min,
            distributionStats.max
          );
          break;
          
        case 'concept':
          // Concept shift: change in relationship (more skewed distribution)
          shiftedValues = this.generateSkewedDistribution(
            distributionStats.mean * 0.9,
            distributionStats.stdDev,
            0.5, // Skewness
            100,
            distributionStats.min,
            distributionStats.max
          );
          break;
          
        case 'prediction':
          // Prediction shift: change in output distribution (bimodal distribution)
          const mix = this.generateBimodalDistribution(
            distributionStats.mean * 0.7,
            distributionStats.mean * 1.1,
            distributionStats.stdDev,
            distributionStats.stdDev,
            0.6, // Mix ratio
            100,
            distributionStats.min,
            distributionStats.max
          );
          shiftedValues = mix;
          break;
          
        default:
          // Default to mean shift
          shiftedValues = this.generateNormalDistribution(
            distributionStats.mean * 0.85,
            distributionStats.stdDev,
            100,
            distributionStats.min,
            distributionStats.max
          );
      }
      
      // Calculate basic statistics
      const originalStats = this.calculateBasicStats(originalValues);
      const shiftedStats = this.calculateBasicStats(shiftedValues);
      
      // Calculate distance metrics
      const metrics = {
        kl_divergence: this.estimateKLDivergence(originalValues, shiftedValues),
        js_divergence: this.estimateJSDivergence(originalValues, shiftedValues),
        wasserstein: this.estimateWassersteinDistance(originalValues, shiftedValues)
      };
      
      return {
        distributions: {
          original: {
            values: originalValues,
            statistics: originalStats
          },
          shifted: {
            values: shiftedValues,
            statistics: shiftedStats
          }
        },
        metrics,
        intensity: shiftIntensity
      };
    }
    
    /**
     * Generate normal distribution samples
     * @param {number} mean - Mean of distribution
     * @param {number} stdDev - Standard deviation
     * @param {number} count - Number of samples
     * @param {number} min - Minimum value
     * @param {number} max - Maximum value
     * @return {Array} Generated samples
     */
    generateNormalDistribution(mean, stdDev, count, min, max) {
      const values = [];
      
      // Generate normally distributed random values
      for (let i = 0; i < count; i++) {
        // Box-Muller transform
        let u1 = Math.random();
        let u2 = Math.random();
        
        // Avoid 0 for log
        if (u1 === 0) u1 = 0.0001;
        
        const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        let value = mean + stdDev * z;
        
        // Clamp to min/max
        value = Math.max(min, Math.min(max, value));
        
        values.push(value);
      }
      
      return values;
    }
    
    /**
     * Generate skewed distribution samples
     * @param {number} mean - Mean of distribution
     * @param {number} stdDev - Standard deviation
     * @param {number} skew - Skewness (-1 to 1)
     * @param {number} count - Number of samples
     * @param {number} min - Minimum value
     * @param {number} max - Maximum value
     * @return {Array} Generated samples
     */
    generateSkewedDistribution(mean, stdDev, skew, count, min, max) {
      const values = [];
      
      for (let i = 0; i < count; i++) {
        // Generate normal random value
        let u1 = Math.random();
        let u2 = Math.random();
        
        if (u1 === 0) u1 = 0.0001;
        
        let z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        
        // Apply skew
        if (skew !== 0) {
          const delta = skew / Math.sqrt(1 + skew * skew);
          const scale = 1 / Math.sqrt(1 - delta * delta);
          
          if (z >= 0) {
            z = scale * (z + delta);
          } else {
            z = scale * (z - delta);
          }
        }
        
        let value = mean + stdDev * z;
        
        // Clamp to min/max
        value = Math.max(min, Math.min(max, value));
        
        values.push(value);
      }
      
      return values;
    }
    
    /**
     * Generate bimodal distribution samples
     * @param {number} mean1 - Mean of first mode
     * @param {number} mean2 - Mean of second mode
     * @param {number} stdDev1 - Standard deviation of first mode
     * @param {number} stdDev2 - Standard deviation of second mode
     * @param {number} mixRatio - Mixing ratio (0-1)
     * @param {number} count - Number of samples
     * @param {number} min - Minimum value
     * @param {number} max - Maximum value
     * @return {Array} Generated samples
     */
    generateBimodalDistribution(mean1, mean2, stdDev1, stdDev2, mixRatio, count, min, max) {
      const values = [];
      
      for (let i = 0; i < count; i++) {
        // Decide which mode to sample from
        const useFirstMode = Math.random() < mixRatio;
        
        // Generate sample from appropriate mode
        const mean = useFirstMode ? mean1 : mean2;
        const stdDev = useFirstMode ? stdDev1 : stdDev2;
        
        // Box-Muller transform
        let u1 = Math.random();
        let u2 = Math.random();
        
        if (u1 === 0) u1 = 0.0001;
        
        const z = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
        let value = mean + stdDev * z;
        
        // Clamp to min/max
        value = Math.max(min, Math.min(max, value));
        
        values.push(value);
      }
      
      return values;
    }
    
    /**
     * Calculate basic statistics for distribution
     * @param {Array} values - Distribution values
     * @return {Object} Statistics
     */
    calculateBasicStats(values) {
      if (!values || values.length === 0) {
        return {
          count: 0,
          min: null,
          max: null,
          mean: null,
          median: null,
          stdDev: null
        };
      }
      
      // Sort for percentile calculations
      const sortedValues = [...values].sort((a, b) => a - b);
      
      // Basic statistics
      const count = values.length;
      const min = sortedValues[0];
      const max = sortedValues[count - 1];
      const sum = values.reduce((acc, val) => acc + val, 0);
      const mean = sum / count;
      
      // Calculate median
      const midPoint = Math.floor(count / 2);
      const median = count % 2 === 0 
        ? (sortedValues[midPoint - 1] + sortedValues[midPoint]) / 2
        : sortedValues[midPoint];
      
      // Calculate standard deviation
      const variance = values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / count;
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
     * Estimate KL divergence between two distributions
     * @param {Array} p - First distribution
     * @param {Array} q - Second distribution
     * @return {number} KL divergence
     */
    estimateKLDivergence(p, q) {
      if (!p || !q || p.length === 0 || q.length === 0) return 0;
      
      // Create histogram representations with equal bins
      const min = Math.min(Math.min(...p), Math.min(...q));
      const max = Math.max(Math.max(...p), Math.max(...q));
      const binCount = 20;
      const binWidth = (max - min) / binCount;
      
      // Count samples in each bin
      const histP = Array(binCount).fill(0);
      const histQ = Array(binCount).fill(0);
      
      p.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histP[binIndex]++;
      });
      
      q.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histQ[binIndex]++;
      });
      
      // Normalize histograms to get probability distributions
      const sumP = histP.reduce((a, b) => a + b, 0);
      const sumQ = histQ.reduce((a, b) => a + b, 0);
      
      const probP = histP.map(count => count / sumP);
      const probQ = histQ.map(count => count / sumQ);
      
      // Calculate KL divergence: sum(p(i) * log(p(i) / q(i)))
      let divergence = 0;
      const epsilon = 1e-10; // Small value to avoid division by zero
      
      for (let i = 0; i < binCount; i++) {
        if (probP[i] > 0) {
          divergence += probP[i] * Math.log((probP[i] + epsilon) / (probQ[i] + epsilon));
        }
      }
      
      return divergence;
    }
    
    /**
     * Estimate JS divergence between two distributions
     * @param {Array} p - First distribution
     * @param {Array} q - Second distribution
     * @return {number} JS divergence
     */
    estimateJSDivergence(p, q) {
      if (!p || !q || p.length === 0 || q.length === 0) return 0;
      
      // Create histogram representations with equal bins
      const min = Math.min(Math.min(...p), Math.min(...q));
      const max = Math.max(Math.max(...p), Math.max(...q));
      const binCount = 20;
      const binWidth = (max - min) / binCount;
      
      // Count samples in each bin
      const histP = Array(binCount).fill(0);
      const histQ = Array(binCount).fill(0);
      
      p.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histP[binIndex]++;
      });
      
      q.forEach(val => {
        const binIndex = Math.min(binCount - 1, Math.floor((val - min) / binWidth));
        histQ[binIndex]++;
      });
      
      // Normalize histograms to get probability distributions
      const sumP = histP.reduce((a, b) => a + b, 0);
      const sumQ = histQ.reduce((a, b) => a + b, 0);
      
      const probP = histP.map(count => count / sumP);
      const probQ = histQ.map(count => count / sumQ);
      
      // Calculate midpoint distribution M = (P + Q) / 2
      const probM = probP.map((val, i) => (val + probQ[i]) / 2);
      
      // Calculate KL(P||M) and KL(Q||M)
      let klPM = 0;
      let klQM = 0;
      const epsilon = 1e-10;
      
      for (let i = 0; i < binCount; i++) {
        if (probP[i] > 0) {
          klPM += probP[i] * Math.log((probP[i] + epsilon) / (probM[i] + epsilon));
        }
        
        if (probQ[i] > 0) {
          klQM += probQ[i] * Math.log((probQ[i] + epsilon) / (probM[i] + epsilon));
        }
      }
      
      // JS = 0.5 * (KL(P||M) + KL(Q||M))
      return 0.5 * (klPM + klQM);
    }
    
    /**
     * Estimate Wasserstein distance between two distributions
     * @param {Array} p - First distribution
     * @param {Array} q - Second distribution
     * @return {number} Wasserstein distance
     */
    estimateWassersteinDistance(p, q) {
      if (!p || !q || p.length === 0 || q.length === 0) return 0;
      
      // Sort distributions
      const sortedP = [...p].sort((a, b) => a - b);
      const sortedQ = [...q].sort((a, b) => a - b);
      
      // Calculate empirical CDFs
      const n = sortedP.length;
      const m = sortedQ.length;
      
      // We need equal length samples, so resample to the minimum length
      const minLength = Math.min(n, m);
      const resampledP = [];
      const resampledQ = [];
      
      for (let i = 0; i < minLength; i++) {
        resampledP.push(sortedP[Math.floor(i * n / minLength)]);
        resampledQ.push(sortedQ[Math.floor(i * m / minLength)]);
      }
      
      // Calculate 1st Wasserstein distance (L1 norm)
      let distance = 0;
      for (let i = 0; i < minLength; i++) {
        distance += Math.abs(resampledP[i] - resampledQ[i]);
      }
      
      return distance / minLength;
    }
  }
  
  export default DistributionData;